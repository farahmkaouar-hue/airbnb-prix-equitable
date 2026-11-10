"""Phase 4 : comparaison de familles de sélection de variables (régression sur log_price_eur).

Protocole (anti-fuite) : échantillon du TRAIN (jeu in_reg_set) ; CV 5 plis groupée par hôte (stratifiée par ville).
Dans chaque pli : prétraitement (imputation, encodage, target encoding) ajusté sur le train du pli, puis chaque méthode
sélectionne ses variables sur le train du pli seulement ; évaluation sur le pli de validation avec un modèle commun
(LightGBM, et Ridge en témoin). Le test final n'est jamais lu.
Usage : python -m src.selection   (résultats par pli mis en cache dans models/_sel_cache pour reprise)
"""
import json
import pickle
import time
import warnings
from itertools import combinations

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV, SelectKBest, SequentialFeatureSelector, chi2, f_classif, f_regression, mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNetCV, LassoCV, Ridge
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedGroupKFold

from src.data import ROOT, load_config
from src.features import build_preprocessor, feature_set

warnings.filterwarnings("ignore")
SEED = 42
K = 60  # nombre de variables conservées par les méthodes à « k fixé »
CACHE = ROOT / "models" / "_sel_cache"
LGB_PARAMS = dict(n_estimators=200, learning_rate=0.1, num_leaves=31, random_state=SEED, n_jobs=16, verbose=-1)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def cum_select(imp: pd.Series, share=0.95) -> list[str]:
    """Variables couvrant `share` de l'importance totale (importances positives)."""
    s = imp.clip(lower=0).sort_values(ascending=False)
    c = s.cumsum() / s.sum()
    return s.index[: int((c < share).sum()) + 1].tolist()


# ------------------------------------------------------------------ méthodes de sélection : (Ztr, ytr, Zs, ctx) -> liste de noms
def m_variance(Z, y, Zs, ctx):
    top = Z.apply(lambda s: s.value_counts(normalize=True).iloc[0])
    return top[top <= 0.995].index.tolist()


def m_corr(Z, y, Zs, ctx):
    sel = SelectKBest(f_regression, k=K).fit(Z, y)
    return Z.columns[sel.get_support()].tolist()


def m_mi(Z, y, Zs, ctx):
    idx = ctx["sub"][:15000]
    disc = (Z.iloc[idx].nunique() <= 2).values
    mi = mutual_info_regression(Z.iloc[idx], y[idx], discrete_features=disc, random_state=SEED, n_jobs=-1)
    ctx["imp"]["mi"] = pd.Series(mi, Z.columns)
    return Z.columns[np.argsort(mi)[::-1][:K]].tolist()


def m_anova(Z, y, Zs, ctx):
    bins = pd.qcut(y, 3, labels=False)
    sel = SelectKBest(f_classif, k=K).fit(Z, bins)
    return Z.columns[sel.get_support()].tolist()


def m_chi2(Z, y, Zs, ctx):
    bins = pd.qcut(y, 3, labels=False)
    mm = (Z - Z.min()) / (Z.max() - Z.min()).replace(0, 1)
    sel = SelectKBest(chi2, k=K).fit(mm, bins)
    return Z.columns[sel.get_support()].tolist()


def m_rfecv(Z, y, Zs, ctx):
    idx = ctx["sub"][:12000]
    g = ctx["groups"][idx]
    cv = list(GroupKFold(3).split(Zs.iloc[idx], groups=g))
    r = RFECV(Ridge(alpha=10), step=8, cv=cv, scoring="neg_root_mean_squared_error", min_features_to_select=10, n_jobs=-1).fit(Zs.iloc[idx], y[idx])
    return Zs.columns[r.support_].tolist()


def m_sfs(Z, y, Zs, ctx):
    idx = ctx["sub"][:8000]
    pool = Zs.columns[SelectKBest(f_regression, k=80).fit(Zs.iloc[idx], y[idx]).get_support()]  # pré-filtre : 80 candidats
    cv = list(GroupKFold(3).split(Zs.iloc[idx], groups=ctx["groups"][idx]))
    s = SequentialFeatureSelector(Ridge(alpha=10), n_features_to_select=25, direction="forward", scoring="neg_root_mean_squared_error", cv=cv, n_jobs=-1)
    s.fit(Zs.iloc[idx][pool], y[idx])
    return pool[s.get_support()].tolist()


def m_lasso(Z, y, Zs, ctx):
    idx = ctx["sub"][:30000]
    m = LassoCV(cv=3, n_alphas=30, random_state=SEED, max_iter=3000, n_jobs=-1).fit(Zs.iloc[idx], y[idx])
    ctx["imp"]["lasso"] = pd.Series(np.abs(m.coef_), Zs.columns)
    return Zs.columns[m.coef_ != 0].tolist()


def m_enet(Z, y, Zs, ctx):
    idx = ctx["sub"][:30000]
    m = ElasticNetCV(l1_ratio=[0.3, 0.7, 0.95], cv=3, n_alphas=30, random_state=SEED, max_iter=3000, n_jobs=-1).fit(Zs.iloc[idx], y[idx])
    return Zs.columns[m.coef_ != 0].tolist()


def m_lgb_gain(Z, y, Zs, ctx):
    m = ctx["lgb_model"]
    imp = pd.Series(m.booster_.feature_importance("gain"), Z.columns)
    ctx["imp"]["lgb_gain"] = imp
    return cum_select(imp)


def m_rf(Z, y, Zs, ctx):
    idx = ctx["sub"][:20000]
    m = RandomForestRegressor(n_estimators=100, max_features=0.3, min_samples_leaf=5, n_jobs=-1, random_state=SEED).fit(Z.iloc[idx], y[idx])
    imp = pd.Series(m.feature_importances_, Z.columns)
    ctx["imp"]["rf_impurity"] = imp
    return cum_select(imp)


def m_perm(Z, y, Zs, ctx):
    # split interne du train du pli (par hôtes) : on n'évalue jamais l'importance sur le pli de validation
    itr, iva = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(Z, groups=ctx["groups"]))
    m = lgb.LGBMRegressor(**LGB_PARAMS).fit(Z.iloc[itr], y[itr])
    iva = iva[:8000]
    r = permutation_importance(m, Z.iloc[iva], y[iva], n_repeats=3, random_state=SEED, scoring="neg_root_mean_squared_error", n_jobs=1)
    ctx["imp"]["permutation"] = pd.Series(r.importances_mean, Z.columns)
    return Z.columns[(r.importances_mean - r.importances_std) > 0].tolist()


def m_shap(Z, y, Zs, ctx):
    idx = ctx["sub"][:5000]
    sv = shap.TreeExplainer(ctx["lgb_model"]).shap_values(Z.iloc[idx])
    imp = pd.Series(np.abs(sv).mean(0), Z.columns)
    ctx["imp"]["shap"] = imp
    return cum_select(imp)


def m_vif(Z, y, Zs, ctx):
    idx = ctx["sub"][:15000]
    X = Zs.iloc[idx]
    X = X.loc[:, X.std() > 0]
    cols = list(X.columns)
    C = np.corrcoef(X.values, rowvar=False)
    keep = np.arange(len(cols))
    while True:
        sub = C[np.ix_(keep, keep)]
        vif = np.diag(np.linalg.inv(sub + 1e-6 * np.eye(len(keep))))
        if vif.max() <= 10 or len(keep) <= 5:
            break
        keep = np.delete(keep, int(vif.argmax()))
    return [cols[i] for i in keep]


def m_random(Z, y, Zs, ctx):
    return list(np.random.RandomState(SEED + ctx["fold"]).choice(Z.columns, K, replace=False))


SELECTORS = [
    ("Toutes les variables (référence)", "témoin", lambda Z, y, Zs, c: Z.columns.tolist()),
    ("60 variables aléatoires (témoin)", "témoin", m_random),
    ("Variance quasi nulle (>99,5 % identique)", "filtre", m_variance),
    ("Corrélation (f_regression, k=60)", "filtre", m_corr),
    ("Information mutuelle (k=60)", "filtre", m_mi),
    ("ANOVA F sur tercile de prix (k=60)", "filtre", m_anova),
    ("Chi² sur tercile de prix (k=60)", "filtre", m_chi2),
    ("RFECV (Ridge)", "wrapper", m_rfecv),
    ("SFS avant (Ridge, 25 parmi 80)", "wrapper", m_sfs),
    ("Lasso (LassoCV)", "embedded", m_lasso),
    ("Elastic Net (CV)", "embedded", m_enet),
    ("LightGBM gain (95 % cumulé)", "embedded", m_lgb_gain),
    ("Random Forest impurité (95 % cumulé)", "embedded", m_rf),
    ("Permutation (LightGBM, split interne)", "agnostique", m_perm),
    ("SHAP |moyenne| (95 % cumulé)", "agnostique", m_shap),
    ("VIF itératif (seuil 10)", "colinéarité", m_vif),
]


def load_data(cfg):
    d = pd.read_parquet(ROOT / "data/processed/listings_features.parquet")
    d = d[~d["is_test"] & d["in_reg_set"]]
    n = cfg["sampling"].get("selection_n", 60000)
    d = d.sample(n, random_state=SEED).reset_index(drop=True)
    return d


def run_fold(k, tr, va, d, feats, y):
    f = CACHE / f"fold{k}.pkl"
    if f.exists():
        return pickle.load(open(f, "rb"))
    Xtr, Xva = d.iloc[tr], d.iloc[va]
    ytr, yva = y[tr], y[va]
    pp = build_preprocessor(feats, Xtr, scale=False)
    Z = pp.fit_transform(Xtr, ytr)
    Zv = pp.transform(Xva)
    mu, sd = Z.mean(), Z.std().replace(0, 1)
    Zs, Zvs = (Z - mu) / sd, (Zv - mu) / sd
    groups = (Xtr["city"] + "_" + Xtr["host_id"].astype(str)).values
    rng = np.random.RandomState(SEED + k)
    ctx = {"sub": rng.permutation(len(Z)), "groups": groups, "imp": {}, "fold": k}
    t0 = time.time()
    ctx["lgb_model"] = lgb.LGBMRegressor(**LGB_PARAMS).fit(Z, ytr)
    t_lgb_full = time.time() - t0
    out = {"fold": k, "n_features_total": Z.shape[1], "methods": {}, "importances": {}}
    for name, fam, fn in SELECTORS:
        t0 = time.time()
        S = fn(Z, ytr, Zs, ctx)
        t_sel = time.time() - t0
        if name.startswith(("LightGBM gain", "SHAP")):
            t_sel += t_lgb_full  # on compte aussi l'entraînement du modèle utilisé
        t0 = time.time()
        g = lgb.LGBMRegressor(**LGB_PARAMS).fit(Z[S], ytr)
        t_fit = time.time() - t0
        pg = g.predict(Zv[S])
        rd = Ridge(alpha=10).fit(Zs[S], ytr)
        pr = rd.predict(Zvs[S])
        out["methods"][name] = dict(family=fam, n=len(S), selected=S, lgb_rmse=rmse(yva, pg), lgb_mae=mae(yva, pg), ridge_rmse=rmse(yva, pr), ridge_mae=mae(yva, pr),
                                    t_select=t_sel, t_fit=t_fit)
        print(f"fold {k} | {name:45s} n={len(S):3d} lgb_rmse={rmse(yva, pg):.4f} t_sel={t_sel:.1f}s", flush=True)
    # PCA (transformation, pas sélection)
    for nm, nc in [("PCA (95 % de variance)", 0.95), ("PCA (60 composantes)", 60)]:
        t0 = time.time()
        p = PCA(n_components=nc, random_state=SEED).fit(Zs)
        P, Pv = p.transform(Zs), p.transform(Zvs)
        t_sel = time.time() - t0
        t0 = time.time()
        g = lgb.LGBMRegressor(**LGB_PARAMS).fit(P, ytr)
        t_fit = time.time() - t0
        pg = g.predict(Pv)
        pr = Ridge(alpha=10).fit(P, ytr).predict(Pv)
        out["methods"][nm] = dict(family="transformation", n=P.shape[1], selected=None, lgb_rmse=rmse(yva, pg), lgb_mae=mae(yva, pg), ridge_rmse=rmse(yva, pr),
                                  ridge_mae=mae(yva, pr), t_select=t_sel, t_fit=t_fit)
        print(f"fold {k} | {nm:45s} n={P.shape[1]:3d} lgb_rmse={rmse(yva, pg):.4f}", flush=True)
    out["importances"] = {k_: v.to_dict() for k_, v in ctx["imp"].items()}
    CACHE.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(f, "wb"))
    return out


def jaccard(sets):
    js = [len(a & b) / len(a | b) for a, b in combinations([set(s) for s in sets], 2)]
    return float(np.mean(js))


def run():
    cfg = load_config()
    CACHE.mkdir(parents=True, exist_ok=True)
    d = load_data(cfg)
    feats = feature_set(d.columns, "core+quote+quality")
    y = d["log_price_eur"].values
    groups = d["city"] + "_" + d["host_id"].astype(str)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = [run_fold(k, tr, va, d, feats, y) for k, (tr, va) in enumerate(cv.split(d, d["city"], groups))]

    rows = []
    for name in folds[0]["methods"]:
        ms = [f["methods"][name] for f in folds]
        sel = [m["selected"] for m in ms]
        rows.append(dict(méthode=name, famille=ms[0]["family"], n_variables=np.mean([m["n"] for m in ms]), lgb_rmse=np.mean([m["lgb_rmse"] for m in ms]),
                         lgb_rmse_std=np.std([m["lgb_rmse"] for m in ms]), lgb_mae=np.mean([m["lgb_mae"] for m in ms]), ridge_rmse=np.mean([m["ridge_rmse"] for m in ms]),
                         ridge_mae=np.mean([m["ridge_mae"] for m in ms]), t_selection_s=np.mean([m["t_select"] for m in ms]), t_entrainement_s=np.mean([m["t_fit"] for m in ms]),
                         jaccard=jaccard(sel) if sel[0] is not None else np.nan))
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "reports" / "selection_results.csv", index=False)

    # fréquence de sélection par variable et méthode (sur les 5 plis)
    freq = {}
    for name in folds[0]["methods"]:
        if folds[0]["methods"][name]["selected"] is None:
            continue
        c = pd.Series(0.0, index=sorted({v for f in folds for v in f["methods"][name]["selected"]}))
        for f in folds:
            c[f["methods"][name]["selected"]] += 1 / len(folds)
        freq[name] = c
    F = pd.DataFrame(freq).fillna(0)
    F.to_csv(ROOT / "reports" / "selection_frequency.csv")
    imps = {}
    for key in folds[0]["importances"]:
        imps[key] = pd.DataFrame([f["importances"][key] for f in folds]).mean()
    pd.DataFrame(imps).to_csv(ROOT / "reports" / "selection_importances.csv")
    print(res.round(4).to_string())


if __name__ == "__main__":
    run()
