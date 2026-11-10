"""Phase 6 : régression du prix (log_price_eur).

Étapes (chaque étape met ses résultats en cache dans models/_reg_cache) :
  course   : 11 modèles + baselines, CV A (5 plis groupés par hôte), échantillon de 80 000 annonces du train
  tune     : Optuna (LightGBM, XGBoost) dans la CV (3 plis groupés par hôte), même échantillon
  final    : modèle réglé sur tout le train (jeu in_reg_set) : prédictions hors échantillon (A), leave-one-city-out (B),
             courbe d'apprentissage, résidus, intervalles de prédiction
Le test final n'est jamais lu. Usage : python -m src.regression [course|tune|final|all]
"""
import json
import pickle
import sys
import time
import warnings

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, cross_val_predict

from src.data import ROOT
from src.evaluate import group_key, mae, mae_eur, r2, rmse
from src.features import build_preprocessor, feature_set
from src.models import CityMedianBaseline, fold_matrices, regression_zoo

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
SEED = 42
N_COURSE = 80000
CACHE = ROOT / "models" / "_reg_cache"
CACHE.mkdir(parents=True, exist_ok=True)
FEATS_NAME = "core+quote+quality"


def load_train():
    d = pd.read_parquet(ROOT / "data/processed/listings_features.parquet")
    return d[~d["is_test"] & d["in_reg_set"]].reset_index(drop=True)


def smear_factor(y, p):
    """Duan smearing : facteur multiplicatif qui corrige le biais de la retransformation exp() (E[exp(e)] > 1)."""
    return float(np.mean(np.exp(np.asarray(y) - np.asarray(p))))


def full_metrics(y, p, smear=None):
    pe, ye = np.exp(p), np.exp(y)
    s = smear if smear is not None else smear_factor(y, p)
    pc = pe * s
    return dict(rmse=rmse(y, p), mae=mae(y, p), r2=r2(y, p), mae_eur=float(np.mean(np.abs(ye - pe))), mae_eur_corr=float(np.mean(np.abs(ye - pc))),
                mape=float(np.mean(np.abs(ye - pc) / ye)), smear=s)


# ===================================================================================== COURSE
def course():
    f = CACHE / "course.pkl"
    if f.exists():
        return pickle.load(open(f, "rb"))
    d = load_train()
    S = d.sample(N_COURSE, random_state=SEED).reset_index(drop=True)
    feats = feature_set(S.columns, FEATS_NAME)
    y = S["log_price_eur"].values
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    folds = list(cv.split(S, S["city"], group_key(S)))
    zoo = regression_zoo()
    oof = {k: np.zeros(len(S)) for k in list(zoo) + ["Médiane de la ville"]}
    times = {k: 0.0 for k in oof}
    for i, (tr, va) in enumerate(folds):
        t0 = time.time()
        Z, Zv, Zs, Zvs = fold_matrices(S, feats, tr, va, y, build_preprocessor)
        b = CityMedianBaseline().fit(S.iloc[tr]["city"], y[tr])
        oof["Médiane de la ville"][va] = b.predict(S.iloc[va]["city"])
        for name, (m, scale) in zoo.items():
            t1 = time.time()
            m.fit(Zs if scale else Z, y[tr])
            oof[name][va] = m.predict(Zvs if scale else Zv)
            times[name] += time.time() - t1
        print(f"course pli {i} : {time.time() - t0:.0f}s", flush=True)
    rows = []
    for k, p in oof.items():
        m = full_metrics(y, p)
        # écart-type entre plis
        fr = [rmse(y[va], p[va]) for _, va in folds]
        rows.append(dict(modèle=k, rmse_std=float(np.std(fr)), temps_s=times[k] / 5, **m))
    # stacking : méta-modèle Ridge sur les prédictions hors échantillon des meilleurs modèles (évalué par CV groupée du méta-modèle)
    base = [k for k in ["Ridge", "Random Forest", "XGBoost", "LightGBM", "HistGradientBoosting"] if k in oof]
    M = np.column_stack([oof[k] for k in base])
    meta = RidgeCV(alphas=[0.01, 0.1, 1, 10])
    ps = cross_val_predict(meta, M, y, cv=GroupKFold(5), groups=group_key(S))
    oof["Stacking (Ridge sur 5 modèles)"] = ps
    rows.append(dict(modèle="Stacking (Ridge sur 5 modèles)", rmse_std=float(np.std([rmse(y[va], ps[va]) for _, va in folds])), temps_s=np.nan, **full_metrics(y, ps)))
    meta.fit(M, y)
    res = dict(table=pd.DataFrame(rows).sort_values("rmse"), oof=oof, y=y, S_index=S[["city", "id", "host_id"]], meta_weights=dict(zip(base, meta.coef_.round(3))), n=len(S))
    pickle.dump(res, open(f, "wb"))
    res["table"].to_csv(ROOT / "reports" / "regression_course.csv", index=False)
    return res


# ===================================================================================== RÉGLAGE
def tune(n_trials_lgb=40, n_trials_xgb=25):
    f = CACHE / "tune.json"
    if f.exists():
        return json.load(open(f))
    d = load_train()
    S = d.sample(N_COURSE, random_state=SEED).reset_index(drop=True)
    feats = feature_set(S.columns, FEATS_NAME)
    y = S["log_price_eur"].values
    folds = list(GroupKFold(3).split(S, groups=group_key(S)))
    mats = [fold_matrices(S, feats, tr, va, y, build_preprocessor)[:2] for tr, va in folds]
    out = {}

    def obj_lgb(t):
        p = dict(n_estimators=t.suggest_int("n_estimators", 200, 900), learning_rate=t.suggest_float("learning_rate", 0.02, 0.15, log=True),
                 num_leaves=t.suggest_int("num_leaves", 15, 255, log=True), min_child_samples=t.suggest_int("min_child_samples", 5, 100, log=True),
                 colsample_bytree=t.suggest_float("colsample_bytree", 0.3, 1.0), subsample=t.suggest_float("subsample", 0.5, 1.0), subsample_freq=1,
                 reg_lambda=t.suggest_float("reg_lambda", 1e-3, 30, log=True), random_state=SEED, n_jobs=16, verbose=-1)
        sc = []
        for (tr, va), (Z, Zv) in zip(folds, mats):
            m = lgb.LGBMRegressor(**p).fit(Z, y[tr])
            sc.append(rmse(y[va], m.predict(Zv)))
            t.report(float(np.mean(sc)), len(sc))
            if t.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(sc))

    def obj_xgb(t):
        p = dict(n_estimators=t.suggest_int("n_estimators", 200, 900), learning_rate=t.suggest_float("learning_rate", 0.02, 0.15, log=True),
                 max_depth=t.suggest_int("max_depth", 4, 11), min_child_weight=t.suggest_float("min_child_weight", 1, 30, log=True),
                 colsample_bytree=t.suggest_float("colsample_bytree", 0.3, 1.0), subsample=t.suggest_float("subsample", 0.5, 1.0),
                 reg_lambda=t.suggest_float("reg_lambda", 1e-3, 30, log=True), tree_method="hist", n_jobs=16, random_state=SEED, verbosity=0)
        sc = []
        for (tr, va), (Z, Zv) in zip(folds, mats):
            m = xgb.XGBRegressor(**p).fit(Z, y[tr])
            sc.append(rmse(y[va], m.predict(Zv)))
            t.report(float(np.mean(sc)), len(sc))
            if t.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(sc))

    for name, obj, n in [("LightGBM", obj_lgb, n_trials_lgb), ("XGBoost", obj_xgb, n_trials_xgb)]:
        st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED), pruner=optuna.pruners.MedianPruner(n_startup_trials=6))
        t0 = time.time()
        st.optimize(obj, n_trials=n)
        hist = [dict(trial=t.number, value=t.value, state=str(t.state)) for t in st.trials]
        out[name] = dict(best_params=st.best_params, best_rmse=st.best_value, seconds=time.time() - t0, history=hist,
                         importances={k: float(v) for k, v in optuna.importance.get_param_importances(st).items()})
        print(name, "meilleur RMSE CV", round(st.best_value, 4), st.best_params, flush=True)
    json.dump(out, open(f, "w"), indent=1)
    return out


# ===================================================================================== FINAL (OOF sur tout le train, LOCO, intervalles)
def final():
    f = CACHE / "final.pkl"
    if f.exists():
        return pickle.load(open(f, "rb"))
    tuned = tune()
    pl = dict(tuned["LightGBM"]["best_params"], subsample_freq=1, random_state=SEED, n_jobs=16, verbose=-1)
    d = load_train()
    feats = feature_set(d.columns, FEATS_NAME)
    y = d["log_price_eur"].values
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(d, d["city"], group_key(d)))
    oof, oof_lo, oof_hi, oof_default = (np.zeros(len(d)) for _ in range(4))
    fold_id = np.zeros(len(d), int)
    default_p = dict(n_estimators=300, learning_rate=0.08, num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED, n_jobs=16, verbose=-1)
    for i, (tr, va) in enumerate(folds):
        t0 = time.time()
        Z, Zv, _, _ = fold_matrices(d, feats, tr, va, y, build_preprocessor)
        oof[va] = lgb.LGBMRegressor(**pl).fit(Z, y[tr]).predict(Zv)
        oof_default[va] = lgb.LGBMRegressor(**default_p).fit(Z, y[tr]).predict(Zv)
        qp = dict(pl, objective="quantile")
        oof_lo[va] = lgb.LGBMRegressor(**qp, alpha=0.05).fit(Z, y[tr]).predict(Zv)
        oof_hi[va] = lgb.LGBMRegressor(**qp, alpha=0.95).fit(Z, y[tr]).predict(Zv)
        fold_id[va] = i
        print(f"final OOF pli {i} : {time.time() - t0:.0f}s", flush=True)
    # courbe d'apprentissage : taille d'entraînement vs erreur (hôtes disjoints), pli 0 comme validation
    tr0, va0 = folds[0]
    lc = []
    rng = np.random.RandomState(SEED)
    for frac in [0.05, 0.1, 0.2, 0.4, 0.7, 1.0]:
        sub = rng.choice(tr0, int(len(tr0) * frac), replace=False)
        Z, Zv, _, _ = fold_matrices(d, feats, sub, va0, y, build_preprocessor)
        m = lgb.LGBMRegressor(**pl).fit(Z, y[sub])
        lc.append(dict(n_train=len(sub), rmse_val=rmse(y[va0], m.predict(Zv)), rmse_train=rmse(y[sub], m.predict(Z))))
        print("courbe d'apprentissage", lc[-1], flush=True)
    # leave-one-city-out avec le modèle réglé (V1 et V3)
    loco = []
    oofB = {"V1": np.zeros(len(d)), "V3": np.zeros(len(d))}
    rng = np.random.RandomState(SEED)
    for c in sorted(d.city.unique()):
        tr, va = np.where(d.city != c)[0], np.where(d.city == c)[0]
        Z, Zv, _, _ = fold_matrices(d, feats, tr, va, y, build_preprocessor)
        drop3 = [k for k in Z.columns if k.startswith("city_") or k == "nbhd_key" or k in ("latitude", "longitude")]
        cal = rng.choice(len(va), min(200, len(va)), replace=False)
        for v, cols in [("V1", list(Z.columns)), ("V3", [k for k in Z.columns if k not in drop3])]:
            p = lgb.LGBMRegressor(**pl).fit(Z[cols], y[tr]).predict(Zv[cols])
            oofB[v][va] = p
            res = y[va] - p
            loco.append(dict(city=c, variante=v, rmse=rmse(y[va], p), bias=float(-res.mean()), rmse_recal200=float(np.sqrt(np.mean((res - res[cal].mean()) ** 2))), n=len(va)))
        print("LOCO", c, flush=True)
    out = dict(d_index=d[["city", "id", "host_id", "neighbourhood_cleansed", "room_type", "price_eur", "price_tier"]], y=y, oof=oof, oof_default=oof_default, oof_lo=oof_lo,
               oof_hi=oof_hi, fold_id=fold_id, learning_curve=lc, loco=pd.DataFrame(loco), oofB=oofB, params=pl)
    pickle.dump(out, open(f, "wb"))
    return out


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step in ("course", "all"):
        course()
    if step in ("tune", "all"):
        tune()
    if step in ("final", "all"):
        final()
