"""Phase 14 : modèle final et ouverture du jeu de test (UNE SEULE FOIS).

1. ajuste le pipeline final sur tout le train (LightGBM réglé en phase 6) + calibration conforme et correction de biais issues de la CV (hors échantillon)
2. sauvegarde models/final_model.joblib (PriceModel) ; ajuste et sauvegarde les classifieurs Superhost et tranche de prix
3. évalue sur le test final (scellé depuis la phase 2) : prix, intervalles, classifications
Usage : python -m src.final_model       (refuse de s'exécuter deux fois, sauf --force)
"""
import json
import sys
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.classification import best_f1_threshold, bin_metrics, multi_metrics, sets_superhost
from src.data import ROOT
from src.evaluate import rmse
from src.features import build_preprocessor, feature_set
from src.pricing import PriceModel
from src.regression import final as reg_final, full_metrics, tune

warnings.filterwarnings("ignore")
SEED = 42
OUT = ROOT / "reports" / "test_results.json"


def conformal_bins(y, p, alpha=0.1, n_bins=10):
    """Intervalle conforme Mondrian : quantile des |résidus| (log) par tranche de prix prédit, avec correction de taille finie."""
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_bins - 1)
    q = []
    for k in range(n_bins):
        r = np.abs(y[b == k] - p[b == k])
        level = min(1.0, np.ceil((len(r) + 1) * (1 - alpha)) / len(r))
        q.append(float(np.quantile(r, level)))
    return edges, np.array(q)


def run(force=False):
    if OUT.exists() and not force:
        print("Le jeu de test a déjà été ouvert (reports/test_results.json). Le test ne doit servir qu'une fois : abandon (utiliser --force pour refaire).")
        return json.load(open(OUT))
    d = pd.read_parquet(ROOT / "data/processed/listings_features.parquet")
    tr = d[~d.is_test & d.in_reg_set].reset_index(drop=True)
    te_all = d[d.is_test & d.has_price].reset_index(drop=True)
    te = te_all[te_all.in_reg_set].reset_index(drop=True)
    feats = feature_set(d.columns, "core+quote+quality")
    y = tr["log_price_eur"].values

    # --------------------------------------------------------- 1. pipeline final
    pre = build_preprocessor(feats, tr, scale=False)
    Z = pre.fit_transform(tr, y)
    pl = reg_final()["params"]
    model = lgb.LGBMRegressor(**pl).fit(Z, y)
    fin = reg_final()
    smear = float(np.mean(np.exp(fin["y"] - fin["oof"])))
    edges, qs = conformal_bins(fin["y"], fin["oof"])
    tmpl = {}
    cat = ["city", "room_type", "property_type", "nbhd_key"]
    cols = [c for c in feats if c not in cat] + cat
    for c, g in tr.groupby("city"):
        s = g[[k for k in cols if k != "city"]].median(numeric_only=True)
        for k in cat[1:]:
            s[k] = g[k].mode().iloc[0]
        for k in [k for k in s.index if k.startswith("am_")]:
            s[k] = 0.0
        s["city"] = c
        tmpl[c] = s
    centers = {c: (g.latitude.median(), g.longitude.median()) for c, g in tr.groupby("city")}
    nbc = tr.groupby(["city", "neighbourhood_cleansed"])[["latitude", "longitude"]].mean().reset_index()
    vocab = json.load(open(ROOT / "models" / "amenity_vocab.json", encoding="utf-8"))
    pm = PriceModel(pre, model, feats, smear, edges, qs, tmpl, centers, nbc, vocab)
    joblib.dump(pm, ROOT / "models" / "final_model.joblib")
    json.dump({"smear": smear, "conformal_q_by_decile": qs.tolist(), "bin_edges": [None if not np.isfinite(e) else float(e) for e in edges]},
              open(ROOT / "models" / "final_model_calibration.json", "w"), indent=1)

    # --------------------------------------------------------- 2. classifieurs
    clf = {}
    allt = d[~d.is_test].dropna(subset=["is_superhost"]).reset_index(drop=True)
    sets_sh = sets_superhost(allt.columns)
    feats_sh = sets_sh["B. + notes d'avis"]
    feats_shA = sets_sh["A. sans avis (caractéristiques seules)"]
    prs = build_preprocessor(feats_sh, allt, scale=False, target_type="binary")
    ys = allt["is_superhost"].astype(int).values
    Zs = prs.fit_transform(allt, ys)
    cP = dict(n_estimators=300, learning_rate=0.06, num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=16, random_state=SEED, verbose=-1)
    from sklearn.model_selection import GroupShuffleSplit

    a, b = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(Zs, groups=allt["city"] + "_" + allt["host_id"].astype(str)))
    thr = best_f1_threshold(ys[b], lgb.LGBMClassifier(**cP).fit(Zs.iloc[a], ys[a]).predict_proba(Zs.iloc[b])[:, 1])  # seuil choisi sur le train seulement
    msh = lgb.LGBMClassifier(**cP).fit(Zs, ys)
    joblib.dump(dict(pre=prs, model=msh, feats=feats_sh, thr=thr), ROOT / "models" / "final_superhost.joblib")
    # variante A (caractéristiques seules, sans circularité avec les critères d'attribution)
    prA = build_preprocessor(feats_shA, allt, scale=False, target_type="binary")
    ZA = prA.fit_transform(allt, ys)
    thrA = best_f1_threshold(ys[b], lgb.LGBMClassifier(**cP).fit(ZA.iloc[a], ys[a]).predict_proba(ZA.iloc[b])[:, 1])
    mshA = lgb.LGBMClassifier(**cP).fit(ZA, ys)
    joblib.dump(dict(pre=prA, model=mshA, feats=feats_shA, thr=thrA), ROOT / "models" / "final_superhost_A.joblib")
    ttr = d[~d.is_test & d.in_reg_set].dropna(subset=["price_tier"]).reset_index(drop=True)
    prt = build_preprocessor(feats, ttr, scale=False, target_type="multiclass")
    yt = ttr["price_tier"].astype(int).values
    Zt = prt.fit_transform(ttr, yt)
    mt = lgb.LGBMClassifier(**cP).fit(Zt, yt)
    joblib.dump(dict(pre=prt, model=mt, feats=feats), ROOT / "models" / "final_price_tier.joblib")

    # --------------------------------------------------------- 3. TEST FINAL
    res = {"n_train_reg": len(tr), "n_test_clean": len(te), "n_test_all_priced": len(te_all), "smear": smear, "threshold_superhost": thr}
    rows = []
    for name, t in [("test propre (in_reg_set)", te), ("test tous prix (outliers et incohérences inclus)", te_all)]:
        out = pm.predict_frame(t)
        yt_ = t["log_price_eur"].values
        m = full_metrics(yt_, out.log_pred.values, smear)
        pe = out.price_eur.values
        ye = t["price_eur"].values
        m.update(within20=float(np.mean(np.abs(pe - ye) / ye <= 0.2)), coverage=float(np.mean((ye >= out.low_eur) & (ye <= out.high_eur))),
                 width_eur=float(np.mean(out.high_eur - out.low_eur)), median_ae_eur=float(np.median(np.abs(pe - ye))), n=len(t))
        res[name] = m
        rows.append(dict(jeu=name, **m))
        if name.startswith("test propre"):
            t = t.assign(pred_eur=pe, low_eur=out.low_eur.values, high_eur=out.high_eur.values, log_pred=out.log_pred.values)
            t[["city", "id", "host_id", "neighbourhood_cleansed", "room_type", "price_eur", "price_tier", "pred_eur", "low_eur", "high_eur", "log_pred", "log_price_eur"]].to_parquet(
                ROOT / "data/processed/test_predictions.parquet", index=False)
    # classification Superhost
    tes = d[d.is_test].dropna(subset=["is_superhost"])
    p = msh.predict_proba(prs.transform(tes))[:, 1]
    res["superhost"] = bin_metrics(tes["is_superhost"].astype(int).values, p, thr)
    res["superhost"]["auc_without_default_thr"] = res["superhost"]["auc"]
    res["superhost_n"] = len(tes)
    pA = mshA.predict_proba(prA.transform(tes))[:, 1]
    res["superhost_A"] = bin_metrics(tes["is_superhost"].astype(int).values, pA, thrA)
    np.save(ROOT / "data/processed/test_superhost_A_proba.npy", pA)
    np.save(ROOT / "data/processed/test_superhost_proba.npy", p)
    # tranche de prix
    tt = te.dropna(subset=["price_tier"])
    P = mt.predict_proba(prt.transform(tt))
    res["price_tier"] = multi_metrics(tt["price_tier"].astype(int).values, P)
    res["price_tier_n"] = len(tt)
    json.dump(res, open(OUT, "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))
    return res


if __name__ == "__main__":
    run(force="--force" in sys.argv)
