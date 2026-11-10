"""Phase 5 : comparaison des schémas de validation A / B / C (régression log_price_eur, LightGBM).
Train uniquement (jeu in_reg_set) ; le test final n'est pas lu. Prétraitement ajusté dans chaque pli sur le train du pli.
Variantes de variables :
  V1 toutes ; V2 sans city_* ni nbhd_key ; V3 = V2 sans latitude/longitude absolues (distance au centre conservée).
Usage : python -m src.validation   -> reports/validation_*.csv
"""
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.data import ROOT
from src.evaluate import metrics, scheme_a, scheme_b, scheme_c
from src.features import build_preprocessor, feature_set

warnings.filterwarnings("ignore")
SEED = 42
PARAMS = dict(n_estimators=300, learning_rate=0.08, num_leaves=63, random_state=SEED, n_jobs=16, verbose=-1)


def variants(cols):
    v1 = list(cols)
    v2 = [c for c in v1 if not c.startswith("city_") and c != "nbhd_key"]
    v3 = [c for c in v2 if c not in ("latitude", "longitude")]
    return {"V1 toutes": v1, "V2 sans ville/quartier": v2, "V3 agnostique ville": v3}


def fit_eval(Z, y, tr, va, cols):
    m = lgb.LGBMRegressor(**PARAMS).fit(Z.iloc[tr][cols], y[tr])
    return m.predict(Z.iloc[va][cols])


def run():
    d = pd.read_parquet(ROOT / "data/processed/listings_features.parquet")
    d = d[~d["is_test"] & d["in_reg_set"]].reset_index(drop=True)
    feats = feature_set(d.columns, "core+quote+quality")
    y = d["log_price_eur"].values
    print(f"{len(d):,} annonces train/régression", flush=True)
    rows_city, rows_sum = [], []

    # ---------------- A : intra-villes
    oof = {v: np.zeros(len(d)) for v in variants(feats)}
    for k, (tr, va) in enumerate(scheme_a(d)):
        t0 = time.time()
        pp = build_preprocessor(feats, d.iloc[tr], scale=False)
        Ztr = pp.fit_transform(d.iloc[tr], y[tr])
        Zva = pp.transform(d.iloc[va])
        Z = pd.concat([Ztr, Zva])
        idx_tr, idx_va = np.arange(len(tr)), np.arange(len(tr), len(tr) + len(va))
        for name, cols in variants(Ztr.columns).items():
            oof[name][va] = fit_eval(Z, np.concatenate([y[tr], y[va]]), idx_tr, idx_va, cols)
        print(f"A pli {k} : {time.time() - t0:.0f}s", flush=True)
    for name, p in oof.items():
        m = metrics(y, p)
        rows_sum.append(dict(schema="A intra-villes (5 plis, hôtes groupés)", variante=name, **m))
        for c in sorted(d.city.unique()):
            i = (d.city == c).values
            rows_city.append(dict(schema="A", variante=name, city=c, bias=float((p[i] - y[i]).mean()), **metrics(y[i], p[i])))

    # ---------------- B : leave-one-city-out (+ recalibration par 200 annonces de la ville cible)
    rng = np.random.RandomState(SEED)
    oofB = {v: np.zeros(len(d)) for v in variants(feats)}
    for tr, va, c in scheme_b(d):
        t0 = time.time()
        pp = build_preprocessor(feats, d.iloc[tr], scale=False)
        Ztr = pp.fit_transform(d.iloc[tr], y[tr])
        Zva = pp.transform(d.iloc[va])
        Z = pd.concat([Ztr, Zva])
        yy = np.concatenate([y[tr], y[va]])
        idx_tr, idx_va = np.arange(len(tr)), np.arange(len(tr), len(tr) + len(va))
        calib = rng.choice(len(va), min(200, len(va)), replace=False)
        for name, cols in variants(Ztr.columns).items():
            p = fit_eval(Z, yy, idx_tr, idx_va, cols)
            oofB[name][va] = p
            res = y[va] - p
            shift = res[calib].mean()  # recalibration du niveau de prix avec 200 annonces de la nouvelle ville
            mm = metrics(y[va], p)
            rows_city.append(dict(schema="B", variante=name, city=c, bias=float(-res.mean()), rmse_recal200=float(np.sqrt(np.mean((res - shift) ** 2))),
                                  rmse_recal_oracle=float(res.std()), corr_within=float(np.corrcoef(y[va], p)[0, 1]), **mm))
        print(f"B {c} : {time.time() - t0:.0f}s", flush=True)
    for name, p in oofB.items():
        rows_sum.append(dict(schema="B leave-one-city-out", variante=name, **metrics(y, p)))

    # ---------------- C : temporel (date de check-in du devis), hôtes disjoints
    tr, late, early = scheme_c(d)
    print(f"C : train {len(tr):,} (check-in < 1er oct. 2026), validation tardive {len(late):,}, contrôle précoce {len(early):,}", flush=True)
    pp = build_preprocessor(feats, d.iloc[tr], scale=False)
    Ztr = pp.fit_transform(d.iloc[tr], y[tr])
    Z = pd.concat([Ztr, pp.transform(d.iloc[late]), pp.transform(d.iloc[early])])
    yy = np.concatenate([y[tr], y[late], y[early]])
    i_tr = np.arange(len(tr))
    i_late = np.arange(len(tr), len(tr) + len(late))
    i_early = np.arange(len(tr) + len(late), len(Z))
    date_cols = {"log_quote_lead_days", "quote_month_sin", "quote_month_cos", "quote_is_weekend"}
    for name, cols in [("toutes les variables", list(Ztr.columns)), ("sans variables de date du devis", [c for c in Ztr.columns if c not in date_cols])]:
        m = lgb.LGBMRegressor(**PARAMS).fit(Z.iloc[i_tr][cols], yy[i_tr])
        for lab, ii in [("C contrôle : hôtes disjoints, même période (check-in sept.)", i_early), ("C temporel : hôtes disjoints, check-in >= oct. 2026", i_late)]:
            p = m.predict(Z.iloc[ii][cols])
            rows_sum.append(dict(schema=lab, variante=name, **metrics(yy[ii], p)))
            if "temporel" in lab:
                lt = d.iloc[late].assign(p=p, y=yy[ii], ci=lambda x: pd.to_datetime(x.scrape_date) + pd.to_timedelta(x.quote_lead_days, unit="D"))
                lt["mois"] = lt.ci.dt.to_period("M").astype(str)
                g = lt.groupby("mois").apply(lambda x: pd.Series(dict(n=len(x), rmse=float(np.sqrt(np.mean((x.y - x.p) ** 2))), bias=float((x.p - x.y).mean()))), include_groups=False)
                g.insert(0, "variante", name)
                g.to_csv(ROOT / "reports" / f"validation_C_by_month_{'all' if name.startswith('toutes') else 'nodate'}.csv")

    pd.DataFrame(rows_sum).to_csv(ROOT / "reports" / "validation_summary.csv", index=False)
    pd.DataFrame(rows_city).to_csv(ROOT / "reports" / "validation_by_city.csv", index=False)
    print(pd.DataFrame(rows_sum).round(4).to_string())


if __name__ == "__main__":
    run()
