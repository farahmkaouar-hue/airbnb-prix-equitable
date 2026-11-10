"""Phase 13 : apport métier (calculs lourds en cache : models/_biz_cache).
 - simulation « et si j'ajoute cet équipement ? » (association, pas causalité)
 - annonces sur / sous-évaluées (résidus hors échantillon de la CV), quartiers sous-évalués
 - régulation : concentration des hôtes, licences, annonces fantômes
 - ville inconnue : combien d'annonces faut-il pour recalibrer le niveau de prix ?
Usage : python -m src.business   (après src.regression et src.final_model)
"""
import pickle
import warnings

import joblib
import numpy as np
import pandas as pd

from src.data import ROOT
from src.regression import final as reg_final

warnings.filterwarnings("ignore")
SEED = 42
CACHE = ROOT / "models" / "_biz_cache"
CACHE.mkdir(parents=True, exist_ok=True)


def amenity_whatif(pm, tr, n=2500):
    """Pour chaque équipement du vocabulaire : variation moyenne du prix prédit quand on l'ajoute à des annonces qui ne l'ont pas."""
    rows = []
    rng = np.random.RandomState(SEED)
    base_all = tr.sample(min(20000, len(tr)), random_state=SEED).reset_index(drop=True)
    p0_all = pm.predict_frame(base_all).log_pred.values
    for a in pm.amenity_vocab:
        col = "am_" + a
        if col not in base_all.columns:
            continue
        m = base_all[col] == 0
        if m.sum() < 200:
            continue
        idx = np.where(m)[0]
        idx = rng.choice(idx, min(n, len(idx)), replace=False)
        X = base_all.iloc[idx].copy()
        X[col] = 1.0
        X["n_amenities"] = X["n_amenities"] + 1
        d = pm.predict_frame(X).log_pred.values - p0_all[idx]
        share = float((base_all[col] == 1).mean())
        rows.append(dict(équipement=a, part_des_annonces=share, effet_pct=float(np.expm1(d.mean()) * 100), ic95_bas=float(np.expm1(d.mean() - 1.96 * d.std() / np.sqrt(len(d))) * 100),
                         ic95_haut=float(np.expm1(d.mean() + 1.96 * d.std() / np.sqrt(len(d))) * 100), n=len(d)))
    return pd.DataFrame(rows).sort_values("effet_pct", ascending=False)


def gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def build():
    f = CACHE / "biz.pkl"
    if f.exists():
        return pickle.load(open(f, "rb"))
    pm = joblib.load(ROOT / "models" / "final_model.joblib")
    d = pd.read_parquet(ROOT / "data/processed/listings_features.parquet")
    tr = d[~d.is_test & d.in_reg_set].reset_index(drop=True)
    fin = reg_final()
    out = {}
    out["whatif"] = amenity_whatif(pm, tr)

    # ---- sur / sous-évaluation (résidus hors échantillon, schéma A)
    idx = fin["d_index"].copy()
    idx["res"] = fin["y"] - fin["oof"]  # > 0 : prix observé au-dessus du prix prédit (sur-évalué), < 0 : en dessous (sous-évalué)
    idx["ratio"] = np.exp(idx["res"])
    out["res_by_city"] = idx.groupby("city").agg(n=("res", "size"), pct_sur=("res", lambda s: (s > 0.5).mean() * 100), pct_sous=("res", lambda s: (s < -0.5).mean() * 100),
                                                 rmse=("res", lambda s: np.sqrt((s ** 2).mean())), biais=("res", "mean")).round(2)
    nb = idx.groupby(["city", "neighbourhood_cleansed"]).agg(n=("res", "size"), res_moy=("res", "mean"), res_sd=("res", "std"), prix_median=("price_eur", "median")).reset_index()
    nb = nb[nb.n >= 40].copy()
    nb["se"] = nb.res_sd / np.sqrt(nb.n)
    nb["ecart_pct"] = np.expm1(nb.res_moy) * 100
    nb["ic_bas"], nb["ic_haut"] = np.expm1(nb.res_moy - 1.96 * nb.se) * 100, np.expm1(nb.res_moy + 1.96 * nb.se) * 100
    out["neighbourhoods"] = nb.sort_values("ecart_pct")
    # annonces extrêmes + croisement avec les annonces fantômes
    ext = d[~d.is_test & d.in_reg_set][["city", "id", "availability_365", "number_of_reviews_ltm", "host_is_pro"]].reset_index(drop=True)
    ext["res"] = idx["res"].values
    ext["ghost"] = (ext.availability_365 == 0) & (ext.number_of_reviews_ltm == 0)
    out["ext_vs_ghost"] = ext.assign(grp=np.where(ext.res > 0.5, "sur-évaluées (>+65 %)", np.where(ext.res < -0.5, "sous-évaluées (<-39 %)", "dans la norme"))).groupby("grp").agg(
        n=("res", "size"), part_fantomes=("ghost", "mean"), part_pro=("host_is_pro", "mean"), dispo_moy=("availability_365", "mean")).round(3)

    # ---- régulation : concentration, licences, fantômes (toutes annonces)
    reg = []
    for c, g in d.groupby("city"):
        n_host = g.groupby("host_id").size().sort_values(ascending=False)
        top10 = n_host.head(max(1, int(len(n_host) * 0.1))).sum() / n_host.sum() * 100
        reg.append(dict(city=c, annonces=len(g), hôtes=len(n_host), part_hôtes_5plus_pct=(g.calculated_host_listings_count >= 5).mean() * 100, top10pct_hôtes_part_annonces=top10,
                        gini_annonces_par_hôte=gini(n_host.values), logements_entiers_pct=(g.room_type == "Entire home/apt").mean() * 100, licence_pct=g.has_license.mean() * 100,
                        fantômes_pct=((g.availability_365 == 0) & (g.number_of_reviews_ltm == 0)).mean() * 100, prix_median_eur=g.price_eur.median()))
    out["regulation"] = pd.DataFrame(reg).set_index("city").round(1)

    # ---- ville inconnue : recalibrer le niveau avec n annonces
    rng = np.random.RandomState(SEED)
    q_dec = joblib.load(ROOT / "models" / "final_model.joblib")
    rows = []
    cov = []
    for v in ["V1", "V3"]:
        p = fin["oofB"][v]
        for c in sorted(fin["d_index"].city.unique()):
            m = (fin["d_index"].city == c).values
            r = fin["y"][m] - p[m]
            for n in [0, 10, 25, 50, 100, 200, 500, 1000]:
                vals = []
                for _ in range(40):
                    shift = r[rng.choice(len(r), n, replace=False)].mean() if n > 0 else 0.0
                    vals.append(np.sqrt(np.mean((r - shift) ** 2)))
                rows.append(dict(variante=v, city=c, n_calib=n, rmse=float(np.mean(vals))))
            # couverture de l'intervalle conforme (calibré sur les villes connues) avant et après recalibration (n = 200)
            lp = p[m]
            edges, q = q_dec.bin_edges, q_dec.bin_q
            b = np.clip(np.searchsorted(edges, lp, side="right") - 1, 0, len(q) - 1)
            for lab, shift in [("sans recalibration", 0.0), ("recalibré (200 annonces)", float(r[rng.choice(len(r), 200, replace=False)].mean()))]:
                yv = fin["y"][m]
                cov.append(dict(variante=v, city=c, cas=lab, couverture=float(np.mean(np.abs(yv - (lp + shift)) <= q[b]))))
    out["recal_curve"] = pd.DataFrame(rows)
    out["recal_cov"] = pd.DataFrame(cov)
    pickle.dump(out, open(f, "wb"))
    return out


if __name__ == "__main__":
    o = build()
    print(o["whatif"].head(10).round(2).to_string())
    print(o["whatif"].tail(5).round(2).to_string())
    print(o["res_by_city"].to_string())
    print(o["regulation"].to_string())
    print(o["recal_cov"].groupby(["variante", "cas"]).couverture.mean().round(3))
