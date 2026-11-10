"""Phase 12 : interprétabilité du modèle final (SHAP global / local / par ville, PDP / ICE). Résultats en cache : models/_interp_cache.
Les « leviers » sont des ASSOCIATIONS apprises, pas des effets causaux garantis.
Exécuter après src.final_model (utilise le modèle sauvegardé ; les exemples locaux viennent du test, ouvert une seule fois en phase 14).
Usage : python -m src.interpret
"""
import pickle
import warnings

import joblib
import numpy as np
import pandas as pd
import shap

from src.data import ROOT
from src.features import feature_set

warnings.filterwarnings("ignore")
SEED = 42
CACHE = ROOT / "models" / "_interp_cache"
CACHE.mkdir(parents=True, exist_ok=True)
PDP_VARS = ["accommodates", "bedrooms", "bath_count", "dist_center_km", "quote_nights", "n_amenities", "review_scores_rating", "host_tenure_years", "latitude"]


def build():
    f = CACHE / "interp.pkl"
    if f.exists():
        return pickle.load(open(f, "rb"))
    pm = joblib.load(ROOT / "models" / "final_model.joblib")
    d = pd.read_parquet(ROOT / "data/processed/listings_features.parquet")
    tr = d[~d.is_test & d.in_reg_set].reset_index(drop=True)
    S = pd.concat([g.sample(min(1500, len(g)), random_state=SEED) for _, g in tr.groupby("city")]).reset_index(drop=True)
    Z = pm.pre.transform(S)
    ex = shap.TreeExplainer(pm.model)
    sv = ex.shap_values(Z)
    out = dict(cols=list(Z.columns), sv=sv, Z=Z.values, base=float(np.ravel(ex.expected_value)[0]), S=S[["city", "id", "neighbourhood_cleansed", "room_type", "price_eur"]])
    # importance par ville (|SHAP| moyen) et corrélations de rangs
    imp_city = pd.DataFrame(np.abs(sv), columns=Z.columns).groupby(S.city.values).mean()
    out["imp_city"] = imp_city
    # PDP / ICE manuels sur la matrice prétraitée (variables numériques clés + climatisation)
    pdp = {}
    rng = np.random.RandomState(SEED)
    ice_idx = rng.choice(len(Z), 120, replace=False)
    for v in PDP_VARS + ["am_air_conditioning", "am_pool", "am_elevator", "am_dishwasher"]:
        if v not in Z.columns:
            continue
        grid = np.unique(np.quantile(Z[v], np.linspace(0.02, 0.98, 25))) if Z[v].nunique() > 2 else np.array([0.0, 1.0])
        curves = []
        for g in grid:
            Zg = Z.copy()
            Zg[v] = g
            curves.append(pm.model.predict(Zg))
        C = np.array(curves)  # (grille, annonces)
        pdp[v] = dict(grid=grid, mean=C.mean(1), ice=C[:, ice_idx], by_city={c: C[:, (S.city.values == c)].mean(1) for c in ["paris", "tokyo", "new-york-city", "mexico-city"]})
    out["pdp"] = pdp
    # explications locales : 3 annonces du test (bon marché, moyenne, luxe), après ouverture du test
    tp = pd.read_parquet(ROOT / "data/processed/test_predictions.parquet")
    full = d.set_index(["city", "id"])
    picks = {}
    q = tp.price_eur.quantile([0.1, 0.5, 0.97]).values
    for name, target in zip(["bon marché", "moyenne", "luxe"], q):
        # annonce de Paris la plus proche du quantile (cohérence d'exemple)
        c = tp[tp.city == "paris"]
        row = c.iloc[(c.price_eur - target).abs().argsort().iloc[0]]
        picks[name] = row
    rows = pd.DataFrame([full.loc[(r.city, r.id)].to_dict() | {"city": r.city, "id": r.id} for r in picks.values()])
    Zl = pm.pre.transform(rows)
    svl = ex.shap_values(Zl)
    out["local"] = dict(names=list(picks), rows=rows[["city", "id", "neighbourhood_cleansed", "room_type", "accommodates", "bedrooms", "bath_count", "dist_center_km", "n_amenities", "price_eur"]],
                        sv=svl, Z=Zl.values, pred=[float(pm.predict_frame(rows.iloc[[i]]).price_eur.iloc[0]) for i in range(3)])
    pickle.dump(out, open(f, "wb"))
    return out


if __name__ == "__main__":
    o = build()
    imp = pd.Series(np.abs(o["sv"]).mean(0), index=o["cols"]).sort_values(ascending=False)
    print(imp.head(15).round(4))
