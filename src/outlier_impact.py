"""Phase 2 : impact de la règle d'outliers de prix (avec / sans / winsorisation).
Entraînement et validation uniquement sur le TRAIN (le test final n'est jamais utilisé ici).
Validation = 20 % des hôtes du train (groupés). Modèle de référence : LightGBM léger, mêmes hyperparamètres.
Usage : python -m src.outlier_impact   (ajoute une section à reports/cleaning_log.md)
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.data import ROOT, load_config

cfg = load_config()
df = pd.read_parquet(ROOT / cfg["paths"]["listings_clean"])
df = df[~df.is_test & df.has_price & ~df.is_geo_outlier & ~df.is_inconsistent].copy()
bounds = pd.read_csv(ROOT / "models" / "price_bounds.csv", index_col=0)

feat = ["accommodates", "bedrooms", "beds", "bath_count", "bath_shared", "latitude", "longitude", "dist_center_km", "n_amenities", "room_type", "city"]
X = df[feat].copy()
for c in ["room_type", "city"]:
    X[c] = X[c].astype("category")
y = df["log_price_eur"].values
groups = df["city"].astype(str) + "_" + df["host_id"].astype(str)
tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=cfg["seed"]).split(X, y, groups))
out_ = df["is_price_outlier"].values
hi = df["city"].map(bounds.high_log).values
y_win = np.minimum(y, hi)

params = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, random_state=cfg["seed"], verbose=-1)
rows = []
for name, mask_tr, ytr in [("Avec outliers (rien retiré)", np.ones(len(df), bool), y),
                           ("Sans outliers (règle 3 x IQR)", ~out_, y),
                           ("Winsorisation (plafonnée à la clôture haute)", np.ones(len(df), bool), y_win)]:
    idx = tr[mask_tr[tr]]
    m = lgb.LGBMRegressor(**params).fit(X.iloc[idx], ytr[idx])
    p = m.predict(X.iloc[va])
    inl = ~out_[va]
    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    mae = lambda a, b: float(np.mean(np.abs(a - b)))
    rows.append([name, len(idx), rmse(y[va][inl], p[inl]), mae(y[va][inl], p[inl]), rmse(y[va], p), mae(y[va], p)])
res = pd.DataFrame(rows, columns=["Stratégie d'entraînement", "n train", "RMSE (validation sans outliers)", "MAE (sans outliers)", "RMSE (validation avec outliers)", "MAE (avec outliers)"]).round(4)
print(res.to_string())
txt = ("\n## Impact de la règle d'outliers (LightGBM de référence, validation sur 20 % des hôtes du train ; cibles = log(price_eur))\n\n"
       + res.to_markdown(index=False)
       + f"\n\nValidation : {len(va):,} annonces dont {int(out_[va].sum())} outliers. Le test final n'est pas utilisé.\n")
p = ROOT / "reports" / "cleaning_log.md"
s = p.read_text(encoding="utf-8")
s = s.split("\n## Impact de la règle d'outliers")[0]
p.write_text(s + txt, encoding="utf-8")
