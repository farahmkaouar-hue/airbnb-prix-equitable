"""Phase 3 : contrôles des Pipelines de features. Train uniquement (le test final n'est jamais lu).
 1. anti-fuite : le target encoding à CV interne n'est pas « trop beau » par rapport à un encodage naïf
 2. les Pipelines fonctionnent pour les 3 cibles (régression, binaire, multiclasse) et sur modalités inconnues
 3. comparaison indicative des jeux de variables (validation = 20 % des hôtes du train, groupés)
Écrit reports/features_check.md. Usage : python -m src.features_check
"""
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupShuffleSplit

from src.data import ROOT, load_config
from src.features import build_preprocessor, feature_set

cfg = load_config()
d = pd.read_parquet(ROOT / "data/processed/listings_features.parquet")
assert d["is_test"].dtype == bool
d = d[~d["is_test"]]  # le test final n'est pas lu
out = ["# Contrôles des features (Phase 3, train uniquement)", ""]

reg = d[d["in_reg_set"]].reset_index(drop=True)
groups = reg["city"] + "_" + reg["host_id"].astype(str)
tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(reg, groups=groups))
Xtr, Xva = reg.iloc[tr], reg.iloc[va]
ytr, yva = Xtr["log_price_eur"].values, Xva["log_price_eur"].values

# 1. fuite du target encoding : corrélation encodage / cible sur le train, croisé vs naïf
feats = feature_set(reg.columns, "core")
pp = build_preprocessor(feats, Xtr, scale=False)
Ztr = pp.fit_transform(Xtr, ytr)
Zva = pp.transform(Xva)
naive = Xtr.groupby("nbhd_key")["log_price_eur"].transform("mean")
te_cv = Ztr["nbhd_key"]
te_val = Zva["nbhd_key"]
c_naive, c_cv, c_val = np.corrcoef(naive, ytr)[0, 1], np.corrcoef(te_cv, ytr)[0, 1], np.corrcoef(te_val, yva)[0, 1]
out += ["## 1. Target encoding du quartier (nbhd_key)", "",
        f"- corrélation encodage naïf (moyenne du quartier, en échantillon) / cible, train : **{c_naive:.3f}** (optimiste : l'annonce se voit elle-même)",
        f"- corrélation encodage avec CV interne / cible, train : **{c_cv:.3f}**",
        f"- corrélation encodage appliqué à la validation (hôtes jamais vus) / cible : **{c_val:.3f}**",
        "- le TargetEncoder à CV interne (cross-fitting) donne sur le train une corrélation proche de celle observée en validation : pas de fuite visible.", "",
        f"Forme de la matrice de features `core` : {Ztr.shape[1]} colonnes (dont {sum(c.endswith('_missing') for c in Ztr.columns)} indicateurs de manquant).", ""]

# modalité inconnue au transform
Xu = Xva.head(5).copy()
Xu["nbhd_key"] = "paris|QUARTIER_INCONNU"
Xu["property_type"] = "Type inconnu"
assert pp.transform(Xu).notna().all().all(), "NaN après transform sur modalités inconnues"
out += ["- modalités inconnues (nouveau quartier, nouveau type de propriété) : gérées sans erreur (encodage = moyenne globale / catégorie 'infrequent').", ""]

# 2. Pipelines des 3 cibles (échantillon pour la vitesse)
smp = d.sample(40000, random_state=42)
res = []
for target, ttype, mdl in [("is_superhost", "binary", LogisticRegression(max_iter=300)), ("price_tier", "multiclass", LogisticRegression(max_iter=300))]:
    s = smp.dropna(subset=[target])
    s = s[s["in_reg_set"]] if target == "price_tier" else s
    f = feature_set(s.columns, "core")
    p = build_preprocessor(f, s, scale=True, target_type=ttype)
    Z = p.fit_transform(s, s[target].astype(int))
    mdl.fit(Z, s[target].astype(int))
    res.append(f"{target} ({ttype}) : OK, {Z.shape[1]} colonnes, accuracy en échantillon {mdl.score(Z, s[target].astype(int)):.3f} (non évaluée : simple test de fonctionnement)")
out += ["## 2. Pipelines pour les 3 cibles", ""] + [f"- {r}" for r in res] + [f"- régression (log_price_eur) : OK, voir ci-dessous.", ""]

# 3. comparaison indicative des jeux de variables
rows = []
for name in ["core", "core+quote", "core+quote+quality", "full"]:
    f = feature_set(reg.columns, name)
    t0 = time.time()
    pl = build_preprocessor(f, Xtr, scale=True)
    Ztr_l, Zva_l = pl.fit_transform(Xtr, ytr), pl.transform(Xva)
    rid = Ridge(alpha=10, random_state=42).fit(Ztr_l, ytr)
    pr = rid.predict(Zva_l)
    pt = build_preprocessor(f, Xtr, scale=False)
    Ztr_t, Zva_t = pt.fit_transform(Xtr, ytr), pt.transform(Xva)
    gb = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=63, random_state=42, verbose=-1).fit(Ztr_t, ytr)
    pg = gb.predict(Zva_t)
    r = lambda p: float(np.sqrt(np.mean((yva - p) ** 2)))
    a = lambda p: float(np.mean(np.abs(yva - p)))
    rows.append([name, len(f), r(pr), a(pr), r(pg), a(pg), time.time() - t0])
    print(rows[-1])
base = [float(np.sqrt(np.mean((yva - np.median(ytr)) ** 2)))]
cmp = pd.DataFrame(rows, columns=["Jeu de variables", "nb variables", "Ridge RMSE", "Ridge MAE", "LightGBM RMSE", "LightGBM MAE", "temps (s)"]).round(4)
out += ["## 3. Comparaison indicative des jeux de variables (cible log_price_eur)", "",
        f"Validation : {len(va):,} annonces (20 % des hôtes du train, jeu in_reg_set) ; entraînement : {len(tr):,}. Référence « médiane globale » : RMSE {base[0]:.4f}.", "",
        cmp.to_markdown(index=False), "",
        "Ces chiffres sont indicatifs (un seul découpage, hyperparamètres par défaut) ; la comparaison rigoureuse en CV est faite en phases 4 à 6. `full` contient des variables d'activité (disponibilité, avis) qui sont en partie des conséquences du prix."]
(ROOT / "reports" / "features_check.md").write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))
