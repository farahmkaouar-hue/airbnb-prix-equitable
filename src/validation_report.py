"""Phase 5 : figures et synthèse à partir de reports/validation_*.csv."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data import ROOT

sns.set_theme(style="whitegrid", context="notebook")
FIG = ROOT / "reports" / "figures"
S = pd.read_csv(ROOT / "reports" / "validation_summary.csv")
C = pd.read_csv(ROOT / "reports" / "validation_by_city.csv")
feat = pd.read_parquet(ROOT / "data/processed/listings_features.parquet", columns=["city", "price_eur", "is_test", "in_reg_set"])
med = feat[~feat.is_test & feat.in_reg_set].groupby("city")["price_eur"].median()

V1, V3 = "V1 toutes", "V3 agnostique ville"
a = C[(C.schema == "A") & (C.variante == V1)].set_index("city")
b = C[(C.schema == "B") & (C.variante == V1)].set_index("city")
b3 = C[(C.schema == "B") & (C.variante == V3)].set_index("city")

# --- fig 1 : RMSE par ville
order = b.rmse.sort_values().index
x = np.arange(len(order))
fig, ax = plt.subplots(figsize=(14, 6))
w = 0.27
ax.bar(x - w, a.loc[order, "rmse"], w, label="A : intra-villes (ville connue)", color="#4c78a8")
ax.bar(x, b.loc[order, "rmse"], w, label="B : ville jamais vue", color="#e45756")
ax.bar(x + w, b.loc[order, "rmse_recal200"], w, label="B + recalibration du niveau (200 annonces)", color="#54a24b")
ax.set_xticks(x, order, rotation=45, ha="right")
ax.set(title="Erreur de prédiction par ville : ville connue, ville inconnue, ville inconnue recalibrée", xlabel="Ville", ylabel="RMSE sur log(prix)")
ax.legend()
plt.tight_layout()
plt.savefig(FIG / "val_01_rmse_by_city.png", dpi=130)
plt.close()

# --- fig 2 : biais B selon le niveau de prix de la ville
fig, ax = plt.subplots(figsize=(10, 7))
xs, ys = np.log(med.loc[b.index]), b.bias
ax.scatter(xs, ys, s=60, color="#e45756")
for c in b.index:
    ax.annotate(c, (np.log(med[c]), b.loc[c, "bias"]), fontsize=9, xytext=(4, 4), textcoords="offset points")
ax.axhline(0, color="k", lw=0.8)
ax.set(title="Ville jamais vue : biais moyen du modèle selon le niveau de prix de la ville", xlabel="log(prix médian de la ville, EUR)",
       ylabel="Biais moyen = prédit - réel, en log (positif = surestimation)")
plt.tight_layout()
plt.savefig(FIG / "val_02_bias_vs_price_level.png", dpi=130)
plt.close()
corr_bias = float(np.corrcoef(xs, ys)[0, 1])

# --- fig 3 : comparaison des schémas
rows = [("A  intra-villes (V1)", S[(S.schema.str.startswith("A")) & (S.variante == V1)].rmse.iloc[0], "#4c78a8"),
        ("A  intra-villes (V3)", S[(S.schema.str.startswith("A")) & (S.variante == V3)].rmse.iloc[0], "#9ecae1"),
        ("B  ville inconnue (V1)", S[(S.schema.str.startswith("B")) & (S.variante == V1)].rmse.iloc[0], "#e45756"),
        ("B  ville inconnue (V2 sans ville/quartier)", S[(S.schema.str.startswith("B")) & (S.variante == "V2 sans ville/quartier")].rmse.iloc[0], "#b22222"),
        ("B  ville inconnue (V3 agnostique)", S[(S.schema.str.startswith("B")) & (S.variante == V3)].rmse.iloc[0], "#f4a6a6"),
        ("B  V1 recalibré (200 annonces)", float(np.sqrt((b.rmse_recal200 ** 2 * b.n).sum() / b.n.sum())), "#54a24b"),
        ("C  contrôle (hôtes disjoints, sept.)", S[S.schema.str.contains("contrôle") & (S.variante == "toutes les variables")].rmse.iloc[0], "#b79bd1"),
        ("C  temporel (check-in >= oct.)", S[S.schema.str.contains("temporel") & (S.variante == "toutes les variables")].rmse.iloc[0], "#8a5fbf")]
fig, ax = plt.subplots(figsize=(11, 6))
ax.barh([r[0] for r in rows][::-1], [r[1] for r in rows][::-1], color=[r[2] for r in rows][::-1])
for i, r in enumerate(rows[::-1]):
    ax.text(r[1] + 0.005, i, f"{r[1]:.3f}", va="center")
ax.set(title="Comparaison des schémas de validation (LightGBM, cible log(prix))", xlabel="RMSE sur log(prix)", ylabel="Schéma (variante de variables)")
plt.tight_layout()
plt.savefig(FIG / "val_03_schemes.png", dpi=130)
plt.close()

# --- fig 4 : C par mois de check-in
m1 = pd.read_csv(ROOT / "reports" / "validation_C_by_month_all.csv")
m2 = pd.read_csv(ROOT / "reports" / "validation_C_by_month_nodate.csv")
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(m1.mois, m1.rmse, "o-", label="avec variables de date du devis", color="#8a5fbf")
ax.plot(m2.mois, m2.rmse, "s--", label="sans variables de date du devis", color="#bbbbbb")
ax.set(title="Schéma C : erreur selon le mois de check-in (entraînement : séjours de septembre 2026 uniquement)", xlabel="Mois du check-in du devis", ylabel="RMSE sur log(prix)")
plt.xticks(rotation=45)
ax.legend()
plt.tight_layout()
plt.savefig(FIG / "val_04_C_by_month.png", dpi=130)
plt.close()

# --- chiffres de décomposition (pondérés par le nombre d'annonces)
n = b.n
A2 = float((a.rmse ** 2 * n).sum() / n.sum())
B2 = float((b.rmse ** 2 * n).sum() / n.sum())
bias2 = float((b.bias ** 2 * n).sum() / n.sum())
orc2 = float((b.rmse_recal_oracle ** 2 * n).sum() / n.sum())
rec2 = float((b.rmse_recal200 ** 2 * n).sum() / n.sum())
lines = ["# Phase 5 : résultats de la comparaison des schémas de validation", "",
         "Régression de log(price_eur), LightGBM (300 arbres), jeu `core+quote+quality`, 309 412 annonces du train (jeu de régression). Prétraitement ajusté dans chaque pli. Test final non utilisé.", "",
         "Variantes : V1 toutes les variables ; V2 sans `city_*` ni `nbhd_key` ; V3 = V2 sans latitude/longitude absolues (distance au centre conservée).", "",
         "## Synthèse", "", S.round(4).rename(columns={"schema": "schéma", "rmse": "RMSE", "mae": "MAE", "mae_eur": "MAE (EUR, exp simple)"}).to_markdown(index=False), "",
         "## Écart A / B : l'effet ville", "",
         f"- RMSE pondéré : A = {np.sqrt(A2):.3f}, B = {np.sqrt(B2):.3f} (V1).",
         f"- Décomposition de l'erreur quadratique de B : biais de niveau par ville (prédit - réel moyen) = {bias2:.3f} ; erreur résiduelle une fois le niveau corrigé = {orc2:.3f} ; "
         f"B = {B2:.3f} ≈ somme. Écart B - A = {B2 - A2:.3f}, dont **{bias2 / (B2 - A2) * 100:.0f} % dû au décalage de niveau de prix** et {(orc2 - A2) / (B2 - A2) * 100:.0f} % à la perte de qualité du classement intra-ville.",
         f"- Une recalibration du niveau avec 200 annonces de la nouvelle ville ramène le RMSE pondéré de {np.sqrt(B2):.3f} à {np.sqrt(rec2):.3f} (oracle avec toute la ville : {np.sqrt(orc2):.3f}).",
         f"- Corrélation entre le biais d'une ville et son niveau de prix (log médian EUR) : {corr_bias:+.2f}.", "",
         "## Par ville (V1)", "",
         pd.DataFrame({"annonces": b.n, "RMSE A": a.rmse, "RMSE B": b.rmse, "biais B": b.bias, "RMSE B recalibré (200)": b.rmse_recal200, "corrélation prédit/réel dans la ville (B)": b.corr_within,
                       "prix médian EUR": med.loc[b.index]}).sort_values("RMSE B").round(3).to_markdown()]
(ROOT / "reports" / "validation_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"A {np.sqrt(A2):.4f} B {np.sqrt(B2):.4f} bias {np.sqrt(bias2):.4f} oracle {np.sqrt(orc2):.4f} recal {np.sqrt(rec2):.4f} share_bias {bias2 / (B2 - A2):.3f} corr_bias_level {corr_bias:.3f}")
