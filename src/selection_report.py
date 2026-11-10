"""Phase 4 : tableau final, consensus entre méthodes et figures, à partir des résultats de src/selection.py."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data import ROOT

sns.set_theme(style="whitegrid", context="notebook")
FIG = ROOT / "reports" / "figures"
res = pd.read_csv(ROOT / "reports" / "selection_results.csv")
F = pd.read_csv(ROOT / "reports" / "selection_frequency.csv", index_col=0)
I = pd.read_csv(ROOT / "reports" / "selection_importances.csv", index_col=0)

# ---------- figure 1 : RMSE vs nombre de variables
fig, ax = plt.subplots(figsize=(11, 7))
fam_col = dict(zip(res.famille.unique(), sns.color_palette("tab10", res.famille.nunique())))
for _, r in res.iterrows():
    ax.errorbar(r.n_variables, r.lgb_rmse, yerr=r.lgb_rmse_std, fmt="o", color=fam_col[r.famille], capsize=3)
    ax.annotate(r.méthode.split(" (")[0], (r.n_variables, r.lgb_rmse), fontsize=7, xytext=(4, 4), textcoords="offset points")
for f, c in fam_col.items():
    ax.plot([], [], "o", color=c, label=f)
ax.set(xscale="log", title="Sélection de variables : erreur de validation (LightGBM) selon le nombre de variables conservées",
       xlabel="Nombre de variables conservées (échelle log)", ylabel="RMSE sur log(prix), moyenne des 5 plis (barres : écart-type)")
ax.legend(title="Famille")
plt.tight_layout()
plt.savefig(FIG / "sel_01_rmse_vs_nfeatures.png", dpi=130)
plt.close()

# ---------- consensus : variables retenues (>= 3 plis sur 5) par méthode
# méthodes SÉLECTIVES (<= 100 variables en moyenne) : les méthodes permissives (Lasso, VIF, RFECV...) gardent presque tout et fausseraient le consensus
n_by = res.set_index("méthode")["n_variables"]
selective = [m for m in F.columns if n_by[m] <= 100 and not m.startswith(("60 variables", "Chi²"))]  # Chi² exclu : performance proche du hasard
sel = F[selective] >= 0.6
cons = sel.sum(axis=1).sort_values(ascending=False)
top = cons.head(40).index
fig, ax = plt.subplots(figsize=(13, 10))
sns.heatmap(F.loc[top, sel.columns], cmap="Blues", vmin=0, vmax=1, cbar_kws={"label": "part des plis où la variable est retenue"}, ax=ax)
ax.set(title="Consensus des méthodes sélectives (<= 100 variables) : 40 variables les plus souvent retenues", xlabel="Méthode de sélection", ylabel="Variable (après prétraitement)")
plt.xticks(rotation=60, ha="right", fontsize=8)
plt.tight_layout()
plt.savefig(FIG / "sel_02_consensus.png", dpi=130)
plt.close()

# ---------- biais des importances : rang selon 4 mesures pour les variables « à forte cardinalité / continues »
keys = [k for k in ["rf_impurity", "lgb_gain", "permutation", "shap"] if k in I.columns]
R = I[keys].rank(ascending=False)
fig, ax = plt.subplots(figsize=(11, 7))
sel_vars = I["shap"].sort_values(ascending=False).head(18).index
M = R.loc[sel_vars, keys]
sns.heatmap(M, annot=True, fmt=".0f", cmap="viridis_r", cbar_kws={"label": "rang (1 = plus important)"}, ax=ax)
ax.set(title="Rang d'importance des 18 variables les plus importantes (SHAP) selon la mesure", xlabel="Mesure d'importance", ylabel="Variable")
plt.tight_layout()
plt.savefig(FIG / "sel_03_importance_ranks.png", dpi=130)
plt.close()

# ---------- tableau markdown
t = res.copy()
t["RMSE LightGBM"] = t.apply(lambda r: f"{r.lgb_rmse:.4f} ± {r.lgb_rmse_std:.4f}", axis=1)
t["Jaccard"] = t.jaccard.map(lambda v: "—" if pd.isna(v) else f"{v:.2f}")
t["nb"] = t.n_variables.map(lambda v: f"{v:.0f}")
tab = t[["méthode", "famille", "nb", "RMSE LightGBM", "lgb_mae", "ridge_rmse", "ridge_mae", "t_selection_s", "t_entrainement_s", "Jaccard"]].round(
    {"lgb_mae": 4, "ridge_rmse": 4, "ridge_mae": 4, "t_selection_s": 1, "t_entrainement_s": 1})
tab.columns = ["Méthode", "Famille", "Nb variables", "RMSE LightGBM (CV)", "MAE LightGBM", "RMSE Ridge", "MAE Ridge", "Temps sélection (s)", "Temps entraînement (s)", "Stabilité (Jaccard)"]
n_meth = sel.shape[1]
top_tab = pd.DataFrame({"méthodes (sur %d)" % n_meth: cons.head(25)})
rf_only = []
txt = ["# Phase 4 : sélection de variables (tableau final)", "",
       "Cible : log(price_eur). Candidates : jeu `core+quote+quality` après prétraitement (target encoding du quartier, one-hot). Échantillon de 60 000 annonces du train, CV 5 plis groupée par hôte ; "
       "prétraitement et sélection ajustés sur le train de chaque pli. Modèle commun d'évaluation : LightGBM (200 arbres) ; Ridge en témoin.", "",
       tab.to_markdown(index=False), "",
       "Stabilité de Jaccard = moyenne des Jaccard sur les 10 paires de plis. « — » : non applicable (PCA, toutes variables).", "",
       f"## Variables retenues par le plus de méthodes sélectives (>= 3 plis sur 5 ; {n_meth} méthodes : {', '.join(selective)})", "", top_tab.to_markdown(), "", "## Importances moyennes (rang 1 = plus important)", "", R.loc[I['shap'].sort_values(ascending=False).head(20).index, keys].astype(int).to_markdown(), ""]
(ROOT / "reports" / "selection_table.md").write_text("\n".join(txt) + "\n", encoding="utf-8")
print(tab.to_string())
print(top_tab.to_string())
print("RF-impurité mais pas permutation (biais ?):", rf_only[:20])
