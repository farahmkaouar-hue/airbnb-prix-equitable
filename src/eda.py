"""Phase 1 : EDA. Produit les figures (reports/figures/eda_*.png) et reports/eda_insights.md.

Usage : python -m src.eda          (Paris d'abord, puis les 15 villes)
Aucune donnée n'est modifiée ; tous les chiffres des insights sont calculés ici.
Les taux de change n'étant pas encore validés, les prix sont comparés entre villes
via l'écart à la médiane de la ville (prix relatif) et le log, pas en devise commune.
"""
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns

from src.data import ROOT, load_config, load_listings

SEED = 42
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")
INS: list[str] = []  # insights chiffrés


def save(name):
    plt.tight_layout()
    plt.savefig(FIG / f"eda_{name}.png", dpi=130)
    plt.close()


def haversine(lat1, lon1, lat2, lon2):
    """Distance en km entre deux points (degrés)."""
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 12742 * np.arcsin(np.sqrt(a))


def add_insight(txt):
    INS.append(txt)
    print("-", txt)


def run():
    cfg = load_config()
    df = load_listings(cfg)
    df["superhost"] = df["host_is_superhost"].map({"t": 1, "f": 0})
    df["log_price"] = np.log(df["price"])
    df["price_rel"] = df["price"] / df.groupby("city")["price"].transform("median")

    # ------------------------------------------------------------------ 1. Manquants
    miss = df.drop(columns=["city", "country", "scrape_date", "superhost", "log_price", "price_rel"]).groupby(df["city"]).apply(lambda x: x.isna().mean() * 100, include_groups=False)
    empty_all = miss.columns[(miss > 99.9).all()].tolist()
    keep = miss.columns[(miss.max() > 1) & ~miss.columns.isin(empty_all)]
    plt.figure(figsize=(16, 7))
    sns.heatmap(miss[keep], cmap="viridis_r", cbar_kws={"label": "% manquant"}, vmin=0, vmax=100)
    plt.title("Valeurs manquantes par ville et par colonne (%, colonnes 100 % vides exclues)")
    plt.xlabel("Colonne")
    plt.ylabel("Ville")
    save("01_missing_heatmap")
    pm = miss["price"]
    add_insight(f"{len(empty_all)} colonnes sont vides (100 %) dans toutes les villes : {', '.join(empty_all)}. À supprimer.")
    add_insight(f"`price` manque pour {df.price.isna().mean() * 100:.1f} % des annonces (de {pm.min():.1f} % à {pm.idxmin()} à {pm.max():.1f} % à {pm.idxmax()}) : "
                f"{df.price.notna().sum():,} annonces exploitables pour la régression.")
    # le manque de prix est-il lié à l'indisponibilité ? (biais de sélection)
    na_av = df.groupby(df.price.isna())["availability_365"].apply(lambda s: (s == 0).mean() * 100)
    add_insight(f"Biais de sélection : parmi les annonces SANS prix, {na_av[True]:.0f} % ont availability_365 = 0, contre {na_av[False]:.0f} % parmi celles avec prix "
                "(le prix n'existe que pour les annonces réservables).")
    sparse = (miss.mean() > 30) & ~miss.columns.isin(empty_all)
    add_insight("Colonnes partiellement vides (>30 % en moyenne) : " + ", ".join(f"{c} ({miss[c].mean():.0f} %)" for c in miss.columns[sparse]) + ".")

    # ------------------------------------------------------------------ 2. Prix
    d = df[df.price.notna()]
    cities = sorted(df.city.unique())
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    paris = d[d.city == "paris"]
    ax[0].hist(paris.price.clip(upper=paris.price.quantile(0.99)), bins=80, color="#4c78a8")
    ax[0].set(title="Paris : prix brut (écrêté au 99e centile)", xlabel="Prix (EUR / nuit)", ylabel="Nombre d'annonces")
    ax[1].hist(paris.log_price, bins=80, color="#f58518")
    ax[1].set(title="Paris : log(prix)", xlabel="log(prix en EUR)", ylabel="Nombre d'annonces")
    save("02_price_paris_raw_vs_log")
    sk_raw, sk_log = paris.price.skew(), paris.log_price.skew()
    add_insight(f"Paris : asymétrie du prix = {sk_raw:.1f} (brut) contre {sk_log:.2f} (log) : le log(prix) est quasi symétrique, d'où la cible log(price).")

    fig, ax = plt.subplots(figsize=(14, 6))
    order = d.groupby("city").log_price.median().sort_values().index
    sns.boxplot(data=d, x="city", y="log_price", order=order, fliersize=1, ax=ax, color="#9ecae1")
    ax.set(title="log(prix) par ville, en devise locale (non convertie)", xlabel="Ville", ylabel="log(prix, devise locale)")
    plt.xticks(rotation=45, ha="right")
    save("03_logprice_by_city_local_currency")
    med = d.groupby("city").price.median()
    cur = d.groupby("city").currency.agg(lambda s: s.mode().iat[0])
    add_insight("Médianes en devise locale (NON comparables sans conversion) : " + ", ".join(f"{c} {med[c]:,.0f} {cur[c]}" for c in med.sort_values().index) + ".")

    fig, ax = plt.subplots(figsize=(12, 6))
    rt = d.room_type.value_counts()
    keep_rt = rt[rt > 200].index
    sns.boxplot(data=d[d.room_type.isin(keep_rt)], x="room_type", y=np.log(d.loc[d.room_type.isin(keep_rt), "price_rel"]), fliersize=1, ax=ax, color="#a1d99b")
    ax.set(title="Prix relatif à la médiane de la ville, par type de logement", xlabel="Type de logement", ylabel="log(prix / médiane de la ville)")
    save("04_rel_price_by_room_type")
    rm = d.groupby("room_type").price_rel.median()
    add_insight("Prix relatif médian (1 = médiane de la ville) par room_type : " + ", ".join(f"{k} {v:.2f}" for k, v in rm.sort_values(ascending=False).items())
                + f" ; répartition : {', '.join(f'{k} {v / len(df) * 100:.0f} %' for k, v in df.room_type.value_counts().items())}.")

    # outliers de prix
    q = d.groupby("city").price.quantile([0.001, 0.99, 0.999]).unstack()
    ratio = (q[0.999] / med).sort_values(ascending=False)
    add_insight(f"Outliers de prix : prix = 0 : {(df.price == 0).sum()} ; prix max global {d.price.max():,.0f} (devise locale, {d.loc[d.price.idxmax(), 'city']}). "
                f"Rapport p99,9 / médiane le plus élevé : {ratio.index[0]} ({ratio.iloc[0]:.0f}x). Prix > 20x la médiane de sa ville : {(d.price_rel > 20).sum()} annonces.")
    add_insight(f"Prix < 0,1x la médiane de sa ville : {(d.price_rel < 0.1).sum()} annonces (prix aberrants bas possibles : devis par mois ou erreurs).")
    qn = d.groupby("city").price_rel.quantile([0.01, 0.99]).unstack()
    add_insight(f"Dispersion du prix relatif (p1 → p99) : médiane des villes {qn[0.01].median():.2f} → {qn[0.99].median():.2f}.")

    # ------------------------------------------------------------------ 3. Numériques
    num = ["price", "accommodates", "bedrooms", "beds", "minimum_nights", "maximum_nights", "availability_365", "number_of_reviews", "number_of_reviews_ltm",
           "reviews_per_month", "review_scores_rating", "review_scores_location", "review_scores_value", "calculated_host_listings_count", "hosts_time_as_host_years",
           "latitude", "longitude"]
    num = [c for c in num if c in df.columns]
    st = d[num].describe().T
    st["skew"] = d[num].skew()
    st["miss_%"] = df[num].isna().mean() * 100
    st.round(2).to_csv(ROOT / "reports" / "eda_numeric_stats.csv")
    cols = ["log_price"] + [c for c in num if c not in ("price", "latitude", "longitude")]
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))
    for a, m in zip(ax, ["pearson", "spearman"]):
        sns.heatmap(d[cols].corr(method=m), annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, annot_kws={"size": 6}, ax=a, cbar=False)
        a.set_title(f"Corrélation de {m.capitalize()}")
    save("05_correlations")
    cs = d[cols].corr("spearman")["log_price"].drop("log_price").sort_values(key=abs, ascending=False)
    add_insight("Corrélations de Spearman avec log(price) : " + ", ".join(f"{k} {v:+.2f}" for k, v in cs.head(5).items()) + ".")
    cm = d[cols].corr("spearman").abs()
    pairs = [(a, b, cm.loc[a, b]) for i, a in enumerate(cols) for b in cols[i + 1:] if cm.loc[a, b] > 0.85 and "log_price" not in (a, b)]
    add_insight("Paires redondantes (|Spearman| > 0,85) : " + ("; ".join(f"{a}~{b} ({v:.2f})" for a, b, v in pairs) or "aucune") + ".")
    sk = st["skew"].abs().sort_values(ascending=False)
    add_insight("Variables les plus asymétriques (à log-transformer) : " + ", ".join(f"{k} ({st.loc[k, 'skew']:.0f})" for k in sk.index[:5]) + ".")

    # ------------------------------------------------------------------ 4. Catégorielles
    cat = ["property_type", "room_type", "neighbourhood_cleansed", "neighbourhood_group_cleansed", "license", "host_location", "source"]
    card = []
    for c in cat:
        if c in df:
            vc = df[c].value_counts(normalize=True)
            card.append((c, df[c].nunique(), (vc < 0.01).sum(), vc[vc < 0.01].sum() * 100))
    pd.DataFrame(card, columns=["colonne", "modalités", "modalités <1 %", "% annonces dans modalités <1 %"]).round(1).to_csv(ROOT / "reports" / "eda_categorical.csv", index=False)
    pt = df.property_type.value_counts(normalize=True)
    nb = df.groupby("city").neighbourhood_cleansed.nunique()
    add_insight(f"property_type : {df.property_type.nunique()} modalités, dont {(pt < 0.01).sum()} rares (<1 %) couvrant {pt[pt < 0.01].sum() * 100:.1f} % des annonces → regroupement nécessaire.")
    add_insight(f"Quartiers (neighbourhood_cleansed) : de {nb.min()} ({nb.idxmin()}) à {nb.max()} ({nb.idxmax()}) par ville ; {df.groupby(['city', 'neighbourhood_cleansed']).ngroups} couples ville-quartier "
                "→ forte cardinalité, target encoding avec CV interne.")

    # ------------------------------------------------------------------ 5. Géographie
    ctr = df.groupby("city")[["latitude", "longitude"]].transform("median")
    df["dist_ctr_km"] = haversine(df.latitude, df.longitude, ctr.latitude, ctr.longitude)
    d = df[df.price.notna()]
    p = d[d.city == "paris"]
    pall = df[df.city == "paris"]
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    ax[0].scatter(pall.longitude, pall.latitude, s=1, alpha=0.3, color="#4c78a8")
    ax[0].set(title="Paris : annonces (n = %s)" % f"{len(pall):,}", xlabel="Longitude (°)", ylabel="Latitude (°)")
    pp = p[p.price < p.price.quantile(0.99)].sort_values("price")
    sc = ax[1].scatter(pp.longitude, pp.latitude, c=np.log(pp.price), s=2, cmap="viridis", alpha=0.8)
    plt.colorbar(sc, ax=ax[1], label="log(prix EUR)")
    ax[1].set(title="Paris : log(prix) par annonce", xlabel="Longitude (°)", ylabel="Latitude (°)")
    save("06_map_paris")
    qm = p.groupby("neighbourhood_cleansed").agg(n=("price", "size"), med=("price", "median")).sort_values("med")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(qm.index, qm.med, color="#f58518")
    ax.set(title="Paris : prix médian par arrondissement", xlabel="Prix médian (EUR / nuit)", ylabel="Arrondissement")
    save("07_paris_median_price_by_neighbourhood")
    add_insight(f"Paris : prix médian par arrondissement de {qm.med.min():.0f} € ({qm.med.idxmin()}) à {qm.med.max():.0f} € ({qm.med.idxmax()}), soit un écart de x{qm.med.max() / qm.med.min():.1f}.")
    dc = d.groupby("city").apply(lambda x: x["dist_ctr_km"].corr(x["log_price"], method="spearman"), include_groups=False).sort_values()
    add_insight(f"Corrélation de Spearman distance au centre ↔ log(prix) : de {dc.iloc[0]:+.2f} ({dc.index[0]}) à {dc.iloc[-1]:+.2f} ({dc.index[-1]}) ; "
                f"la distance (centre = médiane des coordonnées, hypothèse) est négativement liée au prix dans {(dc < 0).sum()} villes sur {len(dc)}.")
    fig, ax = plt.subplots(figsize=(10, 5))
    dc.plot.barh(ax=ax, color="#54a24b")
    ax.set(title="Corrélation distance au centre / log(prix) par ville", xlabel="Spearman", ylabel="Ville")
    save("08_distance_price_corr_by_city")

    # ------------------------------------------------------------------ 6. Hôtes
    sh = df.groupby("city").superhost.mean() * 100
    g = df.groupby(["city", "host_id"]).size().rename("n").reset_index()
    df = df.merge(g, on=["city", "host_id"])
    multi = df.groupby("city").apply(lambda x: (x.n >= 5).mean() * 100, include_groups=False)
    add_insight(f"Superhosts : {df.superhost.mean() * 100:.1f} % des annonces (de {sh.min():.0f} % à {sh.idxmin()} à {sh.max():.0f} % à {sh.idxmax()}).")
    add_insight(f"Hôtes professionnels : {(df.n >= 5).mean() * 100:.1f} % des annonces appartiennent à un hôte avec ≥ 5 annonces dans la ville (de {multi.min():.0f} % à {multi.idxmin()} à {multi.max():.0f} % à {multi.idxmax()}) ; "
                f"{(df.n >= 20).mean() * 100:.1f} % à un hôte avec ≥ 20 annonces.")
    nh = df.groupby("city").host_id.nunique()
    add_insight(f"Annonces par hôte : {len(df):,} annonces pour {df.groupby(['city', 'host_id']).ngroups:,} hôtes distincts → un split par annonce ferait fuiter les hôtes ; split groupé par host_id confirmé.")
    sh_v = df.groupby("room_type").superhost.mean() * 100
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    sh.sort_values().plot.barh(ax=ax[0], color="#e45756")
    ax[0].set(title="Part de Superhosts par ville", xlabel="% des annonces", ylabel="Ville")
    multi.sort_values().plot.barh(ax=ax[1], color="#72b7b2")
    ax[1].set(title="Annonces d'hôtes avec ≥ 5 annonces", xlabel="% des annonces", ylabel="Ville")
    save("09_hosts")
    dd = d.merge(g, on=["city", "host_id"])
    add_insight(f"Les hôtes multi-annonces (≥ 5) ont un prix relatif médian de {dd[dd.n >= 5].price_rel.median():.2f} contre {dd[dd.n < 5].price_rel.median():.2f} pour les autres.")
    ts = df.groupby("superhost")["hosts_time_as_host_years"].median()
    add_insight(f"Ancienneté médiane de l'hôte : {ts[1]:.1f} ans (Superhost) contre {ts[0]:.1f} ans (non Superhost).")

    # ------------------------------------------------------------------ 7. Avis
    nr = (df.number_of_reviews == 0).mean() * 100
    rs = df.review_scores_rating
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].hist(df.number_of_reviews.clip(upper=df.number_of_reviews.quantile(0.99)), bins=60, color="#4c78a8")
    ax[0].set(title="Nombre d'avis par annonce (écrêté au 99e centile)", xlabel="Nombre d'avis", ylabel="Nombre d'annonces", yscale="log")
    ax[1].hist(rs.dropna(), bins=50, color="#f58518")
    ax[1].set(title="Note globale (review_scores_rating)", xlabel="Note (0-5)", ylabel="Nombre d'annonces")
    save("10_reviews")
    add_insight(f"Avis : {nr:.1f} % des annonces n'ont aucun avis ; note moyenne {rs.mean():.2f}, médiane {rs.median():.2f}, {(rs >= 4.5).sum() / rs.notna().sum() * 100:.0f} % des notes ≥ 4,5 : forte positivité, peu discriminante.")
    add_insight(f"Note ({rs.notna().mean() * 100:.0f} % renseignée) : Superhost {df.groupby('superhost').review_scores_rating.mean()[1]:.2f} contre {df.groupby('superhost').review_scores_rating.mean()[0]:.2f} pour les non-Superhosts.")
    # langue des avis (échantillon)
    from langdetect import DetectorFactory, detect

    DetectorFactory.seed = SEED
    random.seed(SEED)
    pf = pq.ParquetFile(ROOT / cfg["paths"]["reviews_raw"])
    smp = []
    for i in range(pf.num_row_groups):
        t = pf.read_row_group(i, columns=["city", "comments"]).to_pandas()
        smp.append(t.sample(min(500, len(t)), random_state=SEED))
    smp = pd.concat(smp).dropna(subset=["comments"])
    smp = smp[smp.comments.str.len() > 20]

    def det(s):
        try:
            return detect(s)
        except Exception:
            return "unk"

    smp["lang"] = smp.comments.map(det)
    lg = smp.lang.value_counts(normalize=True) * 100
    smp.groupby("city").lang.agg(lambda s: s.value_counts(normalize=True).head(3).round(2).to_dict()).to_csv(ROOT / "reports" / "eda_review_langs_by_city.csv")
    add_insight(f"Langues des avis (échantillon de {len(smp):,} avis, langdetect) : " + ", ".join(f"{k} {v:.0f} %" for k, v in lg.head(6).items()) + f" ; {(lg < 1).sum()} langues < 1 % (multilingue → modèles multilingues en phase 9).")

    # ------------------------------------------------------------------ 8. Disponibilité / annonces fantômes
    df["ghost_a"] = df.availability_365 == 0
    df["ghost_b"] = df.ghost_a & (df.number_of_reviews_ltm == 0)
    gh = df.groupby("city")[["ghost_a", "ghost_b"]].mean() * 100
    fig, ax = plt.subplots(figsize=(10, 6))
    gh.sort_values("ghost_b").plot.barh(ax=ax, color=["#bab0ac", "#e45756"])
    ax.set(title="Annonces fantômes par ville", xlabel="% des annonces", ylabel="Ville")
    ax.legend(["availability_365 = 0", "… et aucun avis sur 12 mois"])
    save("11_ghost_listings")
    add_insight(f"Annonces fantômes : {df.ghost_a.mean() * 100:.1f} % ont availability_365 = 0 ; {df.ghost_b.mean() * 100:.1f} % cumulent indisponibilité ET 0 avis sur 12 mois "
                f"(de {gh.ghost_b.min():.0f} % à {gh.ghost_b.idxmin()} à {gh.ghost_b.max():.0f} % à {gh.ghost_b.idxmax()}).")
    add_insight(f"`source` = « previous scrape » pour {df.source.notna().mean() * 100:.0f} % des annonces (non rafraîchies lors de la dernière collecte) ; "
                f"parmi elles, {df[df.source.notna()].price.isna().mean() * 100:.0f} % n'ont pas de prix contre {df[df.source.isna()].price.isna().mean() * 100:.0f} % pour les autres.")

    # ------------------------------------------------------------------ 9. Pièges de fuite
    leak = []
    dl = df[df.price.notna()]
    for c in ["estimated_revenue_l365d", "price_quote_price_per_night", "price_quote_total_price"]:
        if c in dl:
            leak.append((c, dl[c].corr(dl.price, method="spearman")))
    lk = pd.DataFrame(leak, columns=["colonne", "spearman_vs_price"]).round(3)
    lk.to_csv(ROOT / "reports" / "eda_leak_check.csv", index=False)
    rev_resid = (dl.estimated_revenue_l365d - dl.price * dl.estimated_occupancy_l365d).abs()
    ok = dl[dl.estimated_occupancy_l365d > 0]
    r = (ok.estimated_revenue_l365d / (ok.price * ok.estimated_occupancy_l365d)).round(2)
    add_insight(f"FUITE (régression) : estimated_revenue_l365d = prix x occupation estimée : ratio = 1,00 pour {(r == 1).mean() * 100:.0f} % des annonces avec occupation > 0 → à exclure ; "
                f"price_quote_* est le prix lui-même (Spearman {lk.spearman_vs_price.iloc[1]:.2f} avec `price`) → à exclure.")
    sh_df = df.dropna(subset=["review_scores_rating"])
    from sklearn.metrics import roc_auc_score

    auc = roc_auc_score(sh_df.superhost, sh_df.review_scores_rating)
    auc2 = roc_auc_score(df.superhost, df.number_of_reviews_ltm)
    add_insight(f"RISQUE DE FUITE (Superhost) : la seule note review_scores_rating donne un AUC de {auc:.2f} et number_of_reviews_ltm de {auc2:.2f} pour prédire is_superhost ; "
                "ces variables sont à la fois critères d'attribution et conséquences du statut → à tester avec/sans en phase 7.")
    add_insight("Colonnes connues seulement après coup / dérivées de la cible : estimated_revenue_l365d, estimated_occupancy_l365d, price_quote_*, price_rel_city_median (si recréé), "
                "availability_* et reviews_per_month (mesurent l'activité récente, pas les caractéristiques du logement).")

    # ------------------------------------------------------------------ écriture
    head = ["# Insights EDA (Phase 1)", "",
            f"Base : {len(df):,} annonces, {df.city.nunique()} villes, une date de collecte par ville (sept. 2026). Tous les chiffres sont calculés par `src/eda.py`.",
            "Prix en devise locale ; aucune conversion tant que les taux ne sont pas validés. Centre-ville = médiane des coordonnées (hypothèse).", ""]
    body = [f"{i}. {t}" for i, t in enumerate(INS, 1)]
    (ROOT / "reports" / "eda_insights.md").write_text("\n".join(head + body) + "\n", encoding="utf-8")
    print(f"\n{len(INS)} insights écrits.")


if __name__ == "__main__":
    run()
