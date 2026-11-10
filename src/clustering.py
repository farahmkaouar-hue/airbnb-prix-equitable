"""Phase 8 : apprentissage non supervisé (clustering, réduction de dimension, anomalies).
Variables RELATIVES à la ville (prix relatif, rang de distance au centre) pour que les clusters ne reproduisent pas simplement les villes.
Échantillon : 60 000 annonces avec prix (train, jeu in_reg_set) pour k-means / PCA / anomalies ; 10 000 pour le hiérarchique ; 20 000 pour UMAP.
Usage : python -m src.clustering
"""
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.data import ROOT

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
FIG = ROOT / "reports" / "figures"
SEED = 42
FEATS = {"log_price_rel": "prix relatif (log, vs médiane ville)", "accommodates": "capacité", "bedrooms": "chambres", "bath_count": "salles de bain",
         "n_amenities": "nb équipements", "dist_rank": "éloignement du centre (rang dans la ville)", "log_host_listings": "log annonces de l'hôte",
         "is_entire": "logement entier", "review_scores_rating": "note globale", "log_n_reviews": "log nb d'avis", "availability_365": "jours disponibles/an",
         "host_tenure_years": "ancienneté hôte"}


def load():
    d = pd.read_parquet(ROOT / "data/processed/listings_features.parquet")
    d = d[~d["is_test"] & d["in_reg_set"]].copy()
    d["log_price_rel"] = d["log_price_eur"] - d.groupby("city")["log_price_eur"].transform("median")
    d["dist_rank"] = d.groupby("city")["dist_center_km"].rank(pct=True)
    d["is_entire"] = (d["room_type"] == "Entire home/apt").astype(float)
    return d


LABELS = {"log_price_rel": ("prix élevés", "prix bas"), "accommodates": ("grande capacité", "petits logements"), "n_amenities": ("bien équipés", "peu équipés"),
          "dist_rank": ("périphérie", "centre"), "log_host_listings": ("hôtes multi-annonces", "hôtes particuliers"), "is_entire": ("logements entiers", "chambres chez l'habitant"),
          "review_scores_rating": ("très bien notés", "mal notés"), "log_n_reviews": ("beaucoup d'avis", "peu d'avis"), "availability_365": ("très disponibles", "très demandés"),
          "host_tenure_years": ("hôtes anciens", "hôtes récents")}


def name_cluster(z: pd.Series, entire_share: float) -> str:
    """Nom parlant : les 3 caractéristiques les plus écartées de la médiane globale (|z| décroissant), avec leur sens."""
    z = z[list(LABELS)].copy()
    z["is_entire"] = (entire_share - 0.8) * 4  # part de logements entiers (médiane 1 : on mesure l'écart à ~80 %)
    top = z.abs().sort_values(ascending=False).index[:3]
    return " · ".join(LABELS[c][0] if z[c] > 0 else LABELS[c][1] for c in top)


def run():
    d = load()
    S = d.sample(60000, random_state=SEED).reset_index(drop=True)
    cols = list(FEATS)
    X = S[cols].copy()
    X = X.fillna(X.median())
    sc = StandardScaler().fit(X)
    Z = sc.transform(X)
    out = {}

    # ---------------- choix de k (k-means)
    ks = range(2, 13)
    inertia, sil, dbi = [], [], []
    rs = np.random.RandomState(SEED).choice(len(Z), 10000, replace=False)
    for k in ks:
        km = KMeans(k, n_init=5, random_state=SEED).fit(Z)
        inertia.append(km.inertia_)
        sil.append(silhouette_score(Z[rs], km.labels_[rs]))
        dbi.append(davies_bouldin_score(Z, km.labels_))
    sel = pd.DataFrame(dict(k=list(ks), inertie=inertia, silhouette=sil, davies_bouldin=dbi))
    sel.to_csv(ROOT / "reports" / "clust_k_selection.csv", index=False)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].plot(sel.k, sel.inertie, "o-"); ax[0].set(title="Coude (inertie)", xlabel="k", ylabel="Inertie intra-cluster")
    ax[1].plot(sel.k, sel.silhouette, "o-", color="#54a24b"); ax[1].set(title="Silhouette (plus haut = mieux)", xlabel="k", ylabel="Silhouette moyenne")
    ax[2].plot(sel.k, sel.davies_bouldin, "o-", color="#e45756"); ax[2].set(title="Davies-Bouldin (plus bas = mieux)", xlabel="k", ylabel="Indice")
    plt.tight_layout(); plt.savefig(FIG / "clu_01_choice_of_k.png", dpi=130); plt.close()
    # choix : k maximisant la silhouette parmi k >= 4 (au moins 4 profils pour une lecture métier)
    k_best = int(sel[sel.k >= 4].sort_values("silhouette", ascending=False).k.iloc[0])
    km = KMeans(k_best, n_init=10, random_state=SEED).fit(Z)
    S["cluster"] = km.labels_

    # ---------------- profils et noms
    med = S.groupby("cluster")[cols].median()
    glob_std = X.std()
    zprof = (med - X.median()) / glob_std
    ent = S.groupby("cluster")["is_entire"].mean()
    names = {c: name_cluster(zprof.loc[c], ent[c]) for c in zprof.index}
    # unicité des noms
    seen = {}
    for c, n in names.items():
        seen[n] = seen.get(n, 0) + 1
        if seen[n] > 1:
            names[c] = f"{n} ({seen[n]})"
    S["cluster_name"] = S["cluster"].map(names)
    prof = S.groupby("cluster_name").agg(annonces=("id", "size"), part_pct=("id", lambda s: len(s) / len(S) * 100), prix_median_eur=("price_eur", "median"),
                                         prix_relatif=("log_price_rel", lambda s: float(np.exp(s.median()))), capacité=("accommodates", "median"), chambres=("bedrooms", "median"),
                                         logements_entiers_pct=("is_entire", lambda s: s.mean() * 100), dist_centre_rang=("dist_rank", "median"),
                                         annonces_hôte=("calculated_host_listings_count", "median"), note=("review_scores_rating", "median"),
                                         jours_dispo=("availability_365", "median"), superhosts_pct=("is_superhost", lambda s: s.mean() * 100)).round(2)
    prof.sort_values("prix_median_eur", ascending=False).to_csv(ROOT / "reports" / "clust_profiles.csv")
    comp = pd.crosstab(S["cluster_name"], S["city"], normalize="columns") * 100
    comp.round(1).to_csv(ROOT / "reports" / "clust_city_composition.csv")
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(zprof.rename(index=names).rename(columns=FEATS), annot=True, fmt=".1f", cmap="RdBu_r", center=0, ax=ax, cbar_kws={"label": "écart à la médiane globale (en écarts-types)"})
    ax.set(title=f"Profils des {k_best} clusters d'annonces (k-means)", xlabel="Variable", ylabel="Cluster")
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout(); plt.savefig(FIG / "clu_02_profiles.png", dpi=130); plt.close()
    fig, ax = plt.subplots(figsize=(14, 6))
    comp.T.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set(title="Composition des villes par cluster (chaque barre = 100 % des annonces de la ville)", xlabel="Ville", ylabel="% des annonces")
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1))
    plt.tight_layout(); plt.savefig(FIG / "clu_03_city_composition.png", dpi=130); plt.close()
    out["k_best"] = k_best

    # ---------------- hiérarchique (Ward) sur 10 000 annonces + accord avec k-means ; GMM (BIC)
    h = np.random.RandomState(SEED).choice(len(Z), 10000, replace=False)
    L = linkage(Z[h], method="ward")
    hl = fcluster(L, k_best, criterion="maxclust") - 1
    out["ari_ward_kmeans"] = float(adjusted_rand_score(S.loc[h, "cluster"], hl))
    out["sil_ward"] = float(silhouette_score(Z[h], hl))
    fig, ax = plt.subplots(figsize=(12, 5))
    dendrogram(L, truncate_mode="lastp", p=30, ax=ax, color_threshold=0)
    ax.set(title="Dendrogramme (Ward, 10 000 annonces, 30 dernières fusions)", xlabel="Groupes (taille entre parenthèses)", ylabel="Distance de Ward")
    plt.tight_layout(); plt.savefig(FIG / "clu_04_dendrogram.png", dpi=130); plt.close()
    pcs = PCA(8, random_state=SEED).fit_transform(Z)
    bic = []
    for k in range(2, 11):
        g = GaussianMixture(k, covariance_type="full", random_state=SEED, n_init=2).fit(pcs[:30000])
        bic.append(dict(k=k, bic=g.bic(pcs[:30000])))
    bic = pd.DataFrame(bic)
    bic.to_csv(ROOT / "reports" / "clust_gmm_bic.csv", index=False)
    out["gmm_k_bic"] = int(bic.sort_values("bic").k.iloc[0])
    gm = GaussianMixture(k_best, random_state=SEED, n_init=2).fit(pcs[:30000])
    out["ari_gmm_kmeans"] = float(adjusted_rand_score(S["cluster"], gm.predict(pcs)))

    # ---------------- PCA : variance expliquée et composantes
    pca = PCA(random_state=SEED).fit(Z)
    ev = pd.Series(pca.explained_variance_ratio_, index=[f"PC{i + 1}" for i in range(len(cols))])
    loadings = pd.DataFrame(pca.components_[:4].T, index=[FEATS[c] for c in cols], columns=[f"PC{i + 1}" for i in range(4)])
    loadings.to_csv(ROOT / "reports" / "clust_pca_loadings.csv")
    ev.to_csv(ROOT / "reports" / "clust_pca_variance.csv")
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].bar(ev.index, ev.values * 100, color="#4c78a8"); ax[0].plot(ev.index, ev.cumsum().values * 100, "o-", color="#e45756")
    ax[0].set(title="PCA : variance expliquée (barres) et cumulée (ligne)", xlabel="Composante", ylabel="% de variance")
    sns.heatmap(loadings, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax[1], cbar=False)
    ax[1].set(title="PCA : poids des variables sur les 4 premières composantes", xlabel="Composante", ylabel="Variable")
    plt.tight_layout(); plt.savefig(FIG / "clu_05_pca.png", dpi=130); plt.close()
    out["pca_var_top2"] = float(ev.iloc[:2].sum()); out["pca_n_for_80"] = int((ev.cumsum() < 0.8).sum() + 1)

    # ---------------- UMAP (20 000) coloré par ville, prix, cluster
    import umap

    u = np.random.RandomState(SEED).choice(len(Z), 20000, replace=False)
    emb = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=SEED).fit_transform(Z[u])
    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    for a, (col, t, cm) in zip(ax, [(S.loc[u, "city"].astype("category").cat.codes, "ville", "tab20"), (S.loc[u, "log_price_rel"], "log prix relatif à la ville", "viridis"),
                                    (S.loc[u, "cluster"], "cluster k-means", "tab10")]):
        sc_ = a.scatter(emb[:, 0], emb[:, 1], c=col, s=2, cmap=cm, alpha=0.6)
        a.set(title=f"UMAP coloré par {t}", xlabel="UMAP 1", ylabel="UMAP 2")
        if t == "log prix relatif à la ville":
            plt.colorbar(sc_, ax=a, label="log(prix / médiane ville)")
    plt.tight_layout(); plt.savefig(FIG / "clu_06_umap.png", dpi=120); plt.close()

    # ---------------- DBSCAN géographique (Paris) : zones denses
    P = d[d.city == "paris"][["latitude", "longitude", "log_price_rel"]].dropna()
    coords = np.radians(P[["latitude", "longitude"]].values)
    db = DBSCAN(eps=0.15 / 6371, min_samples=40, metric="haversine", algorithm="ball_tree").fit(coords)
    P["zone"] = db.labels_
    n_zones = int((P.zone >= 0).sum() and P.zone.max() + 1)
    out.update(db_zones=n_zones, db_noise_pct=float((P.zone == -1).mean() * 100), db_n=len(P))
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(P.longitude[P.zone == -1], P.latitude[P.zone == -1], s=1, c="lightgrey", label="isolées (bruit)")
    m = P.zone >= 0
    ax.scatter(P.longitude[m], P.latitude[m], s=2, c=P.zone[m] % 20, cmap="tab20", alpha=0.7)
    ax.set(title=f"Paris : zones denses détectées par DBSCAN (rayon 150 m, >= 40 annonces) : {n_zones} zones", xlabel="Longitude (°)", ylabel="Latitude (°)")
    ax.legend(markerscale=6)
    plt.tight_layout(); plt.savefig(FIG / "clu_07_dbscan_paris.png", dpi=130); plt.close()
    top_zones = P[P.zone >= 0].groupby("zone").agg(annonces=("zone", "size"), prix_rel=("log_price_rel", lambda s: float(np.exp(s.median())))).sort_values("annonces", ascending=False).head(8)
    top_zones.to_csv(ROOT / "reports" / "clust_dbscan_paris_top.csv")

    # ---------------- clustering des quartiers
    q = d.groupby(["city", "neighbourhood_cleansed"]).agg(n=("id", "size"), prix_rel=("log_price_rel", "median"), dist=("dist_rank", "median"), entier=("is_entire", "mean"),
                                                         pro=("host_is_pro", "mean"), note=("review_scores_rating", "median"), dispo=("availability_365", "median"),
                                                         capacité=("accommodates", "median")).reset_index()
    q = q[q.n >= 30].dropna()
    Zq = StandardScaler().fit_transform(q[["prix_rel", "dist", "entier", "pro", "note", "dispo", "capacité"]])
    kq = KMeans(5, n_init=10, random_state=SEED).fit(Zq)
    q["groupe"] = kq.labels_
    qp = q.groupby("groupe").agg(quartiers=("n", "size"), annonces=("n", "sum"), prix_rel=("prix_rel", lambda s: float(np.exp(s.median()))), dist=("dist", "median"), entier=("entier", "median"),
                                 pro=("pro", "median"), dispo=("dispo", "median")).round(2)
    qn = {}
    for g, r in qp.iterrows():
        qn[g] = ("chers" if r.prix_rel > 1.15 else "bon marché" if r.prix_rel < 0.85 else "prix moyens") + (" · centraux" if r.dist < 0.4 else " · périphériques" if r.dist > 0.6 else "") + (" · très pros" if r.pro > 0.5 else "")
    q["groupe_nom"] = q["groupe"].map(qn)
    qp["nom"] = qp.index.map(qn)
    q.to_csv(ROOT / "reports" / "clust_neighbourhoods.csv", index=False)
    qp.to_csv(ROOT / "reports" / "clust_neighbourhood_groups.csv")
    out["n_quartiers"] = len(q)

    # ---------------- anomalies (Isolation Forest)
    iso = IsolationForest(n_estimators=300, contamination=0.01, random_state=SEED, n_jobs=-1).fit(Z)
    S["anomaly_score"] = -iso.score_samples(Z)
    S["is_anomaly"] = iso.predict(Z) == -1
    an = S[S.is_anomaly]
    cmp = pd.DataFrame({"anomalies (1 %)": an[cols].median(), "ensemble": S[cols].median()})
    cmp.index = [FEATS[c] for c in cmp.index]
    cmp.round(2).to_csv(ROOT / "reports" / "clust_anomalies_profile.csv")
    an.sort_values("anomaly_score", ascending=False)[["city", "id", "price_eur", "accommodates", "bedrooms", "calculated_host_listings_count", "availability_365", "anomaly_score"]].head(15).to_csv(
        ROOT / "reports" / "clust_anomalies_top.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(S.anomaly_score, bins=80, color="#4c78a8")
    ax.axvline(S.loc[S.is_anomaly, "anomaly_score"].min(), color="#e45756", ls="--", label="seuil (1 % les plus atypiques)")
    ax.set(title="Isolation Forest : score d'atypie des annonces", xlabel="Score d'anomalie (plus haut = plus atypique)", ylabel="Nombre d'annonces")
    ax.legend()
    plt.tight_layout(); plt.savefig(FIG / "clu_08_anomalies.png", dpi=130); plt.close()
    S[["city", "id", "cluster", "cluster_name", "anomaly_score", "is_anomaly"]].to_parquet(ROOT / "data/processed/clusters_sample.parquet", index=False)
    pd.Series(out).to_json(ROOT / "reports" / "clust_summary.json", indent=1)
    print(out)
    print(prof.sort_values("prix_median_eur", ascending=False).to_string())
    print(qp.to_string())


if __name__ == "__main__":
    run()
