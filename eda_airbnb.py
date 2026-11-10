#!/usr/bin/env python3
"""Analyse exploratoire de la base globale : figures dans reports/figures et rapport reports/eda_report.md.

Prérequis : merge_airbnb.py puis clean_airbnb.py.
Les prix sont en devise locale (non convertis) : les comparaisons entre villes passent par
price_rel_city_median (prix / médiane de la ville).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

log = logging.getLogger("eda")

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 14, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "font.size": 11, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
})


def save(fig, fig_dir: Path, name: str) -> str:
    fig.tight_layout()
    fig.savefig(fig_dir / name, dpi=140)
    plt.close(fig)
    return f"figures/{name}"


def hbar(ax, labels, values, fmt="{:,.0f}", color=BLUE, ref=None):
    """Barres horizontales fines avec valeurs en étiquettes directes ; ax.grid sur l'axe des valeurs seulement."""
    y = np.arange(len(labels))
    ax.barh(y, values, height=0.6, color=color)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    top = max(values) if len(values) else 1
    for yi, v in zip(y, values):
        ax.text(v + top * 0.01, yi, fmt.format(v), va="center", fontsize=10, color=INK_2)
    ax.set_xlim(0, top * 1.12)


# --------------------------------------------------------------------------- tables

def city_table(df: pd.DataFrame, n_reviews: pd.Series) -> pd.DataFrame:
    g = df.groupby("city")
    t = pd.DataFrame({
        "pays": g["country"].first(),
        "devise": g["currency"].first(),
        "annonces": g.size(),
        "avis": n_reviews.reindex(g.size().index).fillna(0).astype(int),
        "prix_median": g["price"].median().round(0),
        "pct_prix_dispo": (g["has_price"].mean() * 100).round(1),
        "pct_logement_entier": (g["room_type"].apply(lambda s: (s == "Entire home/apt").mean()) * 100).round(1),
        "pct_hotes_multi": (g["calculated_host_listings_count"].apply(lambda s: (s > 1).mean()) * 100).round(1),
        "note_mediane": g["review_scores_rating"].median().round(2),
        "date_collecte": g["scrape_date"].first().dt.date.astype(str),
    })
    return t.sort_values("annonces", ascending=False)


def missing_pct_by_column(path: Path) -> pd.Series:
    pf = pq.ParquetFile(path)
    n = pf.metadata.num_rows
    nulls = {name: 0 for name in pf.schema_arrow.names}
    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg)
        for name in nulls:
            nulls[name] += table.column(name).null_count
    return (pd.Series(nulls) / n * 100).sort_values(ascending=False)


def reviews_summary(path: Path) -> tuple[pd.Series, pd.Series]:
    """(nombre d'avis par ville, nombre d'avis par année) en ne lisant que deux colonnes."""
    t = pq.read_table(path, columns=["city", "date"])
    by_city = t.group_by("city").aggregate([("city", "count")]).to_pandas().set_index("city")["city_count"]
    years = pc.year(t.column("date"))
    by_year = pd.Series(pc.value_counts(years).to_pylist()).apply(pd.Series).set_index("values")["counts"].sort_index()
    return by_city, by_year


# --------------------------------------------------------------------------- figures

def fig_missing(miss: pd.Series, fig_dir: Path) -> str:
    top = miss[miss > 5].head(25)
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(top) + 1.4))
    hbar(ax, top.index, top.values, fmt="{:.0f} %")
    ax.set_title("Valeurs manquantes par colonne (brut, > 5 %)")
    return save(fig, fig_dir, "01_missing.png")


def fig_counts(t: pd.DataFrame, fig_dir: Path) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.42 * len(t) + 1.6), sharey=False)
    hbar(axes[0], t.index, t["annonces"].values)
    axes[0].set_title("Annonces par ville")
    hbar(axes[1], t.index, (t["avis"] / 1e6).values, fmt="{:.2f} M")
    axes[1].set_title("Avis par ville")
    return save(fig, fig_dir, "02_counts.png")


def fig_price_dist(df: pd.DataFrame, fig_dir: Path) -> str:
    rel = df["price_rel_city_median"].dropna()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(np.log10(rel), bins=80, color=BLUE, edgecolor=SURFACE, linewidth=0.4)
    ticks = [0.1, 0.25, 0.5, 1, 2, 4, 10]
    ax.set_xticks(np.log10(ticks), [f"x{t:g}" for t in ticks])
    ax.set_xlabel("Prix / médiane de la ville (échelle logarithmique)")
    ax.set_ylabel("Annonces")
    ax.grid(axis="x", visible=False)
    ax.set_title("Distribution des prix relatifs, toutes villes")
    return save(fig, fig_dir, "04_price_dist.png")


def fig_room_type(df: pd.DataFrame, fig_dir: Path) -> str:
    med = df.groupby("room_type")["price_rel_city_median"].median().sort_values(ascending=False)
    counts = df["room_type"].value_counts()
    fig, ax = plt.subplots(figsize=(9, 3.2))
    hbar(ax, [f"{r} (n={counts[r]:,})" for r in med.index], med.values, fmt="x{:.2f}")
    ax.set_title("Prix relatif médian par type de logement (1 = médiane de la ville)")
    return save(fig, fig_dir, "05_room_type.png")


def fig_correlations(df: pd.DataFrame, fig_dir: Path) -> tuple[str, pd.Series]:
    num = df.select_dtypes(include=["number", "boolean"]).drop(
        columns=["id", "host_id", "latitude", "longitude", "price", "price_rel_city_median",
                 "estimated_revenue_l365d"], errors="ignore")
    num = num.loc[:, num.nunique() > 1].astype(float)
    corr = num.corrwith(np.log(df["price_rel_city_median"]), method="spearman").dropna()
    corr = corr.reindex(corr.abs().sort_values(ascending=False).index).head(18)
    fig, ax = plt.subplots(figsize=(9, 0.34 * len(corr) + 1.4))
    y = np.arange(len(corr))
    ax.barh(y, corr.values, height=0.6, color=[BLUE if v >= 0 else ORANGE for v in corr.values])
    ax.set_yticks(y, corr.index)
    ax.invert_yaxis()
    ax.axvline(0, color=INK_2, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    for yi, v in zip(y, corr.values):
        ax.text(max(v, 0) + 0.01, yi, f"{v:+.2f}", va="center", fontsize=10, color=INK_2)
    ax.set_xlabel("Corrélation de Spearman (bleu : positive, orange : négative)")
    ax.set_title("Corrélation avec le prix relatif")
    return save(fig, fig_dir, "06_correlations.png"), corr


def fig_rating(df: pd.DataFrame, fig_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(df["review_scores_rating"].dropna(), bins=np.arange(1, 5.06, 0.05), color=BLUE, edgecolor=SURFACE, linewidth=0.4)
    ax.set_xlabel("Note globale")
    ax.set_ylabel("Annonces")
    ax.grid(axis="x", visible=False)
    ax.set_title("Distribution des notes : très concentrée près de 5")
    return save(fig, fig_dir, "07_rating.png")


def fig_reviews_year(by_year: pd.Series, fig_dir: Path) -> str:
    by_year = by_year[by_year.index >= 2010]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(by_year.index, by_year.values / 1e6, color=BLUE, linewidth=2, marker="o", markersize=5)
    ax.set_ylabel("Avis (millions)")
    ax.set_title("Avis publiés par année, toutes villes")
    return save(fig, fig_dir, "08_reviews_year.png")


def fig_quality(t: pd.DataFrame, fig_dir: Path) -> str:
    t = t.sort_values("pct_prix_dispo")
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(t) + 1.6))
    hbar(ax, t.index, t["pct_prix_dispo"].values, fmt="{:.0f} %")
    ax.set_xlim(0, 115)
    ax.set_title("Part des annonces avec un prix renseigné")
    return save(fig, fig_dir, "09_price_available.png")


# --------------------------------------------------------------------------- rapport

def md_table(t: pd.DataFrame) -> str:
    t = t.reset_index()
    head = "| " + " | ".join(t.columns) + " |\n|" + "|".join("---" for _ in t.columns) + "|\n"
    rows = "\n".join("| " + " | ".join(f"{v:,}" if isinstance(v, (int, np.integer)) else str(v) for v in r) + " |"
                     for r in t.itertuples(index=False))
    return head + rows


def build_report(df, t, miss, corr, figs, n_reviews, clean_log: str) -> str:
    stale = df["is_stale"].mean() * 100
    top_pos, top_neg = corr[corr > 0].head(3), corr[corr < 0].head(3)
    lines = [
        "# Analyse exploratoire - Inside Airbnb (base globale)", "",
        f"{len(df):,} annonces, {n_reviews:,} avis, {df['city'].nunique()} villes, "
        f"{df.shape[1]} colonnes après nettoyage.", "",
        "## Vue d'ensemble par ville", "", md_table(t), "",
        "Prix en devise locale ; `pct_prix_dispo` = part des annonces avec un prix ; `pct_hotes_multi` = part des "
        "annonces dont l'hôte en possède plusieurs.", "",
        f"![Annonces et avis]({figs['counts']})", "",
        "Les prix médians du tableau sont en devise locale : pas de comparaison directe entre devises (pas de graphique commun).", "",
        "## Qualité des données", "",
        f"- {df['has_price'].mean() * 100:.1f} % des annonces ont un prix ; {stale:.1f} % proviennent d'une "
        "collecte précédente (`is_stale`) et n'ont en général ni prix ni revenu estimé.",
        f"- {miss.gt(99.9).sum()} colonnes sont vides à 100 % dans la source et ont été retirées.",
        f"- {df['is_price_outlier'].sum():,} prix extrêmes et {df['is_geo_outlier'].sum():,} coordonnées aberrantes sont signalés, non supprimés.", "",
        f"![Valeurs manquantes]({figs['missing']})", "",
        f"![Prix disponible]({figs['quality']})", "",
        "## Prix", "",
        f"![Distribution des prix]({figs['price_dist']})", "",
        f"![Type de logement]({figs['room_type']})", "",
        "## Facteurs associés au prix", "",
        "Les variables les plus liées au prix relatif (Spearman) : "
        + ", ".join(f"`{k}` ({v:+.2f})" for k, v in top_pos.items())
        + (" ; liens négatifs : " + ", ".join(f"`{k}` ({v:+.2f})" for k, v in top_neg.items()) if len(top_neg) else "")
        + ". Ce sont des associations, pas des effets causaux.", "",
        f"![Corrélations]({figs['corr']})", "",
        "## Avis et notes", "",
        f"![Notes]({figs['rating']})", "",
        f"![Avis par année]({figs['reviews_year']})", "",
        "## Journal de nettoyage", "", clean_log,
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--report-dir", type=Path, default=Path("reports"))
    args = p.parse_args()
    fig_dir = args.report_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    proc = args.processed_dir

    df = pd.read_parquet(proc / "listings_clean.parquet")
    by_city, by_year = reviews_summary(proc / "reviews_clean.parquet")
    t = city_table(df, by_city)
    miss = missing_pct_by_column(proc / "listings_all.parquet")
    model = df[df["has_price"] & ~df["is_price_outlier"]]

    figs = {"missing": fig_missing(miss, fig_dir), "counts": fig_counts(t, fig_dir),
            "price_dist": fig_price_dist(model, fig_dir),
            "room_type": fig_room_type(model, fig_dir), "rating": fig_rating(df, fig_dir),
            "reviews_year": fig_reviews_year(by_year, fig_dir), "quality": fig_quality(t, fig_dir)}
    figs["corr"], corr = fig_correlations(model, fig_dir)

    clean_log = (proc / "cleaning_log.md").read_text(encoding="utf-8").split("\n", 2)[-1]
    report = build_report(df, t, miss, corr, figs, int(by_city.sum()), clean_log)
    (args.report_dir / "eda_report.md").write_text(report, encoding="utf-8")
    t.to_csv(args.report_dir / "city_summary.csv", encoding="utf-8")
    log.info("Rapport : %s", args.report_dir / "eda_report.md")


if __name__ == "__main__":
    main()
