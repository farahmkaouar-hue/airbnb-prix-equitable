#!/usr/bin/env python3
"""Fusionne les listings / reviews par ville en Parquet et génère un rapport de données.

Aucune ligne n'est supprimée : seules des colonnes sont ajoutées et le prix est converti en float.

Exemple :
    python merge_airbnb.py
    python merge_airbnb.py --cities paris --output-dir data/processed_test
"""
from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from download_airbnb import CITIES

log = logging.getLogger("merge")

CURRENCY_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY", "R$": "BRL", "CHF": "CHF"}
SYMBOL_RE = re.compile(r"^\s*([^\d\s.,-]+)")
QUOTE_CURRENCY_RE = re.compile(r'"currency":\s*"([A-Z]{3})"')
REVIEW_CHUNK = 500_000


@dataclass
class CityStats:
    city: str
    country: str
    scrape_date: str
    n_listings: int = 0
    n_reviews: int = 0
    pct_missing: float = float("nan")
    currency: str = ""
    absent_columns: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- découverte

def find_runs(raw_dir: Path, only: set[str] | None) -> list[tuple[str, str, Path]]:
    """Liste (slug ville, date, dossier) pour chaque collecte contenant un listings.csv.gz."""
    runs = []
    for listing in sorted(raw_dir.glob("*/*/listings.csv.gz")):
        date_dir = listing.parent
        slug = date_dir.parent.name
        if only is None or slug in only:
            runs.append((slug, date_dir.name, date_dir))
    return runs


def country_of(slug: str) -> str:
    for country, city_slug in CITIES.values():
        if city_slug == slug:
            return country
    return ""


# --------------------------------------------------------------------------- nettoyage

def currency_from_df(df: pd.DataFrame) -> str:
    """Devise ISO lue dans price_quote_raw (fiable) ; sinon déduite du symbole du prix ('' si inconnue)."""
    if "price_quote_raw" in df:
        codes = df["price_quote_raw"].dropna().astype(str).str.extract(QUOTE_CURRENCY_RE)[0].dropna()
        if not codes.empty:
            return codes.mode().iloc[0]
    symbols = df["price"].dropna().astype(str).str.extract(SYMBOL_RE)[0].dropna()
    if symbols.empty:
        return ""
    top = symbols.mode().iloc[0]
    return CURRENCY_SYMBOLS.get(top, top)


def clean_price(raw: pd.Series) -> pd.Series:
    """'$1,250.00' -> 1250.0 (retire symboles et séparateurs de milliers)."""
    text = raw.astype("string").str.replace(r"[^\d.\-]", "", regex=True)
    return pd.to_numeric(text, errors="coerce").astype("float64")


def stringify_mixed_objects(df: pd.DataFrame) -> pd.DataFrame:
    """Colonnes de types mélangés entre villes (ex. int dans l'une, texte dans l'autre) -> texte."""
    for col in df.columns[df.dtypes == object]:
        if df[col].map(type).nunique() > 2:  # str + NaN attendus ; au-delà = types mélangés
            df[col] = df[col].map(lambda v: v if pd.isna(v) or isinstance(v, str) else str(v))
    return df


# --------------------------------------------------------------------------- listings

def load_listings(slug: str, date: str, folder: Path) -> tuple[pd.DataFrame, CityStats]:
    df = pd.read_csv(folder / "listings.csv.gz", low_memory=False)
    stats = CityStats(slug, country_of(slug), date, n_listings=len(df))
    stats.pct_missing = float(df.isna().to_numpy().mean() * 100)  # sur les colonnes d'origine
    if "price" in df:
        stats.currency = currency_from_df(df)
        df["currency"] = stats.currency
        df["price"] = clean_price(df["price"])
    df.insert(0, "city", slug)
    df.insert(1, "country", stats.country)
    df.insert(2, "scrape_date", date)
    return df, stats


def merge_listings(runs: list[tuple[str, str, Path]]) -> tuple[pd.DataFrame, list[CityStats]]:
    frames, stats = [], []
    for slug, date, folder in runs:
        log.info("listings : %s %s", slug, date)
        df, st = load_listings(slug, date, folder)
        frames.append(df)
        stats.append(st)
    all_columns = list(dict.fromkeys(c for f in frames for c in f.columns))
    for f, st in zip(frames, stats):
        st.absent_columns = [c for c in all_columns if c not in f.columns]
        if st.absent_columns:
            log.warning("%s : %d colonnes absentes : %s", st.city, len(st.absent_columns), st.absent_columns)
    merged = pd.concat(frames, ignore_index=True)[all_columns]
    return stringify_mixed_objects(merged), stats


# --------------------------------------------------------------------------- reviews

def reviews_chunks(slug: str, date: str, folder: Path):
    for chunk in pd.read_csv(folder / "reviews.csv.gz", chunksize=REVIEW_CHUNK, low_memory=False):
        if "listing_id" not in chunk:
            raise ValueError(f"{slug}: colonne listing_id absente de reviews.csv.gz")
        chunk.insert(0, "city", slug)
        chunk.insert(1, "scrape_date", date)
        yield chunk


def write_reviews(runs: list[tuple[str, str, Path]], out: Path, stats: dict[tuple[str, str], CityStats]) -> None:
    """Ecrit les avis par blocs (fichiers de plusieurs millions de lignes) dans un seul Parquet."""
    writer, schema = None, None
    try:
        for slug, date, folder in runs:
            if not (folder / "reviews.csv.gz").exists():
                log.warning("reviews manquant pour %s %s", slug, date)
                continue
            log.info("reviews : %s %s", slug, date)
            for chunk in reviews_chunks(slug, date, folder):
                table = pa.Table.from_pandas(chunk, preserve_index=False)
                if writer is None:
                    schema = table.schema
                    writer = pq.ParquetWriter(out, schema, compression="zstd")
                writer.write_table(table.cast(schema))
                stats[(slug, date)].n_reviews += len(chunk)
    finally:
        if writer is not None:
            writer.close()


# --------------------------------------------------------------------------- rapport

def write_report(path: Path, stats: list[CityStats], n_listings: int, n_reviews: int) -> None:
    lines = [
        "# Rapport de données Inside Airbnb",
        "",
        f"Total : {n_listings:,} annonces, {n_reviews:,} avis, {len(stats)} collectes.",
        "",
        "| Ville | Pays | Annonces | Avis | % valeurs manquantes | Date de collecte | Devise |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for s in sorted(stats, key=lambda s: (s.city, s.scrape_date)):
        lines.append(f"| {s.city} | {s.country} | {s.n_listings:,} | {s.n_reviews:,} | "
                     f"{s.pct_missing:.1f} % | {s.scrape_date} | {s.currency or 'inconnue'} |")
    lines += [
        "",
        "## Notes",
        "",
        "- % manquant : cellules vides sur l'ensemble des colonnes d'origine du fichier de la ville "
        "(les colonnes absentes de la ville ne sont pas comptées).",
        "- Devise : code ISO lu dans `price_quote_raw` ; à défaut, déduite du symbole du prix brut "
        "(le `$` est alors ambigu : USD, CAD, AUD, MXN...). Les prix ne sont pas convertis : "
        "ne pas comparer des villes de devises différentes sans conversion.",
        "",
        "## Colonnes absentes de certaines villes",
        "",
    ]
    absent = [s for s in stats if s.absent_columns]
    if absent:
        lines += [f"- **{s.city}** ({s.scrape_date}) : {', '.join(s.absent_columns)}" for s in absent]
    else:
        lines.append("Aucune : toutes les villes ont les mêmes colonnes.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- CLI

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--cities", nargs="+", metavar="SLUG", help="Slugs de villes (ex : paris new-york-city)")
    p.add_argument("--skip-reviews", action="store_true")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    runs = find_runs(args.raw_dir, set(args.cities) if args.cities else None)
    if not runs:
        raise SystemExit(f"Aucun listings.csv.gz trouvé sous {args.raw_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    listings, stats = merge_listings(runs)
    listings.to_parquet(args.output_dir / "listings_all.parquet", index=False)
    log.info("listings_all.parquet : %d lignes, %d colonnes", *listings.shape)
    stats_by_key = {(s.city, s.scrape_date): s for s in stats}
    del listings

    if not args.skip_reviews:
        write_reviews(runs, args.output_dir / "reviews_all.parquet", stats_by_key)
    write_report(args.output_dir / "data_report.md", stats,
                 sum(s.n_listings for s in stats), sum(s.n_reviews for s in stats))
    log.info("Rapport : %s", args.output_dir / "data_report.md")


if __name__ == "__main__":
    main()
