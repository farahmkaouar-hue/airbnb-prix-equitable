#!/usr/bin/env python3
"""Nettoyage de la base globale (sortie de merge_airbnb.py).

Produit dans data/processed/ :
    listings_clean.parquet  toutes les annonces, types corrigés, colonnes inutiles retirées, drapeaux d'anomalies
    listings_model.parquet  sous-ensemble prêt pour la modélisation du prix (prix valide, pas d'anomalie)
    reviews_clean.parquet   avis nettoyés (HTML retiré, dates typées, drapeaux), traités par blocs
    cleaning_log.md         ce qui a été fait, avec les effectifs

Principe : les valeurs douteuses sont signalées par des drapeaux (is_*) plutôt que supprimées en silence ;
seuls les doublons exacts (city, id) des annonces seraient retirés (aucun ici).
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger("clean")

URL_COLS = ["listing_url", "picture_url", "host_url", "host_profile_url", "host_thumbnail_url", "host_picture_url"]
TECHNICAL_COLS = ["scrape_id", "last_scraped", "calendar_last_scraped", "source", "price_quote_checkin_date",
                  "price_quote_checkout_date", "price_quote_total_price", "price_quote_price_per_night",
                  "price_quote_raw"]  # price_quote_* duplique le prix (fuite potentielle de la cible)
BOOL_COLS = ["host_is_superhost", "host_has_profile_pic", "host_identity_verified", "has_availability",
             "instant_bookable"]
PCT_COLS = ["host_response_rate", "host_acceptance_rate"]
DATE_COLS = ["first_review", "last_review", "host_since"]
AMENITY_FLAGS = {"has_wifi": "wifi", "has_kitchen": "kitchen", "has_ac": "air conditioning",
                 "has_washer": "washer", "has_elevator": "elevator", "has_parking": "parking",
                 "has_pool": "pool", "has_workspace": "workspace"}
AUTOMATED_RE = re.compile(r"^(?:the host canceled|the reservation was canceled|this is an automated posting|"
                          r"l'hôte a annulé|la réservation a été annulée|il s'agit d'un message automatique)",
                          re.IGNORECASE)
HTML_RE = re.compile(r"<[^>]+>")
MAX_MIN_NIGHTS, MAX_MAX_NIGHTS = 365, 1125

steps: list[str] = []


def note(msg: str) -> None:
    log.info(msg)
    steps.append(msg)


# --------------------------------------------------------------------------- listings

def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    df["is_stale"] = df["source"].eq("previous scrape") if "source" in df else False
    empty = [c for c in df.columns if df[c].isna().all()]
    constant = [c for c in df.columns if c not in empty and df[c].nunique(dropna=False) == 1
                and c not in ("city", "country", "scrape_date", "currency")]
    to_drop = sorted(set(empty + constant + URL_COLS + TECHNICAL_COLS) & set(df.columns))
    note(f"Colonnes retirées ({len(to_drop)}) : {len(empty)} vides, {len(constant)} constantes, "
         f"URL et colonnes techniques. Liste : {', '.join(to_drop)}")
    return df.drop(columns=to_drop)


def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    for c in BOOL_COLS:
        if c in df:
            df[c] = df[c].map({"t": True, "f": False}).astype("boolean")
    for c in PCT_COLS:
        if c in df:
            df[c] = pd.to_numeric(df[c].astype("string").str.rstrip("%"), errors="coerce") / 100
    for c in DATE_COLS:
        if c in df:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    df["scrape_date"] = pd.to_datetime(df["scrape_date"])
    note("Types : t/f -> booléens, pourcentages -> fractions 0-1, dates -> datetime.")
    return df


def parse_bathrooms(df: pd.DataFrame) -> pd.DataFrame:
    text = df["bathrooms_text"].astype("string").str.lower()
    number = pd.to_numeric(text.str.extract(r"(\d+(?:\.\d+)?)")[0], errors="coerce")
    number = number.where(~(text.str.contains("half", na=False) & number.isna()), 0.5)
    df["bathrooms"] = number.astype("float64")
    df["bath_shared"] = text.str.contains("shared", na=False)
    note("bathrooms (vide dans la source) reconstruit depuis bathrooms_text ; ajout de bath_shared.")
    return df.drop(columns=["bathrooms_text"])


def parse_amenities(df: pd.DataFrame) -> pd.DataFrame:
    def parse(s):
        try:
            return [a.lower() for a in json.loads(s)]
        except (TypeError, ValueError):
            return []

    lists = df["amenities"].map(parse)
    df["n_amenities"] = lists.map(len).astype("int32")
    joined = lists.map(" | ".join)
    for col, kw in AMENITY_FLAGS.items():
        df[col] = joined.str.contains(kw, regex=False)
    note(f"amenities : n_amenities + {len(AMENITY_FLAGS)} indicateurs ({', '.join(AMENITY_FLAGS)}).")
    return df.drop(columns=["amenities"])


def parse_license(df: pd.DataFrame) -> pd.DataFrame:
    lic = df["license"].astype("string").str.lower()
    kind = pd.Series("other", index=df.index)
    kind[lic.isna()] = "missing"
    kind[lic.str.contains("mobilit", na=False)] = "mobility_lease"
    kind[lic.str.startswith("exempt", na=False)] = "exempt"
    df["license_type"] = kind.astype("category")
    df["has_license"] = kind.isin(["other", "exempt", "mobility_lease"])
    note("license -> license_type (missing / exempt / mobility_lease / other) et has_license ; texte brut retiré.")
    return df.drop(columns=["license"])


def clip_and_nullify(df: pd.DataFrame) -> pd.DataFrame:
    n_min, n_max = int((df["minimum_nights"] > MAX_MIN_NIGHTS).sum()), int((df["maximum_nights"] > MAX_MAX_NIGHTS).sum())
    df["minimum_nights"] = df["minimum_nights"].clip(upper=MAX_MIN_NIGHTS)
    df["maximum_nights"] = df["maximum_nights"].clip(upper=MAX_MAX_NIGHTS)
    note(f"minimum_nights plafonné à {MAX_MIN_NIGHTS} ({n_min} valeurs), maximum_nights à {MAX_MAX_NIGHTS} ({n_max}).")
    bad_bed = (df["bedrooms"] > 20) | (df["beds"] > 50)
    df.loc[df["bedrooms"] > 20, "bedrooms"] = np.nan
    df.loc[df["beds"] > 50, "beds"] = np.nan
    note(f"bedrooms > 20 ou beds > 50 mis à NaN ({int(bad_bed.sum())} annonces, saisies aberrantes).")
    score_cols = [c for c in df.columns if c.startswith("review_scores_")]
    zeros = int((df[score_cols] == 0).any(axis=1).sum())
    df[score_cols] = df[score_cols].mask(df[score_cols] == 0)
    note(f"Notes égales à 0 mises à NaN ({zeros} annonces) : l'échelle Airbnb commence à 1.")
    df["has_reviews"] = df["number_of_reviews"] > 0
    return df


def flag_price_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Bornes par ville sur log10(prix) : Q1 - 3*IQR / Q3 + 3*IQR, plus prix <= 0."""
    df["has_price"] = df["price"].gt(0)
    df.loc[~df["has_price"], "price"] = np.nan
    logp = np.log10(df["price"])
    grp = logp.groupby(df["city"])
    q1, q3 = grp.transform(lambda s: s.quantile(0.25)), grp.transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1
    df["is_price_outlier"] = (logp < q1 - 3 * iqr) | (logp > q3 + 3 * iqr)
    median = df.groupby("city")["price"].transform("median")
    df["price_rel_city_median"] = df["price"] / median  # comparable entre villes sans conversion de devise
    per_city = df.groupby("city")["is_price_outlier"].sum().to_dict()
    note(f"Prix : {int((~df['has_price']).sum())} manquants (conservés, has_price=False) ; "
         f"{int(df['is_price_outlier'].sum())} valeurs extrêmes signalées (is_price_outlier), par ville : {per_city}.")
    return df


def flag_geo_outliers(df: pd.DataFrame, max_deg: float = 1.0) -> pd.DataFrame:
    lat_dev = (df["latitude"] - df.groupby("city")["latitude"].transform("median")).abs()
    lon_dev = (df["longitude"] - df.groupby("city")["longitude"].transform("median")).abs()
    df["is_geo_outlier"] = (lat_dev > max_deg) | (lon_dev > max_deg) | df[["latitude", "longitude"]].isna().any(axis=1)
    note(f"Coordonnées : {int(df['is_geo_outlier'].sum())} annonces à plus de {max_deg}° de la médiane de leur ville (is_geo_outlier).")
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.sort_values("scrape_date").drop_duplicates(["city", "id"], keep="last")
    note(f"Doublons (city, id) retirés : {before - len(df)}.")
    return df.reset_index(drop=True)


def clean_listings(df: pd.DataFrame) -> pd.DataFrame:
    note(f"Entrée : {len(df):,} annonces, {df.shape[1]} colonnes, {df['city'].nunique()} villes.")
    df = drop_duplicates(df)
    for step in (drop_columns, convert_types, parse_bathrooms, parse_amenities, parse_license,
                 clip_and_nullify, flag_price_outliers, flag_geo_outliers):
        df = step(df)
    return df


def model_subset(df: pd.DataFrame) -> pd.DataFrame:
    keep = df["has_price"] & ~df["is_price_outlier"] & ~df["is_geo_outlier"]
    note(f"Sous-ensemble modèle : {int(keep.sum()):,} annonces sur {len(df):,} "
         f"(prix valide, hors extrêmes de prix et de géolocalisation).")
    return df[keep].reset_index(drop=True)


# --------------------------------------------------------------------------- reviews

def clean_review_batch(df: pd.DataFrame, listing_ids: set[int]) -> pd.DataFrame:
    comments = df["comments"].astype("string").fillna("")
    comments = comments.str.replace(HTML_RE, " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    out = df.drop(columns=["reviewer_name", "comments"], errors="ignore").copy()
    out["date"] = pd.to_datetime(df["date"], errors="coerce")
    out["scrape_date"] = pd.to_datetime(df["scrape_date"])
    out["comments"] = comments
    out["comment_len"] = comments.str.len().astype("int32")
    out["is_automated"] = comments.str.match(AUTOMATED_RE)
    out["is_orphan"] = ~df["listing_id"].isin(listing_ids)
    return out


def clean_reviews(src: Path, dst: Path, listing_ids: set[int], batch_size: int = 500_000) -> dict[str, int]:
    """Nettoie par blocs. Aucun doublon exact n'existe ; `id` n'est PAS unique (il est partagé entre
    annonces/auteurs différents dans cette version des données), donc on ne dédoublonne pas dessus."""
    stats = {"rows": 0, "empty": 0, "automated": 0, "orphan": 0}
    writer = None
    try:
        for batch in pq.ParquetFile(src).iter_batches(batch_size=batch_size):
            clean = clean_review_batch(batch.to_pandas(), listing_ids)
            stats["rows"] += len(clean)
            stats["empty"] += int((clean["comment_len"] == 0).sum())
            stats["automated"] += int(clean["is_automated"].sum())
            stats["orphan"] += int(clean["is_orphan"].sum())
            table = pa.Table.from_pandas(clean, preserve_index=False)
            writer = writer or pq.ParquetWriter(dst, table.schema, compression="zstd")
            writer.write_table(table.cast(writer.schema))
    finally:
        if writer is not None:
            writer.close()
    note(f"Avis : {stats['rows']:,} lignes conservées ; {stats['empty']:,} commentaires vides, "
         f"{stats['automated']:,} messages automatiques, {stats['orphan']:,} avis dont l'annonce est absente "
         "(drapeaux is_*). Aucune ligne supprimée : pas de doublon exact, et `id` n'est pas unique. "
         "HTML retiré, reviewer_name supprimé (donnée personnelle inutile).")
    return stats


# --------------------------------------------------------------------------- CLI

def write_log(path: Path) -> None:
    body = "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))
    path.write_text(f"# Journal de nettoyage\n\n{body}\n", encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--skip-reviews", action="store_true")
    args = p.parse_args()
    out = args.processed_dir

    listings = clean_listings(pd.read_parquet(out / "listings_all.parquet"))
    listings.to_parquet(out / "listings_clean.parquet", index=False)
    model_subset(listings).to_parquet(out / "listings_model.parquet", index=False)
    if not args.skip_reviews and (out / "reviews_all.parquet").exists():
        clean_reviews(out / "reviews_all.parquet", out / "reviews_clean.parquet", set(listings["id"].tolist()))
    write_log(out / "cleaning_log.md")


if __name__ == "__main__":
    main()
