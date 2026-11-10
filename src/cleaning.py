"""Phase 2 : nettoyage. Produit data/processed/listings_clean.parquet, texts.parquet, split.parquet,
models/amenity_vocab.json, models/price_bounds.csv et reports/cleaning_log.md.

Principe anti-fuite : tout seuil appris sur les données (vocabulaire d'équipements, clôtures de prix,
centre-ville, quantiles de winsorisation) est calculé sur le TRAIN uniquement (split.parquet).
Rien n'est imputé ici (l'imputation se fait dans les Pipelines, ajustée sur le train de chaque pli).
Usage : python -m src.cleaning
"""
import json
import re

import numpy as np
import pandas as pd

from src.data import ROOT, load_config, load_listings, make_split
from src.features import haversine

LOG: list[tuple[str, str, str]] = []  # (étape, lignes/valeurs affectées, pourquoi)


def log(step: str, n, why: str):
    LOG.append((step, n if isinstance(n, str) else f"{int(n):,}", why))
    print(f"[{step}] {LOG[-1][1]} : {why}")


# ---------------------------------------------------------------- utilitaires
def load_fx(cfg: dict) -> pd.DataFrame:
    fx = pd.read_csv(ROOT / cfg["fx"]["file"], parse_dates=["date"])
    return fx.sort_values("date")


def fx_rate_per_city(df: pd.DataFrame, fx: pd.DataFrame) -> pd.Series:
    """Taux (devise pour 1 EUR) par annonce : dernier fixing BCE <= date de collecte de la ville."""
    rates = {}
    for (city, cur), d in df.groupby(["city", "currency"])["scrape_date"].first().items():
        if cur == "EUR":
            rates[city] = 1.0
            continue
        f = fx[(fx.currency == cur) & (fx.date <= pd.Timestamp(d))]
        if f.empty:
            raise ValueError(f"Aucun taux BCE pour {cur} avant {d} ({city}) : à compléter dans data/external/")
        rates[city] = float(f.iloc[-1].rate_per_eur)
    return df["city"].map(rates)


def norm_amenity(a: str) -> str:
    """Normalise un libellé : minuscule, retire les détails après ':' ou ' – ' (marques, débits...)."""
    a = re.split(r"\s[–-]\s|:", str(a).lower().strip())[0].strip()
    return re.sub(r"[^a-z0-9]+", "_", a).strip("_")


def parse_bathrooms(text: pd.Series) -> pd.DataFrame:
    t = text.str.lower()
    num = t.str.extract(r"(\d+(?:\.\d+)?)")[0].astype(float)
    num = num.where(~(t.str.contains("half", na=False) & num.isna()), 0.5)
    num = num.where(~t.str.contains("no bathroom", na=False), 0.0)
    shared = t.str.contains("shared", na=False).astype("float").where(t.notna())
    return pd.DataFrame({"bath_count": num, "bath_shared": shared})


# ---------------------------------------------------------------- nettoyage
def clean_listings(df: pd.DataFrame, split: pd.DataFrame, cfg: dict):
    cc = cfg["cleaning"]
    n0 = len(df)
    df = df.merge(split[["city", "id", "is_test"]], on=["city", "id"], how="left")
    assert df["is_test"].notna().all()
    train = ~df["is_test"].astype(bool)
    log("split", int(df.is_test.sum()), f"annonces du test final ({df.is_test.mean() * 100:.1f} %), groupées par hôte ; tous les seuils ci-dessous sont appris sur les {int(train.sum()):,} annonces train")

    # 1. colonnes entièrement vides
    empty = [c for c in df.columns if df[c].isna().all()]
    df = df.drop(columns=empty)
    log("colonnes 100 % vides", len(empty), "supprimées : " + ", ".join(empty))

    # 1b. contexte du devis de prix : le prix est un devis pour un séjour précis (dates, durée) -> variables saisonnières
    if {"price_quote_checkin_date", "price_quote_checkout_date"} <= set(df.columns):
        ci = pd.to_datetime(df["price_quote_checkin_date"], errors="coerce")
        co = pd.to_datetime(df["price_quote_checkout_date"], errors="coerce")
        df["quote_lead_days"] = (ci - pd.to_datetime(df["scrape_date"])).dt.days
        df["quote_nights"] = (co - ci).dt.days
        df["quote_checkin_month"] = ci.dt.month
        df["quote_checkin_dow"] = ci.dt.dayofweek
        log("contexte du devis", int(ci.notna().sum()), "price_quote_checkin/checkout_date -> quote_lead_days, quote_nights, quote_checkin_month, quote_checkin_dow "
            "(le prix est un devis de 1 à 31 nuits, check-in de 0 à 369 jours après la collecte : saisonnalité et remises mensuelles)")

    # 2. textes conservés à part pour le NLP, puis suppression des textes bruts / données personnelles
    texts = df[["city", "id", "name", "description", "host_about"]].copy()
    drop_cols = ["listing_url", "picture_url", "host_url", "host_profile_url", "host_picture_url", "scrape_id", "last_scraped", "host_profile_id",
                 "calendar_last_scraped", "host_name", "name", "description", "host_about"]
    drop_quote = [c for c in df.columns if c.startswith("price_quote_")]
    absent = [c for c in drop_cols if c not in df.columns]
    df = df.drop(columns=[c for c in drop_cols + drop_quote if c in df.columns])
    log("colonnes inutiles / sensibles", len(drop_cols) - len(absent) + len(drop_quote),
        "URLs, identifiants techniques, price_quote_* (doublon exact du prix, fuite), host_name (RGPD) supprimés ; name/description/host_about copiés dans texts.parquet"
        + (f" (absentes, ignorées : {absent})" if absent else ""))

    # 3. booléens et dates
    for c in ["host_is_superhost", "host_has_profile_pic", "host_identity_verified", "has_availability"]:
        if c in df:
            n_nan = int(df[c].isna().sum())
            df[c] = df[c].map({"t": True, "f": False}).astype("boolean")
            log(f"booléen {c}", len(df) - n_nan, f"'t'/'f' -> booléen ({n_nan} manquants conservés)")
    for c in ["first_review", "last_review", "scrape_date"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    log("dates", len(df), "first_review, last_review, scrape_date -> datetime")
    df["is_stale"] = df["source"].notna() if "source" in df else False
    df = df.drop(columns=["source"], errors="ignore")
    log("source -> is_stale", int(df.is_stale.sum()), "'previous scrape' = annonce non rafraîchie à la dernière collecte (80 % sans prix)")
    df["has_license"] = df["license"].notna()
    df = df.drop(columns=["license"])
    log("license -> has_license", int(df.has_license.sum()), "texte libre (numéros d'enregistrement, donnée quasi personnelle) réduit à un indicateur")

    # 4. salles de bain
    b = parse_bathrooms(df["bathrooms_text"])
    df = pd.concat([df.drop(columns=["bathrooms_text"]), b], axis=1)
    log("bathrooms_text -> bath_count, bath_shared", int(b.bath_count.notna().sum()),
        f"parsé en nombre (demi-salle de bain = 0,5) et indicateur partagée ({int(b.bath_shared.sum()):,} partagées) ; {int(b.bath_count.isna().sum())} non parsables laissés NaN")

    # 5. équipements
    lists = [sorted({norm_amenity(x) for x in json.loads(s)} - {""}) if isinstance(s, str) else [] for s in df["amenities"]]
    lens = np.fromiter((len(x) for x in lists), int)
    flat = pd.Series([a for x in lists for a in x])
    row = np.repeat(np.arange(len(df)), lens)
    tr_rows = train.values[row]
    freq = flat[tr_rows].value_counts() / train.sum()
    vocab = freq[freq >= cc["amenity_min_freq"]].index.tolist()
    with open(ROOT / "models" / "amenity_vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=1)
    idx = {a: i for i, a in enumerate(vocab)}
    mat = np.zeros((len(df), len(vocab)), dtype="int8")
    col = flat.map(idx)
    ok = col.notna().values
    mat[row[ok], col[ok].astype(int).values] = 1
    am = pd.DataFrame(mat, columns=["am_" + a for a in vocab], index=df.index)
    df["n_amenities"] = lens
    df["n_amenities_other"] = lens - mat.sum(axis=1)
    df = pd.concat([df.drop(columns=["amenities"]), am], axis=1)
    log("équipements", len(vocab), f"libellés normalisés (minuscule, détails retirés) : {flat.nunique():,} libellés distincts -> {len(vocab)} conservés (>= {cc['amenity_min_freq']:.0%} des annonces train), "
        f"les autres regroupés dans n_amenities_other ; vocabulaire dans models/amenity_vocab.json")

    # 6. prix -> EUR
    fx = load_fx(cfg)
    rate = fx_rate_per_city(df, fx)
    df["fx_rate_per_eur"] = rate
    df["has_price"] = df["price"].notna() & (df["price"] > 0)
    df["price_eur"] = np.where(df.has_price, df["price"] / rate, np.nan)
    df["log_price_eur"] = np.log(df["price_eur"])
    df = df.rename(columns={"price": "price_local"})
    log("prix -> EUR", int(df.has_price.sum()), "price_eur = prix local / taux BCE du dernier fixing <= date de collecte de la ville ; devise d'origine conservée dans `currency` ; "
        f"{int((~df.has_price).sum()):,} annonces sans prix (aucun prix récupérable : le JSON price_quote_raw n'a pas de price_per_night pour elles) conservées avec price_eur = NaN")

    # 7. winsorisation de valeurs numériques extrêmes
    n_sent = int((df.maximum_nights > cc["max_nights_sentinel"]).sum())
    for c in ["maximum_nights", "maximum_maximum_nights", "minimum_maximum_nights", "maximum_nights_avg_ntm"]:
        if c in df:
            df[c] = df[c].clip(upper=cc["max_nights_sentinel"])
    log("maximum_nights", n_sent, f"valeurs sentinelles (max 2 147 483 647) plafonnées à {cc['max_nights_sentinel']}")
    for c in ["bedrooms", "beds"]:
        cap = df.loc[train, c].quantile(cc["clip_quantile"])
        n_cl = int((df[c] > cap).sum())
        df[c] = df[c].clip(upper=cap)
        log(f"winsorisation {c}", n_cl, f"plafonné au quantile {cc['clip_quantile']} du train = {cap:g}")

    # 8. annonces incohérentes (signalées, pas supprimées ici)
    beds0 = df.beds.fillna(0) == 0
    i1 = (df.accommodates >= 4) & (df.bedrooms.fillna(0) == 0) & beds0
    i2 = df.beds > 3 * df.accommodates
    i3 = df.bedrooms > 2 * df.accommodates
    i4 = df.minimum_nights > df.maximum_nights
    df["is_inconsistent"] = i1 | i2 | i3 | i4
    log("incohérence : capacité >= 4 sans chambre ni lit", int(i1.sum()), "capacité élevée sans couchage déclaré")
    log("incohérence : lits > 3 x capacité", int(i2.sum()), "nombre de lits incompatible avec la capacité")
    log("incohérence : chambres > 2 x capacité", int(i3.sum()), "nombre de chambres incompatible avec la capacité")
    log("incohérence : minimum_nights > maximum_nights", int(i4.sum()), "contradiction de règles de séjour")
    log("is_inconsistent (union)", int(df.is_inconsistent.sum()), "signalées ; exclues de l'entraînement de la régression, jamais supprimées de listings_clean")

    # 9. géographie : coordonnées aberrantes
    ctr = df[train].groupby("city")[["latitude", "longitude"]].median()
    c_ = df[["city"]].join(ctr, on="city")
    df["dist_center_km"] = haversine(df.latitude, df.longitude, c_.latitude, c_.longitude)
    df["is_geo_outlier"] = df.dist_center_km > cc["geo_max_km_from_center"]
    log("coordonnées aberrantes", int(df.is_geo_outlier.sum()), f"> {cc['geo_max_km_from_center']} km de la médiane des coordonnées de la ville (centre = médiane train)")

    # 10. doublons
    log("doublons exacts (city, id)", int(df.duplicated(["city", "id"]).sum()), "aucun")
    key = ["city", "host_id", "latitude", "longitude", "room_type", "accommodates", "bedrooms"]
    df["is_near_duplicate"] = df.duplicated(key, keep="first")
    log("quasi-doublons (même hôte, mêmes coordonnées, type, capacité, chambres)", int(df.is_near_duplicate.sum()),
        "signalés, non supprimés : souvent des unités distinctes d'un même immeuble (hôtels, résidences) ; le split groupé par hôte neutralise la fuite")

    # 11. outliers de prix : clôtures par ville sur log(price_eur), apprises sur le train
    k = cc["price_iqr_k"]
    tp = df[train & df.has_price]
    q = tp.groupby("city")["log_price_eur"].quantile([0.25, 0.75]).unstack()
    iqr = q[0.75] - q[0.25]
    bounds = pd.DataFrame({"low_log": q[0.25] - k * iqr, "high_log": q[0.75] + k * iqr})
    bounds["low_eur"] = np.exp(bounds.low_log)
    bounds["high_eur"] = np.exp(bounds.high_log)
    bounds.round(3).to_csv(ROOT / "models" / "price_bounds.csv")
    lo, hi = df.city.map(bounds.low_log), df.city.map(bounds.high_log)
    below = df.has_price & (df.log_price_eur < lo)
    above = df.has_price & (df.log_price_eur > hi)
    df["is_price_outlier"] = below | above
    log("outliers de prix", int(df.is_price_outlier.sum()),
        f"hors [Q1 - {k} x IQR ; Q3 + {k} x IQR] de log(price_eur) par ville (train) : {int(below.sum())} trop bas, {int(above.sum())} trop hauts "
        f"({df.is_price_outlier.sum() / df.has_price.sum() * 100:.2f} % des prix) ; signalés (is_price_outlier), exclus de l'entraînement seulement ; impact mesuré par outlier_impact")

    # 12. jeu de régression
    df["in_reg_set"] = df.has_price & ~df.is_price_outlier & ~df.is_geo_outlier & ~df.is_inconsistent
    log("jeu de régression (in_reg_set)", int(df.in_reg_set.sum()), f"= avec prix, sans outlier de prix, sans coordonnées aberrantes, sans incohérence ({df.in_reg_set.mean() * 100:.1f} % des annonces)")
    assert len(df) == n0
    return df, texts


def missing_strategy(df: pd.DataFrame) -> str:
    m = df.isna().mean() * 100
    m = m[m > 0].sort_values(ascending=False)
    rows = ["| Colonne | % manquant | Stratégie (appliquée dans le Pipeline, ajustée sur le train) |", "|---|---:|---|"]
    for c, v in m.items():
        if c in ("price_local", "price_eur", "log_price_eur"):
            s = "cible : jamais imputée ; annonces sans prix exclues de la régression"
        elif c.startswith("review_scores") or c in ("first_review", "last_review", "reviews_per_month"):
            s = "absence d'avis : indicateur `has_reviews` + imputation médiane par ville (linéaires) ou NaN natif (boosting)"
        elif str(df[c].dtype) in ("str", "string", "object", "category"):
            s = "catégorie « manquant »"
        elif str(df[c].dtype) == "boolean":
            s = "mode + indicateur de manquant"
        else:
            s = "médiane par ville (train) + indicateur de manquant"
        rows.append(f"| {c} | {v:.1f} | {s} |")
    return "\n".join(rows)


def write_log(df: pd.DataFrame, extra: str = ""):
    out = ["# Journal de nettoyage (Phase 2)", "",
           f"Base d'entrée : listings_all.parquet. Sortie : {len(df):,} lignes x {df.shape[1]} colonnes (listings_clean.parquet). Aucune ligne n'est supprimée : les annonces douteuses sont signalées par des indicateurs.", "",
           "| Étape | Lignes / valeurs affectées | Pourquoi |", "|---|---:|---|"]
    out += [f"| {a} | {b} | {c} |" for a, b, c in LOG]
    out += ["", "## Stratégie d'imputation par colonne (appliquée en phase 3/6 dans les Pipelines)", "", missing_strategy(df), "", extra]
    (ROOT / "reports" / "cleaning_log.md").write_text("\n".join(out) + "\n", encoding="utf-8")


def run():
    cfg = load_config()
    (ROOT / "models").mkdir(exist_ok=True)
    raw = load_listings(cfg)
    raw["scrape_date"] = pd.to_datetime(raw["scrape_date"])
    split = make_split(raw, cfg)
    split.to_parquet(ROOT / "data/processed/split.parquet", index=False)
    df, texts = clean_listings(raw, split, cfg)
    df.to_parquet(ROOT / cfg["paths"]["listings_clean"], index=False)
    texts.to_parquet(ROOT / "data/processed/texts.parquet", index=False)
    write_log(df)
    print(df.shape)


if __name__ == "__main__":
    run()
