"""Phase 3 : feature engineering réutilisable.

Deux familles de transformations :
  1. SANS apprentissage (formules déterministes par ligne) : `build_feature_table` -> data/processed/listings_features.parquet
  2. AVEC apprentissage (imputation par ville, encodage, scaling) : dans des Pipelines sklearn (`build_preprocessor`),
     à ajuster sur le train uniquement (y compris dans chaque pli de CV).
Usage : python -m src.features
"""
import json

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from src.data import ROOT, load_config

SEED = 42


def haversine(lat1, lon1, lat2, lon2):
    """Distance en km entre deux points (degrés)."""
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 12742 * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------------------------------------------
# 1. Variables dérivées (sans apprentissage)
# ---------------------------------------------------------------------------------------------------------------
def add_text_features(df: pd.DataFrame, texts: pd.DataFrame) -> pd.DataFrame:
    """Longueurs de textes (le texte brut reste dans texts.parquet pour le NLP)."""
    t = texts.set_index(["city", "id"])
    desc = t["description"].fillna("")
    out = pd.DataFrame({
        "desc_len": desc.str.len(),
        "desc_words": desc.str.split().str.len(),
        "name_len": t["name"].fillna("").str.len(),
        "host_about_len": t["host_about"].fillna("").str.len(),
    }, index=t.index)
    out["has_host_about"] = (out["host_about_len"] > 0).astype(float)
    return df.join(out, on=["city", "id"])


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Variables dérivées déterministes (aucune statistique apprise)."""
    d = df.copy()
    # booléens nullables -> float (NaN conservés) pour tous les estimateurs
    for c in d.columns:
        if str(d[c].dtype) == "boolean":  # booléens à NaN possibles ; les drapeaux bool (is_test...) restent bool
            d[c] = d[c].astype("float")
    sd = pd.to_datetime(d["scrape_date"])
    # hôte
    d["host_tenure_years"] = d["hosts_time_as_host_years"] + d["hosts_time_as_host_months"] / 12
    d["user_tenure_years"] = d["hosts_time_as_user_years"] + d["hosts_time_as_user_months"] / 12
    d["host_is_pro"] = (d["calculated_host_listings_count"] >= 5).astype(float)
    tot = d["calculated_host_listings_count"].replace(0, np.nan)
    d["host_entire_share"] = d["calculated_host_listings_count_entire_homes"] / tot
    loc = d["host_location"].str.lower()
    cityname = d["city"].str.replace("-", " ", regex=False)
    d["host_is_local"] = [np.nan if pd.isna(l) else float(c.split()[0] in l) for l, c in zip(loc, cityname)]
    # avis
    d["has_reviews"] = (d["number_of_reviews"] > 0).astype(float)
    d["days_since_last_review"] = (sd - pd.to_datetime(d["last_review"])).dt.days
    d["days_since_first_review"] = (sd - pd.to_datetime(d["first_review"])).dt.days
    # logement
    d["is_studio"] = (d["bedrooms"] == 0).astype(float).where(d["bedrooms"].notna())
    acc = d["accommodates"].replace(0, np.nan)
    d["beds_per_person"] = d["beds"] / acc
    d["persons_per_bedroom"] = d["accommodates"] / d["bedrooms"].clip(lower=1)
    d["bath_per_person"] = d["bath_count"] / acc
    # transformations log des variables très asymétriques (utiles aux modèles linéaires, neutres pour les arbres)
    for src, dst in [("minimum_nights", "log_min_nights"), ("maximum_nights", "log_max_nights"), ("calculated_host_listings_count", "log_host_listings"),
                     ("number_of_reviews", "log_n_reviews"), ("number_of_reviews_ltm", "log_n_reviews_ltm"), ("reviews_per_month", "log_reviews_per_month"),
                     ("quote_lead_days", "log_quote_lead_days")]:
        d[dst] = np.log1p(d[src].clip(lower=0))
    # saisonnalité du devis
    m = d["quote_checkin_month"]
    d["quote_month_sin"] = np.sin(2 * np.pi * m / 12)
    d["quote_month_cos"] = np.cos(2 * np.pi * m / 12)
    d["quote_is_weekend"] = d["quote_checkin_dow"].isin([4, 5]).astype(float).where(d["quote_checkin_dow"].notna())
    d["quote_is_monthly"] = (d["quote_nights"] >= 28).astype(float).where(d["quote_nights"].notna())
    # clé quartier (ville + quartier) pour le target encoding
    d["nbhd_key"] = d["city"] + "|" + d["neighbourhood_cleansed"].fillna("missing")
    return d


def make_targets(d: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    """Cibles : log_price_eur (déjà là), is_superhost, price_tier (terciles de prix PAR VILLE, bornes apprises sur le train).
    Limite : les bornes sont calculées sur tout le train, donc légèrement vues par les plis de validation interne."""
    d = d.copy()
    d["is_superhost"] = d["host_is_superhost"]
    tr = d[train_mask & d["in_reg_set"]]
    edges = tr.groupby("city")["price_eur"].quantile([1 / 3, 2 / 3]).unstack()
    edges.columns = ["q33", "q66"]
    edges.to_csv(ROOT / "models" / "price_tier_edges.csv")
    q33, q66 = d["city"].map(edges.q33), d["city"].map(edges.q66)
    tier = np.where(d["price_eur"] <= q33, 0, np.where(d["price_eur"] <= q66, 1, 2)).astype(float)
    d["price_tier"] = np.where(d["price_eur"].notna(), tier, np.nan)  # 0 = bas, 1 = moyen, 2 = haut
    return d


# ---------------------------------------------------------------------------------------------------------------
# Groupes de variables et jeux de variables (pour la sélection de la phase 4 et le test avec/sans de la phase 7)
# ---------------------------------------------------------------------------------------------------------------
def amenity_cols(columns) -> list[str]:
    return [c for c in columns if c.startswith("am_")]


def feature_groups(columns) -> dict[str, list[str]]:
    cols = set(columns)
    g = {
        "property": ["accommodates", "bedrooms", "beds", "bath_count", "bath_shared", "room_type", "property_type", "is_studio", "beds_per_person",
                     "persons_per_bedroom", "bath_per_person", "log_min_nights", "log_max_nights"],
        "amenities": ["n_amenities", "n_amenities_other"] + amenity_cols(columns),
        "location": ["latitude", "longitude", "dist_center_km", "nbhd_key", "city"],
        "host": ["host_tenure_years", "user_tenure_years", "log_host_listings", "host_is_pro", "host_entire_share", "host_identity_verified",
                 "host_has_profile_pic", "host_is_local", "has_license", "has_host_about", "host_about_len"],
        "text": ["desc_len", "desc_words", "name_len"],
        "quality": ["review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness", "review_scores_checkin",
                    "review_scores_communication", "review_scores_location", "review_scores_value"],
        "host_status": ["host_is_superhost"],
        "activity": ["log_n_reviews", "log_n_reviews_ltm", "log_reviews_per_month", "days_since_last_review", "days_since_first_review", "has_reviews",
                     "number_of_reviews_l30d", "availability_30", "availability_60", "availability_90", "availability_365", "has_availability", "is_stale"],
        "quote": ["log_quote_lead_days", "quote_nights", "quote_month_sin", "quote_month_cos", "quote_is_weekend", "quote_is_monthly"],
    }
    return {k: [c for c in v if c in cols] for k, v in g.items()}


def feature_set(columns, name: str = "core") -> list[str]:
    """core = caractéristiques du logement (aucune conséquence du prix) ; +quote = contexte du devis ; +quality ; +activity ; full = tout sauf colonnes à fuite."""
    g = feature_groups(columns)
    core = g["property"] + g["amenities"] + g["location"] + g["host"] + g["text"]
    sets = {"core": core, "core+quote": core + g["quote"], "core+quote+quality": core + g["quote"] + g["quality"] + g["host_status"],
            "full": core + g["quote"] + g["quality"] + g["host_status"] + g["activity"]}
    return sets[name]


# ---------------------------------------------------------------------------------------------------------------
# 2. Transformations avec apprentissage (dans les Pipelines)
# ---------------------------------------------------------------------------------------------------------------
class CityMedianImputer(BaseEstimator, TransformerMixin):
    """Imputation par la médiane de la ville (repli : médiane globale) + indicateurs de manquant.
    Les médianes et la liste des colonnes à indicateur sont apprises sur le train seulement."""

    def __init__(self, num_cols, city_col="city"):
        self.num_cols = num_cols
        self.city_col = city_col

    def fit(self, X, y=None):
        self.global_med_ = X[self.num_cols].median()
        self.city_med_ = X.groupby(self.city_col)[self.num_cols].median()
        self.ind_cols_ = [c for c in self.num_cols if X[c].isna().any()]
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.ind_cols_:
            X[c + "_missing"] = X[c].isna().astype("float")
        med = X[[self.city_col]].join(self.city_med_, on=self.city_col)[self.num_cols]
        med = med.fillna(self.global_med_)
        X[self.num_cols] = X[self.num_cols].astype(float).fillna(med)
        return X


def split_columns(features: list[str], X: pd.DataFrame):
    cat_low = [c for c in ["room_type", "property_type", "city"] if c in features]
    te = [c for c in ["nbhd_key"] if c in features]
    num = [c for c in features if c not in cat_low + te]
    return num, cat_low, te


def build_preprocessor(features: list[str], X_example: pd.DataFrame, scale: bool = False, target_type: str = "continuous") -> Pipeline:
    """Pipeline de prétraitement complet.
    - numériques : médiane par ville + indicateurs de manquant, standardisation si scale=True (modèles linéaires, kNN, SVM, réseaux)
    - catégories faible cardinalité (room_type, property_type, city) : one-hot, modalités < 1 % regroupées
    - nbhd_key (ville|quartier, forte cardinalité) : TargetEncoder à validation croisée interne (cv=5), lissage automatique
    target_type : 'continuous' (régression), 'binary' ou 'multiclass' (classification).
    Si 'city' n'est pas dans les features (ex. leave-one-city-out sans identifiant de ville), la colonne reste nécessaire en entrée de l'imputeur."""
    num, cat_low, te = split_columns(features, X_example)
    imp_cols = num
    steps_num = [("scale", StandardScaler())] if scale else [("pass", "passthrough")]
    ct = ColumnTransformer(
        [("num", Pipeline(steps_num), make_num_selector(num)),
         ("ind", "passthrough", make_ind_selector(num)),
         ("cat", OneHotEncoder(min_frequency=0.01, handle_unknown="infrequent_if_exist", sparse_output=False), cat_low)]
        + ([("te", TargetEncoder(target_type=target_type, cv=5, shuffle=True, random_state=SEED), te)] if te else []),
        remainder="drop", verbose_feature_names_out=False)
    ct.set_output(transform="pandas")
    return Pipeline([("impute", CityMedianImputer(imp_cols)), ("ct", ct)])


def make_num_selector(num):
    return list(num)


def _ind_cols(X):
    # les colonnes *_missing n'existent qu'après fit de l'imputeur : sélection par motif
    return [c for c in X.columns if c.endswith("_missing")]


def make_ind_selector(num):
    return _ind_cols


# ---------------------------------------------------------------------------------------------------------------
# Tableau des variables
# ---------------------------------------------------------------------------------------------------------------
DEFS = {
    "city": ("id/groupe", "Ville (une des 15)"), "country": ("id/groupe", "Pays"), "scrape_date": ("id/groupe", "Date de collecte (une par ville)"),
    "id": ("id/groupe", "Identifiant de l'annonce"), "host_id": ("id/groupe", "Identifiant de l'hôte (groupe de validation)"),
    "price_local": ("cible/dérivée", "Prix du devis en devise locale"), "price_eur": ("cible/dérivée", "Prix par nuit converti en EUR (taux BCE à la date de collecte)"),
    "log_price_eur": ("cible", "CIBLE RÉGRESSION : log(price_eur)"), "is_superhost": ("cible", "CIBLE CLASSIF 1 : l'hôte est Superhost (0/1)"),
    "price_tier": ("cible", "CIBLE CLASSIF 2 : tranche de prix dans la ville (0 bas, 1 moyen, 2 haut ; terciles train)"),
    "currency": ("id/groupe", "Devise d'origine"), "fx_rate_per_eur": ("id/groupe", "Taux BCE utilisé (devise pour 1 EUR)"),
    "is_test": ("drapeau", "Annonce du test final (ne jamais utiliser avant la phase 14)"), "has_price": ("drapeau", "Prix connu"),
    "in_reg_set": ("drapeau", "Utilisable pour la régression (prix connu, ni outlier, ni incohérent, ni coordonnées aberrantes)"),
    "is_price_outlier": ("drapeau", "Prix hors clôtures 3 x IQR de la ville (train)"), "is_inconsistent": ("drapeau", "Capacité / lits / chambres / nuits incohérents"),
    "is_geo_outlier": ("drapeau", "À plus de 100 km du centre de la ville"), "is_near_duplicate": ("drapeau", "Même hôte, mêmes coordonnées, type, capacité, chambres"),
    "estimated_revenue_l365d": ("FUITE prix", "Revenu estimé sur 365 j = prix x occupation : dérivé du prix, exclu de la régression"),
    "estimated_occupancy_l365d": ("FUITE prix", "Occupation estimée sur 365 j (modèle Inside Airbnb) : conséquence, exclue"),
    "hosts_time_as_user_years": ("brute", "Ancienneté de l'hôte comme utilisateur (années entières)"), "hosts_time_as_user_months": ("brute", "idem, mois restants (0-11)"),
    "hosts_time_as_host_years": ("brute", "Ancienneté comme hôte (années entières)"), "hosts_time_as_host_months": ("brute", "idem, mois restants (0-11)"),
    "host_location": ("brute", "Lieu déclaré de l'hôte (texte)"), "host_is_superhost": ("brute", "Statut Superhost (cible de la classification 1)"),
    "host_listings_count": ("brute", "Nombre d'annonces de l'hôte (déclaré)"), "host_has_profile_pic": ("brute", "L'hôte a une photo de profil"),
    "host_identity_verified": ("brute", "Identité de l'hôte vérifiée"), "neighbourhood_cleansed": ("brute", "Quartier (nomenclature Inside Airbnb)"),
    "neighbourhood_group_cleansed": ("brute", "Groupe de quartiers (très incomplet)"), "latitude": ("brute", "Latitude"), "longitude": ("brute", "Longitude"),
    "property_type": ("brute", "Type de propriété (141 modalités ; <1 % regroupées en 'infrequent')"), "room_type": ("brute", "Type de logement (4 modalités)"),
    "accommodates": ("brute", "Capacité d'accueil"), "bedrooms": ("brute", "Nombre de chambres (winsorisé au q99,9 train)"), "beds": ("brute", "Nombre de lits (winsorisé)"),
    "minimum_nights": ("brute", "Nuits minimum"), "maximum_nights": ("brute", "Nuits maximum (plafonné à 1125)"),
    "minimum_minimum_nights": ("brute", "Nuits minimum, plus petite valeur sur 365 j"), "maximum_minimum_nights": ("brute", "Nuits minimum, plus grande valeur"),
    "minimum_maximum_nights": ("brute", "Nuits maximum, plus petite valeur"), "maximum_maximum_nights": ("brute", "Nuits maximum, plus grande valeur"),
    "minimum_nights_avg_ntm": ("brute", "Moyenne des nuits minimum, 12 prochains mois"), "maximum_nights_avg_ntm": ("brute", "Moyenne des nuits maximum, 12 prochains mois"),
    "has_availability": ("brute/activité", "Annonce marquée disponible"), "availability_30": ("activité (conséquence du prix)", "Jours disponibles sur 30 j"),
    "availability_60": ("activité", "Jours disponibles sur 60 j"), "availability_90": ("activité", "Jours disponibles sur 90 j"), "availability_365": ("activité", "Jours disponibles sur 365 j"),
    "availability_eoy": ("activité", "Jours disponibles jusqu'à fin d'année"), "number_of_reviews": ("activité", "Nombre total d'avis"),
    "number_of_reviews_ltm": ("activité", "Avis sur 12 mois"), "number_of_reviews_l30d": ("activité", "Avis sur 30 j"), "number_of_reviews_ly": ("activité", "Avis l'an dernier"),
    "first_review": ("brute", "Date du premier avis"), "last_review": ("brute", "Date du dernier avis"),
    "review_scores_rating": ("qualité", "Note globale (0-5)"), "review_scores_accuracy": ("qualité", "Note exactitude"), "review_scores_cleanliness": ("qualité", "Note propreté"),
    "review_scores_checkin": ("qualité", "Note arrivée"), "review_scores_communication": ("qualité", "Note communication"), "review_scores_location": ("qualité", "Note emplacement"),
    "review_scores_value": ("qualité", "Note rapport qualité/prix"), "calculated_host_listings_count": ("brute", "Annonces de l'hôte dans la ville (calculé Inside Airbnb)"),
    "calculated_host_listings_count_entire_homes": ("brute", "... dont logements entiers"), "calculated_host_listings_count_private_rooms": ("brute", "... dont chambres privées"),
    "calculated_host_listings_count_shared_rooms": ("brute", "... dont chambres partagées"), "reviews_per_month": ("activité", "Avis par mois"),
    "quote_lead_days": ("devis", "Jours entre la collecte et le check-in du devis"), "quote_nights": ("devis", "Durée du séjour du devis (1 à 31 nuits)"),
    "quote_checkin_month": ("devis", "Mois du check-in du devis"), "quote_checkin_dow": ("devis", "Jour de semaine du check-in (0 = lundi)"),
    "is_stale": ("activité", "Annonce non rafraîchie à la dernière collecte (« previous scrape »)"), "has_license": ("hôte", "Un numéro de licence est renseigné"),
    "bath_count": ("logement", "Nombre de salles de bain (parsé ; demi = 0,5)"), "bath_shared": ("logement", "Salle de bain partagée (0/1)"),
    "n_amenities": ("équipements", "Nombre total d'équipements normalisés"), "n_amenities_other": ("équipements", "Équipements hors vocabulaire (<1 % des annonces)"),
    "dist_center_km": ("localisation", "Distance haversine au centre (médiane des coordonnées train de la ville)"),
    "desc_len": ("texte", "Longueur de la description (caractères)"), "desc_words": ("texte", "Nombre de mots de la description"), "name_len": ("texte", "Longueur du titre"),
    "host_about_len": ("texte", "Longueur du « à propos » de l'hôte"), "has_host_about": ("texte", "L'hôte a rempli son « à propos »"),
    "host_tenure_years": ("dérivée", "Ancienneté comme hôte = années + mois/12"), "user_tenure_years": ("dérivée", "Ancienneté comme utilisateur"),
    "host_is_pro": ("dérivée", "Hôte avec >= 5 annonces dans la ville"), "host_entire_share": ("dérivée", "Part de logements entiers parmi les annonces de l'hôte"),
    "host_is_local": ("dérivée", "Le nom de la ville figure dans host_location (heuristique)"), "has_reviews": ("dérivée", "Au moins un avis"),
    "days_since_last_review": ("dérivée", "Jours entre le dernier avis et la collecte"), "days_since_first_review": ("dérivée", "Jours entre le premier avis et la collecte"),
    "is_studio": ("dérivée", "0 chambre (studio)"), "beds_per_person": ("dérivée", "Ratio lits / capacité"), "persons_per_bedroom": ("dérivée", "Capacité par chambre"),
    "bath_per_person": ("dérivée", "Salles de bain par personne"), "log_min_nights": ("dérivée", "log1p(minimum_nights)"), "log_max_nights": ("dérivée", "log1p(maximum_nights)"),
    "log_host_listings": ("dérivée", "log1p(annonces de l'hôte)"), "log_n_reviews": ("dérivée", "log1p(nombre d'avis)"), "log_n_reviews_ltm": ("dérivée", "log1p(avis 12 mois)"),
    "log_reviews_per_month": ("dérivée", "log1p(avis par mois)"), "log_quote_lead_days": ("dérivée", "log1p(délai du devis)"),
    "quote_month_sin": ("saisonnalité", "sin(2π mois/12) du check-in du devis"), "quote_month_cos": ("saisonnalité", "cos(2π mois/12)"),
    "quote_is_weekend": ("saisonnalité", "Check-in un vendredi ou samedi"), "quote_is_monthly": ("saisonnalité", "Séjour >= 28 nuits (tarif mensuel)"),
    "nbhd_key": ("localisation", "Clé ville|quartier (target encoding à CV interne)"),
}


def variables_table(d: pd.DataFrame) -> str:
    g = feature_groups(d.columns)
    inv = {c: k for k, v in g.items() for c in v}
    rows = ["| Variable | Famille | Définition | Type | % manquant | Jeu de variables |", "|---|---|---|---|---:|---|"]
    for c in d.columns:
        if c.startswith("am_"):
            continue
        fam, de = DEFS.get(c, ("?", "(non documentée)"))
        rows.append(f"| {c} | {fam} | {de} | {d[c].dtype} | {d[c].isna().mean() * 100:.1f} | {inv.get(c, '—')} |")
    am = amenity_cols(d.columns)
    rows.append(f"| am_* ({len(am)} colonnes) | équipements | Équipement présent (0/1) ; vocabulaire dans models/amenity_vocab.json | int8 | 0.0 | amenities |")
    return "\n".join(rows)


def run():
    cfg = load_config()
    df = pd.read_parquet(ROOT / cfg["paths"]["listings_clean"])
    texts = pd.read_parquet(ROOT / "data/processed/texts.parquet")
    d = add_text_features(df, texts)
    d = add_derived_features(d)
    d = make_targets(d, ~d["is_test"].astype(bool))
    d.to_parquet(ROOT / "data/processed/listings_features.parquet", index=False)
    unk = [c for c in d.columns if not c.startswith("am_") and c not in DEFS]
    print("colonnes non documentées :", unk)
    md = ["# Tableau des variables (Phase 3)", "",
          f"{len(d):,} annonces x {d.shape[1]} colonnes dans `listings_features.parquet`. Jeux de variables : `core` (caractéristiques du logement), `core+quote` (+ contexte du devis), "
          "`core+quote+quality` (+ notes et statut Superhost), `full` (+ activité : disponibilités, avis). Voir `feature_set()`.", "",
          "Non créées : surface par personne (aucune colonne de surface dans les données) ; langue de la description (99,4 % détectée en anglais, texte traduit à la source : aucune variance).", "",
          variables_table(d)]
    (ROOT / "reports" / "features_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(d.shape)


if __name__ == "__main__":
    run()
