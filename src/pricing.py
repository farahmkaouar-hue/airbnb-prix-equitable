"""Outil de recommandation de prix (Phase 13) : classe PriceModel (prétraitement + LightGBM + intervalle conforme).

Le prix prédit est le prix par nuit (EUR) d'un devis de séjour ; la fourchette est un intervalle de prédiction conforme « Mondrian »
(calibré par tranche de prix prédit sur les résidus hors échantillon de la validation croisée), de couverture visée 90 %.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.cleaning import norm_amenity
from src.features import haversine

ROOT = Path(__file__).resolve().parents[1]


class PriceModel:
    def __init__(self, pre, model, feats, smear, bin_edges, bin_q, templates, centers, nb_centroids, amenity_vocab, alpha=0.1):
        self.pre, self.model, self.feats = pre, model, feats
        self.smear, self.bin_edges, self.bin_q = smear, np.asarray(bin_edges), np.asarray(bin_q)
        self.templates, self.centers, self.nb_centroids, self.amenity_vocab, self.alpha = templates, centers, nb_centroids, amenity_vocab, alpha

    # ----------------------------------------------------------------------------------------- prédiction sur une table de features
    def predict_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        Z = self.pre.transform(X)
        lp = self.model.predict(Z)
        b = np.clip(np.searchsorted(self.bin_edges, lp, side="right") - 1, 0, len(self.bin_q) - 1)
        q = self.bin_q[b]
        # prix recommandé = estimation MÉDIANE exp(log_pred) (minimise l'erreur absolue en euros) ; l'intervalle conforme est défini en log, donc sans smearing ;
        # mean_eur = prix MOYEN attendu (avec correction de biais de la retransformation), utile pour des totaux, pas pour un prix individuel
        out = pd.DataFrame({"log_pred": lp, "price_eur": np.exp(lp), "low_eur": np.exp(lp - q), "high_eur": np.exp(lp + q), "mean_eur": np.exp(lp) * self.smear}, index=X.index)
        return out

    # ----------------------------------------------------------------------------------------- une annonce décrite par un dictionnaire
    def build_row(self, listing: dict) -> pd.DataFrame:
        """Complète une annonce partielle avec le profil médian de sa ville, puis recalcule les variables dérivées."""
        city = listing["city"]
        if city not in self.templates:
            raise ValueError(f"Ville inconnue du modèle : {city}. Villes : {sorted(self.templates)}. (Pour une nouvelle ville : recalibrer, voir le rapport.)")
        row = self.templates[city].copy()
        for k, v in listing.items():
            if k in row.index:
                row[k] = v
        lat, lon = float(row["latitude"]), float(row["longitude"])
        c_lat, c_lon = self.centers[city]
        row["dist_center_km"] = float(haversine(lat, lon, c_lat, c_lon))
        if "neighbourhood_cleansed" in listing:
            row["nbhd_key"] = f"{city}|{listing['neighbourhood_cleansed']}"
        else:  # quartier le plus proche des coordonnées
            nb = self.nb_centroids[self.nb_centroids.city == city]
            j = int(np.argmin(haversine(lat, lon, nb.latitude.values, nb.longitude.values)))
            row["nbhd_key"] = f"{city}|{nb.neighbourhood_cleansed.iloc[j]}"
        acc = max(float(row["accommodates"]), 1.0)
        beds, bedrooms = float(row["beds"]), float(row["bedrooms"])
        row["beds_per_person"], row["persons_per_bedroom"] = beds / acc, acc / max(bedrooms, 1.0)
        row["bath_per_person"] = float(row["bath_count"]) / acc
        row["is_studio"] = float(bedrooms == 0)
        if "minimum_nights" in listing:
            row["log_min_nights"] = np.log1p(listing["minimum_nights"])
        if "maximum_nights" in listing:
            row["log_max_nights"] = np.log1p(min(listing["maximum_nights"], 1125))
        # contexte du devis (par défaut : séjour de 2 nuits, départ dans 14 jours)
        nights = listing.get("quote_nights", 2)
        lead = listing.get("quote_lead_days", 14)
        month = listing.get("quote_checkin_month", pd.Timestamp.today().month)
        row["quote_nights"], row["log_quote_lead_days"] = nights, np.log1p(lead)
        row["quote_month_sin"], row["quote_month_cos"] = np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)
        row["quote_is_monthly"] = float(nights >= 28)
        row["quote_is_weekend"] = float(listing.get("quote_checkin_weekend", 0))
        # équipements : liste de libellés (normalisés comme au nettoyage)
        if "amenities" in listing:
            names = {norm_amenity(a) for a in listing["amenities"]}
            for a in self.amenity_vocab:
                row["am_" + a] = float(a in names)
            row["n_amenities_other"] = len(names - set(self.amenity_vocab))
            row["n_amenities"] = len(names)
        return row.to_frame().T.astype({c: float for c in row.index if c not in ("city", "room_type", "property_type", "nbhd_key")})

    def predict_price(self, listing: dict) -> dict:
        X = self.build_row(listing)
        X = X.assign(**{c: listing["city"] if c == "city" else X[c] for c in ["city"]})
        p = self.predict_frame(X).iloc[0]
        return {"prix_recommande_eur": round(float(p.price_eur), 1), "fourchette_eur": (round(float(p.low_eur), 1), round(float(p.high_eur), 1)), "prix_moyen_attendu_eur": round(float(p.mean_eur), 1), "couverture_visee": 1 - self.alpha}


def load_model(path=ROOT / "models" / "final_model.joblib") -> PriceModel:
    import joblib

    return joblib.load(path)
