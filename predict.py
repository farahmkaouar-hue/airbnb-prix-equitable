"""Exemple d'utilisation du modèle final de prix : python predict.py

Le modèle (models/final_model.joblib) contient le Pipeline complet : imputation par ville, encodage, LightGBM réglé, intervalle conforme à 90 %.
Le prix renvoyé est le prix par nuit (EUR) d'un devis de séjour (par défaut : 2 nuits, départ dans 14 jours) ; la fourchette vise 90 % de couverture
pour des villes connues du modèle. Ce sont des ASSOCIATIONS apprises sur des prix affichés, pas des prix payés ni des effets garantis.
"""
from src.pricing import load_model

pm = load_model()

# Une annonce est décrite par un dictionnaire PARTIEL : les champs manquants sont complétés par le profil médian de la ville.
studio_paris = {"city": "paris", "room_type": "Entire home/apt", "accommodates": 2, "bedrooms": 0, "beds": 1, "bath_count": 1,
                "latitude": 48.8566, "longitude": 2.3522, "amenities": ["Wifi", "Kitchen", "Elevator", "Hair dryer", "Hot water"], "minimum_nights": 2}
grand_tokyo = {"city": "tokyo", "room_type": "Entire home/apt", "accommodates": 6, "bedrooms": 2, "beds": 4, "bath_count": 1.5,
               "latitude": 35.69, "longitude": 139.70, "amenities": ["Wifi", "Kitchen", "Air conditioning", "Washer"]}

for nom, annonce in [("Studio à Paris (centre)", studio_paris), ("2 chambres à Tokyo", grand_tokyo)]:
    out = pm.predict_price(annonce)
    lo, hi = out["fourchette_eur"]
    print(f"{nom} : {out['prix_recommande_eur']:.0f} EUR / nuit, fourchette à 90 % : {lo:.0f} - {hi:.0f} EUR (prix moyen attendu : {out['prix_moyen_attendu_eur']:.0f} EUR)")

# « Et si j'ajoute la climatisation ? » : même annonce, un seul changement (association apprise, pas effet causal)
sans, avec = dict(studio_paris), dict(studio_paris, amenities=studio_paris["amenities"] + ["Air conditioning"])
a, b = pm.predict_price(sans)["prix_recommande_eur"], pm.predict_price(avec)["prix_recommande_eur"]
print(f"Studio à Paris avec climatisation : {a:.0f} -> {b:.0f} EUR ({(b / a - 1) * 100:+.1f} %)")

# Nouvelle ville ? Le modèle refuse les villes inconnues : il faut recalibrer le niveau de prix (voir le rapport, chapitre « Apport métier »).
try:
    pm.predict_price(dict(studio_paris, city="vienne"))
except ValueError as e:
    print("Ville inconnue :", e)
