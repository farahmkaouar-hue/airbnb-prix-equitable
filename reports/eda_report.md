# Analyse exploratoire - Inside Airbnb (base globale)

485,128 annonces, 21,726,192 avis, 15 villes, 83 colonnes après nettoyage.

## Vue d'ensemble par ville

| city | pays | devise | annonces | avis | prix_median | pct_prix_dispo | pct_logement_entier | pct_hotes_multi | note_mediane | date_collecte |
|---|---|---|---|---|---|---|---|---|---|---|
| london | united-kingdom | GBP | 91,954 | 2,323,969 | 175.0 | 65.1 | 66.4 | 55.5 | 4.83 | 2026-09-17 |
| paris | france | EUR | 77,149 | 2,503,875 | 218.0 | 56.4 | 88.6 | 36.1 | 4.85 | 2026-09-16 |
| los-angeles | united-states | USD | 43,781 | 1,890,200 | 215.0 | 83.8 | 73.8 | 65.4 | 4.91 | 2026-09-14 |
| rome | italy | EUR | 37,451 | 2,622,221 | 188.0 | 90.6 | 74.5 | 61.6 | 4.86 | 2026-09-17 |
| tokyo | japan | JPY | 35,970 | 1,677,064 | 20322.0 | 93.4 | 84.1 | 91.9 | 4.83 | 2026-09-25 |
| mexico-city | mexico | MXN | 30,468 | 1,795,294 | 1500.0 | 93.1 | 66.8 | 69.6 | 4.84 | 2026-09-23 |
| new-york-city | united-states | USD | 29,918 | 999,099 | 182.0 | 64.9 | 54.8 | 56.7 | 4.86 | 2026-09-14 |
| lisbon | portugal | EUR | 25,020 | 1,950,076 | 140.0 | 88.1 | 74.6 | 75.1 | 4.77 | 2026-09-19 |
| madrid | spain | EUR | 23,109 | 1,524,765 | 151.0 | 81.3 | 70.0 | 73.5 | 4.75 | 2026-09-17 |
| toronto | canada | CAD | 22,050 | 729,202 | 161.0 | 76.0 | 67.9 | 46.8 | 4.9 | 2026-09-16 |
| sydney | australia | AUD | 20,833 | 829,163 | 338.0 | 85.1 | 78.9 | 63.4 | 4.84 | 2026-09-16 |
| barcelona | spain | EUR | 15,236 | 1,127,248 | 191.0 | 82.8 | 69.6 | 80.5 | 4.73 | 2026-09-17 |
| berlin | germany | EUR | 12,376 | 708,768 | 130.0 | 65.9 | 68.2 | 46.4 | 4.86 | 2026-09-19 |
| amsterdam | the-netherlands | EUR | 10,612 | 571,927 | 285.0 | 56.9 | 80.8 | 18.7 | 4.92 | 2026-09-16 |
| lyon | france | EUR | 9,201 | 473,321 | 110.0 | 55.1 | 79.7 | 36.0 | 4.83 | 2026-09-17 |

Prix en devise locale ; `pct_prix_dispo` = part des annonces avec un prix ; `pct_hotes_multi` = part des annonces dont l'hôte en possède plusieurs.

![Annonces et avis](figures/02_counts.png)

Les prix médians du tableau sont en devise locale : pas de comparaison directe entre devises (pas de graphique commun).

## Qualité des données

- 74.7 % des annonces ont un prix ; 27.3 % proviennent d'une collecte précédente (`is_stale`) et n'ont en général ni prix ni revenu estimé.
- 13 colonnes sont vides à 100 % dans la source et ont été retirées.
- 633 prix extrêmes et 0 coordonnées aberrantes sont signalés, non supprimés.

![Valeurs manquantes](figures/01_missing.png)

![Prix disponible](figures/09_price_available.png)

## Prix

![Distribution des prix](figures/04_price_dist.png)

![Type de logement](figures/05_room_type.png)

## Facteurs associés au prix

Les variables les plus liées au prix relatif (Spearman) : `accommodates` (+0.61), `beds` (+0.51), `bedrooms` (+0.50) ; liens négatifs : `bath_shared` (-0.40), `calculated_host_listings_count_private_rooms` (-0.37), `minimum_nights` (-0.19). Ce sont des associations, pas des effets causaux.

![Corrélations](figures/06_correlations.png)

## Avis et notes

![Notes](figures/07_rating.png)

![Avis par année](figures/08_reviews_year.png)

## Journal de nettoyage

1. Doublons (city, id) retirés : 0.
2. Colonnes retirées (27) : 13 vides, 0 constantes, URL et colonnes techniques. Liste : bathrooms, calendar_last_scraped, calendar_updated, host_acceptance_rate, host_neighbourhood, host_picture_url, host_profile_url, host_response_rate, host_response_time, host_since, host_thumbnail_url, host_total_listings_count, host_url, host_verifications, instant_bookable, last_scraped, listing_url, neighborhood_overview, neighbourhood, picture_url, price_quote_checkin_date, price_quote_checkout_date, price_quote_price_per_night, price_quote_raw, price_quote_total_price, scrape_id, source
3. Types : t/f -> booléens, pourcentages -> fractions 0-1, dates -> datetime.
4. bathrooms (vide dans la source) reconstruit depuis bathrooms_text ; ajout de bath_shared.
5. amenities : n_amenities + 8 indicateurs (has_wifi, has_kitchen, has_ac, has_washer, has_elevator, has_parking, has_pool, has_workspace).
6. license -> license_type (missing / exempt / mobility_lease / other) et has_license ; texte brut retiré.
7. minimum_nights plafonné à 365 (130 valeurs), maximum_nights à 1125 (125).
8. bedrooms > 20 ou beds > 50 mis à NaN (41 annonces, saisies aberrantes).
9. Notes égales à 0 mises à NaN (45 annonces) : l'échelle Airbnb commence à 1.
10. Prix : 122535 manquants (conservés, has_price=False) ; 633 valeurs extrêmes signalées (is_price_outlier), par ville : {'amsterdam': 6, 'barcelona': 1, 'berlin': 3, 'lisbon': 56, 'london': 61, 'los-angeles': 11, 'lyon': 6, 'madrid': 38, 'mexico-city': 59, 'new-york-city': 7, 'paris': 97, 'rome': 126, 'sydney': 44, 'tokyo': 118, 'toronto': 0}.
11. Coordonnées : 0 annonces à plus de 1.0° de la médiane de leur ville (is_geo_outlier).
12. Sous-ensemble modèle : 361,960 annonces sur 485,128 (prix valide, hors extrêmes de prix et de géolocalisation).
13. Avis : 21,726,192 lignes conservées ; 2,399 commentaires vides, 96 messages automatiques, 0 avis dont l'annonce est absente (drapeaux is_*). Aucune ligne supprimée : pas de doublon exact, et `id` n'est pas unique. HTML retiré, reviewer_name supprimé (donnée personnelle inutile).

