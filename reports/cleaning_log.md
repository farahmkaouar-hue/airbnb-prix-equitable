# Journal de nettoyage (Phase 2)

Base d'entrée : listings_all.parquet. Sortie : 485,128 lignes x 244 colonnes (listings_clean.parquet). Aucune ligne n'est supprimée : les annonces douteuses sont signalées par des indicateurs.

| Étape | Lignes / valeurs affectées | Pourquoi |
|---|---:|---|
| split | 69,304 | annonces du test final (14.3 %), groupées par hôte ; tous les seuils ci-dessous sont appris sur les 415,824 annonces train |
| colonnes 100 % vides | 13 | supprimées : neighborhood_overview, host_since, host_response_time, host_response_rate, host_acceptance_rate, host_thumbnail_url, host_neighbourhood, host_total_listings_count, host_verifications, neighbourhood, bathrooms, calendar_updated, instant_bookable |
| contexte du devis | 381,464 | price_quote_checkin/checkout_date -> quote_lead_days, quote_nights, quote_checkin_month, quote_checkin_dow (le prix est un devis de 1 à 31 nuits, check-in de 0 à 369 jours après la collecte : saisonnalité et remises mensuelles) |
| colonnes inutiles / sensibles | 18 | URLs, identifiants techniques, price_quote_* (doublon exact du prix, fuite), host_name (RGPD) supprimés ; name/description/host_about copiés dans texts.parquet |
| booléen host_is_superhost | 485,128 | 't'/'f' -> booléen (0 manquants conservés) |
| booléen host_has_profile_pic | 485,128 | 't'/'f' -> booléen (0 manquants conservés) |
| booléen host_identity_verified | 485,128 | 't'/'f' -> booléen (0 manquants conservés) |
| booléen has_availability | 481,105 | 't'/'f' -> booléen (4023 manquants conservés) |
| dates | 485,128 | first_review, last_review, scrape_date -> datetime |
| source -> is_stale | 132,539 | 'previous scrape' = annonce non rafraîchie à la dernière collecte (80 % sans prix) |
| license -> has_license | 237,299 | texte libre (numéros d'enregistrement, donnée quasi personnelle) réduit à un indicateur |
| bathrooms_text -> bath_count, bath_shared | 484,570 | parsé en nombre (demi-salle de bain = 0,5) et indicateur partagée (58,800 partagées) ; 558 non parsables laissés NaN |
| équipements | 164 | libellés normalisés (minuscule, détails retirés) : 32,755 libellés distincts -> 164 conservés (>= 1% des annonces train), les autres regroupés dans n_amenities_other ; vocabulaire dans models/amenity_vocab.json |
| prix -> EUR | 362,593 | price_eur = prix local / taux BCE du dernier fixing <= date de collecte de la ville ; devise d'origine conservée dans `currency` ; 122,535 annonces sans prix (aucun prix récupérable : le JSON price_quote_raw n'a pas de price_per_night pour elles) conservées avec price_eur = NaN |
| maximum_nights | 125 | valeurs sentinelles (max 2 147 483 647) plafonnées à 1125 |
| winsorisation bedrooms | 353 | plafonné au quantile 0.999 du train = 9 |
| winsorisation beds | 409 | plafonné au quantile 0.999 du train = 15 |
| incohérence : capacité >= 4 sans chambre ni lit | 187 | capacité élevée sans couchage déclaré |
| incohérence : lits > 3 x capacité | 678 | nombre de lits incompatible avec la capacité |
| incohérence : chambres > 2 x capacité | 925 | nombre de chambres incompatible avec la capacité |
| incohérence : minimum_nights > maximum_nights | 6 | contradiction de règles de séjour |
| is_inconsistent (union) | 1,548 | signalées ; exclues de l'entraînement de la régression, jamais supprimées de listings_clean |
| coordonnées aberrantes | 0 | > 100 km de la médiane des coordonnées de la ville (centre = médiane train) |
| doublons exacts (city, id) | 0 | aucun |
| quasi-doublons (même hôte, mêmes coordonnées, type, capacité, chambres) | 24,729 | signalés, non supprimés : souvent des unités distinctes d'un même immeuble (hôtels, résidences) ; le split groupé par hôte neutralise la fuite |
| outliers de prix | 651 | hors [Q1 - 3.0 x IQR ; Q3 + 3.0 x IQR] de log(price_eur) par ville (train) : 0 trop bas, 651 trop hauts (0.18 % des prix) ; signalés (is_price_outlier), exclus de l'entraînement seulement ; impact mesuré par outlier_impact |
| jeu de régression (in_reg_set) | 360,782 | = avec prix, sans outlier de prix, sans coordonnées aberrantes, sans incohérence (74.4 % des annonces) |

## Stratégie d'imputation par colonne (appliquée en phase 3/6 dans les Pipelines)

| Colonne | % manquant | Stratégie (appliquée dans le Pipeline, ajustée sur le train) |
|---|---:|---|
| neighbourhood_group_cleansed | 69.2 | catégorie « manquant » |
| price_local | 25.3 | cible : jamais imputée ; annonces sans prix exclues de la régression |
| estimated_revenue_l365d | 25.3 | médiane par ville (train) + indicateur de manquant |
| price_eur | 25.3 | cible : jamais imputée ; annonces sans prix exclues de la régression |
| log_price_eur | 25.3 | cible : jamais imputée ; annonces sans prix exclues de la régression |
| host_location | 25.2 | catégorie « manquant » |
| quote_lead_days | 21.4 | médiane par ville (train) + indicateur de manquant |
| quote_checkin_dow | 21.4 | médiane par ville (train) + indicateur de manquant |
| quote_checkin_month | 21.4 | médiane par ville (train) + indicateur de manquant |
| quote_nights | 21.4 | médiane par ville (train) + indicateur de manquant |
| review_scores_value | 18.6 | absence d'avis : indicateur `has_reviews` + imputation médiane par ville (linéaires) ou NaN natif (boosting) |
| review_scores_location | 18.6 | absence d'avis : indicateur `has_reviews` + imputation médiane par ville (linéaires) ou NaN natif (boosting) |
| review_scores_checkin | 18.6 | absence d'avis : indicateur `has_reviews` + imputation médiane par ville (linéaires) ou NaN natif (boosting) |
| review_scores_communication | 18.6 | absence d'avis : indicateur `has_reviews` + imputation médiane par ville (linéaires) ou NaN natif (boosting) |
| review_scores_accuracy | 18.6 | absence d'avis : indicateur `has_reviews` + imputation médiane par ville (linéaires) ou NaN natif (boosting) |
| review_scores_cleanliness | 18.6 | absence d'avis : indicateur `has_reviews` + imputation médiane par ville (linéaires) ou NaN natif (boosting) |
| reviews_per_month | 18.6 | absence d'avis : indicateur `has_reviews` + imputation médiane par ville (linéaires) ou NaN natif (boosting) |
| last_review | 18.6 | absence d'avis : indicateur `has_reviews` + imputation médiane par ville (linéaires) ou NaN natif (boosting) |
| review_scores_rating | 18.6 | absence d'avis : indicateur `has_reviews` + imputation médiane par ville (linéaires) ou NaN natif (boosting) |
| first_review | 18.6 | absence d'avis : indicateur `has_reviews` + imputation médiane par ville (linéaires) ou NaN natif (boosting) |
| minimum_minimum_nights | 3.4 | médiane par ville (train) + indicateur de manquant |
| minimum_maximum_nights | 3.4 | médiane par ville (train) + indicateur de manquant |
| maximum_minimum_nights | 3.4 | médiane par ville (train) + indicateur de manquant |
| minimum_nights | 3.4 | médiane par ville (train) + indicateur de manquant |
| maximum_nights | 3.4 | médiane par ville (train) + indicateur de manquant |
| maximum_maximum_nights | 3.4 | médiane par ville (train) + indicateur de manquant |
| beds | 2.3 | médiane par ville (train) + indicateur de manquant |
| has_availability | 0.8 | mode + indicateur de manquant |
| bath_shared | 0.1 | médiane par ville (train) + indicateur de manquant |
| bath_count | 0.1 | médiane par ville (train) + indicateur de manquant |
| bedrooms | 0.1 | médiane par ville (train) + indicateur de manquant |
| hosts_time_as_host_years | 0.0 | médiane par ville (train) + indicateur de manquant |
| hosts_time_as_host_months | 0.0 | médiane par ville (train) + indicateur de manquant |
| hosts_time_as_user_months | 0.0 | médiane par ville (train) + indicateur de manquant |
| hosts_time_as_user_years | 0.0 | médiane par ville (train) + indicateur de manquant |



## Impact de la règle d'outliers (LightGBM de référence, validation sur 20 % des hôtes du train ; cibles = log(price_eur))

| Stratégie d'entraînement                     |   n train |   RMSE (validation sans outliers) |   MAE (sans outliers) |   RMSE (validation avec outliers) |   MAE (avec outliers) |
|:---------------------------------------------|----------:|----------------------------------:|----------------------:|----------------------------------:|----------------------:|
| Avec outliers (rien retiré)                  |    246400 |                            0.4654 |                0.345  |                            0.4836 |                0.3496 |
| Sans outliers (règle 3 x IQR)                |    245950 |                            0.4642 |                0.3438 |                            0.4829 |                0.3485 |
| Winsorisation (plafonnée à la clôture haute) |    246400 |                            0.4652 |                0.3448 |                            0.4837 |                0.3494 |

Validation : 63,574 annonces dont 112 outliers. Le test final n'est pas utilisé.
