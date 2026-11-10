# Tableau des variables (Phase 3)

485,128 annonces x 275 colonnes dans `listings_features.parquet`. Jeux de variables : `core` (caractéristiques du logement), `core+quote` (+ contexte du devis), `core+quote+quality` (+ notes et statut Superhost), `full` (+ activité : disponibilités, avis). Voir `feature_set()`.

Non créées : surface par personne (aucune colonne de surface dans les données) ; langue de la description (99,4 % détectée en anglais, texte traduit à la source : aucune variance).

| Variable | Famille | Définition | Type | % manquant | Jeu de variables |
|---|---|---|---|---:|---|
| city | id/groupe | Ville (une des 15) | str | 0.0 | location |
| country | id/groupe | Pays | str | 0.0 | — |
| scrape_date | id/groupe | Date de collecte (une par ville) | datetime64[us] | 0.0 | — |
| id | id/groupe | Identifiant de l'annonce | int64 | 0.0 | — |
| host_id | id/groupe | Identifiant de l'hôte (groupe de validation) | int64 | 0.0 | — |
| hosts_time_as_user_years | brute | Ancienneté de l'hôte comme utilisateur (années entières) | float64 | 0.0 | — |
| hosts_time_as_user_months | brute | idem, mois restants (0-11) | float64 | 0.0 | — |
| hosts_time_as_host_years | brute | Ancienneté comme hôte (années entières) | float64 | 0.0 | — |
| hosts_time_as_host_months | brute | idem, mois restants (0-11) | float64 | 0.0 | — |
| host_location | brute | Lieu déclaré de l'hôte (texte) | str | 25.2 | — |
| host_is_superhost | brute | Statut Superhost (cible de la classification 1) | float64 | 0.0 | host_status |
| host_listings_count | brute | Nombre d'annonces de l'hôte (déclaré) | int64 | 0.0 | — |
| host_has_profile_pic | brute | L'hôte a une photo de profil | float64 | 0.0 | host |
| host_identity_verified | brute | Identité de l'hôte vérifiée | float64 | 0.0 | host |
| neighbourhood_cleansed | brute | Quartier (nomenclature Inside Airbnb) | str | 0.0 | — |
| neighbourhood_group_cleansed | brute | Groupe de quartiers (très incomplet) | str | 69.2 | — |
| latitude | brute | Latitude | float64 | 0.0 | location |
| longitude | brute | Longitude | float64 | 0.0 | location |
| property_type | brute | Type de propriété (141 modalités ; <1 % regroupées en 'infrequent') | str | 0.0 | property |
| room_type | brute | Type de logement (4 modalités) | str | 0.0 | property |
| accommodates | brute | Capacité d'accueil | int64 | 0.0 | property |
| bedrooms | brute | Nombre de chambres (winsorisé au q99,9 train) | float64 | 0.1 | property |
| beds | brute | Nombre de lits (winsorisé) | float64 | 2.3 | property |
| price_local | cible/dérivée | Prix du devis en devise locale | float64 | 25.3 | — |
| minimum_nights | brute | Nuits minimum | float64 | 3.4 | — |
| maximum_nights | brute | Nuits maximum (plafonné à 1125) | float64 | 3.4 | — |
| minimum_minimum_nights | brute | Nuits minimum, plus petite valeur sur 365 j | float64 | 3.4 | — |
| maximum_minimum_nights | brute | Nuits minimum, plus grande valeur | float64 | 3.4 | — |
| minimum_maximum_nights | brute | Nuits maximum, plus petite valeur | float64 | 3.4 | — |
| maximum_maximum_nights | brute | Nuits maximum, plus grande valeur | float64 | 3.4 | — |
| minimum_nights_avg_ntm | brute | Moyenne des nuits minimum, 12 prochains mois | float64 | 0.0 | — |
| maximum_nights_avg_ntm | brute | Moyenne des nuits maximum, 12 prochains mois | float64 | 0.0 | — |
| has_availability | brute/activité | Annonce marquée disponible | float64 | 0.8 | activity |
| availability_30 | activité (conséquence du prix) | Jours disponibles sur 30 j | int64 | 0.0 | activity |
| availability_60 | activité | Jours disponibles sur 60 j | int64 | 0.0 | activity |
| availability_90 | activité | Jours disponibles sur 90 j | int64 | 0.0 | activity |
| availability_365 | activité | Jours disponibles sur 365 j | int64 | 0.0 | activity |
| number_of_reviews | activité | Nombre total d'avis | int64 | 0.0 | — |
| number_of_reviews_ltm | activité | Avis sur 12 mois | int64 | 0.0 | — |
| number_of_reviews_l30d | activité | Avis sur 30 j | int64 | 0.0 | activity |
| availability_eoy | activité | Jours disponibles jusqu'à fin d'année | int64 | 0.0 | — |
| number_of_reviews_ly | activité | Avis l'an dernier | int64 | 0.0 | — |
| estimated_occupancy_l365d | FUITE prix | Occupation estimée sur 365 j (modèle Inside Airbnb) : conséquence, exclue | int64 | 0.0 | — |
| estimated_revenue_l365d | FUITE prix | Revenu estimé sur 365 j = prix x occupation : dérivé du prix, exclu de la régression | float64 | 25.3 | — |
| first_review | brute | Date du premier avis | datetime64[us] | 18.6 | — |
| last_review | brute | Date du dernier avis | datetime64[us] | 18.6 | — |
| review_scores_rating | qualité | Note globale (0-5) | float64 | 18.6 | quality |
| review_scores_accuracy | qualité | Note exactitude | float64 | 18.6 | quality |
| review_scores_cleanliness | qualité | Note propreté | float64 | 18.6 | quality |
| review_scores_checkin | qualité | Note arrivée | float64 | 18.6 | quality |
| review_scores_communication | qualité | Note communication | float64 | 18.6 | quality |
| review_scores_location | qualité | Note emplacement | float64 | 18.6 | quality |
| review_scores_value | qualité | Note rapport qualité/prix | float64 | 18.6 | quality |
| calculated_host_listings_count | brute | Annonces de l'hôte dans la ville (calculé Inside Airbnb) | int64 | 0.0 | — |
| calculated_host_listings_count_entire_homes | brute | ... dont logements entiers | int64 | 0.0 | — |
| calculated_host_listings_count_private_rooms | brute | ... dont chambres privées | int64 | 0.0 | — |
| calculated_host_listings_count_shared_rooms | brute | ... dont chambres partagées | int64 | 0.0 | — |
| reviews_per_month | activité | Avis par mois | float64 | 18.6 | — |
| currency | id/groupe | Devise d'origine | str | 0.0 | — |
| is_test | drapeau | Annonce du test final (ne jamais utiliser avant la phase 14) | bool | 0.0 | — |
| quote_lead_days | devis | Jours entre la collecte et le check-in du devis | float64 | 21.4 | — |
| quote_nights | devis | Durée du séjour du devis (1 à 31 nuits) | float64 | 21.4 | quote |
| quote_checkin_month | devis | Mois du check-in du devis | float64 | 21.4 | — |
| quote_checkin_dow | devis | Jour de semaine du check-in (0 = lundi) | float64 | 21.4 | — |
| is_stale | activité | Annonce non rafraîchie à la dernière collecte (« previous scrape ») | bool | 0.0 | activity |
| has_license | hôte | Un numéro de licence est renseigné | bool | 0.0 | host |
| bath_count | logement | Nombre de salles de bain (parsé ; demi = 0,5) | float64 | 0.1 | property |
| bath_shared | logement | Salle de bain partagée (0/1) | float64 | 0.1 | property |
| n_amenities | équipements | Nombre total d'équipements normalisés | int64 | 0.0 | amenities |
| n_amenities_other | équipements | Équipements hors vocabulaire (<1 % des annonces) | int64 | 0.0 | amenities |
| fx_rate_per_eur | id/groupe | Taux BCE utilisé (devise pour 1 EUR) | float64 | 0.0 | — |
| has_price | drapeau | Prix connu | bool | 0.0 | — |
| price_eur | cible/dérivée | Prix par nuit converti en EUR (taux BCE à la date de collecte) | float64 | 25.3 | — |
| log_price_eur | cible | CIBLE RÉGRESSION : log(price_eur) | float64 | 25.3 | — |
| is_inconsistent | drapeau | Capacité / lits / chambres / nuits incohérents | bool | 0.0 | — |
| dist_center_km | localisation | Distance haversine au centre (médiane des coordonnées train de la ville) | float64 | 0.0 | location |
| is_geo_outlier | drapeau | À plus de 100 km du centre de la ville | bool | 0.0 | — |
| is_near_duplicate | drapeau | Même hôte, mêmes coordonnées, type, capacité, chambres | bool | 0.0 | — |
| is_price_outlier | drapeau | Prix hors clôtures 3 x IQR de la ville (train) | bool | 0.0 | — |
| in_reg_set | drapeau | Utilisable pour la régression (prix connu, ni outlier, ni incohérent, ni coordonnées aberrantes) | bool | 0.0 | — |
| desc_len | texte | Longueur de la description (caractères) | int64 | 0.0 | text |
| desc_words | texte | Nombre de mots de la description | int64 | 0.0 | text |
| name_len | texte | Longueur du titre | int64 | 0.0 | text |
| host_about_len | texte | Longueur du « à propos » de l'hôte | int64 | 0.0 | host |
| has_host_about | texte | L'hôte a rempli son « à propos » | float64 | 0.0 | host |
| host_tenure_years | dérivée | Ancienneté comme hôte = années + mois/12 | float64 | 0.0 | host |
| user_tenure_years | dérivée | Ancienneté comme utilisateur | float64 | 0.0 | host |
| host_is_pro | dérivée | Hôte avec >= 5 annonces dans la ville | float64 | 0.0 | host |
| host_entire_share | dérivée | Part de logements entiers parmi les annonces de l'hôte | float64 | 0.0 | host |
| host_is_local | dérivée | Le nom de la ville figure dans host_location (heuristique) | float64 | 25.2 | host |
| has_reviews | dérivée | Au moins un avis | float64 | 0.0 | activity |
| days_since_last_review | dérivée | Jours entre le dernier avis et la collecte | float64 | 18.6 | activity |
| days_since_first_review | dérivée | Jours entre le premier avis et la collecte | float64 | 18.6 | activity |
| is_studio | dérivée | 0 chambre (studio) | float64 | 0.1 | property |
| beds_per_person | dérivée | Ratio lits / capacité | float64 | 2.3 | property |
| persons_per_bedroom | dérivée | Capacité par chambre | float64 | 0.1 | property |
| bath_per_person | dérivée | Salles de bain par personne | float64 | 0.1 | property |
| log_min_nights | dérivée | log1p(minimum_nights) | float64 | 3.4 | property |
| log_max_nights | dérivée | log1p(maximum_nights) | float64 | 3.4 | property |
| log_host_listings | dérivée | log1p(annonces de l'hôte) | float64 | 0.0 | host |
| log_n_reviews | dérivée | log1p(nombre d'avis) | float64 | 0.0 | activity |
| log_n_reviews_ltm | dérivée | log1p(avis 12 mois) | float64 | 0.0 | activity |
| log_reviews_per_month | dérivée | log1p(avis par mois) | float64 | 18.6 | activity |
| log_quote_lead_days | dérivée | log1p(délai du devis) | float64 | 21.4 | quote |
| quote_month_sin | saisonnalité | sin(2π mois/12) du check-in du devis | float64 | 21.4 | quote |
| quote_month_cos | saisonnalité | cos(2π mois/12) | float64 | 21.4 | quote |
| quote_is_weekend | saisonnalité | Check-in un vendredi ou samedi | float64 | 21.4 | quote |
| quote_is_monthly | saisonnalité | Séjour >= 28 nuits (tarif mensuel) | float64 | 21.4 | quote |
| nbhd_key | localisation | Clé ville|quartier (target encoding à CV interne) | str | 0.0 | location |
| is_superhost | cible | CIBLE CLASSIF 1 : l'hôte est Superhost (0/1) | float64 | 0.0 | — |
| price_tier | cible | CIBLE CLASSIF 2 : tranche de prix dans la ville (0 bas, 1 moyen, 2 haut ; terciles train) | float64 | 25.3 | — |
| am_* (164 colonnes) | équipements | Équipement présent (0/1) ; vocabulaire dans models/amenity_vocab.json | int8 | 0.0 | amenities |
