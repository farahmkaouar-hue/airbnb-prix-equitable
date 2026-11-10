# Phase 4 : sélection de variables (tableau final)

Cible : log(price_eur). Candidates : jeu `core+quote+quality` après prétraitement (target encoding du quartier, one-hot). Échantillon de 60 000 annonces du train, CV 5 plis groupée par hôte ; prétraitement et sélection ajustés sur le train de chaque pli. Modèle commun d'évaluation : LightGBM (200 arbres) ; Ridge en témoin.

| Méthode                                  | Famille        |   Nb variables | RMSE LightGBM (CV)   |   MAE LightGBM |   RMSE Ridge |   MAE Ridge |   Temps sélection (s) |   Temps entraînement (s) | Stabilité (Jaccard)   |
|:-----------------------------------------|:---------------|---------------:|:---------------------|---------------:|-------------:|------------:|----------------------:|-------------------------:|:----------------------|
| Toutes les variables (référence)         | témoin         |            259 | 0.3854 ± 0.0036      |         0.2746 |       0.4288 |      0.3104 |                   0   |                      0.9 | 1.00                  |
| 60 variables aléatoires (témoin)         | témoin         |             60 | 0.4812 ± 0.0435      |         0.3572 |       0.5839 |      0.4426 |                   0   |                      0.4 | 0.13                  |
| Variance quasi nulle (>99,5 % identique) | filtre         |            251 | 0.3857 ± 0.0036      |         0.2748 |       0.4289 |      0.3105 |                   0.2 |                      0.9 | 1.00                  |
| Corrélation (f_regression, k=60)         | filtre         |             60 | 0.4293 ± 0.0051      |         0.3113 |       0.4728 |      0.3464 |                   0.1 |                      0.4 | 0.91                  |
| Information mutuelle (k=60)              | filtre         |             60 | 0.3966 ± 0.0036      |         0.2834 |       0.4613 |      0.3372 |                   2.9 |                      0.5 | 0.84                  |
| ANOVA F sur tercile de prix (k=60)       | filtre         |             60 | 0.4318 ± 0.0054      |         0.3125 |       0.4724 |      0.346  |                   0.2 |                      0.4 | 0.97                  |
| Chi² sur tercile de prix (k=60)          | filtre         |             60 | 0.4796 ± 0.0054      |         0.3531 |       0.5143 |      0.3822 |                   0.4 |                      0.4 | 0.95                  |
| RFECV (Ridge)                            | wrapper        |            131 | 0.3875 ± 0.0040      |         0.2764 |       0.4302 |      0.3114 |                   1.6 |                      0.7 | 0.59                  |
| SFS avant (Ridge, 25 parmi 80)           | wrapper        |             25 | 0.4313 ± 0.0066      |         0.314  |       0.4719 |      0.3463 |                  87.2 |                      0.2 | 0.53                  |
| Lasso (LassoCV)                          | embedded       |            226 | 0.3876 ± 0.0039      |         0.2765 |       0.4288 |      0.3104 |                  11.2 |                      0.7 | 0.85                  |
| Elastic Net (CV)                         | embedded       |            227 | 0.3875 ± 0.0039      |         0.2763 |       0.4288 |      0.3104 |                  39.5 |                      0.8 | 0.85                  |
| LightGBM gain (95 % cumulé)              | embedded       |             39 | 0.3912 ± 0.0043      |         0.2794 |       0.4563 |      0.3335 |                   0.9 |                      0.4 | 0.86                  |
| Random Forest impurité (95 % cumulé)     | embedded       |             79 | 0.3884 ± 0.0044      |         0.2774 |       0.4387 |      0.3191 |                   1.7 |                      0.6 | 0.82                  |
| Permutation (LightGBM, split interne)    | agnostique     |            143 | 0.3858 ± 0.0037      |         0.275  |       0.4365 |      0.3167 |                  22.4 |                      0.8 | 0.61                  |
| SHAP |moyenne| (95 % cumulé)             | agnostique     |             81 | 0.3876 ± 0.0040      |         0.2762 |       0.4401 |      0.32   |                   2.1 |                      0.6 | 0.84                  |
| VIF itératif (seuil 10)                  | colinéarité    |            236 | 0.3879 ± 0.0029      |         0.2773 |       0.4294 |      0.3108 |                   0.4 |                      0.8 | 0.96                  |
| PCA (95 % de variance)                   | transformation |            191 | 0.4507 ± 0.0027      |         0.3316 |       0.4433 |      0.3229 |                   0.2 |                      2.1 | —                     |
| PCA (60 composantes)                     | transformation |             60 | 0.4570 ± 0.0027      |         0.3379 |       0.4757 |      0.3528 |                   0.2 |                      0.7 | —                     |

Stabilité de Jaccard = moyenne des Jaccard sur les 10 paires de plis. « — » : non applicable (PCA, toutes variables).

## Variables retenues par le plus de méthodes sélectives (>= 3 plis sur 5 ; 7 méthodes : Corrélation (f_regression, k=60), Information mutuelle (k=60), ANOVA F sur tercile de prix (k=60), SFS avant (Ridge, 25 parmi 80), LightGBM gain (95 % cumulé), Random Forest impurité (95 % cumulé), SHAP |moyenne| (95 % cumulé))

|                           |   méthodes (sur 7) |
|:--------------------------|-------------------:|
| accommodates              |                  7 |
| nbhd_key                  |                  7 |
| am_dishwasher             |                  7 |
| bath_count                |                  7 |
| am_lock_on_bedroom_door   |                  7 |
| city_paris                |                  7 |
| bedrooms                  |                  7 |
| bath_per_person           |                  7 |
| city_mexico-city          |                  7 |
| bath_shared               |                  7 |
| review_scores_location    |                  7 |
| host_entire_share         |                  6 |
| persons_per_bedroom       |                  6 |
| n_amenities               |                  6 |
| log_min_nights            |                  6 |
| room_type_Entire home/apt |                  6 |
| am_elevator               |                  6 |
| room_type_Private room    |                  6 |
| quote_nights              |                  6 |
| has_license               |                  6 |
| am_hair_dryer             |                  5 |
| am_tv                     |                  5 |
| beds_per_person           |                  5 |
| city_tokyo                |                  5 |
| am_crib                   |                  5 |

## Importances moyennes (rang 1 = plus important)

|                           |   rf_impurity |   lgb_gain |   permutation |   shap |
|:--------------------------|--------------:|-----------:|--------------:|-------:|
| nbhd_key                  |             1 |          1 |             1 |      1 |
| accommodates              |             2 |          2 |             3 |      2 |
| quote_nights              |             9 |          5 |             2 |      3 |
| bedrooms                  |             3 |          3 |             4 |      4 |
| dist_center_km            |            14 |         12 |             6 |      5 |
| latitude                  |             5 |          7 |             7 |      6 |
| bath_count                |             4 |          4 |             5 |      7 |
| log_quote_lead_days       |            18 |          8 |             8 |      8 |
| bath_shared               |             7 |          6 |            10 |      9 |
| am_dishwasher             |            20 |         13 |            15 |     10 |
| city_new-york-city        |            25 |          9 |             9 |     11 |
| review_scores_location    |            19 |         14 |            13 |     12 |
| review_scores_cleanliness |            26 |         19 |            16 |     13 |
| room_type_Entire home/apt |             6 |         10 |            14 |     14 |
| host_entire_share         |            16 |         21 |            17 |     15 |
| longitude                 |             8 |         11 |            11 |     16 |
| quote_is_weekend          |            53 |         28 |            18 |     17 |
| city_mexico-city          |            13 |         17 |            12 |     18 |
| review_scores_value       |            36 |         33 |            19 |     19 |
| am_air_conditioning       |            43 |         31 |            20 |     20 |

