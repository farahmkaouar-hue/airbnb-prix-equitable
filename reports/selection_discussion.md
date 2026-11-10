# Phase 4 : discussion de la sélection de variables

Tous les chiffres viennent de `reports/selection_results.csv`, `selection_importances.csv` et `selection_frequency.csv` (5 plis groupés par hôte, 60 000 annonces du train).

## Lecture du tableau
- **Référence (toutes les variables, 259 colonnes après prétraitement)** : RMSE LightGBM 0,3854. Un tirage de 60 variables au hasard donne 0,4812 : la sélection a donc un vrai sens.
- **Aucune méthode ne bat la référence.** LightGBM supporte bien les variables peu utiles ; le gain d'une sélection est la simplicité et l'interprétabilité, pas la performance.
- **Meilleur compromis** : SHAP (95 % de l'importance cumulée) garde 81 variables pour 0,3876 (+0,002 vs référence) ; LightGBM-gain en garde 39 pour 0,3912 (+0,006). La permutation (143 variables, 0,3858) et RFECV (131, 0,3875) restent aussi proches.
- **Les filtres univariés sont les moins bons à k = 60** : corrélation 0,4293, ANOVA 0,4318, et surtout chi² 0,4796 (proche du hasard ; explication probable, non testée : le chi² sur variables min-max et tercile de prix favorise des indicatrices peu liées au prix). L'information mutuelle (0,3966) est nettement meilleure car elle capte les relations non linéaires (latitude, longitude, encodage du quartier). Un filtre univarié ignore les interactions et la redondance.
- **Wrappers** : RFECV (Ridge) garde 131 variables, performance équivalente à la référence. Le SFS avant, limité à 25 variables parmi 80 pré-filtrées et guidé par un modèle linéaire, est mauvais pour LightGBM (0,4313) et coûteux (87 s contre < 3 s pour les filtres) : la sélection guidée par Ridge ne retient pas les variables utiles à un modèle non linéaire.
- **Embedded linéaires** : Lasso (226) et Elastic Net (227) ne suppriment presque rien, car les 259 colonnes portent toutes un peu d'information ; pour Ridge, la référence reste la meilleure (0,4288).
- **VIF (seuil 10)** : ne retire que 23 colonnes sur 259, sans effet sur l'erreur (0,3879) : la multicolinéarité est faible une fois les colonnes parfaitement redondantes retirées, et elle gêne l'interprétation des coefficients linéaires, pas la prédiction des arbres.
- **PCA (transformation)** : 0,4507 avec 191 composantes (95 % de variance) et 0,4570 avec 60, nettement pire que la sélection. Hypothèse (non testée) : les axes de variance maximale ne sont pas ceux qui expliquent le prix ; la PCA détruit en outre l'interprétabilité. À réserver à la visualisation (phase 8).

## Stabilité (Jaccard moyen sur les 10 paires de plis)
Filtres univariés : 0,84 à 0,97 (très stables). Embedded : 0,82 à 0,86. VIF : 0,96. Wrappers et permutation : 0,53 à 0,61, moins stables (RFECV 0,59, SFS 0,53, permutation 0,61), car ils dépendent de petits écarts de performance entre sous-ensembles et du bruit de l'échantillon de validation interne. Un sous-ensemble stable n'est pas pour autant meilleur (chi² : Jaccard 0,95 mais mauvaise erreur).

## Variables retenues par plusieurs méthodes
Consensus des 7 méthodes sélectives (≤ 100 variables, hors chi²) : retenues par les 7 méthodes dans au moins 3 plis sur 5 : `accommodates`, `bedrooms`, `bath_count`, `bath_per_person`, `bath_shared`, `nbhd_key` (quartier encodé), `am_dishwasher`, `am_lock_on_bedroom_door`, `review_scores_location`, `city_paris`, `city_mexico-city`. Les rangs d'importance (SHAP / permutation) placent en tête : `nbhd_key` (rang 1 pour toutes les mesures), `accommodates`, `quote_nights`, `bedrooms`, `bath_count`, `dist_center_km`, `latitude`, `log_quote_lead_days`.
Les variables du contexte du devis (`quote_nights`, `log_quote_lead_days`) sont parmi les plus importantes : le prix observé dépend beaucoup de la durée de séjour du devis (remises mensuelles).

## Divergences : biais des importances d'arbres
Rang d'importance selon la mesure (1 = plus important) :

| Variable | RF impureté | LightGBM gain | Permutation | SHAP |
|---|---:|---:|---:|---:|
| desc_len | 29 | 44 | 70 | 65 |
| desc_words | 31 | 45 | 78 | 67 |
| name_len | 30 | 46 | 66 | 44 |
| host_about_len | 34 | 40 | 68 | 36 |
| am_air_conditioning | 43 | 31 | 20 | 20 |
| quote_nights | 9 | 5 | 2 | 3 |

Les variables continues à forte cardinalité (longueurs de textes) sont **surévaluées par l'impureté RF** (rang ≈ 30) par rapport à la permutation (rang ≈ 70) : un arbre a beaucoup de seuils candidats pour ces variables et peut y ajuster du bruit. À l'inverse, des variables binaires utiles (climatisation) sont sous-évaluées par l'impureté (rang 43) contre 20 en permutation et SHAP. La permutation et SHAP, calculées hors échantillon d'entraînement (permutation sur un split interne), sont plus fiables. Le quartier encodé (`nbhd_key`, forte cardinalité mais encodé par target encoding avec CV interne) reste premier pour toutes les mesures : la cohérence entre les mesures indique que ce n'est pas un artefact de cardinalité, mais cet encodage reste une variable puissante à surveiller en généralisation à une ville inconnue (phase 5).
La permutation répartit l'importance entre variables corrélées (latitude / longitude / distance au centre) : chacune paraît moins importante prise isolément.

## Décisions pour la suite
- **Jeu retenu pour la régression** : `core+quote+quality`. Les 81 variables SHAP (liste dans `models/selected_features.json`, clé `shap_95`) servent de jeu réduit pour les modèles linéaires, l'interprétation et les outils métier ; les modèles d'arbres de la phase 6 peuvent aussi utiliser toutes les variables.
- Les variables d'activité (`activity`) restent exclues (conséquences du prix).
- Phase 5 : tester sans `city_*` et sans `nbhd_key` pour mesurer l'effet ville.
- Limites : un échantillon de 60 000 annonces (12 % du train) ; une graine ; hyperparamètres fixes (200 arbres) ; k = 60 non optimisé pour les filtres ; SFS limité à 25 variables pré-filtrées ; RFECV et SFS utilisent Ridge en interne.
