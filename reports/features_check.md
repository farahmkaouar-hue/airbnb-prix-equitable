# Contrôles des features (Phase 3, train uniquement)

## 1. Target encoding du quartier (nbhd_key)

- corrélation encodage naïf (moyenne du quartier, en échantillon) / cible, train : **0.553** (optimiste : l'annonce se voit elle-même)
- corrélation encodage avec CV interne / cible, train : **0.545**
- corrélation encodage appliqué à la validation (hôtes jamais vus) / cible : **0.548**
- le TargetEncoder à CV interne (cross-fitting) donne sur le train une corrélation proche de celle observée en validation : pas de fuite visible.

Forme de la matrice de features `core` : 237 colonnes (dont 12 indicateurs de manquant).

- modalités inconnues (nouveau quartier, nouveau type de propriété) : gérées sans erreur (encodage = moyenne globale / catégorie 'infrequent').

## 2. Pipelines pour les 3 cibles

- is_superhost (binary) : OK, 237 colonnes, accuracy en échantillon 0.754 (non évaluée : simple test de fonctionnement)
- price_tier (multiclass) : OK, 237 colonnes, accuracy en échantillon 0.701 (non évaluée : simple test de fonctionnement)
- régression (log_price_eur) : OK, voir ci-dessous.

## 3. Comparaison indicative des jeux de variables (cible log_price_eur)

Validation : 63,167 annonces (20 % des hôtes du train, jeu in_reg_set) ; entraînement : 246,245. Référence « médiane globale » : RMSE 0.8015.

| Jeu de variables   |   nb variables |   Ridge RMSE |   Ridge MAE |   LightGBM RMSE |   LightGBM MAE |   temps (s) |
|:-------------------|---------------:|-------------:|------------:|----------------:|---------------:|------------:|
| core               |            198 |       0.4407 |      0.3189 |          0.3963 |         0.2837 |     13.6613 |
| core+quote         |            204 |       0.4328 |      0.3122 |          0.3739 |         0.2662 |     13.4624 |
| core+quote+quality |            212 |       0.4274 |      0.3078 |          0.3657 |         0.2589 |     14.1519 |
| full               |            225 |       0.417  |      0.2997 |          0.3538 |         0.2497 |     15.807  |

Ces chiffres sont indicatifs (un seul découpage, hyperparamètres par défaut) ; la comparaison rigoureuse en CV est faite en phases 4 à 6. `full` contient des variables d'activité (disponibilité, avis) qui sont en partie des conséquences du prix.
