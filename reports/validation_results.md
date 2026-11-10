# Phase 5 : résultats de la comparaison des schémas de validation

Régression de log(price_eur), LightGBM (300 arbres), jeu `core+quote+quality`, 309 412 annonces du train (jeu de régression). Prétraitement ajusté dans chaque pli. Test final non utilisé.

Variantes : V1 toutes les variables ; V2 sans `city_*` ni `nbhd_key` ; V3 = V2 sans latitude/longitude absolues (distance au centre conservée).

## Synthèse

| schéma                                                      | variante                        |   RMSE |    MAE |     r2 |   MAE (EUR, exp simple) |      n |
|:------------------------------------------------------------|:--------------------------------|-------:|-------:|-------:|------------------------:|-------:|
| A intra-villes (5 plis, hôtes groupés)                      | V1 toutes                       | 0.3655 | 0.2578 | 0.7959 |                 67.7516 | 309412 |
| A intra-villes (5 plis, hôtes groupés)                      | V2 sans ville/quartier          | 0.3681 | 0.26   | 0.793  |                 68.4251 | 309412 |
| A intra-villes (5 plis, hôtes groupés)                      | V3 agnostique ville             | 0.4191 | 0.3056 | 0.7318 |                 78.0687 | 309412 |
| B leave-one-city-out                                        | V1 toutes                       | 0.5828 | 0.445  | 0.4812 |                104.675  | 309412 |
| B leave-one-city-out                                        | V2 sans ville/quartier          | 0.6757 | 0.5299 | 0.3027 |                117.879  | 309412 |
| B leave-one-city-out                                        | V3 agnostique ville             | 0.5936 | 0.4532 | 0.4619 |                105.912  | 309412 |
| C contrôle : hôtes disjoints, même période (check-in sept.) | toutes les variables            | 0.3918 | 0.2785 | 0.7789 |                 79.1264 |  82959 |
| C temporel : hôtes disjoints, check-in >= oct. 2026         | toutes les variables            | 0.3914 | 0.2889 | 0.7332 |                 60.594  |  72854 |
| C contrôle : hôtes disjoints, même période (check-in sept.) | sans variables de date du devis | 0.4004 | 0.2852 | 0.7691 |                 80.9085 |  82959 |
| C temporel : hôtes disjoints, check-in >= oct. 2026         | sans variables de date du devis | 0.4186 | 0.3151 | 0.6949 |                 65.1246 |  72854 |

## Écart A / B : l'effet ville

- RMSE pondéré : A = 0.366, B = 0.583 (V1).
- Décomposition de l'erreur quadratique de B : biais de niveau par ville (prédit - réel moyen) = 0.148 ; erreur résiduelle une fois le niveau corrigé = 0.191 ; B = 0.340 ≈ somme. Écart B - A = 0.206, dont **72 % dû au décalage de niveau de prix** et 28 % à la perte de qualité du classement intra-ville.
- Une recalibration du niveau avec 200 annonces de la nouvelle ville ramène le RMSE pondéré de 0.583 à 0.439 (oracle avec toute la ville : 0.438).
- Corrélation entre le biais d'une ville et son niveau de prix (log médian EUR) : -0.87.

## Par ville (V1)

| city          |   annonces |   RMSE A |   RMSE B |   biais B |   RMSE B recalibré (200) |   corrélation prédit/réel dans la ville (B) |   prix médian EUR |
|:--------------|-----------:|---------:|---------:|----------:|-------------------------:|--------------------------------------------:|------------------:|
| barcelona     |      10704 |    0.351 |    0.366 |    -0.012 |                    0.367 |                                       0.903 |           196.415 |
| rome          |      28867 |    0.327 |    0.405 |    -0.05  |                    0.402 |                                       0.722 |           188.7   |
| madrid        |      15959 |    0.366 |    0.418 |    -0.015 |                    0.418 |                                       0.835 |           153     |
| lisbon        |      18668 |    0.343 |    0.419 |     0.201 |                    0.368 |                                       0.818 |           140     |
| berlin        |       6954 |    0.348 |    0.472 |     0.239 |                    0.407 |                                       0.795 |           125     |
| sydney        |      15097 |    0.342 |    0.515 |    -0.293 |                    0.424 |                                       0.839 |           209.415 |
| london        |      51053 |    0.369 |    0.532 |    -0.287 |                    0.449 |                                       0.82  |           203.507 |
| amsterdam     |       5178 |    0.32  |    0.535 |    -0.393 |                    0.374 |                                       0.767 |           284     |
| lyon          |       4348 |    0.314 |    0.54  |     0.421 |                    0.341 |                                       0.807 |           108.735 |
| tokyo         |      28623 |    0.412 |    0.559 |     0.251 |                    0.501 |                                       0.622 |           113.523 |
| los-angeles   |      31251 |    0.361 |    0.562 |    -0.325 |                    0.462 |                                       0.864 |           185.707 |
| toronto       |      14345 |    0.351 |    0.576 |     0.428 |                    0.386 |                                       0.894 |           101.666 |
| paris         |      37069 |    0.361 |    0.585 |    -0.369 |                    0.454 |                                       0.815 |           216.87  |
| new-york-city |      17111 |    0.376 |    0.791 |    -0.625 |                    0.487 |                                       0.844 |           158.714 |
| mexico-city   |      24185 |    0.413 |    0.978 |     0.865 |                    0.459 |                                       0.777 |            75.377 |
