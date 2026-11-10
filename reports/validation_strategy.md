# Stratégie de validation (définie en Phase 0, avant toute exploration du test)

Contraintes constatées dans les données : **une seule date de collecte par ville** (14→25 sept. 2026) et **aucun fichier calendar** téléchargé.
Le `price` est un devis (price quote) pour des dates précises (de sept. 2026 à mai 2027 selon l'annonce), absent pour ~25 % des annonces.

## Jeu de test final (utilisé une seule fois, Phase 14)
- 15 % des annonces **à prix connu**, tirage fixé une fois (random_state=42), identifiants sauvegardés dans `data/processed/test_ids.parquet`.
- **Groupé par `host_id`** (StratifiedGroupKFold, stratifié par ville) : un même hôte a souvent plusieurs annonces quasi identiques (description, photos, équipements). Un split par annonce ferait fuiter l'hôte entre train et test et gonflerait les scores.
- Aucun nettoyage ni sélection ne regarde ce jeu ; les transformations sont apprises sur le train.

## Schémas de CV (sur le train uniquement)
| Schéma | Implémentation | Mesure |
|---|---|---|
| A | StratifiedGroupKFold(k=5), strate = ville, groupe = host_id | performance « intra-villes » |
| B | Leave-One-City-Out (15 plis, groupe = city) | généralisation à une ville jamais vue |
| C | Temporel : **non réalisable** (1 date/ville) | voir ci-dessous |

Les écarts A vs B mesurent l'« effet ville » (niveau de prix, devise, quartiers inconnus). En B, les encodages par quartier/ville sont inutilisables : on comparera donc aussi des variantes sans identifiants de ville.

## Schéma C et Phase 10
Sans calendar, la phase 10 (séries temporelles) est impossible. Le schéma C temporel pourrait être remplacé par un split sur le calendar (train = jours anciens, test = jours récents), mais cela évalue la prédiction de prix par jour, pas par annonce. Les URLs calendar existent dans `data/raw/index.csv` (non téléchargées ; volume potentiellement de l'ordre de 100 M+ de lignes pour 15 villes → échantillon recommandé).

## Règles anti-fuite
- Tout transformeur (imputation, scaling, encodage, sélection, PCA, TF-IDF) dans un Pipeline, ajusté dans chaque pli sur le train du pli.
- Cible de classification « tranche de prix » : terciles calculés par ville **sur le train** uniquement.
- Colonnes à risque listées en Phase 1 (ex. `estimated_revenue_l365d`, `price_quote_*`, `review_scores_*` pour Superhost).

---
# Résultats de la phase 5 (détails : `validation_results.md`, figures `val_*.png`)

Schéma C réalisé sous forme de variante : entraînement sur les hôtes du groupe 1 avec check-in du devis avant le 1er octobre 2026 (81 940 annonces) ; validation sur des hôtes disjoints (groupe 2) avec check-in à partir du 1er octobre 2026 (72 854) ; contrôle sur les mêmes hôtes disjoints avec check-in en septembre (82 959). Ce n'est pas un split temporel de séries de prix (aucune annonce n'est observée à deux dates) : il teste la généralisation à d'autres dates de séjour.

| Schéma | Variante | RMSE log(prix) | R² | MAE (EUR, exp simple) |
|---|---|---:|---:|---:|
| A intra-villes (5 plis) | V1 toutes | 0,366 | 0,796 | 68 |
| A | V3 agnostique ville | 0,419 | 0,732 | 78 |
| B ville inconnue | V1 toutes | 0,583 | 0,481 | 105 |
| B | V2 sans ville/quartier | 0,676 | 0,303 | 118 |
| B | V3 agnostique ville | 0,594 | 0,462 | 106 |
| B | V1 + recalibration (200 annonces) | 0,439 | — | — |
| C contrôle (sept., hôtes disjoints) | toutes | 0,392 | 0,779 | 79 |
| C temporel (check-in >= oct.) | toutes | 0,391 | 0,733 | 61 |
| C temporel | sans variables de date du devis | 0,419 | 0,695 | 65 |

## Effet ville (A contre B)
- Passer d'une ville connue (A) à une ville jamais vue (B) fait passer le RMSE de 0,366 à 0,583, le R² de 0,80 à 0,48.
- **72 % de l'écart quadratique vient d'un décalage de niveau de prix par ville** (biais moyen pondéré 0,385 en log), 28 % d'une moins bonne hiérarchie des prix à l'intérieur de la ville. Une recalibration du niveau avec seulement 200 annonces de la nouvelle ville ramène le RMSE à 0,439 (l'oracle avec toute la ville donne 0,438) : le modèle classe bien les annonces d'une ville inconnue (corrélation prédit/réel de 0,62 à Tokyo à 0,90 à Barcelone) mais pas leur niveau.
- Le biais est corrélé au niveau de prix de la ville (−0,87) : le modèle tire les prédictions vers la moyenne des autres villes (Mexico surestimé de +0,87 en log, soit un facteur de 2,4 ; New York et Amsterdam sous-estimés de −0,63 et −0,39).
- Sans identifiants de ville ni de quartier (V2), la généralisation se dégrade (0,676) : latitude et longitude absolues deviennent un identifiant déguisé de la ville, sans lien avec le niveau de prix d'une ville nouvelle. Retirer aussi latitude/longitude (V3) revient à 0,594, presque au niveau de V1 (0,583). Pour une ville inconnue, il faut donc des variables géographiques **relatives** (distance au centre) ; en contrepartie, la performance intra-ville baisse (0,419 contre 0,366).
- Pour la comparaison de modèles (phase 6), les deux schémas seront rapportés : A pour l'usage « ville connue », B pour l'usage « nouvelle ville ».

## Schéma C
- Avec toutes les variables, la validation tardive (0,391) est aussi bonne que le contrôle (0,392) : à composition comparable, le modèle généralise aux séjours d'octobre. Mais le RMSE par mois monte de 0,34 (octobre) à environ 0,45 à partir de novembre, avec une surestimation moyenne de +0,05 à +0,16 en log. L'agrégat est dominé par octobre (57 % de la validation tardive).
- Sans les variables de date du devis, l'erreur tardive passe à 0,419 : la date du séjour aide, mais ces variables ne peuvent pas extrapoler à des saisons jamais vues en entraînement (septembre seulement).
- Limite : les séjours lointains (check-in en 2027) concernent des annonces de composition différente (sélection par disponibilité, séjours mensuels) ; l'écart de RMSE n'est donc pas uniquement de la saisonnalité. Un modèle de saisonnalité demande le calendar (phase 10).
