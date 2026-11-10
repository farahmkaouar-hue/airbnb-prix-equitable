# Le juste prix d'un logement Airbnb : rapport final

*Projet de machine learning sur les données Inside Airbnb (15 villes, septembre 2026). Notebook complet : `notebooks/00_projet_complet.ipynb`. Modèle final : `models/final_model.joblib` ; exemple : `python predict.py`.*

---

## 1. Résumé exécutif

**Question.** Comment aider un hôte ou une plateforme à estimer le prix juste d'un logement et à identifier les facteurs de qualité, de manière généralisable à plusieurs villes ?

**Données.** 485 128 annonces dans 15 villes, 21,7 millions d'avis, 15 calendriers de 365 jours. Le prix est un **devis pour un séjour précis** (1 à 31 nuits), manquant pour 25 % des annonces.

**Résultats principaux (jeu de test final, hôtes jamais vus, ouvert une seule fois).**
- **Prix** : RMSE de 0,354 sur log(prix), **R² de 0,815**, erreur absolue moyenne de **68 €** par nuit (médiane 27 €), **53 % des annonces à ±20 %** du prix prédit. Résultats cohérents avec la validation croisée (RMSE 0,356).
- **Fourchette de prix** à 90 % (intervalle conforme adaptatif) : **90,4 % de couverture** sur le test, largeur moyenne 245 €.
- **Superhost** : ROC-AUC 0,829 avec les seules caractéristiques du logement et de l'hôte, 0,886 avec les notes (qui recopient en partie les critères d'attribution). **Tranche de prix** (bas / moyen / haut dans la ville) : accuracy 0,756.
- **Généralisation à une ville inconnue** : l'effet ville est massif. Le RMSE passe de 0,366 (ville connue) à 0,583 (ville jamais vue). **72 % de l'écart vient d'un décalage de niveau de prix**, corrigeable : avec 25 annonces de la nouvelle ville le RMSE moyen par ville passe de 0,555 à 0,419, et recalibrer niveau et largeur de l'intervalle sur 100 à 200 annonces ramène la couverture à 90 %.
- **Boosting contre réseau de neurones** : LightGBM bat le réseau dense dans chacun des 5 plis (RMSE 0,3674 contre 0,3965) pour un réglage 4 fois plus court.

**Ce que le modèle ne sait pas.** Il sous-estime les logements très chers (la fourchette ne contient le prix réel que pour 68 % des annonces dont le prix réel dépasse 600 €), il n'est pas exploitable dans une ville nouvelle sans recalibration, et ses « leviers » (équipements, caractéristiques) sont des associations, pas des causes.

**Cinq actions recommandées**, chacune reliée à un résultat chiffré (voir section 10) : publier la fourchette et pas seulement le prix ; ajuster selon la durée de séjour (−35 % de prix par nuit entre 1 et 90 nuits) et les jours demandés ; signaler les annonces très écartées du prix attendu et peu réservées ; cibler la régulation sur la concentration des hôtes ; ne jamais déployer dans une nouvelle ville sans recalibrer.

---

## 2. Contexte et problématique
Parties prenantes : **hôtes** (fixer le bon prix), **plateforme** (qualité, confiance, annonces fantômes), **investisseurs et villes** (marché, saisonnalité, régulation). Le projet suit des règles strictes : graine 42, versions figées (`requirements.txt`), une seule configuration (`config.yaml`), **aucune fuite de données** (tout ce qui apprend est ajusté sur le train, dans des Pipelines, y compris dans la validation croisée), test mis sous scellés dès le nettoyage.

## 3. Données
15 villes (Amsterdam, Barcelone, Berlin, Lisbonne, Londres, Los Angeles, Lyon, Madrid, Mexico, New York, Paris, Rome, Sydney, Tokyo, Toronto), une collecte par ville (14 au 25 septembre 2026). 13 colonnes sont vides dans toutes les villes. Le prix manque pour 25 % des annonces (de 6,6 % à Tokyo à 44,9 % à Lyon) ; ce manque est lié à l'indisponibilité (77 % des annonces sans prix sont indisponibles toute l'année contre 5 % parmi celles avec prix) : **biais de sélection vers les annonces réservables**. Les prix sont convertis en euros avec les **taux de référence de la BCE** de la date de collecte de chaque ville (`data/external/ecb_fx_2026-09.csv`).

## 4. Analyse exploratoire
Le log(prix) est quasi symétrique (asymétrie 0,6 contre 57 en brut à Paris). Facteurs principaux : capacité du logement, quartier (à Paris, de 146 € à 339 € de prix médian selon l'arrondissement), durée du séjour du devis. 38 % des annonces appartiennent à un hôte avec 5 annonces ou plus (de 7 % à Amsterdam à 74 % à Tokyo) ; 15 % sont des annonces fantômes probables. Pièges de fuite repérés : `estimated_revenue_l365d` vaut exactement prix × occupation ; `price_quote_*` copie le prix ; disponibilités et avis récents sont des conséquences du prix. Détail : `reports/eda_insights.md`.

## 5. Nettoyage
Aucune ligne supprimée : les annonces douteuses sont signalées (651 prix extrêmes sur la règle 3 × IQR de log(prix) par ville, 1 548 annonces incohérentes, 24 729 quasi-doublons) et les modèles filtrent. 164 équipements normalisés conservés (≥ 1 % des annonces du train). Impact de la règle d'outliers mesuré : RMSE 0,4654 → 0,4642. Journal complet : `reports/cleaning_log.md`.

## 6. Variables
275 colonnes (`reports/features_table.md`) : ratios, distance au centre, ancienneté de l'hôte, longueur des textes et **contexte du devis** (durée, délai, saison), qui fait passer le RMSE de LightGBM de 0,396 à 0,374. Quartier encodé par target encoding à validation croisée interne (corrélation 0,545 sur le train contre 0,548 sur des hôtes jamais vus : pas de fuite visible).

## 7. Sélection de variables
Seize méthodes comparées (filtres, wrappers, embedded, permutation, SHAP, VIF, PCA) en CV groupée par hôte. **Aucune ne bat « toutes les variables »** (RMSE 0,3854) ; SHAP garde 81 variables sur 259 pour 0,3876. L'importance d'impureté des arbres surévalue les variables continues à forte cardinalité (rang 29 contre 70 en permutation pour la longueur de description).

## 8. Validation
Trois schémas : A (5 plis groupés par hôte, stratifiés par ville), B (leave-one-city-out), C (temporel sur la date de séjour du devis). RMSE : A 0,366 ; B 0,583 ; sans identifiants de ville (V3) 0,594. **Latitude et longitude absolues sont des identifiants de ville déguisés.**

## 9. Modélisation : tableau comparatif unique

### 9.1 Régression du prix (cible : log(prix en EUR))
| Modèle | Validation croisée | RMSE (log) | MAE (log) | R² | MAE (EUR) | TEST FINAL (une seule ouverture) |
|---|---|---:|---:|---:|---:|---|
| Médiane de la ville | 80 000 annonces, 5 plis A | 0,747 ± 0,018 | 0,576 | 0,154 | 131 | — |
| Régression linéaire | 80 000 annonces, 5 plis A | 0,427 ± 0,006 | 0,309 | 0,723 | 80 | — |
| Ridge | 80 000 annonces, 5 plis A | 0,428 ± 0,006 | 0,309 | 0,722 | 80 | — |
| Lasso | 80 000 annonces, 5 plis A | 0,428 ± 0,007 | 0,309 | 0,722 | 80 | — |
| ElasticNet | 80 000 annonces, 5 plis A | 0,428 ± 0,006 | 0,309 | 0,722 | 80 | — |
| kNN (k=15) | 80 000 annonces, 5 plis A | 0,518 ± 0,010 | 0,388 | 0,592 | 98 | — |
| Arbre de décision | 80 000 annonces, 5 plis A | 0,472 ± 0,011 | 0,344 | 0,661 | 89 | — |
| Random Forest | 80 000 annonces, 5 plis A | 0,401 ± 0,011 | 0,286 | 0,755 | 75 | — |
| Gradient Boosting (sklearn) | 80 000 annonces, 5 plis A | 0,386 ± 0,009 | 0,275 | 0,774 | 72 | — |
| HistGradientBoosting | 80 000 annonces, 5 plis A | 0,380 ± 0,010 | 0,270 | 0,781 | 71 | — |
| XGBoost | 80 000 annonces, 5 plis A | 0,374 ± 0,009 | 0,265 | 0,787 | 70 | — |
| LightGBM | 80 000 annonces, 5 plis A | 0,375 ± 0,009 | 0,266 | 0,786 | 70 | — |
| Stacking (Ridge sur 5 modèles) | 80 000 annonces, 5 plis A | 0,371 ± 0,009 | 0,262 | 0,791 | 70 | — |
| Réseau dense (embeddings) réglé | 80 000 annonces, 5 plis A | 0,396 | 0,286 | 0,761 | 74 | — |
| LightGBM réglé (même échantillon) | 80 000 annonces, 5 plis A | 0,367 | 0,259 | 0,795 | 69 | — |
| LightGBM par défaut | 309 412 annonces, 5 plis A | 0,365 | — | — | — | — |
| **LightGBM réglé (modèle final)** | 309 412 annonces, 5 plis A | **0,356** | 0,249 | 0,806 | 66 | **RMSE 0,354 · MAE 0,249 · R² 0,815 · MAE 68 €** |

*Le jeu de test n'est utilisé que pour le modèle final retenu, une seule fois. Les autres modèles ne sont comparés qu'en validation croisée.* Le réglage (Optuna, 40 essais LightGBM, 25 XGBoost) ramène le RMSE de 0,365 à 0,356. L'empilement de la course gagne environ 0,004 sur LightGBM par défaut (échantillon de 80 000 annonces), mais n'a pas été reproduit sur le train complet réglé. La correction de biais du `exp()` (smearing, facteur 1,089) dégrade la MAE en euros (la MAE est minimisée par la médiane) : le prix recommandé est l'estimation médiane.

### 9.2 Classification
| Tâche | Modèle | Validation croisée (60 000 annonces) | Test final |
|---|---|---|---|
| Superhost, caractéristiques seules (jeu A) | LightGBM | ROC-AUC 0,817, PR-AUC 0,637, F1 0,561 | ROC-AUC 0,829, PR-AUC 0,661, F1 0,636 (seuil 0,28) |
| Superhost, avec les notes (jeu B) | LightGBM | ROC-AUC 0,875, PR-AUC 0,749, F1 0,666 | ROC-AUC 0,886, PR-AUC 0,771, F1 0,704 (seuil 0,33) |
| Superhost, avec notes et activité des avis (jeu C, fuite) | LightGBM | ROC-AUC 0,902 | non retenu |
| Superhost (jeu A) | Régression logistique | ROC-AUC 0,778 ± 0,008 | — |
| Superhost (jeu A) | kNN (k=25) | ROC-AUC 0,749 ± 0,008 | — |
| Superhost (jeu A) | SVM (RBF, 12 000 ex.) | ROC-AUC 0,772 ± 0,007 | — |
| Superhost (jeu A) | Arbre de décision | ROC-AUC 0,758 ± 0,009 | — |
| Superhost (jeu A) | Random Forest | ROC-AUC 0,800 ± 0,010 | — |
| Superhost (jeu A) | HistGradientBoosting | ROC-AUC 0,813 ± 0,003 | — |
| Superhost (jeu A) | LightGBM | ROC-AUC 0,817 ± 0,003 | — |
| Tranche de prix | Régression logistique | accuracy 0,704, F1 macro 0,705, AUC 0,869 | — |
| Tranche de prix | kNN (k=25) | accuracy 0,613, F1 macro 0,617, AUC 0,798 | — |
| Tranche de prix | SVM (RBF, 12 000 ex.) | accuracy 0,697, F1 macro 0,698, AUC 0,864 | — |
| Tranche de prix | Arbre de décision | accuracy 0,658, F1 macro 0,661, AUC 0,831 | — |
| Tranche de prix | Random Forest | accuracy 0,704, F1 macro 0,705, AUC 0,872 | — |
| Tranche de prix | HistGradientBoosting | accuracy 0,732, F1 macro 0,733, AUC 0,890 | — |
| Tranche de prix | LightGBM | accuracy 0,736, F1 macro 0,737, AUC 0,894 | accuracy 0,756, F1 macro 0,756, AUC 0,907 |

### 9.3 Apprentissage non supervisé
Les annonces forment un continuum plus que des familles (silhouette de 0,12 à k = 5 ; coude progressif). Cinq profils sont retenus pour leur lisibilité (grande capacité et prix élevés ; chambres chez l'habitant ; deux groupes de logements entiers ; un petit groupe de 2 % d'annonces mal notées d'hôtes multi-annonces). La PCA : deux composantes expliquent 38 % de la variance (taille et standing ; maturité de l'annonce), il en faut 7 pour 80 %. DBSCAN trouve 48 zones denses à Paris (une zone centrale géante par chaînage). L'Isolation Forest signale 1 % d'annonces atypiques : de très grandes propriétés d'opérateurs professionnels.

### 9.4 NLP sur les avis (échantillon : 8,100 annonces, 64,528 avis)
6.2 % d'avis négatifs (biais de positivité) ; sentiment multilingue (`lxyuan/distilbert-base-multilingual-cased-sentiments-student`) corrélé à la note (Spearman 0,50). Premier reproche : la **propreté** (34 % des avis négatifs), puis le bruit et les équipements en panne. **Valeur ajoutée du texte** : nulle sur le prix (RMSE 0,3387 → 0,3382) ; sur le Superhost sans les notes, l'AUC passe de 0,735 à 0,774 ; avec les notes, aucun gain (le texte recopie la note).

### 9.5 Séries temporelles (calendar, sans prix)
Série étudiée : part de nuits indisponibles (annonces actives), instantané de 365 jours. Courbe en U selon l'horizon, week-ends plus demandés (jusqu'à +3,4 points à Madrid ; jeudi à Paris), pics autour du 29-31 décembre dans 8 villes sur 15. **Aucun modèle ne bat la baseline saisonnière naïve** à 28 jours (MAE 1,56 point ; LightGBM 1,66, ETS 2,12, SARIMA 2,31).

### 9.6 Deep learning
Réseau dense à embeddings (PyTorch), réglé par Optuna (20 essais, 82 min) : RMSE 0,3965 contre 0,3674 pour LightGBM (40 essais, 20 min), à l'avantage du boosting dans les 5 plis, les 15 villes et les 10 déciles de prix.

## 10. Apport métier
**Hôtes.** Équipements associés à la plus forte hausse du prix prédit : piscine (+7.9 %), lave-vaisselle (+7.4 %), cheminée (+6.8 %), climatisation (+6.1 %) ; ce sont des **associations**, pas des effets garantis (le verrou de chambre est associé à -7.3 % : il signale des chambres chez l'habitant).
**Plateforme.** 15 % d'annonces fantômes probables (jusqu'à 33 % à Lyon) ; les annonces sur-évaluées (>+65 % du prix attendu) sont disponibles 247 jours par an contre 215 ; premier reproche des voyageurs : la propreté.
**Investisseurs et villes.** 160 quartiers sur 693 ont un écart significatif entre prix observé et attendu (pistes à vérifier, pas des recommandations d'achat). Concentration : de 7 % (Amsterdam) à 74 % (Tokyo) d'annonces d'hôtes à ≥ 5 annonces ; licences non renseignées à Londres et Mexico.
**Ville inconnue.** Sans recalibration : couverture de la fourchette de 65 % ; avec niveau seul (200 annonces) 83 % ; avec niveau et largeur, 90 %.

**Recommandations** : (1) publier la fourchette à 90 % ; (2) ajuster selon la durée de séjour et les jours demandés ; (3) signaler les annonces très écartées du prix attendu et peu réservées ; (4) cibler la régulation sur la concentration et renseigner les licences ; (5) recalibrer niveau (25 à 100 annonces) puis largeur (100 à 200) avant tout déploiement dans une nouvelle ville.

## 11. Limites et éthique
Prix affiché et non payé ; aucune donnée de réservation ; biais de sélection (25 % sans prix) ; une collecte par ville ; calendar sans prix ; variables absentes (photos, vue, surface). Régression vers la moyenne (surestimation des logements bon marché, sous-estimation des plus chers). Associations et non causalité. RGPD : nom des hôtes retiré, licence réduite à un indicateur. Risques : conversion de logements en meublés touristiques, homogénéisation des prix et coordination algorithmique ; l'outil doit rester une aide à la décision avec fourchette.

## 12. Pistes d'amélioration
Plusieurs collectes pour mesurer la dérive et la saisonnalité ; données de réservation ; photos et surface ; modèle par ville ou hiérarchique ; HDBSCAN pour les zones denses ; recherche d'hyperparamètres plus longue ; analyse causale (expériences de prix).

## Annexes
- **Figures clés** (dans `reports/figures/`) : 1 `eda_06_map_paris.png` (carte des prix à Paris) ; 2 `eda_02_price_paris_raw_vs_log.png` (log du prix) ; 3 `sel_01_rmse_vs_nfeatures.png` (sélection de variables) ; 4 `sel_03_importance_ranks.png` (biais des importances) ; 5 `val_03_schemes.png` (schémas de validation) ; 6 `val_02_bias_vs_price_level.png` (biais par ville inconnue) ; 7 `clu_06_umap.png` (UMAP) ; 8 `clu_02_profiles.png` (profils de clusters) ; 9 `ts_03_weekday_profile.png` (profil hebdomadaire) ; 10 `ts_05_forecast_scores.png` (prévision). Les autres graphiques sont dans le notebook.
- **Checklist de reproductibilité** : graine 42 ; `requirements.txt` figé ; `config.yaml` ; split fixé dans `data/processed/split.parquet` ; test ouvert une seule fois (`src/final_model.py` refuse un second passage) ; `python run_all.py` ; chaque étape lourde est mise en cache.
- **Décisions à justifier à l'oral** : voir le dernier chapitre du notebook (12 décisions).
