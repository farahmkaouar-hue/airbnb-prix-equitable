# Prix juste d'un logement Airbnb

Comment estimer le prix juste d'un logement et identifier les facteurs de qualité, de façon généralisable à plusieurs villes ?

Données : Inside Airbnb, **485 128 annonces dans 15 villes**, 21,7 millions d'avis et les calendriers de disponibilité.

## Résultats

- **Prix** : R² de **0,815** sur le logarithme du prix, erreur absolue moyenne de 68 € par nuit (médiane 27 €). 53 % des annonces sont à ±20 % du prix prédit.
- **Fourchette de prix à 90 %** (intervalle conforme) : **90,4 % de couverture** sur l'échantillon de test.
- **Ville inconnue** : l'erreur augmente fortement (RMSE de 0,366 à 0,583 sur le logarithme du prix). 72 % de cet écart vient d'un simple décalage du niveau de prix de la ville.
- **LightGBM** fait mieux qu'un réseau de neurones dense dans chacun des 5 plis de validation, pour un temps de réglage 4 fois plus court.

## Limites

Le modèle sous-estime les logements très chers : au-delà de 600 € la nuit, la fourchette ne contient le prix réel que dans 68 % des cas.

## Contenu

| Dossier | Contenu |
|---|---|
| `notebooks/` | `00_projet_complet.ipynb` (le projet entier), puis exploration, nettoyage, variables et sélection |
| `src/` | Code Python : nettoyage, variables, modèles, NLP sur les avis, clustering, intervalles de prédiction |
| `reports/` | Rapport final |
| `Projet_prix_Airbnb_rapport_detaille.pdf` | Rapport détaillé |

Les données brutes (plusieurs Go) ne sont pas incluses. Elles sont téléchargeables sur [Inside Airbnb](https://insideairbnb.com).

Technologies : Python, pandas, LightGBM, PyTorch, scikit-learn.
