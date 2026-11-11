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

| Fichier | Contenu |
|---|---|
| `notebooks/00_projet_complet.ipynb` | Le projet entier : audit, nettoyage, variables, sélection, validation, modèles, NLP sur les avis, interprétation |
| `src/` | Le code de chaque étape, lancé dans l'ordre par `run_all.py` |
| `config.yaml` | Paramètres du projet : graine, chemins, règles de nettoyage, schémas de validation |
| `download_airbnb.py`, `merge_airbnb.py` | Téléchargement et fusion des données des 15 villes |
| `predict.py` | Prix estimé et fourchette à 90 % pour une nouvelle annonce |
| `Projet_prix_Airbnb_rapport_detaille.pdf` | Rapport détaillé |

## Lancer le projet

```bash
pip install -r requirements.txt
python download_airbnb.py && python merge_airbnb.py   # données Inside Airbnb (plusieurs Go)
python run_all.py                                      # toutes les étapes, graine fixée à 42
```

## Données

Les données ne sont pas incluses (plusieurs Go). Elles proviennent d'[Inside Airbnb](https://insideairbnb.com/get-the-data/), sous licence [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Inside Airbnb n'est pas affilié à Airbnb, Inc.

Technologies : Python, pandas, LightGBM, PyTorch, scikit-learn, Optuna.
