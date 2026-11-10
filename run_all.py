"""Exécute le projet de bout en bout (reproductible, random_state = 42, configuration dans config.yaml).

Usage :
    python run_all.py              # tout (plusieurs heures sur un portable à 16 cœurs ; les étapes lourdes sont mises en cache)
    python run_all.py --from 5     # reprendre à l'étape 5
    python run_all.py --only 3     # une seule étape
Les téléchargements initiaux (download_airbnb.py, merge_airbnb.py) sont supposés déjà faits : data/processed/listings_all.parquet et reviews_all.parquet.
"""
import subprocess
import sys
import time

STEPS = [
    ("0  Audit des données", ["-m", "src.audit"]),
    ("1  Analyse exploratoire", ["-m", "src.eda"]),
    ("2  Nettoyage + split train/test (test scellé)", ["-m", "src.cleaning"]),
    ("2b Impact de la règle d'outliers", ["-m", "src.outlier_impact"]),
    ("3  Variables + contrôles anti-fuite", ["-m", "src.features"]),
    ("3b Contrôles des features", ["-m", "src.features_check"]),
    ("4  Sélection de variables", ["-m", "src.selection"]),
    ("4b Rapport de sélection", ["-m", "src.selection_report"]),
    ("5  Schémas de validation A/B/C", ["-m", "src.validation"]),
    ("5b Rapport de validation", ["-m", "src.validation_report"]),
    ("6  Régression (course, réglage Optuna, OOF, LOCO)", ["-m", "src.regression", "all"]),
    ("7  Classification (Superhost, tranche de prix)", ["-m", "src.classification"]),
    ("8  Clustering, PCA, UMAP, anomalies", ["-m", "src.clustering"]),
    ("9  NLP : échantillon, sentiment, embeddings", ["-m", "src.nlp", "all"]),
    ("9b NLP : analyses et valeur ajoutée", ["-m", "src.nlp_analysis"]),
    ("10 Séries temporelles (calendar)", ["-m", "src.timeseries"]),
    ("11 Deep learning", ["-m", "src.deep"]),
    ("14 Modèle final + ouverture UNIQUE du test", ["-m", "src.final_model"]),
    ("12 Interprétabilité (SHAP, PDP)", ["-m", "src.interpret"]),
    ("13 Apport métier", ["-m", "src.business"]),
    ("   Notebook final", ["tools/build_notebook.py"]),
]

if __name__ == "__main__":
    a = sys.argv[1:]
    lo = int(a[a.index("--from") + 1]) if "--from" in a else 1
    only = int(a[a.index("--only") + 1]) if "--only" in a else None
    for i, (name, cmd) in enumerate(STEPS, 1):
        if (only and i != only) or i < lo:
            continue
        t0 = time.time()
        print(f"\n=== [{i}/{len(STEPS)}] {name} ===", flush=True)
        r = subprocess.run([sys.executable] + cmd)
        print(f"--- {name} : {'OK' if r.returncode == 0 else 'ÉCHEC'} en {time.time() - t0:.0f}s", flush=True)
        if r.returncode != 0:
            sys.exit(r.returncode)
