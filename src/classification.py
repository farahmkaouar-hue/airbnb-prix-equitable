"""Phase 7 : classification.
  Tâche 1 : is_superhost (déséquilibré, ~30 % de positifs) ; test avec / sans les variables suspectes de fuite (notes, activité des avis)
  Tâche 2 : price_tier (bas / moyen / haut, terciles de prix PAR ville)
Validation : CV A (5 plis groupés par hôte) sur un échantillon du train ; schéma B (leave-one-city-out) pour le meilleur modèle.
Le test final n'est jamais lu. Usage : python -m src.classification [test]   (« test » = version rapide)
"""
import pickle
import sys
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.data import ROOT
from src.evaluate import group_key
from src.features import build_preprocessor, feature_groups, feature_set
from src.models import fold_matrices

warnings.filterwarnings("ignore")
SEED = 42
CACHE = ROOT / "models" / "_clf_cache"
CACHE.mkdir(parents=True, exist_ok=True)
FAST = len(sys.argv) > 1 and sys.argv[1] == "test"
N = 15000 if FAST else 60000


def sets_superhost(columns):
    g = feature_groups(columns)
    core = g["property"] + g["amenities"] + g["location"] + g["host"] + g["text"]
    return {"A. sans avis (caractéristiques seules)": core,
            "B. + notes d'avis": core + g["quality"],
            "C. + notes et activité des avis (suspect de fuite)": core + g["quality"] + [c for c in g["activity"] if c not in ("is_stale",)]}


def zoo(multiclass=False, n_jobs=16):
    return {
        "Régression logistique": (LogisticRegression(max_iter=500, C=0.5), True),
        "kNN (k=25)": (KNeighborsClassifier(n_neighbors=25, n_jobs=n_jobs), True),
        "SVM (RBF, 12 000 ex.)": (SVC(C=1.0, probability=True, random_state=SEED), True),
        "Arbre de décision": (DecisionTreeClassifier(max_depth=10, min_samples_leaf=30, random_state=SEED), False),
        "Random Forest": (RandomForestClassifier(n_estimators=250, min_samples_leaf=3, max_features="sqrt", n_jobs=n_jobs, random_state=SEED), False),
        "HistGradientBoosting": (HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=SEED), False),
        "LightGBM": (lgb.LGBMClassifier(n_estimators=300, learning_rate=0.06, num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=n_jobs,
                                        random_state=SEED, verbose=-1), False),
    }


def best_f1_threshold(y, p):
    ts = np.linspace(0.05, 0.95, 91)
    f = [f1_score(y, p >= t) for t in ts]
    return float(ts[int(np.argmax(f))])


def bin_metrics(y, p, thr=0.5):
    yh = p >= thr
    return dict(auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p), accuracy=accuracy_score(y, yh), precision=precision_score(y, yh), recall=recall_score(y, yh),
                f1=f1_score(y, yh), f1_macro=f1_score(y, yh, average="macro"), thr=thr)


def multi_metrics(y, P):
    yh = P.argmax(1)
    return dict(accuracy=accuracy_score(y, yh), bal_acc=balanced_accuracy_score(y, yh), f1_macro=f1_score(y, yh, average="macro"), auc_ovr=roc_auc_score(y, P, multi_class="ovr"))


def load(task):
    d = pd.read_parquet(ROOT / "data/processed/listings_features.parquet")
    d = d[~d["is_test"]]
    if task == "superhost":
        d = d.dropna(subset=["is_superhost"])
    else:
        d = d[d["in_reg_set"]].dropna(subset=["price_tier"])
    return d.sample(min(N, len(d)), random_state=SEED).reset_index(drop=True)


# ----------------------------------------------------------------------------------------- Tâche 1
def superhost():
    f = CACHE / ("superhost_fast.pkl" if FAST else "superhost.pkl")
    if f.exists():
        return pickle.load(open(f, "rb"))
    S = load("superhost")
    y = S["is_superhost"].astype(int).values
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(S, y, group_key(S)))
    rows, oofs = [], {}
    # (1) comparaison des modèles sur le jeu A
    # (2) test avec / sans variables suspectes avec LightGBM et régression logistique
    feat_sets = sets_superhost(S.columns)
    for set_name, feats in feat_sets.items():
        models = zoo() if set_name.startswith("A.") else {k: v for k, v in zoo().items() if k in ("Régression logistique", "LightGBM")}
        oof = {k: np.zeros(len(S)) for k in models}
        tm = {k: 0.0 for k in models}
        for i, (tr, va) in enumerate(folds):
            Z, Zv, Zs, Zvs = fold_matrices(S, feats, tr, va, y, lambda ft, X, scale: build_preprocessor(ft, X, scale=scale, target_type="binary"))
            for name, (m, scale) in models.items():
                t0 = time.time()
                Xtr, Xva, ytr = (Zs, Zvs, y[tr]) if scale else (Z, Zv, y[tr])
                if name.startswith("SVM"):
                    sub = np.random.RandomState(SEED).choice(len(Xtr), min(12000 if not FAST else 4000, len(Xtr)), replace=False)
                    m.fit(Xtr.iloc[sub], ytr[sub])
                else:
                    m.fit(Xtr, ytr)
                oof[name][va] = m.predict_proba(Xva)[:, 1]
                tm[name] += time.time() - t0
            print(f"superhost {set_name[:2]} pli {i}", flush=True)
        for k, p in oof.items():
            m = bin_metrics(y, p, 0.5)
            m.update(modèle=k, jeu=set_name, temps_s=tm[k] / 5, auc_std=float(np.std([roc_auc_score(y[va], p[va]) for _, va in folds])))
            rows.append(m)
        oofs[set_name] = oof
    table = pd.DataFrame(rows)

    # (3) déséquilibre : class_weight, SMOTE (dans la CV, train du pli seulement), seuil optimisé (choisi sur un split interne du train du pli)
    feats = feat_sets["B. + notes d'avis"]
    variants = {"LightGBM standard": None, "LightGBM class_weight='balanced'": "cw", "LightGBM + SMOTE": "smote", "LightGBM seuil optimisé (F1)": "thr"}
    oofv = {k: np.zeros(len(S)) for k in variants}
    thr_used = {k: [] for k in variants}
    for i, (tr, va) in enumerate(folds):
        Z, Zv, _, _ = fold_matrices(S, feats, tr, va, y, lambda ft, X, scale: build_preprocessor(ft, X, scale=scale, target_type="binary"))
        P = dict(n_estimators=300, learning_rate=0.06, num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=16, random_state=SEED, verbose=-1)
        for k, mode in variants.items():
            if mode is None:
                oofv[k][va] = lgb.LGBMClassifier(**P).fit(Z, y[tr]).predict_proba(Zv)[:, 1]
            elif mode == "cw":
                oofv[k][va] = lgb.LGBMClassifier(**P, class_weight="balanced").fit(Z, y[tr]).predict_proba(Zv)[:, 1]
            elif mode == "smote":
                Xs, ys = SMOTE(random_state=SEED).fit_resample(Z, y[tr])
                oofv[k][va] = lgb.LGBMClassifier(**P).fit(Xs, ys).predict_proba(Zv)[:, 1]
            else:
                a, b = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(Z, groups=group_key(S).values[tr]))
                mm = lgb.LGBMClassifier(**P).fit(Z.iloc[a], y[tr][a])
                t = best_f1_threshold(y[tr][b], mm.predict_proba(Z.iloc[b])[:, 1])
                thr_used[k].append(t)
                oofv[k][va] = lgb.LGBMClassifier(**P).fit(Z, y[tr]).predict_proba(Zv)[:, 1]
        print(f"superhost déséquilibre pli {i}", flush=True)
    imb = []
    for k, p in oofv.items():
        thr = float(np.mean(thr_used[k])) if thr_used[k] else 0.5
        imb.append(dict(variante=k, **bin_metrics(y, p, thr)))
    imb = pd.DataFrame(imb)

    # (4) schéma B (leave-one-city-out) avec LightGBM, jeu B
    lo = []
    cities = S["city"].values
    for c in sorted(S.city.unique()):
        tr, va = np.where(cities != c)[0], np.where(cities == c)[0]
        Z, Zv, _, _ = fold_matrices(S, feats, tr, va, y, lambda ft, X, scale: build_preprocessor(ft, X, scale=scale, target_type="binary"))
        p = lgb.LGBMClassifier(**P).fit(Z, y[tr]).predict_proba(Zv)[:, 1]
        lo.append(dict(city=c, auc=roc_auc_score(y[va], p), pr_auc=average_precision_score(y[va], p), base_rate=y[va].mean(), n=len(va)))
    loco = pd.DataFrame(lo)
    res = dict(table=table, imbalance=imb, oof=oofs, oofv=oofv, y=y, loco=loco, S_index=S[["city", "id", "host_id"]], n=len(S))
    pickle.dump(res, open(f, "wb"))
    return res


# ----------------------------------------------------------------------------------------- Tâche 2
def price_tier():
    f = CACHE / ("tier_fast.pkl" if FAST else "tier.pkl")
    if f.exists():
        return pickle.load(open(f, "rb"))
    S = load("tier")
    y = S["price_tier"].astype(int).values
    feats = feature_set(S.columns, "core+quote+quality")
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(S, y, group_key(S)))
    models = zoo(multiclass=True)
    oof = {k: np.zeros((len(S), 3)) for k in models}
    tm = {k: 0.0 for k in models}
    for i, (tr, va) in enumerate(folds):
        Z, Zv, Zs, Zvs = fold_matrices(S, feats, tr, va, y, lambda ft, X, scale: build_preprocessor(ft, X, scale=scale, target_type="multiclass"))
        for name, (m, scale) in models.items():
            t0 = time.time()
            Xtr, Xva = (Zs, Zvs) if scale else (Z, Zv)
            if name.startswith("SVM"):
                sub = np.random.RandomState(SEED).choice(len(Xtr), min(12000 if not FAST else 4000, len(Xtr)), replace=False)
                m.fit(Xtr.iloc[sub], y[tr][sub])
            else:
                m.fit(Xtr, y[tr])
            oof[name][va] = m.predict_proba(Xva)
            tm[name] += time.time() - t0
        print(f"tier pli {i}", flush=True)
    table = pd.DataFrame([dict(modèle=k, temps_s=tm[k] / 5, **multi_metrics(y, P)) for k, P in oof.items()])
    # schéma B pour LightGBM : tranches relatives à la ville -> généralisation à une ville inconnue
    lo = []
    cities = S["city"].values
    for c in sorted(S.city.unique()):
        tr, va = np.where(cities != c)[0], np.where(cities == c)[0]
        Z, Zv, _, _ = fold_matrices(S, feats, tr, va, y, lambda ft, X, scale: build_preprocessor(ft, X, scale=scale, target_type="multiclass"))
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.06, num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=16, random_state=SEED, verbose=-1)
        P = m.fit(Z, y[tr]).predict_proba(Zv)
        lo.append(dict(city=c, n=len(va), **multi_metrics(y[va], P)))
    res = dict(table=table, oof=oof, y=y, loco=pd.DataFrame(lo), S_index=S[["city", "id", "host_id"]], n=len(S))
    pickle.dump(res, open(f, "wb"))
    return res


if __name__ == "__main__":
    a = superhost()
    print(a["table"].round(4).to_string())
    print(a["imbalance"].round(4).to_string())
    b = price_tier()
    print(b["table"].round(4).to_string())
