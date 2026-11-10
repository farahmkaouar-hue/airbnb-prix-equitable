"""Catalogue de modèles (Phases 6, 7, 11) et utilitaires communs de validation croisée sur matrices prétraitées."""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

SEED = 42


class CityMedianBaseline:
    """Baseline : médiane de log(prix) de la ville (repli : médiane globale pour une ville inconnue)."""

    def fit(self, cities, y):
        s = pd.Series(np.asarray(y)).groupby(np.asarray(cities)).median()
        self.med_, self.glob_ = s, float(np.median(y))
        return self

    def predict(self, cities):
        return pd.Series(np.asarray(cities)).map(self.med_).fillna(self.glob_).values


def regression_zoo(n_jobs=16):
    """nom -> (estimateur, a_besoin_de_scaling). Hyperparamètres « par défaut raisonnables », identiques pour tous les plis."""
    import lightgbm as lgb
    import xgboost as xgb

    return {
        "Régression linéaire": (LinearRegression(), True),
        "Ridge": (Ridge(alpha=10, random_state=SEED), True),
        "Lasso": (Lasso(alpha=0.001, max_iter=5000, random_state=SEED), True),
        "ElasticNet": (ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000, random_state=SEED), True),
        "kNN (k=15)": (KNeighborsRegressor(n_neighbors=15, weights="distance", n_jobs=n_jobs), True),
        "Arbre de décision": (DecisionTreeRegressor(max_depth=12, min_samples_leaf=20, random_state=SEED), False),
        "Random Forest": (RandomForestRegressor(n_estimators=200, min_samples_leaf=3, max_features=0.33, n_jobs=n_jobs, random_state=SEED), False),
        "Gradient Boosting (sklearn)": (GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8, random_state=SEED), False),
        "HistGradientBoosting": (HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=SEED), False),
        "XGBoost": (xgb.XGBRegressor(n_estimators=400, learning_rate=0.08, max_depth=7, subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=n_jobs,
                                     random_state=SEED, verbosity=0), False),
        "LightGBM": (lgb.LGBMRegressor(n_estimators=300, learning_rate=0.08, num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                       n_jobs=n_jobs, random_state=SEED, verbose=-1), False),
    }


def fold_matrices(d, feats, tr, va, y, build_preprocessor):
    """Prétraitement ajusté sur le train du pli ; renvoie (Z_tr, Z_va, Zs_tr, Zs_va) (Zs = standardisé avec les stats du train du pli)."""
    pp = build_preprocessor(feats, d.iloc[tr], scale=False)
    Z = pp.fit_transform(d.iloc[tr], y[tr])
    Zv = pp.transform(d.iloc[va])
    mu, sd = Z.mean(), Z.std().replace(0, 1)
    return Z, Zv, (Z - mu) / sd, (Zv - mu) / sd
