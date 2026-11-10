"""Phase 11 : réseau de neurones dense (PyTorch, CPU) sur les données tabulaires vs boosting.

- catégories (type de logement, type de propriété, ville, quartier) -> couches d'embeddings (modalités rares regroupées)
- numériques : imputation médiane par ville + indicateurs de manquant + standardisation (ajustés sur le train de chaque pli)
- dropout, BatchNorm, AdamW, early stopping sur 12 % des hôtes du train du pli (jamais sur le pli de validation)
- comparaison HONNÊTE avec LightGBM : mêmes 80 000 annonces, mêmes plis A (5 plis groupés par hôte), réglage Optuna des deux (budgets annoncés)
Usage : python -m src.deep
"""
import json
import pickle
import time
import warnings

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from src.data import ROOT
from src.evaluate import group_key, rmse, mae
from src.features import CityMedianImputer, feature_set
from src.regression import N_COURSE, load_train, tune
from src.models import fold_matrices
from src.features import build_preprocessor

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
SEED = 42
CACHE = ROOT / "models" / "_deep_cache"
CACHE.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(16)
CATS = ["room_type", "property_type", "city", "nbhd_key"]


class Net(nn.Module):
    def __init__(self, n_num, cat_sizes, emb_dims, hidden, dropout):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(s, e) for s, e in zip(cat_sizes, emb_dims)])
        self.emb_drop = nn.Dropout(dropout / 2)
        d_in = n_num + sum(emb_dims)
        layers = []
        for h in hidden:
            layers += [nn.Linear(d_in, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d_in = h
        layers.append(nn.Linear(d_in, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, xn, xc):
        e = torch.cat([emb(xc[:, i]) for i, emb in enumerate(self.embs)], 1)
        return self.mlp(torch.cat([xn, self.emb_drop(e)], 1)).squeeze(1)


class TabPrep:
    """Prétraitement NN ajusté sur le train : numériques (imputation par ville + standardisation) et indices de catégories."""

    def __init__(self, feats, min_count=20):
        self.cat = [c for c in CATS if c in feats]
        self.num = [c for c in feats if c not in self.cat]
        self.min_count = min_count

    def fit(self, X):
        self.imp = CityMedianImputer(self.num).fit(X[self.num + ["city"]] if "city" not in self.num else X[self.num])
        Xi = self.imp.transform(X[self.num + ["city"]])
        self.cols = [c for c in Xi.columns if c != "city"]
        self.sc = StandardScaler().fit(Xi[self.cols].astype(float))
        self.maps = {}
        for c in self.cat:
            vc = X[c].astype(str).value_counts()
            self.maps[c] = {v: i + 1 for i, v in enumerate(vc[vc >= self.min_count].index)}  # 0 = rare / inconnu
        return self

    def transform(self, X):
        Xi = self.imp.transform(X[self.num + ["city"]])
        xn = self.sc.transform(Xi[self.cols].astype(float)).astype("float32")
        xc = np.column_stack([X[c].astype(str).map(self.maps[c]).fillna(0).astype(int).values for c in self.cat]).astype("int64")
        return xn, xc

    def cat_sizes(self):
        return [len(m) + 1 for m in self.maps.values()]


def fit_nn(Xtr, ytr, groups, prep, params, Xva=None, yva=None, max_epochs=60, patience=8, seed=SEED, log_curve=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    itr, iva = next(GroupShuffleSplit(1, test_size=0.12, random_state=seed).split(Xtr, groups=groups))
    xn, xc = prep.transform(Xtr)
    y = ytr.astype("float32")
    ymu, ysd = y[itr].mean(), y[itr].std()
    dims = [min(32, max(4, (s + 1) // 2)) for s in prep.cat_sizes()]
    net = Net(xn.shape[1], prep.cat_sizes(), dims, params["hidden"], params["dropout"])
    opt = torch.optim.AdamW(net.parameters(), lr=params["lr"], weight_decay=params["wd"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    T = lambda a: torch.from_numpy(a)
    xn_t, xc_t, y_t = T(xn), T(xc), T((y - ymu) / ysd)
    best, best_state, bad, curve = 1e9, None, 0, []
    bs = params["batch"]
    for ep in range(max_epochs):
        net.train()
        perm = np.random.permutation(itr)
        tl = 0.0
        for s in range(0, len(perm), bs):
            b = perm[s:s + bs]
            if len(b) < 8:
                continue
            opt.zero_grad()
            loss = nn.functional.mse_loss(net(xn_t[b], xc_t[b]), y_t[b])
            loss.backward()
            opt.step()
            tl += loss.item() * len(b)
        net.eval()
        with torch.no_grad():
            vl = nn.functional.mse_loss(net(xn_t[iva], xc_t[iva]), y_t[iva]).item()
        sched.step(vl)
        curve.append((ep + 1, tl / len(itr), vl))
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()

    def predict(X):
        a, c = prep.transform(X)
        with torch.no_grad():
            return net(T(a), T(c)).numpy() * ysd + ymu

    return (predict, curve) if log_curve else predict


def run():
    f = CACHE / "deep.pkl"
    if f.exists():
        return pickle.load(open(f, "rb"))
    d = load_train()
    S = d.sample(N_COURSE, random_state=SEED).reset_index(drop=True)
    feats = feature_set(S.columns, "core+quote+quality")
    y = S["log_price_eur"].values
    g = group_key(S).values

    # ---------------- réglage Optuna du réseau (3 plis groupés, 20 essais) ; LightGBM : 40 essais déjà faits en phase 6
    folds3 = list(GroupKFold(3).split(S, groups=g))

    def obj(t):
        p = dict(hidden=[t.suggest_categorical("h1", [128, 256, 512]), t.suggest_categorical("h2", [64, 128, 256]), t.suggest_categorical("h3", [32, 64])],
                 dropout=t.suggest_float("dropout", 0.0, 0.4), lr=t.suggest_float("lr", 5e-4, 5e-3, log=True), wd=t.suggest_float("wd", 1e-6, 1e-2, log=True),
                 batch=t.suggest_categorical("batch", [512, 1024, 2048]))
        sc = []
        for tr, va in folds3[:2]:  # 2 plis pour limiter le coût
            prep = TabPrep(feats).fit(S.iloc[tr])
            pr = fit_nn(S.iloc[tr], y[tr], g[tr], prep, p, max_epochs=35, patience=5)
            sc.append(rmse(y[va], pr(S.iloc[va])))
            t.report(float(np.mean(sc)), len(sc))
            if t.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(sc))

    t0 = time.time()
    st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED), pruner=optuna.pruners.MedianPruner(n_startup_trials=4))
    st.optimize(obj, n_trials=20)
    nn_secs = time.time() - t0
    bp = st.best_params
    best = dict(hidden=[bp["h1"], bp["h2"], bp["h3"]], dropout=bp["dropout"], lr=bp["lr"], wd=bp["wd"], batch=bp["batch"])
    print("NN meilleurs réglages", best, round(st.best_value, 4), f"{nn_secs:.0f}s", flush=True)

    # ---------------- évaluation sur les 5 plis A (mêmes plis que la course de la phase 6)
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(S, S["city"], group_key(S)))
    tuned = tune()
    lp = dict(tuned["LightGBM"]["best_params"], subsample_freq=1, random_state=SEED, n_jobs=16, verbose=-1)
    oof_nn, oof_lgb = np.zeros(len(S)), np.zeros(len(S))
    curves = []
    for i, (tr, va) in enumerate(folds):
        prep = TabPrep(feats).fit(S.iloc[tr])
        pr, curve = fit_nn(S.iloc[tr], y[tr], g[tr], prep, best, max_epochs=80, patience=10, log_curve=True)
        oof_nn[va] = pr(S.iloc[va])
        curves.append(curve)
        Z, Zv, _, _ = fold_matrices(S, feats, tr, va, y, build_preprocessor)
        oof_lgb[va] = lgb.LGBMRegressor(**lp).fit(Z, y[tr]).predict(Zv)
        print(f"pli {i}: NN {rmse(y[va], oof_nn[va]):.4f} | LightGBM {rmse(y[va], oof_lgb[va]):.4f}", flush=True)
    res = dict(best=best, nn_trials=[dict(n=t.number, v=t.value, s=str(t.state)) for t in st.trials], nn_seconds=nn_secs, nn_best_cv=st.best_value, lgb_seconds=tuned["LightGBM"]["seconds"],
               lgb_best_cv=tuned["LightGBM"]["best_rmse"], oof_nn=oof_nn, oof_lgb=oof_lgb, y=y, curves=curves, folds=[(tr.tolist(), va.tolist()) for tr, va in folds],
               city=S["city"].values)
    pickle.dump(res, open(f, "wb"))
    return res


if __name__ == "__main__":
    r = run()
    print("RMSE NN", rmse(r["y"], r["oof_nn"]), "LightGBM", rmse(r["y"], r["oof_lgb"]), "moyenne des deux", rmse(r["y"], (r["oof_nn"] + r["oof_lgb"]) / 2))
