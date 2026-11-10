"""Métriques et schémas de validation réutilisables (Phase 5)."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def r2(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def mae_eur(y_log, p_log):
    """MAE en euros après retransformation exp() simple (la correction de biais est traitée en phase 6)."""
    return float(np.mean(np.abs(np.exp(np.asarray(y_log)) - np.exp(np.asarray(p_log)))))


def metrics(y, p) -> dict:
    return dict(rmse=rmse(y, p), mae=mae(y, p), r2=r2(y, p), mae_eur=mae_eur(y, p), n=len(y))


def group_key(df: pd.DataFrame) -> pd.Series:
    """Groupe de validation = hôte (un hôte est propre à une ville)."""
    return df["city"].astype(str) + "_" + df["host_id"].astype(str)


def scheme_a(df: pd.DataFrame, k: int = 5):
    """A : K-fold stratifié par ville, groupé par hôte (intra-villes)."""
    cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=SEED)
    yield from cv.split(df, df["city"], group_key(df))


def scheme_b(df: pd.DataFrame):
    """B : leave-one-city-out. Rend (idx_train, idx_val, ville)."""
    cities = df["city"].values
    for c in sorted(df["city"].unique()):
        yield np.where(cities != c)[0], np.where(cities == c)[0], c


def scheme_c(df: pd.DataFrame, cutoff="2026-10-01"):
    """C (temporel via la date de check-in du devis) : hôtes disjoints ; entraînement = hôtes du groupe 1 avec check-in < cutoff ;
    validation « tardive » = hôtes du groupe 2 avec check-in >= cutoff ; contrôle « précoce » = hôtes du groupe 2 avec check-in < cutoff."""
    ci = pd.to_datetime(df["scrape_date"]) + pd.to_timedelta(df["quote_lead_days"], unit="D")
    hosts = group_key(df).drop_duplicates()
    rng = np.random.RandomState(SEED)
    g1 = set(hosts[rng.rand(len(hosts)) < 0.5])
    in1 = group_key(df).isin(g1).values
    early = (ci < pd.Timestamp(cutoff)).values
    known = ci.notna().values
    return (np.where(in1 & early & known)[0], np.where(~in1 & ~early & known)[0], np.where(~in1 & early & known)[0])
