"""Chargement des données et audit (Phase 0)."""
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = ROOT / "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_listings(cfg: dict | None = None, city: str | None = None, columns=None) -> pd.DataFrame:
    """Charge les annonces brutes (optionnellement une seule ville, ex. 'paris')."""
    cfg = cfg or load_config()
    filters = [("city", "==", city)] if city else None
    return pd.read_parquet(ROOT / cfg["paths"]["listings_raw"], columns=columns, filters=filters)


def reviews_metadata(cfg: dict | None = None) -> pq.FileMetaData:
    """Métadonnées des avis sans charger les 2,6 Go en mémoire."""
    cfg = cfg or load_config()
    return pq.ParquetFile(ROOT / cfg["paths"]["reviews_raw"]).metadata


def make_split(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Test final : 1 pli sur n_splits, groupé par (city, host_id) et stratifié par ville. Fixé une fois."""
    from sklearn.model_selection import StratifiedGroupKFold

    cfg = cfg or load_config()
    k = cfg["validation"]["final_test"]["n_splits"]
    groups = df["city"].astype(str) + "_" + df["host_id"].astype(str)
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=cfg["seed"])
    _, test_idx = next(sgkf.split(df, df["city"], groups))
    out = df[["city", "id", "host_id"]].copy()
    out["is_test"] = False
    out.iloc[test_idx, out.columns.get_loc("is_test")] = True
    return out
