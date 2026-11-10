"""Détection de langue des descriptions d'annonces (langdetect, seed 42), en parallèle.
Fonction déterministe par ligne, sans apprentissage sur les données : pas de fuite.
Usage : python -m src.lang   -> data/processed/listing_lang.parquet
"""
import re

import pandas as pd
from joblib import Parallel, delayed

from src.data import ROOT


def _detect_chunk(texts):
    from langdetect import DetectorFactory, detect

    DetectorFactory.seed = 42
    out = []
    for t in texts:
        t = re.sub(r"<[^>]+>|http\S+|[\d\W_]+", " ", t or "").strip()[:400]
        if len(t) < 20:
            out.append("unk")
            continue
        try:
            out.append(detect(t))
        except Exception:
            out.append("unk")
    return out


def run():
    d = pd.read_parquet(ROOT / "data/processed/listings_all.parquet", columns=["city", "id", "description", "name"])
    txt = d["description"].fillna(d["name"]).fillna("").tolist()
    chunks = [txt[i:i + 5000] for i in range(0, len(txt), 5000)]
    res = Parallel(n_jobs=14)(delayed(_detect_chunk)(c) for c in chunks)
    d["desc_lang"] = [x for r in res for x in r]
    d[["city", "id", "desc_lang"]].to_parquet(ROOT / "data/processed/listing_lang.parquet", index=False)
    print(d["desc_lang"].value_counts(normalize=True).head(10).round(3).to_dict())


if __name__ == "__main__":
    run()
