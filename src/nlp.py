"""Phase 9 : NLP sur les avis (et descriptions).

Étapes (résultats en cache dans models/_nlp_cache) :
  1. échantillon d'avis (stratifié par ville), détection de langue, nettoyage
  2. sentiment : modèle multilingue pré-entraîné (lxyuan/distilbert-base-multilingual-cased-sentiments-student), agrégé par annonce
  3. représentations : TF-IDF + modèle linéaire ; embeddings de phrases multilingues (paraphrase-multilingual-MiniLM-L12-v2) ;
     tâche : prédire la note de l'annonce (note < médiane) et comparer
  4. topics : LDA sur avis en anglais (thèmes appréciés / reprochés), + lexique de friction (propreté, bruit, localisation, accueil)
  5. valeur ajoutée : le sentiment / les thèmes améliorent-ils la prédiction du prix et du Superhost ? (même validation, avec / sans)
Échantillonnage explicite : ~8 000 annonces du train (jeu de régression, >= 8 avis, 540 par ville max), leurs 8 avis les plus récents (~60 000 avis). Usage : python -m src.nlp [etape]
"""
import pickle
import re
import sys
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from src.data import ROOT
from src.evaluate import group_key

warnings.filterwarnings("ignore")
SEED = 42
CACHE = ROOT / "models" / "_nlp_cache"
CACHE.mkdir(parents=True, exist_ok=True)
N_REVIEWS = 60000
SENT_MODEL = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
EMB_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def clean(t: str) -> str:
    t = re.sub(r"<[^>]+>|http\S+", " ", str(t))
    return re.sub(r"\s+", " ", t).strip()


def sample_listings(per_city=540, min_reviews=8):
    """Annonces du TRAIN avec prix (jeu de régression) et >= 8 avis, échantillonnées par ville (échantillonnage explicite, voir en-tête)."""
    d = pd.read_parquet(ROOT / "data/processed/listings_features.parquet", columns=["city", "id", "host_id", "number_of_reviews", "is_test", "in_reg_set"])
    d = d[~d.is_test & d.in_reg_set & (d.number_of_reviews >= min_reviews)]
    return pd.concat([g.sample(min(per_city, len(g)), random_state=SEED) for _, g in d.groupby("city")])[["city", "id"]]


def sample_reviews(k_per_listing=8):
    """Les k avis les plus récents de chaque annonce échantillonnée, nettoyés, avec langue détectée."""
    f = CACHE / "reviews_sample.parquet"
    if f.exists():
        return pd.read_parquet(f)
    L = sample_listings()
    ids = set(zip(L.city, L.id))
    pf = pq.ParquetFile(ROOT / "data/processed/reviews_all.parquet")
    parts = []
    for i in range(pf.num_row_groups):
        t = pf.read_row_group(i, columns=["city", "listing_id", "date", "comments"]).to_pandas()
        t = t[[(c, l) in ids for c, l in zip(t.city, t.listing_id)]]
        parts.append(t.dropna(subset=["comments"]))
    r = pd.concat(parts, ignore_index=True)
    r["comments"] = r["comments"].map(clean)
    r = r[r["comments"].str.len().between(30, 1200)]
    r["date"] = pd.to_datetime(r["date"], errors="coerce")
    r = r.sort_values("date", ascending=False).groupby(["city", "listing_id"]).head(k_per_listing).reset_index(drop=True)
    from langdetect import DetectorFactory, detect

    DetectorFactory.seed = SEED

    def det(s):
        try:
            return detect(s[:300])
        except Exception:
            return "unk"

    r["lang"] = r["comments"].map(det)
    r.to_parquet(f, index=False)
    return r


def sentiment(r):
    f = CACHE / "sentiment.parquet"
    if f.exists():
        return pd.read_parquet(f)
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(16)
    tok = AutoTokenizer.from_pretrained(SENT_MODEL)
    mdl = AutoModelForSequenceClassification.from_pretrained(SENT_MODEL).eval()
    labels = [mdl.config.id2label[i].lower() for i in range(mdl.config.num_labels)]
    out = []
    texts = r["comments"].tolist()
    t0 = time.time()
    order = np.argsort([len(t) for t in texts])
    probs = np.zeros((len(texts), len(labels)), dtype="float32")
    with torch.no_grad():
        for s in range(0, len(texts), 64):
            idx = order[s:s + 64]
            b = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=128, return_tensors="pt")
            probs[idx] = torch.softmax(mdl(**b).logits, -1).numpy()
            if (s // 64) % 100 == 0:
                print(f"sentiment {s}/{len(texts)} {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(probs, columns=[f"p_{l}" for l in labels])
    df["sentiment"] = df.get("p_positive", 0) - df.get("p_negative", 0)
    df.to_parquet(f, index=False)
    return df


def embeddings(r):
    f = CACHE / "embeddings.npy"
    if f.exists():
        return np.load(f)
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(16)
    m = SentenceTransformer(EMB_MODEL, device="cpu")
    E = m.encode(r["comments"].tolist(), batch_size=128, show_progress_bar=False, normalize_embeddings=True)
    np.save(f, E.astype("float32"))
    return E


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    r = sample_reviews()
    print(len(r), "avis ;", r.lang.value_counts(normalize=True).head(6).round(3).to_dict(), flush=True)
    if step in ("sentiment", "all"):
        sentiment(r)
    if step in ("emb", "all"):
        embeddings(r)
