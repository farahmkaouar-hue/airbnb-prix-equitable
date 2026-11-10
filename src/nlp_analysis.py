"""Phase 9 (suite) : analyses NLP à partir des avis échantillonnés, du sentiment et des embeddings (src/nlp.py).
Tout ce qui apprend sur le texte (TF-IDF, LDA, PCA des embeddings) est ajusté dans chaque pli sur les annonces d'entraînement seulement.
Usage : python -m src.nlp_analysis   -> models/_nlp_cache/analysis.pkl + reports/nlp_*.csv
"""
import pickle
import re
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from src.data import ROOT
from src.evaluate import group_key, rmse
from src.features import build_preprocessor, feature_set
from src.models import fold_matrices
from src.nlp import CACHE, embeddings, sample_reviews, sentiment

warnings.filterwarnings("ignore")
SEED = 42
FRICTION = {
    "propreté": r"\b(dirty|filthy|stain\w*|smell\w*|stink\w*|dusty|mold\w*|mould\w*|cockroach\w*|bugs?|bedbugs?|hair on|not clean|unclean|grim\w*)\b",
    "bruit": r"\b(noise|noisy|loud|thin walls?|music|party|construction|street noise|couldn't sleep|could not sleep|barking)\b",
    "accueil / communication": r"\b(rude|unresponsive|no response|didn't answer|did not answer|never answered|unhelpful|unfriendly|hard to reach|cancel\w*)\b",
    "literie / confort": r"\b(uncomfortable|hard mattress|old mattress|sagging|pillows? (were|was)|tiny|cramped|too small|no air|stuffy)\b",
    "équipements en panne": r"\b(no hot water|broken|not working|didn't work|did not work|no heating|no wifi|wifi (was )?(down|bad|slow)|no air conditioning|no a/?c|leak\w*|mice|ants)\b",
    "localisation / sécurité": r"\b(far from|unsafe|sketchy|dangerous|isolated|hard to find|inconvenient|steep stairs|no elevator)\b",
    "conformité à l'annonce": r"\b(misleading|not as described|not as advertised|different from (the )?(photos|pictures)|photos? (are|were) (old|misleading)|false advertis\w*|disappointed)\b",
    "arrivée / clés": r"\b(lockbox|key|keys|check-?in|check in|late arrival|waited|waiting|couldn't get in|could not get in|code)\b.*\b(problem|issue|trouble|wait\w*|late|difficult|confus\w*)\b",
}


def build():
    f = CACHE / "analysis.pkl"
    if f.exists():
        return pickle.load(open(f, "rb"))
    r = sample_reviews()
    s = sentiment(r)
    E = embeddings(r)
    r = pd.concat([r.reset_index(drop=True), s.reset_index(drop=True)], axis=1)
    r["neg"] = (r["p_negative"] > 0.5).astype(int)
    en = r["lang"] == "en"
    txt = r["comments"].str.lower()
    for k, rx in FRICTION.items():
        r["fr_" + k] = (en & txt.str.contains(rx, regex=True)).astype(int)
    out = {"n_reviews": len(r), "langs": r.lang.value_counts(normalize=True).head(8).round(3).to_dict()}

    # ---- sentiment : distribution par langue et ville ; validité externe (corrélation avec la note de l'annonce)
    out["sent_by_lang"] = r.groupby("lang").agg(n=("sentiment", "size"), sentiment_moyen=("sentiment", "mean"), part_negatifs=("neg", "mean")).query("n >= 800").sort_values("n", ascending=False).round(3)
    out["sent_by_city"] = r.groupby("city").agg(sentiment_moyen=("sentiment", "mean"), part_negatifs=("neg", "mean"), n=("sentiment", "size")).round(3)
    L = pd.read_parquet(ROOT / "data/processed/listings_features.parquet")
    L = L[~L.is_test & L.in_reg_set]
    key = r.groupby(["city", "listing_id"]).agg(sent_mean=("sentiment", "mean"), p_pos=("p_positive", "mean"), p_neg=("p_negative", "mean"), neg_share=("neg", "mean"), n_rev=("neg", "size"),
                                                 **{"fr_" + k: ("fr_" + k, "mean") for k in FRICTION}).reset_index().rename(columns={"listing_id": "id"})
    D = key.merge(L, on=["city", "id"], how="inner").reset_index(drop=True)
    out["n_listings"] = len(D)
    out["sent_vs_rating_spearman"] = float(spearmanr(D["sent_mean"], D["review_scores_rating"], nan_policy="omit")[0])
    out["sent_vs_price_spearman"] = float(spearmanr(D["sent_mean"], D["log_price_eur"])[0])

    # ---- lexique de friction (avis en anglais) : fréquence dans les avis négatifs vs positifs, par ville
    e = r[en]
    fr = pd.DataFrame({k: [e.loc[e.neg == 1, "fr_" + k].mean() * 100, e.loc[e.neg == 0, "fr_" + k].mean() * 100] for k in FRICTION}, index=["avis négatifs (%)", "avis non négatifs (%)"]).T
    fr["rapport"] = fr["avis négatifs (%)"] / fr["avis non négatifs (%)"].replace(0, np.nan)
    out["friction"] = fr.sort_values("avis négatifs (%)", ascending=False).round(2)
    out["friction_by_city"] = (e[e.neg == 1].groupby("city")[["fr_" + k for k in FRICTION]].mean() * 100).rename(columns=lambda c: c[3:]).round(1)
    out["n_neg_en"] = int((e.neg == 1).sum())
    out["share_neg"] = float(r.neg.mean())

    # ---- topics LDA (descriptif, avis en anglais)
    cv = CountVectorizer(stop_words="english", min_df=30, max_df=0.4, max_features=8000, token_pattern=r"(?u)\b[a-zA-Z]{3,}\b")
    Xc = cv.fit_transform(e["comments"])
    lda = LatentDirichletAllocation(8, learning_method="batch", max_iter=25, random_state=SEED, n_jobs=-1).fit(Xc)
    vocab = np.array(cv.get_feature_names_out())
    out["topics"] = pd.DataFrame({f"thème {i + 1}": vocab[np.argsort(c)[::-1][:12]] for i, c in enumerate(lda.components_)})
    tn = lda.transform(Xc)
    out["topic_sentiment"] = pd.DataFrame({"thème dominant": tn.argmax(1) + 1, "sentiment": e["sentiment"].values, "neg": e["neg"].values}).groupby("thème dominant").agg(
        part_des_avis=("sentiment", lambda s: len(s) / len(e)), sentiment_moyen=("sentiment", "mean"), part_negatifs=("neg", "mean")).round(3)

    # ---- texte par annonce : TF-IDF / embeddings (comparaison) puis valeur ajoutée (même validation)
    r = r.merge(D[["city", "id"]].rename(columns={"id": "listing_id"}), on=["city", "listing_id"])
    docs = r.groupby(["city", "listing_id"])["comments"].apply(lambda x: " ".join(x)).reset_index().rename(columns={"listing_id": "id"})
    D = D.merge(docs, on=["city", "id"])
    idx_map = {(c, l): i for i, (c, l) in enumerate(zip(r.city, r.listing_id))}
    Er = E[[i for i, (c, l) in enumerate(zip(sample_reviews().city, sample_reviews().listing_id)) if (c, l) in idx_map]] if False else None
    base_r = sample_reviews().reset_index(drop=True)
    pos = base_r.groupby(["city", "listing_id"]).indices
    Emb = np.vstack([E[pos[(c, i)]].mean(0) for c, i in zip(D.city, D.id)])
    y_price = D["log_price_eur"].values
    y_sh = D["is_superhost"].astype(int).values
    y_hi = (D["review_scores_rating"] >= 4.8).astype(float).values.copy()
    y_hi[D["review_scores_rating"].isna().values] = np.nan
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(D, D["city"], group_key(D)))
    feats_price = feature_set(D.columns, "core+quote+quality")
    # ATTENTION : pour prédire le Superhost, le jeu ne doit PAS contenir host_is_superhost (la cible). On réutilise les jeux de la classification (phase 7).
    from src.classification import sets_superhost

    sh_sets = sets_superhost(D.columns)
    feats_sh = {"A. sans notes": sh_sets["A. sans avis (caractéristiques seules)"], "B. avec notes": sh_sets["B. + notes d'avis"]}
    assert all("host_is_superhost" not in v for v in feats_sh.values())
    tfi_rows, val_rows = [], []
    fr_cols = ["fr_" + k for k in FRICTION]
    sent_cols = ["sent_mean", "p_pos", "p_neg", "neg_share"]
    P = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, n_jobs=16, random_state=SEED, verbose=-1)
    for i, (tr, va) in enumerate(folds):
        Dtr, Dva = D.iloc[tr], D.iloc[va]
        # TF-IDF vs embeddings (texte seul) : note élevée, Superhost, prix
        tf = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=60000, sublinear_tf=True).fit(Dtr["comments"])
        Ttr, Tva = tf.transform(Dtr["comments"]), tf.transform(Dva["comments"])
        ok_tr, ok_va = ~np.isnan(y_hi[tr]), ~np.isnan(y_hi[va])
        for task, y, kind in [("note ≥ 4,8", y_hi, "cls"), ("Superhost", y_sh, "cls"), ("log prix", y_price, "reg")]:
            ytr, yva = y[tr], y[va]
            mtr, mva = (~np.isnan(ytr), ~np.isnan(yva))
            for rep, (A, B) in {"TF-IDF (1-2 grammes)": (Ttr, Tva), "Embeddings multilingues (moyenne)": (Emb[tr], Emb[va])}.items():
                if kind == "cls":
                    m = LogisticRegression(C=4 if "TF" in rep else 2, max_iter=1000).fit(A[mtr], ytr[mtr].astype(int))
                    sc = roc_auc_score(yva[mva], m.predict_proba(B[mva])[:, 1])
                else:
                    m = Ridge(alpha=1.0 if "TF" in rep else 0.3).fit(A, ytr)
                    sc = rmse(yva, m.predict(B))
                tfi_rows.append(dict(pli=i, tâche=task, représentation=rep, score=sc))
        # valeur ajoutée sur les variables tabulaires
        pcs = PCA(16, random_state=SEED).fit(Emb[tr])
        Ptr, Pva = pcs.transform(Emb[tr]), pcs.transform(Emb[va])
        cvf = CountVectorizer(stop_words="english", min_df=20, max_df=0.4, max_features=5000, token_pattern=r"(?u)\b[a-zA-Z]{3,}\b")
        en_tr = r[(r.lang == "en")].merge(Dtr[["city", "id"]].rename(columns={"id": "listing_id"}), on=["city", "listing_id"])
        Xl = cvf.fit_transform(en_tr["comments"])
        ldaf = LatentDirichletAllocation(8, learning_method="batch", max_iter=15, random_state=SEED, n_jobs=-1).fit(Xl)

        def topic_feats(Dsub):
            rr = r[r.lang == "en"].merge(Dsub[["city", "id"]].rename(columns={"id": "listing_id"}), on=["city", "listing_id"])
            T = pd.DataFrame(ldaf.transform(cvf.transform(rr["comments"])), columns=[f"topic{j + 1}" for j in range(8)])
            T["city"], T["id"] = rr["city"].values, rr["listing_id"].values
            T = T.groupby(["city", "id"]).mean().reset_index()
            return Dsub[["city", "id"]].merge(T, on=["city", "id"], how="left").drop(columns=["city", "id"]).values

        Ttop_tr, Ttop_va = topic_feats(Dtr), topic_feats(Dva)
        extra = {"sentiment": (Dtr[sent_cols].values, Dva[sent_cols].values), "friction": (Dtr[fr_cols].values, Dva[fr_cols].values),
                 "thèmes LDA": (Ttop_tr, Ttop_va), "embeddings (PCA 16)": (Ptr, Pva),
                 "tout le texte": (np.hstack([Dtr[sent_cols].values, Dtr[fr_cols].values, Ttop_tr, Ptr]), np.hstack([Dva[sent_cols].values, Dva[fr_cols].values, Ttop_va, Pva]))}
        tasks = [("prix (log)", feats_price, y_price, "reg", None)] + [(f"Superhost {k}", v, y_sh, "cls", k) for k, v in feats_sh.items()]
        for tname, ft, y, kind, _ in tasks:
            Z, Zv, _, _ = fold_matrices(D, ft, tr, va, y, lambda f_, X, scale: build_preprocessor(f_, X, scale=scale, target_type="continuous" if kind == "reg" else "binary"))
            variants = {"sans texte (référence)": (Z.values, Zv.values)}
            for k, (a, b) in extra.items():
                variants["+ " + k] = (np.hstack([Z.values, a]), np.hstack([Zv.values, b]))
            for vn, (A, B) in variants.items():
                if kind == "reg":
                    p = lgb.LGBMRegressor(**P).fit(A, y[tr]).predict(B)
                    sc = rmse(y[va], p)
                else:
                    p = lgb.LGBMClassifier(**P).fit(A, y[tr]).predict_proba(B)[:, 1]
                    sc = roc_auc_score(y[va], p)
                val_rows.append(dict(pli=i, tâche=tname, variante=vn, score=sc))
        print(f"NLP pli {i} terminé", flush=True)
    out["rep_compare"] = pd.DataFrame(tfi_rows)
    out["value_added"] = pd.DataFrame(val_rows)
    pickle.dump(out, open(CACHE / "analysis.pkl", "wb"))
    return out


if __name__ == "__main__":
    o = build()
    print(o["n_reviews"], o["n_listings"], o["langs"])
    print(o["sent_by_lang"])
    print(o["friction"])
    print(o["topics"])
    print(o["rep_compare"].groupby(["tâche", "représentation"]).score.mean().round(4))
    v = o["value_added"].pivot_table(index=["tâche", "variante"], values="score", aggfunc=["mean", "std"]).round(4)
    print(v)
    print("sent~rating", o["sent_vs_rating_spearman"], "sent~price", o["sent_vs_price_spearman"])
