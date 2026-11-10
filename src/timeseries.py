"""Phase 10 : séries temporelles sur le calendar (disponibilité par jour).

ATTENTION aux limites des données (documentées dans le rapport) :
  - le calendar 2026 ne contient PAS de prix (colonnes : listing_id, date, available, minimum_nights, maximum_nights) : on modélise la
    disponibilité (part de nuits indisponibles = réservées OU bloquées par l'hôte), proxy de la demande ;
  - c'est UN instantané pris à la date de collecte, qui regarde 365 jours vers l'avenir : la part indisponible dépend de l'horizon
    (forte à court terme : réservations + blocages ; minimale vers 6 mois ; qui remonte ensuite : calendriers pas encore ouverts)
    -> l'effet « horizon » est mélangé à la saisonnalité annuelle, que l'on ne peut donc pas isoler avec un seul instantané.
Usage : python -m src.timeseries   -> data/processed/daily_availability.parquet, reports/ts_*.csv, figures ts_*.png
"""
import time
import warnings

import holidays
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.data import ROOT

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
FIG = ROOT / "reports" / "figures"
SEED = 42
COUNTRY = {"amsterdam": "NL", "barcelona": "ES", "berlin": "DE", "lisbon": "PT", "london": "GB", "los-angeles": "US", "lyon": "FR", "madrid": "ES",
           "mexico-city": "MX", "new-york-city": "US", "paris": "FR", "rome": "IT", "sydney": "AU", "tokyo": "JP", "toronto": "CA"}


# ------------------------------------------------------------------------------------------------ agrégation
def aggregate():
    out = ROOT / "data/processed/daily_availability.parquet"
    if out.exists():
        return pd.read_parquet(out)
    feat = pd.read_parquet(ROOT / "data/processed/listings_features.parquet", columns=["city", "id", "room_type", "neighbourhood_cleansed", "scrape_date"])
    rows = []
    for city in sorted(COUNTRY):
        f = next((ROOT / "data/raw" / city).glob("*/calendar.csv.gz"))
        t0 = time.time()
        c = pd.read_csv(f, usecols=["listing_id", "date", "available", "minimum_nights"], dtype={"available": "string"})
        c["unavail"] = (c["available"] == "f").astype("int8")
        # annonces « actives » : au moins une nuit disponible dans l'année (les autres sont bloquées en permanence = fantômes)
        act = c.groupby("listing_id")["unavail"].min().eq(0)
        n_all, n_act = len(act), int(act.sum())
        c = c[c["listing_id"].isin(act[act].index)]
        meta = feat[feat.city == city].set_index("id")[["room_type", "neighbourhood_cleansed"]]
        c = c.join(meta, on="listing_id")
        scrape = pd.to_datetime(feat.loc[feat.city == city, "scrape_date"].iloc[0])
        for seg_name, keys in [("ville", []), ("room_type", ["room_type"])] + ([("quartier", ["neighbourhood_cleansed"])] if city == "paris" else []):
            g = c.groupby(keys + ["date"]).agg(n_active=("unavail", "size"), share_unavail=("unavail", "mean"), min_nights_mean=("minimum_nights", "mean")).reset_index()
            g["segment"] = seg_name + ("" if not keys else ":" + g[keys[0]].astype(str))
            g = g.drop(columns=keys)
            g["city"], g["scrape_date"], g["n_listings_all"], g["n_listings_active"] = city, scrape, n_all, n_act
            rows.append(g)
        print(f"{city}: {n_act:,}/{n_all:,} annonces actives, {time.time() - t0:.0f}s", flush=True)
    df = pd.concat(rows, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df["days_ahead"] = (df["date"] - df["scrape_date"]).dt.days
    df.to_parquet(out, index=False)
    return df


def calendar_features(dates, city, scrape):
    h = holidays.country_holidays(COUNTRY[city], years=sorted({d.year for d in dates}))
    hd = pd.Series(sorted(pd.Timestamp(k) for k in h.keys()))
    x = pd.DataFrame({"date": dates})
    x["dow"] = x.date.dt.dayofweek
    x["is_weekend_night"] = x.dow.isin([4, 5]).astype(int)  # nuit de vendredi ou samedi
    x["month"] = x.date.dt.month
    x["doy_sin"] = np.sin(2 * np.pi * x.date.dt.dayofyear / 365.25)
    x["doy_cos"] = np.cos(2 * np.pi * x.date.dt.dayofyear / 365.25)
    x["is_holiday"] = x.date.isin(hd).astype(int)
    nxt = [(hd[hd >= d].iloc[0] - d).days if (hd >= d).any() else 60 for d in x.date]
    x["days_to_holiday"] = np.clip(nxt, 0, 60)
    x["xmas_ny"] = (((x.date.dt.month == 12) & (x.date.dt.day >= 22)) | ((x.date.dt.month == 1) & (x.date.dt.day <= 3))).astype(int)
    x["days_ahead"] = (x.date - scrape).dt.days
    return x


# ------------------------------------------------------------------------------------------------ prévision
def smape(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return float(np.mean(2 * np.abs(y - p) / (np.abs(y) + np.abs(p) + 1e-9)) * 100)


def forecast_eval(daily, horizon=28, n_splits=5):
    ser = {c: g.sort_values("date").set_index("date") for c, g in daily[daily.segment == "ville"].groupby("city")}
    T = min(len(s) for s in ser.values())
    cuts = [T - horizon * (n_splits - i) for i in range(n_splits)]  # origines successives (fenêtre extensible)
    rows, example = [], {}
    for k, cut in enumerate(cuts):
        # LightGBM global (toutes les villes) : variables calendaires + niveau récent connu à l'origine
        tr_parts, te_parts = [], []
        for c, s in ser.items():
            x = calendar_features(s.index, c, s.scrape_date.iloc[0])
            x["city"] = c
            x["y"] = s.share_unavail.values
            x["lvl7"] = s.share_unavail.iloc[cut - 7:cut].mean()
            x["lvl28"] = s.share_unavail.iloc[max(cut - 28, 0):cut].mean()
            x["slope"] = s.share_unavail.iloc[cut - 14:cut].mean() - s.share_unavail.iloc[cut - 28:cut - 14].mean()
            tr_parts.append(x.iloc[:cut].assign(lvl7=lambda z: z.y.rolling(7, min_periods=1).mean().shift(1).bfill(), lvl28=lambda z: z.y.rolling(28, min_periods=1).mean().shift(1).bfill(),
                                                slope=0.0))
            te_parts.append(x.iloc[cut:cut + horizon])
        tr, te = pd.concat(tr_parts), pd.concat(te_parts)
        cats = pd.CategoricalDtype(sorted(tr["city"].unique()))
        tr["city"], te["city"] = tr["city"].astype(cats), te["city"].astype(cats)
        feats = ["dow", "is_weekend_night", "month", "doy_sin", "doy_cos", "is_holiday", "days_to_holiday", "xmas_ny", "days_ahead", "lvl7", "lvl28", "city"]
        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=10, random_state=SEED, verbose=-1, n_jobs=8).fit(tr[feats], tr["y"])
        te["p_lgb"] = m.predict(te[feats])
        for c, s in ser.items():
            y = s.share_unavail.values
            trn, tst = y[:cut], y[cut:cut + horizon]
            preds = {}
            preds["Naïf saisonnier (semaine précédente)"] = np.tile(trn[-7:], horizon // 7 + 1)[:horizon]
            try:
                ets = ExponentialSmoothing(trn, trend="add", damped_trend=True, seasonal="add", seasonal_periods=7).fit()
                preds["ETS (tendance amortie + saison. hebdo)"] = ets.forecast(horizon)
            except Exception:
                preds["ETS (tendance amortie + saison. hebdo)"] = preds["Naïf saisonnier (semaine précédente)"]
            try:
                sar = SARIMAX(trn, order=(1, 1, 1), seasonal_order=(1, 0, 1, 7)).fit(disp=False, maxiter=60)
                preds["SARIMA(1,1,1)(1,0,1,7)"] = sar.forecast(horizon)
            except Exception:
                preds["SARIMA(1,1,1)(1,0,1,7)"] = preds["Naïf saisonnier (semaine précédente)"]
            preds["LightGBM global (calendrier + niveau récent)"] = te[te.city == c]["p_lgb"].values
            for name, p in preds.items():
                rows.append(dict(origine=k, city=c, modèle=name, mae=float(np.mean(np.abs(tst - p))) * 100, smape=smape(tst, p)))
            if c == "paris" and k == n_splits - 1:
                example = dict(dates=s.index[:cut + horizon], y=y[:cut + horizon], cut=cut, preds=preds)
        print(f"origine {k + 1}/{n_splits} évaluée", flush=True)
    return pd.DataFrame(rows), example


def run():
    daily = aggregate()
    city = daily[daily.segment == "ville"].copy()
    # ---- figure 1 : courbes par ville
    fig, axes = plt.subplots(3, 5, figsize=(20, 9), sharey=True)
    for ax, (c, g) in zip(axes.ravel(), city.groupby("city")):
        g = g.sort_values("date")
        ax.plot(g.date, g.share_unavail * 100, lw=0.8, color="#4c78a8")
        ax.plot(g.date, g.share_unavail.rolling(7, center=True).mean() * 100, lw=1.6, color="#e45756")
        ax.set_title(f"{c} ({g.n_listings_active.iloc[0]:,} annonces actives)", fontsize=9)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
    fig.suptitle("Part de nuits indisponibles par jour (annonces actives) : instantané du calendar, 365 jours vers l'avenir (bleu : jour ; rouge : moyenne mobile 7 j)", fontsize=12)
    fig.supylabel("% de nuits indisponibles")
    plt.tight_layout()
    plt.savefig(FIG / "ts_01_unavailability_by_city.png", dpi=120)
    plt.close()

    # ---- décomposition STL (Paris) + profil hebdomadaire par ville (résidu après retrait de la tendance d'horizon)
    p = city[city.city == "paris"].sort_values("date").set_index("date")["share_unavail"] * 100
    stl = STL(p, period=7, robust=True).fit()
    fig, ax = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
    for a, (s, t) in zip(ax, [(p, "Série observée"), (stl.trend, "Tendance (effet d'horizon + saison longue)"), (stl.seasonal, "Saisonnalité hebdomadaire"), (stl.resid, "Résidu (événements)")]):
        a.plot(s, lw=1)
        a.set_title(t, fontsize=10)
        a.set_ylabel("points de %")
    fig.suptitle("Paris : décomposition STL de la part de nuits indisponibles (période 7 jours)", y=1.0)
    plt.tight_layout()
    plt.savefig(FIG / "ts_02_stl_paris.png", dpi=120)
    plt.close()
    prof = {}
    for c, g in city.groupby("city"):
        g = g.sort_values("date").set_index("date")["share_unavail"] * 100
        s = STL(g, period=7, robust=True).fit()
        d = (s.seasonal + s.resid).groupby(g.index.dayofweek).mean()
        prof[c] = d
    prof = pd.DataFrame(prof).T
    prof.columns = ["lun", "mar", "mer", "jeu", "ven", "sam", "dim"]
    prof.to_csv(ROOT / "reports" / "ts_weekday_profile.csv")
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(prof, annot=True, fmt=".1f", cmap="RdBu_r", center=0, cbar_kws={"label": "écart à la tendance, points de %"}, ax=ax)
    ax.set(title="Profil hebdomadaire : nuits plus (rouge) ou moins (bleu) indisponibles que la tendance", xlabel="Jour de la nuit", ylabel="Ville")
    plt.tight_layout()
    plt.savefig(FIG / "ts_03_weekday_profile.png", dpi=130)
    plt.close()

    # ---- événements : jours dont le résidu STL est le plus fort / faible (par ville)
    ev = []
    for c, g in city.groupby("city"):
        g = g.sort_values("date").set_index("date")["share_unavail"] * 100
        r = STL(g, period=7, robust=True).fit().resid
        r = r.iloc[2:-21]  # on écarte les bords de la fenêtre (1er jour : annonces vues le jour de collecte ; 3 dernières semaines : calendriers non ouverts)
        z = (r - r.median()) / (r.abs().median() * 1.4826)
        hol = holidays.country_holidays(COUNTRY[c], years=[2026, 2027])
        for d, v in z.sort_values(ascending=False).head(4).items():
            ev.append(dict(city=c, date=d.date(), type="pic de demande", z=float(v), residu_pts=float(r[d]), jour_férié=hol.get(d.date(), "")))
        for d, v in z.sort_values().head(3).items():
            ev.append(dict(city=c, date=d.date(), type="creux", z=float(v), residu_pts=float(r[d]), jour_férié=hol.get(d.date(), "")))
    pd.DataFrame(ev).to_csv(ROOT / "reports" / "ts_events.csv", index=False)

    # ---- prévision : fenêtre glissante, 5 origines de 28 jours
    res, ex = forecast_eval(daily)
    res.to_csv(ROOT / "reports" / "ts_forecast_scores.csv", index=False)
    s = res.groupby("modèle").agg(mae=("mae", "mean"), smape=("smape", "mean")).sort_values("mae")
    base = s.loc["Naïf saisonnier (semaine précédente)"]
    s["gain_mae_vs_naïf_%"] = (1 - s.mae / base.mae) * 100
    s.to_csv(ROOT / "reports" / "ts_forecast_summary.csv")
    fig, ax = plt.subplots(figsize=(13, 5))
    d = ex["dates"]
    ax.plot(d[-120:], ex["y"][-120:] * 100, color="k", lw=1.2, label="observé")
    for name, p in ex["preds"].items():
        ax.plot(d[ex["cut"]:], np.asarray(p) * 100, lw=1.3, label=name)
    ax.axvline(d[ex["cut"]], color="grey", ls="--")
    ax.set(title="Paris : prévision à 28 jours (dernière origine de la validation glissante)", xlabel="Date", ylabel="% de nuits indisponibles")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "ts_04_forecast_paris.png", dpi=130)
    plt.close()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=res, y="modèle", x="mae", order=s.index, ax=ax, color="#4c78a8", errorbar="sd")
    ax.set(title="Erreur de prévision à 28 jours (MAE, moyenne sur 15 villes x 5 origines)", xlabel="MAE (points de % d'indisponibilité)", ylabel="Modèle")
    plt.tight_layout()
    plt.savefig(FIG / "ts_05_forecast_scores.png", dpi=130)
    plt.close()
    # ---- Paris : arrondissements (heatmap hebdomadaire)
    q = daily[(daily.city == "paris") & daily.segment.str.startswith("quartier")].copy()
    q["quartier"] = q.segment.str.split(":").str[1]
    q["mois"] = q.date.dt.to_period("M").astype(str)
    pv = q.pivot_table(index="quartier", columns="mois", values="share_unavail", aggfunc="mean") * 100
    fig, ax = plt.subplots(figsize=(13, 7))
    sns.heatmap(pv, cmap="viridis_r", cbar_kws={"label": "% nuits indisponibles"}, ax=ax)
    ax.set(title="Paris : indisponibilité moyenne par arrondissement et par mois de séjour", xlabel="Mois de la nuit", ylabel="Arrondissement")
    plt.tight_layout()
    plt.savefig(FIG / "ts_06_paris_neighbourhood_month.png", dpi=130)
    plt.close()
    print(s.round(3).to_string())


if __name__ == "__main__":
    run()
