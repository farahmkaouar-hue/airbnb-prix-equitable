"""Phase 0 : audit des données brutes. Écrit reports/phase0_audit.md."""
import pandas as pd
import pyarrow.parquet as pq

from src.data import ROOT, load_config, load_listings

cfg = load_config()
df = load_listings(cfg)
out = ["# Phase 0 : audit des données brutes\n"]

out.append(f"## Annonces\n- shape : {df.shape[0]:,} lignes x {df.shape[1]} colonnes")
out.append(f"- mémoire (pandas) : {df.memory_usage(deep=True).sum() / 1e6:,.0f} Mo")
out.append(f"- doublons sur `id` : {df['id'].duplicated().sum():,} ; doublons (city, id) : {df.duplicated(['city', 'id']).sum():,}")
out.append("\n### Types\n" + df.dtypes.astype(str).value_counts().to_frame("nb_colonnes").to_markdown())
out.append("\n### Dtype par colonne\n" + df.dtypes.astype(str).to_frame("dtype").T.to_markdown())

d = pd.to_datetime(df["scrape_date"])
out.append(f"\n### Dates de collecte\n- plage : {d.min().date()} → {d.max().date()}")
out.append("- nb de dates distinctes par ville : " + str(df.groupby("city")["scrape_date"].nunique().max()) + " (max)")

g = df.groupby("city").agg(country=("country", "first"), devise=("currency", lambda s: s.mode().iat[0] if s.notna().any() else None),
                           annonces=("id", "size"), date=("scrape_date", "first"))
out.append("\n### Annonces par ville\n" + g.sort_values("annonces", ascending=False).to_markdown())

# colonnes 100 % vides par ville
empty = df.groupby("city").apply(lambda x: x.columns[x.isna().all()].tolist(), include_groups=False)
out.append("\n### Colonnes 100 % vides par ville\n" + "\n".join(f"- {c} : {v}" for c, v in empty.items() if v) or "aucune")
miss = df.isna().mean().sort_values(ascending=False)
out.append("\n### Colonnes les plus manquantes (global)\n" + (miss[miss > 0.3] * 100).round(1).to_frame("% manquant").to_markdown())

# types de colonnes: price brut
out.append("\n### Échantillon colonnes sensibles\n" + df[["price", "currency", "host_is_superhost", "host_response_rate", "bathrooms_text", "amenities"]]
           .head(3).to_markdown() if {"price", "amenities"} <= set(df.columns) else "colonnes brutes attendues absentes")
sh = df["host_is_superhost"].value_counts(dropna=False) if "host_is_superhost" in df else None
if sh is not None:
    out.append("\n### host_is_superhost\n" + sh.to_frame().to_markdown())

# avis
m = pq.ParquetFile(ROOT / cfg["paths"]["reviews_raw"]).metadata
out.append(f"\n## Avis\n- {m.num_rows:,} lignes x {m.num_columns} colonnes (fichier {ROOT.joinpath(cfg['paths']['reviews_raw']).stat().st_size / 1e9:.2f} Go)")
rv = pd.read_parquet(ROOT / cfg["paths"]["reviews_raw"], columns=["city", "date"])
rv["date"] = pd.to_datetime(rv["date"], errors="coerce")
out.append(f"- plage des dates d'avis : {rv['date'].min().date()} → {rv['date'].max().date()}")
out.append("\n### Avis par ville\n" + rv.groupby("city").size().to_frame("avis").to_markdown())

# calendar
cal = list((ROOT / cfg["paths"]["raw"]).glob("*/*/calendar*"))
out.append(f"\n## Calendar\n- fichiers calendar trouvés dans data/raw : {len(cal)}")

(ROOT / "reports" / "phase0_audit.md").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))
