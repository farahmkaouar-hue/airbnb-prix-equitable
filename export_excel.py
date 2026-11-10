#!/usr/bin/env python3
"""Exporte la base globale des annonces dans un seul classeur Excel.

Par défaut : une seule feuille 'annonces' (brutes ou nettoyées, toutes villes), identique à la base de travail.
Avec --with-extra-sheets : ajoute LISEZ-MOI et resume_villes.
Les avis ne sont PAS exportés : Excel est limité à 1 048 576 lignes par feuille et la base en compte des millions ;
ils restent dans reviews_all.parquet / reviews_clean.parquet.

Exemples :
    python export_excel.py                      # annonces brutes -> data/processed/airbnb_global.xlsx
    python export_excel.py --source clean       # annonces nettoyées
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import xlsxwriter

EXCEL_MAX_ROWS = 1_048_576
EXCEL_MAX_CELL = 32_767  # limite Excel
TEXT_LIMIT = EXCEL_MAX_CELL  # pas de troncature autre que la limite d'Excel
ILLEGAL_XML = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
CHUNK = 20_000


def sanitize(df: pd.DataFrame) -> pd.DataFrame:
    """Texte compatible Excel : caractères de contrôle retirés, textes tronqués à TEXT_LIMIT caractères."""
    df = df.copy()
    for col in df.columns[df.dtypes.map(lambda t: t == object or str(t).startswith("str") or str(t) == "string")]:
        s = df[col].astype("string").str.replace(ILLEGAL_XML, "", regex=True).str.slice(0, TEXT_LIMIT)
        df[col] = s
    for col in df.columns[df.dtypes.map(lambda t: "datetime" in str(t))]:
        df[col] = df[col].dt.strftime("%Y-%m-%d")
    for col in df.columns[df.dtypes.map(lambda t: str(t) in ("boolean", "bool"))]:
        df[col] = df[col].astype("Int8")  # 1 / 0, lisible et filtrable dans Excel
    for col in df.columns[df.dtypes.map(lambda t: str(t) == "category")]:
        df[col] = df[col].astype("string")
    return df


def write_rows(ws, df: pd.DataFrame, fmt_header, start_row: int = 0) -> None:
    ws.write_row(start_row, 0, list(df.columns), fmt_header)
    row = start_row + 1
    for i in range(0, len(df), CHUNK):
        block = df.iloc[i:i + CHUNK].astype(object).where(df.iloc[i:i + CHUNK].notna(), None).to_numpy()
        for values in block:
            for col, v in enumerate(values):
                if v is not None:
                    ws.write(row, col, v)
            row += 1


def add_readme(wb, fmt, n_rows: int, n_cols: int, cities: list[str], source: str) -> None:
    ws = wb.add_worksheet("LISEZ-MOI")
    kind = "brutes (avant nettoyage)" if source == "all" else "nettoyées"
    lines = [
        "Base globale Inside Airbnb",
        f"Annonces {kind} : {n_rows:,} lignes, {n_cols} colonnes, {len(cities)} villes : {', '.join(cities)}.",
        "Feuille 'annonces' : une ligne par annonce, toutes villes ; colonne 'city' pour filtrer.",
        "Feuille 'resume_villes' : effectifs et prix médian par ville (devise locale, non convertie).",
        "Les avis ne sont pas dans ce classeur (limite Excel d'un million de lignes) : voir les fichiers .parquet.",
        f"Les cellules de texte sont limitées à {TEXT_LIMIT:,} caractères par Excel (aucune donnée tronquée en pratique pour cette base).",
        "Source : Inside Airbnb, https://insideairbnb.com/get-the-data/ , licence CC BY 4.0.",
        "Attribution : Inside Airbnb ; données fusionnées par ce projet. Projet indépendant, non affilié à Airbnb.",
    ]
    ws.set_column(0, 0, 130)
    for i, line in enumerate(lines):
        ws.write(i, 0, line, fmt if i == 0 else None)


def city_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("city")
    out = pd.DataFrame({"pays": g["country"].first(), "devise": g["currency"].first(),
                        "annonces": g.size(), "prix_median": g["price"].median().round(2),
                        "pct_prix_renseigne": (g["price"].apply(lambda s: s.notna().mean()) * 100).round(1),
                        "date_collecte": g["scrape_date"].first().astype(str).str.slice(0, 10)})
    return out.sort_values("annonces", ascending=False).reset_index()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--source", choices=["all", "clean"], default="all")
    p.add_argument("--output", type=Path)
    p.add_argument("--with-extra-sheets", action="store_true",
                   help="Ajoute les feuilles LISEZ-MOI et resume_villes (absentes de la base de travail)")
    args = p.parse_args()
    src = args.processed_dir / ("listings_all.parquet" if args.source == "all" else "listings_clean.parquet")
    out = args.output or args.processed_dir / ("airbnb_global.xlsx" if args.source == "all" else "airbnb_global_clean.xlsx")

    df = pd.read_parquet(src)
    summary = city_summary(df)
    df = sanitize(df)
    wb = xlsxwriter.Workbook(out, {"constant_memory": True, "strings_to_formulas": False,
                                   "strings_to_urls": False, "nan_inf_to_errors": True})
    wb.use_zip64()
    bold = wb.add_format({"bold": True})
    if args.with_extra_sheets:
        add_readme(wb, bold, len(df), df.shape[1], sorted(df["city"].unique()), args.source)
        ws = wb.add_worksheet("resume_villes")
        write_rows(ws, summary, bold)
        ws.set_column(0, len(summary.columns) - 1, 18)
    n_sheets = -(-len(df) // (EXCEL_MAX_ROWS - 1))
    for k in range(n_sheets):
        part = df.iloc[k * (EXCEL_MAX_ROWS - 1):(k + 1) * (EXCEL_MAX_ROWS - 1)]
        ws = wb.add_worksheet("annonces" if n_sheets == 1 else f"annonces_{k + 1}")
        write_rows(ws, part, bold)
        ws.freeze_panes(1, 0)
    wb.close()
    print(f"{out} : {len(df):,} lignes sur {n_sheets} feuille(s) d'annonces, {out.stat().st_size / 1e6:.0f} Mo")


if __name__ == "__main__":
    main()
