#!/usr/bin/env python3
"""Télécharge les données Inside Airbnb (listings / reviews / calendar) pour une liste de villes.

Exemples :
    python download_airbnb.py --cities Paris          # test sur une ville
    python download_airbnb.py                          # les 15 villes, collecte la plus récente
    python download_airbnb.py --with-calendar --all-dates
"""
from __future__ import annotations

import argparse
import csv
import gzip
import logging
import random
import re
import sys
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE_URL = "https://insideairbnb.com/get-the-data/"
DEFAULT_UA = "InsideAirbnbResearchDownloader/1.0 (personal ML project; polite: 1-2 s delay)"

# Nom affiché -> (slug pays, slug ville) tels qu'ils apparaissent dans les URL du site.
CITIES: dict[str, tuple[str, str]] = {
    "Paris": ("france", "paris"),
    "Lyon": ("france", "lyon"),
    "Barcelona": ("spain", "barcelona"),
    "Madrid": ("spain", "madrid"),
    "London": ("united-kingdom", "london"),
    "Amsterdam": ("the-netherlands", "amsterdam"),
    "Berlin": ("germany", "berlin"),
    "Rome": ("italy", "rome"),
    "Lisbon": ("portugal", "lisbon"),
    "New York City": ("united-states", "new-york-city"),
    "Los Angeles": ("united-states", "los-angeles"),
    "Toronto": ("canada", "toronto"),
    "Sydney": ("australia", "sydney"),
    "Tokyo": ("japan", "tokyo"),
    "Mexico City": ("mexico", "mexico-city"),
}

URL_RE = re.compile(
    r"https://data\.insideairbnb\.com/(?P<country>[^/\"'<>\s]+)/(?P<region>[^/\"'<>\s]+)/"
    r"(?P<city>[^/\"'<>\s]+)/(?P<date>\d{4}-\d{2}-\d{2})/data/"
    r"(?P<kind>listings|reviews|calendar)\.csv\.gz"
)

log = logging.getLogger("airbnb")


@dataclass(frozen=True)
class Entry:
    city: str  # slug de la ville
    country: str  # slug du pays
    date: str
    kind: str  # listings | reviews | calendar
    url: str


# --------------------------------------------------------------------------- réseau

class Client:
    """Session HTTP polie : User-Agent explicite, pause 1-2 s, retries à backoff exponentiel."""

    def __init__(self, user_agent: str, retries: int = 5, pause: tuple[float, float] = (1.0, 2.0)):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent
        self.retries = retries
        self.pause = pause

    def get(self, url: str, **kwargs) -> requests.Response:
        for attempt in range(1, self.retries + 1):
            time.sleep(random.uniform(*self.pause))
            try:
                resp = self.session.get(url, timeout=(15, 120), **kwargs)
                if resp.status_code in (404, 403):  # définitif : inutile de réessayer
                    resp.raise_for_status()
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise requests.HTTPError(f"HTTP {resp.status_code}", response=resp)
                resp.raise_for_status()
                return resp
            except requests.RequestException as exc:
                status = getattr(exc.response, "status_code", None)
                if status in (403, 404) or attempt == self.retries:
                    raise
                wait = 2 ** attempt + random.random()
                log.warning("Echec %s (%s), nouvel essai %d/%d dans %.0fs", url, exc, attempt, self.retries, wait)
                time.sleep(wait)
        raise RuntimeError("unreachable")


# --------------------------------------------------------------------------- index

def entries_from_text(text: str) -> list[Entry]:
    seen: dict[str, Entry] = {}
    for m in URL_RE.finditer(text):
        seen[m.group(0)] = Entry(m["city"], m["country"], m["date"], m["kind"], m.group(0))
    return list(seen.values())


def entries_from_bs4(html: str) -> list[Entry]:
    hrefs = " ".join(a["href"] for a in BeautifulSoup(html, "html.parser").find_all("a", href=True))
    return entries_from_text(hrefs)


def html_from_playwright(url: str, user_agent: str) -> str:
    from playwright.sync_api import sync_playwright  # import tardif : dépendance optionnelle

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(user_agent=user_agent)
        page.goto(url, wait_until="networkidle")
        html = page.content()
        browser.close()
    return html


def build_index(client: Client, user_agent: str) -> list[Entry]:
    """requests + regex, puis BeautifulSoup, puis Playwright en dernier recours."""
    log.info("Téléchargement de la page %s", BASE_URL)
    html = client.get(BASE_URL).content.decode("utf-8", errors="replace")
    for name, parser in (("regex", entries_from_text), ("BeautifulSoup", entries_from_bs4)):
        entries = parser(html)
        if entries:
            log.info("Index construit via %s : %d fichiers", name, len(entries))
            return entries
    log.warning("Aucun lien trouvé en HTML statique, tentative Playwright")
    entries = entries_from_bs4(html_from_playwright(BASE_URL, user_agent))
    log.info("Index construit via Playwright : %d fichiers", len(entries))
    return entries


def save_index(entries: list[Entry], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(entries[0])) if entries else ["url"])
        writer.writeheader()
        writer.writerows(asdict(e) for e in entries)


# --------------------------------------------------------------------------- sélection

def select_entries(index: list[Entry], wanted: dict[str, tuple[str, str]],
                   kinds: set[str], all_dates: bool) -> tuple[list[Entry], list[str]]:
    """Retourne (entrées à télécharger, noms de villes introuvables)."""
    selected, missing = [], []
    for name, (country, slug) in wanted.items():
        city_entries = [e for e in index if e.city == slug and e.country == country and e.kind in kinds]
        if not any(e.kind == "listings" for e in city_entries):
            missing.append(name)
            continue
        dates = sorted({e.date for e in city_entries if e.kind == "listings"}, reverse=True)
        keep = set(dates) if all_dates else {dates[0]}
        selected += [e for e in city_entries if e.date in keep]
    return selected, missing


def target_path(root: Path, e: Entry) -> Path:
    return root / e.city / e.date / f"{e.kind}.csv.gz"


# --------------------------------------------------------------------------- intégrité

def is_valid(path: Path) -> bool:
    """gzip complet (CRC vérifié à la fin du flux) et CSV lisible par pandas."""
    try:
        with gzip.open(path, "rb") as f:
            while f.read(1 << 20):
                pass
        pd.read_csv(path, nrows=100)
        return True
    except (OSError, EOFError, zlib.error, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return False


def download_file(client: Client, e: Entry, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    resp = client.get(e.url, stream=True)
    total = int(resp.headers.get("Content-Length", 0)) or None
    with tmp.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True, leave=False,
                                   desc=f"{e.city}/{e.kind}") as bar:
        for chunk in resp.iter_content(1 << 20):
            f.write(chunk)
            bar.update(len(chunk))
    tmp.replace(dest)


def fetch_entry(client: Client, e: Entry, root: Path) -> str:
    """Retourne 'skipped', 'downloaded' ou 'failed'."""
    dest = target_path(root, e)
    if dest.exists() and is_valid(dest):
        log.info("Déjà présent et valide : %s", dest)
        return "skipped"
    if dest.exists():
        log.warning("Fichier corrompu, retéléchargement : %s", dest)
        dest.unlink()
    try:
        download_file(client, e, dest)
    except requests.RequestException as exc:
        log.error("Echec téléchargement %s : %s", e.url, exc)
        return "failed"
    if not is_valid(dest):
        log.error("Fichier invalide après téléchargement : %s", dest)
        dest.unlink(missing_ok=True)
        return "failed"
    log.info("Téléchargé : %s (%.1f Mo)", dest, dest.stat().st_size / 1e6)
    return "downloaded"


# --------------------------------------------------------------------------- CLI

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cities", nargs="+", metavar="VILLE",
                   help="Sous-ensemble de villes (défaut : les 15). Ex : --cities Paris Lyon")
    p.add_argument("--all-dates", action="store_true", help="Toutes les collectes, pas seulement la plus récente")
    p.add_argument("--with-calendar", action="store_true", help="Télécharge aussi calendar.csv.gz (très volumineux)")
    p.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--log-file", type=Path, default=Path("logs/download.log"))
    p.add_argument("--user-agent", default=DEFAULT_UA)
    p.add_argument("--retries", type=int, default=5)
    p.add_argument("--dry-run", action="store_true", help="Construit l'index et affiche la sélection sans télécharger")
    return p.parse_args()


def setup_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(path, encoding="utf-8")])


def resolve_cities(names: list[str] | None) -> dict[str, tuple[str, str]]:
    if not names:
        return CITIES
    lookup = {k.lower(): k for k in CITIES}
    unknown = [n for n in names if n.lower() not in lookup]
    if unknown:
        sys.exit(f"Villes inconnues : {unknown}. Disponibles : {list(CITIES)}")
    return {lookup[n.lower()]: CITIES[lookup[n.lower()]] for n in names}


def print_summary(selected: list[Entry], missing: list[str], results: dict[str, int],
                  failed: list[Entry], root: Path) -> None:
    found = sorted({e.city for e in selected})
    size = sum(target_path(root, e).stat().st_size for e in selected if target_path(root, e).exists())
    print("\n===== Récapitulatif =====")
    print(f"Villes trouvées   ({len(found)}) : {', '.join(found) or '-'}")
    print(f"Villes manquantes ({len(missing)}) : {', '.join(missing) or '-'}")
    print(f"Fichiers : {results['downloaded']} téléchargés, {results['skipped']} déjà présents, "
          f"{results['failed']} en échec")
    for e in failed:
        print(f"  ECHEC : {e.url}")
    print(f"Taille totale sélection sur disque : {size / 1e6:.1f} Mo ({size / 1e9:.2f} Go)")
    print("Journal : logs/download.log")


def main() -> int:
    args = parse_args()
    setup_logging(args.log_file)
    client = Client(args.user_agent, retries=args.retries)
    index = build_index(client, args.user_agent)
    save_index(index, args.output_dir / "index.csv")

    kinds = {"listings", "reviews"} | ({"calendar"} if args.with_calendar else set())
    selected, missing = select_entries(index, resolve_cities(args.cities), kinds, args.all_dates)
    if missing:
        log.warning("Villes introuvables : %s", missing)
    if args.dry_run:
        for e in selected:
            print(e.url)
        print(f"Manquantes : {missing}")
        return 0

    results = {"downloaded": 0, "skipped": 0, "failed": 0}
    failed: list[Entry] = []
    for e in tqdm(selected, desc="Fichiers", unit="fichier"):
        status = fetch_entry(client, e, args.output_dir)
        results[status] += 1
        if status == "failed":
            failed.append(e)
    print_summary(selected, missing, results, failed, args.output_dir)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
