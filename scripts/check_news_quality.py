"""
scripts/check_news_quality.py

Eyeball the news query against the hardest tickers before committing to a
full 150-ticker run.

Short acronym tickers are where a name-based search goes wrong: ACC, BSE,
MRF, SAIL, OIL and similar collide with unrelated organisations and
ordinary words. This prints what the feed actually returns for a sample so
the precision can be judged rather than assumed.

    python scripts/check_news_quality.py
    python scripts/check_news_quality.py --tickers ACC BSE MRF
"""

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from urllib.parse import unquote

from src.ingestion.fetch_news import build_query, fetch_ticker_news
from src.utils.config import load_config

DEFAULT_SAMPLE = ["ACC", "BSE", "MRF", "SAIL", "OIL", "TATACOMM", "BHARATFORG"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="*", default=DEFAULT_SAMPLE)
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()

    config = load_config()
    path = REPO_ROOT / config["tickers"]["master_list_path"]
    df = pd.read_csv(path)

    if "company_name" not in df.columns:
        print("No company_name column yet. Run: python scripts/enrich_ticker_names.py")
        return 1

    names = dict(zip(df["symbol"], df["company_name"].fillna(df["symbol"])))
    lookback = config["ingestion"]["news"]["lookback_days"]

    for sym in args.tickers:
        name = names.get(sym, sym)
        print("=" * 72)
        print(f"{sym}  ({name})")
        print("query:", unquote(build_query(name, lookback).split("q=")[1].split("&")[0]))
        print("-" * 72)

        payload = fetch_ticker_news(sym, name, lookback, args.limit)
        if not payload["data"]:
            print("  (no articles)")
        for a in payload["data"]:
            print(f"  [{a['publisher'] or '?':<22}] {a['title'][:78]}")
        print()
        time.sleep(1.5)

    print("=" * 72)
    print("Scan for headlines that clearly aren't about the Indian company.")
    print("A few misses are tolerable -- sentiment averages over many headlines,")
    print("so occasional noise dilutes rather than biases. Systematic misses")
    print("(most results wrong for a ticker) are not tolerable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
