"""
scripts/enrich_ticker_names.py

One-time helper: adds a `company_name` column to the ticker master list.

Google News is searched by company name, not ticker symbol. That is the
whole point of the switch away from marketaux -- searching "ACC" against a
global news index returns American Campus Communities (a US student-housing
REIT), whereas searching "ACC Ltd" returns the Indian cement company. Ticker
symbols are not unique across exchanges; company names very nearly are.

Names come from yfinance's `longName`. Run once and commit the result --
the names don't change, and re-running costs ~150 Yahoo requests you don't
need to spend.

    python scripts/enrich_ticker_names.py
    python scripts/enrich_ticker_names.py --sleep 3    # slower, if throttled
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_config


def fetch_company_name(symbol: str, suffix: str) -> str:
    """Best-effort company name. Returns "" on failure -- the caller falls
    back to the bare symbol, which still produces a usable (if noisier)
    search query."""
    try:
        info = yf.Ticker(f"{symbol}{suffix}").info
        return info.get("longName") or info.get("shortName") or ""
    except Exception as e:
        print(f"  {symbol}: lookup failed ({type(e).__name__})")
        return ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sleep", type=float, default=2.0,
                        help="Seconds between Yahoo requests (default 2.0)")
    args = parser.parse_args()

    config = load_config()
    path = REPO_ROOT / config["tickers"]["master_list_path"]
    suffix = config["tickers"]["suffix"]

    df = pd.read_csv(path)
    if "company_name" in df.columns:
        missing = df["company_name"].isna().sum()
        print(f"company_name column already present ({missing} still blank)")
        if missing == 0:
            print("Nothing to do.")
            return 0
    else:
        df["company_name"] = pd.NA

    total = len(df)
    resolved = 0

    for i, row in df.iterrows():
        if pd.notna(row.get("company_name")) and str(row["company_name"]).strip():
            resolved += 1
            continue

        symbol = row["symbol"]
        name = fetch_company_name(symbol, suffix)
        if name:
            df.at[i, "company_name"] = name
            resolved += 1
            print(f"[{i + 1:>3}/{total}] {symbol:<14} -> {name}")
        else:
            df.at[i, "company_name"] = symbol
            print(f"[{i + 1:>3}/{total}] {symbol:<14} -> (no name, using symbol)")

        df.to_csv(path, index=False)
        time.sleep(args.sleep)

    print(f"\nDone: {resolved}/{total} names resolved -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
