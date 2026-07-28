"""
src/ingestion/fetch_prices.py

Pulls 5-year OHLCV history for each Nifty150 midcap ticker via yfinance
and writes the raw, untouched response straight to S3 (raw/prices/...).
No transformation happens here -- that's the ETL stage's job.
"""

import logging
import time

import pandas as pd
import yfinance as yf

from src.utils.s3_io import write_parquet_s3, dated_key

from src.utils.config import load_config as _load_config

logger = logging.getLogger("equirisk.ingestion.prices")




def _load_ticker_list(config: dict) -> list:
    """Reads the master ticker CSV (local, versioned metadata file --
    not "data" in the no-local-storage sense, just a small reference
    table checked into the repo). Expects a 'symbol' column."""
    path = config["tickers"]["master_list_path"]
    df = pd.read_csv(path)
    suffix = config["tickers"]["suffix"]
    return [f"{sym}{suffix}" for sym in df["symbol"].tolist()]


def fetch_ticker_price_history(ticker: str, years: int) -> pd.DataFrame:
    """Fetch OHLCV history for one ticker. Returns empty DataFrame on
    failure rather than raising -- one bad ticker shouldn't kill the
    whole ingestion run."""
    try:
        period = f"{years}y"
        df = yf.Ticker(ticker).history(period=period, interval="1d")
        if df.empty:
            logger.warning(f"No data returned for {ticker}")
            return pd.DataFrame()
        df = df.reset_index()
        df["ticker"] = ticker
        return df
    except Exception as e:
        logger.error(f"Failed to fetch {ticker}: {e}")
        return pd.DataFrame()


def fetch_all_tickers_prices(config_path: str = None, sleep_sec: float = 2.0) -> None:
    """Main ingestion entrypoint, called by the orchestrator. Fetches
    every ticker in the master list and writes each one's raw OHLCV to
    S3 as parquet under raw/prices/{ticker}/{date}.parquet.

    sleep_sec throttles requests. Yahoo rate-limits aggressively on its
    undocumented endpoints and returns HTTP 429 once tripped, with a
    cooldown that retrying only extends. 0.5s was not enough across 150
    tickers; 2.0s costs about five minutes total, which is nothing against
    a full pipeline run.
    """
    config = _load_config(config_path)
    tickers = _load_ticker_list(config)
    years = config["ingestion"]["price_history_years"]
    raw_prefix = config["s3"]["paths"]["raw_prices"]
    bucket = config["s3"]["bucket"]

    logger.info(f"Fetching {years}-year price history for {len(tickers)} tickers")

    success_count = 0
    for ticker in tickers:
        df = fetch_ticker_price_history(ticker, years)
        if df.empty:
            continue

        key = dated_key(raw_prefix, ticker.replace(".NS", ""), ext="parquet")
        write_parquet_s3(df, key, bucket)
        success_count += 1
        time.sleep(sleep_sec)

    logger.info(f"Price ingestion complete: {success_count}/{len(tickers)} tickers succeeded")

    if success_count == 0:
        raise RuntimeError("Price ingestion failed for all tickers -- check network/yfinance status")


if __name__ == "__main__":
    from src.utils.logging_config import setup_logging

    setup_logging()
    fetch_all_tickers_prices()