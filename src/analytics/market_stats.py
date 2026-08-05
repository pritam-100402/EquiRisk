"""
src/analytics/market_stats.py

Computes beta, Sharpe ratio, and annualized volatility for a selected
lookback window. These are NOT part of the Spark ETL/feature table --
they're computed on demand in the dashboard from:
  - the ticker's own price history (already in S3, via the feature table)
  - a live benchmark index pull (yfinance) for beta

Kept separate from feature_engineering.py because these are
presentation-layer analytics computed per user-selected duration, not
fixed rolling-window features baked into the training data.
"""

import logging

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger("equirisk.analytics.market_stats")

TRADING_DAYS_PER_YEAR = 252


def annualized_volatility(daily_returns: pd.Series) -> float:
    """Std dev of daily returns, annualized by sqrt(252)."""
    if daily_returns.empty or daily_returns.isna().all():
        return float("nan")
    return float(daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def sharpe_ratio(daily_returns: pd.Series, risk_free_rate_annual: float) -> float:
    """Annualized Sharpe ratio: (annualized mean return - risk-free rate)
    / annualized volatility. Uses a flat risk-free rate assumption from
    config.yaml rather than a live bond yield feed -- reasonable
    documented simplification -- see the README's limitations section."""
    if daily_returns.empty or daily_returns.isna().all():
        return float("nan")
    mean_annual_return = float(daily_returns.mean() * TRADING_DAYS_PER_YEAR)
    vol = annualized_volatility(daily_returns)
    if vol == 0 or np.isnan(vol):
        return float("nan")
    return (mean_annual_return - risk_free_rate_annual) / vol


def fetch_benchmark_returns(benchmark_ticker: str, start_date, end_date) -> pd.Series:
    """Live-fetches the benchmark index's daily returns for the given
    date range. Not cached at this layer -- the dashboard page calling
    this should wrap it in st.cache_data, since Streamlit's cache
    survives across reruns within a session and this doesn't need to
    hit Yahoo on every widget interaction."""
    try:
        bench = yf.Ticker(benchmark_ticker).history(start=start_date, end=end_date, interval="1d")
        if bench.empty:
            logger.warning(f"No benchmark data returned for {benchmark_ticker}")
            return pd.Series(dtype=float)
        bench = bench.reset_index()
        bench["date"] = pd.to_datetime(bench["Date"]).dt.tz_localize(None)
        bench = bench.set_index("date")
        return bench["Close"].pct_change().dropna()
    except Exception as e:
        logger.error(f"Failed to fetch benchmark {benchmark_ticker}: {e}")
        return pd.Series(dtype=float)


def compute_beta(stock_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Beta = cov(stock, benchmark) / var(benchmark), aligned on date.
    Returns NaN if there's not enough overlapping data to compute a
    meaningful covariance (e.g. benchmark fetch failed).

    Both indices are normalized to pandas Timestamp (midnight) before
    aligning -- without this, a stock_returns index made of plain
    Python `date` objects (which is what you can get back from a
    Spark-written DATE column depending on the pyarrow/pandas version)
    won't match a benchmark_returns index of pandas Timestamps even
    though they represent the same calendar days. pd.concat's inner
    join then silently finds zero overlap and this returns NaN with no
    error -- exactly the "beta always N/A" symptom this fixes."""
    stock_returns = stock_returns.copy()
    stock_returns.index = pd.to_datetime(stock_returns.index).normalize()

    benchmark_returns = benchmark_returns.copy()
    benchmark_returns.index = pd.to_datetime(benchmark_returns.index).normalize()

    aligned = pd.concat([stock_returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(aligned) < 20:
        logger.warning(f"Only {len(aligned)} overlapping dates between stock and benchmark -- beta unreliable")
        return float("nan")
    aligned.columns = ["stock", "benchmark"]
    covariance = aligned["stock"].cov(aligned["benchmark"])
    benchmark_variance = aligned["benchmark"].var()
    if benchmark_variance == 0:
        return float("nan")
    return float(covariance / benchmark_variance)


def compute_window_stats(
    ticker_df: pd.DataFrame,
    window_days: int,
    benchmark_returns: pd.Series,
    risk_free_rate_annual: float,
) -> dict:
    """Main entrypoint: given a ticker's full price history (must have
    'date', 'close', 'daily_return' columns, sorted ascending), a
    lookback window in trading days, and an ALREADY-FETCHED benchmark
    returns series (see fetch_benchmark_returns -- fetch it once,
    cached, in the caller rather than here), returns current price,
    beta, Sharpe ratio, and annualized volatility for that window.

    benchmark_returns is passed in rather than fetched here on purpose:
    this function may be called many times per page render (once per
    duration change, once per ticker switch); fetching from Yahoo on
    every single call gets rate-limited fast. Fetch once per session
    (st.cache_data in the dashboard) and reuse across calls.
    """
    window_df = ticker_df.sort_values("date").tail(window_days)
    if window_df.empty:
        return {"current_price": float("nan"), "beta": float("nan"), "sharpe": float("nan"), "volatility": float("nan")}

    current_price = float(window_df["close"].iloc[-1])
    vol = annualized_volatility(window_df["daily_return"])
    sharpe = sharpe_ratio(window_df["daily_return"], risk_free_rate_annual)

    stock_returns = window_df.set_index("date")["daily_return"]
    beta = compute_beta(stock_returns, benchmark_returns)

    return {
        "current_price": current_price,
        "beta": beta,
        "sharpe": sharpe,
        "volatility": vol,
    }