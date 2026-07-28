"""
src/analytics/fundamentals.py

Live fetch of company fundamentals (P/E, net income, profit margin,
debt-to-equity) via yfinance's `.info` dict. These are slow-changing
values (quarterly/annual), so this is intentionally NOT part of the
daily Spark ETL -- it's fetched on demand and cached by the dashboard
layer (st.cache_data with a long TTL, see config.yaml's
analytics.fundamentals_cache_ttl_seconds).

yfinance's `.info` dict is somewhat unreliable field-to-field (Yahoo
changes what's populated fairly often) -- every field access below
uses .get() with a None fallback rather than assuming presence.
"""

import logging

import yfinance as yf

logger = logging.getLogger("equirisk.analytics.fundamentals")


def fetch_fundamentals(ticker: str) -> dict:
    """Returns a dict of fundamental metrics for one ticker. Any field
    yfinance doesn't have for this company comes back as None -- the
    dashboard should render "N/A" for those rather than erroring."""
    try:
        info = yf.Ticker(ticker).info
    except Exception as e:
        logger.error(f"Failed to fetch fundamentals for {ticker}: {e}")
        return {
            "pe_ratio": None, "net_income": None, "profit_margin": None,
            "debt_to_equity": None, "market_cap": None, "sector": None,
        }

    return {
        "pe_ratio": info.get("trailingPE"),
        "net_income": info.get("netIncomeToCommon"),
        "profit_margin": info.get("profitMargins"),  # fraction, e.g. 0.12 = 12%
        "debt_to_equity": info.get("debtToEquity"),  # yfinance reports this as a percentage-like ratio (e.g. 45.2)
        "market_cap": info.get("marketCap"),
        "sector": info.get("sector"),
    }


def format_large_number(value) -> str:
    """Formats a raw number into a readable Indian-context string
    (crores) for display -- e.g. 1234567890 -> '₹123.5 Cr'. Returns
    'N/A' for None/NaN input."""
    if value is None:
        return "N/A"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "N/A"

    crore = value / 1e7
    if abs(crore) >= 100:
        return f"\u20B9{crore:,.0f} Cr"
    return f"\u20B9{crore:,.2f} Cr"