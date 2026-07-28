"""
src/ingestion/fetch_news.py

Fetches recent news headlines per ticker and writes the raw response to
S3 under raw/news/{ticker}/{date}.json.

SOURCE: Google News RSS.

This replaced marketaux, which was evaluated first and rejected. Two
findings drove the switch, both reproducible from the archived responses:

  1. Coverage. 97 of 100 tickers returned zero articles -- 8 articles
     total across 100 Indian midcaps.

  2. Symbol collision, which was the more dangerous problem. marketaux
     resolves ticker symbols against a US-centric namespace, so a query
     for "ACC" (ACC Ltd, Indian cement) returned articles about American
     Campus Communities, a US student-housing REIT. Those rows joined
     cleanly and silently, feeding a cement company's risk model
     sentiment derived from American real estate news. A wrong answer
     that looks right is worse than no answer.

Google News avoids both. It indexes Indian financial press (Economic
Times, Moneycontrol, Business Standard, Livemint) directly, and because
the query is by COMPANY NAME rather than ticker, there is no symbol
namespace to collide in. The ticker association is by construction: we
asked for this company's news, so this company is what we got.

It also needs no API key and imposes no daily quota, which removes the
100-requests/day ceiling that truncated the marketaux run at exactly
100 of 150 tickers.
"""

import hashlib
import logging
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote

import pandas as pd
import requests

from src.utils.config import load_config as _load_config
from src.utils.s3_io import put_json, dated_key

logger = logging.getLogger("equirisk.ingestion.news")

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"

# Google returns different editions per locale; the Indian English edition
# surfaces domestic financial press that the US edition largely omits.
LOCALE_PARAMS = "hl=en-IN&gl=IN&ceid=IN:en"

# Corporate suffixes are noise in a search query and occasionally harm
# recall ("Tata Communications Limited" matches less than "Tata
# Communications"). Stripped from the end of the name only.
_SUFFIX_RE = re.compile(
    r"\s+(Limited|Ltd\.?|Inc\.?|Corporation|Corp\.?|PLC|Company|Co\.?)$",
    re.IGNORECASE,
)


def _search_name(company_name: str) -> str:
    """Trim corporate suffixes for a cleaner search phrase.

    Except for short acronyms. "Tata Communications" is more findable than
    "Tata Communications Limited", but "ACC" on its own collides with the
    American College of Cardiology, and "OIL", "MRF", "SAIL", "BSE" are
    similarly ambiguous. Below five characters the suffix is doing useful
    disambiguating work, so it stays.
    """
    name = (company_name or "").strip()
    for _ in range(3):  # e.g. "... Company Limited"
        stripped = _SUFFIX_RE.sub("", name).strip()
        if stripped == name:
            break
        name = stripped

    if len(name) < 5 and len(name) < len(company_name or ""):
        return (company_name or "").strip()
    return name


def _article_id(link: str, title: str) -> str:
    """RSS has no stable article UUID, so derive one. Used downstream to
    deduplicate the same story syndicated across outlets."""
    return hashlib.sha1(f"{link}|{title}".encode("utf-8")).hexdigest()[:16]


def _parse_pubdate(raw: str) -> str:
    """RSS pubDate is RFC 822 ("Mon, 20 Jul 2026 10:30:00 GMT"), which
    Spark's to_date() will not parse. Normalise to ISO 8601 here rather
    than fighting it in the ETL."""
    if not raw:
        return ""
    try:
        return parsedate_to_datetime(raw).astimezone(timezone.utc).isoformat()
    except Exception:
        return ""


# Context terms that anchor a query to Indian equities. Deliberately
# excludes bare "share"/"shares"/"stock": those are ordinary English words
# that appear as verbs and nouns in unrelated coverage, which is how a
# search for ACC (the cement company) surfaced the American College of
# Cardiology -- "ACC shares updated recommendations for managing HFpEF".
# The anchors below are near-unambiguous in an Indian financial context.
MARKET_ANCHORS = '(NSE OR BSE OR Nifty OR Sensex OR "share price" OR crore OR earnings)'


def build_query(company_name: str, lookback_days: int) -> str:
    """Google News search operators: quoted phrase for the company, a
    stock-context term to filter out unrelated coverage, and when:Nd to
    bound recency."""
    name = _search_name(company_name)
    terms = f'"{name}" {MARKET_ANCHORS} when:{lookback_days}d'
    return f"{GOOGLE_NEWS_RSS}?q={quote(terms)}&{LOCALE_PARAMS}"


def fetch_ticker_news(symbol: str, company_name: str, lookback_days: int,
                      limit: int, timeout: int = 20) -> dict:
    """Fetch headlines for one company. Returns a dict in this project's
    own schema (not a passthrough of the provider's), so the ETL is
    insulated from any future source change.

    Returns a well-formed empty payload on failure rather than raising --
    one unreachable feed shouldn't kill a 150-ticker run.
    """
    payload = {
        "ticker": symbol,
        "company_name": company_name,
        "source": "google_news_rss",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "data": [],
    }

    url = build_query(company_name, lookback_days)

    try:
        resp = requests.get(url, timeout=timeout,
                            headers={"User-Agent": "Mozilla/5.0 (EquiRisk research pipeline)"})
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except Exception as e:
        logger.warning(f"{symbol}: fetch failed ({type(e).__name__}: {e})")
        return payload

    for item in list(root.iter("item"))[:limit]:
        title = (item.findtext("title") or "").strip()
        if not title:
            continue
        link = (item.findtext("link") or "").strip()
        published = _parse_pubdate(item.findtext("pubDate") or "")
        if not published:
            continue

        payload["data"].append({
            "article_id": _article_id(link, title),
            "title": title,
            # Google's RSS description is an HTML link blob, not prose, so
            # it is deliberately not carried through as a text field.
            "description": None,
            "published_at": published,
            "link": link,
            "publisher": (item.findtext("source") or "").strip() or None,
        })

    return payload


def _load_ticker_table(config: dict) -> pd.DataFrame:
    """Ticker master list, which must carry a company_name column. Run
    scripts/enrich_ticker_names.py once to populate it."""
    path = config["tickers"]["master_list_path"]
    df = pd.read_csv(path)

    if "company_name" not in df.columns:
        raise RuntimeError(
            f"{path} has no 'company_name' column. Google News is searched by "
            f"company name, not ticker. Run:  python scripts/enrich_ticker_names.py"
        )

    df["company_name"] = df["company_name"].fillna(df["symbol"])
    return df


def fetch_all_tickers_news(config_path: str = None, sleep_sec: float = 1.5) -> None:
    """Main news ingestion entrypoint, called by the orchestrator.

    sleep_sec paces requests to stay well within what Google tolerates for
    RSS. There is no published quota, but hammering it invites throttling.
    """
    config = _load_config(config_path)
    news_config = config["ingestion"]["news"]
    lookback_days = news_config["lookback_days"]
    limit = news_config["articles_per_ticker"]
    raw_prefix = config["s3"]["paths"]["raw_news"]
    bucket = config["s3"]["bucket"]

    tickers = _load_ticker_table(config)

    logger.info(
        f"Fetching Google News for {len(tickers)} tickers "
        f"(last {lookback_days} days, up to {limit} articles each)"
    )

    total_articles = 0
    tickers_with_news = 0

    for _, row in tickers.iterrows():
        symbol = row["symbol"]
        company_name = row["company_name"]

        payload = fetch_ticker_news(symbol, company_name, lookback_days, limit)
        n = len(payload["data"])

        key = dated_key(raw_prefix, symbol, ext="json")
        put_json(key, payload, bucket)

        total_articles += n
        if n:
            tickers_with_news += 1
        else:
            logger.debug(f"{symbol}: no articles")

        time.sleep(sleep_sec)

    logger.info(
        f"News ingestion complete: {total_articles} articles across "
        f"{tickers_with_news}/{len(tickers)} tickers"
    )

    if tickers_with_news == 0:
        raise RuntimeError(
            "No news retrieved for any ticker. Check network access to "
            "news.google.com and that company_name values look sensible."
        )


if __name__ == "__main__":
    from src.utils.logging_config import setup_logging

    setup_logging()
    fetch_all_tickers_news()
