"""
src/rag/build_index.py

Builds a per-ticker FAISS index over a small corpus assembled from:
  - recent news headlines/descriptions (from the processed feature table)
  - latest computed stats (risk label, volatility, sentiment trend)
  - a plain-language "risk explanation" blurb

This runs after ML inference (predict.py) so the corpus can include the
freshly computed risk score/label -- the chat should be able to explain
*why* a stock got its current label, not just recite news.

Each ticker gets its own small FAISS index (rather than one giant index
across all 150 tickers) since queries are always scoped to one company
in this dashboard -- keeps retrieval fast and avoids cross-ticker noise.
"""

import io
import logging

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.analytics.fundamentals import fetch_fundamentals, format_large_number
from src.utils.s3_io import (
    read_hive_partitioned_parquet_s3,
    read_parquet_s3,
    put_bytes,
    put_json,
    vectorstore_key,
    predictions_key,
)

from src.utils.config import load_config as _load_config

logger = logging.getLogger("equirisk.rag.build_index")

_embedder = None  # lazy-loaded, shared across tickers in one run




def _get_embedder(model_name: str) -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(model_name)
    return _embedder


def _chunk_text(text: str, chunk_size_tokens: int, overlap_tokens: int) -> list:
    """Simple word-count-based chunking (approximates tokens closely
    enough for this corpus size -- headlines/blurbs are short, so exact
    tokenization isn't critical here). Swap in a real tokenizer-based
    splitter if you find chunks running noticeably over/under target."""
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(chunk_size_tokens - overlap_tokens, 1)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + chunk_size_tokens])
        if chunk.strip():
            chunks.append(chunk)
        if start + chunk_size_tokens >= len(words):
            break
    return chunks


def _has_headlines(value) -> bool:
    """True if `value` holds at least one headline.

    A plain `if value:` breaks here. The headlines column is a Spark
    ARRAY<STRING>, which comes back through parquet into pandas as a numpy
    array, and numpy raises ValueError("truth value of an array with more
    than one element is ambiguous") on bool(). It only appeared to work
    while every row's headlines were None."""
    if value is None:
        return False
    try:
        return len(value) > 0
    except TypeError:
        return False



def _fundamentals_doc(ticker: str, suffix: str = ".NS") -> str:
    """A prose paragraph of company fundamentals for the RAG corpus.

    Written as sentences rather than a table so the embedding model can match
    it against natural-language questions ("is this company profitable?",
    "how much debt does it carry?") and so the LLM can quote it directly.

    Returns an empty string if yfinance has nothing useful, so that a company
    with no fundamentals data simply contributes no document rather than one
    full of "N/A".
    """
    try:
        f = fetch_fundamentals(f"{ticker}{suffix}")
    except Exception as e:
        logger.warning(f"{ticker}: fundamentals lookup failed ({e})")
        return ""

    if not any(v is not None for v in f.values()):
        return ""

    parts = [f"Company fundamentals for {ticker}."]

    if f.get("sector"):
        parts.append(f"It operates in the {f['sector']} sector.")
    if f.get("market_cap") is not None:
        parts.append(f"Market capitalisation is {format_large_number(f['market_cap'])}.")
    if f.get("pe_ratio") is not None:
        parts.append(
            f"The trailing price-to-earnings ratio is {f['pe_ratio']:.1f}, "
            f"which indicates how much investors pay per rupee of earnings."
        )
    if f.get("net_income") is not None:
        parts.append(f"Net income attributable to shareholders is "
                     f"{format_large_number(f['net_income'])}.")
    if f.get("profit_margin") is not None:
        # yfinance reports this as a fraction (0.12 = 12%)
        parts.append(
            f"The net profit margin is {f['profit_margin'] * 100:.1f} percent, "
            f"meaning that share of revenue is retained as profit."
        )
    if f.get("debt_to_equity") is not None:
        # yfinance reports this percentage-style, e.g. 45.2 meaning 0.452x
        ratio = f["debt_to_equity"] / 100.0
        leverage = ("relatively low leverage" if ratio < 0.5
                    else "moderate leverage" if ratio < 1.5
                    else "high leverage")
        parts.append(
            f"The debt-to-equity ratio is {ratio:.2f}, indicating {leverage}. "
            f"Higher leverage generally amplifies volatility, because a fixed "
            f"interest burden magnifies the effect of changes in operating income."
        )

    return " ".join(parts)


def build_ticker_corpus(ticker_df: pd.DataFrame, ticker: str,
                        predicted_label: str = None) -> list:
    """Assembles the raw text documents for one ticker: recent headlines
    (most recent 20 rows' worth), plus one summary doc describing the
    latest risk stats in plain language. Returns a list of raw text
    strings, pre-chunking.

    `predicted_label` comes from predict.py's output. The feature table's
    own risk_label is null on the latest row by construction -- there is no
    complete forward window yet -- so reading it here put the literal
    string "None" into the corpus, and the chat assistant would dutifully
    tell users the risk label was None. The model's prediction is the
    answer the dashboard is actually showing, so it belongs here."""
    docs = []

    recent = ticker_df.sort_values("date").tail(20)
    for _, row in recent.iterrows():
        if _has_headlines(row.get("headlines")):
            for headline in row["headlines"]:
                if headline:
                    docs.append(f"[{row['date']}] {headline}")

    latest = ticker_df.sort_values("date").iloc[-1]

    label = predicted_label if predicted_label else latest.get("risk_label")
    label_text = (
        f"a predicted risk label of {label}" if predicted_label
        else (f"a risk label of {label}" if pd.notna(label) else "no risk label available yet")
    )

    latest_date = pd.to_datetime(latest["date"]).strftime("%Y-%m-%d")
    summary = (
        f"As of {latest_date}, {ticker} has {label_text}. "
        f"20-day volatility is {latest.get('volatility_20d', float('nan')):.4f}, "
        f"60-day volatility is {latest.get('volatility_60d', float('nan')):.4f}. "
        f"Recent average news sentiment (3-day) is {latest.get('sentiment_3d_avg', 0.0):.3f} "
        f"(range -1 very negative to +1 very positive), based on {int(latest.get('article_count', 0))} "
        f"recent articles."
    )
    docs.append(summary)

    # Fundamentals are fetched live from yfinance rather than stored in the
    # feature table, because they change quarterly rather than daily and are
    # not model inputs. They give the assistant something to say about the
    # business itself, not only its price behaviour.
    fundamentals = _fundamentals_doc(ticker)
    if fundamentals:
        docs.append(fundamentals)

    return docs


def build_index_for_ticker(ticker: str, ticker_df: pd.DataFrame, config: dict,
                           predicted_label: str = None) -> None:
    rag_config = config["rag"]
    bucket = config["s3"]["bucket"]

    docs = build_ticker_corpus(ticker_df, ticker, predicted_label)
    chunks = []
    for doc in docs:
        chunks.extend(_chunk_text(doc, rag_config["chunk_size_tokens"], rag_config["chunk_overlap_tokens"]))

    if not chunks:
        logger.warning(f"No chunks generated for {ticker} -- skipping index build")
        return

    embedder = _get_embedder(rag_config["embedding_model"])
    embeddings = embedder.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)

    index = faiss.IndexFlatIP(embeddings.shape[1])  # inner product on normalized vecs = cosine sim
    index.add(embeddings.astype("float32"))

    index_buf = io.BytesIO()
    faiss.write_index(index, faiss.PyCallbackIOWriter(index_buf.write))
    put_bytes(vectorstore_key(ticker, "index.faiss"), index_buf.getvalue(), bucket)

    put_json(vectorstore_key(ticker, "chunks.json"), {"chunks": chunks}, bucket)

    logger.info(f"Built index for {ticker}: {len(chunks)} chunks")


def refresh_all_indices(config_path: str = None) -> None:
    """Main RAG entrypoint, called by the orchestrator after ML
    inference completes. Rebuilds every ticker's index from scratch --
    simpler and safer for a course project than incremental updates,
    and the data volume per ticker is small enough that full rebuild is
    fast."""
    config = _load_config(config_path)
    bucket = config["s3"]["bucket"]
    processed_prefix = config["s3"]["paths"]["processed_features"]

    full_df = read_hive_partitioned_parquet_s3(processed_prefix, bucket, partition_col="ticker")

    # Predictions are optional -- if inference hasn't run yet the corpus
    # just falls back to whatever label the feature table has.
    predicted_labels = {}
    try:
        preds = read_parquet_s3(predictions_key(), bucket)
        predicted_labels = dict(zip(preds["ticker"], preds["predicted_risk_label"]))
        logger.info(f"Loaded {len(predicted_labels)} predicted labels for the RAG corpus")
    except Exception as e:
        logger.warning(f"No predictions available for the RAG corpus ({e}) -- continuing without them")

    tickers = full_df["ticker"].unique().tolist()
    logger.info(f"Refreshing RAG indices for {len(tickers)} tickers")

    for ticker in tickers:
        ticker_df = full_df[full_df["ticker"] == ticker]
        try:
            build_index_for_ticker(ticker, ticker_df, config, predicted_labels.get(ticker))
        except Exception as e:
            logger.error(f"Failed to build index for {ticker}: {e}")

    logger.info("RAG index refresh complete")


if __name__ == "__main__":
    from src.utils.logging_config import setup_logging

    setup_logging()
    refresh_all_indices()