"""
src/rag/retriever.py

Loads a given ticker's FAISS index + chunk text from S3 and returns the
top-k most relevant chunks for a user query. Called by llm_client.py
right before the Groq prompt is assembled.
"""

import io
import logging

import faiss
from sentence_transformers import SentenceTransformer

from src.utils.s3_io import get_bytes, get_json, vectorstore_key

from src.utils.config import load_config as _load_config

logger = logging.getLogger("equirisk.rag.retriever")

_embedder = None




def _get_embedder(model_name: str) -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(model_name)
    return _embedder


def load_ticker_index(ticker: str, bucket: str):
    """Loads the FAISS index + chunk text list for one ticker.
    Raises FileNotFoundError-style errors upward if the index doesn't
    exist yet -- callers should catch this and tell the user to run the
    pipeline first rather than crashing the whole chat."""
    index_bytes = get_bytes(vectorstore_key(ticker, "index.faiss"), bucket)
    reader = faiss.PyCallbackIOReader(io.BytesIO(index_bytes).read)
    index = faiss.read_index(reader)

    chunks_data = get_json(vectorstore_key(ticker, "chunks.json"), bucket)
    return index, chunks_data["chunks"]


def retrieve_relevant_chunks(ticker: str, query: str, config_path: str = None) -> list:
    """Returns the top_k most relevant text chunks for `query`, scoped
    to `ticker`'s index. Returns an empty list (not an error) if no
    index exists yet -- llm_client.py should fall back to a
    "no data available, run the pipeline first" response in that case."""
    config = _load_config(config_path)
    bucket = config["s3"]["bucket"]
    rag_config = config["rag"]

    try:
        index, chunks = load_ticker_index(ticker, bucket)
    except Exception as e:
        logger.warning(f"No index available for {ticker}: {e}")
        return []

    embedder = _get_embedder(rag_config["embedding_model"])
    query_vec = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")

    top_k = min(rag_config["top_k_retrieval"], len(chunks))
    scores, indices = index.search(query_vec, top_k)

    results = [chunks[i] for i in indices[0] if 0 <= i < len(chunks)]

    # Always include the two generated summary documents, whatever the query.
    #
    # Pure top-k similarity is the wrong tool for broad questions. "Should I
    # invest in this stock?" is lexically closer to a random headline than to
    # a paragraph of ratios, so the fundamentals and risk statistics -- the
    # two documents most likely to be needed -- can be crowded out by news.
    # They are short, so pinning them costs little context and guarantees the
    # model always has the company's numbers in front of it.
    pinned = [c for c in chunks
              if c.startswith("Company fundamentals for")
              or c.startswith("As of ")]
    for doc in pinned:
        if doc not in results:
            results.insert(0, doc)

    return results


if __name__ == "__main__":
    # Quick manual test -- run after build_index.py has populated at
    # least one ticker's index.
    test_ticker = "TATAMOTORS"
    test_query = "why is this stock considered risky right now"
    chunks = retrieve_relevant_chunks(test_ticker, test_query)
    print(f"Retrieved {len(chunks)} chunks for {test_ticker}:")
    for c in chunks:
        print("-", c)