"""
tests/test_rag.py

Unit tests for the RAG layer's text handling.

Two of these are regression tests for bugs that were invisible while the
news pipeline returned nothing: the headlines column arrives as a numpy
array (so `if row["headlines"]:` raises), and the latest row's risk_label
is null by construction (so the corpus was advertising a risk label of
"None" to the LLM).
"""

import pandas as pd
import pytest

pytest.importorskip("faiss", reason="faiss-cpu not installed")
pytest.importorskip("sentence_transformers", reason="sentence-transformers not installed")

from src.rag.build_index import _chunk_text, _has_headlines, build_ticker_corpus  # noqa: E402


class TestChunkText:

    def test_empty_text_yields_no_chunks(self):
        assert _chunk_text("", 100, 20) == []
        assert _chunk_text("   ", 100, 20) == []

    def test_short_text_is_one_chunk(self):
        chunks = _chunk_text("a short headline about the company", 100, 20)
        assert len(chunks) == 1

    def test_long_text_splits_into_multiple_chunks(self):
        text = " ".join(f"word{i}" for i in range(250))
        chunks = _chunk_text(text, chunk_size_tokens=100, overlap_tokens=20)
        assert len(chunks) > 1
        assert all(len(c.split()) <= 100 for c in chunks)

    def test_chunks_overlap_as_configured(self):
        text = " ".join(f"word{i}" for i in range(200))
        chunks = _chunk_text(text, chunk_size_tokens=100, overlap_tokens=20)
        first_words = chunks[0].split()
        second_words = chunks[1].split()
        assert first_words[-20:] == second_words[:20]

    def test_no_infinite_loop_when_overlap_exceeds_chunk_size(self):
        """step = max(chunk_size - overlap, 1) guards this; if that guard
        is ever removed the range() step goes <= 0 and this hangs."""
        chunks = _chunk_text(" ".join(f"w{i}" for i in range(50)), 10, 50)
        assert len(chunks) > 0

    def test_covers_all_input_words(self):
        text = " ".join(f"word{i}" for i in range(150))
        chunks = _chunk_text(text, 100, 20)
        assert "word149" in chunks[-1]


class TestHasHeadlines:
    """Regression: a Spark ARRAY<STRING> read through parquet into pandas
    is a numpy array, and bool(numpy_array_with_2+_elements) raises
    ValueError rather than returning True."""

    def test_none_is_false(self):
        assert _has_headlines(None) is False

    def test_empty_containers_are_false(self):
        import numpy as np
        assert _has_headlines([]) is False
        assert _has_headlines(np.array([])) is False

    def test_numpy_array_of_headlines_is_true(self):
        import numpy as np
        arr = np.array(["first headline", "second headline"], dtype=object)
        assert _has_headlines(arr) is True

    def test_plain_list_is_true(self):
        assert _has_headlines(["a headline"]) is True

    def test_non_container_is_false_not_an_exception(self):
        assert _has_headlines(3.14) is False


class TestBuildTickerCorpus:

    def _frame(self, headlines=None, risk_label=None):
        return pd.DataFrame([{
            "date": pd.Timestamp("2026-07-24"),
            "risk_label": risk_label,
            "volatility_20d": 0.0209,
            "volatility_60d": 0.0175,
            "sentiment_3d_avg": 0.0,
            "article_count": 0 if headlines is None else len(headlines),
            "headlines": headlines,
        }])

    def test_summary_document_is_always_produced(self):
        docs = build_ticker_corpus(self._frame(), "TESTCO")
        assert len(docs) == 1
        assert "TESTCO" in docs[0]

    def test_numpy_headlines_do_not_raise(self):
        """This is the exact shape that came out of parquet and broke
        build_index for every ticker that had news."""
        import numpy as np
        arr = np.array(["Profits up sharply", "New plant announced"], dtype=object)
        docs = build_ticker_corpus(self._frame(headlines=arr), "TESTCO")
        assert len(docs) == 3  # two headlines + one summary
        assert any("Profits up sharply" in d for d in docs)

    def test_null_label_does_not_leak_the_string_none(self):
        """The latest row's risk_label is null by design. Rendering it
        directly produced 'has a risk label of None', which the assistant
        then reported to users as fact."""
        docs = build_ticker_corpus(self._frame(risk_label=None), "TESTCO")
        assert "risk label of None" not in docs[-1]
        assert "no risk label available yet" in docs[-1]

    def test_predicted_label_is_used_when_supplied(self):
        docs = build_ticker_corpus(self._frame(risk_label=None), "TESTCO", predicted_label="High")
        assert "predicted risk label of High" in docs[-1]
