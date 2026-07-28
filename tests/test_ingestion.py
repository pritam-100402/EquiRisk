"""
tests/test_ingestion.py

Unit tests for news ingestion (Google News RSS).

No network: the RSS body is injected. These cover the parsing surface that
broke silently with the previous provider -- date formats Spark can't read,
unstable article IDs, and failures that must not raise mid-run.
"""

import types
import xml.etree.ElementTree as ET

import pytest

from src.ingestion.fetch_news import (
    _article_id,
    _parse_pubdate,
    _search_name,
    build_query,
    fetch_ticker_news,
)

SAMPLE_RSS = b"""<?xml version="1.0"?><rss version="2.0"><channel>
<title>ACC - Google News</title>
<item>
  <title>ACC Ltd Q1 profit rises 18% on strong cement demand</title>
  <link>https://economictimes.indiatimes.com/acc-q1</link>
  <pubDate>Mon, 20 Jul 2026 10:30:00 GMT</pubDate>
  <source url="https://economictimes.indiatimes.com">Economic Times</source>
</item>
<item>
  <title>Cement stocks slip as input costs bite</title>
  <link>https://moneycontrol.com/cement-slip</link>
  <pubDate>Tue, 21 Jul 2026 04:15:00 GMT</pubDate>
  <source url="https://moneycontrol.com">Moneycontrol</source>
</item>
<item>
  <title></title>
  <link>https://example.com/empty</link>
  <pubDate>Tue, 21 Jul 2026 05:00:00 GMT</pubDate>
</item>
</channel></rss>"""


@pytest.fixture
def stub_feed(monkeypatch):
    """Replace requests.get with a canned RSS response."""
    import src.ingestion.fetch_news as fn

    class FakeResp:
        content = SAMPLE_RSS

        def raise_for_status(self):
            pass

    monkeypatch.setattr(fn, "requests", types.SimpleNamespace(get=lambda *a, **k: FakeResp()))


class TestSearchName:
    """Corporate suffixes hurt recall in a quoted phrase search -- except
    on short acronyms, where they are the only thing disambiguating the
    company from an unrelated organisation."""

    @pytest.mark.parametrize("raw,expected", [
        ("Tata Communications Limited", "Tata Communications"),
        ("Godrej Industries Ltd", "Godrej Industries"),
        ("3M India Limited", "3M India"),
        ("Bharat Forge Limited", "Bharat Forge"),
    ])
    def test_strips_suffix_from_distinctive_names(self, raw, expected):
        assert _search_name(raw) == expected

    @pytest.mark.parametrize("raw", ["ACC Ltd.", "BSE Limited", "MRF Limited", "SRF Limited"])
    def test_keeps_suffix_on_short_acronyms(self, raw):
        """Bare "ACC" collides with the American College of Cardiology;
        "OIL", "MRF", "SAIL", "BSE" are similarly ambiguous. Below five
        characters the suffix earns its place."""
        assert _search_name(raw) == raw

    def test_leaves_clean_names_alone(self):
        assert _search_name("Bharat Forge") == "Bharat Forge"

    def test_handles_empty_input(self):
        assert _search_name("") == ""
        assert _search_name(None) == ""


class TestBuildQuery:

    def test_uses_indian_edition(self):
        """The US edition omits most Indian financial press."""
        q = build_query("ACC Limited", 30)
        assert "hl=en-IN" in q and "gl=IN" in q

    def test_excludes_ambiguous_keywords(self):
        """Bare "shares"/"stock" match unrelated coverage -- "ACC shares
        updated recommendations" is a cardiology headline, not equity news."""
        from urllib.parse import unquote
        q = unquote(build_query("ACC Limited", 30))
        assert "OR shares)" not in q and "(stock OR" not in q
        assert "NSE" in q and "Sensex" in q

    def test_encodes_recency_bound(self):
        assert "when%3A30d" in build_query("ACC Limited", 30)
        assert "when%3A7d" in build_query("ACC Limited", 7)


class TestParsePubdate:
    """RSS emits RFC 822; Spark's to_date() needs ISO 8601."""

    def test_converts_rfc822_to_iso(self):
        assert _parse_pubdate("Mon, 20 Jul 2026 10:30:00 GMT").startswith("2026-07-20T10:30:00")

    def test_bad_input_returns_empty_not_exception(self):
        assert _parse_pubdate("") == ""
        assert _parse_pubdate("not a date") == ""
        assert _parse_pubdate(None) == ""


class TestArticleId:

    def test_is_stable(self):
        assert _article_id("http://x/1", "T") == _article_id("http://x/1", "T")

    def test_differs_by_link(self):
        assert _article_id("http://x/1", "T") != _article_id("http://x/2", "T")


class TestFetchTickerNews:

    def test_parses_articles(self, stub_feed):
        p = fetch_ticker_news("ACC", "ACC Limited", 30, 20)
        assert p["ticker"] == "ACC"
        assert p["source"] == "google_news_rss"
        assert len(p["data"]) == 2

    def test_ticker_comes_from_the_request_not_the_feed(self, stub_feed):
        """The structural fix for the previous provider's symbol collision:
        we asked for this company, so this company is what we record. There
        is no resolved-symbol field that could disagree."""
        p = fetch_ticker_news("ACC", "ACC Limited", 30, 20)
        assert all("symbol" not in a for a in p["data"])
        assert p["ticker"] == "ACC"

    def test_drops_empty_titles(self, stub_feed):
        titles = [a["title"] for a in fetch_ticker_news("ACC", "ACC Ltd", 30, 20)["data"]]
        assert "" not in titles

    def test_respects_limit(self, stub_feed):
        assert len(fetch_ticker_news("ACC", "ACC Ltd", 30, 1)["data"]) == 1

    def test_dates_are_iso(self, stub_feed):
        for a in fetch_ticker_news("ACC", "ACC Ltd", 30, 20)["data"]:
            assert a["published_at"].startswith("2026-")

    def test_network_failure_returns_empty_payload(self, monkeypatch):
        """One unreachable feed must not abort a 150-ticker run."""
        import src.ingestion.fetch_news as fn

        def boom(*a, **k):
            raise ConnectionError("no network")

        monkeypatch.setattr(fn, "requests", types.SimpleNamespace(get=boom))
        p = fetch_ticker_news("ACC", "ACC Ltd", 30, 20)
        assert p["data"] == []
        assert p["ticker"] == "ACC"

    def test_malformed_xml_returns_empty_payload(self, monkeypatch):
        import src.ingestion.fetch_news as fn

        class BadResp:
            content = b"<html>rate limited</html>"

            def raise_for_status(self):
                pass

        monkeypatch.setattr(fn, "requests", types.SimpleNamespace(get=lambda *a, **k: BadResp()))
        assert fetch_ticker_news("ACC", "ACC Ltd", 30, 20)["data"] == []
