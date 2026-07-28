"""
tests/test_etl.py

Unit tests for the ETL layer's pure logic.

Scope note: these deliberately avoid starting a SparkSession or touching
S3. Spark-dependent transforms are integration-tested by running the
pipeline and inspecting the output in notebooks/02 and /03; what's worth
unit-testing here is the logic that has real branching and silent-failure
modes -- sentiment scoring of raw text, and the chronological cutoff rule
that both feature_engineering and train must agree on.
"""

import pytest

vader = pytest.importorskip("vaderSentiment", reason="vaderSentiment not installed")

from src.etl.sentiment import score_headlines  # noqa: E402


class TestScoreHeadlines:
    """score_headlines is the one place raw text becomes a number, and it
    has to treat 'no news' and 'bad news' as different things."""

    def test_empty_input_is_neutral_not_missing(self):
        # A no-news day must read as 0.0 (no signal), not NaN -- the whole
        # downstream pipeline assumes it never has to handle nulls here.
        assert score_headlines([]) == 0.0
        assert score_headlines(None) == 0.0

    def test_positive_headline_scores_positive(self):
        score = score_headlines(["Company reports record profits and strong growth"])
        assert score > 0

    def test_negative_headline_scores_negative(self):
        score = score_headlines(["Company reports massive losses amid fraud investigation"])
        assert score < 0

    def test_result_stays_in_valid_range(self):
        headlines = [
            "Excellent results, shares surge",
            "Terrible quarter, shares crash",
            "Company announces routine board meeting",
        ]
        assert -1.0 <= score_headlines(headlines) <= 1.0

    def test_averages_across_headlines(self):
        positive = "Record profits and excellent growth"
        negative = "Devastating losses and fraud"
        mixed = score_headlines([positive, negative])
        assert abs(mixed) < abs(score_headlines([positive]))

    def test_ignores_none_entries_in_list(self):
        # collect_list can yield nulls if an article had a null title.
        with_nulls = score_headlines(["Record profits and excellent growth", None])
        without = score_headlines(["Record profits and excellent growth"])
        assert with_nulls == pytest.approx(without)

    def test_all_null_entries_fall_back_to_neutral(self):
        assert score_headlines([None, None]) == 0.0


class TestTrainingCutoffDate:
    """The cutoff rule is duplicated in Spark (feature_engineering) and
    pandas (train). They MUST agree -- if they drift, the risk-label
    quantiles get fitted on a different period than the model trains on,
    which is exactly the leak the chronological split was added to close.
    """

    def test_pandas_cutoff_splits_on_dates_not_rows(self):
        pd = pytest.importorskip("pandas")
        pytest.importorskip("sklearn")
        from src.ml.train import training_cutoff_date

        # 10 distinct dates, but wildly unbalanced row counts per date.
        dates = pd.to_datetime([f"2024-01-{d:02d}" for d in range(1, 11)])
        rows = []
        for i, d in enumerate(dates):
            rows.extend([d] * (100 if i == 0 else 1))
        df = pd.DataFrame({"date": rows})

        # 20% test => cutoff is the 8th of the 10 distinct dates,
        # regardless of the row imbalance.
        assert training_cutoff_date(df, 0.2) == dates[7]

    def test_cutoff_rejects_empty_input(self):
        pd = pytest.importorskip("pandas")
        pytest.importorskip("sklearn")
        from src.ml.train import training_cutoff_date

        with pytest.raises(ValueError):
            training_cutoff_date(pd.DataFrame({"date": []}), 0.2)
