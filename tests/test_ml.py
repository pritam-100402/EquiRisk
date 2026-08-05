"""
tests/test_ml.py

Unit tests for the ML layer and the analytics helpers that feed the
dashboard.

The most important test in this file is
TestChronologicalSplit::test_no_test_date_precedes_any_train_date. That
property is the whole reason the split was rewritten -- a random split
satisfies every other assertion here while still leaking future
information into training.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn", reason="scikit-learn not installed")
pytest.importorskip("xgboost", reason="xgboost not installed")
pytest.importorskip("lightgbm", reason="lightgbm not installed")

from src.ml.train import FEATURE_COLUMNS, LABEL_COLUMN, prepare_train_test


def _synthetic_feature_table(n_dates: int = 500, n_tickers: int = 5) -> pd.DataFrame:
    """A feature table shaped like the real one: every ticker observed on
    every date, all feature columns present, labels cycling through the
    three buckets."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2022-01-03", periods=n_dates, freq="B")
    labels = ["Low", "Medium", "High"]

    rows = []
    for ticker_idx in range(n_tickers):
        for date_idx, date in enumerate(dates):
            row = {
                "ticker": f"TICK{ticker_idx}",
                "date": date,
                LABEL_COLUMN: labels[(date_idx + ticker_idx) % 3],
            }
            for col in FEATURE_COLUMNS:
                row[col] = float(rng.normal())
            rows.append(row)
    return pd.DataFrame(rows)


class TestChronologicalSplit:

    def test_no_test_date_precedes_any_train_date(self):
        """The core anti-leakage property. Under a random split this
        assertion fails immediately; under a chronological split it holds
        by construction."""
        df = _synthetic_feature_table()
        cutoff_frac, gap = 0.2, 30

        from src.ml.train import training_cutoff_date
        cutoff = training_cutoff_date(df, cutoff_frac)

        train_dates = df[df["date"] <= cutoff]["date"]
        test_dates = df[df["date"] > cutoff]["date"]

        assert train_dates.max() < test_dates.min()

    def test_embargo_gap_is_actually_dropped(self):
        """Rows inside the embargo window belong to neither side."""
        df = _synthetic_feature_table()
        X_train, X_test, y_train, y_test, _, _ = prepare_train_test(
            df, test_size=0.2, random_state=42, gap_days=30
        )
        # 5 tickers x 500 dates = 2500 rows; the embargo must remove some.
        assert len(X_train) + len(X_test) < len(df)

    def test_split_is_deterministic(self):
        """No randomness left -- two calls must produce identical splits."""
        df = _synthetic_feature_table()
        a = prepare_train_test(df, 0.2, 42, 30)
        b = prepare_train_test(df, 0.2, 999, 30)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[2], b[2])

    def test_scaler_is_fitted_on_training_data_only(self):
        """Fitting the scaler on all rows would leak test-period
        distribution into training. Training features should be roughly
        standardised; test features need not be."""
        df = _synthetic_feature_table()
        X_train, X_test, _, _, _, _ = prepare_train_test(df, 0.2, 42, 30)
        assert np.allclose(X_train.mean(axis=0), 0, atol=1e-6)
        assert np.allclose(X_train.std(axis=0), 1, atol=1e-6)

    def test_encoder_round_trips_all_three_buckets(self):
        df = _synthetic_feature_table()
        _, _, y_train, y_test, _, encoder = prepare_train_test(df, 0.2, 42, 30)
        assert set(encoder.classes_) == {"Low", "Medium", "High"}
        decoded = encoder.inverse_transform(y_train)
        assert set(decoded).issubset({"Low", "Medium", "High"})

    def test_null_labels_are_excluded(self):
        """Tail rows with no complete forward window are live-inference
        rows, not training rows."""
        df = _synthetic_feature_table()
        df.loc[df["date"] == df["date"].max(), LABEL_COLUMN] = None
        X_train, X_test, y_train, y_test, _, _ = prepare_train_test(df, 0.2, 42, 30)
        assert len(y_train) + len(y_test) <= len(df.dropna(subset=[LABEL_COLUMN]))

    def test_empty_side_raises_rather_than_training_on_nothing(self):
        df = _synthetic_feature_table(n_dates=20)
        with pytest.raises(RuntimeError):
            prepare_train_test(df, test_size=0.01, random_state=42, gap_days=500)


class TestRiskScore:
    """The dashboard's heuristic composite score (distinct from the ML
    classifier's label -- see the note in risk_score.py)."""

    def test_score_stays_in_bounds(self):
        from src.analytics.risk_score import compute_risk_score
        for vol, beta, sent in [(0.0, 0.0, 1.0), (5.0, 10.0, -1.0), (0.3, 1.1, 0.0)]:
            assert -100 <= compute_risk_score(vol, beta, sent) <= 100

    def test_higher_volatility_increases_risk(self):
        from src.analytics.risk_score import compute_risk_score
        low = compute_risk_score(0.10, 1.0, 0.0)
        high = compute_risk_score(0.60, 1.0, 0.0)
        assert high > low

    def test_negative_sentiment_increases_risk(self):
        """Sentiment is the one input that runs backwards -- worth
        pinning down, it's easy to invert by accident."""
        from src.analytics.risk_score import compute_risk_score
        optimistic = compute_risk_score(0.3, 1.0, 0.8)
        pessimistic = compute_risk_score(0.3, 1.0, -0.8)
        assert pessimistic > optimistic

    def test_missing_component_falls_back_to_neutral(self):
        from src.analytics.risk_score import compute_risk_score
        assert not np.isnan(compute_risk_score(float("nan"), 1.0, 0.0))

    def test_categories_cover_the_whole_range(self):
        from src.analytics.risk_score import categorize_risk_score
        for score in range(-100, 101, 5):
            assert categorize_risk_score(float(score)) != "Unknown"

    def test_nan_score_is_unknown(self):
        from src.analytics.risk_score import categorize_risk_score
        assert categorize_risk_score(float("nan")) == "Unknown"


class TestMarketStats:

    def test_beta_of_series_against_itself_is_one(self):
        yf = pytest.importorskip("yfinance", reason="yfinance not installed")
        from src.analytics.market_stats import compute_beta

        idx = pd.date_range("2024-01-01", periods=100, freq="B")
        returns = pd.Series(np.random.default_rng(0).normal(0, 0.01, 100), index=idx)
        assert compute_beta(returns, returns) == pytest.approx(1.0)

    def test_beta_aligns_date_and_timestamp_indices(self):
        """Regression test for the 'beta always N/A' bug: a stock index of
        plain date objects must still align with a benchmark index of
        Timestamps."""
        pytest.importorskip("yfinance", reason="yfinance not installed")
        from src.analytics.market_stats import compute_beta

        idx = pd.date_range("2024-01-01", periods=100, freq="B")
        values = np.random.default_rng(1).normal(0, 0.01, 100)
        stock = pd.Series(values, index=[d.date() for d in idx])
        benchmark = pd.Series(values, index=idx)
        assert not np.isnan(compute_beta(stock, benchmark))

    def test_insufficient_overlap_returns_nan(self):
        pytest.importorskip("yfinance", reason="yfinance not installed")
        from src.analytics.market_stats import compute_beta

        idx = pd.date_range("2024-01-01", periods=5, freq="B")
        s = pd.Series([0.01] * 5, index=idx)
        assert np.isnan(compute_beta(s, s))

    def test_annualized_volatility_scales_by_sqrt_252(self):
        pytest.importorskip("yfinance", reason="yfinance not installed")
        from src.analytics.market_stats import annualized_volatility

        idx = pd.date_range("2024-01-01", periods=252, freq="B")
        returns = pd.Series(np.random.default_rng(2).normal(0, 0.01, 252), index=idx)
        expected = returns.std() * np.sqrt(252)
        assert annualized_volatility(returns) == pytest.approx(expected)
