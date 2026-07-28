"""
src/etl/feature_engineering.py

Reads the sentiment-scored table (processed/sentiment/ written by
sentiment.py), computes returns, rolling volatility, moving averages,
RSI, MACD, and the forward-looking risk label, and writes the final
feature table to processed/features/. This is the direct input to ML
training/inference, RAG index building, and the dashboard.

Kept separate from clean_transform.py so you can re-run just the feature
math (e.g. to try a different rolling window) without re-doing the raw
price/news join every time.
"""

import logging

from pyspark.sql import DataFrame, Window, functions as F

from src.etl.spark_session import get_spark_session, s3a_path

from src.utils.config import load_config as _load_config

logger = logging.getLogger("equirisk.etl.feature_engineering")




def add_returns(df: DataFrame) -> DataFrame:
    """Daily simple return = (close_t - close_t-1) / close_t-1, computed
    per ticker via a lag window ordered by date."""
    w = Window.partitionBy("ticker").orderBy("date")
    return df.withColumn("prev_close", F.lag("close", 1).over(w)).withColumn(
        "daily_return", (F.col("close") - F.col("prev_close")) / F.col("prev_close")
    )


def add_rolling_volatility(df: DataFrame, windows: list) -> DataFrame:
    """Rolling stddev of daily returns over each window (in trading
    days) -- the core "risk" signal. rowsBetween uses trading-day counts,
    not calendar days, since that's what the raw data actually has."""
    for w_days in windows:
        w = Window.partitionBy("ticker").orderBy("date").rowsBetween(-(w_days - 1), 0)
        df = df.withColumn(f"volatility_{w_days}d", F.stddev("daily_return").over(w))
    return df


def add_moving_averages(df: DataFrame, windows: list) -> DataFrame:
    for w_days in windows:
        w = Window.partitionBy("ticker").orderBy("date").rowsBetween(-(w_days - 1), 0)
        df = df.withColumn(f"ma_{w_days}d", F.avg("close").over(w))
    return df


def add_rsi(df: DataFrame, period: int = 14) -> DataFrame:
    """Standard 14-day RSI. Computed from average gain/loss over the
    window using the daily_return column already added by add_returns."""
    w_order = Window.partitionBy("ticker").orderBy("date")
    df = df.withColumn("gain", F.when(F.col("daily_return") > 0, F.col("daily_return")).otherwise(0.0))
    df = df.withColumn("loss", F.when(F.col("daily_return") < 0, -F.col("daily_return")).otherwise(0.0))

    w_roll = Window.partitionBy("ticker").orderBy("date").rowsBetween(-(period - 1), 0)
    df = df.withColumn("avg_gain", F.avg("gain").over(w_roll))
    df = df.withColumn("avg_loss", F.avg("loss").over(w_roll))

    df = df.withColumn(
        "rsi_14",
        F.when(F.col("avg_loss") == 0, 100.0).otherwise(
            100.0 - (100.0 / (1.0 + (F.col("avg_gain") / F.col("avg_loss"))))
        ),
    )
    return df.drop("gain", "loss", "avg_gain", "avg_loss")


def add_macd(df: DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> DataFrame:
    """MACD via EMA approximated over a rolling window (Spark has no
    native EMA/recursive window function, so this uses a simple-average
    approximation over the fast/slow windows rather than a true
    exponential decay). Good enough as a technical-indicator feature for
    this project, but it is not a faithful MACD -- documented as a known
    limitation in the README."""
    w_fast = Window.partitionBy("ticker").orderBy("date").rowsBetween(-(fast - 1), 0)
    w_slow = Window.partitionBy("ticker").orderBy("date").rowsBetween(-(slow - 1), 0)
    w_signal = Window.partitionBy("ticker").orderBy("date").rowsBetween(-(signal - 1), 0)

    df = df.withColumn("ema_fast_approx", F.avg("close").over(w_fast))
    df = df.withColumn("ema_slow_approx", F.avg("close").over(w_slow))
    df = df.withColumn("macd_line", F.col("ema_fast_approx") - F.col("ema_slow_approx"))
    df = df.withColumn("macd_signal", F.avg("macd_line").over(w_signal))
    return df.drop("ema_fast_approx", "ema_slow_approx")


def training_cutoff_date(df: DataFrame, test_size: float):
    """The date that separates the training period from the test period.

    Indexes into the sorted list of distinct dates rather than splitting
    on rows, so every ticker is cut at the same calendar point. This
    exact same rule is reimplemented in pandas in src/ml/train.py --
    both operate on the sorted distinct dates of the same table, so they
    always agree on where the boundary falls. Change one, change both.
    """
    dates = [r["date"] for r in df.select("date").distinct().orderBy("date").collect()]
    if not dates:
        raise ValueError("Cannot compute a training cutoff on an empty table")
    idx = max(int(len(dates) * (1.0 - test_size)) - 1, 0)
    return dates[idx]



def add_short_horizon_volatility(df: DataFrame, windows: list = [5, 10, 30]) -> DataFrame:
    """Realised volatility at shorter lookbacks, including one matched to
    the label horizon.

    Volatility clustering is strongest at short lags -- yesterday's
    turbulence predicts tomorrow's far better than last quarter's does. The
    existing 20/60/90-day windows miss that entirely. The 30-day window is
    deliberately matched to the forward horizon being predicted: the single
    best predictor of the next 30 days of volatility is usually the last 30.
    """
    for w_days in windows:
        w = Window.partitionBy("ticker").orderBy("date").rowsBetween(-(w_days - 1), 0)
        df = df.withColumn(f"volatility_{w_days}d", F.stddev("daily_return").over(w))
    return df


def add_garman_klass_volatility(df: DataFrame, window_days: int = 20) -> DataFrame:
    """Garman-Klass (1980) estimator -- uses all four OHLC values.

    Parkinson uses only the high-low range and so ignores where the price
    opened and closed within it. Garman-Klass adds the open-to-close move,
    making it more efficient again -- roughly 7x close-to-close, against
    Parkinson's 5x.

        sigma^2 = 0.5*ln(H/L)^2 - (2*ln2 - 1)*ln(C/O)^2
    """
    valid = (F.col("high") > 0) & (F.col("low") > 0) & (F.col("open") > 0) & (F.col("close") > 0)
    gk = (
        0.5 * F.pow(F.log(F.col("high") / F.col("low")), 2)
        - (2 * F.log(F.lit(2.0)) - 1) * F.pow(F.log(F.col("close") / F.col("open")), 2)
    )
    df = df.withColumn("_gk", F.when(valid, gk))

    w = Window.partitionBy("ticker").orderBy("date").rowsBetween(-(window_days - 1), 0)
    # Clamped at zero: the estimator can go slightly negative on a single
    # day when the open-to-close term dominates a narrow range.
    df = df.withColumn(
        f"garman_klass_vol_{window_days}d",
        F.sqrt(F.greatest(F.avg("_gk").over(w), F.lit(0.0))),
    )
    return df.drop("_gk")


def add_volatility_of_volatility(df: DataFrame, window_days: int = 60) -> DataFrame:
    """How unstable the volatility itself has been.

    Two stocks can share the same 20-day volatility while one has held it
    steady and the other has swung between calm and chaotic. The second is
    much more likely to move again. This is the second moment of the risk
    measure, and it carries information the level alone cannot.
    """
    w = Window.partitionBy("ticker").orderBy("date").rowsBetween(-(window_days - 1), 0)
    df = df.withColumn("vol_of_vol_60d", F.stddev("volatility_20d").over(w))
    df = df.withColumn(
        "vol_of_vol_ratio",
        F.when(F.col("volatility_20d") > 0,
               F.col("vol_of_vol_60d") / F.col("volatility_20d")),
    )
    return df


def add_downside_risk(df: DataFrame, window_days: int = 20) -> DataFrame:
    """Semi-deviation: volatility computed on losses only.

    Risk is asymmetric in a way plain standard deviation cannot express --
    it treats a 5% gain and a 5% loss identically. Downside deviation is
    what actually concerns an investor, and downside-heavy volatility tends
    to persist differently from upside-heavy volatility.
    """
    df = df.withColumn(
        "_neg_ret",
        F.when(F.col("daily_return") < 0, F.col("daily_return")),
    )
    w = Window.partitionBy("ticker").orderBy("date").rowsBetween(-(window_days - 1), 0)
    df = df.withColumn("downside_vol_20d", F.stddev("_neg_ret").over(w))
    df = df.withColumn(
        "downside_ratio",
        F.when(F.col("volatility_20d") > 0,
               F.col("downside_vol_20d") / F.col("volatility_20d")),
    )
    return df.drop("_neg_ret")


def add_overnight_gap_features(df: DataFrame, window_days: int = 20) -> DataFrame:
    """Volatility of the overnight gap (previous close to next open).

    Information arriving while the market is shut shows up as a gap rather
    than as intraday movement. Frequent large gaps mark a stock whose price
    is being driven by news flow -- earnings, regulatory decisions, sector
    announcements -- and those names carry more forward volatility than
    their intraday numbers alone would suggest.
    """
    w_lag = Window.partitionBy("ticker").orderBy("date")
    df = df.withColumn("_prev_close", F.lag("close", 1).over(w_lag))
    df = df.withColumn(
        "_gap",
        F.when(F.col("_prev_close") > 0, F.col("open") / F.col("_prev_close") - 1.0),
    )

    w = Window.partitionBy("ticker").orderBy("date").rowsBetween(-(window_days - 1), 0)
    df = df.withColumn("overnight_gap_vol_20d", F.stddev("_gap").over(w))
    df = df.withColumn("avg_abs_gap_20d", F.avg(F.abs(F.col("_gap"))).over(w))
    return df.drop("_prev_close", "_gap")


def add_return_distribution_shape(df: DataFrame, window_days: int = 60) -> DataFrame:
    """Skewness and kurtosis of the recent return distribution.

    Fat tails (high kurtosis) mean extreme moves are more common than a
    normal distribution implies, and that property persists -- it is one of
    the most reliable stylised facts in financial returns. Negative skew
    flags crash-prone names. Neither is visible in a standard deviation.
    """
    w = Window.partitionBy("ticker").orderBy("date").rowsBetween(-(window_days - 1), 0)
    df = df.withColumn("return_skew_60d", F.skewness("daily_return").over(w))
    df = df.withColumn("return_kurt_60d", F.kurtosis("daily_return").over(w))
    return df


def add_drawdown_features(df: DataFrame, window_days: int = 60) -> DataFrame:
    """Current drawdown from the rolling peak, and the worst drawdown in
    the window.

    A stock already well below its recent high behaves differently from one
    at the high: falling markets are more volatile than rising ones (the
    leverage effect), so drawdown state is genuinely predictive rather than
    merely descriptive.
    """
    w = Window.partitionBy("ticker").orderBy("date").rowsBetween(-(window_days - 1), 0)
    df = df.withColumn("_roll_max", F.max("close").over(w))
    df = df.withColumn(
        "drawdown_from_peak",
        F.when(F.col("_roll_max") > 0, F.col("close") / F.col("_roll_max") - 1.0),
    )
    df = df.withColumn("max_drawdown_60d", F.min("drawdown_from_peak").over(w))
    return df.drop("_roll_max")


def add_liquidity_features(df: DataFrame, window_days: int = 20) -> DataFrame:
    """Amihud (2002) illiquidity: average price impact per unit of turnover.

        ILLIQ = mean( |return| / (volume * close) )

    Thinly traded stocks move further on the same order flow, so illiquidity
    is mechanically linked to volatility. It also captures something the
    volume ratio does not: that ratio is relative to the stock's own
    history, whereas this is an absolute measure comparable across the
    universe. Scaled by 1e6 to keep it in a numerically sane range.
    """
    df = df.withColumn("_turnover", F.col("volume") * F.col("close"))
    df = df.withColumn(
        "_illiq",
        F.when(F.col("_turnover") > 0, F.abs(F.col("daily_return")) / F.col("_turnover") * 1e6),
    )
    w = Window.partitionBy("ticker").orderBy("date").rowsBetween(-(window_days - 1), 0)
    df = df.withColumn("amihud_illiq_20d", F.avg("_illiq").over(w))
    return df.drop("_turnover", "_illiq")


def add_price_position_features(df: DataFrame) -> DataFrame:
    """Where the price sits within its 52-week range, plus momentum.

    Stocks near 52-week lows are typically more volatile than those near
    highs -- distress and uncertainty cluster at the bottom of the range.
    Momentum is included separately from the moving-average ratios because
    it measures the size of the move rather than the current deviation.

    Windows here are partial at the start of each ticker's series (Spark
    computes over whatever rows exist rather than returning null). That is
    acceptable for features -- it makes early rows noisier, not
    forward-looking -- and it avoids discarding the first year of data.
    """
    w_52w = Window.partitionBy("ticker").orderBy("date").rowsBetween(-251, 0)
    df = df.withColumn("_high_52w", F.max("close").over(w_52w))
    df = df.withColumn("_low_52w", F.min("close").over(w_52w))
    df = df.withColumn(
        "pct_of_52w_range",
        F.when(F.col("_high_52w") > F.col("_low_52w"),
               (F.col("close") - F.col("_low_52w")) / (F.col("_high_52w") - F.col("_low_52w"))),
    )

    w_lag = Window.partitionBy("ticker").orderBy("date")
    for lag in (20, 60):
        df = df.withColumn(
            f"momentum_{lag}d",
            F.when(F.lag("close", lag).over(w_lag) > 0,
                   F.col("close") / F.lag("close", lag).over(w_lag) - 1.0),
        )
    return df.drop("_high_52w", "_low_52w")


def add_market_sensitivity(df: DataFrame, window_days: int = 60) -> DataFrame:
    """Rolling beta and correlation against an equal-weighted market return
    built from the panel itself.

    A high-beta name inherits market turbulence and amplifies it, so beta is
    a direct channel for forward volatility. The market return is the
    cross-sectional mean of daily_return on each date -- contemporaneous and
    observable, not lookahead. Using the panel avoids a second data source
    and keeps the benchmark consistent with the universe being modelled.
    """
    w_date = Window.partitionBy("date")
    df = df.withColumn("market_return", F.avg("daily_return").over(w_date))

    w = Window.partitionBy("ticker").orderBy("date").rowsBetween(-(window_days - 1), 0)
    df = df.withColumn("_cov", F.covar_samp("daily_return", "market_return").over(w))
    df = df.withColumn("_mvar", F.var_samp("market_return").over(w))
    df = df.withColumn(
        "beta_60d",
        F.when(F.col("_mvar") > 0, F.col("_cov") / F.col("_mvar")),
    )
    df = df.withColumn(
        "corr_market_60d",
        F.corr("daily_return", "market_return").over(w),
    )
    return df.drop("_cov", "_mvar")


def add_extreme_move_features(df: DataFrame, window_days: int = 20) -> DataFrame:
    """Count of outsized daily moves in the recent window.

    Thresholded against the stock's own 90-day volatility, so it measures
    "unusual for this stock" rather than "large in absolute terms". A
    cluster of extreme moves is the clearest signal that a name has entered
    a turbulent regime.
    """
    df = df.withColumn(
        "_is_extreme",
        F.when(
            (F.col("volatility_90d") > 0)
            & (F.abs(F.col("daily_return")) > 2 * F.col("volatility_90d")),
            1.0,
        ).otherwise(0.0),
    )
    w = Window.partitionBy("ticker").orderBy("date").rowsBetween(-(window_days - 1), 0)
    df = df.withColumn("extreme_move_count_20d", F.sum("_is_extreme").over(w))
    return df.drop("_is_extreme")


def add_price_relative_features(df: DataFrame, windows: list) -> DataFrame:
    """Converts price-level features into scale-free ratios.

    ma_20d and friends are raw rupee amounts. MRF trades near Rs 150,000 and
    IDEA near Rs 10, so pooling 150 tickers and fitting one global scaler
    makes those columns encode "which company is this" rather than anything
    about risk. A moving average is only informative relative to the current
    price, so that is what gets fed to the model.

    Same for MACD: a Rs 40 MACD line means something very different on a
    Rs 100 stock than on a Rs 100,000 one. Dividing by close makes it
    comparable across the universe.
    """
    for w_days in windows:
        df = df.withColumn(
            f"ma_ratio_{w_days}d",
            F.when(F.col(f"ma_{w_days}d") > 0,
                   F.col("close") / F.col(f"ma_{w_days}d") - 1.0),
        )

    df = df.withColumn(
        "macd_norm",
        F.when(F.col("close") > 0, F.col("macd_line") / F.col("close")),
    )
    df = df.withColumn(
        "macd_signal_norm",
        F.when(F.col("close") > 0, F.col("macd_signal") / F.col("close")),
    )
    return df


def add_range_volatility(df: DataFrame, window_days: int = 20) -> DataFrame:
    """Parkinson (1980) high-low range volatility estimator.

    The pipeline ingests full OHLC and then uses only close. That discards
    real information: the intraday range says how much the price actually
    moved, whereas close-to-close cannot distinguish a quiet day from one
    that swung wildly and closed flat. Parkinson's estimator is roughly five
    times more statistically efficient than close-to-close at the same
    sample size, which makes it one of the stronger volatility predictors
    available here.

        sigma^2 = mean( ln(high/low)^2 ) / (4 ln 2)
    """
    ln_hl_sq = F.when(
        (F.col("high") > 0) & (F.col("low") > 0),
        F.pow(F.log(F.col("high") / F.col("low")), 2),
    )
    df = df.withColumn("_ln_hl_sq", ln_hl_sq)

    w = Window.partitionBy("ticker").orderBy("date").rowsBetween(-(window_days - 1), 0)
    df = df.withColumn(
        f"parkinson_vol_{window_days}d",
        F.sqrt(F.avg("_ln_hl_sq").over(w) / (4 * F.log(F.lit(2.0)))),
    )
    return df.drop("_ln_hl_sq")


def add_volatility_term_structure(df: DataFrame) -> DataFrame:
    """Ratio of short- to long-horizon volatility.

    Levels alone say how volatile a stock has been; the ratio says whether
    volatility is currently rising or falling relative to its own baseline.
    A value above 1 means the recent window is more turbulent than the
    longer one, which is the classic regime-change signal and is what
    actually precedes a shift in forward volatility.
    """
    df = df.withColumn(
        "vol_ratio_20_60",
        F.when(F.col("volatility_60d") > 0,
               F.col("volatility_20d") / F.col("volatility_60d")),
    )
    df = df.withColumn(
        "vol_ratio_20_90",
        F.when(F.col("volatility_90d") > 0,
               F.col("volatility_20d") / F.col("volatility_90d")),
    )
    return df


def add_volume_features(df: DataFrame, window_days: int = 20) -> DataFrame:
    """Volume relative to its own recent average.

    Raw volume is not comparable across tickers (a large-cap trades orders
    of magnitude more shares than a small one), so it is expressed as a
    ratio to the stock's own 20-day average. Unusual volume commonly leads
    volatility -- it marks the arrival of information before the price has
    finished reacting to it.
    """
    w = Window.partitionBy("ticker").orderBy("date").rowsBetween(-(window_days - 1), 0)
    df = df.withColumn("_avg_volume", F.avg("volume").over(w))
    df = df.withColumn(
        "volume_ratio",
        F.when(F.col("_avg_volume") > 0, F.col("volume") / F.col("_avg_volume")),
    )
    return df.drop("_avg_volume")


def add_market_regime_features(df: DataFrame) -> DataFrame:
    """Cross-sectional market volatility, and each stock's volatility
    relative to it.

    Volatility is heavily systematic -- when the market is turbulent,
    almost everything is turbulent together. Without a market-level term the
    model has no way to distinguish "this stock got riskier" from "everything
    got riskier", and those have very different implications for a forward
    forecast.

    Computed as the mean of volatility_20d across all tickers on each date.
    This is CONTEMPORANEOUS information -- every stock's trailing volatility
    is observable today -- so it is available at prediction time and is not
    lookahead.
    """
    w_date = Window.partitionBy("date")
    df = df.withColumn("market_vol_20d", F.avg("volatility_20d").over(w_date))
    df = df.withColumn(
        "rel_vol_20d",
        F.when(F.col("market_vol_20d") > 0,
               F.col("volatility_20d") / F.col("market_vol_20d")),
    )
    return df


def add_cross_sectional_rank_features(df: DataFrame) -> DataFrame:
    """Each stock's percentile rank against the other tickers on the SAME
    date, for the main trailing risk measures.

    This is the most direct answer to a mismatch in the problem setup: the
    LABEL is a cross-sectional rank ("is this stock among the riskier third
    today?"), but every other feature is an absolute level. That forces the
    model to infer, from raw numbers alone and separately on every date,
    whether 0.024 is high relative to the other 149 names -- and the answer
    depends entirely on the market regime that day. Supplying the rank
    directly removes that burden.

    It also exposes the property that makes this target learnable at all:
    relative volatility rank is persistent. A stock in the top tercile of
    trailing volatility is usually in the top tercile of forward volatility.
    The absolute features can only hint at this; the rank states it.

    Strictly TRAILING data ranked within a date -- every input is observable
    at prediction time, so there is no lookahead.
    """
    ranked = [
        "volatility_20d", "volatility_60d", "volatility_90d",
        "parkinson_vol_20d", "garman_klass_vol_20d",
        "downside_vol_20d", "vol_of_vol_60d",
        "beta_60d", "amihud_illiq_20d",
    ]
    for col in ranked:
        w = Window.partitionBy("date").orderBy(F.col(col).asc_nulls_last())
        df = df.withColumn(
            f"xs_rank_{col}",
            F.when(F.col(col).isNotNull(), F.percent_rank().over(w)),
        )
    return df


def add_rank_persistence_features(df: DataFrame, window_days: int = 60) -> DataFrame:
    """How stable this stock's cross-sectional risk rank has been.

    Rank persistence is the mechanism the whole cross-sectional target rests
    on, so it is worth stating explicitly rather than leaving the model to
    infer it. The mean says where the stock usually sits in the risk
    ordering; the standard deviation says how reliably it stays there. A
    name that has held the top tercile for three months is a much safer bet
    to remain there than one that arrived last week.

    The drift term captures direction: current rank against the trailing
    average, so a stock climbing the risk ordering is distinguishable from
    one that has always been near the top.
    """
    w = Window.partitionBy("ticker").orderBy("date").rowsBetween(-(window_days - 1), 0)

    df = df.withColumn("xs_rank_mean_60d", F.avg("xs_rank_volatility_20d").over(w))
    df = df.withColumn("xs_rank_std_60d", F.stddev("xs_rank_volatility_20d").over(w))
    df = df.withColumn(
        "xs_rank_drift",
        F.col("xs_rank_volatility_20d") - F.col("xs_rank_mean_60d"),
    )
    return df


def add_forward_volatility_label(df: DataFrame, horizon_days: int, buckets: list,
                                 test_size: float = 0.2,
                                 mode: str = "cross_sectional",
                                 tail_fraction: float = 0.33) -> DataFrame:
    """Forward-looking volatility over the next horizon_days, bucketed.

    Two target definitions, selected by `mode`:

    "absolute" -- buckets against fixed thresholds fitted on the training
        period. Asks "will realised volatility exceed 0.023?". Sounds like
        the natural question, but it is largely unlearnable at a 30-day
        horizon: whether any month clears a fixed threshold is dominated by
        market regime (crashes, rate shocks, elections), which is driven by
        news that has not happened yet. Daily technicals cannot recover it.

    "cross_sectional" (default) -- buckets by rank against the other tickers
        on the SAME date. Asks "will this stock be among the riskier third
        of the universe next month?". Relative volatility is a stock
        characteristic and is strongly persistent: a name that is volatile
        relative to peers in calm markets is usually volatile relative to
        peers in turbulent ones. Market regime lifts every stock together
        and largely cancels out of the ranking.

    The cross-sectional label uses peers' forward volatility on the same
    date, which is future information -- but so is any forward-looking
    label. What matters is that no FEATURE sees the future: at inference the
    model scores today's features alone. This is the standard cross-
    sectional setup in quantitative finance.

    Both modes share the full-window guard: rowsBetween(1, horizon_days)
    quietly computes over however many future rows exist rather than
    failing, so rows near the end of a ticker's series would otherwise be
    labelled from 2- or 7-day windows as if they were full 30-day labels.
    """
    w_forward = Window.partitionBy("ticker").orderBy("date").rowsBetween(1, horizon_days)

    df = df.withColumn("forward_volatility_raw", F.stddev("daily_return").over(w_forward))
    df = df.withColumn("forward_window_rows", F.count("daily_return").over(w_forward))
    df = df.withColumn(
        "forward_volatility",
        F.when(F.col("forward_window_rows") >= horizon_days, F.col("forward_volatility_raw")),
    ).drop("forward_volatility_raw", "forward_window_rows")

    if mode.startswith("cross_sectional"):
        # percent_rank within each date: 0.0 = lowest forward vol that day,
        # 1.0 = highest. Nulls sort out of the ranking automatically because
        # they are filtered before the window is applied.
        w_date = Window.partitionBy("date").orderBy("forward_volatility")
        df = df.withColumn(
            "forward_volatility_rank",
            F.when(F.col("forward_volatility").isNotNull(),
                   F.percent_rank().over(w_date)),
        )
        rank = F.col("forward_volatility_rank")

        if mode == "cross_sectional_binary":
            # Two classes, with the ambiguous middle band DROPPED (labelled
            # null, so prepare_train_test's dropna removes it).
            #
            # The three-class version spends much of its error budget on the
            # tercile boundaries: a stock at rank 0.33 and one at 0.34 have
            # essentially identical volatility and receive different labels,
            # so those rows are close to coin flips no matter how good the
            # features are. That is irreducible label noise, not model
            # weakness.
            #
            # Removing the middle band asks the question a risk dashboard
            # actually needs answered -- "is this clearly risky or clearly
            # safe?" -- and discards the rows where no honest answer exists.
            # It is a legitimate reframing, but the baseline moves from 1/3
            # to 1/2 and BOTH must be reported for the numbers to mean
            # anything.
            bucket_expr = (
                F.when(rank.isNull(), None)
                .when(rank < tail_fraction, buckets[0])
                .when(rank >= 1.0 - tail_fraction, buckets[-1])
                .otherwise(None)
            )
            logger.info(
                f"Risk label: CROSS-SECTIONAL BINARY. Bottom {tail_fraction:.0%} "
                f"-> '{buckets[0]}', top {tail_fraction:.0%} -> '{buckets[-1]}', "
                f"middle {1 - 2 * tail_fraction:.0%} dropped as ambiguous. "
                f"Baseline accuracy is 50%, NOT 33% -- report it alongside."
            )
            return df.withColumn("risk_label", bucket_expr)

        n = len(buckets)
        bucket_expr = F.when(rank.isNull(), None)
        for i in range(n - 1):
            bucket_expr = bucket_expr.when(rank < (i + 1) / n, buckets[i])
        bucket_expr = bucket_expr.otherwise(buckets[-1])

        logger.info(
            f"Risk label: CROSS-SECTIONAL rank within each date "
            f"({n} equal buckets). Classes are balanced in every period by "
            f"construction, so a regime shift cannot skew the test set. "
            f"Baseline accuracy is {100.0 / n:.1f}%."
        )
        return df.withColumn("risk_label", bucket_expr)

    # --- absolute mode ---
    cutoff = training_cutoff_date(df, test_size)
    train_only = df.filter(F.col("date") <= F.lit(cutoff))

    probabilities = [1 / len(buckets) * i for i in range(1, len(buckets))]
    quantiles = train_only.approxQuantile("forward_volatility", probabilities, 0.01)
    logger.info(f"Risk label: ABSOLUTE. Quantiles fitted on dates <= {cutoff}: {quantiles}")

    bucket_expr = F.when(F.col("forward_volatility").isNull(), None)
    for i, q in enumerate(quantiles):
        bucket_expr = bucket_expr.when(F.col("forward_volatility") <= q, buckets[i])
    bucket_expr = bucket_expr.otherwise(buckets[-1])

    return df.withColumn("risk_label", bucket_expr)


def run_feature_engineering(config_path: str = None) -> None:
    """Main feature engineering entrypoint. Reads the sentiment-scored
    table, adds all feature columns and the risk label, writes the final
    feature table to processed_features.

    Reads and writes DIFFERENT S3 prefixes on purpose -- see the note in
    sentiment.run_sentiment_scoring() for why an in-place overwrite is
    unsafe with Spark's lazy evaluation."""
    config = _load_config(config_path)
    bucket = config["s3"]["bucket"]
    in_prefix = config["s3"]["paths"]["processed_sentiment"]
    out_prefix = config["s3"]["paths"]["processed_features"]
    fe_config = config["feature_engineering"]
    label_config = config["ml"]["risk_label"]
    test_size = config["ml"]["train_test_split"]

    spark = get_spark_session(
        app_name=config["spark"]["app_name"] + "-FeatureEng",
        master=config["spark"]["master"],
        driver_memory=config["spark"].get("driver_memory", "2g"),
        driver_max_result_size=config["spark"].get("driver_max_result_size", "512m"),
    )

    try:
        in_path = s3a_path(bucket, in_prefix)
        out_path = s3a_path(bucket, out_prefix)

        df = spark.read.parquet(in_path)

        df = add_returns(df)
        df = add_rolling_volatility(df, fe_config["rolling_windows_days"])
        df = add_moving_averages(df, fe_config["rolling_windows_days"])
        df = add_rsi(df)
        df = add_macd(df)

        # Derived features. Order matters: each of these reads columns the
        # block above produced.
        df = add_price_relative_features(df, fe_config["rolling_windows_days"])
        df = add_range_volatility(df)
        df = add_volatility_term_structure(df)
        df = add_volume_features(df)
        df = add_market_regime_features(df)

        # Wave 2. Order matters -- several read columns produced above
        # (vol_of_vol needs volatility_20d, extreme moves need volatility_90d).
        df = add_short_horizon_volatility(df)
        df = add_garman_klass_volatility(df)
        df = add_volatility_of_volatility(df)
        df = add_downside_risk(df)
        df = add_overnight_gap_features(df)
        df = add_return_distribution_shape(df)
        df = add_drawdown_features(df)
        df = add_liquidity_features(df)
        df = add_price_position_features(df)
        df = add_market_sensitivity(df)
        df = add_extreme_move_features(df)

        # Wave 3 -- must come last: ranks the wave-2 columns.
        df = add_cross_sectional_rank_features(df)
        df = add_rank_persistence_features(df)
        df = add_forward_volatility_label(
            df,
            label_config["horizon_days"],
            label_config["buckets"],
            test_size,
            label_config.get("mode", "cross_sectional"),
            label_config.get("binary_tail_fraction", 0.33),
        )

        # repartition("ticker") before the partitioned write so each ticker
        # lands in exactly one file. Without it, shuffle_partitions (8) lets
        # each ticker be written by up to 8 separate tasks, producing ~1200
        # small objects rather than 150 -- and S3 write cost is dominated by
        # request count, not bytes. This was most of a 32-minute write.
        (
            df.repartition("ticker")
            .write.mode("overwrite")
            .partitionBy("ticker")
            .parquet(out_path)
        )
        logger.info(f"Wrote feature table -> {out_path} (partitioned by ticker)")
    finally:
        spark.stop()


if __name__ == "__main__":
    from src.utils.logging_config import setup_logging

    setup_logging()
    run_feature_engineering()