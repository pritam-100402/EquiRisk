"""
src/etl/clean_transform.py

Reads raw prices + raw news from S3, cleans/standardizes both, joins them
on ticker + date, and writes the combined base table to
processed/base/ (partitioned by ticker) as the input for sentiment.py,
which in turn feeds feature_engineering.py.

This module owns "get raw data into one clean, joined Spark DataFrame."
Rolling-window stats, technical indicators, and sentiment scoring live in
feature_engineering.py / sentiment.py -- kept separate so each stage is
independently testable/inspectable in the notebooks.
"""

import logging

from pyspark.sql import DataFrame, functions as F
from pyspark.sql.types import ArrayType, StringType, StructField, StructType

from src.etl.spark_session import get_spark_session, s3a_path

from src.utils.config import load_config as _load_config

logger = logging.getLogger("equirisk.etl.clean_transform")

# Declared rather than inferred. Schema inference reads the input to work out
# its shape, and it cannot determine the fields of an array of structs when
# every file's `data` array is empty -- which happens whenever the news source
# returns nothing for any ticker. Inference then types `data` as an array of
# strings, and `article.title` fails at analysis time with a confusing error
# about a struct field on a non-struct column. Declaring the schema makes an
# empty run produce zero article rows instead, which the downstream LEFT join
# already handles correctly. It also skips the inference pass, which is free
# speed.
NEWS_ARTICLE_SCHEMA = StructType([
    StructField("article_id", StringType(), True),
    StructField("title", StringType(), True),
    StructField("description", StringType(), True),
    StructField("published_at", StringType(), True),
    StructField("link", StringType(), True),
    StructField("publisher", StringType(), True),
])

NEWS_FILE_SCHEMA = StructType([
    StructField("ticker", StringType(), True),
    StructField("company_name", StringType(), True),
    StructField("source", StringType(), True),
    StructField("fetched_at", StringType(), True),
    StructField("data", ArrayType(NEWS_ARTICLE_SCHEMA), True),
])




def read_raw_prices(spark, bucket: str, raw_prefix: str) -> DataFrame:
    """Reads all partitions under raw/prices/ -- each file already has a
    'ticker' column written during ingestion, so no path-parsing needed."""
    path = s3a_path(bucket, raw_prefix) + "/*/*.parquet"
    df = spark.read.parquet(path)
    logger.info(f"Read raw prices: {df.count()} rows")
    return df


def read_raw_news(spark, bucket: str, raw_prefix: str) -> DataFrame:
    """Raw news lands as one JSON blob per ticker/day, in this project's
    own schema (see src/ingestion/fetch_news.py): a `ticker` at the root
    and a `data` array of articles. Explode that array into one row per
    article.

    Note the ticker comes from the FILE, not from any field inside the
    article. That is the structural fix for the marketaux failure: it
    returned an `entities` array of resolved symbols, which had to be
    exploded and matched against NSE tickers, and that match silently
    attached US companies sharing a symbol (ACC Ltd vs American Campus
    Communities) to the wrong Indian stock. Querying per company and
    keeping the association at the file level makes the mismatch
    impossible rather than merely unlikely.
    """
    path = s3a_path(bucket, raw_prefix) + "/*/*.json"
    raw = (
        spark.read
        .schema(NEWS_FILE_SCHEMA)
        .option("multiLine", "true")
        .json(path)
    )

    exploded = raw.select(
        F.col("ticker"),
        F.explode("data").alias("article"),
    )
    news = exploded.select(
        F.col("ticker"),
        F.col("article.article_id").alias("article_id"),
        F.col("article.title").alias("title"),
        F.col("article.description").alias("description"),
        F.col("article.published_at").alias("published_at"),
    )
    logger.info(f"Read raw news: {news.count()} article rows")
    return news


def clean_prices(df: DataFrame) -> DataFrame:
    """Standardize column names/types, drop duplicate ticker+date rows,
    drop rows with null close price (can't compute returns without it)."""
    df = (
        df.withColumnRenamed("Date", "date")
        .withColumnRenamed("Open", "open")
        .withColumnRenamed("High", "high")
        .withColumnRenamed("Low", "low")
        .withColumnRenamed("Close", "close")
        .withColumnRenamed("Volume", "volume")
        .withColumn("ticker", F.regexp_replace("ticker", "\\.NS$", ""))
        .withColumn("date", F.to_date("date"))
        .dropDuplicates(["ticker", "date"])
        .filter(F.col("close").isNotNull())
    )
    return df.select("ticker", "date", "open", "high", "low", "close", "volume")


def clean_news(df: DataFrame) -> DataFrame:
    """Standardize the news date to a plain date (for joining against
    daily price rows) and drop rows missing the fields RAG/sentiment
    need downstream."""
    df = (
        df.withColumn("news_date", F.to_date("published_at"))
        .filter(F.col("title").isNotNull())
        .dropDuplicates(["article_id"])
    )
    return df.select("ticker", "news_date", "article_id", "title", "description", "published_at")


def aggregate_news_daily(news_df: DataFrame) -> DataFrame:
    """Collapse multiple articles/day into one row per ticker+date, since
    price rows are daily. Keeps a concatenated headline list (used later
    by sentiment.py) and an article count (itself a mild signal --
    news volume spikes often coincide with volatility)."""
    return news_df.groupBy("ticker", "news_date").agg(
        F.collect_list("title").alias("headlines"),
        F.collect_list("description").alias("descriptions"),
        F.count("article_id").alias("article_count"),
    )


def diagnose_news_join(prices_df: DataFrame, news_daily_df: DataFrame) -> dict:
    """Reports how well the news symbols line up with the price symbols
    BEFORE the join silently swallows a mismatch.

    Since the switch to Google News the ticker is carried on the news file
    itself, so a symbology mismatch is no longer structurally possible.
    What this still catches is coverage collapse: if the feed returns
    nothing for most tickers, every row survives the LEFT join and simply
    comes back with null headlines and article_count 0. The pipeline
    completes, the model trains, and the sentiment features are silently
    constant zero. Nothing errors. Log the overlap so that failure mode is
    visible instead of invisible.
    """
    price_symbols = {r["ticker"] for r in prices_df.select("ticker").distinct().collect()}
    news_symbols = {r["ticker"] for r in news_daily_df.select("ticker").distinct().collect()}
    matched = price_symbols & news_symbols

    stats = {
        "price_symbols": len(price_symbols),
        "news_symbols": len(news_symbols),
        "matched_symbols": len(matched),
        "unmatched_news_examples": sorted(news_symbols - price_symbols)[:10],
    }

    if not news_symbols:
        logger.error(
            "NEWS JOIN: no news symbols at all -- raw/news/ is empty or every "
            "response had no 'data' array. Sentiment features will be constant 0."
        )
    elif not matched:
        logger.error(
            f"NEWS JOIN: {len(news_symbols)} news symbols, {len(price_symbols)} price "
            f"symbols, ZERO overlap. The symbologies do not match, so all sentiment "
            f"features will be constant 0. Example news symbols: "
            f"{stats['unmatched_news_examples']}"
        )
    else:
        pct = 100.0 * len(matched) / len(price_symbols)
        log = logger.info if pct >= 50 else logger.warning
        log(f"NEWS JOIN: {len(matched)}/{len(price_symbols)} price symbols matched ({pct:.1f}%)")

    return stats


def join_prices_news(prices_df: DataFrame, news_daily_df: DataFrame) -> DataFrame:
    """Left join -- keep every price row even on days with no news
    (most days, for most midcap tickers). Downstream sentiment scoring
    should treat null headlines as neutral/no-signal, not missing data."""
    joined = prices_df.join(
        news_daily_df,
        (prices_df.ticker == news_daily_df.ticker) & (prices_df.date == news_daily_df.news_date),
        how="left",
    ).drop(news_daily_df.ticker).drop(news_daily_df.news_date)

    joined = joined.fillna({"article_count": 0})
    return joined


def run_etl(config_path: str = None) -> None:
    """Main ETL entrypoint, called by the orchestrator."""
    config = _load_config(config_path)
    bucket = config["s3"]["bucket"]
    raw_prices_prefix = config["s3"]["paths"]["raw_prices"]
    raw_news_prefix = config["s3"]["paths"]["raw_news"]
    out_prefix = config["s3"]["paths"]["processed_base"]

    spark = get_spark_session(
        app_name=config["spark"]["app_name"],
        master=config["spark"]["master"],
        driver_memory=config["spark"].get("driver_memory", "2g"),
        driver_max_result_size=config["spark"].get("driver_max_result_size", "512m"),
    )

    try:
        prices = clean_prices(read_raw_prices(spark, bucket, raw_prices_prefix))
        news = clean_news(read_raw_news(spark, bucket, raw_news_prefix))
        news_daily = aggregate_news_daily(news)

        diagnose_news_join(prices, news_daily)
        base_table = join_prices_news(prices, news_daily)

        out_path = s3a_path(bucket, out_prefix)
        (
            base_table.write.mode("overwrite")
            .partitionBy("ticker")
            .parquet(out_path)
        )
        logger.info(f"Wrote joined base table -> {out_path} (partitioned by ticker)")
    finally:
        spark.stop()


if __name__ == "__main__":
    from src.utils.logging_config import setup_logging

    setup_logging()
    run_etl()