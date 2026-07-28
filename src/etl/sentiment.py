"""
src/etl/sentiment.py

Scores the aggregated daily headlines/descriptions (produced by
clean_transform.py's aggregate_news_daily) using VADER, and adds
sentiment columns to the feature table. Runs as a Spark UDF so it slots
into the same pipeline as the rest of the ETL rather than a separate
pass over the data.

Swap in FinBERT here later if VADER's generic sentiment proves too
noisy for financial headlines -- the function signature
(list[str] -> float) stays the same either way, so nothing else in the
pipeline needs to change.
"""

import logging

from pyspark.sql import DataFrame, functions as F
from pyspark.sql.types import DoubleType
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.etl.spark_session import get_spark_session, s3a_path

from src.utils.config import load_config as _load_config

logger = logging.getLogger("equirisk.etl.sentiment")

_analyzer = SentimentIntensityAnalyzer()




def score_headlines(headlines) -> float:
    """Average VADER compound score across a list of headlines.
    Returns 0.0 (neutral) for empty/null input -- a no-news day should
    read as "no signal," not missing data, so the model doesn't need
    special-casing for nulls downstream."""
    if not headlines:
        return 0.0
    scores = [_analyzer.polarity_scores(h)["compound"] for h in headlines if h]
    return float(sum(scores) / len(scores)) if scores else 0.0


score_headlines_udf = F.udf(score_headlines, DoubleType())


def add_sentiment_scores(df: DataFrame) -> DataFrame:
    """Adds a daily_sentiment column (VADER compound, -1 to +1) from the
    'headlines' array column, plus a 3-day rolling average to smooth
    single-day noise -- one very positive/negative headline shouldn't
    swing the risk model on its own."""
    df = df.withColumn("daily_sentiment", score_headlines_udf(F.col("headlines")))

    from pyspark.sql import Window
    w = Window.partitionBy("ticker").orderBy("date").rowsBetween(-2, 0)
    df = df.withColumn("sentiment_3d_avg", F.avg("daily_sentiment").over(w))

    return df


def run_sentiment_scoring(config_path: str = None) -> None:
    """Pipeline stage: reads the joined base table, adds sentiment
    columns, writes to processed_sentiment.

    Reads and writes DIFFERENT S3 prefixes on purpose. Spark evaluates
    lazily, so `df.write.mode("overwrite")` to the same path the DataFrame
    was read from can clear the input directory before the read has
    materialised -- either a FileNotFoundException or, worse, a silently
    truncated table."""
    config = _load_config(config_path)
    bucket = config["s3"]["bucket"]
    in_prefix = config["s3"]["paths"]["processed_base"]
    out_prefix = config["s3"]["paths"]["processed_sentiment"]

    spark = get_spark_session(
        app_name=config["spark"]["app_name"] + "-Sentiment",
        master=config["spark"]["master"],
        driver_memory=config["spark"].get("driver_memory", "2g"),
        driver_max_result_size=config["spark"].get("driver_max_result_size", "512m"),
    )

    try:
        in_path = s3a_path(bucket, in_prefix)
        out_path = s3a_path(bucket, out_prefix)

        df = spark.read.parquet(in_path)
        df = add_sentiment_scores(df)
        df.write.mode("overwrite").partitionBy("ticker").parquet(out_path)
        logger.info(f"Wrote sentiment-scored table -> {out_path}")
    finally:
        spark.stop()


if __name__ == "__main__":
    from src.utils.logging_config import setup_logging

    setup_logging()
    run_sentiment_scoring()