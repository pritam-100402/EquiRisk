"""
src/ml/predict.py

Loads the model saved by train.py and scores the current "live" row for
every ticker -- i.e. the most recent row per ticker, which has a NULL
risk_label because add_forward_volatility_label() (feature_engineering.py)
only labels rows that have a COMPLETE forward window, and the most recent
horizon_days rows do not yet. Those null-label rows are exactly the ones
the dashboard needs a prediction for.

Writes predictions to their own S3 location (processed/predictions/latest.parquet)
rather than trying to patch values back into the Spark-partitioned feature
table -- keeps this module simple pandas-in/pandas-out, and keeps the
Spark-written feature table untouched by non-Spark writes.
"""

import io
import logging
from datetime import datetime, timezone

import joblib
import pandas as pd

from src.ml.train import FEATURE_COLUMNS
from src.utils.s3_io import (
    read_hive_partitioned_parquet_s3,
    write_parquet_s3,
    get_bytes,
    model_key,
    predictions_key,
)

from src.utils.config import load_config as _load_config

logger = logging.getLogger("equirisk.ml.predict")

PREDICTIONS_KEY = predictions_key()


def load_model_bundle(version: str, bucket: str) -> dict:
    """Loads the {model, scaler, features, labels} bundle train.py saved."""
    model_bytes = get_bytes(model_key(version, "model.pkl"), bucket)
    return joblib.load(io.BytesIO(model_bytes))


def load_feature_table(bucket: str, processed_prefix: str) -> pd.DataFrame:
    return read_hive_partitioned_parquet_s3(processed_prefix, bucket, partition_col="ticker")


def get_latest_rows(df: pd.DataFrame) -> pd.DataFrame:
    """One row per ticker: the most recent date. This row has a null
    risk_label -- there is no complete forward window to compute one from
    yet -- which is exactly why it needs a model prediction."""
    return df.sort_values("date").groupby("ticker").tail(1).reset_index(drop=True)


def run_inference(config_path: str = None) -> pd.DataFrame:
    """Main inference entrypoint, called by the orchestrator after
    training. Scores the latest row per ticker and writes predictions
    to S3. Returns the predictions DataFrame too, mainly for notebook/
    CLI convenience."""
    config = _load_config(config_path)
    bucket = config["s3"]["bucket"]
    processed_prefix = config["s3"]["paths"]["processed_features"]
    version = config["ml"]["model_version"]

    bundle = load_model_bundle(version, bucket)
    model = bundle["model"]
    scaler = bundle["scaler"]
    encoder = bundle["encoder"]
    feature_cols = bundle["features"]

    df = load_feature_table(bucket, processed_prefix)
    latest = get_latest_rows(df)

    # Drop rows missing any feature the model needs -- can't predict
    # without a complete feature vector. Log which tickers get skipped
    # so a low prediction count is traceable rather than silently short.
    complete = latest.dropna(subset=feature_cols)
    skipped = set(latest["ticker"]) - set(complete["ticker"])
    if skipped:
        logger.warning(f"Skipping {len(skipped)} tickers with incomplete features: {sorted(skipped)}")

    if complete.empty:
        raise RuntimeError("No tickers have complete features for live prediction -- check ETL/feature output")

    X = scaler.transform(complete[feature_cols])
    predictions_encoded = model.predict(X)
    predictions = encoder.inverse_transform(predictions_encoded)

    result = complete[["ticker", "date"]].copy()
    result["predicted_risk_label"] = predictions
    result["predicted_at"] = datetime.now(timezone.utc).isoformat()

    write_parquet_s3(result, PREDICTIONS_KEY, bucket)
    logger.info(f"Wrote {len(result)} live predictions -> s3://{bucket}/{PREDICTIONS_KEY}")

    return result


if __name__ == "__main__":
    from src.utils.logging_config import setup_logging

    setup_logging()
    preds = run_inference()
    print(preds)