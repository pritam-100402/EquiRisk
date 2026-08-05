"""
src/utils/s3_io.py

Thin wrapper around boto3/s3fs for all S3 reads/writes used across the
project. Centralizing this means ingestion, ETL, ML, and RAG modules never
touch boto3 directly -- easier to swap libraries or add retries/logging
later, and easier to unit test with a mocked client.

No local files are ever written by these helpers -- everything round-trips
through in-memory buffers.
"""

import io
import json
import logging
import os
import re
from datetime import datetime, timezone

from concurrent.futures import ThreadPoolExecutor

import boto3
import pandas as pd
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("equirisk.s3_io")

_BUCKET = os.environ.get("S3_BUCKET", "equirisk-data")


def _client():
    return boto3.client(
        "s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name=os.environ.get("AWS_DEFAULT_REGION", "ap-south-1"),
    )


def put_json(key: str, data: dict, bucket: str = _BUCKET) -> None:
    """Write a dict as JSON to s3://<bucket>/<key>."""
    body = json.dumps(data, default=str).encode("utf-8")
    _client().put_object(Bucket=bucket, Key=key, Body=body)
    logger.info(f"Wrote JSON -> s3://{bucket}/{key}")


def get_json(key: str, bucket: str = _BUCKET) -> dict:
    """Read a JSON object from s3://<bucket>/<key>."""
    obj = _client().get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))


def object_exists(key: str, bucket: str = _BUCKET) -> bool:
    """Note the exception class comes from botocore directly rather than
    from `_client().exceptions` -- referencing it off a client would build
    a brand new boto3 client mid-exception just to look up a class."""
    try:
        _client().head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False


def list_keys(prefix: str, bucket: str = _BUCKET) -> list:
    """List all keys under a prefix (handles pagination)."""
    paginator = _client().get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def read_parquet_s3(key: str, bucket: str = _BUCKET) -> pd.DataFrame:
    """Read a single parquet object into a pandas DataFrame.
    For partitioned datasets written by Spark, prefer spark.read.parquet()
    on the s3a:// path directly -- this helper is for smaller, single-file
    reads (e.g. a model's feature snapshot, RAG source docs)."""
    obj = _client().get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))


def read_hive_partitioned_parquet_s3(prefix: str, bucket: str = _BUCKET, partition_col: str = "ticker") -> pd.DataFrame:
    """Reads every parquet file under a Spark partitionBy() prefix and
    reconstructs the partition column from each key's folder name.

    Spark's partitionBy("ticker") writes data under
    <prefix>/ticker=SYMBOL/part-*.parquet and STRIPS the ticker column
    out of the file itself -- the value only lives in the folder name
    (standard Hive-style partitioning). A plain pd.read_parquet() on one
    of those files has no idea about that convention, so the partition
    column is silently missing unless something adds it back. This is
    that something -- use it (instead of a bare loop over
    read_parquet_s3) any time you're reading a Spark-partitioned table
    via pandas rather than Spark itself."""
    keys = [k for k in list_keys(prefix, bucket) if k.endswith(".parquet")]
    if not keys:
        raise RuntimeError(f"No parquet files found under s3://{bucket}/{prefix}")

    pattern = re.compile(rf"{re.escape(partition_col)}=([^/]+)/")

    def _read_one(key):
        match = pattern.search(key)
        if not match:
            logger.warning(f"Could not extract '{partition_col}' from key: {key} -- skipping")
            return None
        df = read_parquet_s3(key, bucket)
        df[partition_col] = match.group(1)
        return df

    # Fetched in parallel. Each partition is a separate S3 GET, and from
    # outside AWS the cost is dominated by per-request round-trip latency
    # rather than bandwidth -- so a sequential loop over 150+ partitions
    # spends almost all its time waiting. boto3 clients are not thread-safe,
    # but _client() builds a fresh one per call, so each worker gets its own.
    logger.info(f"Reading {len(keys)} parquet partitions from s3://{bucket}/{prefix}")
    with ThreadPoolExecutor(max_workers=16) as pool:
        frames = [f for f in pool.map(_read_one, keys) if f is not None]

    if not frames:
        raise RuntimeError(f"No files under s3://{bucket}/{prefix} matched the expected {partition_col}=... pattern")

    return pd.concat(frames, ignore_index=True)


def write_parquet_s3(df: pd.DataFrame, key: str, bucket: str = _BUCKET) -> None:
    """Writes microsecond-precision timestamps (coerce_timestamps="us")
    rather than pandas/pyarrow's nanosecond default -- Spark's built-in
    Parquet reader cannot read INT64 TIMESTAMP(NANOS) columns at all and
    throws 'Illegal Parquet type' the moment it tries to read the
    schema. This matters for any DataFrame with a tz-aware datetime
    column, e.g. yfinance's price history index."""
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, coerce_timestamps="us", allow_truncated_timestamps=True)
    buf.seek(0)
    _client().put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    logger.info(f"Wrote parquet -> s3://{bucket}/{key}")


def read_csv_s3(key: str, bucket: str = _BUCKET) -> pd.DataFrame:
    obj = _client().get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))


def put_bytes(key: str, data: bytes, bucket: str = _BUCKET) -> None:
    _client().put_object(Bucket=bucket, Key=key, Body=data)
    logger.info(f"Wrote bytes -> s3://{bucket}/{key}")


def get_bytes(key: str, bucket: str = _BUCKET) -> bytes:
    obj = _client().get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()


def dated_key(prefix: str, ticker: str, ext: str = "json") -> str:
    """e.g. raw/news/RELIANCE/2026-07-26.json"""
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    prefix = prefix.strip("/")
    return f"{prefix}/{ticker}/{date_str}.{ext}"


def model_key(version: str, filename: str = "model.pkl") -> str:
    return f"models/risk_model_{version}/{filename}"


def vectorstore_key(ticker: str, filename: str = "index.faiss") -> str:
    return f"vectorstore/{ticker}/{filename}"


def predictions_key() -> str:
    """Where predict.py writes live predictions and the RAG/dashboard
    layers read them from. Lives here with the other key helpers so
    consumers don't have to import the ML module (and its xgboost/
    lightgbm dependency chain) just to learn one path."""
    return "processed/predictions/latest.parquet"