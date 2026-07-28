"""
src/etl/spark_session.py

Creates a SparkSession configured to read/write directly to S3 via the
S3A connector, using credentials from environment variables (loaded via
python-dotenv). No local data is ever persisted — all reads/writes
target s3a:// paths.

Prereqs on your machine (Zorin OS, JDK11, Hadoop already installed):
- Spark's `--packages` will pull the matching hadoop-aws + aws-java-sdk-bundle
  jars on first run (needs internet the first time; cached afterwards in
  ~/.ivy2). If you'd rather not depend on internet at runtime, download the
  matching jar versions once and place them in $SPARK_HOME/jars/ instead,
  then drop the .config("spark.jars.packages", ...) line below.
- Match hadoop-aws version to the Hadoop version bundled with your PySpark
  build. PySpark 3.5.x ships with Hadoop 3.3.4 by default — hadoop-aws:3.3.4
  is the safe pairing. If `pyspark --version` reports a different Hadoop
  version, adjust HADOOP_AWS_VERSION below to match.
"""

import os
from dotenv import load_dotenv
from pyspark.sql import SparkSession

load_dotenv()

HADOOP_AWS_VERSION = "3.3.4"
AWS_SDK_VERSION = "1.12.262"


def get_spark_session(app_name: str = "EquiRiskETL", master: str = "local[*]",
                      driver_memory: str = "2g",
                      driver_max_result_size: str = "512m") -> SparkSession:
    """Build (or fetch existing) SparkSession wired for S3A access.

    driver_memory is set explicitly rather than left at Spark's 1g default.
    In local[*] mode there are no separate executor JVMs -- the driver does
    everything -- so this single value is the entire memory budget for the
    ETL. Note that raising it is not free on a small machine: the JVM
    reserves the heap up front, and the RAG stage separately needs ~1.5-2GB
    for torch. Two Spark sessions alive at once (e.g. a Jupyter kernel plus
    a CLI run) each take this much again.
    """

    aws_access_key = os.environ["AWS_ACCESS_KEY_ID"]
    aws_secret_key = os.environ["AWS_SECRET_ACCESS_KEY"]
    aws_region = os.environ.get("AWS_DEFAULT_REGION", "ap-south-1")

    spark = (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        .config(
            "spark.jars.packages",
            f"org.apache.hadoop:hadoop-aws:{HADOOP_AWS_VERSION},"
            f"com.amazonaws:aws-java-sdk-bundle:{AWS_SDK_VERSION}",
        )
        .config("spark.hadoop.fs.s3a.access.key", aws_access_key)
        .config("spark.hadoop.fs.s3a.secret.key", aws_secret_key)
        .config("spark.hadoop.fs.s3a.endpoint.region", aws_region)
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                 "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
        .config("spark.driver.memory", driver_memory)
        .config("spark.driver.maxResultSize", driver_max_result_size)
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    return spark


def s3a_path(bucket: str, key_prefix: str) -> str:
    """Build a clean s3a:// path from bucket + key prefix."""
    key_prefix = key_prefix.strip("/")
    return f"s3a://{bucket}/{key_prefix}"


if __name__ == "__main__":
    # Quick smoke test: reads nothing, just confirms the session builds
    # and S3A jars resolve correctly.
    spark = get_spark_session()
    print("Spark version:", spark.version)
    print("Session created successfully with S3A support.")
    spark.stop()