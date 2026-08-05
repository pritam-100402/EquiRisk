"""
src/pipeline/orchestrator.py

Single entrypoint that stitches the full pipeline together, in order:
  ingest prices -> ingest news -> ETL join -> sentiment scoring ->
  feature engineering -> ML training -> ML live prediction -> RAG index refresh

Sentiment runs BEFORE feature engineering because each ETL stage reads one
S3 prefix and writes the next one in the chain (base -> sentiment ->
features), and feature engineering produces the final table. See the
paths block in config.yaml.

This is what BOTH the Streamlit "Refresh Pipeline" button and the CLI
script (scripts/run_pipeline_cli.py) call. Neither of them should
duplicate this logic -- they only call run_full_pipeline().

Each stage function is a thin wrapper that imports the real work from
src/ingestion/, src/etl/, src/ml/, src/rag/ -- this file owns ORDER,
status reporting, and error handling, not the stage internals.

Note: ML training re-trains and overwrites the saved model on every
single run, including every dashboard "Refresh Pipeline" click. That's
fine for a course project's data volume/scale, but if you ever want a
faster refresh that skips retraining (e.g. only re-predict on new
data with the existing model), pull stage_ml_train out of stage_defs
below and run it separately/less often instead.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger("equirisk.orchestrator")


@dataclass
class StageResult:
    name: str
    success: bool
    duration_sec: float
    message: str = ""


@dataclass
class PipelineRunResult:
    stages: list = field(default_factory=list)
    overall_success: bool = True

    def add(self, result: StageResult):
        self.stages.append(result)
        if not result.success:
            self.overall_success = False


def _run_stage(name: str, fn: Callable, status_callback: Optional[Callable] = None) -> StageResult:
    """Run one stage, time it, catch errors so one failed stage doesn't
    kill the whole run silently -- it's recorded and surfaced instead."""
    if status_callback:
        status_callback(f"Running: {name}...")
    logger.info(f"--- Stage start: {name} ---")
    t0 = time.time()
    try:
        fn()
        duration = time.time() - t0
        logger.info(f"--- Stage OK: {name} ({duration:.1f}s) ---")
        return StageResult(name=name, success=True, duration_sec=duration)
    except Exception as e:
        duration = time.time() - t0
        logger.exception(f"--- Stage FAILED: {name} ---")
        return StageResult(name=name, success=False, duration_sec=duration, message=str(e))


def stage_ingest_prices():
    from src.ingestion.fetch_prices import fetch_all_tickers_prices
    fetch_all_tickers_prices()


def stage_ingest_news():
    from src.ingestion.fetch_news import fetch_all_tickers_news
    fetch_all_tickers_news()


def stage_etl():
    from src.etl.clean_transform import run_etl
    run_etl()


def stage_feature_engineering():
    from src.etl.feature_engineering import run_feature_engineering
    run_feature_engineering()


def stage_sentiment():
    from src.etl.sentiment import run_sentiment_scoring
    run_sentiment_scoring()


def stage_ml_train():
    from src.ml.train import run_training
    run_training()


def stage_ml_predict():
    from src.ml.predict import run_inference
    run_inference()


def stage_rag_refresh():
    from src.rag.build_index import refresh_all_indices
    refresh_all_indices()


def run_full_pipeline(status_callback: Optional[Callable] = None) -> PipelineRunResult:
    """Runs all stages in order. status_callback(str) is optional --
    Streamlit passes something like st.status().update(label=...) so the
    dashboard shows live progress; the CLI script can pass print instead.

    Stages after a failure are skipped by design (ETL can't run on data
    that failed to ingest) -- check result.overall_success and
    result.stages[i].message to see what broke.
    """
    result = PipelineRunResult()

    stage_defs = [
        ("Ingest prices", stage_ingest_prices),
        ("Ingest news", stage_ingest_news),
        ("ETL (Spark) -- join", stage_etl),
        ("ETL (Spark) -- sentiment", stage_sentiment),
        ("ETL (Spark) -- feature engineering", stage_feature_engineering),
        ("ML training", stage_ml_train),
        ("ML risk prediction", stage_ml_predict),
        ("RAG index refresh", stage_rag_refresh),
    ]

    for name, fn in stage_defs:
        stage_result = _run_stage(name, fn, status_callback)
        result.add(stage_result)
        if not stage_result.success:
            logger.warning(f"Stopping pipeline after failure in: {name}")
            break

    if status_callback:
        status_callback(
            "Pipeline complete." if result.overall_success else "Pipeline failed -- see logs."
        )

    return result


if __name__ == "__main__":
    from src.utils.logging_config import setup_logging

    setup_logging()
    run_result = run_full_pipeline(status_callback=print)
    for s in run_result.stages:
        status = "OK" if s.success else "FAILED"
        print(f"[{status}] {s.name} ({s.duration_sec:.1f}s) {s.message}")