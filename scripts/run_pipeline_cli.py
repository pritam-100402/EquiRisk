"""
scripts/run_pipeline_cli.py

Command-line entrypoint for the full EquiRisk pipeline.

Deliberately thin. All stage ordering, error handling, and status
reporting live in src/pipeline/orchestrator.py -- this file only parses
arguments, configures logging, and prints a summary. The Streamlit
dashboard's "Refresh Pipeline" button calls the exact same
run_full_pipeline(), so the two can never drift apart.

Run from the repository root:

    python scripts/run_pipeline_cli.py                  # full pipeline
    python scripts/run_pipeline_cli.py --list-stages    # show stage order
    python scripts/run_pipeline_cli.py --stage etl      # run one stage only
    python scripts/run_pipeline_cli.py --verbose        # DEBUG logging

Exit code is 0 if every stage succeeded, 1 otherwise -- so this is safe
to drop into cron or a CI job.
"""

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.orchestrator import (
    run_full_pipeline,
    stage_ingest_prices,
    stage_ingest_news,
    stage_etl,
    stage_sentiment,
    stage_feature_engineering,
    stage_ml_train,
    stage_ml_predict,
    stage_rag_refresh,
)
from src.utils.logging_config import setup_logging

INDIVIDUAL_STAGES = {
    "prices": ("Ingest prices", stage_ingest_prices),
    "news": ("Ingest news", stage_ingest_news),
    "etl": ("ETL (Spark) -- join", stage_etl),
    "sentiment": ("ETL (Spark) -- sentiment", stage_sentiment),
    "features": ("ETL (Spark) -- feature engineering", stage_feature_engineering),
    "train": ("ML training", stage_ml_train),
    "predict": ("ML risk prediction", stage_ml_predict),
    "rag": ("RAG index refresh", stage_rag_refresh),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the EquiRisk data pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        choices=list(INDIVIDUAL_STAGES),
        help="Run a single stage instead of the full pipeline. Stages are not "
             "independent -- each reads what the previous one wrote -- so this "
             "is for re-running a failed stage, not for arbitrary ordering.",
    )
    parser.add_argument(
        "--list-stages",
        action="store_true",
        help="Print the stage order and exit without running anything.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="DEBUG-level logging.",
    )
    return parser.parse_args()


def _print_summary(stages: list) -> None:
    print("\n" + "=" * 68)
    print("PIPELINE SUMMARY")
    print("=" * 68)
    for s in stages:
        status = "OK    " if s.success else "FAILED"
        print(f"[{status}] {s.name:<40} {s.duration_sec:>7.1f}s")
        if s.message:
            print(f"           -> {s.message}")
    print("=" * 68)


def main() -> int:
    args = _parse_args()

    if args.list_stages:
        print("Pipeline stages, in order:\n")
        for key, (label, _) in INDIVIDUAL_STAGES.items():
            print(f"  {key:<10} {label}")
        return 0

    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    if args.stage:
        label, fn = INDIVIDUAL_STAGES[args.stage]
        print(f"Running single stage: {label}")
        try:
            fn()
        except Exception as e:
            print(f"\n[FAILED] {label}: {e}")
            return 1
        print(f"\n[OK] {label}")
        return 0

    result = run_full_pipeline(status_callback=print)
    _print_summary(result.stages)

    return 0 if result.overall_success else 1


if __name__ == "__main__":
    sys.exit(main())
