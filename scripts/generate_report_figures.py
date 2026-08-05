"""
scripts/generate_report_figures.py

Generates the two model figures the project report needs, as PNG files ready to
drop into Overleaf:

    report_figures/confusion_matrix.png
    report_figures/feature_importance.png

Reproduces the exact train/test split used by train.py, loads the saved model
from S3, and evaluates on the held-out test period -- so the numbers here match
the numbers in the training log rather than being recomputed differently.

    python scripts/generate_report_figures.py
    python scripts/generate_report_figures.py --top 25    # more features shown
"""

import argparse
import io
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.ml.train import FEATURE_COLUMNS, prepare_train_test
from src.utils.config import load_config
from src.utils.logging_config import setup_logging
from src.utils.s3_io import get_bytes, model_key, read_hive_partitioned_parquet_s3

OUT_DIR = REPO_ROOT / "report_figures"

NAVY = "#12395c"
MIDBLUE = "#4682b4"


def _load_bundle(config):
    bucket = config["s3"]["bucket"]
    version = config["ml"]["model_version"]
    raw = get_bytes(model_key(version), bucket)
    return joblib.load(io.BytesIO(raw))


def plot_confusion(y_true, y_pred, labels, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=axes[0], cmap="Blues", colorbar=False, values_format="d"
    )
    axes[0].set_title("Counts", fontsize=11)
    axes[0].grid(False)

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    ConfusionMatrixDisplay(cm_norm, display_labels=labels).plot(
        ax=axes[1], cmap="Blues", colorbar=False, values_format=".2f"
    )
    axes[1].set_title("Row-normalised (recall per class)", fontsize=11)
    axes[1].grid(False)

    for ax in axes:
        ax.set_xlabel("Predicted label", fontsize=10)
        ax.set_ylabel("True label", fontsize=10)

    fig.suptitle(
        f"Confusion matrix on held-out test period  "
        f"(accuracy {accuracy_score(y_true, y_pred):.3f}, "
        f"macro-F1 {f1_score(y_true, y_pred, average='macro'):.3f})",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_importance(model, feature_names, out_path, top_n=20):
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        xlabel = "Feature importance (mean decrease in impurity)"
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_).mean(axis=0)
        xlabel = "Mean absolute coefficient (across classes)"
    else:
        print("  model exposes neither feature_importances_ nor coef_ -- skipping")
        return

    order = np.argsort(values)[-top_n:]
    names = [feature_names[i] for i in order]
    vals = values[order]

    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.32 * top_n)))
    ax.barh(range(len(vals)), vals, color=MIDBLUE, edgecolor=NAVY, height=0.72)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(f"Top {top_n} features by importance", fontsize=12)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=20,
                        help="Number of features in the importance plot")
    args = parser.parse_args()

    setup_logging()
    OUT_DIR.mkdir(exist_ok=True)

    config = load_config()
    ml_config = config["ml"]

    print("Loading feature table from S3...")
    df = read_hive_partitioned_parquet_s3(
        config["s3"]["paths"]["processed_features"],
        config["s3"]["bucket"],
        partition_col="ticker",
    )
    print(f"  {len(df):,} rows")

    print("Reproducing the chronological split...")
    _, X_test, _, y_test, _, encoder = prepare_train_test(
        df,
        ml_config["train_test_split"],
        ml_config["random_state"],
        ml_config.get("train_test_gap_days", 30),
    )

    print("Loading trained model from S3...")
    bundle = _load_bundle(config)
    model = bundle["model"]
    labels = bundle["labels"]

    y_pred = encoder.inverse_transform(model.predict(X_test))
    y_true = encoder.inverse_transform(y_test)

    print("\nClassification report:\n")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    print("Generating figures...")
    plot_confusion(y_true, y_pred, labels, OUT_DIR / "confusion_matrix.png")
    plot_importance(model, FEATURE_COLUMNS, OUT_DIR / "feature_importance.png", args.top)

    print(f"\nDone. Both PNGs are in: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
