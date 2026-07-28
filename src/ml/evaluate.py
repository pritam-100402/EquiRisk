"""
src/ml/evaluate.py

Shared metric/plotting helpers used by both src/ml/train.py (production
training script) and notebooks/04_model_training_comparison.ipynb (where
models are compared and plotted). Keeping this logic
in one place means the notebook's comparison numbers and the production
training script's numbers can never quietly drift apart.
"""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)


def classification_metrics(y_true, y_pred, labels: list) -> dict:
    """Core metrics for the Low/Medium/High risk classification task.
    Uses macro averaging since the risk buckets should be treated as
    equally important even if one class is less frequent."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", labels=labels),
        "precision_macro": precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
    }


def confusion_matrix_df(y_true, y_pred, labels: list) -> pd.DataFrame:
    """Returns the confusion matrix as a labeled DataFrame -- easier to
    read/plot in a notebook than a raw numpy array."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])


def full_report(y_true, y_pred, labels: list) -> str:
    """Sklearn's per-class precision/recall/f1 report as text -- handy
    to print directly in a notebook cell."""
    return classification_report(y_true, y_pred, labels=labels, zero_division=0)


def compare_models(results: dict, labels: list) -> pd.DataFrame:
    """Builds a comparison table across candidate models.

    results: {"model_name": (y_true, y_pred), ...}
    Returns a DataFrame with one row per model and one column per metric
    -- the table notebook 04 displays to justify the selected model."""
    rows = []
    for model_name, (y_true, y_pred) in results.items():
        metrics = classification_metrics(y_true, y_pred, labels)
        metrics["model"] = model_name
        rows.append(metrics)
    df = pd.DataFrame(rows).set_index("model")
    return df.sort_values("f1_macro", ascending=False)


def feature_importance_df(model, feature_names: list, top_n: int = 20) -> pd.DataFrame:
    """Works for tree-based models (RandomForest, XGBoost, LightGBM)
    that expose feature_importances_. Returns top_n features sorted
    descending -- plot this in the notebook to sanity-check that the
    model is leaning on sensible features (volatility/sentiment) rather
    than something spurious."""
    if not hasattr(model, "feature_importances_"):
        raise ValueError(f"{type(model).__name__} has no feature_importances_ attribute")

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    return df.head(top_n)