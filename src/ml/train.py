"""
src/ml/train.py

Trains and compares the candidate models defined in config.yaml
(ml.candidate_models) on the feature table produced by the ETL stage,
picks the best one by macro F1, and saves it to S3
(models/risk_model_{version}/model.pkl).

The actual "which model is best and why" exploration/plotting belongs in
notebooks/04_model_training_comparison.ipynb -- that notebook should
import train_all_candidates() and compare_models() from here/evaluate.py
rather than reimplementing training, so the notebook's numbers and this
script's chosen model are guaranteed to match.
"""

import io
import logging

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import lightgbm as lgb

from src.ml.evaluate import compare_models
from src.utils.s3_io import read_hive_partitioned_parquet_s3, put_bytes, model_key

from src.utils.config import load_config as _load_config

logger = logging.getLogger("equirisk.ml.train")

# Every feature here must be SCALE-FREE -- a ratio, a rate, or a bounded
# index -- because all 150 tickers are pooled into one model with one global
# scaler. Raw price levels (ma_20d, macd_line) were removed for exactly this
# reason: MRF trades near Rs 150,000 and IDEA near Rs 10, so those columns
# encoded which company a row belonged to rather than anything about its risk.
FEATURE_COLUMNS = [
    # --- Returns and realised volatility, multiple horizons ---------------
    # volatility_30d is deliberately matched to the label horizon: the best
    # single predictor of the next 30 days is usually the last 30.
    "daily_return",
    "volatility_5d", "volatility_10d", "volatility_20d",
    "volatility_30d", "volatility_60d", "volatility_90d",

    # --- Range-based estimators (use the OHLC, not just close) ------------
    "parkinson_vol_20d", "garman_klass_vol_20d",

    # --- Volatility dynamics ---------------------------------------------
    "vol_ratio_20_60", "vol_ratio_20_90",
    "vol_of_vol_60d", "vol_of_vol_ratio",

    # --- Downside / tail risk --------------------------------------------
    "downside_vol_20d", "downside_ratio",
    "return_skew_60d", "return_kurt_60d",
    "extreme_move_count_20d",

    # --- Overnight information arrival -----------------------------------
    "overnight_gap_vol_20d", "avg_abs_gap_20d",

    # --- Market regime and sensitivity -----------------------------------
    "market_vol_20d", "rel_vol_20d",
    "beta_60d", "corr_market_60d",

    # --- Price position and momentum (all scale-free) ---------------------
    "ma_ratio_20d", "ma_ratio_60d", "ma_ratio_90d",
    "drawdown_from_peak", "max_drawdown_60d",
    "pct_of_52w_range", "momentum_20d", "momentum_60d",

    # --- Oscillators ------------------------------------------------------
    "rsi_14", "macd_norm", "macd_signal_norm",

    # --- Liquidity and volume --------------------------------------------
    "volume_ratio", "amihud_illiq_20d",

    # --- News sentiment ---------------------------------------------------
    # Retained because the pipeline computes it and the dashboard shows it,
    # but near-constant across most of the training period: Google News RSS
    # returns roughly 30 days of history, not five years. Tree models will
    # report these as unused.
    "daily_sentiment", "sentiment_3d_avg", "article_count",

    # --- Cross-sectional rank features ------------------------------------
    # The label is a rank within each date, so these put the features in the
    # same space as the target rather than making the model infer the
    # ordering from absolute levels on every date.
    "xs_rank_volatility_20d", "xs_rank_volatility_60d", "xs_rank_volatility_90d",
    "xs_rank_parkinson_vol_20d", "xs_rank_garman_klass_vol_20d",
    "xs_rank_downside_vol_20d", "xs_rank_vol_of_vol_60d",
    "xs_rank_beta_60d", "xs_rank_amihud_illiq_20d",

    # Rank persistence -- the property the cross-sectional target relies on.
    "xs_rank_mean_60d", "xs_rank_std_60d", "xs_rank_drift",
]
LABEL_COLUMN = "risk_label"

MODEL_REGISTRY = {
    "logistic_regression": lambda: LogisticRegression(max_iter=1000),
    "random_forest": lambda: RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42),
    "xgboost": lambda: xgb.XGBClassifier(n_estimators=300, max_depth=6, eval_metric="mlogloss", random_state=42),
    "lightgbm": lambda: lgb.LGBMClassifier(n_estimators=300, max_depth=6, random_state=42),
}




def load_feature_table(bucket: str, processed_prefix: str) -> pd.DataFrame:
    """Reads all per-ticker parquet partitions (reconstructing the
    'ticker' column Spark strips out into the folder name) and
    concatenates into one pandas DataFrame for scikit-learn/XGBoost/
    LightGBM training. This is the one place a full Spark->pandas
    handoff happens -- fine at this data scale (150 tickers x ~5 years
    of daily rows), but if the dataset grows much larger, train
    directly against Spark ML instead."""
    df = read_hive_partitioned_parquet_s3(processed_prefix, bucket, partition_col="ticker")
    logger.info(f"Loaded feature table: {df.shape[0]} rows")
    return df


def training_cutoff_date(df: pd.DataFrame, test_size: float):
    """The date separating the training period from the test period.

    Mirrors feature_engineering.training_cutoff_date() exactly: index into
    the sorted distinct dates, so every ticker is cut at the same calendar
    point and the risk-label quantiles fitted during feature engineering
    were fitted on precisely this training period. Change one, change both.
    """
    dates = pd.Series(sorted(pd.unique(df["date"])))
    if dates.empty:
        raise ValueError("Cannot compute a training cutoff on an empty table")
    idx = max(int(len(dates) * (1.0 - test_size)) - 1, 0)
    return dates.iloc[idx]


def prepare_train_test(df: pd.DataFrame, test_size: float, random_state: int,
                       gap_days: int = 30):
    """Splits CHRONOLOGICALLY, not randomly.

    A random split is wrong for this dataset in a way that quietly
    inflates every metric. The label looks 30 trading days forward and the
    features are rolling windows over the preceding 20-90 days, so two
    rows a day apart share almost all of their underlying information. A
    random split scatters those near-duplicate rows across train and test,
    and the model gets credit for recognising rows it has effectively
    already seen. The earliest `1 - test_size` of the timeline trains; the
    latest `test_size` tests.

    `gap_days` then drops a further block of trading days between the two
    periods (an embargo). Without it, the last training rows have labels
    computed from forward windows that reach into the test period -- the
    boundary itself leaks. Should be >= the label horizon.

    Rows with a null label are dropped: those are the tail rows per ticker
    with no complete forward window (see
    feature_engineering.add_forward_volatility_label), which is exactly
    the set predict.py scores live. Rows with any null feature are dropped
    rather than imputed -- "we train only on complete rows" is a cleaner
    thing to defend than a silent imputation choice.

    Labels are integer-encoded once here rather than per-model: XGBoost's
    sklearn wrapper requires integer classes, the others accept strings,
    and encoding once guarantees every model trains on identical labels.
    The encoder is returned so predictions decode back to Low/Medium/High.

    `random_state` no longer affects the split (there is nothing random
    left in it) and is retained only so model seeding stays configurable.
    """
    df = df.dropna(subset=[LABEL_COLUMN] + FEATURE_COLUMNS).sort_values("date")

    cutoff = training_cutoff_date(df, test_size)

    trading_days = pd.Series(sorted(pd.unique(df["date"])))
    cutoff_idx = int(trading_days[trading_days == cutoff].index[0])
    embargo_end_idx = min(cutoff_idx + gap_days, len(trading_days) - 1)
    test_start = trading_days.iloc[embargo_end_idx]

    train_df = df[df["date"] <= cutoff]
    test_df = df[df["date"] > test_start]

    if train_df.empty or test_df.empty:
        raise RuntimeError(
            f"Chronological split produced an empty side "
            f"(train={len(train_df)}, test={len(test_df)}). "
            f"Check test_size={test_size} and gap_days={gap_days}."
        )

    logger.info(
        f"Chronological split: train <= {cutoff} ({len(train_df)} rows), "
        f"test > {test_start} ({len(test_df)} rows), "
        f"{gap_days}-day embargo between them"
    )

    encoder = LabelEncoder()
    encoder.fit(df[LABEL_COLUMN])
    y_train = encoder.transform(train_df[LABEL_COLUMN])
    y_test = encoder.transform(test_df[LABEL_COLUMN])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_df[FEATURE_COLUMNS])
    X_test_scaled = scaler.transform(test_df[FEATURE_COLUMNS])

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, encoder


def train_all_candidates(X_train, y_train, X_test, candidate_names: list, encoder: LabelEncoder) -> dict:
    """Trains every candidate model listed in config.yaml. Returns
    {"model_name": (fitted_model, y_pred_labels)} -- y_pred_labels is
    already decoded back to the original string labels via `encoder`,
    so callers (evaluate.compare_models, etc.) never need to know
    encoding happened."""
    results = {}
    for name in candidate_names:
        if name not in MODEL_REGISTRY:
            logger.warning(f"Unknown model '{name}' in candidate_models -- skipping")
            continue
        logger.info(f"Training {name}...")
        model = MODEL_REGISTRY[name]()
        model.fit(X_train, y_train)
        y_pred_encoded = model.predict(X_test)
        y_pred_labels = encoder.inverse_transform(y_pred_encoded)
        results[name] = (model, y_pred_labels)
    return results


def run_training(config_path: str = None) -> str:
    """Main training entrypoint. Trains all candidates, picks the best
    by macro F1, saves it + the scaler to S3. Returns the winning model
    name (mainly useful for logging/notebook display)."""
    config = _load_config(config_path)
    bucket = config["s3"]["bucket"]
    processed_prefix = config["s3"]["paths"]["processed_features"]
    ml_config = config["ml"]
    labels = ml_config["risk_label"]["buckets"]

    df = load_feature_table(bucket, processed_prefix)
    X_train, X_test, y_train, y_test, scaler, encoder = prepare_train_test(
        df,
        ml_config["train_test_split"],
        ml_config["random_state"],
        ml_config.get("train_test_gap_days", 30),
    )

    trained = train_all_candidates(X_train, y_train, X_test, ml_config["candidate_models"], encoder)
    if not trained:
        raise RuntimeError("No candidate models were successfully trained")

    y_test_labels = encoder.inverse_transform(y_test)
    comparison_input = {name: (y_test_labels, y_pred) for name, (model, y_pred) in trained.items()}
    comparison_table = compare_models(comparison_input, labels)
    logger.info(f"Model comparison:\n{comparison_table}")

    # Soft-voting ensemble: average the predicted class probabilities
    # across all candidates. The models make different errors -- a linear
    # model and a tree ensemble fail on different rows -- so averaging their
    # probabilities usually beats every individual member by a point or two.
    # Entered as a candidate and selected only if it actually wins.
    try:
        import numpy as np
        probas = [m.predict_proba(X_test) for m, _ in trained.values()]
        ensemble_pred = encoder.inverse_transform(np.mean(probas, axis=0).argmax(axis=1))
        y_test_labels = encoder.inverse_transform(y_test)

        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        comparison_table.loc["ensemble_soft_vote"] = {
            "accuracy": accuracy_score(y_test_labels, ensemble_pred),
            "f1_macro": f1_score(y_test_labels, ensemble_pred, average="macro"),
            "precision_macro": precision_score(y_test_labels, ensemble_pred,
                                               average="macro", zero_division=0),
            "recall_macro": recall_score(y_test_labels, ensemble_pred, average="macro"),
        }
        comparison_table = comparison_table.sort_values("f1_macro", ascending=False)
        logger.info(f"Model comparison (with ensemble):\n{comparison_table}")
    except Exception as e:
        logger.warning(f"Ensemble evaluation skipped: {e}")

    # The ensemble is not a single fitted estimator, so it cannot be
    # serialised through the existing bundle format. If it wins, that is
    # worth reporting -- but the best single model is what gets deployed.
    ranked = [n for n in comparison_table.index if n != "ensemble_soft_vote"]
    if comparison_table.index[0] == "ensemble_soft_vote":
        logger.info(
            "Soft-vote ensemble scored highest; deploying the best single "
            "model instead, since the bundle format holds one estimator."
        )
    best_model_name = ranked[0]
    best_model, _ = trained[best_model_name]
    logger.info(f"Best model: {best_model_name} (f1_macro={comparison_table.loc[best_model_name, 'f1_macro']:.4f})")

    version = ml_config["model_version"]
    model_buf = io.BytesIO()
    joblib.dump(
        {"model": best_model, "scaler": scaler, "encoder": encoder, "features": FEATURE_COLUMNS, "labels": labels},
        model_buf,
    )
    put_bytes(model_key(version, "model.pkl"), model_buf.getvalue(), bucket)

    logger.info(f"Saved {best_model_name} -> s3://{bucket}/{model_key(version)}")
    return best_model_name


if __name__ == "__main__":
    from src.utils.logging_config import setup_logging

    setup_logging()
    run_training()