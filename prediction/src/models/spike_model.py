"""
LightGBM spike prediction model for lead_spike_60 target.

Binary classifier predicting whether an RTM price spike will occur
within the next 60 minutes, trained per settlement point with
time-series splits.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

LABEL_COLS = ["spike_event", "lead_spike_60", "regime"]
TARGET = "lead_spike_60"

# Time-series split boundaries (UTC)
TRAIN_END = pd.Timestamp("2025-01-01", tz="UTC")
VAL_END = pd.Timestamp("2026-01-01", tz="UTC")

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "spike"
FEATURE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "spike_features"


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Return feature columns (everything except labels)."""
    return [c for c in df.columns if c not in LABEL_COLS]


def load_data(sp: str, feature_dir: Path = FEATURE_DIR) -> pd.DataFrame:
    """Load parquet for a settlement point."""
    path = feature_dir / f"{sp}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No feature file: {path}")
    df = pd.read_parquet(path)
    # Drop rows where target is NaN
    df = df.dropna(subset=[TARGET])
    return df


def split_data(
    df: pd.DataFrame,
    train_end: pd.Timestamp = TRAIN_END,
    val_end: pd.Timestamp = VAL_END,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train/val/test split. Index must be datetime."""
    train = df[df.index < train_end]
    val = df[(df.index >= train_end) & (df.index < val_end)]
    test = df[df.index >= val_end]
    return train, val, test


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    params: Optional[Dict] = None,
    num_boost_round: int = 2000,
    early_stopping_rounds: int = 50,
) -> lgb.Booster:
    """Train a LightGBM model with early stopping on validation set."""
    # Calculate class imbalance ratio
    pos = (train_df[TARGET] == 1).sum()
    neg = (train_df[TARGET] == 0).sum()
    scale_pos_weight = neg / max(pos, 1)

    default_params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 50,
        "scale_pos_weight": scale_pos_weight,
        "verbose": -1,
    }
    if params:
        default_params.update(params)

    logger.info(
        "Training: {} rows, pos_rate={:.4f}, scale_pos_weight={:.1f}",
        len(train_df), pos / max(pos + neg, 1), scale_pos_weight,
    )

    train_ds = lgb.Dataset(train_df[feature_cols], label=train_df[TARGET])
    val_ds = lgb.Dataset(val_df[feature_cols], label=val_df[TARGET], reference=train_ds)

    model = lgb.train(
        default_params,
        train_ds,
        num_boost_round=num_boost_round,
        valid_sets=[val_ds],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds),
            lgb.log_evaluation(100),
        ],
    )
    return model


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict:
    """Compute classification metrics for spike prediction."""
    metrics = {}
    if len(np.unique(y_true)) < 2:
        logger.warning("Only one class in y_true, metrics will be limited")
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))

    # Precision at recall >= 0.5
    if len(np.unique(y_true)) >= 2:
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        mask = recall >= 0.5
        if mask.any():
            metrics["precision_at_recall_50"] = float(precision[mask].max())
        else:
            metrics["precision_at_recall_50"] = 0.0
    else:
        metrics["precision_at_recall_50"] = float("nan")

    y_pred = (y_prob >= threshold).astype(int)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    metrics["precision"] = float(tp / max(tp + fp, 1))
    metrics["recall"] = float(tp / max(tp + fn, 1))
    metrics["n_positive"] = int(y_true.sum())
    metrics["n_total"] = int(len(y_true))

    return metrics


def compute_event_recall(
    timestamps: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    gap_minutes: int = 60,
) -> Dict:
    """
    Event-level recall: group consecutive lead_spike_60=1 intervals into events,
    count how many events had at least one prediction above threshold.
    """
    pos_mask = y_true == 1
    if not pos_mask.any():
        return {"n_events": 0, "events_detected": 0, "event_recall": float("nan")}

    # Find event boundaries: gaps > gap_minutes between positive intervals
    pos_times = timestamps[pos_mask]
    pos_probs = y_prob[pos_mask]

    time_diffs = pos_times[1:] - pos_times[:-1]
    gap_threshold = pd.Timedelta(minutes=gap_minutes)

    # Assign event IDs
    event_ids = np.zeros(len(pos_times), dtype=int)
    event_id = 0
    for i in range(1, len(pos_times)):
        if time_diffs[i - 1] > gap_threshold:
            event_id += 1
        event_ids[i] = event_id

    n_events = event_id + 1
    events_detected = 0
    for eid in range(n_events):
        event_probs = pos_probs[event_ids == eid]
        if (event_probs >= threshold).any():
            events_detected += 1

    return {
        "n_events": n_events,
        "events_detected": events_detected,
        "event_recall": events_detected / n_events if n_events > 0 else float("nan"),
    }


def feature_importance(model: lgb.Booster, top_n: int = 10) -> List[Tuple[str, float]]:
    """Return top-N features by importance (gain)."""
    names = model.feature_name()
    importance = model.feature_importance(importance_type="gain")
    pairs = sorted(zip(names, importance), key=lambda x: x[1], reverse=True)
    return pairs[:top_n]
