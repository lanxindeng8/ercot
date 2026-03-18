#!/usr/bin/env python3
"""Spike detection model training — CatBoost binary classifier with Optuna tuning."""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT / "checkpoints"
DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "training"

SETTLEMENT_POINTS = ["hb_west", "hb_houston", "hb_north", "hb_south", "lz_lcra", "lz_west"]

TARGET_PRICE = "rtm_lmp"
SPIKE_THRESHOLD = 100.0  # absolute floor ($/MWh)
SPIKE_MULTIPLIER = 3.0   # multiplier on rolling mean
ROLLING_WINDOW = 24      # hours for rolling mean
LABEL_LOOKAHEAD = 1      # predict next-hour spikes

LEAKAGE_PRONE_FEATURES = [
    "rtm_roll_24h_mean", "rtm_roll_24h_std", "rtm_roll_24h_min", "rtm_roll_24h_max",
    "rtm_roll_168h_mean", "rtm_roll_168h_std", "rtm_roll_168h_min", "rtm_roll_168h_max",
    "dam_rtm_spread", "spread_roll_24h_mean", "spread_roll_168h_mean",
]

# Base features from training parquets (80 unified features)
BASE_FEATURES = [
    # Temporal (7)
    "hour_of_day", "day_of_week", "month", "is_weekend", "is_peak_hour",
    "is_holiday", "is_summer",
    # DAM lags (4)
    "dam_lag_1h", "dam_lag_4h", "dam_lag_24h", "dam_lag_168h",
    # RTM lags (4)
    "rtm_lag_1h", "rtm_lag_4h", "rtm_lag_24h", "rtm_lag_168h",
    # DAM rolling (8)
    "dam_roll_24h_mean", "dam_roll_24h_std", "dam_roll_24h_min", "dam_roll_24h_max",
    "dam_roll_168h_mean", "dam_roll_168h_std", "dam_roll_168h_min", "dam_roll_168h_max",
    # RTM rolling (8)
    "rtm_roll_24h_mean", "rtm_roll_24h_std", "rtm_roll_24h_min", "rtm_roll_24h_max",
    "rtm_roll_168h_mean", "rtm_roll_168h_std", "rtm_roll_168h_min", "rtm_roll_168h_max",
    # Cross-market (3)
    "dam_rtm_spread", "spread_roll_24h_mean", "spread_roll_168h_mean",
    # Fuel mix pct (6)
    "wind_pct", "solar_pct", "gas_pct", "nuclear_pct", "coal_pct", "hydro_pct",
    # Ancillary service (13)
    "regdn", "regup", "rrs", "nspin", "ecrs",
    "reg_spread", "total_as_cost",
    "regup_lag_24h", "rrs_lag_24h", "nspin_lag_24h", "total_as_lag_24h",
    "total_as_roll_24h_mean", "total_as_roll_24h_std",
    # RTM components (6)
    "congestion_pct", "loss_pct", "energy_pct",
    "congestion_ma_4h", "congestion_volatility_24h", "high_congestion_flag",
    # Fuel gen MW (15)
    "gas_gen_mw", "gas_cc_gen_mw", "coal_gen_mw", "nuclear_gen_mw",
    "solar_gen_mw", "wind_gen_mw", "hydro_gen_mw", "biomass_gen_mw",
    "total_gen_mw", "renewable_ratio", "thermal_ratio", "net_load_mw",
    "solar_ramp_1h", "wind_ramp_1h", "gas_ramp_1h",
    # Cross-domain (6)
    "dam_as_ratio", "reg_spread_roll_24h_mean", "ecrs_lag_24h",
    "gas_cc_share", "wind_ramp_4h", "solar_ramp_4h",
]

# Nullable columns (optional data sources) — filled with 0 instead of dropping
NULLABLE_COLS = [c for c in BASE_FEATURES if c not in [
    "hour_of_day", "day_of_week", "month", "is_weekend", "is_peak_hour",
    "is_holiday", "is_summer",
    "dam_lag_1h", "dam_lag_4h", "dam_lag_24h", "dam_lag_168h",
    "rtm_lag_1h", "rtm_lag_4h", "rtm_lag_24h", "rtm_lag_168h",
    "dam_roll_24h_mean", "dam_roll_24h_std", "dam_roll_24h_min", "dam_roll_24h_max",
    "dam_roll_168h_mean", "dam_roll_168h_std", "dam_roll_168h_min", "dam_roll_168h_max",
    "rtm_roll_24h_mean", "rtm_roll_24h_std", "rtm_roll_24h_min", "rtm_roll_24h_max",
    "rtm_roll_168h_mean", "rtm_roll_168h_std", "rtm_roll_168h_min", "rtm_roll_168h_max",
    "dam_rtm_spread", "spread_roll_24h_mean", "spread_roll_168h_mean",
]]

# Spike-specific engineered features (added during preprocessing)
SPIKE_FEATURES = [
    "price_accel",           # 2nd derivative of RTM price
    "volatility_regime",     # rolling_std / rolling_mean
    "hour_spike_prob",       # historical spike rate per hour
    "price_momentum",        # rtm_lmp - rtm_roll_24h_mean
    "price_ratio_to_mean",   # rtm_lmp / rtm_roll_24h_mean
    "rtm_range_24h",         # rtm_roll_24h_max - rtm_roll_24h_min
]

FEATURE_COLS = BASE_FEATURES + SPIKE_FEATURES

CORE_FEATURES = [c for c in BASE_FEATURES if c not in NULLABLE_COLS]

CAT_FEATURES = [
    "hour_of_day", "day_of_week", "month", "is_weekend", "is_peak_hour",
    "is_holiday", "is_summer",
]


# ---------------------------------------------------------------------------
# Label generation
# ---------------------------------------------------------------------------

def generate_spike_labels(df: pd.DataFrame) -> pd.Series:
    """Generate binary spike labels.

    Label at time t predicts whether the next hour spikes, using only RTM
    history available strictly before time t.
    """
    future_price = df[TARGET_PRICE].shift(-LABEL_LOOKAHEAD)
    rolling_mean = df[TARGET_PRICE].shift(1).rolling(window=ROLLING_WINDOW, min_periods=1).mean()
    threshold = pd.Series(
        np.maximum(SPIKE_THRESHOLD, SPIKE_MULTIPLIER * rolling_mean.fillna(0.0)),
        index=df.index,
    )
    return (future_price > threshold).where(future_price.notna()).astype(float)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def add_spike_features(df: pd.DataFrame, train_hour_probs: dict | None = None) -> pd.DataFrame:
    """Add spike-specific engineered features to the dataframe.

    Args:
        df: DataFrame with base features and rtm_lmp.
        train_hour_probs: Pre-computed hour->spike_prob mapping (from training set).
            If None, computes from the data itself (only valid for training).

    Returns:
        DataFrame with spike features added.
    """
    df = df.copy()

    for col in LEAKAGE_PRONE_FEATURES:
        if col in df.columns:
            df[col] = df[col].shift(1)

    # Price acceleration (2nd derivative): diff of rtm_lag differences
    price_diff1 = df["rtm_lag_1h"] - df["rtm_lag_4h"]
    price_diff2 = df["rtm_lag_4h"] - df["rtm_lag_24h"]
    df["price_accel"] = price_diff1 - price_diff2

    # Volatility regime: rolling_std / rolling_mean
    rolling_mean = df["rtm_roll_24h_mean"].replace(0, np.nan)
    df["volatility_regime"] = df["rtm_roll_24h_std"] / rolling_mean.abs()
    df["volatility_regime"] = df["volatility_regime"].fillna(0)

    # Hour-specific spike probability (historical)
    if train_hour_probs is not None:
        df["hour_spike_prob"] = df["hour_of_day"].map(train_hour_probs).fillna(0)
    else:
        # Compute from this data (training set)
        labels = generate_spike_labels(df)
        hour_probs = labels.groupby(df["hour_of_day"]).mean()
        df["hour_spike_prob"] = df["hour_of_day"].map(hour_probs).fillna(0)

    # Price momentum: last observed price vs trailing rolling mean
    df["price_momentum"] = df["rtm_lag_1h"] - df["rtm_roll_24h_mean"]

    # Price ratio to rolling mean
    df["price_ratio_to_mean"] = df["rtm_lag_1h"] / df["rtm_roll_24h_mean"].replace(0, np.nan)
    df["price_ratio_to_mean"] = df["price_ratio_to_mean"].fillna(1)

    # 24h price range
    df["rtm_range_24h"] = df["rtm_roll_24h_max"] - df["rtm_roll_24h_min"]

    return df


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(sp: str):
    """Load train/val/test parquets, add spike features and labels."""
    sp_dir = DATA_DIR / sp
    dfs = {}
    for split in ("train", "val", "test"):
        path = sp_dir / f"{split}.parquet"
        df = pd.read_parquet(path)
        for col in NULLABLE_COLS:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)
        before = len(df)
        df = df.dropna(subset=CORE_FEATURES + [TARGET_PRICE])
        log.info(f"  {split}: {before} -> {len(df)} rows (dropped {before - len(df)} NaN)")
        dfs[split] = df

    # Add spike features — compute hour probs from training set only
    train_df = dfs["train"]
    train_df = add_spike_features(train_df, train_hour_probs=None)
    train_labels = generate_spike_labels(train_df)
    hour_probs = train_labels.groupby(train_df["hour_of_day"]).mean().to_dict()

    dfs["train"] = train_df
    for split in ("val", "test"):
        dfs[split] = add_spike_features(dfs[split], train_hour_probs=hour_probs)

    # Generate labels
    for split in ("train", "val", "test"):
        dfs[split]["spike"] = generate_spike_labels(dfs[split])
        dfs[split] = dfs[split][dfs[split]["spike"].notna()].copy()
        dfs[split]["spike"] = dfs[split]["spike"].astype(int)

    return dfs["train"], dfs["val"], dfs["test"], hour_probs


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                   y_prob: np.ndarray) -> dict:
    """Compute precision, recall, F1, AUC-ROC, confusion matrix."""
    tn, fp, fn, tp = binary_confusion_counts(y_true, y_pred)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc_roc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc_roc = float("nan")

    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "auc_roc": round(float(auc_roc), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_positive": int(y_true.sum()),
        "n_negative": int((1 - y_true).sum()),
        "positive_rate": round(float(y_true.mean()), 4),
    }


def binary_confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    """Return TN, FP, FN, TP even when only one class is present."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return int(tn), int(fp), int(fn), int(tp)


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def _catboost_objective(trial, X_train, y_train, X_val, y_val, scale_pos_weight):
    """Optuna objective optimizing F1 on validation set."""
    params = {
        "iterations": trial.suggest_int("iterations", 300, 2000),
        "depth": trial.suggest_int("depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "random_strength": trial.suggest_float("random_strength", 0.0, 3.0),
        "scale_pos_weight": scale_pos_weight,
        "loss_function": "Logloss",
        "eval_metric": "F1",
        "verbose": 0,
        "random_seed": 42,
        "allow_writing_files": False,
        "auto_class_weights": None,
    }
    # Threshold tuning for precision-oriented classification
    threshold = trial.suggest_float("threshold", 0.3, 0.8)

    cat_idx = [list(X_train.columns).index(c) for c in CAT_FEATURES if c in X_train.columns]
    train_pool = Pool(X_train, y_train, cat_features=cat_idx)
    val_pool = Pool(X_val, y_val, cat_features=cat_idx)

    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=100, verbose=0)

    probs = model.predict_proba(X_val)[:, 1]
    preds = (probs >= threshold).astype(int)

    # Optimize for F1 but penalize low precision (we want fewer false alarms)
    f1 = f1_score(y_val, preds, zero_division=0)
    prec = precision_score(y_val, preds, zero_division=0)
    # Weighted score: reward high precision
    score = 0.6 * f1 + 0.4 * prec
    return score


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_settlement_point(sp: str, n_trials: int = 50) -> dict:
    """Train spike detection model for one settlement point."""
    log.info(f"=== Training spike detector: {sp.upper()} ===")
    train_df, val_df, test_df, hour_probs = load_data(sp)

    y_train = train_df["spike"].values
    y_val = val_df["spike"].values
    y_test = test_df["spike"].values

    # Use only features that exist in the dataframe
    available_features = [c for c in FEATURE_COLS if c in train_df.columns]
    X_train = train_df[available_features].copy()
    X_val = val_df[available_features].copy()
    X_test = test_df[available_features].copy()

    # Class imbalance stats
    pos_rate = y_train.mean()
    scale_pos_weight = (1 - pos_rate) / max(pos_rate, 1e-6)
    log.info(f"  Spike rate: {pos_rate:.4f} ({y_train.sum()}/{len(y_train)})")
    log.info(f"  scale_pos_weight: {scale_pos_weight:.2f}")

    results = {
        "settlement_point": sp,
        "spike_rate_train": round(float(pos_rate), 4),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "hour_spike_probs": {str(k): round(float(v), 4) for k, v in hour_probs.items()},
        "features_used": available_features,
    }

    # --- Optuna hyperparameter tuning ---
    log.info(f"  Tuning CatBoost ({n_trials} trials)...")
    t0 = time.time()
    study = optuna.create_study(direction="maximize", study_name=f"{sp}_spike")
    study.optimize(
        lambda trial: _catboost_objective(
            trial, X_train, y_train, X_val, y_val, scale_pos_weight
        ),
        n_trials=n_trials,
    )
    tune_time = time.time() - t0
    log.info(f"  Best trial score: {study.best_value:.4f} ({tune_time:.0f}s)")

    # --- Retrain with best params ---
    best_params = {k: v for k, v in study.best_params.items() if k != "threshold"}
    best_threshold = study.best_params["threshold"]
    best_params.update({
        "scale_pos_weight": scale_pos_weight,
        "loss_function": "Logloss",
        "eval_metric": "F1",
        "verbose": 0,
        "random_seed": 42,
        "allow_writing_files": False,
    })

    cat_idx = [list(X_train.columns).index(c) for c in CAT_FEATURES if c in X_train.columns]
    train_pool = Pool(X_train, y_train, cat_features=cat_idx)
    val_pool = Pool(X_val, y_val, cat_features=cat_idx)

    model = CatBoostClassifier(**best_params)
    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=100, verbose=0)

    # --- Predict ---
    val_probs = model.predict_proba(X_val)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]
    val_preds = (val_probs >= best_threshold).astype(int)
    test_preds = (test_probs >= best_threshold).astype(int)

    val_metrics = compute_classification_metrics(y_val, val_preds, val_probs)
    test_metrics = compute_classification_metrics(y_test, test_preds, test_probs)

    results["best_params"] = {
        k: (int(v) if isinstance(v, (np.integer,)) else
            float(v) if isinstance(v, (np.floating,)) else v)
        for k, v in best_params.items()
    }
    results["best_threshold"] = round(float(best_threshold), 4)
    results["val_metrics"] = val_metrics
    results["test_metrics"] = test_metrics
    results["tuning_seconds"] = round(tune_time, 1)

    # --- Save model ---
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = CHECKPOINT_DIR / f"{sp}_spike_catboost.cbm"
    model.save_model(str(model_path))
    results["model_path"] = str(model_path)

    # Save threshold and hour probs for inference
    meta_path = CHECKPOINT_DIR / f"{sp}_spike_meta.json"
    with open(meta_path, "w") as f:
        json.dump({
            "threshold": best_threshold,
            "hour_spike_probs": {str(k): float(v) for k, v in hour_probs.items()},
            "features": available_features,
        }, f, indent=2)

    # Save test predictions for evaluate.py
    np.save(CHECKPOINT_DIR / f"{sp}_spike_test_probs.npy", test_probs)

    # Feature importance
    importance = model.get_feature_importance()
    fi = sorted(zip(available_features, importance), key=lambda x: -x[1])
    results["top_features"] = [
        {"feature": f, "importance": round(float(v), 2)} for f, v in fi[:15]
    ]

    log.info(f"  Val:  P={val_metrics['precision']:.3f} R={val_metrics['recall']:.3f} "
             f"F1={val_metrics['f1']:.3f} AUC={val_metrics['auc_roc']:.3f}")
    log.info(f"  Test: P={test_metrics['precision']:.3f} R={test_metrics['recall']:.3f} "
             f"F1={test_metrics['f1']:.3f} AUC={test_metrics['auc_roc']:.3f}")
    log.info(f"  Threshold: {best_threshold:.3f}")
    log.info(f"  Model saved to {model_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train spike detection models")
    parser.add_argument("--points", nargs="*", default=["hb_west"],
                        help="Settlement points to train (default: hb_west)")
    parser.add_argument("--trials", type=int, default=50,
                        help="Optuna trials (default: 50)")
    args = parser.parse_args()

    all_results = {}
    for sp in args.points:
        result = train_settlement_point(sp, n_trials=args.trials)
        all_results[sp] = result

    # Save training report
    report_path = ROOT / "training_report.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"Training report saved to {report_path}")

    # Print summary
    print("\n" + "=" * 90)
    print(f"{'SP':<12} {'Prec':>8} {'Recall':>8} {'F1':>8} {'AUC':>8} "
          f"{'Thresh':>8} {'Spk Rate':>10} {'TP':>6} {'FP':>6} {'FN':>6}")
    print("-" * 90)
    for sp, r in all_results.items():
        tm = r["test_metrics"]
        cm = tm["confusion_matrix"]
        print(f"{sp:<12} {tm['precision']:>8.3f} {tm['recall']:>8.3f} {tm['f1']:>8.3f} "
              f"{tm['auc_roc']:>8.3f} {r['best_threshold']:>8.3f} "
              f"{r['spike_rate_train']:>10.4f} {cm['tp']:>6} {cm['fp']:>6} {cm['fn']:>6}")
    print("=" * 90)


if __name__ == "__main__":
    main()
