#!/usr/bin/env python3
"""Sprint 2 Task 3: Train & evaluate spread prediction models on unified parquet data.

Trains CatBoost and LightGBM models to predict DAM-RTM spread direction and
magnitude using the unified training pipeline parquets (data/training/hb_west/).

Usage:
    python prediction/models/delta-spread/train_and_eval.py [--settlement-point hb_west]
"""

import argparse
import json
import sys
from pathlib import Path

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "training"
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"

# Columns that are all NaN or are targets (not features)
EXCLUDE_COLS = {
    "delivery_date",
    "dam_lmp",
    "rtm_lmp",
    "dam_rtm_spread",
    "wind_pct",
    "solar_pct",
    "gas_pct",
    "nuclear_pct",
    "coal_pct",
    "hydro_pct",
}

CAT_FEATURES = [
    "hour_of_day",
    "day_of_week",
    "month",
    "is_weekend",
    "is_peak_hour",
    "is_holiday",
    "is_summer",
]


def load_split(sp: str, split: str) -> pd.DataFrame:
    path = DATA_DIR / sp / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return pd.read_parquet(path)


def prepare_features(df: pd.DataFrame):
    """Return (X, y_spread, y_direction) dropping excluded/NaN columns."""
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feature_cols].copy()
    # Drop columns that are entirely NaN
    X = X.dropna(axis=1, how="all")
    y_spread = df["dam_rtm_spread"].values
    y_direction = (y_spread > 0).astype(int)  # 1 = DAM > RTM
    return X, y_spread, y_direction


def compute_regression_metrics(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "directional_accuracy": float(
            np.mean(np.sign(y_pred) == np.sign(y_true))
        ),
    }


def compute_classification_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
    }


def train_catboost_regression(X_train, y_train, X_val, y_val, cat_idx):
    """Train CatBoost regressor for spread magnitude."""
    model = cb.CatBoostRegressor(
        iterations=1500,
        depth=7,
        learning_rate=0.05,
        loss_function="MAE",
        random_seed=42,
        verbose=100,
        early_stopping_rounds=150,
        cat_features=cat_idx,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
    )
    return model


def train_catboost_classifier(X_train, y_train, X_val, y_val, cat_idx):
    """Train CatBoost binary classifier for spread direction."""
    model = cb.CatBoostClassifier(
        iterations=1500,
        depth=7,
        learning_rate=0.05,
        loss_function="Logloss",
        auto_class_weights="Balanced",
        random_seed=42,
        verbose=100,
        early_stopping_rounds=150,
        cat_features=cat_idx,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
    )
    return model


def train_lgbm_regression(X_train, y_train, X_val, y_val, cat_cols):
    """Train LightGBM regressor for spread magnitude."""
    train_ds = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
    val_ds = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols, reference=train_ds)
    params = {
        "objective": "mae",
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "max_depth": 7,
        "min_child_samples": 20,
        "seed": 42,
        "verbose": -1,
    }
    model = lgb.train(
        params,
        train_ds,
        num_boost_round=1500,
        valid_sets=[val_ds],
        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(100)],
    )
    return model


def naive_baseline(y_test, lag_col):
    """Naive baseline: predict spread = last known spread (lag_1h equivalent)."""
    pred = lag_col.values
    valid = ~np.isnan(pred) & ~np.isnan(y_test)
    return compute_regression_metrics(y_test[valid], pred[valid])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--settlement-point", default="hb_west")
    args = parser.parse_args()
    sp = args.settlement_point

    print(f"=== Training spread models for {sp.upper()} ===\n")

    # Load data
    df_train = load_split(sp, "train")
    df_val = load_split(sp, "val")
    df_test = load_split(sp, "test")

    X_train, y_train_spread, y_train_dir = prepare_features(df_train)
    X_val, y_val_spread, y_val_dir = prepare_features(df_val)
    X_test, y_test_spread, y_test_dir = prepare_features(df_test)

    feature_cols = list(X_train.columns)
    cat_idx = [i for i, c in enumerate(feature_cols) if c in CAT_FEATURES]
    cat_cols_present = [c for c in feature_cols if c in CAT_FEATURES]

    # Cast categoricals to int
    for df_x in [X_train, X_val, X_test]:
        for c in cat_cols_present:
            df_x[c] = df_x[c].astype(int)

    print(f"Features: {len(feature_cols)}, Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Spread stats — Train mean: {y_train_spread.mean():.2f}, std: {y_train_spread.std():.2f}")
    print(f"Direction balance — Train positive: {y_train_dir.mean():.1%}, Test positive: {y_test_dir.mean():.1%}\n")

    results = {}

    # --- Naive baseline ---
    print("--- Naive Baseline (dam_lag_1h - rtm_lag_1h) ---")
    lag_spread = df_test["dam_lag_1h"] - df_test["rtm_lag_1h"]
    results["naive_baseline"] = naive_baseline(y_test_spread, lag_spread)
    print(json.dumps(results["naive_baseline"], indent=2), "\n")

    # --- CatBoost Regression ---
    print("--- CatBoost Regression (spread magnitude) ---")
    cb_reg = train_catboost_regression(X_train, y_train_spread, X_val, y_val_spread, cat_idx)
    pred_reg = cb_reg.predict(X_test)
    results["catboost_regression"] = compute_regression_metrics(y_test_spread, pred_reg)
    print(json.dumps(results["catboost_regression"], indent=2), "\n")

    # --- CatBoost Classifier ---
    print("--- CatBoost Classifier (spread direction) ---")
    cb_cls = train_catboost_classifier(X_train, y_train_dir, X_val, y_val_dir, cat_idx)
    pred_cls = cb_cls.predict(X_test).astype(int)
    results["catboost_classifier"] = compute_classification_metrics(y_test_dir, pred_cls)
    pred_proba = cb_cls.predict_proba(X_test)[:, 1]
    print(json.dumps(results["catboost_classifier"], indent=2))
    print(classification_report(y_test_dir, pred_cls, target_names=["DAM<RTM", "DAM>RTM"]), "\n")

    # --- LightGBM Regression ---
    print("--- LightGBM Regression (spread magnitude) ---")
    lgb_reg = train_lgbm_regression(X_train, y_train_spread, X_val, y_val_spread, cat_cols_present)
    pred_lgb = lgb_reg.predict(X_test)
    results["lgbm_regression"] = compute_regression_metrics(y_test_spread, pred_lgb)
    print(json.dumps(results["lgbm_regression"], indent=2), "\n")

    # --- Save checkpoints ---
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    cb_reg_path = CHECKPOINT_DIR / f"{sp}_catboost_regression.cbm"
    cb_cls_path = CHECKPOINT_DIR / f"{sp}_catboost_classifier.cbm"
    lgb_path = CHECKPOINT_DIR / f"{sp}_lgbm_regression.txt"

    cb_reg.save_model(str(cb_reg_path))
    cb_cls.save_model(str(cb_cls_path))
    lgb_reg.save_model(str(lgb_path))
    print(f"Checkpoints saved to {CHECKPOINT_DIR}/")

    # --- Save predictions for backtesting ---
    pred_df = pd.DataFrame({
        "delivery_date": df_test["delivery_date"].values,
        "hour_ending": df_test["hour_ending"].values,
        "dam_lmp": df_test["dam_lmp"].values,
        "rtm_lmp": df_test["rtm_lmp"].values,
        "actual_spread": y_test_spread,
        "actual_direction": y_test_dir,
        "cb_pred_spread": pred_reg,
        "cb_pred_direction": pred_cls,
        "cb_pred_proba": pred_proba,
        "lgb_pred_spread": pred_lgb,
    })
    pred_path = CHECKPOINT_DIR / f"{sp}_predictions.parquet"
    pred_df.to_parquet(pred_path, index=False)
    print(f"Predictions saved to {pred_path}")

    # --- Save results ---
    results_path = CHECKPOINT_DIR / f"{sp}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print(f"{'Model':<30} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'Dir Acc':>8}")
    print("-" * 70)
    for name in ["naive_baseline", "catboost_regression", "lgbm_regression"]:
        m = results[name]
        print(f"{name:<30} {m['mae']:>8.2f} {m['rmse']:>8.2f} {m['r2']:>8.4f} {m['directional_accuracy']:>8.1%}")
    print("-" * 70)
    m = results["catboost_classifier"]
    print(f"{'catboost_classifier':<30} {'—':>8} {'—':>8} {'—':>8} {m['accuracy']:>8.1%}")
    print("=" * 70)


if __name__ == "__main__":
    main()
