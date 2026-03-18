#!/usr/bin/env python
"""
Train Load Forecast Models — Sprint 4.2

Trains CatBoost + LightGBM on features built by build_features_from_sqlite.py.
Evaluates with MAE, RMSE, MAPE. Saves checkpoints to checkpoints/.

Usage:
    python train_load_models.py                       # train both
    python train_load_models.py --model catboost      # CatBoost only
    python train_load_models.py --model lightgbm      # LightGBM only
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import joblib

from build_features_from_sqlite import (
    CATEGORICAL_FEATURES,
    TARGET_COL,
    get_feature_columns,
)


# ---------------------------------------------------------------------------
# Best hyperparameters from Optuna notebooks
# ---------------------------------------------------------------------------
CATBOOST_PARAMS = {
    "iterations": 1083,
    "depth": 3,
    "learning_rate": 0.052042,
    "l2_leaf_reg": 1e-6,
    "bagging_temperature": 0.697156,
    "random_strength": 0.04638,
    "border_count": 254,
    "loss_function": "MAE",
    "verbose": 100,
    "random_seed": 42,
}

LIGHTGBM_PARAMS = {
    "n_estimators": 417,
    "max_depth": 8,
    "learning_rate": 0.033736,
    "num_leaves": 18,
    "min_child_samples": 50,
    "subsample": 0.821078,
    "colsample_bytree": 0.893556,
    "reg_alpha": 0.001424,
    "reg_lambda": 1e-6,
    "objective": "mae",
    "verbosity": -1,
    "random_state": 42,
    "n_jobs": -1,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_and_split(
    data_path: Path,
    train_end: str = "2023-06-30",
    val_end: str = "2023-12-31",
):
    """Load parquet and split by date into train / val / test."""
    df = pd.read_parquet(data_path)
    df["valid_time"] = pd.to_datetime(df["valid_time"])
    df = df.sort_values("valid_time")

    train = df[df["valid_time"] <= train_end].copy()
    val = df[(df["valid_time"] > train_end) & (df["valid_time"] <= val_end)].copy()
    test = df[df["valid_time"] > val_end].copy()

    logger.info(f"Train: {len(train)}  ({train['valid_time'].min()} → {train['valid_time'].max()})")
    logger.info(f"Val:   {len(val)}  ({val['valid_time'].min()} → {val['valid_time'].max()})")
    logger.info(f"Test:  {len(test)}  ({test['valid_time'].min()} → {test['valid_time'].max()})")
    return train, val, test


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> dict:
    """Compute MAE, RMSE, MAPE."""
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

    logger.info(f"{label}  MAE={mae:.1f}  RMSE={rmse:.1f}  MAPE={mape:.2f}%")
    return {"mae": mae, "rmse": rmse, "mape": mape}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_catboost(X_train, y_train, X_val, y_val, cat_indices):
    """Train CatBoost regressor with notebook-optimised hyperparameters."""
    from catboost import CatBoostRegressor, Pool

    train_pool = Pool(X_train, y_train, cat_features=cat_indices)
    val_pool = Pool(X_val, y_val, cat_features=cat_indices)

    model = CatBoostRegressor(**CATBOOST_PARAMS)
    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50, use_best_model=True)
    return model


def train_lightgbm(X_train, y_train, X_val, y_val, cat_features):
    """Train LightGBM regressor with notebook-optimised hyperparameters."""
    import lightgbm as lgb

    train_ds = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
    val_ds = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_features, reference=train_ds)

    params = {k: v for k, v in LIGHTGBM_PARAMS.items()
              if k not in ("n_estimators", "random_state", "n_jobs")}
    params["metric"] = "mae"

    model = lgb.train(
        params,
        train_ds,
        num_boost_round=LIGHTGBM_PARAMS["n_estimators"],
        valid_sets=[val_ds],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
    )
    return model


# ---------------------------------------------------------------------------
# Checkpoint saving
# ---------------------------------------------------------------------------
def save_checkpoint(model, metrics: dict, name: str, checkpoint_dir: Path, feature_cols: list):
    """Save model + metadata JSON."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_path = checkpoint_dir / f"{name}.joblib"
    joblib.dump(model, model_path)

    meta = {
        "model_name": name,
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
        "feature_count": len(feature_cols),
        "features": feature_cols,
        "categorical_features": CATEGORICAL_FEATURES,
    }
    meta_path = checkpoint_dir / f"{name}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Saved checkpoint: {model_path}")
    logger.info(f"Saved metadata:   {meta_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train load forecast models (Sprint 4.2)")
    parser.add_argument("--model", type=str, default="all", choices=["catboost", "lightgbm", "all"])
    parser.add_argument("--data", type=str, default=None,
                        help="Path to features_augmented.parquet (default: data/features_augmented.parquet)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--train-end", type=str, default="2023-06-30")
    parser.add_argument("--val-end", type=str, default="2023-12-31")
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    data_path = Path(args.data) if args.data else base_dir / "data" / "features_augmented.parquet"
    checkpoint_dir = base_dir / args.checkpoint_dir

    if not data_path.exists():
        logger.error(f"Data not found: {data_path}. Run build_features_from_sqlite.py first.")
        return

    # Load & split
    train_df, val_df, test_df = load_and_split(data_path, args.train_end, args.val_end)
    feature_cols = get_feature_columns(train_df)
    logger.info(f"Features ({len(feature_cols)}): {feature_cols}")

    # Resolve categorical feature indices
    cat_indices = [feature_cols.index(c) for c in CATEGORICAL_FEATURES if c in feature_cols]
    cat_names = [c for c in CATEGORICAL_FEATURES if c in feature_cols]
    logger.info(f"Categorical indices: {cat_indices}")

    # Use DataFrames to preserve dtypes for CatBoost categorical handling
    X_train_df = train_df[feature_cols].copy()
    X_val_df = val_df[feature_cols].copy()
    X_test_df = test_df[feature_cols].copy()

    # Cast categorical columns to int
    for c in cat_names:
        X_train_df[c] = X_train_df[c].astype(int)
        X_val_df[c] = X_val_df[c].astype(int)
        X_test_df[c] = X_test_df[c].astype(int)

    y_train = train_df[TARGET_COL].values
    y_val = val_df[TARGET_COL].values
    y_test = test_df[TARGET_COL].values

    # ====================== CatBoost ======================
    if args.model in ("catboost", "all"):
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING CATBOOST")
        logger.info("=" * 60)

        cb_model = train_catboost(X_train_df, y_train, X_val_df, y_val, cat_indices)

        cb_val_metrics = evaluate(y_val, cb_model.predict(X_val_df), "CatBoost val")
        cb_test_metrics = evaluate(y_test, cb_model.predict(X_test_df), "CatBoost test")

        save_checkpoint(
            cb_model,
            {"val": cb_val_metrics, "test": cb_test_metrics},
            "load_catboost",
            checkpoint_dir,
            feature_cols,
        )

    # ====================== LightGBM ======================
    if args.model in ("lightgbm", "all"):
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING LIGHTGBM")
        logger.info("=" * 60)

        # LightGBM needs DataFrame with categorical dtype
        X_train_lgb = X_train_df.copy()
        X_val_lgb = X_val_df.copy()
        for c in cat_names:
            X_train_lgb[c] = X_train_lgb[c].astype("category")
            X_val_lgb[c] = X_val_lgb[c].astype("category")

        lgb_model = train_lightgbm(X_train_lgb, y_train, X_val_lgb, y_val, cat_names)

        # Predict (LightGBM booster predicts from raw arrays fine)
        lgb_val_pred = lgb_model.predict(X_val_df.values)
        lgb_test_pred = lgb_model.predict(X_test_df.values)

        lgb_val_metrics = evaluate(y_val, lgb_val_pred, "LightGBM val")
        lgb_test_metrics = evaluate(y_test, lgb_test_pred, "LightGBM test")

        save_checkpoint(
            lgb_model,
            {"val": lgb_val_metrics, "test": lgb_test_metrics},
            "load_lightgbm",
            checkpoint_dir,
            feature_cols,
        )

    # ====================== Ensemble ======================
    if args.model == "all":
        logger.info("\n" + "=" * 60)
        logger.info("ENSEMBLE (CatBoost + LightGBM average)")
        logger.info("=" * 60)

        ens_pred = (cb_model.predict(X_test_df) + lgb_model.predict(X_test_df.values)) / 2
        ens_metrics = evaluate(y_test, ens_pred, "Ensemble test")

        # Save ensemble config
        ens_meta = {
            "type": "catboost_lightgbm_ensemble",
            "n_models": 2,
            "components": ["load_catboost", "load_lightgbm"],
            "test_metrics": ens_metrics,
            "timestamp": datetime.now().isoformat(),
        }
        with open(checkpoint_dir / "ensemble_config.json", "w") as f:
            json.dump(ens_meta, f, indent=2)
        logger.info(f"Saved ensemble config: {checkpoint_dir / 'ensemble_config.json'}")

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
