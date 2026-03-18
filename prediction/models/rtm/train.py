#!/usr/bin/env python3
"""RTM multi-horizon model training with CatBoost/LightGBM + Optuna tuning.

Trains separate models for 1h, 4h, and 24h ahead RTM price forecasting.
Each horizon gets both CatBoost and LightGBM candidates; best is selected by val MAE.
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor, early_stopping
from sklearn.metrics import mean_absolute_error, mean_squared_error

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

SETTLEMENT_POINTS = ["hb_west", "hb_north", "hb_south", "hb_houston", "hb_busavg"]

BASE_TARGET = "rtm_lmp"

# Multi-horizon definitions: name -> shift (in hourly rows)
HORIZONS = {
    "rtm_lmp_1h": 1,
    "rtm_lmp_4h": 4,
    "rtm_lmp_24h": 24,
}

FUEL_COLS = ["wind_pct", "solar_pct", "gas_pct", "nuclear_pct", "coal_pct", "hydro_pct"]

FEATURE_COLS = [
    # Temporal
    "hour_of_day", "day_of_week", "month", "is_weekend", "is_peak_hour",
    "is_holiday", "is_summer",
    # DAM lags
    "dam_lag_1h", "dam_lag_4h", "dam_lag_24h", "dam_lag_168h",
    # RTM lags
    "rtm_lag_1h", "rtm_lag_4h", "rtm_lag_24h", "rtm_lag_168h",
    # DAM rolling
    "dam_roll_24h_mean", "dam_roll_24h_std", "dam_roll_24h_min", "dam_roll_24h_max",
    "dam_roll_168h_mean", "dam_roll_168h_std", "dam_roll_168h_min", "dam_roll_168h_max",
    # RTM rolling
    "rtm_roll_24h_mean", "rtm_roll_24h_std", "rtm_roll_24h_min", "rtm_roll_24h_max",
    "rtm_roll_168h_mean", "rtm_roll_168h_std", "rtm_roll_168h_min", "rtm_roll_168h_max",
    # Cross-market
    "dam_rtm_spread", "spread_roll_24h_mean", "spread_roll_168h_mean",
    # Fuel mix
    *FUEL_COLS,
]

# RTM-specific extra features derived during loading
RTM_EXTRA_FEATURES = [
    "rtm_volatility_24h",       # rtm_roll_24h_std / (rtm_roll_24h_mean + 1)
    "rtm_momentum_4h",          # rtm_lag_1h - rtm_lag_4h
    "rtm_mean_revert_signal",   # rtm_lag_1h - rtm_roll_24h_mean
]

ALL_FEATURE_COLS = FEATURE_COLS + RTM_EXTRA_FEATURES

CORE_FEATURES = [c for c in ALL_FEATURE_COLS if c not in FUEL_COLS]

CAT_FEATURES = [
    "hour_of_day", "day_of_week", "month", "is_weekend", "is_peak_hour",
    "is_holiday", "is_summer",
]


def create_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Create multi-horizon target columns by shifting rtm_lmp forward.

    rtm_lmp_Xh = the RTM price X hours into the future (what we want to predict).
    Uses negative shift so row t gets the price from t+X.
    """
    df = df.copy()
    for target_name, shift in HORIZONS.items():
        df[target_name] = df[BASE_TARGET].shift(-shift)
    return df


def add_rtm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add RTM-specific derived features."""
    df = df.copy()
    df["rtm_volatility_24h"] = df["rtm_roll_24h_std"] / (df["rtm_roll_24h_mean"].abs() + 1.0)
    df["rtm_momentum_4h"] = df["rtm_lag_1h"] - df["rtm_lag_4h"]
    df["rtm_mean_revert_signal"] = df["rtm_lag_1h"] - df["rtm_roll_24h_mean"]
    return df


def load_data(sp: str, target: str):
    """Load train/val/test parquet for a settlement point and horizon target."""
    sp_dir = DATA_DIR / sp
    dfs = {}
    for split in ("train", "val", "test"):
        path = sp_dir / f"{split}.parquet"
        df = pd.read_parquet(path)
        # Create multi-horizon targets
        df = create_targets(df)
        # Add RTM-specific features
        df = add_rtm_features(df)
        # Fill missing fuel data
        for col in FUEL_COLS:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)
        before = len(df)
        # Drop rows where features or target are NaN (target NaN at tail due to shift)
        df = df.dropna(subset=CORE_FEATURES + [target])
        log.info(f"  {split}: {before} -> {len(df)} rows (dropped {before - len(df)} NaN)")
        dfs[split] = df
    return dfs["train"], dfs["val"], dfs["test"]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MAPE, RMSE, MAE, and directional accuracy."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # MAPE: avoid division by zero on near-zero prices
    mask = np.abs(y_true) > 1.0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = float("nan")
    # Directional accuracy vs prior step
    if len(y_true) > 1:
        actual_dir = np.diff(y_true) > 0
        pred_dir = np.diff(y_pred) > 0
        dir_acc = np.mean(actual_dir == pred_dir) * 100
    else:
        dir_acc = float("nan")
    return {
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "mape_pct": round(float(mape), 2),
        "directional_accuracy_pct": round(float(dir_acc), 2),
    }


def naive_baseline_metrics(val_df: pd.DataFrame, target: str, horizon_shift: int) -> dict:
    """Naive baseline: use current RTM price (lag=horizon) as forecast.

    For 1h ahead, naive = rtm_lag_1h; for 4h = rtm_lag_4h; for 24h = rtm_lag_24h.
    """
    lag_col = f"rtm_lag_{horizon_shift}h"
    if horizon_shift == 168:
        lag_col = "rtm_lag_168h"
    mask = val_df[lag_col].notna() & val_df[target].notna()
    y_true = val_df.loc[mask, target].values
    y_pred = val_df.loc[mask, lag_col].values
    return compute_metrics(y_true, y_pred)


# ---------------------------------------------------------------------------
# Optuna objectives
# ---------------------------------------------------------------------------

def _catboost_objective(trial, X_train, y_train, X_val, y_val):
    params = {
        "iterations": trial.suggest_int("iterations", 500, 3000),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
        "loss_function": "MAE",
        "verbose": 0,
        "random_seed": 42,
        "allow_writing_files": False,
    }
    cat_idx = [list(X_train.columns).index(c) for c in CAT_FEATURES if c in X_train.columns]
    train_pool = Pool(X_train, y_train, cat_features=cat_idx)
    val_pool = Pool(X_val, y_val, cat_features=cat_idx)
    model = CatBoostRegressor(**params)
    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=100, verbose=0)
    preds = model.predict(X_val)
    return mean_absolute_error(y_val, preds)


def _lgbm_objective(trial, X_train, y_train, X_val, y_val):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 500, 3000),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "objective": "mae",
        "random_state": 42,
        "verbosity": -1,
    }
    cat_cols = [c for c in CAT_FEATURES if c in X_train.columns]
    model = LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[early_stopping(stopping_rounds=100, verbose=False)],
        categorical_feature=cat_cols,
    )
    preds = model.predict(X_val)
    return mean_absolute_error(y_val, preds)


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train_horizon(sp: str, target: str, horizon_shift: int, n_trials: int = 30) -> dict:
    """Train CatBoost + LightGBM for one settlement point and one horizon."""
    log.info(f"--- Horizon: {target} (shift={horizon_shift}h) ---")
    train_df, val_df, test_df = load_data(sp, target)

    X_train = train_df[ALL_FEATURE_COLS].copy()
    y_train = train_df[target].values
    X_val = val_df[ALL_FEATURE_COLS].copy()
    y_val = val_df[target].values
    X_test = test_df[ALL_FEATURE_COLS].copy()
    y_test = test_df[target].values

    # --- Naive baseline ---
    baseline = naive_baseline_metrics(val_df, target, horizon_shift)
    log.info(f"  Naive baseline (val): MAE={baseline['mae']}, MAPE={baseline['mape_pct']}%")

    result = {"target": target, "horizon_hours": horizon_shift, "naive_baseline_val": baseline}

    # --- CatBoost with Optuna ---
    log.info(f"  Tuning CatBoost ({n_trials} trials)...")
    t0 = time.time()
    cb_study = optuna.create_study(direction="minimize", study_name=f"{sp}_{target}_catboost")
    cb_study.optimize(
        lambda trial: _catboost_objective(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
    )
    cb_time = time.time() - t0
    log.info(f"  CatBoost best val MAE: {cb_study.best_value:.4f} ({cb_time:.0f}s)")

    # Retrain best CatBoost
    cb_params = cb_study.best_params
    cb_params.update({"loss_function": "MAE", "verbose": 0, "random_seed": 42, "allow_writing_files": False})
    cat_idx = [list(X_train.columns).index(c) for c in CAT_FEATURES if c in X_train.columns]
    cb_model = CatBoostRegressor(**cb_params)
    cb_model.fit(
        Pool(X_train, y_train, cat_features=cat_idx),
        eval_set=Pool(X_val, y_val, cat_features=cat_idx),
        early_stopping_rounds=100, verbose=0,
    )
    cb_val_preds = cb_model.predict(X_val)
    cb_test_preds = cb_model.predict(X_test)
    cb_val_metrics = compute_metrics(y_val, cb_val_preds)
    cb_test_metrics = compute_metrics(y_test, cb_test_preds)
    result["catboost"] = {
        "best_params": cb_params,
        "val_metrics": cb_val_metrics,
        "test_metrics": cb_test_metrics,
        "tuning_seconds": round(cb_time, 1),
    }

    # --- LightGBM with Optuna ---
    log.info(f"  Tuning LightGBM ({n_trials} trials)...")
    t0 = time.time()
    lgbm_study = optuna.create_study(direction="minimize", study_name=f"{sp}_{target}_lgbm")
    lgbm_study.optimize(
        lambda trial: _lgbm_objective(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
    )
    lgbm_time = time.time() - t0
    log.info(f"  LightGBM best val MAE: {lgbm_study.best_value:.4f} ({lgbm_time:.0f}s)")

    # Retrain best LightGBM
    lgbm_params = lgbm_study.best_params
    lgbm_params.update({"objective": "mae", "random_state": 42, "verbosity": -1})
    cat_cols = [c for c in CAT_FEATURES if c in X_train.columns]
    lgbm_model = LGBMRegressor(**lgbm_params)
    lgbm_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[early_stopping(stopping_rounds=100, verbose=False)],
        categorical_feature=cat_cols,
    )
    lgbm_val_preds = lgbm_model.predict(X_val)
    lgbm_test_preds = lgbm_model.predict(X_test)
    lgbm_val_metrics = compute_metrics(y_val, lgbm_val_preds)
    lgbm_test_metrics = compute_metrics(y_test, lgbm_test_preds)
    result["lightgbm"] = {
        "best_params": {k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (np.floating,)) else v) for k, v in lgbm_params.items()},
        "val_metrics": lgbm_val_metrics,
        "test_metrics": lgbm_test_metrics,
        "tuning_seconds": round(lgbm_time, 1),
    }

    # --- Pick winner ---
    cb_val_mae = cb_val_metrics["mae"]
    lgbm_val_mae = lgbm_val_metrics["mae"]
    winner = "catboost" if cb_val_mae <= lgbm_val_mae else "lightgbm"
    result["best_model"] = winner
    log.info(f"  Winner: {winner} (val MAE: CB={cb_val_mae:.4f}, LGBM={lgbm_val_mae:.4f})")

    # --- Save best model ---
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    model_tag = f"{sp}_{target}"
    if winner == "catboost":
        model_path = CHECKPOINT_DIR / f"{model_tag}_catboost.cbm"
        cb_model.save_model(str(model_path))
        result["model_path"] = str(model_path)
        np.save(CHECKPOINT_DIR / f"{model_tag}_test_preds.npy", cb_test_preds)
    else:
        import joblib
        model_path = CHECKPOINT_DIR / f"{model_tag}_lightgbm.joblib"
        joblib.dump(lgbm_model, str(model_path))
        result["model_path"] = str(model_path)
        np.save(CHECKPOINT_DIR / f"{model_tag}_test_preds.npy", lgbm_test_preds)

    # Save feature importance
    if winner == "catboost":
        importance = cb_model.get_feature_importance()
    else:
        importance = lgbm_model.feature_importances_
    fi = sorted(zip(ALL_FEATURE_COLS, importance), key=lambda x: -x[1])
    result["top_features"] = [{"feature": f, "importance": round(float(v), 2)} for f, v in fi[:10]]

    log.info(f"  Test metrics ({winner}): {result[winner]['test_metrics']}")
    return result


def train_settlement_point(sp: str, n_trials: int = 30) -> dict:
    """Train all horizons for one settlement point."""
    log.info(f"=== Training RTM models for {sp.upper()} ===")
    results = {"settlement_point": sp, "horizons": {}}
    for target, shift in HORIZONS.items():
        result = train_horizon(sp, target, shift, n_trials=n_trials)
        results["horizons"][target] = result
    return results


def main():
    parser = argparse.ArgumentParser(description="Train RTM multi-horizon models")
    parser.add_argument("--points", nargs="*", default=["hb_west"],
                        help="Settlement points to train (default: hb_west)")
    parser.add_argument("--trials", type=int, default=30,
                        help="Optuna trials per model type per horizon")
    args = parser.parse_args()

    all_results = {}
    for sp in args.points:
        result = train_settlement_point(sp, n_trials=args.trials)
        all_results[sp] = result

    # Save evaluation report
    report_path = ROOT / "training_report.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"Training report saved to {report_path}")

    # Print summary table
    print("\n" + "=" * 100)
    print(f"{'SP':<12} {'Horizon':<14} {'Model':<10} {'Val MAE':>10} {'Test MAE':>10} {'Test RMSE':>10} {'MAPE%':>8} {'DirAcc%':>8} {'Baseline':>10}")
    print("-" * 100)
    for sp, sp_result in all_results.items():
        for target, r in sp_result["horizons"].items():
            w = r["best_model"]
            tm = r[w]["test_metrics"]
            bl = r["naive_baseline_val"]["mae"]
            print(f"{sp:<12} {target:<14} {w:<10} {r[w]['val_metrics']['mae']:>10.2f} {tm['mae']:>10.2f} {tm['rmse']:>10.2f} {tm['mape_pct']:>8.1f} {tm['directional_accuracy_pct']:>8.1f} {bl:>10.2f}")
    print("=" * 100)


if __name__ == "__main__":
    main()
