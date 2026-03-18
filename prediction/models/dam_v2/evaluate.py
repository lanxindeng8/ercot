#!/usr/bin/env python3
"""DAM v2 model evaluation — detailed metrics, error analysis, and plots."""

import argparse
import json
import logging
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT / "checkpoints"
DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "training"
OUTPUT_DIR = ROOT / "evaluation_plots"

SETTLEMENT_POINTS = ["hb_west", "hb_north", "hb_south", "hb_houston", "hb_busavg"]

TARGET = "dam_lmp"

FUEL_COLS = ["wind_pct", "solar_pct", "gas_pct", "nuclear_pct", "coal_pct", "hydro_pct"]

FEATURE_COLS = [
    "hour_of_day", "day_of_week", "month", "is_weekend", "is_peak_hour",
    "is_holiday", "is_summer",
    "dam_lag_1h", "dam_lag_4h", "dam_lag_24h", "dam_lag_168h",
    "rtm_lag_1h", "rtm_lag_4h", "rtm_lag_24h", "rtm_lag_168h",
    "dam_roll_24h_mean", "dam_roll_24h_std", "dam_roll_24h_min", "dam_roll_24h_max",
    "dam_roll_168h_mean", "dam_roll_168h_std", "dam_roll_168h_min", "dam_roll_168h_max",
    "rtm_roll_24h_mean", "rtm_roll_24h_std", "rtm_roll_24h_min", "rtm_roll_24h_max",
    "rtm_roll_168h_mean", "rtm_roll_168h_std", "rtm_roll_168h_min", "rtm_roll_168h_max",
    "dam_rtm_spread", "spread_roll_24h_mean", "spread_roll_168h_mean",
    *FUEL_COLS,
]

CORE_FEATURES = [c for c in FEATURE_COLS if c not in FUEL_COLS]

CAT_FEATURES = [
    "hour_of_day", "day_of_week", "month", "is_weekend", "is_peak_hour",
    "is_holiday", "is_summer",
]

PEAK_HOURS = set(range(7, 23))  # hours 7-22 (ERCOT peak)


def load_model(sp: str):
    """Load the best saved model for a settlement point."""
    cb_path = CHECKPOINT_DIR / f"{sp}_catboost.cbm"
    lgbm_path = CHECKPOINT_DIR / f"{sp}_lightgbm.joblib"
    if cb_path.exists():
        model = CatBoostRegressor()
        model.load_model(str(cb_path))
        return model, "catboost"
    elif lgbm_path.exists():
        model = joblib.load(str(lgbm_path))
        return model, "lightgbm"
    else:
        raise FileNotFoundError(f"No model found for {sp} in {CHECKPOINT_DIR}")


def predict(model, model_type: str, X: pd.DataFrame) -> np.ndarray:
    """Generate predictions from model."""
    if model_type == "catboost":
        cat_idx = [list(X.columns).index(c) for c in CAT_FEATURES if c in X.columns]
        return model.predict(Pool(X, cat_features=cat_idx))
    else:
        return model.predict(X)


def evaluate_sp(sp: str) -> dict:
    """Full evaluation for one settlement point."""
    log.info(f"=== Evaluating {sp.upper()} ===")

    # Load model and test data
    model, model_type = load_model(sp)
    test_df = pd.read_parquet(DATA_DIR / sp / "test.parquet")
    for col in FUEL_COLS:
        if col in test_df.columns:
            test_df[col] = test_df[col].fillna(0.0)
    test_df = test_df.dropna(subset=CORE_FEATURES + [TARGET])

    X_test = test_df[FEATURE_COLS].copy()
    y_true = test_df[TARGET].values
    y_pred = predict(model, model_type, X_test)

    hours = test_df["hour_of_day"].values
    residuals = y_true - y_pred

    # --- Overall metrics ---
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mask_nonzero = np.abs(y_true) > 1.0
    mape = np.mean(np.abs((y_true[mask_nonzero] - y_pred[mask_nonzero]) / y_true[mask_nonzero])) * 100

    # --- Hourly MAPE ---
    hourly_mape = {}
    for h in range(1, 25):
        h_mask = (hours == h) & mask_nonzero
        if h_mask.sum() > 0:
            hourly_mape[int(h)] = round(float(
                np.mean(np.abs((y_true[h_mask] - y_pred[h_mask]) / y_true[h_mask])) * 100
            ), 2)

    # --- Peak vs off-peak ---
    peak_mask = np.isin(hours, list(PEAK_HOURS))
    offpeak_mask = ~peak_mask

    def _segment_metrics(mask):
        if mask.sum() == 0:
            return {}
        yt, yp = y_true[mask], y_pred[mask]
        nz = np.abs(yt) > 1.0
        return {
            "mae": round(float(mean_absolute_error(yt, yp)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(yt, yp))), 4),
            "mape_pct": round(float(np.mean(np.abs((yt[nz] - yp[nz]) / yt[nz])) * 100), 2) if nz.sum() > 0 else None,
            "n_samples": int(mask.sum()),
        }

    # --- Error distribution ---
    error_dist = {
        "mean": round(float(np.mean(residuals)), 4),
        "std": round(float(np.std(residuals)), 4),
        "median": round(float(np.median(residuals)), 4),
        "p5": round(float(np.percentile(residuals, 5)), 4),
        "p25": round(float(np.percentile(residuals, 25)), 4),
        "p75": round(float(np.percentile(residuals, 75)), 4),
        "p95": round(float(np.percentile(residuals, 95)), 4),
    }

    result = {
        "settlement_point": sp,
        "model_type": model_type,
        "n_test_samples": int(len(y_true)),
        "overall": {"mae": round(float(mae), 4), "rmse": round(float(rmse), 4), "mape_pct": round(float(mape), 2)},
        "peak": _segment_metrics(peak_mask),
        "offpeak": _segment_metrics(offpeak_mask),
        "hourly_mape": hourly_mape,
        "error_distribution": error_dist,
    }

    # --- Generate plots ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _plot_pred_vs_actual(y_true, y_pred, sp)
    _plot_residual_distribution(residuals, sp)
    _plot_hourly_error_heatmap(hours, residuals, sp)

    log.info(f"  Overall: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.1f}%")
    return result


def _plot_pred_vs_actual(y_true, y_pred, sp):
    """Scatter plot of predicted vs actual prices."""
    fig, ax = plt.subplots(figsize=(8, 8))
    # Sample for readability if too many points
    n = len(y_true)
    if n > 5000:
        idx = np.random.RandomState(42).choice(n, 5000, replace=False)
        yt, yp = y_true[idx], y_pred[idx]
    else:
        yt, yp = y_true, y_pred
    ax.scatter(yt, yp, alpha=0.15, s=4, color="steelblue")
    lo = min(yt.min(), yp.min())
    hi = max(np.percentile(yt, 99), np.percentile(yp, 99))
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Actual DAM LMP ($/MWh)")
    ax.set_ylabel("Predicted DAM LMP ($/MWh)")
    ax.set_title(f"{sp.upper()} — Predicted vs Actual")
    ax.legend()
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{sp}_pred_vs_actual.png", dpi=150)
    plt.close(fig)


def _plot_residual_distribution(residuals, sp):
    """Histogram of residuals (errors)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    clipped = np.clip(residuals, np.percentile(residuals, 1), np.percentile(residuals, 99))
    ax.hist(clipped, bins=80, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Residual (Actual - Predicted, $/MWh)")
    ax.set_ylabel("Count")
    ax.set_title(f"{sp.upper()} — Residual Distribution (1st-99th percentile)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{sp}_residual_dist.png", dpi=150)
    plt.close(fig)


def _plot_hourly_error_heatmap(hours, residuals, sp):
    """Heatmap of MAE by hour and absolute error quartile."""
    fig, ax = plt.subplots(figsize=(12, 4))
    hourly_mae = []
    hourly_labels = []
    for h in range(1, 25):
        mask = hours == h
        if mask.sum() > 0:
            hourly_mae.append(np.mean(np.abs(residuals[mask])))
        else:
            hourly_mae.append(0)
        hourly_labels.append(str(h))

    colors = plt.cm.YlOrRd(np.array(hourly_mae) / (max(hourly_mae) + 1e-6))
    bars = ax.bar(range(24), hourly_mae, color=colors, edgecolor="white")
    ax.set_xticks(range(24))
    ax.set_xticklabels(hourly_labels)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("MAE ($/MWh)")
    ax.set_title(f"{sp.upper()} — MAE by Hour of Day")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{sp}_hourly_mae.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate DAM v2 models")
    parser.add_argument("--points", nargs="*", default=SETTLEMENT_POINTS)
    args = parser.parse_args()

    all_results = {}
    for sp in args.points:
        try:
            result = evaluate_sp(sp)
            all_results[sp] = result
        except FileNotFoundError as e:
            log.warning(f"Skipping {sp}: {e}")

    report_path = ROOT / "detailed_evaluation.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"Detailed evaluation saved to {report_path}")
    log.info(f"Plots saved to {OUTPUT_DIR}")

    # Summary
    print("\n" + "=" * 70)
    print(f"{'SP':<12} {'MAE':>8} {'RMSE':>8} {'MAPE%':>8} {'Peak MAE':>10} {'OffPk MAE':>10}")
    print("-" * 70)
    for sp, r in all_results.items():
        o = r["overall"]
        print(f"{sp:<12} {o['mae']:>8.2f} {o['rmse']:>8.2f} {o['mape_pct']:>8.1f} {r['peak']['mae']:>10.2f} {r['offpeak']['mae']:>10.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
