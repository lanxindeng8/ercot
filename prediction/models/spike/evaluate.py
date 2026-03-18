#!/usr/bin/env python3
"""Spike model evaluation — precision@k, lead time, revenue impact analysis."""

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

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

# Import from train module
from train import (
    BASE_FEATURES,
    CAT_FEATURES,
    FEATURE_COLS,
    FUEL_COLS,
    CORE_FEATURES,
    TARGET_PRICE,
    SPIKE_THRESHOLD,
    SPIKE_MULTIPLIER,
    ROLLING_WINDOW,
    add_spike_features,
    generate_spike_labels,
)

# Battery economics for revenue impact
BATTERY_CAPACITY_MWH = 100  # MWh BESS capacity
BATTERY_POWER_MW = 25       # MW charge/discharge rate
HOLD_SOC_PENALTY = 5.0      # $/MWh opportunity cost of holding SOC


def load_model_and_meta(sp: str):
    """Load saved model and metadata for a settlement point."""
    model_path = CHECKPOINT_DIR / f"{sp}_spike_catboost.cbm"
    meta_path = CHECKPOINT_DIR / f"{sp}_spike_meta.json"

    if not model_path.exists():
        raise FileNotFoundError(f"No spike model for {sp} at {model_path}")

    model = CatBoostClassifier()
    model.load_model(str(model_path))

    with open(meta_path) as f:
        meta = json.load(f)

    return model, meta


def precision_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: int) -> float:
    """Precision among top-k highest probability predictions."""
    if k <= 0 or len(y_true) == 0:
        return 0.0
    k = min(k, len(y_true))
    top_k_idx = np.argsort(y_prob)[-k:]
    return float(y_true[top_k_idx].sum() / k)


def compute_lead_time(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute how many hours before a spike the model first raises alarm.

    Looks at each contiguous spike event and finds the earliest true-positive
    prediction before/during the event.
    """
    # Find spike event boundaries
    spike_starts = []
    in_spike = False
    for i in range(len(y_true)):
        if y_true[i] == 1 and not in_spike:
            spike_starts.append(i)
            in_spike = True
        elif y_true[i] == 0:
            in_spike = False

    if not spike_starts:
        return {"mean_lead_hours": None, "median_lead_hours": None, "n_events": 0}

    lead_times = []
    for start in spike_starts:
        # Look back up to 12 hours (12 rows at hourly resolution) for alarm
        lookback = max(0, start - 12)
        alarmed = False
        for i in range(lookback, start + 1):
            if y_pred[i] == 1:
                lead_hours = start - i
                lead_times.append(lead_hours)
                alarmed = True
                break
        if not alarmed:
            lead_times.append(0)  # no advance warning

    lead_arr = np.array(lead_times)
    return {
        "mean_lead_hours": round(float(lead_arr.mean()), 2),
        "median_lead_hours": round(float(np.median(lead_arr)), 2),
        "max_lead_hours": int(lead_arr.max()),
        "n_events_detected": int((lead_arr > 0).sum()),
        "n_events_total": len(spike_starts),
        "detection_rate": round(float((lead_arr > 0).sum() / len(spike_starts)), 4),
    }


def compute_revenue_impact(y_true: np.ndarray, y_pred: np.ndarray,
                           prices: np.ndarray) -> dict:
    """Estimate revenue impact of spike prediction for battery dispatch.

    Strategy: When spike is predicted, hold SOC (don't charge) and discharge
    during spike hours. Compare vs no-prediction baseline.

    Revenue model:
    - True positive: capture spike revenue = price * BATTERY_POWER_MW * 1hr
    - False positive: pay opportunity cost = HOLD_SOC_PENALTY * BATTERY_CAPACITY_MWH
    - False negative: missed spike revenue
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # Revenue from correctly predicted spikes (discharge at spike price)
    tp_mask = (y_true == 1) & (y_pred == 1)
    tp_revenue = float(prices[tp_mask].sum() * BATTERY_POWER_MW) if tp_mask.any() else 0.0

    # Missed revenue from undetected spikes
    fn_mask = (y_true == 1) & (y_pred == 0)
    missed_revenue = float(prices[fn_mask].sum() * BATTERY_POWER_MW) if fn_mask.any() else 0.0

    # Cost of false alarms (holding SOC unnecessarily)
    false_alarm_cost = float(fp * HOLD_SOC_PENALTY * BATTERY_CAPACITY_MWH)

    # Net value of spike prediction
    net_value = tp_revenue - false_alarm_cost
    total_spike_revenue = tp_revenue + missed_revenue

    return {
        "tp_revenue_usd": round(tp_revenue, 2),
        "missed_revenue_usd": round(missed_revenue, 2),
        "false_alarm_cost_usd": round(false_alarm_cost, 2),
        "net_value_usd": round(net_value, 2),
        "capture_rate": round(tp_revenue / total_spike_revenue, 4) if total_spike_revenue > 0 else 0.0,
        "avg_spike_price": round(float(prices[y_true == 1].mean()), 2) if y_true.sum() > 0 else 0.0,
        "avg_false_alarm_price": round(float(prices[(y_pred == 1) & (y_true == 0)].mean()), 2) if fp > 0 else 0.0,
        "battery_capacity_mwh": BATTERY_CAPACITY_MWH,
        "battery_power_mw": BATTERY_POWER_MW,
        "hold_soc_penalty_per_mwh": HOLD_SOC_PENALTY,
    }


def evaluate_sp(sp: str) -> dict:
    """Full evaluation for one settlement point."""
    log.info(f"=== Evaluating spike model: {sp.upper()} ===")

    model, meta = load_model_and_meta(sp)
    threshold = meta["threshold"]
    hour_probs = {int(k): v for k, v in meta["hour_spike_probs"].items()}
    features = meta["features"]

    # Load test data
    test_df = pd.read_parquet(DATA_DIR / sp / "test.parquet")
    for col in FUEL_COLS:
        if col in test_df.columns:
            test_df[col] = test_df[col].fillna(0.0)
    test_df = test_df.dropna(subset=CORE_FEATURES + [TARGET_PRICE])
    test_df = add_spike_features(test_df, train_hour_probs=hour_probs)

    y_true = generate_spike_labels(test_df).values
    prices = test_df[TARGET_PRICE].values

    available_features = [c for c in features if c in test_df.columns]
    X_test = test_df[available_features].copy()
    cat_idx = [list(X_test.columns).index(c) for c in CAT_FEATURES if c in X_test.columns]
    probs = model.predict_proba(Pool(X_test, cat_features=cat_idx))[:, 1]
    preds = (probs >= threshold).astype(int)

    # --- Classification metrics ---
    precision = precision_score(y_true, preds, zero_division=0)
    recall = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    try:
        auc_roc = roc_auc_score(y_true, probs)
    except ValueError:
        auc_roc = float("nan")

    cm = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # --- Precision@K ---
    n_spikes = int(y_true.sum())
    p_at_k = {}
    for k_mult in [1, 2, 5]:
        k = n_spikes * k_mult
        if k > 0:
            p_at_k[f"precision_at_{k_mult}x_spikes"] = round(precision_at_k(y_true, probs, k), 4)

    # --- Lead time ---
    lead_time = compute_lead_time(y_true, preds)

    # --- Revenue impact ---
    revenue = compute_revenue_impact(y_true, preds, prices)

    # --- Hourly breakdown ---
    hours = test_df["hour_of_day"].values
    hourly_stats = {}
    for h in range(1, 25):
        h_mask = hours == h
        if h_mask.sum() == 0:
            continue
        h_true = y_true[h_mask]
        h_preds = preds[h_mask]
        hourly_stats[int(h)] = {
            "n_samples": int(h_mask.sum()),
            "n_spikes": int(h_true.sum()),
            "spike_rate": round(float(h_true.mean()), 4),
            "precision": round(float(precision_score(h_true, h_preds, zero_division=0)), 4),
            "recall": round(float(recall_score(h_true, h_preds, zero_division=0)), 4),
        }

    result = {
        "settlement_point": sp,
        "threshold": round(float(threshold), 4),
        "n_test": int(len(y_true)),
        "n_spikes": n_spikes,
        "spike_rate": round(float(y_true.mean()), 4),
        "metrics": {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "auc_roc": round(float(auc_roc), 4),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        },
        "precision_at_k": p_at_k,
        "lead_time": lead_time,
        "revenue_impact": revenue,
        "hourly_breakdown": hourly_stats,
    }

    # --- Plots ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _plot_precision_recall_curve(y_true, probs, sp)
    _plot_roc_curve(y_true, probs, sp)
    _plot_spike_distribution(y_true, preds, prices, sp)
    _plot_hourly_spike_rates(hourly_stats, sp)

    log.info(f"  Precision={precision:.3f} Recall={recall:.3f} F1={f1:.3f} AUC={auc_roc:.3f}")
    log.info(f"  Lead time: mean={lead_time['mean_lead_hours']}h, "
             f"detection={lead_time.get('detection_rate', 0):.1%}")
    log.info(f"  Revenue impact: net=${revenue['net_value_usd']:,.0f}")

    return result


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_precision_recall_curve(y_true, probs, sp):
    prec_vals, rec_vals, _ = precision_recall_curve(y_true, probs)
    pr_auc = auc(rec_vals, prec_vals)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(rec_vals, prec_vals, color="steelblue", linewidth=2,
            label=f"PR AUC = {pr_auc:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{sp.upper()} — Precision-Recall Curve")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{sp}_pr_curve.png", dpi=150)
    plt.close(fig)


def _plot_roc_curve(y_true, probs, sp):
    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc_val = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="steelblue", linewidth=2,
            label=f"ROC AUC = {roc_auc_val:.3f}")
    ax.plot([0, 1], [0, 1], "r--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{sp.upper()} — ROC Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{sp}_roc_curve.png", dpi=150)
    plt.close(fig)


def _plot_spike_distribution(y_true, preds, prices, sp):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Price distribution for spikes vs non-spikes
    ax = axes[0]
    ax.hist(prices[y_true == 0], bins=50, alpha=0.6, label="Non-spike", color="steelblue",
            density=True)
    if y_true.sum() > 0:
        ax.hist(prices[y_true == 1], bins=30, alpha=0.6, label="Spike", color="red",
                density=True)
    ax.set_xlabel("RTM LMP ($/MWh)")
    ax.set_ylabel("Density")
    ax.set_title("Price Distribution")
    ax.legend()

    # Predicted probabilities by class
    ax = axes[1]
    ax.hist(prices[preds == 0], bins=50, alpha=0.6, label="Predicted non-spike",
            color="steelblue", density=True)
    if preds.sum() > 0:
        ax.hist(prices[preds == 1], bins=30, alpha=0.6, label="Predicted spike",
                color="orange", density=True)
    ax.set_xlabel("RTM LMP ($/MWh)")
    ax.set_ylabel("Density")
    ax.set_title("Predicted Classification")
    ax.legend()

    fig.suptitle(f"{sp.upper()} — Spike Price Distributions")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{sp}_spike_dist.png", dpi=150)
    plt.close(fig)


def _plot_hourly_spike_rates(hourly_stats, sp):
    hours = sorted(hourly_stats.keys())
    rates = [hourly_stats[h]["spike_rate"] for h in hours]
    precisions = [hourly_stats[h]["precision"] for h in hours]

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.bar(hours, rates, alpha=0.6, color="steelblue", label="Spike rate")
    ax1.set_xlabel("Hour of Day")
    ax1.set_ylabel("Spike Rate", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    ax2 = ax1.twinx()
    ax2.plot(hours, precisions, "o-", color="red", label="Precision")
    ax2.set_ylabel("Precision", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    fig.suptitle(f"{sp.upper()} — Hourly Spike Rate & Precision")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{sp}_hourly_spikes.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate spike detection models")
    parser.add_argument("--points", nargs="*", default=["hb_west"])
    args = parser.parse_args()

    all_results = {}
    for sp in args.points:
        try:
            result = evaluate_sp(sp)
            all_results[sp] = result
        except FileNotFoundError as e:
            log.warning(f"Skipping {sp}: {e}")

    report_path = ROOT / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"Evaluation report saved to {report_path}")
    log.info(f"Plots saved to {OUTPUT_DIR}")

    # Summary
    print("\n" + "=" * 95)
    print(f"{'SP':<12} {'Prec':>8} {'Recall':>8} {'F1':>8} {'AUC':>8} "
          f"{'LeadH':>8} {'NetRev$':>12} {'Capture':>8}")
    print("-" * 95)
    for sp, r in all_results.items():
        m = r["metrics"]
        lt = r["lead_time"]
        rv = r["revenue_impact"]
        lead_h = lt["mean_lead_hours"] if lt["mean_lead_hours"] is not None else 0
        print(f"{sp:<12} {m['precision']:>8.3f} {m['recall']:>8.3f} {m['f1']:>8.3f} "
              f"{m['auc_roc']:>8.3f} {lead_h:>8.1f} {rv['net_value_usd']:>12,.0f} "
              f"{rv['capture_rate']:>8.1%}")
    print("=" * 95)


if __name__ == "__main__":
    main()
