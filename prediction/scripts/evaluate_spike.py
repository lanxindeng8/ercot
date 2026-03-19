#!/usr/bin/env python
"""Evaluate trained LightGBM spike models and generate reports.

Usage:
    python -m prediction.scripts.evaluate_spike [--sp HB_WEST] [--all]
"""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import precision_recall_curve

from prediction.src.models.spike_model import (
    FEATURE_DIR,
    MODEL_DIR,
    TARGET,
    compute_event_recall,
    compute_metrics,
    feature_importance,
    get_feature_cols,
    load_data,
    split_data,
)

AVAILABLE_SPS = [
    "HB_BUSAVG", "HB_HOUSTON", "HB_NORTH", "HB_PAN", "HB_SOUTH", "HB_WEST",
    "LZ_AEN", "LZ_CPS", "LZ_HOUSTON", "LZ_LCRA", "LZ_NORTH", "LZ_RAYBN",
    "LZ_SOUTH", "LZ_WEST",
]


def load_model(sp: str, model_dir: Path = MODEL_DIR) -> lgb.Booster:
    """Load a trained model for a settlement point."""
    path = model_dir / f"{sp}_lead60.lgb"
    if not path.exists():
        raise FileNotFoundError(f"No model found: {path}")
    return lgb.Booster(model_file=str(path))


def evaluate_sp(sp: str, feature_dir: Path, model_dir: Path) -> dict:
    """Evaluate a single SP model on test data."""
    model = load_model(sp, model_dir)
    df = load_data(sp, feature_dir)
    feature_cols = get_feature_cols(df)
    _, _, test_df = split_data(df)

    if len(test_df) == 0:
        logger.warning("{}: no test data", sp)
        return {}

    y_true = test_df[TARGET].values
    y_prob = model.predict(test_df[feature_cols])

    metrics = compute_metrics(y_true, y_prob)
    event = compute_event_recall(test_df.index, y_true, y_prob)
    metrics.update({f"event_{k}": v for k, v in event.items()})

    # PR curve sample points
    if len(np.unique(y_true)) >= 2:
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        # Downsample for report
        idx = np.linspace(0, len(prec) - 1, 20, dtype=int)
        metrics["pr_curve"] = {
            "precision": [float(prec[i]) for i in idx],
            "recall": [float(rec[i]) for i in idx],
        }

    # Feature importance
    top_feats = feature_importance(model)
    metrics["top_features"] = [{"name": n, "gain": float(g)} for n, g in top_feats]

    return {"settlement_point": sp, **metrics}


def case_study_lz_cps(feature_dir: Path, model_dir: Path):
    """Case study: 2025-12-14 LZ_CPS event — $686 spike at 20:15 CT."""
    sp = "LZ_CPS"
    try:
        model = load_model(sp, model_dir)
    except FileNotFoundError:
        logger.warning("No model for LZ_CPS, skipping case study")
        return

    df = load_data(sp, feature_dir)
    feature_cols = get_feature_cols(df)

    # 20:15 CT = 02:15 UTC on 2025-12-15 (CT is UTC-6)
    event_time = pd.Timestamp("2025-12-15 02:15:00", tz="UTC")
    # Show 4 hours before the event
    window_start = event_time - pd.Timedelta(hours=4)
    window_end = event_time + pd.Timedelta(hours=1)

    window = df[(df.index >= window_start) & (df.index <= window_end)]
    if len(window) == 0:
        logger.warning("No data in case study window for LZ_CPS")
        return

    y_prob = model.predict(window[feature_cols])
    window = window.copy()
    window["pred_prob"] = y_prob

    logger.info("\n=== Case Study: LZ_CPS 2025-12-14 $686 Spike ===")
    logger.info("Event time: 2025-12-14 20:15 CT (2025-12-15 02:15 UTC)")
    logger.info(
        "{:<25} {:>8} {:>12} {:>10}",
        "Timestamp (UTC)", "LMP", "lead_spike_60", "Pred Prob",
    )
    for ts, row in window.iterrows():
        logger.info(
            "{:<25} {:>8.1f} {:>12.0f} {:>10.4f}",
            str(ts)[:25],
            row.get("lmp_lag1", float("nan")),
            row[TARGET],
            row["pred_prob"],
        )


def main():
    parser = argparse.ArgumentParser(description="Evaluate spike prediction models")
    parser.add_argument("--sp", type=str, nargs="*", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--feature-dir", type=str, default=str(FEATURE_DIR))
    parser.add_argument("--model-dir", type=str, default=str(MODEL_DIR))
    parser.add_argument("--case-study", action="store_true", default=True,
                        help="Include LZ_CPS case study")
    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)
    model_dir = Path(args.model_dir)
    sps = [s.upper() for s in args.sp] if args.sp else AVAILABLE_SPS

    results = []
    for sp in sps:
        try:
            r = evaluate_sp(sp, feature_dir, model_dir)
            if r:
                results.append(r)
        except Exception as e:
            logger.error("Failed {}: {}", sp, e)

    # Summary table
    logger.info("\n=== Evaluation Summary (Test Set: 2026-01+) ===")
    header = f"{'SP':<15} {'PR-AUC':>8} {'ROC-AUC':>9} {'P@R50':>7} {'Events':>8} {'Detected':>9} {'EvRecall':>9}"
    logger.info(header)
    for r in results:
        logger.info(
            "{:<15} {:>8.4f} {:>9.4f} {:>7.4f} {:>8} {:>9} {:>9.2f}",
            r["settlement_point"],
            r.get("pr_auc", float("nan")),
            r.get("roc_auc", float("nan")),
            r.get("precision_at_recall_50", float("nan")),
            r.get("event_n_events", 0),
            r.get("event_events_detected", 0),
            r.get("event_event_recall", float("nan")),
        )

    # Save report
    report_path = model_dir / "evaluation_report.json"
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Report saved: {}", report_path)

    # Feature importance across SPs
    logger.info("\n=== Top Features (by avg gain across SPs) ===")
    feat_gains = {}
    for r in results:
        for feat in r.get("top_features", []):
            feat_gains.setdefault(feat["name"], []).append(feat["gain"])
    avg_gains = sorted(
        [(n, np.mean(g)) for n, g in feat_gains.items()],
        key=lambda x: x[1], reverse=True,
    )
    for name, gain in avg_gains[:10]:
        logger.info("  {:<30} avg_gain={:.1f}", name, gain)

    if args.case_study:
        case_study_lz_cps(feature_dir, model_dir)


if __name__ == "__main__":
    main()
