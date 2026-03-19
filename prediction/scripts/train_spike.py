#!/usr/bin/env python
"""Train LightGBM spike prediction models per settlement point.

Usage:
    python -m prediction.scripts.train_spike [--sp HB_WEST] [--all]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from loguru import logger

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
    train_model,
)

# SPs that have feature parquets (excludes HB_HUBAVG which has no parquet)
AVAILABLE_SPS = [
    "HB_BUSAVG", "HB_HOUSTON", "HB_NORTH", "HB_PAN", "HB_SOUTH", "HB_WEST",
    "LZ_AEN", "LZ_CPS", "LZ_HOUSTON", "LZ_LCRA", "LZ_NORTH", "LZ_RAYBN",
    "LZ_SOUTH", "LZ_WEST",
]


def train_sp(sp: str, feature_dir: Path = FEATURE_DIR, model_dir: Path = MODEL_DIR) -> dict:
    """Train and evaluate model for a single settlement point."""
    logger.info("=== {} ===", sp)

    df = load_data(sp, feature_dir)
    feature_cols = get_feature_cols(df)
    train_df, val_df, test_df = split_data(df)

    logger.info(
        "Split: train={} val={} test={}", len(train_df), len(val_df), len(test_df)
    )

    if len(train_df) == 0 or (train_df[TARGET] == 1).sum() == 0:
        logger.warning("Insufficient training data for {}, skipping", sp)
        return {}

    # Train
    model = train_model(train_df, val_df, feature_cols)

    # Evaluate on validation and test sets
    results = {"settlement_point": sp, "feature_cols": feature_cols}

    for split_name, split_df in [("val", val_df), ("test", test_df)]:
        if len(split_df) == 0 or split_df[TARGET].sum() == 0:
            logger.warning("No positive samples in {} set for {}", split_name, sp)
            continue

        y_true = split_df[TARGET].values
        y_prob = model.predict(split_df[feature_cols])

        metrics = compute_metrics(y_true, y_prob)
        event = compute_event_recall(split_df.index, y_true, y_prob)
        metrics.update({f"event_{k}": v for k, v in event.items()})

        results[split_name] = metrics

        logger.info(
            "  {}: PR-AUC={:.4f}  ROC-AUC={:.4f}  P@R50={:.4f}  EventRecall={}/{} ({:.2f})",
            split_name,
            metrics.get("pr_auc", float("nan")),
            metrics.get("roc_auc", float("nan")),
            metrics.get("precision_at_recall_50", float("nan")),
            event["events_detected"],
            event["n_events"],
            event.get("event_recall", float("nan")),
        )

    # Feature importance
    top_feats = feature_importance(model)
    results["top_features"] = [{"name": n, "gain": float(g)} for n, g in top_feats]
    logger.info("  Top features: {}", [n for n, _ in top_feats[:5]])

    # Save model and metrics
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{sp}_lead60.lgb"
    model.save_model(str(model_path))
    logger.info("  Model saved: {}", model_path)

    metrics_path = model_dir / f"{sp}_lead60_metrics.json"
    # Convert for JSON serialization
    serializable = {k: v for k, v in results.items() if k != "feature_cols"}
    with open(metrics_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    return results


def main():
    parser = argparse.ArgumentParser(description="Train spike prediction models")
    parser.add_argument("--sp", type=str, nargs="*", default=None,
                        help="Settlement points to train (default: all)")
    parser.add_argument("--all", action="store_true", help="Train all SPs")
    parser.add_argument("--feature-dir", type=str, default=str(FEATURE_DIR))
    parser.add_argument("--model-dir", type=str, default=str(MODEL_DIR))
    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)
    model_dir = Path(args.model_dir)

    if args.sp:
        sps = [s.upper() for s in args.sp]
    else:
        sps = AVAILABLE_SPS

    logger.info("Training spike models for {} SPs", len(sps))

    all_results = []
    t0 = time.time()

    for sp in sps:
        try:
            result = train_sp(sp, feature_dir, model_dir)
            if result:
                all_results.append(result)
        except Exception as e:
            logger.error("Failed to train {}: {}", sp, e)

    # Summary table
    logger.info("\n=== Summary ===")
    header = f"{'SP':<15} {'Val PR-AUC':>10} {'Val ROC-AUC':>11} {'Test PR-AUC':>11} {'Test ROC-AUC':>12} {'Event Recall':>13}"
    logger.info(header)
    for r in all_results:
        val = r.get("val", {})
        test = r.get("test", {})
        ev = test.get("event_n_events", 0)
        ed = test.get("event_events_detected", 0)
        er = f"{ed}/{ev}" if ev > 0 else "n/a"
        logger.info(
            "{:<15} {:>10.4f} {:>11.4f} {:>11.4f} {:>12.4f} {:>13}",
            r["settlement_point"],
            val.get("pr_auc", float("nan")),
            val.get("roc_auc", float("nan")),
            test.get("pr_auc", float("nan")),
            test.get("roc_auc", float("nan")),
            er,
        )

    logger.info("Total time: {:.1f}s", time.time() - t0)


if __name__ == "__main__":
    main()
