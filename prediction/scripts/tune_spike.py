#!/usr/bin/env python
"""Optuna hyperparameter tuning for LightGBM spike prediction models.

Usage:
    python -m prediction.scripts.tune_spike [--sp HB_WEST] [--n-trials 50]
"""

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
from loguru import logger

from prediction.src.models.spike_model import (
    FEATURE_DIR,
    MODEL_DIR,
    TARGET,
    compute_event_recall,
    compute_metrics,
    get_feature_cols,
    load_data,
    split_data,
    train_model,
)

AVAILABLE_SPS = [
    "HB_BUSAVG", "HB_HOUSTON", "HB_NORTH", "HB_PAN", "HB_SOUTH", "HB_WEST",
    "LZ_AEN", "LZ_CPS", "LZ_HOUSTON", "LZ_LCRA", "LZ_NORTH", "LZ_RAYBN",
    "LZ_SOUTH", "LZ_WEST",
]


def create_objective(train_df, val_df, feature_cols, neg_pos_ratio):
    """Create an Optuna objective function that maximizes PR-AUC."""

    def objective(trial):
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": 5,
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "scale_pos_weight": trial.suggest_float(
                "scale_pos_weight", 50, neg_pos_ratio * 1.5
            ),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
            "max_depth": trial.suggest_categorical("max_depth", [-1, 3, 5, 7, 10, 15]),
            "min_split_gain": trial.suggest_float("min_split_gain", 0, 1.0),
            "objective": "binary",
            "metric": ["binary_logloss", "auc"],
            "verbose": -1,
        }

        model = train_model(
            train_df, val_df, feature_cols,
            params=params,
            num_boost_round=2000,
            early_stopping_rounds=50,
        )

        y_prob = model.predict(val_df[feature_cols])
        y_true = val_df[TARGET].values
        metrics = compute_metrics(y_true, y_prob)
        return metrics["pr_auc"]

    return objective


def tune_sp(
    sp: str,
    n_trials: int = 50,
    feature_dir: Path = FEATURE_DIR,
    model_dir: Path = MODEL_DIR,
) -> dict:
    """Run Optuna tuning for a single settlement point."""
    logger.info("=== Tuning {} ({} trials) ===", sp, n_trials)

    df = load_data(sp, feature_dir)
    feature_cols = get_feature_cols(df)
    train_df, val_df, test_df = split_data(df)

    logger.info("Split: train={} val={} test={}", len(train_df), len(val_df), len(test_df))

    pos = (train_df[TARGET] == 1).sum()
    neg = (train_df[TARGET] == 0).sum()
    if pos == 0:
        logger.warning("No positive samples in training data for {}, skipping", sp)
        return {}
    neg_pos_ratio = neg / max(pos, 1)

    # Load baseline metrics
    baseline_metrics_path = model_dir / f"{sp}_lead60_metrics.json"
    baseline_metrics = {}
    if baseline_metrics_path.exists():
        with open(baseline_metrics_path) as f:
            baseline_metrics = json.load(f)

    # Run Optuna study
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", study_name=f"spike_{sp}")

    objective = create_objective(train_df, val_df, feature_cols, neg_pos_ratio)

    def progress_callback(study, trial):
        if (trial.number + 1) % 10 == 0:
            logger.info(
                "  Trial {}/{}: best PR-AUC={:.4f}",
                trial.number + 1, n_trials, study.best_value,
            )

    study.optimize(objective, n_trials=n_trials, callbacks=[progress_callback])

    best_params = study.best_params
    logger.info("Best PR-AUC (val): {:.4f}", study.best_value)
    logger.info("Best params: {}", best_params)

    # Retrain with best params on full train set
    best_params.update({
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "bagging_freq": 5,
        "verbose": -1,
    })
    tuned_model = train_model(
        train_df, val_df, feature_cols,
        params=best_params,
        num_boost_round=2000,
        early_stopping_rounds=50,
    )

    # Evaluate tuned model on test set
    tuned_test_metrics = {}
    tuned_event = {}
    if len(test_df) > 0 and test_df[TARGET].sum() > 0:
        y_true_test = test_df[TARGET].values
        y_prob_test = tuned_model.predict(test_df[feature_cols])
        tuned_test_metrics = compute_metrics(y_true_test, y_prob_test)
        tuned_event = compute_event_recall(test_df.index, y_true_test, y_prob_test)

    # Evaluate baseline on test set (reload baseline model)
    baseline_test_metrics = {}
    baseline_event = {}
    baseline_model_path = model_dir / f"{sp}_lead60.lgb"
    if baseline_model_path.exists() and len(test_df) > 0 and test_df[TARGET].sum() > 0:
        baseline_model = lgb.Booster(model_file=str(baseline_model_path))
        y_true_test = test_df[TARGET].values
        y_prob_base = baseline_model.predict(test_df[feature_cols])
        baseline_test_metrics = compute_metrics(y_true_test, y_prob_base)
        baseline_event = compute_event_recall(test_df.index, y_true_test, y_prob_base)

    # Save tuned model
    model_dir.mkdir(parents=True, exist_ok=True)
    tuned_model_path = model_dir / f"{sp}_lead60_tuned.lgb"
    tuned_model.save_model(str(tuned_model_path))
    logger.info("Tuned model saved: {}", tuned_model_path)

    # Save study results
    study_results = {
        "settlement_point": sp,
        "n_trials": n_trials,
        "best_trial": study.best_trial.number,
        "best_val_pr_auc": study.best_value,
        "best_params": {k: v for k, v in best_params.items()
                        if k not in ("objective", "metric", "verbose")},
        "tuned_test_metrics": tuned_test_metrics,
        "tuned_test_event": tuned_event,
    }
    results_path = model_dir / f"{sp}_tuning_results.json"
    with open(results_path, "w") as f:
        json.dump(study_results, f, indent=2, default=str)

    # Comparison dict
    comparison = {
        "settlement_point": sp,
        "baseline": {
            "pr_auc": baseline_test_metrics.get("pr_auc"),
            "roc_auc": baseline_test_metrics.get("roc_auc"),
            "event_recall": baseline_event.get("event_recall"),
        },
        "tuned": {
            "pr_auc": tuned_test_metrics.get("pr_auc"),
            "roc_auc": tuned_test_metrics.get("roc_auc"),
            "event_recall": tuned_event.get("event_recall"),
        },
    }
    return comparison


def print_comparison_table(comparisons):
    """Print a side-by-side comparison table of baseline vs tuned metrics."""
    header = (
        f"{'SP':<15} {'Base PR-AUC':>11} {'Tuned PR-AUC':>12} "
        f"{'Base ROC':>9} {'Tuned ROC':>10} "
        f"{'Base EvRec':>10} {'Tuned EvRec':>11}"
    )
    logger.info("\n=== Baseline vs Tuned (Test Set) ===")
    logger.info(header)
    for c in comparisons:
        b = c["baseline"]
        t = c["tuned"]
        logger.info(
            "{:<15} {:>11.4f} {:>12.4f} {:>9.4f} {:>10.4f} {:>10.2f} {:>11.2f}",
            c["settlement_point"],
            b.get("pr_auc") or float("nan"),
            t.get("pr_auc") or float("nan"),
            b.get("roc_auc") or float("nan"),
            t.get("roc_auc") or float("nan"),
            b.get("event_recall") or float("nan"),
            t.get("event_recall") or float("nan"),
        )


def main():
    parser = argparse.ArgumentParser(description="Tune spike models with Optuna")
    parser.add_argument("--sp", type=str, nargs="*", default=None,
                        help="Settlement points to tune (default: all)")
    parser.add_argument("--n-trials", type=int, default=50,
                        help="Number of Optuna trials per SP (default: 50)")
    parser.add_argument("--feature-dir", type=str, default=str(FEATURE_DIR))
    parser.add_argument("--model-dir", type=str, default=str(MODEL_DIR))
    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)
    model_dir = Path(args.model_dir)

    sps = [s.upper() for s in args.sp] if args.sp else AVAILABLE_SPS

    logger.info("Tuning spike models for {} SPs, {} trials each", len(sps), args.n_trials)

    all_comparisons = []
    t0 = time.time()

    for sp in sps:
        try:
            comparison = tune_sp(sp, args.n_trials, feature_dir, model_dir)
            if comparison:
                all_comparisons.append(comparison)
        except Exception as e:
            logger.error("Failed to tune {}: {}", sp, e)

    if all_comparisons:
        print_comparison_table(all_comparisons)

        # Save comparison file
        comparison_path = model_dir / "tuning_comparison.json"
        with open(comparison_path, "w") as f:
            json.dump(all_comparisons, f, indent=2, default=str)
        logger.info("Comparison saved: {}", comparison_path)

    logger.info("Total time: {:.1f}s", time.time() - t0)


if __name__ == "__main__":
    main()
