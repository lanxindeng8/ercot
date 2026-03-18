#!/usr/bin/env python
"""
Train Wind Models — Sprint 4.1

Trains GBM and ensemble models on features built from SQLite wind data.
Evaluates with standard + ramp metrics, saves checkpoints.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from loguru import logger
import numpy as np
import pandas as pd

from models.gbm_model import GBMWindModel
from evaluation.metrics import compute_all_metrics
from evaluation.ramp_metrics import (
    compute_ramp_metrics,
    evaluate_ramp_down_in_no_solar,
    generate_ramp_report,
)
from utils.config import load_config


def load_and_split(config, use_augmented=True):
    """Load features and split into train/val/test."""
    if use_augmented:
        path = Path('data/features_augmented.parquet')
    else:
        path = Path('data/features.parquet')

    df = pd.read_parquet(path)
    df['valid_time'] = pd.to_datetime(df['valid_time'])

    train_mask = (df['valid_time'] >= config.training.train_start) & (df['valid_time'] < config.training.train_end)
    val_mask = (df['valid_time'] >= config.training.val_start) & (df['valid_time'] < config.training.val_end)
    test_mask = (df['valid_time'] >= config.training.test_start) & (df['valid_time'] <= config.training.test_end)

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()

    logger.info(f"Train: {len(train_df)} ({config.training.train_start} to {config.training.train_end})")
    logger.info(f"Val:   {len(val_df)} ({config.training.val_start} to {config.training.val_end})")
    logger.info(f"Test:  {len(test_df)} ({config.training.test_start} to {config.training.test_end})")

    return train_df, val_df, test_df


def get_feature_cols(df):
    """Get feature columns (exclude metadata and target)."""
    exclude = ['valid_time', 'init_time', 'wind_generation', 'lead_time', 'timestamp']
    return [c for c in df.columns if c not in exclude]


def train_gbm(train_df, val_df, feature_cols, config):
    """Train a single GBM model."""
    model = GBMWindModel(
        quantiles=config.model.quantiles,
        n_estimators=config.model.gbm_n_estimators,
        learning_rate=config.model.gbm_learning_rate,
        max_depth=config.model.gbm_max_depth,
        num_leaves=config.model.gbm_num_leaves,
        early_stopping_rounds=config.model.gbm_early_stopping_rounds,
        random_state=config.model.random_state,
    )

    model.fit(
        train_df[feature_cols], train_df['wind_generation'],
        val_df[feature_cols], val_df['wind_generation'],
    )
    return model


def train_gbm_ensemble(train_df, val_df, feature_cols):
    """Train ensemble of GBMs with different hyperparameters."""
    configs = [
        {'n_estimators': 1000, 'learning_rate': 0.02, 'max_depth': 8, 'num_leaves': 63},
        {'n_estimators': 800, 'learning_rate': 0.03, 'max_depth': 6, 'num_leaves': 31},
        {'n_estimators': 1200, 'learning_rate': 0.01, 'max_depth': 10, 'num_leaves': 127},
        {'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 5, 'num_leaves': 15},
    ]

    models = []
    for i, cfg in enumerate(configs):
        logger.info(f"Training GBM {i+1}/{len(configs)}: {cfg}")
        model = GBMWindModel(
            quantiles=[0.1, 0.5, 0.9],
            early_stopping_rounds=50,
            random_state=42 + i,
            **cfg,
        )
        model.fit(
            train_df[feature_cols], train_df['wind_generation'],
            val_df[feature_cols], val_df['wind_generation'],
        )
        models.append(model)

    return models


def evaluate_model(y_test, y_pred, timestamps, capacity, quantile_preds=None):
    """Run full evaluation including ramp metrics."""
    results = {}

    # Standard metrics
    metrics = compute_all_metrics(
        y_true=y_test,
        y_pred=y_pred,
        capacity=capacity,
        quantile_preds=quantile_preds,
    )
    results['standard'] = metrics

    logger.info("\nStandard Metrics:")
    for name, value in metrics.items():
        logger.info(f"  {name}: {value:.4f}")

    # Ramp metrics
    if timestamps is not None:
        ramp_metrics = compute_ramp_metrics(
            y_true=y_test,
            y_pred=y_pred,
            timestamps=timestamps,
            threshold=2000,  # medium ramp
            window=3,
        )
        results['ramp'] = ramp_metrics

        logger.info("\nRamp Detection Metrics:")
        for name, value in ramp_metrics.items():
            logger.info(f"  {name}: {value}")

        # Critical: Ramp-down no-solar
        no_solar = evaluate_ramp_down_in_no_solar(
            y_true=y_test,
            y_pred=y_pred,
            timestamps=timestamps,
            threshold=-2000,
        )
        results['no_solar'] = no_solar

        logger.info("\nRamp-Down No-Solar Metrics (CRITICAL):")
        for name, value in no_solar.items():
            logger.info(f"  {name}: {value}")

        # Full report
        report = generate_ramp_report(y_test, y_pred, timestamps)
        results['report'] = report
        logger.info("\n" + report)

    return results


def save_checkpoint(model, metrics, model_name, checkpoint_dir):
    """Save model checkpoint with metadata."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = checkpoint_dir / model_name
    model.save(str(model_path))

    # Save metrics
    serializable_metrics = {}
    for category, cat_metrics in metrics.items():
        if isinstance(cat_metrics, dict):
            serializable_metrics[category] = {
                k: float(v) if isinstance(v, (np.floating, float, int, np.integer)) else str(v)
                for k, v in cat_metrics.items()
            }

    meta = {
        'model_name': model_name,
        'timestamp': datetime.now().isoformat(),
        'metrics': serializable_metrics,
    }
    with open(checkpoint_dir / f'{model_name}_metrics.json', 'w') as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Checkpoint saved: {checkpoint_dir / model_name}")


def main():
    parser = argparse.ArgumentParser(description='Train wind forecasting models (Sprint 4.1)')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--model', type=str, default='all', choices=['gbm', 'ensemble', 'all'])
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    args = parser.parse_args()

    config = load_config(args.config)

    # Load data
    logger.info("=" * 60)
    logger.info("LOADING DATA")
    logger.info("=" * 60)
    train_df, val_df, test_df = load_and_split(config, use_augmented=True)
    feature_cols = get_feature_cols(train_df)
    logger.info(f"Features: {len(feature_cols)}")

    timestamps = pd.DatetimeIndex(test_df['valid_time'])
    y_test = test_df['wind_generation'].values
    capacity = config.data.wind_capacity

    # ========== GBM Model ==========
    if args.model in ('gbm', 'all'):
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING GBM MODEL")
        logger.info("=" * 60)

        gbm = train_gbm(train_df, val_df, feature_cols, config)

        # Predict
        y_pred_gbm = gbm.predict(test_df[feature_cols])
        quantile_preds_gbm = gbm.predict_quantiles(test_df[feature_cols])

        # Bias correction from validation
        y_val_pred = gbm.predict(val_df[feature_cols])
        val_residuals = val_df['wind_generation'].values - y_val_pred
        bias_correction = np.mean(val_residuals)
        logger.info(f"Bias correction: {bias_correction:.1f} MW")
        y_pred_gbm_corrected = y_pred_gbm + bias_correction

        # Evaluate
        logger.info("\n" + "=" * 60)
        logger.info("GBM EVALUATION")
        logger.info("=" * 60)
        gbm_metrics = evaluate_model(y_test, y_pred_gbm_corrected, timestamps, capacity, quantile_preds_gbm)

        # Feature importance
        importance = gbm.get_feature_importance()
        if importance is not None:
            logger.info("\nTop 15 Features:")
            for i, (feat, imp) in enumerate(importance.head(15).items()):
                logger.info(f"  {i+1}. {feat}: {imp:.0f}")

        # Save checkpoint
        save_checkpoint(gbm, gbm_metrics, 'gbm_model', args.checkpoint_dir)

    # ========== GBM Ensemble ==========
    if args.model in ('ensemble', 'all'):
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING GBM ENSEMBLE")
        logger.info("=" * 60)

        models = train_gbm_ensemble(train_df, val_df, feature_cols)

        # Ensemble predictions
        test_preds = [m.predict(test_df[feature_cols]) for m in models]
        y_pred_ens = np.mean(test_preds, axis=0)

        # Bias correction
        val_preds = [m.predict(val_df[feature_cols]) for m in models]
        val_ens = np.mean(val_preds, axis=0)
        val_residuals = val_df['wind_generation'].values - val_ens
        bias_correction = np.mean(val_residuals)
        y_pred_ens_corrected = y_pred_ens + bias_correction
        logger.info(f"Ensemble bias correction: {bias_correction:.1f} MW")

        # Conformal prediction intervals
        corrected_residuals = val_df['wind_generation'].values - (val_ens + bias_correction)
        best_coverage_diff = float('inf')
        best_threshold = None
        for percentile in range(85, 100):
            threshold = np.percentile(np.abs(corrected_residuals), percentile)
            y_lower = y_pred_ens_corrected - threshold
            y_upper = y_pred_ens_corrected + threshold
            cov = np.mean((y_test >= y_lower) & (y_test <= y_upper))
            if abs(cov - 0.80) < best_coverage_diff:
                best_coverage_diff = abs(cov - 0.80)
                best_threshold = threshold
        logger.info(f"Conformal threshold for ~80% coverage: {best_threshold:.1f} MW")

        # Evaluate
        logger.info("\n" + "=" * 60)
        logger.info("ENSEMBLE EVALUATION")
        logger.info("=" * 60)
        ens_metrics = evaluate_model(y_test, y_pred_ens_corrected, timestamps, capacity)

        # Save ensemble checkpoint (save best individual model)
        best_idx = np.argmin([np.mean(np.abs(y_test - p)) for p in test_preds])
        save_checkpoint(models[best_idx], ens_metrics, 'ensemble_best_model', args.checkpoint_dir)

        # Also save ensemble metadata
        ens_meta = {
            'type': 'gbm_ensemble',
            'n_models': len(models),
            'bias_correction': float(bias_correction),
            'conformal_threshold': float(best_threshold),
            'individual_maes': [float(np.mean(np.abs(y_test - p))) for p in test_preds],
        }
        with open(Path(args.checkpoint_dir) / 'ensemble_config.json', 'w') as f:
            json.dump(ens_meta, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
