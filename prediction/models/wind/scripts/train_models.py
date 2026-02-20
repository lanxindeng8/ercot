#!/usr/bin/env python
"""
Train Models Script

Trains wind forecasting models and evaluates performance.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from loguru import logger
import numpy as np
import pandas as pd

from models.gbm_model import GBMWindModel
from models.lstm_model import LSTMWindModel
from models.ensemble import EnsembleWindModel
from evaluation.metrics import compute_all_metrics, compute_metrics_by_horizon
from evaluation.ramp_metrics import (
    compute_ramp_metrics,
    evaluate_ramp_down_in_no_solar,
    generate_ramp_report,
)
from utils.config import load_config


def load_data(
    features_path: str,
    target_path: str,
    train_start: str,
    train_end: str,
    val_start: str,
    val_end: str,
    test_start: str,
    test_end: str,
):
    """
    Load and split data.

    Returns:
        Tuple of train, val, test DataFrames
    """
    # Load features
    features_df = pd.read_parquet(features_path)

    # Load targets (wind generation)
    if Path(target_path).exists():
        targets_df = pd.read_parquet(target_path)
        df = features_df.merge(targets_df, on='valid_time', how='inner')
    else:
        logger.warning(f"Target file not found: {target_path}")
        # For demo, create synthetic target
        logger.info("Creating synthetic target for demonstration")
        capacity = 40000  # MW
        df = features_df.copy()
        df['wind_generation'] = df['normalized_power_mean'] * capacity * (0.8 + 0.4 * np.random.rand(len(df)))

    # Ensure datetime index
    df['valid_time'] = pd.to_datetime(df['valid_time'])

    # Split data
    train_mask = (df['valid_time'] >= train_start) & (df['valid_time'] < train_end)
    val_mask = (df['valid_time'] >= val_start) & (df['valid_time'] < val_end)
    test_mask = (df['valid_time'] >= test_start) & (df['valid_time'] < test_end)

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()

    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    return train_df, val_df, test_df


def prepare_features(df: pd.DataFrame, feature_cols: list):
    """Extract features and target from DataFrame."""
    X = df[feature_cols].copy()
    y = df['wind_generation'].copy()
    return X, y


def main():
    parser = argparse.ArgumentParser(description='Train wind forecasting models')
    parser.add_argument(
        '--features',
        type=str,
        default='data/features.parquet',
        help='Features file',
    )
    parser.add_argument(
        '--targets',
        type=str,
        default='data/wind_generation.parquet',
        help='Target file',
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Configuration file',
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gbm',
        choices=['gbm', 'lstm', 'ensemble'],
        help='Model to train',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models',
        help='Output directory for trained models',
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Load data
    train_df, val_df, test_df = load_data(
        features_path=args.features,
        target_path=args.targets,
        train_start=config.training.train_start,
        train_end=config.training.train_end,
        val_start=config.training.val_start,
        val_end=config.training.val_end,
        test_start=config.training.test_start,
        test_end=config.training.test_end,
    )

    # Define feature columns (exclude metadata and target)
    exclude_cols = ['valid_time', 'init_time', 'wind_generation', 'lead_time']
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    logger.info(f"Using {len(feature_cols)} features")

    # Prepare data
    X_train, y_train = prepare_features(train_df, feature_cols)
    X_val, y_val = prepare_features(val_df, feature_cols)
    X_test, y_test = prepare_features(test_df, feature_cols)

    # Train model
    if args.model == 'gbm':
        logger.info("Training LightGBM model...")
        model = GBMWindModel(
            quantiles=config.model.quantiles,
            n_estimators=config.model.gbm_n_estimators,
            learning_rate=config.model.gbm_learning_rate,
            max_depth=config.model.gbm_max_depth,
            num_leaves=config.model.gbm_num_leaves,
            early_stopping_rounds=config.model.gbm_early_stopping_rounds,
            random_state=config.model.random_state,
        )

    elif args.model == 'lstm':
        logger.info("Training LSTM model...")
        model = LSTMWindModel(
            quantiles=config.model.quantiles,
            seq_length=config.model.lstm_seq_length,
            hidden_dim=config.model.lstm_hidden_dim,
            num_layers=config.model.lstm_num_layers,
            dropout=config.model.lstm_dropout,
            learning_rate=config.model.lstm_learning_rate,
            batch_size=config.model.lstm_batch_size,
            epochs=config.model.lstm_epochs,
            patience=config.model.lstm_patience,
        )

    elif args.model == 'ensemble':
        logger.info("Training ensemble model...")
        gbm = GBMWindModel(
            quantiles=config.model.quantiles,
            n_estimators=config.model.gbm_n_estimators,
            learning_rate=config.model.gbm_learning_rate,
            random_state=config.model.random_state,
        )
        lstm = LSTMWindModel(
            quantiles=config.model.quantiles,
            seq_length=config.model.lstm_seq_length,
            hidden_dim=config.model.lstm_hidden_dim,
            epochs=50,  # Reduced for ensemble
        )
        model = EnsembleWindModel(models=[gbm, lstm])

    # Train
    model.fit(X_train, y_train, X_val, y_val)

    # Generate predictions
    y_pred = model.predict(X_test)
    quantile_preds = model.predict_quantiles(X_test)

    # Evaluate
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)

    # Standard metrics
    metrics = compute_all_metrics(
        y_true=y_test.values,
        y_pred=y_pred,
        capacity=config.data.wind_capacity,
        quantile_preds=quantile_preds,
    )

    logger.info("\nStandard Metrics:")
    for name, value in metrics.items():
        logger.info(f"  {name}: {value:.4f}")

    # Ramp metrics
    if 'valid_time' in test_df.columns:
        timestamps = pd.DatetimeIndex(test_df['valid_time'])

        logger.info("\nRamp Detection Metrics:")
        ramp_metrics = compute_ramp_metrics(
            y_true=y_test.values,
            y_pred=y_pred,
            timestamps=timestamps,
            threshold=config.ramp.ramp_threshold_medium,
            window=config.ramp.ramp_window,
        )
        for name, value in ramp_metrics.items():
            logger.info(f"  {name}: {value}")

        # Critical: Ramp-down during no-solar
        logger.info("\nRamp-Down No-Solar Metrics (CRITICAL):")
        no_solar_metrics = evaluate_ramp_down_in_no_solar(
            y_true=y_test.values,
            y_pred=y_pred,
            timestamps=timestamps,
            threshold=-config.ramp.ramp_threshold_medium,
        )
        for name, value in no_solar_metrics.items():
            logger.info(f"  {name}: {value}")

        # Full report
        report = generate_ramp_report(y_test.values, y_pred, timestamps)
        logger.info("\n" + report)

    # Save model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{args.model}_model"
    model.save(str(model_path))
    logger.info(f"\nModel saved to {model_path}")


if __name__ == '__main__':
    main()
