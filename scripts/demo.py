#!/usr/bin/env python
"""
Demo Script - Wind Forecasting with Synthetic Data

Demonstrates the full pipeline without requiring real HRRR data.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
from loguru import logger

from features.wind_features import WindFeatureEngineer
from features.temporal_features import TemporalFeatureEngineer
from features.ramp_features import RampFeatureEngineer
from models.gbm_model import GBMWindModel
from evaluation.metrics import compute_all_metrics
from evaluation.ramp_metrics import (
    compute_ramp_metrics,
    evaluate_ramp_down_in_no_solar,
    generate_ramp_report,
)


def generate_synthetic_data(n_days: int = 60) -> pd.DataFrame:
    """
    Generate synthetic wind data for demonstration.

    Simulates realistic patterns:
    - Diurnal cycle (wind typically stronger at night in Texas)
    - Seasonal variation
    - Random weather events (ramps)
    """
    np.random.seed(42)

    # Generate hourly timestamps
    timestamps = pd.date_range('2024-01-01', periods=n_days * 24, freq='h')
    n = len(timestamps)

    # Base wind speed with diurnal pattern
    hours = timestamps.hour.values
    diurnal = 2 * np.sin(2 * np.pi * (hours - 6) / 24)  # Peak around midnight

    # Seasonal pattern (stronger in spring)
    days = timestamps.dayofyear.values
    seasonal = 1.5 * np.sin(2 * np.pi * (days - 90) / 365)  # Peak in spring

    # Base wind speed
    base_ws = 8 + diurnal + seasonal + np.random.normal(0, 2, n)
    base_ws = np.clip(base_ws, 0, 25)

    # Add some ramp events
    ramp_indices = np.random.choice(n - 6, size=10, replace=False)
    for idx in ramp_indices:
        # Random ramp direction
        if np.random.rand() > 0.5:
            # Ramp up
            base_ws[idx:idx+4] += np.linspace(0, 6, 4)
        else:
            # Ramp down
            base_ws[idx:idx+4] -= np.linspace(0, 6, 4)

    base_ws = np.clip(base_ws, 0, 25)

    # Convert to power (simplified power curve)
    wind_engineer = WindFeatureEngineer()
    normalized_power = wind_engineer.apply_power_curve(base_ws)
    capacity = 40000  # MW
    wind_generation = normalized_power * capacity * (0.85 + 0.15 * np.random.rand(n))

    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'ws_80m_mean': base_ws,
        'ws_80m_std': np.abs(np.random.normal(1.5, 0.5, n)),
        'normalized_power': normalized_power,
        'wind_generation': wind_generation,
    })

    # Add temporal features
    temporal_eng = TemporalFeatureEngineer()
    temporal_features = temporal_eng.compute_features(timestamps)
    df = pd.concat([df, temporal_features.reset_index(drop=True)], axis=1)

    return df


def main():
    logger.info("=" * 60)
    logger.info("Wind Forecasting Demo with Synthetic Data")
    logger.info("=" * 60)

    # Generate data
    logger.info("\n1. Generating synthetic data...")
    df = generate_synthetic_data(n_days=90)
    logger.info(f"   Generated {len(df)} hourly samples")

    # Split data
    train_end = int(len(df) * 0.7)
    val_end = int(len(df) * 0.85)

    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]

    logger.info(f"   Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Prepare features
    feature_cols = [
        'ws_80m_mean', 'ws_80m_std', 'normalized_power',
        'hour_sin', 'hour_cos', 'doy_sin', 'doy_cos',
        'is_peak_hour', 'is_ramp_prone_hour',
    ]

    X_train = train_df[feature_cols]
    y_train = train_df['wind_generation']
    X_val = val_df[feature_cols]
    y_val = val_df['wind_generation']
    X_test = test_df[feature_cols]
    y_test = test_df['wind_generation']

    # Train model
    logger.info("\n2. Training LightGBM model...")
    model = GBMWindModel(
        quantiles=[0.1, 0.5, 0.9],
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        verbose=-1,
    )
    model.fit(X_train, y_train, X_val, y_val)

    # Predict
    logger.info("\n3. Generating predictions...")
    y_pred = model.predict(X_test)
    quantile_preds = model.predict_quantiles(X_test)

    # Evaluate
    logger.info("\n4. Evaluation Results")
    logger.info("-" * 40)

    # Standard metrics
    metrics = compute_all_metrics(
        y_true=y_test.values,
        y_pred=y_pred,
        capacity=40000,
        quantile_preds=quantile_preds,
    )

    logger.info("\nStandard Metrics:")
    logger.info(f"   MAE:  {metrics['mae']:.1f} MW")
    logger.info(f"   RMSE: {metrics['rmse']:.1f} MW")
    logger.info(f"   NMAE: {metrics['nmae']:.2f}%")
    logger.info(f"   Coverage (80%): {metrics.get('coverage_80', 0):.2%}")

    # Ramp metrics
    timestamps = pd.DatetimeIndex(test_df['timestamp'])

    ramp_metrics = compute_ramp_metrics(
        y_true=y_test.values,
        y_pred=y_pred,
        timestamps=timestamps,
        threshold=3000,
        window=3,
    )

    logger.info("\nRamp Detection (threshold=3000 MW):")
    logger.info(f"   Actual events:    {ramp_metrics['n_actual_events']}")
    logger.info(f"   Predicted events: {ramp_metrics['n_predicted_events']}")
    logger.info(f"   POD (hit rate):   {ramp_metrics['pod']:.2f}")
    logger.info(f"   FAR:              {ramp_metrics['far']:.2f}")
    logger.info(f"   CSI:              {ramp_metrics['csi']:.2f}")

    # Ramp-down in no-solar (critical scenario)
    no_solar_metrics = evaluate_ramp_down_in_no_solar(
        y_true=y_test.values,
        y_pred=y_pred,
        timestamps=timestamps,
        threshold=-3000,
    )

    logger.info("\nRamp-Down in No-Solar Period (CRITICAL):")
    logger.info(f"   Events:    {no_solar_metrics['n_actual_ramp_down_no_solar']}")
    logger.info(f"   POD:       {no_solar_metrics['pod_no_solar']:.2f}")
    logger.info(f"   Miss rate: {no_solar_metrics['miss_rate_no_solar']:.2f}")

    # Feature importance
    logger.info("\n5. Feature Importance (Top 5):")
    importance = model.get_feature_importance()
    if importance is not None:
        for feat, imp in importance.head(5).items():
            logger.info(f"   {feat}: {imp:.0f}")

    # Sample predictions
    logger.info("\n6. Sample Predictions (first 5 test hours):")
    logger.info("-" * 60)
    logger.info(f"{'Time':<20} {'Actual':>10} {'Pred':>10} {'P10':>10} {'P90':>10}")
    logger.info("-" * 60)

    for i in range(5):
        ts = test_df['timestamp'].iloc[i]
        actual = y_test.iloc[i]
        pred = y_pred[i]
        p10 = quantile_preds[0.1][i]
        p90 = quantile_preds[0.9][i]
        logger.info(f"{str(ts):<20} {actual:>10.0f} {pred:>10.0f} {p10:>10.0f} {p90:>10.0f}")

    logger.info("\n" + "=" * 60)
    logger.info("Demo complete!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
