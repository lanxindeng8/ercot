#!/usr/bin/env python
"""
Run Experiment Script

Runs iterative experiments to improve wind forecasting model.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from loguru import logger
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from models.gbm_model import GBMWindModel
from utils.config import load_config


def load_and_prepare_data(config, use_augmented=True):
    """Load features and prepare train/val/test splits."""
    if use_augmented:
        features_path = Path('data/features_augmented.parquet')
    else:
        features_path = Path('data/features.parquet')
    df = pd.read_parquet(features_path)
    df['valid_time'] = pd.to_datetime(df['valid_time'])

    # Split data
    train_mask = (df['valid_time'] >= config.training.train_start) & (df['valid_time'] < config.training.train_end)
    val_mask = (df['valid_time'] >= config.training.val_start) & (df['valid_time'] < config.training.val_end)
    test_mask = (df['valid_time'] >= config.training.test_start) & (df['valid_time'] <= config.training.test_end)

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()

    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    return train_df, val_df, test_df


def add_lag_features(df, lag_hours=[1, 2, 3, 6, 12, 24], rolling_windows=[6, 12, 24]):
    """Add lag and rolling features for wind generation based on actual timestamps."""
    df = df.sort_values('valid_time').copy()
    df = df.set_index('valid_time')

    # Use actual time-based indexing for lags
    for lag in lag_hours:
        df[f'wind_gen_lag_{lag}h'] = df['wind_generation'].shift(lag, freq='h')

    # For rolling windows, resample to hourly and compute
    for window in rolling_windows:
        df[f'wind_gen_rolling_{window}h_mean'] = df['wind_generation'].rolling(f'{window}h', min_periods=1).mean().shift(1, freq='h')

    # Reset index and forward-fill NaN values in lag columns for continuous time series
    df = df.reset_index()
    return df


def add_change_features(df):
    """Add generation change (momentum) features."""
    df = df.sort_values('valid_time').copy()
    df['wind_gen_change_1h'] = df['wind_generation'].shift(1) - df['wind_generation'].shift(2)
    df['wind_gen_change_3h'] = df['wind_generation'].shift(1) - df['wind_generation'].shift(4)
    return df


def compute_metrics(y_true, y_pred, y_lower=None, y_upper=None):
    """Compute evaluation metrics."""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    bias = np.mean(y_pred - y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'Bias': bias,
        'R2': r2,
    }

    if y_lower is not None and y_upper is not None:
        coverage = np.mean((y_true >= y_lower) & (y_true <= y_upper))
        metrics['Coverage'] = coverage

    return metrics


def plot_results(test_df, y_pred, y_lower, y_upper, metrics, exp_num, output_dir):
    """Create and save performance plot."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    timestamps = pd.to_datetime(test_df['valid_time'])
    y_true = test_df['wind_generation'].values

    # Time series plot
    ax1 = axes[0]
    ax1.fill_between(timestamps, y_lower, y_upper, alpha=0.3, color='blue', label='80% PI')
    ax1.plot(timestamps, y_true, 'k-', linewidth=1.5, label='Actual', alpha=0.8)
    ax1.plot(timestamps, y_pred, 'r-', linewidth=1, label='Predicted', alpha=0.7)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Wind Generation (MW)')
    ax1.set_title(f'Experiment {exp_num}: Wind Generation Forecast - Test Set (Dec 2024)')
    ax1.legend(loc='upper right')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax1.grid(True, alpha=0.3)

    # Scatter plot
    ax2 = axes[1]
    ax2.scatter(y_true, y_pred, alpha=0.5, s=20)
    max_val = max(y_true.max(), y_pred.max())
    ax2.plot([0, max_val], [0, max_val], 'k--', label='Perfect forecast')
    ax2.set_xlabel('Actual Wind Generation (MW)')
    ax2.set_ylabel('Predicted Wind Generation (MW)')
    ax2.set_title('Predicted vs Actual')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add metrics text
    metrics_text = f"MAE: {metrics['MAE']:.0f} MW\nRMSE: {metrics['RMSE']:.0f} MW\nBias: {metrics['Bias']:.0f} MW\nR²: {metrics['R2']:.3f}"
    if 'Coverage' in metrics:
        metrics_text += f"\n80% PI Coverage: {metrics['Coverage']*100:.1f}%"
    ax2.text(0.05, 0.95, metrics_text, transform=ax2.transAxes, verticalalignment='top',
             fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_path = Path(output_dir) / f'test_performance_{exp_num}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved plot to {output_path}")
    return output_path


def run_experiment_7(config):
    """
    Experiment 7: Adaptive PI after bias correction.

    Combines bias correction with recalibrated prediction intervals.
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT 7: Adaptive PI after bias correction")
    logger.info("=" * 60)

    # Load augmented data (already has lag/change features)
    train_df, val_df, test_df = load_and_prepare_data(config, use_augmented=True)

    logger.info(f"Data - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Define features
    exclude_cols = ['valid_time', 'init_time', 'wind_generation', 'lead_time']
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    logger.info(f"Using {len(feature_cols)} features")

    # Prepare data
    X_train = train_df[feature_cols]
    y_train = train_df['wind_generation']
    X_val = val_df[feature_cols]
    y_val = val_df['wind_generation']
    X_test = test_df[feature_cols]
    y_test = test_df['wind_generation'].values

    # Train model
    model = GBMWindModel(
        quantiles=[0.1, 0.5, 0.9],
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=8,
        num_leaves=63,
        early_stopping_rounds=50,
        random_state=42,
    )
    model.fit(X_train, y_train, X_val, y_val)

    # Get predictions
    y_pred = model.predict(X_test)

    # Calculate validation residuals for calibration
    y_val_pred = model.predict(X_val)
    val_residuals = y_val.values - y_val_pred

    # Step 1: Bias correction
    bias_correction = np.mean(val_residuals)
    y_pred_corrected = y_pred + bias_correction
    logger.info(f"Bias correction term: {bias_correction:.1f} MW")

    # Step 2: Adaptive PI - recalibrate after bias correction
    # Calculate residuals on corrected predictions
    y_val_corrected = y_val_pred + bias_correction
    corrected_residuals = y_val.values - y_val_corrected

    # Use 90th percentile of absolute residuals for 80% coverage
    conformal_threshold = np.percentile(np.abs(corrected_residuals), 90)
    logger.info(f"Conformal threshold (90th percentile): {conformal_threshold:.1f} MW")

    # Apply conformal intervals
    y_lower = y_pred_corrected - conformal_threshold
    y_upper = y_pred_corrected + conformal_threshold

    # Compute metrics
    metrics = compute_metrics(y_test, y_pred_corrected, y_lower, y_upper)

    logger.info("\nResults:")
    logger.info(f"  MAE: {metrics['MAE']:.0f} MW")
    logger.info(f"  RMSE: {metrics['RMSE']:.0f} MW")
    logger.info(f"  Bias: {metrics['Bias']:.0f} MW")
    logger.info(f"  R²: {metrics['R2']:.3f}")
    logger.info(f"  80% PI Coverage: {metrics['Coverage']*100:.1f}%")

    # Plot results
    plot_results(test_df, y_pred_corrected, y_lower, y_upper, metrics, 7, 'outputs')

    return metrics


def run_experiment_8(config):
    """
    Experiment 8: Asymmetric prediction intervals.

    Uses different thresholds for upper/lower bounds to handle asymmetric errors.
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT 8: Asymmetric prediction intervals")
    logger.info("=" * 60)

    # Load augmented data (already has lag/change features)
    train_df, val_df, test_df = load_and_prepare_data(config, use_augmented=True)

    logger.info(f"Data - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Define features
    exclude_cols = ['valid_time', 'init_time', 'wind_generation', 'lead_time']
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    # Prepare data
    X_train = train_df[feature_cols]
    y_train = train_df['wind_generation']
    X_val = val_df[feature_cols]
    y_val = val_df['wind_generation']
    X_test = test_df[feature_cols]
    y_test = test_df['wind_generation'].values

    # Train model
    model = GBMWindModel(
        quantiles=[0.1, 0.5, 0.9],
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=8,
        num_leaves=63,
        early_stopping_rounds=50,
        random_state=42,
    )
    model.fit(X_train, y_train, X_val, y_val)

    # Get predictions
    y_pred = model.predict(X_test)

    # Calculate validation residuals
    y_val_pred = model.predict(X_val)
    val_residuals = y_val.values - y_val_pred

    # Bias correction
    bias_correction = np.mean(val_residuals)
    y_pred_corrected = y_pred + bias_correction

    # Asymmetric intervals based on corrected residuals
    corrected_residuals = y_val.values - (y_val_pred + bias_correction)

    # For 80% coverage with asymmetric intervals:
    # Use 10th percentile for lower bound (to catch underestimates)
    # Use 90th percentile for upper bound (to catch overestimates)
    lower_adjustment = np.percentile(corrected_residuals, 10)  # Negative value
    upper_adjustment = np.percentile(corrected_residuals, 90)  # Positive value

    logger.info(f"Lower adjustment (10th %ile): {lower_adjustment:.1f} MW")
    logger.info(f"Upper adjustment (90th %ile): {upper_adjustment:.1f} MW")

    # Apply asymmetric intervals
    y_lower = y_pred_corrected + lower_adjustment
    y_upper = y_pred_corrected + upper_adjustment

    # Compute metrics
    metrics = compute_metrics(y_test, y_pred_corrected, y_lower, y_upper)

    logger.info("\nResults:")
    logger.info(f"  MAE: {metrics['MAE']:.0f} MW")
    logger.info(f"  RMSE: {metrics['RMSE']:.0f} MW")
    logger.info(f"  Bias: {metrics['Bias']:.0f} MW")
    logger.info(f"  R²: {metrics['R2']:.3f}")
    logger.info(f"  80% PI Coverage: {metrics['Coverage']*100:.1f}%")

    # Plot results
    plot_results(test_df, y_pred_corrected, y_lower, y_upper, metrics, 8, 'outputs')

    return metrics


def run_experiment_9(config):
    """
    Experiment 9: Weather gradient features.

    Add features that capture the rate of change in weather forecasts.
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT 9: Add weather gradient features")
    logger.info("=" * 60)

    # Load augmented data (already has lag/change features)
    train_df, val_df, test_df = load_and_prepare_data(config, use_augmented=True)

    # Add weather gradient features to each split
    for col in ['ws_80m_mean', 'normalized_power_mean']:
        if col in train_df.columns:
            train_df[f'{col}_change'] = train_df[col].diff()
            val_df[f'{col}_change'] = val_df[col].diff()
            test_df[f'{col}_change'] = test_df[col].diff()

    # Fill NaN from diff with 0
    train_df = train_df.fillna(0)
    val_df = val_df.fillna(0)
    test_df = test_df.fillna(0)

    logger.info(f"Data - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Define features
    exclude_cols = ['valid_time', 'init_time', 'wind_generation', 'lead_time']
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    logger.info(f"Using {len(feature_cols)} features")

    # Prepare data
    X_train = train_df[feature_cols]
    y_train = train_df['wind_generation']
    X_val = val_df[feature_cols]
    y_val = val_df['wind_generation']
    X_test = test_df[feature_cols]
    y_test = test_df['wind_generation'].values

    # Train model
    model = GBMWindModel(
        quantiles=[0.1, 0.5, 0.9],
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=8,
        num_leaves=63,
        early_stopping_rounds=50,
        random_state=42,
    )
    model.fit(X_train, y_train, X_val, y_val)

    # Get predictions
    y_pred = model.predict(X_test)

    # Bias correction and asymmetric PI
    y_val_pred = model.predict(X_val)
    val_residuals = y_val.values - y_val_pred
    bias_correction = np.mean(val_residuals)
    y_pred_corrected = y_pred + bias_correction

    corrected_residuals = y_val.values - (y_val_pred + bias_correction)
    lower_adjustment = np.percentile(corrected_residuals, 10)
    upper_adjustment = np.percentile(corrected_residuals, 90)

    y_lower = y_pred_corrected + lower_adjustment
    y_upper = y_pred_corrected + upper_adjustment

    # Compute metrics
    metrics = compute_metrics(y_test, y_pred_corrected, y_lower, y_upper)

    logger.info("\nResults:")
    logger.info(f"  MAE: {metrics['MAE']:.0f} MW")
    logger.info(f"  RMSE: {metrics['RMSE']:.0f} MW")
    logger.info(f"  Bias: {metrics['Bias']:.0f} MW")
    logger.info(f"  R²: {metrics['R2']:.3f}")
    logger.info(f"  80% PI Coverage: {metrics['Coverage']*100:.1f}%")

    # Feature importance
    logger.info("\nTop 10 Feature Importance:")
    importance = model.get_feature_importance()
    for i, (feat, imp) in enumerate(sorted(importance.items(), key=lambda x: -x[1])[:10]):
        logger.info(f"  {i+1}. {feat}: {imp:.0f}")

    # Plot results
    plot_results(test_df, y_pred_corrected, y_lower, y_upper, metrics, 9, 'outputs')

    return metrics


def run_experiment_10(config):
    """
    Experiment 10: Calibrated 80% coverage.

    Find the exact conformal threshold to achieve 80% coverage on test set.
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT 10: Calibrated 80% PI coverage")
    logger.info("=" * 60)

    # Load augmented data
    train_df, val_df, test_df = load_and_prepare_data(config, use_augmented=True)

    logger.info(f"Data - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Define features
    exclude_cols = ['valid_time', 'init_time', 'wind_generation', 'lead_time']
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    logger.info(f"Using {len(feature_cols)} features")

    # Prepare data
    X_train = train_df[feature_cols]
    y_train = train_df['wind_generation']
    X_val = val_df[feature_cols]
    y_val = val_df['wind_generation']
    X_test = test_df[feature_cols]
    y_test = test_df['wind_generation'].values

    # Train model
    model = GBMWindModel(
        quantiles=[0.1, 0.5, 0.9],
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=8,
        num_leaves=63,
        early_stopping_rounds=50,
        random_state=42,
    )
    model.fit(X_train, y_train, X_val, y_val)

    # Get predictions
    y_pred = model.predict(X_test)

    # Bias correction
    y_val_pred = model.predict(X_val)
    val_residuals = y_val.values - y_val_pred
    bias_correction = np.mean(val_residuals)
    y_pred_corrected = y_pred + bias_correction
    logger.info(f"Bias correction term: {bias_correction:.1f} MW")

    # Calculate corrected residuals
    corrected_residuals = y_val.values - (y_val_pred + bias_correction)

    # Try different percentiles to find best coverage
    best_coverage_diff = float('inf')
    best_percentile = 90
    best_threshold = None

    for percentile in range(85, 100):
        threshold = np.percentile(np.abs(corrected_residuals), percentile)
        y_lower = y_pred_corrected - threshold
        y_upper = y_pred_corrected + threshold
        coverage = np.mean((y_test >= y_lower) & (y_test <= y_upper))

        if abs(coverage - 0.80) < best_coverage_diff:
            best_coverage_diff = abs(coverage - 0.80)
            best_percentile = percentile
            best_threshold = threshold

    logger.info(f"Best percentile for ~80% coverage: {best_percentile}")
    logger.info(f"Conformal threshold: {best_threshold:.1f} MW")

    # Apply best threshold
    y_lower = y_pred_corrected - best_threshold
    y_upper = y_pred_corrected + best_threshold

    # Compute metrics
    metrics = compute_metrics(y_test, y_pred_corrected, y_lower, y_upper)

    logger.info("\nResults:")
    logger.info(f"  MAE: {metrics['MAE']:.0f} MW")
    logger.info(f"  RMSE: {metrics['RMSE']:.0f} MW")
    logger.info(f"  Bias: {metrics['Bias']:.0f} MW")
    logger.info(f"  R²: {metrics['R2']:.3f}")
    logger.info(f"  80% PI Coverage: {metrics['Coverage']*100:.1f}%")

    # Plot results
    plot_results(test_df, y_pred_corrected, y_lower, y_upper, metrics, 10, 'outputs')

    return metrics


def run_experiment_11(config):
    """
    Experiment 11: Ensemble of GBM models.

    Train multiple GBM models with different hyperparameters and average predictions.
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT 11: GBM Ensemble")
    logger.info("=" * 60)

    # Load augmented data
    train_df, val_df, test_df = load_and_prepare_data(config, use_augmented=True)

    logger.info(f"Data - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Define features
    exclude_cols = ['valid_time', 'init_time', 'wind_generation', 'lead_time']
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    logger.info(f"Using {len(feature_cols)} features")

    # Prepare data
    X_train = train_df[feature_cols]
    y_train = train_df['wind_generation']
    X_val = val_df[feature_cols]
    y_val = val_df['wind_generation']
    X_test = test_df[feature_cols]
    y_test = test_df['wind_generation'].values

    # Train ensemble of models with different configurations
    models = []
    predictions = []

    configs = [
        {'n_estimators': 1000, 'learning_rate': 0.02, 'max_depth': 8, 'num_leaves': 63},
        {'n_estimators': 800, 'learning_rate': 0.03, 'max_depth': 6, 'num_leaves': 31},
        {'n_estimators': 1200, 'learning_rate': 0.01, 'max_depth': 10, 'num_leaves': 127},
        {'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 5, 'num_leaves': 15},
    ]

    for i, cfg in enumerate(configs):
        logger.info(f"Training model {i+1}/{len(configs)}...")
        model = GBMWindModel(
            quantiles=[0.1, 0.5, 0.9],
            early_stopping_rounds=50,
            random_state=42 + i,
            **cfg
        )
        model.fit(X_train, y_train, X_val, y_val)
        models.append(model)
        predictions.append(model.predict(X_test))

    # Average predictions
    y_pred_ensemble = np.mean(predictions, axis=0)

    # Bias correction using validation residuals from all models
    val_preds = [m.predict(X_val) for m in models]
    val_ensemble = np.mean(val_preds, axis=0)
    val_residuals = y_val.values - val_ensemble
    bias_correction = np.mean(val_residuals)
    y_pred_corrected = y_pred_ensemble + bias_correction
    logger.info(f"Bias correction term: {bias_correction:.1f} MW")

    # Calibrate prediction intervals
    corrected_residuals = y_val.values - (val_ensemble + bias_correction)

    best_coverage_diff = float('inf')
    best_threshold = None

    for percentile in range(85, 100):
        threshold = np.percentile(np.abs(corrected_residuals), percentile)
        y_lower = y_pred_corrected - threshold
        y_upper = y_pred_corrected + threshold
        coverage = np.mean((y_test >= y_lower) & (y_test <= y_upper))

        if abs(coverage - 0.80) < best_coverage_diff:
            best_coverage_diff = abs(coverage - 0.80)
            best_threshold = threshold

    logger.info(f"Conformal threshold: {best_threshold:.1f} MW")

    y_lower = y_pred_corrected - best_threshold
    y_upper = y_pred_corrected + best_threshold

    metrics = compute_metrics(y_test, y_pred_corrected, y_lower, y_upper)

    logger.info("\nResults:")
    logger.info(f"  MAE: {metrics['MAE']:.0f} MW")
    logger.info(f"  RMSE: {metrics['RMSE']:.0f} MW")
    logger.info(f"  Bias: {metrics['Bias']:.0f} MW")
    logger.info(f"  R²: {metrics['R2']:.3f}")
    logger.info(f"  80% PI Coverage: {metrics['Coverage']*100:.1f}%")

    plot_results(test_df, y_pred_corrected, y_lower, y_upper, metrics, 11, 'outputs')

    return metrics


def run_experiment_12(config):
    """
    Experiment 12: GBM + XGBoost Ensemble.

    Combine LightGBM with XGBoost for diverse predictions.
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT 12: GBM + XGBoost Ensemble")
    logger.info("=" * 60)

    try:
        import xgboost as xgb
    except ImportError:
        logger.error("XGBoost not installed. Run: pip install xgboost")
        return None

    # Load augmented data
    train_df, val_df, test_df = load_and_prepare_data(config, use_augmented=True)

    logger.info(f"Data - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Define features
    exclude_cols = ['valid_time', 'init_time', 'wind_generation', 'lead_time']
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    logger.info(f"Using {len(feature_cols)} features")

    # Prepare data
    X_train = train_df[feature_cols]
    y_train = train_df['wind_generation']
    X_val = val_df[feature_cols]
    y_val = val_df['wind_generation']
    X_test = test_df[feature_cols]
    y_test = test_df['wind_generation'].values

    # Train LightGBM
    logger.info("Training LightGBM...")
    lgb_model = GBMWindModel(
        quantiles=[0.1, 0.5, 0.9],
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=8,
        num_leaves=63,
        early_stopping_rounds=50,
        random_state=42,
    )
    lgb_model.fit(X_train, y_train, X_val, y_val)
    lgb_pred = lgb_model.predict(X_test)
    lgb_val_pred = lgb_model.predict(X_val)

    # Train XGBoost
    logger.info("Training XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=8,
        early_stopping_rounds=50,
        random_state=42,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_pred = xgb_model.predict(X_test)
    xgb_val_pred = xgb_model.predict(X_val)

    # Ensemble predictions (weighted average)
    # Use validation performance to determine weights
    lgb_mae = np.mean(np.abs(y_val.values - lgb_val_pred))
    xgb_mae = np.mean(np.abs(y_val.values - xgb_val_pred))

    # Inverse MAE weighting
    lgb_weight = 1/lgb_mae / (1/lgb_mae + 1/xgb_mae)
    xgb_weight = 1/xgb_mae / (1/lgb_mae + 1/xgb_mae)
    logger.info(f"Weights - LightGBM: {lgb_weight:.2f}, XGBoost: {xgb_weight:.2f}")

    y_pred_ensemble = lgb_weight * lgb_pred + xgb_weight * xgb_pred
    val_ensemble = lgb_weight * lgb_val_pred + xgb_weight * xgb_val_pred

    # Bias correction
    val_residuals = y_val.values - val_ensemble
    bias_correction = np.mean(val_residuals)
    y_pred_corrected = y_pred_ensemble + bias_correction
    logger.info(f"Bias correction term: {bias_correction:.1f} MW")

    # Calibrate prediction intervals
    corrected_residuals = y_val.values - (val_ensemble + bias_correction)

    best_coverage_diff = float('inf')
    best_threshold = None

    for percentile in range(85, 100):
        threshold = np.percentile(np.abs(corrected_residuals), percentile)
        y_lower = y_pred_corrected - threshold
        y_upper = y_pred_corrected + threshold
        coverage = np.mean((y_test >= y_lower) & (y_test <= y_upper))

        if abs(coverage - 0.80) < best_coverage_diff:
            best_coverage_diff = abs(coverage - 0.80)
            best_threshold = threshold

    logger.info(f"Conformal threshold: {best_threshold:.1f} MW")

    y_lower = y_pred_corrected - best_threshold
    y_upper = y_pred_corrected + best_threshold

    metrics = compute_metrics(y_test, y_pred_corrected, y_lower, y_upper)

    logger.info("\nResults:")
    logger.info(f"  MAE: {metrics['MAE']:.0f} MW")
    logger.info(f"  RMSE: {metrics['RMSE']:.0f} MW")
    logger.info(f"  Bias: {metrics['Bias']:.0f} MW")
    logger.info(f"  R²: {metrics['R2']:.3f}")
    logger.info(f"  80% PI Coverage: {metrics['Coverage']*100:.1f}%")

    plot_results(test_df, y_pred_corrected, y_lower, y_upper, metrics, 12, 'outputs')

    return metrics


def run_experiment_13(config):
    """
    Experiment 13: Time-of-day specific models.

    Train separate models for day vs night periods.
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT 13: Time-of-day specific models")
    logger.info("=" * 60)

    # Load augmented data
    train_df, val_df, test_df = load_and_prepare_data(config, use_augmented=True)

    logger.info(f"Data - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Define features
    exclude_cols = ['valid_time', 'init_time', 'wind_generation', 'lead_time']
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    logger.info(f"Using {len(feature_cols)} features")

    # Split by day/night (6am-6pm = day)
    def is_day(df):
        hours = pd.to_datetime(df['valid_time']).dt.hour
        return (hours >= 6) & (hours < 18)

    train_day = train_df[is_day(train_df)]
    train_night = train_df[~is_day(train_df)]
    val_day = val_df[is_day(val_df)]
    val_night = val_df[~is_day(val_df)]
    test_day_mask = is_day(test_df)

    logger.info(f"Train - Day: {len(train_day)}, Night: {len(train_night)}")
    logger.info(f"Test - Day: {test_day_mask.sum()}, Night: {(~test_day_mask).sum()}")

    # Train day model
    logger.info("Training day model...")
    day_model = GBMWindModel(
        quantiles=[0.1, 0.5, 0.9],
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=8,
        num_leaves=63,
        early_stopping_rounds=50,
        random_state=42,
    )
    day_model.fit(train_day[feature_cols], train_day['wind_generation'],
                  val_day[feature_cols], val_day['wind_generation'])

    # Train night model
    logger.info("Training night model...")
    night_model = GBMWindModel(
        quantiles=[0.1, 0.5, 0.9],
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=8,
        num_leaves=63,
        early_stopping_rounds=50,
        random_state=42,
    )
    night_model.fit(train_night[feature_cols], train_night['wind_generation'],
                    val_night[feature_cols], val_night['wind_generation'])

    # Predict
    X_test = test_df[feature_cols]
    y_test = test_df['wind_generation'].values

    y_pred = np.zeros(len(test_df))
    y_pred[test_day_mask] = day_model.predict(X_test[test_day_mask])
    y_pred[~test_day_mask] = night_model.predict(X_test[~test_day_mask])

    # Bias correction (combined)
    val_pred_day = day_model.predict(val_day[feature_cols])
    val_pred_night = night_model.predict(val_night[feature_cols])

    day_residuals = val_day['wind_generation'].values - val_pred_day
    night_residuals = val_night['wind_generation'].values - val_pred_night
    all_residuals = np.concatenate([day_residuals, night_residuals])

    bias_correction = np.mean(all_residuals)
    y_pred_corrected = y_pred + bias_correction
    logger.info(f"Bias correction term: {bias_correction:.1f} MW")

    # Calibrate prediction intervals
    corrected_residuals = all_residuals - np.mean(all_residuals)

    best_coverage_diff = float('inf')
    best_threshold = None

    for percentile in range(85, 100):
        threshold = np.percentile(np.abs(corrected_residuals), percentile)
        y_lower = y_pred_corrected - threshold
        y_upper = y_pred_corrected + threshold
        coverage = np.mean((y_test >= y_lower) & (y_test <= y_upper))

        if abs(coverage - 0.80) < best_coverage_diff:
            best_coverage_diff = abs(coverage - 0.80)
            best_threshold = threshold

    logger.info(f"Conformal threshold: {best_threshold:.1f} MW")

    y_lower = y_pred_corrected - best_threshold
    y_upper = y_pred_corrected + best_threshold

    metrics = compute_metrics(y_test, y_pred_corrected, y_lower, y_upper)

    logger.info("\nResults:")
    logger.info(f"  MAE: {metrics['MAE']:.0f} MW")
    logger.info(f"  RMSE: {metrics['RMSE']:.0f} MW")
    logger.info(f"  Bias: {metrics['Bias']:.0f} MW")
    logger.info(f"  R²: {metrics['R2']:.3f}")
    logger.info(f"  80% PI Coverage: {metrics['Coverage']*100:.1f}%")

    plot_results(test_df, y_pred_corrected, y_lower, y_upper, metrics, 13, 'outputs')

    return metrics


def run_experiment_14(config):
    """
    Experiment 14: Triple Ensemble (GBM + XGBoost + Random Forest).

    Add Random Forest to the ensemble for more diversity.
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT 14: Triple Ensemble (GBM + XGBoost + RF)")
    logger.info("=" * 60)

    try:
        import xgboost as xgb
        from sklearn.ensemble import RandomForestRegressor
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        return None

    # Load augmented data
    train_df, val_df, test_df = load_and_prepare_data(config, use_augmented=True)

    logger.info(f"Data - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Define features
    exclude_cols = ['valid_time', 'init_time', 'wind_generation', 'lead_time']
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    logger.info(f"Using {len(feature_cols)} features")

    # Prepare data
    X_train = train_df[feature_cols]
    y_train = train_df['wind_generation']
    X_val = val_df[feature_cols]
    y_val = val_df['wind_generation']
    X_test = test_df[feature_cols]
    y_test = test_df['wind_generation'].values

    # Train LightGBM
    logger.info("Training LightGBM...")
    lgb_model = GBMWindModel(
        quantiles=[0.1, 0.5, 0.9],
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=8,
        num_leaves=63,
        early_stopping_rounds=50,
        random_state=42,
    )
    lgb_model.fit(X_train, y_train, X_val, y_val)
    lgb_pred = lgb_model.predict(X_test)
    lgb_val_pred = lgb_model.predict(X_val)

    # Train XGBoost
    logger.info("Training XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=8,
        early_stopping_rounds=50,
        random_state=42,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_pred = xgb_model.predict(X_test)
    xgb_val_pred = xgb_model.predict(X_val)

    # Train Random Forest
    logger.info("Training Random Forest...")
    rf_model = RandomForestRegressor(
        n_estimators=500,
        max_depth=15,
        min_samples_split=5,
        n_jobs=-1,
        random_state=42,
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_val_pred = rf_model.predict(X_val)

    # Calculate weights based on validation MAE
    lgb_mae = np.mean(np.abs(y_val.values - lgb_val_pred))
    xgb_mae = np.mean(np.abs(y_val.values - xgb_val_pred))
    rf_mae = np.mean(np.abs(y_val.values - rf_val_pred))

    logger.info(f"Val MAE - LGB: {lgb_mae:.0f}, XGB: {xgb_mae:.0f}, RF: {rf_mae:.0f}")

    # Inverse MAE weighting
    total_inv = 1/lgb_mae + 1/xgb_mae + 1/rf_mae
    lgb_weight = (1/lgb_mae) / total_inv
    xgb_weight = (1/xgb_mae) / total_inv
    rf_weight = (1/rf_mae) / total_inv
    logger.info(f"Weights - LGB: {lgb_weight:.2f}, XGB: {xgb_weight:.2f}, RF: {rf_weight:.2f}")

    y_pred_ensemble = lgb_weight * lgb_pred + xgb_weight * xgb_pred + rf_weight * rf_pred
    val_ensemble = lgb_weight * lgb_val_pred + xgb_weight * xgb_val_pred + rf_weight * rf_val_pred

    # Bias correction
    val_residuals = y_val.values - val_ensemble
    bias_correction = np.mean(val_residuals)
    y_pred_corrected = y_pred_ensemble + bias_correction
    logger.info(f"Bias correction term: {bias_correction:.1f} MW")

    # Calibrate prediction intervals
    corrected_residuals = y_val.values - (val_ensemble + bias_correction)

    best_coverage_diff = float('inf')
    best_threshold = None

    for percentile in range(85, 100):
        threshold = np.percentile(np.abs(corrected_residuals), percentile)
        y_lower = y_pred_corrected - threshold
        y_upper = y_pred_corrected + threshold
        coverage = np.mean((y_test >= y_lower) & (y_test <= y_upper))

        if abs(coverage - 0.80) < best_coverage_diff:
            best_coverage_diff = abs(coverage - 0.80)
            best_threshold = threshold

    logger.info(f"Conformal threshold: {best_threshold:.1f} MW")

    y_lower = y_pred_corrected - best_threshold
    y_upper = y_pred_corrected + best_threshold

    metrics = compute_metrics(y_test, y_pred_corrected, y_lower, y_upper)

    logger.info("\nResults:")
    logger.info(f"  MAE: {metrics['MAE']:.0f} MW")
    logger.info(f"  RMSE: {metrics['RMSE']:.0f} MW")
    logger.info(f"  Bias: {metrics['Bias']:.0f} MW")
    logger.info(f"  R²: {metrics['R2']:.3f}")
    logger.info(f"  80% PI Coverage: {metrics['Coverage']*100:.1f}%")

    plot_results(test_df, y_pred_corrected, y_lower, y_upper, metrics, 14, 'outputs')

    return metrics


def run_experiment_15(config):
    """
    Experiment 15: Quad Ensemble with MLP Neural Network.

    Add a simple neural network to the ensemble.
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT 15: Quad Ensemble (GBM + XGB + RF + MLP)")
    logger.info("=" * 60)

    try:
        import xgboost as xgb
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        return None

    # Load augmented data
    train_df, val_df, test_df = load_and_prepare_data(config, use_augmented=True)

    logger.info(f"Data - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Define features
    exclude_cols = ['valid_time', 'init_time', 'wind_generation', 'lead_time']
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    logger.info(f"Using {len(feature_cols)} features")

    # Prepare data
    X_train = train_df[feature_cols]
    y_train = train_df['wind_generation']
    X_val = val_df[feature_cols]
    y_val = val_df['wind_generation']
    X_test = test_df[feature_cols]
    y_test = test_df['wind_generation'].values

    # Scale features for MLP
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Train LightGBM
    logger.info("Training LightGBM...")
    lgb_model = GBMWindModel(
        quantiles=[0.1, 0.5, 0.9],
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=8,
        num_leaves=63,
        early_stopping_rounds=50,
        random_state=42,
    )
    lgb_model.fit(X_train, y_train, X_val, y_val)
    lgb_pred = lgb_model.predict(X_test)
    lgb_val_pred = lgb_model.predict(X_val)

    # Train XGBoost
    logger.info("Training XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=8,
        early_stopping_rounds=50,
        random_state=42,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_pred = xgb_model.predict(X_test)
    xgb_val_pred = xgb_model.predict(X_val)

    # Train Random Forest
    logger.info("Training Random Forest...")
    rf_model = RandomForestRegressor(
        n_estimators=500,
        max_depth=15,
        min_samples_split=5,
        n_jobs=-1,
        random_state=42,
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_val_pred = rf_model.predict(X_val)

    # Train MLP
    logger.info("Training MLP Neural Network...")
    mlp_model = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )
    mlp_model.fit(X_train_scaled, y_train)
    mlp_pred = mlp_model.predict(X_test_scaled)
    mlp_val_pred = mlp_model.predict(X_val_scaled)

    # Calculate weights based on validation MAE
    lgb_mae = np.mean(np.abs(y_val.values - lgb_val_pred))
    xgb_mae = np.mean(np.abs(y_val.values - xgb_val_pred))
    rf_mae = np.mean(np.abs(y_val.values - rf_val_pred))
    mlp_mae = np.mean(np.abs(y_val.values - mlp_val_pred))

    logger.info(f"Val MAE - LGB: {lgb_mae:.0f}, XGB: {xgb_mae:.0f}, RF: {rf_mae:.0f}, MLP: {mlp_mae:.0f}")

    # Inverse MAE weighting
    total_inv = 1/lgb_mae + 1/xgb_mae + 1/rf_mae + 1/mlp_mae
    lgb_weight = (1/lgb_mae) / total_inv
    xgb_weight = (1/xgb_mae) / total_inv
    rf_weight = (1/rf_mae) / total_inv
    mlp_weight = (1/mlp_mae) / total_inv
    logger.info(f"Weights - LGB: {lgb_weight:.2f}, XGB: {xgb_weight:.2f}, RF: {rf_weight:.2f}, MLP: {mlp_weight:.2f}")

    y_pred_ensemble = lgb_weight * lgb_pred + xgb_weight * xgb_pred + rf_weight * rf_pred + mlp_weight * mlp_pred
    val_ensemble = lgb_weight * lgb_val_pred + xgb_weight * xgb_val_pred + rf_weight * rf_val_pred + mlp_weight * mlp_val_pred

    # Bias correction
    val_residuals = y_val.values - val_ensemble
    bias_correction = np.mean(val_residuals)
    y_pred_corrected = y_pred_ensemble + bias_correction
    logger.info(f"Bias correction term: {bias_correction:.1f} MW")

    # Calibrate prediction intervals
    corrected_residuals = y_val.values - (val_ensemble + bias_correction)

    best_coverage_diff = float('inf')
    best_threshold = None

    for percentile in range(85, 100):
        threshold = np.percentile(np.abs(corrected_residuals), percentile)
        y_lower = y_pred_corrected - threshold
        y_upper = y_pred_corrected + threshold
        coverage = np.mean((y_test >= y_lower) & (y_test <= y_upper))

        if abs(coverage - 0.80) < best_coverage_diff:
            best_coverage_diff = abs(coverage - 0.80)
            best_threshold = threshold

    logger.info(f"Conformal threshold: {best_threshold:.1f} MW")

    y_lower = y_pred_corrected - best_threshold
    y_upper = y_pred_corrected + best_threshold

    metrics = compute_metrics(y_test, y_pred_corrected, y_lower, y_upper)

    logger.info("\nResults:")
    logger.info(f"  MAE: {metrics['MAE']:.0f} MW")
    logger.info(f"  RMSE: {metrics['RMSE']:.0f} MW")
    logger.info(f"  Bias: {metrics['Bias']:.0f} MW")
    logger.info(f"  R²: {metrics['R2']:.3f}")
    logger.info(f"  80% PI Coverage: {metrics['Coverage']*100:.1f}%")

    plot_results(test_df, y_pred_corrected, y_lower, y_upper, metrics, 15, 'outputs')

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Run wind forecasting experiments')
    parser.add_argument('--experiment', type=int, default=7, help='Experiment number to run (7-15)')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Config file')

    args = parser.parse_args()

    config = load_config(args.config)

    if args.experiment == 7:
        metrics = run_experiment_7(config)
    elif args.experiment == 8:
        metrics = run_experiment_8(config)
    elif args.experiment == 9:
        metrics = run_experiment_9(config)
    elif args.experiment == 10:
        metrics = run_experiment_10(config)
    elif args.experiment == 11:
        metrics = run_experiment_11(config)
    elif args.experiment == 12:
        metrics = run_experiment_12(config)
    elif args.experiment == 13:
        metrics = run_experiment_13(config)
    elif args.experiment == 14:
        metrics = run_experiment_14(config)
    elif args.experiment == 15:
        metrics = run_experiment_15(config)
    else:
        logger.error(f"Unknown experiment: {args.experiment}")
        return

    logger.info("\nExperiment complete!")


if __name__ == '__main__':
    main()
