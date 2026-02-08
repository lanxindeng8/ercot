#!/usr/bin/env python3
"""
Train DAM Price Prediction Models (V2)

Trains 24 per-hour CatBoost models using the proven DAM_Price_Forecast approach.
Uses 35 features and trains separate models for each hour for better accuracy.

Usage:
    python scripts/train_dam_v2.py --input /path/to/dam_data.csv --output-dir models/dam_v2

Expected MAE: ~$8.50 (vs naive ~$10.58)
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def main():
    parser = argparse.ArgumentParser(description="Train DAM V2 models")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to DAM price CSV (date,hour,dam_price)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/dam_v2",
        help="Output directory for trained models",
    )
    parser.add_argument(
        "--train-start",
        type=str,
        default="2022-01-01",
        help="Training start date",
    )
    parser.add_argument(
        "--test-start",
        type=str,
        default="2026-01-01",
        help="Test start date",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DAM Price Prediction Model Training (V2)")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output: {output_dir}")
    print(f"Train period: {args.train_start} to {args.test_start}")
    print(f"Test period: {args.test_start} onwards")
    print()

    # Load data and extract features
    from features.dam_features_v2 import DAMFeatureEngineer, load_dam_data_from_csv

    print("Loading DAM price data...")
    dam_df = load_dam_data_from_csv(args.input)
    print(f"  Records: {len(dam_df):,}")
    print(f"  Date range: {dam_df.index.min()} to {dam_df.index.max()}")
    print()

    print("Extracting features (35 features)...")
    fe = DAMFeatureEngineer()
    features_df = fe.extract_features(dam_df, verbose=True)
    print()

    # Filter by date range
    train_start = pd.Timestamp(args.train_start)
    test_start = pd.Timestamp(args.test_start)

    features_df = features_df[features_df['timestamp'] >= train_start]
    train_df = features_df[features_df['timestamp'] < test_start]
    test_df = features_df[features_df['timestamp'] >= test_start]

    print(f"Train samples: {len(train_df):,}")
    print(f"Test samples: {len(test_df):,}")
    print()

    if len(train_df) == 0:
        print("ERROR: No training data available!")
        return 1

    # Train 24 per-hour models
    feature_names = fe.get_feature_names()
    cat_indices = fe.get_categorical_indices()

    all_test_preds = []
    all_test_actuals = []
    metrics_by_hour = []

    print("Training per-hour models...")
    print("-" * 60)

    for hour in range(1, 25):
        print(f"\nHour {hour:02d}:")

        # Filter data for this hour
        hour_train = train_df[train_df['hour'] == hour]
        hour_test = test_df[test_df['hour'] == hour]

        if len(hour_train) < 50:
            print(f"  WARNING: Only {len(hour_train)} training samples, skipping")
            continue

        X_train = hour_train[feature_names]
        y_train = hour_train['target']
        X_test = hour_test[feature_names] if len(hour_test) > 0 else None
        y_test = hour_test['target'] if len(hour_test) > 0 else None

        # Train CatBoost model (same config as DAM_Price_Forecast)
        model = CatBoostRegressor(
            iterations=2000,
            depth=6,
            learning_rate=0.2,
            l2_leaf_reg=3.0,
            random_strength=1.0,
            min_data_in_leaf=20,
            subsample=0.8,
            bootstrap_type='Bernoulli',
            loss_function='MAE',
            random_seed=42,
            verbose=0,
            allow_writing_files=False,
        )

        model.fit(X_train, y_train)

        # Save model
        model_path = output_dir / f"dam_hour_{hour:02d}.cbm"
        model.save_model(str(model_path))

        # Evaluate
        if X_test is not None and len(X_test) > 0:
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            # Naive baseline (yesterday same hour)
            naive_pred = X_test['dam_lag_24h']
            naive_mae = mean_absolute_error(y_test, naive_pred)

            improvement = (naive_mae - mae) / naive_mae * 100

            print(f"  Train: {len(X_train):,}, Test: {len(X_test):,}")
            print(f"  MAE: ${mae:.2f}, Naive: ${naive_mae:.2f}, Improvement: {improvement:.1f}%")

            all_test_preds.extend(y_pred)
            all_test_actuals.extend(y_test)
            metrics_by_hour.append({
                'hour': hour,
                'mae': mae,
                'rmse': rmse,
                'naive_mae': naive_mae,
                'improvement_pct': improvement,
                'n_test': len(X_test),
            })
        else:
            print(f"  Train: {len(X_train):,}, Test: 0 (no test data)")

    print()
    print("=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)

    if all_test_preds:
        overall_mae = mean_absolute_error(all_test_actuals, all_test_preds)
        overall_rmse = np.sqrt(mean_squared_error(all_test_actuals, all_test_preds))
        overall_naive = np.mean([m['naive_mae'] for m in metrics_by_hour])

        print(f"Overall MAE: ${overall_mae:.2f}")
        print(f"Overall RMSE: ${overall_rmse:.2f}")
        print(f"Naive MAE: ${overall_naive:.2f}")
        print(f"Improvement: {(overall_naive - overall_mae) / overall_naive * 100:.1f}%")

    print()
    print(f"Models saved to: {output_dir}")
    print(f"  24 files: dam_hour_01.cbm to dam_hour_24.cbm")

    # Save metrics
    metrics_df = pd.DataFrame(metrics_by_hour)
    metrics_path = output_dir / "training_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"  Metrics: {metrics_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
