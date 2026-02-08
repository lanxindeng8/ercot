#!/usr/bin/env python3
"""
DAM Model Training Script

Trains XGBoost, LightGBM, and CatBoost models for DAM price prediction.
Can fetch data from InfluxDB or use a local CSV file.

Usage:
    # Using InfluxDB (requires env vars: INFLUXDB_URL, INFLUXDB_TOKEN, INFLUXDB_ORG, INFLUXDB_BUCKET)
    python train_dam_models.py --source influxdb --settlement-point LZ_HOUSTON

    # Using CSV file
    python train_dam_models.py --source csv --input /path/to/dam_prices.csv

    # Generate features only (for debugging)
    python train_dam_models.py --source influxdb --features-only
"""

import argparse
import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_data_from_influxdb(settlement_point: str = "LZ_HOUSTON") -> pd.DataFrame:
    """Load DAM prices from InfluxDB"""
    from data.influxdb_fetcher import create_fetcher_from_env

    print("Connecting to InfluxDB...")
    fetcher = create_fetcher_from_env()

    print(f"Fetching DAM prices for {settlement_point}...")
    df = fetcher.fetch_dam_prices(settlement_point=settlement_point)
    fetcher.close()

    if df.empty:
        raise ValueError(f"No data found for {settlement_point}")

    print(f"Loaded {len(df)} records from InfluxDB")
    return df


def load_data_from_csv(filepath: str) -> pd.DataFrame:
    """Load DAM prices from CSV file"""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)

    # Handle different CSV formats
    if 'date' in df.columns and 'hour' in df.columns:
        # Format: date, hour, dam_price
        df['date'] = pd.to_datetime(df['date'])
        df['timestamp'] = df['date'] + pd.to_timedelta((df['hour'] - 1), unit='h')
        df = df.set_index('timestamp')
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        df['hour'] = df.index.hour + 1

    # Rename price column if needed
    if 'SettlementPointPrice' in df.columns:
        df = df.rename(columns={'SettlementPointPrice': 'dam_price'})
    if 'lmp' in df.columns:
        df = df.rename(columns={'lmp': 'dam_price'})

    print(f"Loaded {len(df)} records from CSV")
    return df


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract DAM prediction features"""
    from features.dam_features import DAMFeatureEngineer

    print("Extracting features...")
    engineer = DAMFeatureEngineer()
    features_df = engineer.extract_features(df, verbose=True)
    return features_df


def prepare_train_test(
    features_df: pd.DataFrame,
    train_end: str = "2024-12-31"
) -> tuple:
    """Split data into train and test sets"""
    print(f"Splitting data (train end: {train_end})...")

    # Feature columns (exclude target and timestamp)
    feature_cols = [c for c in features_df.columns if c not in ['target', 'timestamp']]

    train_mask = features_df['timestamp'] <= train_end
    train_df = features_df[train_mask]
    test_df = features_df[~train_mask]

    X_train = train_df[feature_cols]
    y_train = train_df['target']
    X_test = test_df[feature_cols]
    y_test = test_df['target']

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    return X_train, X_test, y_train, y_test, feature_cols


def train_xgboost(X_train, y_train):
    """Train XGBoost model"""
    print("\nTraining XGBoost...")
    model = XGBRegressor(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=10,
        objective='reg:absoluteerror',
        tree_method='hist',
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    return model


def train_lightgbm(X_train, y_train):
    """Train LightGBM model"""
    print("\nTraining LightGBM...")
    model = LGBMRegressor(
        n_estimators=1000,
        num_leaves=31,
        max_depth=8,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_samples=20,
        objective='mae',
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_catboost(X_train, y_train, cat_features=None):
    """Train CatBoost model"""
    print("\nTraining CatBoost...")
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
        verbose=100,
        allow_writing_files=False,
    )
    model.fit(X_train, y_train, cat_features=cat_features)
    return model


def evaluate_model(model, X_test, y_test, name: str) -> dict:
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"{name} Results:")
    print(f"  MAE: ${mae:.2f}")
    print(f"  RMSE: ${rmse:.2f}")

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4)
    }


def main():
    parser = argparse.ArgumentParser(description="Train DAM price prediction models")
    parser.add_argument("--source", choices=["influxdb", "csv"], default="csv",
                        help="Data source type")
    parser.add_argument("--input", type=str, help="Input CSV file path")
    parser.add_argument("--settlement-point", default="LZ_HOUSTON",
                        help="ERCOT settlement point")
    parser.add_argument("--output", default="../models/dam",
                        help="Output directory for models")
    parser.add_argument("--train-end", default="2024-12-31",
                        help="End date for training data")
    parser.add_argument("--features-only", action="store_true",
                        help="Only extract features, don't train models")
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).parent / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    if args.source == "influxdb":
        df = load_data_from_influxdb(args.settlement_point)
    else:
        if not args.input:
            print("Error: --input required for CSV source")
            sys.exit(1)
        df = load_data_from_csv(args.input)

    # Extract features
    features_df = extract_features(df)

    if args.features_only:
        output_path = output_dir / "dam_features.csv"
        features_df.to_csv(output_path, index=False)
        print(f"Features saved to {output_path}")
        return

    # Prepare train/test split
    X_train, X_test, y_train, y_test, feature_cols = prepare_train_test(
        features_df, args.train_end
    )

    # Get categorical feature indices
    from features.dam_features import CATEGORICAL_INDICES
    cat_indices = [i for i, c in enumerate(feature_cols) if i in CATEGORICAL_INDICES]

    results = {}

    # Train XGBoost
    xgb_model = train_xgboost(X_train, y_train)
    results['xgboost'] = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    joblib.dump(xgb_model, output_dir / "xgboost_dam.joblib")
    print(f"Saved XGBoost model to {output_dir / 'xgboost_dam.joblib'}")

    # Train LightGBM
    lgb_model = train_lightgbm(X_train, y_train)
    results['lightgbm'] = evaluate_model(lgb_model, X_test, y_test, "LightGBM")
    joblib.dump(lgb_model, output_dir / "lightgbm_dam.joblib")
    print(f"Saved LightGBM model to {output_dir / 'lightgbm_dam.joblib'}")

    # Train CatBoost
    cat_model = train_catboost(X_train, y_train, cat_features=cat_indices)
    results['catboost'] = evaluate_model(cat_model, X_test, y_test, "CatBoost")
    cat_model.save_model(str(output_dir / "catboost_dam.cbm"))
    print(f"Saved CatBoost model to {output_dir / 'catboost_dam.cbm'}")

    # Ensemble evaluation
    xgb_pred = xgb_model.predict(X_test)
    lgb_pred = lgb_model.predict(X_test)
    cat_pred = cat_model.predict(X_test)
    ensemble_pred = (xgb_pred + lgb_pred + cat_pred) / 3

    ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
    ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
    print(f"\nEnsemble Results:")
    print(f"  MAE: ${ensemble_mae:.2f}")
    print(f"  RMSE: ${ensemble_rmse:.2f}")
    results['ensemble'] = {
        "mae": round(ensemble_mae, 4),
        "rmse": round(ensemble_rmse, 4)
    }

    # Save metadata
    metadata = {
        "trained_at": datetime.utcnow().isoformat(),
        "settlement_point": args.settlement_point,
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "feature_columns": feature_cols,
        "categorical_indices": cat_indices,
        "train_end_date": args.train_end,
        "results": results
    }

    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Models saved to: {output_dir}")
    print(f"XGBoost MAE: ${results['xgboost']['mae']:.2f}")
    print(f"LightGBM MAE: ${results['lightgbm']['mae']:.2f}")
    print(f"CatBoost MAE: ${results['catboost']['mae']:.2f}")
    print(f"Ensemble MAE: ${results['ensemble']['mae']:.2f}")


if __name__ == "__main__":
    main()
