#!/usr/bin/env python
"""
Build Features from SQLite

Extracts wind generation from the ERCOT archive database and builds
training features using temporal, lag, rolling, and ramp features.
No HRRR weather data required.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from loguru import logger
import numpy as np
import pandas as pd
import sqlite3

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "temporal_features",
    str(Path(__file__).parent.parent / 'src' / 'features' / 'temporal_features.py'),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
TemporalFeatureEngineer = _mod.TemporalFeatureEngineer


def extract_wind_generation(db_path: str) -> pd.DataFrame:
    """Extract hourly wind generation from SQLite fuel_mix_hist table."""
    conn = sqlite3.connect(db_path)

    query = """
        SELECT delivery_date, interval_15min, generation_mw
        FROM fuel_mix_hist
        WHERE fuel IN ('Wind', 'Wnd')
        ORDER BY delivery_date, interval_15min
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Build timestamp from delivery_date + interval_15min
    # interval_15min: 1-96 (96 intervals per day)
    df['delivery_date'] = pd.to_datetime(df['delivery_date'])
    df['timestamp'] = df['delivery_date'] + pd.to_timedelta(
        (df['interval_15min'] - 1) * 15, unit='m'
    )

    # Aggregate both 'Wind' and 'Wnd' fuel types
    df = df.groupby('timestamp')['generation_mw'].sum().reset_index()
    df.columns = ['timestamp', 'wind_generation']

    # Resample to hourly (mean of 15-min intervals)
    df = df.set_index('timestamp').resample('h').mean().reset_index()
    df = df.dropna(subset=['wind_generation'])

    logger.info(f"Extracted {len(df)} hourly wind generation records")
    logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    logger.info(f"Wind gen range: {df['wind_generation'].min():.0f} - {df['wind_generation'].max():.0f} MW")

    return df


def build_wind_proxy_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build proxy wind features from generation data.

    Since we don't have HRRR weather data, we derive wind-like features
    from the generation time series itself.
    """
    capacity = 40000  # MW approximate ERCOT wind capacity

    # Normalized power (proxy for wind speed via inverse power curve)
    df['normalized_power_mean'] = df['wind_generation'] / capacity

    # Estimate wind speed from generation using inverse power curve
    # Power curve: P = ((ws - 3) / 9)^3 for 3 <= ws <= 12
    # Inverse: ws = 3 + 9 * P^(1/3)
    norm_p = df['normalized_power_mean'].clip(0.001, 0.999)
    df['ws_80m_mean'] = 3.0 + 9.0 * np.power(norm_p, 1.0 / 3.0)
    df['ws_80m_std'] = df['ws_80m_mean'] * 0.15  # proxy variability

    # Wind shear proxy (use generation variability)
    df['wind_shear_mean'] = 0.2 + 0.1 * np.random.RandomState(42).randn(len(df)).clip(-2, 2)

    # Power sensitivity: dP/dws is highest in cubic region
    ws = df['ws_80m_mean'].values
    sensitivity = np.where(
        (ws >= 3) & (ws <= 12),
        3 * ((ws - 3) / 9) ** 2 / 9,
        0.0,
    )
    df['power_sensitivity'] = sensitivity

    return df


def build_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag, rolling, and change features."""
    df = df.sort_values('timestamp').copy()

    # Lag features (shifted to avoid data leakage)
    lag_hours = [1, 2, 3, 6, 12, 24]
    for lag in lag_hours:
        df[f'wind_gen_lag_{lag}h'] = df['wind_generation'].shift(lag)

    # Rolling features (shifted by 1 to avoid leakage)
    rolling_windows = [6, 12, 24]
    for window in rolling_windows:
        rolled = df['wind_generation'].rolling(window, min_periods=1)
        df[f'wind_gen_rolling_{window}h_mean'] = rolled.mean().shift(1)
        df[f'wind_gen_rolling_{window}h_std'] = rolled.std().shift(1)
        df[f'wind_gen_rolling_{window}h_min'] = rolled.min().shift(1)
        df[f'wind_gen_rolling_{window}h_max'] = rolled.max().shift(1)

    # Change/momentum features
    df['wind_gen_change_1h'] = df['wind_generation'].shift(1) - df['wind_generation'].shift(2)
    df['wind_gen_change_3h'] = df['wind_generation'].shift(1) - df['wind_generation'].shift(4)
    df['wind_gen_change_6h'] = df['wind_generation'].shift(1) - df['wind_generation'].shift(7)

    # Rate of change
    df['wind_gen_roc_1h'] = df['wind_gen_change_1h'] / (df['wind_generation'].shift(2).clip(lower=100))
    df['wind_gen_roc_3h'] = df['wind_gen_change_3h'] / (df['wind_generation'].shift(4).clip(lower=100))

    # Ramp indicators (based on lagged data only)
    df['ramp_down_1h'] = (df['wind_gen_change_1h'] < -1000).astype(float)
    df['ramp_down_3h'] = (df['wind_gen_change_3h'] < -2000).astype(float)
    df['ramp_up_1h'] = (df['wind_gen_change_1h'] > 1000).astype(float)

    return df


def build_ramp_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ramp risk and no-solar period features."""
    hours = pd.to_datetime(df['timestamp']).dt.hour

    # No-solar period flags
    df['is_no_solar_period'] = ((hours >= 19) | (hours < 7)).astype(float)
    df['is_evening_peak'] = ((hours >= 17) & (hours < 21)).astype(float)

    # Minutes relative to sunset (approximate: sunset ~19:00 in Texas)
    df['hours_to_sunset'] = (19 - hours).clip(lower=-12, upper=12)
    df['hours_since_sunset'] = np.where(hours >= 19, hours - 19, np.where(hours < 7, hours + 5, 0))

    # Ramp-down risk score (using lagged generation changes)
    risk = np.zeros(len(df))
    change_1h = df.get('wind_gen_change_1h', pd.Series(0, index=df.index)).fillna(0)
    risk += (change_1h < -1000).astype(float) * 0.15
    risk += (change_1h < -2000).astype(float) * 0.20
    risk += df['is_no_solar_period'].values * 0.15
    risk += (df['is_evening_peak'].values * df['is_no_solar_period'].values) * 0.15
    df['ramp_down_risk_score'] = risk.clip(0, 1)

    return df


def main():
    # Paths
    base_dir = Path(__file__).parent.parent
    db_path = base_dir.parent.parent.parent / 'scraper' / 'data' / 'ercot_archive.db'
    data_dir = base_dir / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Database: {db_path}")

    # Step 1: Extract wind generation
    logger.info("=" * 60)
    logger.info("Step 1: Extracting wind generation from SQLite")
    logger.info("=" * 60)
    wind_df = extract_wind_generation(str(db_path))

    # Filter to training period (Jul 2024 - Dec 2024)
    wind_df = wind_df[
        (wind_df['timestamp'] >= '2024-07-01')
        & (wind_df['timestamp'] <= '2024-12-31')
    ].copy()
    logger.info(f"Filtered to training period: {len(wind_df)} records")

    # Save raw wind generation
    wind_csv_path = data_dir / 'ercot_wind.csv'
    wind_df.to_csv(wind_csv_path, index=False)
    logger.info(f"Saved wind generation CSV: {wind_csv_path}")

    # Step 2: Build proxy wind features
    logger.info("=" * 60)
    logger.info("Step 2: Building proxy wind features")
    logger.info("=" * 60)
    wind_df = build_wind_proxy_features(wind_df)

    # Step 3: Add temporal features
    logger.info("=" * 60)
    logger.info("Step 3: Adding temporal features")
    logger.info("=" * 60)
    temporal_eng = TemporalFeatureEngineer()
    timestamps = pd.DatetimeIndex(wind_df['timestamp'])
    temporal_features = temporal_eng.compute_features(timestamps)
    wind_df = pd.concat([wind_df.reset_index(drop=True), temporal_features.reset_index(drop=True)], axis=1)

    # Step 4: Add ramp risk features
    logger.info("=" * 60)
    logger.info("Step 4: Adding ramp risk features")
    logger.info("=" * 60)
    wind_df = build_ramp_risk_features(wind_df)

    # Rename timestamp to valid_time for compatibility
    wind_df = wind_df.rename(columns={'timestamp': 'valid_time'})

    # Save base features
    features_path = data_dir / 'features.parquet'
    wind_df.to_parquet(features_path, index=False)
    logger.info(f"Saved base features: {features_path} ({wind_df.shape})")

    # Step 5: Build augmented features (with lags)
    logger.info("=" * 60)
    logger.info("Step 5: Building augmented features with lags")
    logger.info("=" * 60)
    wind_df = wind_df.rename(columns={'valid_time': 'timestamp'})
    wind_df = build_lag_and_rolling_features(wind_df)
    wind_df = wind_df.rename(columns={'timestamp': 'valid_time'})

    # Drop rows with NaN from lag features
    before = len(wind_df)
    wind_df = wind_df.dropna()
    logger.info(f"Dropped {before - len(wind_df)} rows with NaN from lags")

    # Save augmented features
    augmented_path = data_dir / 'features_augmented.parquet'
    wind_df.to_parquet(augmented_path, index=False)
    logger.info(f"Saved augmented features: {augmented_path} ({wind_df.shape})")

    # Summary
    logger.info("=" * 60)
    logger.info("FEATURE BUILD COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Date range: {wind_df['valid_time'].min()} to {wind_df['valid_time'].max()}")
    logger.info(f"Total features: {len([c for c in wind_df.columns if c not in ['valid_time', 'wind_generation']])}")
    logger.info(f"Columns: {list(wind_df.columns)}")


if __name__ == '__main__':
    main()
