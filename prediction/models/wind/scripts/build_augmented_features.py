#!/usr/bin/env python
"""
Build Augmented Features Script

Creates features with properly computed lag features from hourly ERCOT data.
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from loguru import logger
import numpy as np
import pandas as pd


def main():
    # Load ERCOT wind data (hourly)
    logger.info("Loading ERCOT wind data...")
    ercot = pd.read_csv('data/ercot_wind.csv', parse_dates=['timestamp'])
    ercot = ercot.sort_values('timestamp')
    logger.info(f"ERCOT data: {len(ercot)} rows")

    # Add lag features based on hourly data
    logger.info("Adding lag features...")
    lag_hours = [1, 2, 3, 6, 12, 24]
    for lag in lag_hours:
        ercot[f'wind_gen_lag_{lag}h'] = ercot['wind_generation'].shift(lag)

    # Add rolling features
    rolling_windows = [6, 12, 24]
    for window in rolling_windows:
        ercot[f'wind_gen_rolling_{window}h_mean'] = ercot['wind_generation'].rolling(window, min_periods=1).mean().shift(1)

    # Add change features
    logger.info("Adding change features...")
    ercot['wind_gen_change_1h'] = ercot['wind_generation'].shift(1) - ercot['wind_generation'].shift(2)
    ercot['wind_gen_change_3h'] = ercot['wind_generation'].shift(1) - ercot['wind_generation'].shift(4)

    # Rename for merge
    ercot = ercot.rename(columns={'timestamp': 'valid_time'})

    # Load original HRRR features
    logger.info("Loading HRRR features...")
    hrrr = pd.read_parquet('data/features.parquet')
    hrrr['valid_time'] = pd.to_datetime(hrrr['valid_time'])
    logger.info(f"HRRR features: {len(hrrr)} rows")

    # Drop wind_generation from HRRR (we'll use ERCOT's)
    if 'wind_generation' in hrrr.columns:
        hrrr = hrrr.drop(columns=['wind_generation'])

    # Merge HRRR features with augmented ERCOT data
    logger.info("Merging HRRR features with augmented ERCOT data...")
    merged = hrrr.merge(ercot, on='valid_time', how='inner')

    # Drop rows with NaN in lag features
    before_dropna = len(merged)
    merged = merged.dropna()
    logger.info(f"Dropped {before_dropna - len(merged)} rows with NaN")

    # Save augmented features
    output_path = Path('data/features_augmented.parquet')
    merged.to_parquet(output_path, index=False)

    logger.info(f"Saved augmented features to {output_path}")
    logger.info(f"Shape: {merged.shape}")
    logger.info(f"Columns: {list(merged.columns)}")

    # Show date range
    logger.info(f"Date range: {merged['valid_time'].min()} to {merged['valid_time'].max()}")


if __name__ == '__main__':
    main()
