#!/usr/bin/env python
"""
Build Features Script

Processes HRRR data and generates training features.
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from loguru import logger
import numpy as np
import pandas as pd
import xarray as xr

from features.wind_features import WindFeatureEngineer
from features.ramp_features import RampFeatureEngineer
from features.temporal_features import TemporalFeatureEngineer
from data.hrrr_client import compute_wind_speed
from utils.config import load_config


def process_hrrr_file(
    file_path: Path,
    wind_engineer: WindFeatureEngineer,
    ramp_engineer: RampFeatureEngineer,
    temporal_engineer: TemporalFeatureEngineer,
) -> pd.DataFrame:
    """
    Process a single HRRR file into features.

    Args:
        file_path: Path to zarr or netCDF file
        wind_engineer: Wind feature engineer
        ramp_engineer: Ramp feature engineer
        temporal_engineer: Temporal feature engineer

    Returns:
        DataFrame with computed features
    """
    logger.info(f"Processing {file_path.name}...")

    # Load data - handle both zarr and netCDF
    if file_path.suffix == '.zarr' or file_path.is_dir():
        ds = xr.open_zarr(file_path)
        data = ds['hrrr']
    else:
        data = xr.open_dataarray(file_path)

    all_features = []

    # Process each lead time
    for lead_idx, lead_time in enumerate(data.coords['lead_time'].values):
        lead_hours = lead_time / np.timedelta64(1, 'h')

        # Extract variables at this lead time
        lead_data = data.isel(lead_time=lead_idx)

        # Get wind components (assuming variable dimension exists)
        u80m = lead_data.sel(variable='u80m').values
        v80m = lead_data.sel(variable='v80m').values
        u10m = lead_data.sel(variable='u10m').values
        v10m = lead_data.sel(variable='v10m').values

        # Compute wind speeds
        ws_80m = compute_wind_speed(u80m, v80m)
        ws_10m = compute_wind_speed(u10m, v10m)

        # Regional aggregation (mean over spatial dims)
        features = {
            'lead_time': lead_hours,
            'ws_80m_mean': np.nanmean(ws_80m),
            'ws_80m_std': np.nanstd(ws_80m),
            'ws_80m_max': np.nanmax(ws_80m),
            'ws_80m_min': np.nanmin(ws_80m),
            'ws_10m_mean': np.nanmean(ws_10m),
        }

        # Power curve features
        normalized_power = wind_engineer.apply_power_curve(ws_80m)
        features['normalized_power_mean'] = np.nanmean(normalized_power)
        features['normalized_power_std'] = np.nanstd(normalized_power)

        # Wind shear
        shear = wind_engineer.compute_wind_shear(ws_10m, ws_80m)
        features['wind_shear_mean'] = np.nanmean(shear)

        all_features.append(features)

    return pd.DataFrame(all_features)


def main():
    parser = argparse.ArgumentParser(description='Build training features')
    parser.add_argument(
        '--input-dir',
        type=str,
        default='data/hrrr',
        help='Directory with HRRR data files',
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='data/features.parquet',
        help='Output features file',
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Configuration file',
    )
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date (YYYY-MM-DD)',
    )
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date (YYYY-MM-DD)',
    )
    parser.add_argument(
        '--ercot-file',
        type=str,
        default='data/ercot_wind.csv',
        help='ERCOT wind generation data file',
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Initialize feature engineers
    wind_engineer = WindFeatureEngineer(
        cut_in_speed=config.features.cut_in_speed,
        rated_speed=config.features.rated_speed,
        cut_out_speed=config.features.cut_out_speed,
    )
    ramp_engineer = RampFeatureEngineer()
    temporal_engineer = TemporalFeatureEngineer()

    # Find input files (zarr directories or nc files)
    input_dir = Path(args.input_dir)
    zarr_files = sorted(input_dir.glob('*.zarr'))
    nc_files = sorted(input_dir.glob('*.nc'))
    files = zarr_files if zarr_files else nc_files

    if not files:
        logger.error(f"No HRRR data files found in {input_dir}")
        return

    logger.info(f"Found {len(files)} files to process")

    # Process each file
    all_dfs = []
    for file_path in files:
        try:
            df = process_hrrr_file(
                file_path,
                wind_engineer,
                ramp_engineer,
                temporal_engineer,
            )

            # Extract timestamp from filename (hrrr_YYYYMMDD_HH.zarr or .nc)
            parts = file_path.stem.split('_')
            if len(parts) >= 3:
                date_str = parts[1]
                hour = int(parts[2])
                init_time = datetime.strptime(date_str, '%Y%m%d').replace(hour=hour)
                df['init_time'] = init_time
                df['valid_time'] = df.apply(
                    lambda row: init_time + pd.Timedelta(hours=row['lead_time']),
                    axis=1
                )

            all_dfs.append(df)

        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_dfs:
        logger.error("No files processed successfully")
        return

    # Combine all features
    features_df = pd.concat(all_dfs, ignore_index=True)

    # Add temporal features
    if 'valid_time' in features_df.columns:
        timestamps = pd.DatetimeIndex(features_df['valid_time'])
        temporal_features = temporal_engineer.compute_features(timestamps)
        features_df = pd.concat([features_df, temporal_features.reset_index(drop=True)], axis=1)

    # Merge with ERCOT wind generation data
    ercot_path = Path(args.ercot_file)
    if ercot_path.exists():
        logger.info(f"Loading ERCOT data from {ercot_path}")
        ercot_df = pd.read_csv(ercot_path, parse_dates=['timestamp'])
        ercot_df = ercot_df.rename(columns={'timestamp': 'valid_time'})

        # Merge on valid_time
        features_df = features_df.merge(
            ercot_df,
            on='valid_time',
            how='left',
        )
        logger.info(f"Merged with ERCOT data: {features_df['wind_generation'].notna().sum()} matched rows")
    else:
        logger.warning(f"ERCOT data file not found: {ercot_path}")

    # Save
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(output_path, index=False)

    logger.info(f"Saved features to {output_path}")
    logger.info(f"Shape: {features_df.shape}")
    logger.info(f"Columns: {list(features_df.columns)}")


if __name__ == '__main__':
    main()
