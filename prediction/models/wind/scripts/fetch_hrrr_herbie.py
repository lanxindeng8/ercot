#!/usr/bin/env python3
"""
Fetch HRRR data using Herbie library.

Alternative to earth2studio-based fetch_hrrr_data.py.
Downloads wind variables for Texas region.
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import xarray as xr
from herbie import Herbie
from loguru import logger

# Texas bounding box (lon in 0-360 format for HRRR)
TEXAS_BOUNDS = {
    'lat_min': 25.8,
    'lat_max': 36.5,
    'lon_min': 253.4,   # -106.6 + 360
    'lon_max': 266.5,   # -93.5 + 360
}

# Variables to fetch (search patterns for herbie)
WIND_VARS = {
    'u80m': ':UGRD:80 m above ground',
    'v80m': ':VGRD:80 m above ground',
    'u10m': ':UGRD:10 m above ground',
    'v10m': ':VGRD:10 m above ground',
    't2m': ':TMP:2 m above ground',
    'sp': ':PRES:surface',
}


def fetch_hrrr_forecast(
    init_time: datetime,
    lead_times: list,
    output_dir: Path,
) -> Path:
    """
    Fetch HRRR forecast for given init time and lead times.

    Args:
        init_time: Initialization time (UTC)
        lead_times: List of forecast hours (0-12)
        output_dir: Output directory

    Returns:
        Path to saved zarr file
    """
    all_data = []

    for fxx in lead_times:
        logger.info(f"  Fetching F{fxx:02d}...")

        try:
            H = Herbie(
                init_time.strftime("%Y-%m-%d %H:%M"),
                model="hrrr",
                product="sfc",
                fxx=fxx,
            )

            # Fetch each variable
            var_arrays = {}
            for var_name, search_pattern in WIND_VARS.items():
                try:
                    ds = H.xarray(search_pattern, remove_grib=True)
                    # Get the data variable (first one that's not a coord)
                    data_var = [v for v in ds.data_vars if v not in ['latitude', 'longitude']][0]
                    var_arrays[var_name] = ds[data_var]
                except Exception as e:
                    logger.warning(f"    Failed to get {var_name}: {e}")
                    continue

            if not var_arrays:
                logger.warning(f"  No data retrieved for F{fxx:02d}")
                continue

            # Combine variables into single dataset
            # Use first variable to get coordinates
            first_var = list(var_arrays.values())[0]
            lat = first_var.latitude.values
            lon = first_var.longitude.values

            # Subset to Texas
            lat_mask = (lat >= TEXAS_BOUNDS['lat_min']) & (lat <= TEXAS_BOUNDS['lat_max'])
            lon_mask = (lon >= TEXAS_BOUNDS['lon_min']) & (lon <= TEXAS_BOUNDS['lon_max'])

            # Find bounding indices
            y_indices = np.where(lat_mask.any(axis=1))[0]
            x_indices = np.where(lon_mask.any(axis=0))[0]

            if len(y_indices) == 0 or len(x_indices) == 0:
                logger.warning("  No Texas data in grid, using full extent")
                y_slice = slice(None)
                x_slice = slice(None)
            else:
                y_slice = slice(y_indices[0], y_indices[-1] + 1)
                x_slice = slice(x_indices[0], x_indices[-1] + 1)

            # Extract Texas subset for each variable
            lead_data = {}
            for var_name, da in var_arrays.items():
                texas_data = da.isel(y=y_slice, x=x_slice).values
                lead_data[var_name] = texas_data

            # Store coordinates from first variable
            texas_lat = lat[y_slice, x_slice]
            texas_lon = lon[y_slice, x_slice]

            all_data.append({
                'lead_time': fxx,
                'data': lead_data,
                'lat': texas_lat,
                'lon': texas_lon,
            })

        except Exception as e:
            logger.error(f"  Failed F{fxx:02d}: {e}")
            continue

    if not all_data:
        logger.error("No data retrieved!")
        return None

    # Build xarray dataset
    lead_times_arr = np.array([d['lead_time'] for d in all_data])
    variables = list(all_data[0]['data'].keys())

    # Stack all lead times
    data_arrays = {}
    for var in variables:
        var_data = np.stack([d['data'][var] for d in all_data], axis=0)
        data_arrays[var] = (['lead_time', 'y', 'x'], var_data)

    ds = xr.Dataset(
        data_arrays,
        coords={
            'lead_time': ('lead_time', [timedelta(hours=int(h)) for h in lead_times_arr]),
            'lat': (['y', 'x'], all_data[0]['lat']),
            'lon': (['y', 'x'], all_data[0]['lon']),
            'init_time': init_time,
        },
    )

    # Combine into single DataArray for compatibility with existing code
    var_list = list(variables)
    combined = np.stack([ds[v].values for v in var_list], axis=1)

    da = xr.DataArray(
        combined,
        dims=['lead_time', 'variable', 'y', 'x'],
        coords={
            'lead_time': ds.lead_time,
            'variable': var_list,
            'lat': ds.lat,
            'lon': ds.lon,
        },
        attrs={'init_time': str(init_time)},
    )

    # Save
    output_file = output_dir / f"hrrr_{init_time.strftime('%Y%m%d')}_{init_time.hour:02d}.zarr"
    da.to_dataset(name='hrrr').to_zarr(output_file, mode='w')

    logger.info(f"  Saved to {output_file}")
    logger.info(f"  Shape: {da.shape}")

    return output_file


def main():
    parser = argparse.ArgumentParser(description='Fetch HRRR data using Herbie')
    parser.add_argument('--date', type=str, required=True, help='Date (YYYY-MM-DD)')
    parser.add_argument('--hours', type=int, nargs='+', default=[0, 6, 12, 18],
                        help='Initialization hours (UTC)')
    parser.add_argument('--lead-times', type=int, nargs='+', default=list(range(0, 13)),
                        help='Lead times in hours')
    parser.add_argument('--output-dir', type=str, default='data/hrrr',
                        help='Output directory')

    args = parser.parse_args()

    date = datetime.strptime(args.date, '%Y-%m-%d')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Fetching HRRR data for {args.date}")
    logger.info(f"Init hours: {args.hours}")
    logger.info(f"Lead times: {args.lead_times}")

    for hour in args.hours:
        init_time = date.replace(hour=hour)
        logger.info(f"\nProcessing init={init_time}...")

        fetch_hrrr_forecast(init_time, args.lead_times, output_dir)

    logger.info("\nDone!")


if __name__ == '__main__':
    main()
