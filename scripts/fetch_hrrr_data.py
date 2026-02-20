#!/usr/bin/env python
"""
Fetch HRRR Data Script

Downloads HRRR forecast data for Texas/ERCOT region.
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from loguru import logger
import numpy as np
import pandas as pd

from data.hrrr_client import HRRRWindClient, WIND_VARIABLES


def main():
    parser = argparse.ArgumentParser(description='Fetch HRRR forecast data')
    parser.add_argument(
        '--date',
        type=str,
        required=True,
        help='Forecast date (YYYY-MM-DD)',
    )
    parser.add_argument(
        '--hours',
        type=int,
        nargs='+',
        default=list(range(0, 24, 6)),
        help='Initialization hours (UTC)',
    )
    parser.add_argument(
        '--lead-times',
        type=int,
        nargs='+',
        default=list(range(0, 13)),
        help='Lead times in hours',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/hrrr',
        help='Output directory',
    )
    parser.add_argument(
        '--source',
        type=str,
        default='aws',
        choices=['aws', 'google', 'nomads'],
        help='Data source',
    )

    args = parser.parse_args()

    # Parse date
    date = datetime.strptime(args.date, '%Y-%m-%d')

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize client
    client = HRRRWindClient(source=args.source, cache=True)

    logger.info(f"Fetching HRRR data for {args.date}")
    logger.info(f"Init hours: {args.hours}")
    logger.info(f"Lead times: {args.lead_times}")

    for hour in args.hours:
        init_time = date.replace(hour=hour)

        try:
            logger.info(f"Fetching init={init_time}...")

            # Fetch Texas subset
            data = client.fetch_forecast_texas(
                init_time=init_time,
                lead_times=args.lead_times,
            )

            # Save to zarr (more robust than netCDF for complex data)
            output_file = output_dir / f"hrrr_{args.date.replace('-', '')}_{hour:02d}.zarr"
            data.to_dataset(name='hrrr').to_zarr(output_file, mode='w')

            logger.info(f"Saved to {output_file}")
            logger.info(f"Shape: {data.shape}")

        except Exception as e:
            logger.error(f"Failed to fetch {init_time}: {e}")
            continue

    logger.info("Done!")


if __name__ == '__main__':
    main()
