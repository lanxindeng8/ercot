#!/usr/bin/env python
"""
CLI script to fetch Open-Meteo weather data for all ERCOT weather stations
and store it in the SQLite archive.

Usage:
    python prediction/scripts/fetch_weather.py \
        --start-year 2015 --end-year 2026 \
        --db-path scraper/data/ercot_archive.db
"""

import argparse
import sys
from pathlib import Path

from loguru import logger

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prediction.src.data.weather.openmeteo_client import fetch_all_stations, save_to_sqlite


def main():
    parser = argparse.ArgumentParser(description="Fetch Open-Meteo weather data for ERCOT zones")
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument(
        "--db-path",
        type=str,
        default="scraper/data/ercot_archive.db",
    )
    args = parser.parse_args()

    db_path = Path(args.db_path)
    if not db_path.parent.exists():
        logger.error(f"Parent directory does not exist: {db_path.parent}")
        sys.exit(1)

    logger.info(f"Fetching weather data {args.start_year}-{args.end_year} → {db_path}")

    df = fetch_all_stations(
        start_year=args.start_year, end_year=args.end_year, db_path=db_path
    )

    if df.empty:
        logger.warning("No data fetched")
        sys.exit(1)

    logger.info(f"Total rows fetched: {len(df)}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
