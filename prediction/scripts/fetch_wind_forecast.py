#!/usr/bin/env python3
"""
CLI script to fetch ERCOT wind forecast data (NP4-732-CD).

Usage:
    python prediction/scripts/fetch_wind_forecast.py \
        --start-date 2025-01-01 --end-date 2025-12-31 \
        --db-path scraper/data/ercot_archive.db
"""

import argparse
from datetime import datetime
from pathlib import Path

from dateutil.relativedelta import relativedelta
from loguru import logger

from scraper.src.ercot_client import create_client_from_env
from prediction.src.data.ercot.wind_forecast import (
    fetch_wind_forecast,
    deduplicate_latest,
    pivot_to_regions,
    save_to_sqlite,
)


def monthly_chunks(start_date: str, end_date: str):
    """Yield (chunk_start, chunk_end) date strings in monthly intervals."""
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    while current <= end:
        chunk_end = min(current + relativedelta(months=1, days=-1), end)
        yield current.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        current = chunk_end + relativedelta(days=1)


def main():
    parser = argparse.ArgumentParser(description="Fetch ERCOT wind forecast data")
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--db-path",
        default="scraper/data/ercot_archive.db",
        help="Path to SQLite database",
    )
    args = parser.parse_args()

    db_path = Path(args.db_path)
    client = create_client_from_env()

    chunks = list(monthly_chunks(args.start_date, args.end_date))
    logger.info(f"Fetching wind forecast data in {len(chunks)} monthly chunks")

    total_rows = 0
    for i, (chunk_start, chunk_end) in enumerate(chunks, 1):
        logger.info(f"[{i}/{len(chunks)}] {chunk_start} to {chunk_end}")
        try:
            df = fetch_wind_forecast(client, chunk_start, chunk_end)
            if df.empty:
                continue
            df = deduplicate_latest(df)
            df = pivot_to_regions(df)
            save_to_sqlite(df, db_path)
            total_rows += len(df)
        except Exception:
            logger.exception(f"Failed to fetch chunk {chunk_start} to {chunk_end}")

    logger.info(f"Done. Total rows saved: {total_rows}")


if __name__ == "__main__":
    main()
