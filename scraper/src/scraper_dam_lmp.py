#!/usr/bin/env python3
"""
Day-Ahead Market LMP Scraper

Fetches DAM LMP (Day-Ahead Market Locational Marginal Price) data from ERCOT
and writes to:
1. SQLite (all settlement points) - primary storage
2. InfluxDB (filtered settlement points) - for frontend

Usage:
    python scraper_dam_lmp.py                     # Normal operation (from last timestamp)
    python scraper_dam_lmp.py --start-date 2026-01-01  # Backfill from specific date
"""

import argparse
import os
import sys
from datetime import datetime, timedelta

from ercot_client import create_client_from_env
from influxdb_writer import create_writer_from_env
from sqlite_archive import create_archive_from_env

# Settlement points to write to InfluxDB (for frontend)
# All data goes to SQLite, only these go to InfluxDB
INFLUXDB_SETTLEMENT_POINTS = {
    # Hubs
    "HB_BUSAVG", "HB_HOUSTON", "HB_HUBAVG", "HB_NORTH", "HB_PAN", "HB_SOUTH", "HB_WEST",
    # Load Zones
    "LZ_AEN", "LZ_CPS", "LZ_HOUSTON", "LZ_LCRA", "LZ_NORTH", "LZ_RAYBN", "LZ_SOUTH", "LZ_WEST",
}


def main():
    """Main scraper function for DAM LMP data"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="DAM LMP Scraper")
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date for backfill (YYYY-MM-DD format). Overrides auto-detection.",
    )
    args = parser.parse_args()

    print(f"Starting DAM LMP scraper at {datetime.utcnow().isoformat()}")

    try:
        # Initialize clients
        print("Initializing ERCOT client...")
        ercot = create_client_from_env()

        print("Initializing SQLite archive (primary storage)...")
        sqlite = create_archive_from_env()

        print("Initializing InfluxDB writer (for frontend)...")
        influxdb = create_writer_from_env()

        # Get start date
        print("Determining start date...")

        if args.start_date:
            # Use command-line specified start date
            start_date = datetime.fromisoformat(args.start_date)
            print(f"Using specified start date: {start_date.isoformat()}")
        else:
            # Auto-detect from last timestamp in SQLite
            last_timestamp = sqlite.get_last_time("dam_lmp")

            if last_timestamp:
                # Start from last timestamp
                start_date = last_timestamp
                print(f"Found last timestamp in SQLite: {start_date.isoformat()}")
            else:
                # Try InfluxDB as fallback
                last_timestamp = influxdb.get_last_timestamp("dam_lmp")
                if last_timestamp:
                    start_date = last_timestamp
                    print(f"Found last timestamp in InfluxDB: {start_date.isoformat()}")
                else:
                    # Default: start from 7 days ago
                    start_date = datetime.utcnow() - timedelta(days=7)
                    print(f"No existing data, starting from: {start_date.isoformat()}")

        # Fetch and write data
        print("Fetching DAM LMP data from ERCOT...")
        total_sqlite = 0
        total_influx = 0

        for page_data in ercot.fetch_spp_day_ahead_hourly(start_date):
            print(f"Received {len(page_data)} records")

            # Write ALL data to SQLite (primary storage)
            sqlite_written = sqlite.write_dam_lmp_raw(page_data)
            total_sqlite += sqlite_written

            # Filter for InfluxDB (only frontend settlement points)
            filtered_data = [
                r for r in page_data
                if (r.get("SettlementPoint") or r.get("settlementPoint", "")) in INFLUXDB_SETTLEMENT_POINTS
            ]

            # Write filtered data to InfluxDB
            if filtered_data:
                influx_written = influxdb.write_dam_lmp_data(filtered_data)
                total_influx += influx_written

        print(f"Completed! SQLite: {total_sqlite} records, InfluxDB: {total_influx} records")

        # Close connections
        sqlite.close()
        influxdb.close()

        return 0

    except Exception as e:
        print(f"Error in DAM LMP scraper: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
