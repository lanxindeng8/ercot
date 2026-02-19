#!/usr/bin/env python3
"""
Real-Time Market LMP Scraper (API Version)

Fetches RTM LMP (Real-Time Market Locational Marginal Price) data from ERCOT
Public API and writes to:
1. SQLite (all settlement points) - primary storage
2. InfluxDB (filtered settlement points) - for frontend

Note: The ERCOT Public API has a ~6 hour delay. This scraper is used to
backfill historical data. Use scraper_rtm_lmp_realtime.py for real-time data.
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

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

# File to track the last API timestamp (separate from CDR data)
API_TIMESTAMP_FILE = Path(__file__).parent.parent / "logs" / "api_rtm_last_timestamp.txt"


def get_api_last_timestamp() -> datetime:
    """
    Get the last timestamp from API data (stored in file).
    This is separate from CDR real-time data.
    """
    try:
        if API_TIMESTAMP_FILE.exists():
            timestamp_str = API_TIMESTAMP_FILE.read_text().strip()
            return datetime.fromisoformat(timestamp_str)
    except Exception as e:
        print(f"Error reading API timestamp file: {e}")
    return None


def save_api_last_timestamp(timestamp: datetime):
    """Save the last API timestamp to file."""
    try:
        API_TIMESTAMP_FILE.parent.mkdir(parents=True, exist_ok=True)
        API_TIMESTAMP_FILE.write_text(timestamp.isoformat())
    except Exception as e:
        print(f"Error saving API timestamp: {e}")


def main():
    """Main scraper function for RTM LMP data"""
    print(f"Starting RTM LMP API scraper at {datetime.utcnow().isoformat()}")

    try:
        # Initialize clients
        print("Initializing ERCOT client...")
        ercot = create_client_from_env()

        print("Initializing SQLite archive (primary storage)...")
        sqlite = create_archive_from_env()

        print("Initializing InfluxDB writer (for frontend)...")
        influxdb = create_writer_from_env()

        # Get start date for API data (from file, not InfluxDB)
        print("Determining start date for API data...")
        last_timestamp = get_api_last_timestamp()

        if last_timestamp:
            # Start from last API timestamp
            start_date = last_timestamp
            print(f"Found last API timestamp: {start_date.isoformat()}")
        else:
            # Default: start from 7 days ago
            start_date = datetime.utcnow() - timedelta(days=7)
            print(f"No existing API timestamp file, starting from: {start_date.isoformat()}")

        # Fetch and write data
        print("Fetching RTM LMP data from ERCOT...")
        total_sqlite = 0
        total_influx = 0
        latest_timestamp = None

        for page_data in ercot.fetch_lmp_by_settlement_point(start_date):
            print(f"Received {len(page_data)} records")

            # Track latest timestamp in this batch
            for record in page_data:
                ts_str = record.get("SCEDTimestamp") or record.get("scedTimestamp")
                if ts_str:
                    try:
                        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        ts = ts.replace(tzinfo=None)  # Store as naive UTC
                        if latest_timestamp is None or ts > latest_timestamp:
                            latest_timestamp = ts
                    except:
                        pass

            # Write ALL data to SQLite (primary storage)
            sqlite_written = sqlite.write_rtm_lmp_raw(page_data)
            total_sqlite += sqlite_written

            # Filter for InfluxDB (only frontend settlement points)
            filtered_data = [
                r for r in page_data
                if (r.get("SettlementPoint") or r.get("settlementPoint", "")) in INFLUXDB_SETTLEMENT_POINTS
            ]

            # Write filtered data to InfluxDB
            if filtered_data:
                influx_written = influxdb.write_rtm_lmp_data(filtered_data)
                total_influx += influx_written

        # Save the latest timestamp for next run
        if latest_timestamp:
            save_api_last_timestamp(latest_timestamp)
            print(f"Saved last API timestamp: {latest_timestamp.isoformat()}")

        print(f"Completed! SQLite: {total_sqlite} records, InfluxDB: {total_influx} records")

        # Close connections
        sqlite.close()
        influxdb.close()

        return 0

    except Exception as e:
        print(f"Error in RTM LMP scraper: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
