#!/usr/bin/env python3
"""
Fetch February 28, 2026 data from InfluxDB to check for updates.
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

INFLUXDB_URL = os.getenv("INFLUXDB_URL", "https://us-east-1-1.aws.cloud2.influxdata.com")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG", "0691bd05e35a51b2")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET", "ercot")

def main():
    if not INFLUXDB_TOKEN:
        print("ERROR: INFLUXDB_TOKEN not set")
        return 1

    try:
        from influxdb_client_3 import InfluxDBClient3
    except ImportError:
        print("ERROR: influxdb3-python not installed")
        return 1

    client = InfluxDBClient3(
        host=INFLUXDB_URL.replace("https://", "").replace("http://", ""),
        token=INFLUXDB_TOKEN,
        org=INFLUXDB_ORG,
        database=INFLUXDB_BUCKET,
    )

    print("Querying InfluxDB for Feb 28, 2026 data...")

    # Check DAM data for Feb 28
    dam_query = """
    SELECT time, settlement_point, lmp
    FROM "dam_lmp"
    WHERE settlement_point IN ('HB_WEST', 'LZ_WEST')
    AND time >= '2026-02-28T00:00:00Z'
    AND time <= '2026-02-28T23:59:59Z'
    ORDER BY time ASC
    """

    print("\nDAM Data:")
    result = client.query(dam_query)
    df_dam = result.to_pandas()
    print(f"  Total records: {len(df_dam)}")
    if not df_dam.empty:
        print(f"  Time range: {df_dam['time'].min()} to {df_dam['time'].max()}")
        print(f"\n  Sample data:")
        print(df_dam.head(10).to_string())

    # Check RTM data for Feb 28
    rtm_query = """
    SELECT time, settlement_point, lmp
    FROM "rtm_lmp_api"
    WHERE settlement_point IN ('HB_WEST', 'LZ_WEST')
    AND time >= '2026-02-28T00:00:00Z'
    AND time <= '2026-02-28T23:59:59Z'
    ORDER BY time ASC
    """

    print("\n\nRTM Data:")
    result = client.query(rtm_query)
    df_rtm = result.to_pandas()
    print(f"  Total records: {len(df_rtm)}")
    if not df_rtm.empty:
        print(f"  Time range: {df_rtm['time'].min()} to {df_rtm['time'].max()}")

        # Group by hour to see hourly coverage
        df_rtm['hour'] = df_rtm['time'].dt.floor('h')
        hourly_counts = df_rtm.groupby(['hour', 'settlement_point']).size()
        print(f"\n  Hourly coverage:")
        for (hour, sp), count in hourly_counts.items():
            hour_ct = hour - timedelta(hours=6)  # Convert to Central Time
            print(f"    {hour_ct.strftime('%Y-%m-%d %H:%M')} {sp}: {count} records")

    client.close()

if __name__ == "__main__":
    sys.exit(main())
