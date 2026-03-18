#!/usr/bin/env python3
"""
Complete February 28, 2026 data by fetching missing hours from InfluxDB.
"""

import os
import sys
import csv
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

INFLUXDB_URL = os.getenv("INFLUXDB_URL", "https://us-east-1-1.aws.cloud2.influxdata.com")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG", "0691bd05e35a51b2")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET", "ercot")

CSV_PATH = Path(__file__).parent.parent / "data" / "feb2026_dam_rtm_west.csv"

def fetch_feb28_complete():
    """Fetch complete Feb 28 data from InfluxDB"""

    if not INFLUXDB_TOKEN:
        print("ERROR: INFLUXDB_TOKEN not set")
        return None

    try:
        from influxdb_client_3 import InfluxDBClient3
    except ImportError:
        print("ERROR: influxdb3-python not installed")
        return None

    client = InfluxDBClient3(
        host=INFLUXDB_URL.replace("https://", "").replace("http://", ""),
        token=INFLUXDB_TOKEN,
        org=INFLUXDB_ORG,
        database=INFLUXDB_BUCKET,
    )

    data = defaultdict(dict)

    # Fetch DAM data for Feb 28 (need to query Feb 28-Mar 1 UTC to cover full day in CT)
    print("Fetching DAM data for Feb 28...")
    dam_query = """
    SELECT time, settlement_point, lmp
    FROM "dam_lmp"
    WHERE settlement_point IN ('HB_WEST', 'LZ_WEST')
    AND time >= '2026-02-28T00:00:00Z'
    AND time <= '2026-03-01T05:59:59Z'
    ORDER BY time ASC
    """

    result = client.query(dam_query)
    df_dam = result.to_pandas()
    print(f"  Found {len(df_dam)} DAM records")

    # Convert UTC to Central Time and filter for Feb 28 only
    df_dam['ct_time'] = df_dam['time'] - timedelta(hours=6)
    df_dam['date'] = df_dam['ct_time'].dt.strftime('%Y-%m-%d')
    df_dam_feb28 = df_dam[df_dam['date'] == '2026-02-28']

    for _, row in df_dam_feb28.iterrows():
        ct_time = row['ct_time']
        date_str = ct_time.strftime('%Y-%m-%d')
        hour_ending = ct_time.hour + 1

        sp = row['settlement_point']
        key = (date_str, hour_ending, sp)
        data[key]['dam_price'] = round(row['lmp'], 2)

    # Fetch RTM data for Feb 28 (need to query Feb 28-29 UTC to cover full day in CT)
    print("Fetching RTM data for Feb 28...")
    rtm_query = """
    SELECT time, settlement_point, lmp
    FROM "rtm_lmp_api"
    WHERE settlement_point IN ('HB_WEST', 'LZ_WEST')
    AND time >= '2026-02-28T00:00:00Z'
    AND time <= '2026-03-01T05:59:59Z'
    ORDER BY time ASC
    """

    result = client.query(rtm_query)
    df_rtm = result.to_pandas()
    print(f"  Found {len(df_rtm)} RTM records")

    # Group by hour and average
    df_rtm['ct_time'] = df_rtm['time'] - timedelta(hours=6)
    df_rtm['date'] = df_rtm['ct_time'].dt.strftime('%Y-%m-%d')
    df_rtm['hour'] = df_rtm['ct_time'].dt.hour

    # Filter only Feb 28 in Central Time
    df_rtm_feb28 = df_rtm[df_rtm['date'] == '2026-02-28']

    for (sp, hour), group in df_rtm_feb28.groupby(['settlement_point', 'hour']):
        avg_lmp = group['lmp'].mean()
        hour_ending = hour + 1
        key = ('2026-02-28', hour_ending, sp)
        data[key]['rtm_price'] = round(avg_lmp, 2)

    client.close()
    return data

def read_existing_csv():
    """Read existing CSV"""
    existing = {}
    with open(CSV_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row['oper_date'], int(row['hour_ending']), row['settlement_point'])
            existing[key] = {
                'dam_price': row['dam_price'],
                'rtm_price': row['rtm_price']
            }
    return existing

def merge_and_write(existing, new_data):
    """Merge new data and write CSV"""
    # Update existing with new data
    updated_count = 0
    for key, values in new_data.items():
        if key not in existing:
            existing[key] = values
            updated_count += 1
        else:
            if 'dam_price' in values and not existing[key]['dam_price']:
                existing[key]['dam_price'] = values['dam_price']
                updated_count += 1
            if 'rtm_price' in values and not existing[key]['rtm_price']:
                existing[key]['rtm_price'] = values['rtm_price']
                updated_count += 1

    print(f"Updated {updated_count} records")

    # Write back to CSV
    sorted_keys = sorted(existing.keys(), key=lambda x: (x[0], x[1], x[2]))

    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['oper_date', 'hour_ending', 'settlement_point', 'dam_price', 'rtm_price'])

        for key in sorted_keys:
            oper_date, hour_ending, sp = key
            values = existing[key]
            writer.writerow([
                oper_date,
                hour_ending,
                sp,
                values.get('dam_price', ''),
                values.get('rtm_price', '')
            ])

def main():
    print("Completing February 28, 2026 data...")

    # Fetch new data
    new_data = fetch_feb28_complete()
    if not new_data:
        return 1

    print(f"Fetched {len(new_data)} records from InfluxDB")

    # Read existing CSV
    existing = read_existing_csv()
    print(f"Read {len(existing)} existing records")

    # Merge and write
    merge_and_write(existing, new_data)

    print("Done! CSV updated.")

    # Verify Feb 28
    print("\nVerifying Feb 28 data...")
    feb28_count = sum(1 for k in existing.keys() if k[0] == '2026-02-28')
    print(f"  Feb 28 records: {feb28_count} (should be 48 for 24 hours × 2 nodes)")

    return 0

if __name__ == "__main__":
    sys.exit(main())
