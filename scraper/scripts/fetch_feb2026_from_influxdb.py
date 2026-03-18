#!/usr/bin/env python3
"""
Fetch February 2026 DAM/RTM data from InfluxDB and fill the CSV file.
"""

import os
import sys
import csv
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# InfluxDB configuration
INFLUXDB_URL = os.getenv("INFLUXDB_URL", "https://us-east-1-1.aws.cloud2.influxdata.com")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG", "0691bd05e35a51b2")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET", "ercot")

CSV_PATH = Path(__file__).parent.parent / "data" / "feb2026_dam_rtm_west.csv"

def fetch_from_influxdb(settlement_points=["HB_WEST", "LZ_WEST"]):
    """Fetch February 2026 data from InfluxDB"""

    if not INFLUXDB_TOKEN:
        print("ERROR: INFLUXDB_TOKEN not set in environment")
        print("Please set it using:")
        print("  export INFLUXDB_TOKEN='your_token_here'")
        print("\nOr provide it in a .env file")
        return None

    try:
        from influxdb_client_3 import InfluxDBClient3
    except ImportError:
        print("ERROR: influxdb_client_3 not installed")
        print("Install it with: pip install influxdb3-python")
        return None

    print(f"Connecting to InfluxDB...")
    print(f"  URL: {INFLUXDB_URL}")
    print(f"  ORG: {INFLUXDB_ORG}")
    print(f"  Database: {INFLUXDB_BUCKET}")

    client = InfluxDBClient3(
        host=INFLUXDB_URL.replace("https://", "").replace("http://", ""),
        token=INFLUXDB_TOKEN,
        org=INFLUXDB_ORG,
        database=INFLUXDB_BUCKET,
    )

    data = defaultdict(dict)

    # Fetch DAM data
    print("\nFetching DAM data for February 2026...")
    for sp in settlement_points:
        dam_query = f"""
        SELECT time, lmp
        FROM "dam_lmp"
        WHERE settlement_point = '{sp}'
        AND time >= '2026-02-01T00:00:00Z'
        AND time <= '2026-02-28T23:59:59Z'
        ORDER BY time ASC
        """

        try:
            result = client.query(dam_query)
            df = result.to_pandas()

            if not df.empty:
                print(f"  Found {len(df)} DAM records for {sp}")

                for _, row in df.iterrows():
                    timestamp = row['time']
                    # Convert UTC to Central Time (CST = UTC-6)
                    # Note: This is simplified; in production use pytz for DST handling
                    ct_timestamp = timestamp - timedelta(hours=6)

                    date_str = ct_timestamp.strftime('%Y-%m-%d')
                    hour_ending = ct_timestamp.hour + 1  # Convert to HE1-HE24

                    key = (date_str, hour_ending, sp)
                    data[key]['dam_price'] = round(row['lmp'], 2)
            else:
                print(f"  No DAM data for {sp}")

        except Exception as e:
            print(f"  Error fetching DAM for {sp}: {e}")

    # Fetch RTM data
    print("\nFetching RTM data for February 2026...")
    for sp in settlement_points:
        rtm_query = f"""
        SELECT time, lmp
        FROM "rtm_lmp_api"
        WHERE settlement_point = '{sp}'
        AND time >= '2026-02-01T00:00:00Z'
        AND time <= '2026-02-28T23:59:59Z'
        ORDER BY time ASC
        """

        try:
            result = client.query(rtm_query)
            df = result.to_pandas()

            if not df.empty:
                print(f"  Found {len(df)} RTM records for {sp}")

                # Group by hour and take average
                df['time'] = df['time'].dt.floor('H')
                hourly_avg = df.groupby('time')['lmp'].mean()

                for timestamp, avg_lmp in hourly_avg.items():
                    # Convert UTC to Central Time
                    ct_timestamp = timestamp - timedelta(hours=6)

                    date_str = ct_timestamp.strftime('%Y-%m-%d')
                    hour_ending = ct_timestamp.hour + 1

                    key = (date_str, hour_ending, sp)
                    if key in data:
                        data[key]['rtm_price'] = round(avg_lmp, 2)
                    else:
                        data[key] = {'rtm_price': round(avg_lmp, 2)}
            else:
                print(f"  No RTM data for {sp}")

        except Exception as e:
            print(f"  Error fetching RTM for {sp}: {e}")

    client.close()
    return data

def read_existing_csv():
    """Read existing CSV data"""
    existing_data = {}

    if not CSV_PATH.exists():
        print(f"CSV file not found: {CSV_PATH}")
        return existing_data

    with open(CSV_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row['oper_date'], int(row['hour_ending']), row['settlement_point'])
            existing_data[key] = {
                'dam_price': row['dam_price'],
                'rtm_price': row['rtm_price']
            }

    print(f"Read {len(existing_data)} rows from existing CSV")
    return existing_data

def merge_data(existing_data, influx_data):
    """Merge InfluxDB data with existing CSV data"""
    merged = {}

    # Start with existing CSV data
    for key, values in existing_data.items():
        merged[key] = values.copy()

    # Fill in from InfluxDB
    filled_count = 0
    for key, influx_values in influx_data.items():
        if key not in merged:
            # New row from InfluxDB
            merged[key] = {
                'dam_price': influx_values.get('dam_price', ''),
                'rtm_price': influx_values.get('rtm_price', '')
            }
            filled_count += 1
        else:
            # Fill missing values
            if not merged[key]['rtm_price'] and 'rtm_price' in influx_values:
                merged[key]['rtm_price'] = influx_values['rtm_price']
                filled_count += 1
            if not merged[key]['dam_price'] and 'dam_price' in influx_values:
                merged[key]['dam_price'] = influx_values['dam_price']
                filled_count += 1

    print(f"Filled {filled_count} values from InfluxDB")
    return merged

def write_csv(data):
    """Write merged data to CSV"""
    sorted_keys = sorted(data.keys(), key=lambda x: (x[0], x[1], x[2]))

    print(f"Writing {len(sorted_keys)} rows to CSV...")

    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['oper_date', 'hour_ending', 'settlement_point', 'dam_price', 'rtm_price'])

        for key in sorted_keys:
            oper_date, hour_ending, settlement_point = key
            values = data[key]
            writer.writerow([
                oper_date,
                hour_ending,
                settlement_point,
                values.get('dam_price', ''),
                values.get('rtm_price', '')
            ])

    print(f"Successfully wrote CSV to {CSV_PATH}")

def main():
    print("=" * 60)
    print("Fetch February 2026 Data from InfluxDB")
    print("=" * 60)

    # Fetch from InfluxDB
    influx_data = fetch_from_influxdb()

    if influx_data is None:
        print("\nFailed to fetch data from InfluxDB")
        print("Please configure INFLUXDB_TOKEN and try again")
        return 1

    # Read existing CSV
    existing_data = read_existing_csv()

    # Merge data
    merged_data = merge_data(existing_data, influx_data)

    # Write updated CSV
    write_csv(merged_data)

    # Print summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Total rows: {len(merged_data)}")

    # Count missing values
    missing_rtm = sum(1 for v in merged_data.values() if not v.get('rtm_price'))
    missing_dam = sum(1 for v in merged_data.values() if not v.get('dam_price'))
    print(f"  Missing RTM prices: {missing_rtm}")
    print(f"  Missing DAM prices: {missing_dam}")

    # Date range
    dates = sorted(set(k[0] for k in merged_data.keys()))
    if dates:
        print(f"  Date range: {dates[0]} to {dates[-1]}")
    print("=" * 60)

    return 0

if __name__ == "__main__":
    sys.exit(main())
