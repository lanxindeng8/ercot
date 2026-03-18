#!/usr/bin/env python3
"""
Fill missing February 2026 DAM/RTM data for West region
Reads from ercot_archive.db and updates CSV file
"""

import sqlite3
import csv
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta

# Paths
DB_PATH = Path(__file__).parent.parent / "data" / "ercot_archive.db"
CSV_PATH = Path(__file__).parent.parent / "data" / "feb2026_dam_rtm_west.csv"

def fetch_data_from_db(settlement_points=["HB_WEST", "LZ_WEST"]):
    """Fetch all February 2026 data from database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    data = defaultdict(dict)

    # Query DAM data
    print("Fetching DAM data from database...")
    dam_query = """
        SELECT delivery_date, hour_ending, settlement_point, lmp
        FROM dam_lmp_hist
        WHERE delivery_date LIKE '2026-02%'
        AND settlement_point IN ({})
        ORDER BY delivery_date, hour_ending, settlement_point
    """.format(','.join(['?' for _ in settlement_points]))

    cursor.execute(dam_query, settlement_points)
    dam_rows = cursor.fetchall()
    print(f"Found {len(dam_rows)} DAM records")

    for row in dam_rows:
        delivery_date, hour_ending, settlement_point, price = row
        key = (delivery_date, hour_ending, settlement_point)
        data[key]['dam_price'] = price

    # Query RTM data - aggregate by hour (average of all intervals)
    print("Fetching RTM data from database...")
    rtm_query = """
        SELECT delivery_date, delivery_hour, settlement_point, AVG(lmp) as avg_lmp
        FROM rtm_lmp_hist
        WHERE delivery_date LIKE '2026-02%'
        AND settlement_point IN ({})
        GROUP BY delivery_date, delivery_hour, settlement_point
        ORDER BY delivery_date, delivery_hour, settlement_point
    """.format(','.join(['?' for _ in settlement_points]))

    cursor.execute(rtm_query, settlement_points)
    rtm_rows = cursor.fetchall()
    print(f"Found {len(rtm_rows)} RTM aggregated records")

    for row in rtm_rows:
        delivery_date, delivery_hour, settlement_point, avg_lmp = row
        # RTM uses delivery_hour (0-23), but CSV uses hour_ending (1-24)
        hour_ending = delivery_hour + 1
        key = (delivery_date, hour_ending, settlement_point)
        if key in data:
            data[key]['rtm_price'] = round(avg_lmp, 2)
        else:
            data[key] = {'rtm_price': round(avg_lmp, 2)}

    conn.close()
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

def merge_data(existing_data, db_data):
    """Merge database data with existing CSV data"""
    merged = {}

    # Start with existing CSV data
    for key, values in existing_data.items():
        merged[key] = values.copy()

    # Fill in missing values from database
    filled_count = 0
    for key, db_values in db_data.items():
        if key not in merged:
            # Completely new row from database
            merged[key] = {
                'dam_price': db_values.get('dam_price', ''),
                'rtm_price': db_values.get('rtm_price', '')
            }
            filled_count += 1
        else:
            # Fill in missing RTM prices
            if not merged[key]['rtm_price'] and 'rtm_price' in db_values:
                merged[key]['rtm_price'] = db_values['rtm_price']
                filled_count += 1
            # Fill in missing DAM prices (unlikely but check anyway)
            if not merged[key]['dam_price'] and 'dam_price' in db_values:
                merged[key]['dam_price'] = db_values['dam_price']
                filled_count += 1

    print(f"Filled {filled_count} missing values from database")
    return merged

def write_csv(data):
    """Write merged data to CSV"""
    # Sort data by date, hour, settlement_point
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
    print("Starting data fill process...")
    print(f"Database: {DB_PATH}")
    print(f"CSV file: {CSV_PATH}")

    # Fetch data from database
    db_data = fetch_data_from_db()

    # Read existing CSV
    existing_data = read_existing_csv()

    # Merge data
    merged_data = merge_data(existing_data, db_data)

    # Write updated CSV
    write_csv(merged_data)

    # Print summary
    print("\nSummary:")
    print(f"Total rows in merged data: {len(merged_data)}")

    # Count missing RTM prices
    missing_rtm = sum(1 for v in merged_data.values() if not v.get('rtm_price'))
    print(f"Rows with missing RTM prices: {missing_rtm}")

    # Date range
    dates = sorted(set(k[0] for k in merged_data.keys()))
    if dates:
        print(f"Date range: {dates[0]} to {dates[-1]}")

if __name__ == "__main__":
    main()
