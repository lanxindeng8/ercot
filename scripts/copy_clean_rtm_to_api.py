#!/usr/bin/env python3
"""
Copy clean RTM LMP data to rtm_lmp_api measurement.
Only copies data before CDR scraper started (2026-02-08T15:34:50Z).
"""

import os
import sys
import time
from datetime import datetime, timedelta

sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

from dotenv import load_dotenv
load_dotenv()

from influxdb_client_3 import InfluxDBClient3, Point

# CDR scraper started at this time - only copy data before this
CDR_START = '2026-02-08T15:34:50Z'
DATA_START = '2026-01-22T00:00:00Z'

BATCH_SIZE = 5000
BATCH_DELAY = 1  # seconds between batches


def main():
    url = os.environ.get('INFLUXDB_URL')
    token = os.environ.get('INFLUXDB_TOKEN')
    org = os.environ.get('INFLUXDB_ORG')

    client = InfluxDBClient3(
        host=url.replace('https://', '').replace('http://', ''),
        token=token,
        org=org,
        database='ercot',
    )

    print("=" * 60)
    print("Copy Clean RTM Data to rtm_lmp_api")
    print("=" * 60)
    print(f"Source: rtm_lmp (before {CDR_START})")
    print(f"Target: rtm_lmp_api")
    print()

    # Get total count
    count_query = f'''
    SELECT COUNT(*) as count FROM "rtm_lmp"
    WHERE time >= '{DATA_START}' AND time < '{CDR_START}'
    '''
    result = client.query(count_query)
    df = result.to_pandas()
    total = int(df['count'].iloc[0]) if not df.empty else 0
    print(f"Total records to copy: {total:,}")
    print()

    # Copy day by day
    current_start = datetime.fromisoformat(DATA_START.replace('Z', '+00:00'))
    final_end = datetime.fromisoformat(CDR_START.replace('Z', '+00:00'))

    copied = 0
    day_num = 0
    start_time = time.time()

    while current_start < final_end:
        current_end = min(current_start + timedelta(days=1), final_end)
        day_num += 1

        query = f'''
        SELECT time, settlement_point, lmp, energy_component, congestion_component, loss_component
        FROM "rtm_lmp"
        WHERE time >= '{current_start.isoformat()}' AND time < '{current_end.isoformat()}'
        '''

        try:
            result = client.query(query)
            df = result.to_pandas()

            if not df.empty:
                # Convert to points
                points = []
                for _, row in df.iterrows():
                    point = (
                        Point("rtm_lmp_api")
                        .tag("settlement_point", str(row.get('settlement_point', '') or ''))
                        .field("lmp", float(row.get('lmp', 0) or 0))
                        .field("energy_component", float(row.get('energy_component', 0) or 0))
                        .field("congestion_component", float(row.get('congestion_component', 0) or 0))
                        .field("loss_component", float(row.get('loss_component', 0) or 0))
                        .time(row['time'])
                    )
                    points.append(point)

                # Write in batches
                for i in range(0, len(points), BATCH_SIZE):
                    batch = points[i:i + BATCH_SIZE]
                    client.write(record=batch)
                    time.sleep(BATCH_DELAY)

                copied += len(points)

            elapsed = time.time() - start_time
            rate = copied / elapsed if elapsed > 0 else 0
            eta = (total - copied) / rate if rate > 0 else 0

            print(f"Day {day_num:2d}: {current_start.date()} - {len(df) if not df.empty else 0:6,} records | "
                  f"Total: {copied:,}/{total:,} ({100*copied/total:.1f}%) | "
                  f"ETA: {eta/60:.1f} min")

        except Exception as e:
            print(f"Error on day {day_num}: {e}")

        current_start = current_end

    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print(f"Completed! Copied {copied:,} records in {elapsed/60:.1f} minutes")
    print("=" * 60)

    client.close()


if __name__ == "__main__":
    main()
