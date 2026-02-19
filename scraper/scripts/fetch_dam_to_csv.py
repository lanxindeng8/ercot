#!/usr/bin/env python3
"""
Fetch DAM LMP data from ERCOT API and save to CSV file.

This script bypasses InfluxDB to fetch historical data that exceeds
the InfluxDB Cloud free tier retention limit (30 days).

Usage:
    python scripts/fetch_dam_to_csv.py --start-date 2026-01-01 --output dam_2026.csv
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Load environment variables from .env file
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    import os
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value

from ercot_client import create_client_from_env


def main():
    parser = argparse.ArgumentParser(description="Fetch DAM LMP data to CSV")
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD format)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dam_data.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--settlement-point",
        type=str,
        default=None,
        help="Filter by settlement point (default: all)",
    )
    args = parser.parse_args()

    start_date = datetime.fromisoformat(args.start_date)
    output_path = Path(args.output)

    print(f"Fetching DAM data from {start_date.strftime('%Y-%m-%d')}...")
    print(f"Output file: {output_path}")
    if args.settlement_point:
        print(f"Settlement point filter: {args.settlement_point}")

    # Initialize ERCOT client
    print("Initializing ERCOT client...")
    ercot = create_client_from_env()

    # Fetch data
    print("Fetching data from ERCOT API...")
    all_records = []

    for page_data in ercot.fetch_spp_day_ahead_hourly(start_date):
        print(f"  Received {len(page_data)} records")

        for record in page_data:
            # Extract fields
            delivery_date = record.get("DeliveryDate") or record.get("deliveryDate")
            hour_ending = record.get("HourEnding") or record.get("hourEnding")
            settlement_point = record.get("SettlementPoint") or record.get("settlementPoint")
            price = record.get("SettlementPointPrice") or record.get("settlementPointPrice")

            if not delivery_date or not hour_ending or price is None:
                continue

            # Filter by settlement point if specified
            if args.settlement_point and settlement_point != args.settlement_point:
                continue

            # Parse hour from HourEnding (format: "01:00", "02:00", etc.)
            hour = int(hour_ending.split(":")[0])

            # Convert date format from YYYY-MM-DD to MM/DD/YYYY
            date_obj = datetime.strptime(delivery_date, "%Y-%m-%d")
            date_str = date_obj.strftime("%m/%d/%Y")

            all_records.append({
                "date": date_str,
                "hour": hour,
                "dam_price": float(price),
                "settlement_point": settlement_point,
            })

    print(f"\nTotal records fetched: {len(all_records)}")

    # Write to CSV
    print(f"Writing to {output_path}...")

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "hour", "dam_price", "settlement_point"])
        writer.writeheader()
        writer.writerows(all_records)

    print(f"Done! Wrote {len(all_records)} records to {output_path}")

    # Show summary
    if all_records:
        dates = sorted(set(r["date"] for r in all_records))
        points = sorted(set(r["settlement_point"] for r in all_records))
        print(f"\nDate range: {dates[0]} to {dates[-1]}")
        print(f"Settlement points: {len(points)}")
        print(f"  {', '.join(points[:5])}{'...' if len(points) > 5 else ''}")


if __name__ == "__main__":
    main()
