#!/usr/bin/env python3
"""Fetch ONE missing month of wind forecast data, then exit.

Designed to be called by cron every hour. Finds the first missing month,
fetches it, and exits. If rate-limited (429), exits immediately without retry.
When all months are filled, prints COMPLETE and exits 0.
"""
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

# Must run with PYTHONPATH=~/projects/ercot
from scraper.src.ercot_client import create_client_from_env
from prediction.src.data.ercot.wind_forecast import (
    fetch_wind_forecast,
    deduplicate_latest,
    pivot_to_regions,
    save_to_sqlite,
)

DB_PATH = Path(__file__).resolve().parents[2] / "scraper" / "data" / "ercot_archive.db"
TARGET_START = "2022-12"
TARGET_END = "2026-03"


def get_missing_months(db_path: Path) -> list[str]:
    """Return list of YYYY-MM strings missing from wind_forecast table."""
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT DISTINCT substr(delivery_date, 1, 7) FROM wind_forecast"
    ).fetchall()
    conn.close()
    have = {r[0] for r in rows}

    missing = []
    cur = datetime.strptime(TARGET_START, "%Y-%m")
    end = datetime.strptime(TARGET_END, "%Y-%m")
    while cur <= end:
        ym = cur.strftime("%Y-%m")
        if ym not in have:
            missing.append(ym)
        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1)
        else:
            cur = cur.replace(month=cur.month + 1)
    return missing


def main():
    missing = get_missing_months(DB_PATH)
    if not missing:
        print("COMPLETE — all wind forecast months filled")
        return

    target = missing[0]
    print(f"Missing months: {len(missing)} — fetching {target}")

    # Compute date range for this month
    year, month = int(target[:4]), int(target[5:7])
    start_date = f"{year}-{month:02d}-01"
    if month == 12:
        end_date = f"{year}-12-31"
    else:
        from datetime import date
        import calendar
        last_day = calendar.monthrange(year, month)[1]
        end_date = f"{year}-{month:02d}-{last_day}"

    # Create client with max_retries=0 so 429 fails immediately
    from scraper.src.ercot_client import ErcotClient
    import os
    client = ErcotClient(
        username=os.environ["ERCOT_API_USERNAME"],
        password=os.environ["ERCOT_API_PASSWORD"],
        public_subscription_key=os.environ["ERCOT_PUBLIC_API_SUBSCRIPTION_KEY"],
        max_retries=1,  # one retry then give up
    )

    try:
        df = fetch_wind_forecast(client, start_date, end_date)
        if df.empty:
            print(f"No data returned for {target}")
            return
        df = deduplicate_latest(df)
        df = pivot_to_regions(df)
        save_to_sqlite(df, DB_PATH)
        print(f"OK — saved {len(df)} rows for {target}")

        # Check if done
        remaining = get_missing_months(DB_PATH)
        if not remaining:
            print("COMPLETE — all wind forecast months filled")
    except Exception as e:
        if "429" in str(e):
            print(f"RATE_LIMITED — skipping {target}, will retry next hour")
        else:
            print(f"ERROR — {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
