#!/usr/bin/env python3
"""Fetch missing wind forecast months sequentially until rate-limited or done.

Designed to be called by cron every hour. Fetches missing months one by one
in order. Stops on 429 rate limit or when all months are filled.
"""
import calendar
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

from scraper.src.ercot_client import ErcotClient
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

    print(f"Missing months: {len(missing)} — {missing}")

    client = ErcotClient(
        username=os.environ["ERCOT_API_USERNAME"],
        password=os.environ["ERCOT_API_PASSWORD"],
        public_subscription_key=os.environ["ERCOT_PUBLIC_API_SUBSCRIPTION_KEY"],
        max_retries=1,  # one retry then give up on 429
    )

    fetched = 0
    for target in missing:
        year, month = int(target[:4]), int(target[5:7])
        start_date = f"{year}-{month:02d}-01"
        last_day = calendar.monthrange(year, month)[1]
        end_date = f"{year}-{month:02d}-{last_day}"

        print(f"Fetching {target} ({start_date} → {end_date})...")
        try:
            df = fetch_wind_forecast(client, start_date, end_date)
            if df.empty:
                print(f"  No data returned for {target}, skipping")
                continue
            df = deduplicate_latest(df)
            df = pivot_to_regions(df)
            save_to_sqlite(df, DB_PATH)
            fetched += 1
            print(f"  OK — saved {len(df)} rows for {target}")
        except Exception as e:
            if "429" in str(e):
                print(f"RATE_LIMITED after {fetched} months — stopping, will retry next hour")
                return
            else:
                print(f"  ERROR on {target}: {e}")
                continue

    remaining = get_missing_months(DB_PATH)
    if not remaining:
        print(f"COMPLETE — fetched {fetched} months, all wind forecast months filled")
    else:
        print(f"Done this run: {fetched} months fetched, {len(remaining)} still missing")


if __name__ == "__main__":
    main()
