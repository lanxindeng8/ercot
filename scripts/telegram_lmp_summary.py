#!/usr/bin/env python3
"""
Telegram Bot: Daily LMP Summary

Sends daily summary of ERCOT LMP data (RTM & DAM) to Telegram.
Runs at 2 AM daily to report on previous day's data.

Data source priority:
1. SQLite archive (populated by archive-to-sqlite at 1 AM)
2. Fallback to InfluxDB if SQLite data is incomplete

Usage:
    python scripts/telegram_lmp_summary.py [--date YYYY-MM-DD]
"""

import os
import sys
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta, timezone
from argparse import ArgumentParser
import requests

from dotenv import load_dotenv
load_dotenv()


# Configuration
SETTLEMENT_POINT = "LZ_WEST"
PRICE_THRESHOLD = 60.0  # $60
RTM_EXPECTED_INTERVALS = 288  # 5-min intervals × 24 hours
DAM_EXPECTED_HOURS = 24
RTM_MIN_THRESHOLD = 200  # Minimum RTM records to consider SQLite data "complete"
DAM_MIN_THRESHOLD = 20   # Minimum DAM records to consider SQLite data "complete"

# SQLite database path
SQLITE_DB_PATH = Path("/Users/nancy/projects/trueflux/ercot-scraper/data/ercot_archive.db")


def get_utc_range_for_date(date_str: str) -> tuple:
    """
    Convert Central Time date to UTC range.
    Central Time midnight = 06:00 UTC
    """
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    start_utc = date_obj.replace(hour=6, minute=0, second=0, tzinfo=timezone.utc)
    end_utc = start_utc + timedelta(days=1)
    return start_utc.isoformat(), end_utc.isoformat()


def get_influxdb_client():
    """Create InfluxDB client from environment variables"""
    try:
        from influxdb_client_3 import InfluxDBClient3
    except ImportError:
        print("Warning: influxdb_client_3 not installed, InfluxDB fallback disabled")
        return None

    url = os.environ.get("INFLUXDB_URL")
    token = os.environ.get("INFLUXDB_TOKEN")
    org = os.environ.get("INFLUXDB_ORG")
    database = os.environ.get("INFLUXDB_BUCKET", "ercot")

    if not all([url, token, org]):
        print("Warning: InfluxDB credentials not configured, fallback disabled")
        return None

    return InfluxDBClient3(
        host=url.replace("https://", "").replace("http://", ""),
        token=token,
        org=org,
        database=database,
    )


# ============== RTM Data Fetching (Per-bucket fallback) ==============

def generate_rtm_bucket_keys(date_str: str) -> list:
    """Generate all 288 5-minute time bucket keys for a day (in UTC, normalized format)"""
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    start_utc = date_obj.replace(hour=6, minute=0, second=0, tzinfo=timezone.utc)

    bucket_keys = []
    for i in range(288):
        bucket = start_utc + timedelta(minutes=5 * i)
        bucket_keys.append(normalize_bucket_key(bucket))
    return bucket_keys


def normalize_bucket_key(dt: datetime) -> str:
    """Normalize datetime to a consistent bucket key format (no timezone suffix)"""
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    bucket = dt.replace(second=0, microsecond=0)
    bucket = bucket.replace(minute=(bucket.minute // 5) * 5)
    return bucket.strftime("%Y-%m-%dT%H:%M:%S")


def fetch_sqlite_rtm_by_table(conn: sqlite3.Connection, table: str, date_str: str) -> dict:
    """Fetch RTM data from a specific SQLite table, return as {bucket_key: lmp}"""
    start_utc, end_utc = get_utc_range_for_date(date_str)
    cursor = conn.cursor()
    result = {}

    try:
        cursor.execute(f"""
            SELECT time, lmp FROM {table}
            WHERE settlement_point = ?
            AND time >= ? AND time < ?
        """, (SETTLEMENT_POINT, start_utc, end_utc))

        for time_str, lmp in cursor.fetchall():
            # Parse SQLite time (may or may not have timezone)
            if 'Z' in time_str:
                dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
            elif '+' in time_str or time_str.endswith('-00:00'):
                dt = datetime.fromisoformat(time_str)
            else:
                # No timezone - assume UTC
                dt = datetime.fromisoformat(time_str).replace(tzinfo=timezone.utc)

            bucket_key = normalize_bucket_key(dt)
            if bucket_key not in result:
                result[bucket_key] = lmp
    except sqlite3.OperationalError:
        pass

    return result


def fetch_influxdb_rtm_by_table(client, table: str, date_str: str) -> dict:
    """Fetch RTM data from a specific InfluxDB table, return as {bucket_key: lmp}"""
    start_utc, end_utc = get_utc_range_for_date(date_str)
    result = {}

    query = f"""
    SELECT time, lmp
    FROM "{table}"
    WHERE settlement_point = '{SETTLEMENT_POINT}'
    AND time >= '{start_utc}'
    AND time < '{end_utc}'
    """

    try:
        query_result = client.query(query)
        df = query_result.to_pandas()
        if not df.empty:
            for _, row in df.iterrows():
                dt = row['time'].to_pydatetime()
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                bucket_key = normalize_bucket_key(dt)
                if bucket_key not in result:
                    result[bucket_key] = row['lmp']
    except Exception as e:
        print(f"  Error querying InfluxDB {table}: {e}")

    return result


def fetch_rtm_data_with_fallback(conn, influx_client, date_str: str) -> tuple:
    """
    Fetch RTM data with per-bucket fallback priority:
    1. SQLite rtm_lmp_cdr
    2. SQLite rtm_lmp_api
    3. InfluxDB rtm_lmp_realtime
    4. InfluxDB rtm_lmp_api

    Returns: (prices_list, source_stats_dict)
    """
    bucket_keys = generate_rtm_bucket_keys(date_str)

    # Fetch all data sources
    sources = {}

    # SQLite sources
    if conn:
        sources['sqlite_cdr'] = fetch_sqlite_rtm_by_table(conn, 'rtm_lmp_cdr', date_str)
        sources['sqlite_api'] = fetch_sqlite_rtm_by_table(conn, 'rtm_lmp_api', date_str)
        print(f"  SQLite rtm_lmp_cdr: {len(sources['sqlite_cdr'])} records")
        print(f"  SQLite rtm_lmp_api: {len(sources['sqlite_api'])} records")

    # InfluxDB sources
    if influx_client:
        sources['influx_cdr'] = fetch_influxdb_rtm_by_table(influx_client, 'rtm_lmp_realtime', date_str)
        sources['influx_api'] = fetch_influxdb_rtm_by_table(influx_client, 'rtm_lmp_api', date_str)
        print(f"  InfluxDB rtm_lmp_realtime: {len(sources['influx_cdr'])} records")
        print(f"  InfluxDB rtm_lmp_api: {len(sources['influx_api'])} records")

    # Priority order for fallback
    priority = ['sqlite_cdr', 'sqlite_api', 'influx_cdr', 'influx_api']

    # Build final prices with per-bucket fallback
    prices = []
    source_counts = {k: 0 for k in priority}

    for bucket_key in bucket_keys:
        lmp = None
        used_source = None

        for source_name in priority:
            if source_name in sources and bucket_key in sources[source_name]:
                lmp = sources[source_name][bucket_key]
                used_source = source_name
                break

        if lmp is not None:
            prices.append(lmp)
            source_counts[used_source] += 1

    return prices, source_counts


def fetch_dam_data_sqlite(conn: sqlite3.Connection, date_str: str) -> list:
    """Fetch DAM LMP data for a specific date from SQLite"""
    start_utc, end_utc = get_utc_range_for_date(date_str)
    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT time, lmp FROM dam_lmp
            WHERE settlement_point = ?
            AND time >= ? AND time < ?
            ORDER BY time ASC
        """, (SETTLEMENT_POINT, start_utc, end_utc))

        rows = cursor.fetchall()
        return [lmp for _, lmp in rows]
    except sqlite3.OperationalError:
        return []


# ============== DAM Data Fetching ==============

def fetch_dam_data_influxdb(client, date_str: str) -> list:
    """Fetch DAM LMP data for a specific date from InfluxDB"""
    start_utc, end_utc = get_utc_range_for_date(date_str)

    query = f"""
    SELECT time, lmp
    FROM "dam_lmp"
    WHERE settlement_point = '{SETTLEMENT_POINT}'
    AND time >= '{start_utc}'
    AND time < '{end_utc}'
    ORDER BY time ASC
    """

    try:
        result = client.query(query)
        df = result.to_pandas()
        return df['lmp'].tolist() if not df.empty else []
    except Exception as e:
        print(f"Error fetching DAM data from InfluxDB: {e}")
        return []


# ============== Main Logic ==============

def calculate_stats(prices: list) -> dict:
    """Calculate LMP statistics."""
    count = len(prices)

    if not prices:
        return {
            "count": 0,
            "over_threshold": 0,
            "over_pct": 0,
            "avg_under_threshold": None,
        }

    over_threshold = sum(1 for p in prices if p > PRICE_THRESHOLD)
    over_pct = (over_threshold / count) * 100 if count > 0 else 0

    under_prices = [p for p in prices if p <= PRICE_THRESHOLD]
    avg_under = sum(under_prices) / len(under_prices) if under_prices else None

    return {
        "count": count,
        "over_threshold": over_threshold,
        "over_pct": over_pct,
        "avg_under_threshold": avg_under,
    }


def format_message(date_str: str, rtm_stats: dict, dam_stats: dict) -> str:
    """Format the Telegram message"""
    rtm_avg = f"${rtm_stats['avg_under_threshold']:.2f}" if rtm_stats['avg_under_threshold'] else "N/A"
    rtm_icon = "✅" if rtm_stats['count'] >= RTM_EXPECTED_INTERVALS else "❓"
    rtm_data_status = f"{rtm_stats['count']}/{RTM_EXPECTED_INTERVALS} {rtm_icon}"
    rtm_over_hours = rtm_stats['over_threshold'] / 12

    dam_avg = f"${dam_stats['avg_under_threshold']:.2f}" if dam_stats['avg_under_threshold'] else "N/A"
    dam_icon = "✅" if dam_stats['count'] >= DAM_EXPECTED_HOURS else "❓"
    dam_data_status = f"{dam_stats['count']}/{DAM_EXPECTED_HOURS} {dam_icon}"

    message = f"""ERCOT LMP Daily Summary
Date: {date_str} | {SETTLEMENT_POINT}

RTM: {rtm_data_status}
  >${PRICE_THRESHOLD:.0f}: {rtm_over_hours:.1f} hours
  Avg (<=${PRICE_THRESHOLD:.0f}): {rtm_avg}

DAM: {dam_data_status}
  >${PRICE_THRESHOLD:.0f}: {dam_stats['over_threshold']} hours
  Avg (<=${PRICE_THRESHOLD:.0f}): {dam_avg}"""

    return message


def send_telegram_message(message: str) -> bool:
    """Send message via Telegram Bot API to all configured chat IDs"""
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_ids_str = os.environ.get("TELEGRAM_CHAT_IDS", os.environ.get("TELEGRAM_CHAT_ID", ""))

    if not bot_token or not chat_ids_str:
        print("Error: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_IDS not set")
        return False

    chat_ids = [cid.strip() for cid in chat_ids_str.split(",") if cid.strip()]
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    success = True

    for chat_id in chat_ids:
        payload = {"chat_id": chat_id, "text": message}

        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                print(f"Message sent to {chat_id}")
            else:
                print(f"Error sending to {chat_id}: {response.text}")
                success = False
        except Exception as e:
            print(f"Error sending to {chat_id}: {e}")
            success = False

    return success


def get_yesterday() -> str:
    """Get yesterday's date in YYYY-MM-DD format"""
    yesterday = datetime.now() - timedelta(days=1)
    return yesterday.strftime("%Y-%m-%d")


def main():
    parser = ArgumentParser(description="Send daily LMP summary to Telegram")
    parser.add_argument("--date", type=str, default=None, help="Date to report on (YYYY-MM-DD)")
    args = parser.parse_args()

    target_date = args.date or get_yesterday()

    print("=" * 60)
    print("ERCOT LMP Daily Summary Bot")
    print("=" * 60)
    print(f"Date: {target_date}")
    print(f"Settlement Point: {SETTLEMENT_POINT}")
    print()

    # Open connections
    conn = None
    influx_client = None

    if SQLITE_DB_PATH.exists():
        conn = sqlite3.connect(str(SQLITE_DB_PATH))
        print(f"SQLite: {SQLITE_DB_PATH}")
    else:
        print(f"SQLite not found: {SQLITE_DB_PATH}")

    influx_client = get_influxdb_client()
    if influx_client:
        print("InfluxDB: connected")
    else:
        print("InfluxDB: not available")

    print()
    print("Fetching RTM data (per-bucket fallback)...")

    # Fetch RTM with per-bucket fallback
    rtm_prices, rtm_source_counts = fetch_rtm_data_with_fallback(conn, influx_client, target_date)

    # Determine primary source for display
    if rtm_source_counts:
        primary_source = max(rtm_source_counts, key=rtm_source_counts.get)
        source_summary = ", ".join(f"{k}:{v}" for k, v in rtm_source_counts.items() if v > 0)
        print(f"  RTM source breakdown: {source_summary}")
    else:
        primary_source = "N/A"

    print()
    print("Fetching DAM data...")

    # Fetch DAM (simpler, just SQLite then InfluxDB fallback)
    dam_prices = []

    if conn:
        dam_prices = fetch_dam_data_sqlite(conn, target_date)
        print(f"  SQLite DAM: {len(dam_prices)} records")

    if len(dam_prices) < DAM_MIN_THRESHOLD and influx_client:
        dam_prices_influx = fetch_dam_data_influxdb(influx_client, target_date)
        print(f"  InfluxDB DAM: {len(dam_prices_influx)} records")
        if len(dam_prices_influx) > len(dam_prices):
            dam_prices = dam_prices_influx

    # Close connections
    if conn:
        conn.close()
    if influx_client:
        influx_client.close()

    print()
    print(f"Final data: RTM={len(rtm_prices)}, DAM={len(dam_prices)}")

    # Calculate statistics
    rtm_stats = calculate_stats(rtm_prices)
    dam_stats = calculate_stats(dam_prices)

    # Format and send message
    message = format_message(target_date, rtm_stats, dam_stats)
    print()
    print("Message:")
    print("-" * 40)
    print(message)
    print("-" * 40)
    print()

    # Send to Telegram
    print("Sending to Telegram...")
    success = send_telegram_message(message)

    print("=" * 60)
    print("Done!" if success else "Failed to send message")
    print("=" * 60)


if __name__ == "__main__":
    main()
