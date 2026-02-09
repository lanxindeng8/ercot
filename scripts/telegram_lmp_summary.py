#!/usr/bin/env python3
"""
Telegram Bot: Daily LMP Summary

Sends daily summary of ERCOT LMP data (RTM & DAM) to Telegram.
Runs at 2 AM daily to report on previous day's data.

Usage:
    python scripts/telegram_lmp_summary.py [--date YYYY-MM-DD]
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from argparse import ArgumentParser
import requests

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from influxdb_client_3 import InfluxDBClient3


# Configuration
SETTLEMENT_POINT = "LZ_WEST"
PRICE_THRESHOLD = 60.0  # $60
RTM_EXPECTED_INTERVALS = 288  # 5-min intervals × 24 hours
DAM_EXPECTED_HOURS = 24


def get_influxdb_client():
    """Create InfluxDB client from environment variables"""
    url = os.environ.get("INFLUXDB_URL")
    token = os.environ.get("INFLUXDB_TOKEN")
    org = os.environ.get("INFLUXDB_ORG")
    database = os.environ.get("INFLUXDB_BUCKET", "ercot")

    return InfluxDBClient3(
        host=url.replace("https://", "").replace("http://", ""),
        token=token,
        org=org,
        database=database,
    )


def get_utc_range_for_date(date_str: str) -> tuple:
    """
    Convert Central Time date to UTC range.
    Central Time midnight = 06:00 UTC
    """
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    start_utc = f"{date_str}T06:00:00Z"
    next_date = date_obj + timedelta(days=1)
    end_utc = f"{next_date.strftime('%Y-%m-%d')}T06:00:00Z"
    return start_utc, end_utc


def fetch_rtm_data(client, date_str: str) -> list:
    """Fetch RTM LMP data for a specific date (5-minute intervals, 288 per day)"""
    start_utc, end_utc = get_utc_range_for_date(date_str)

    query = f"""
    SELECT time, lmp
    FROM "rtm_lmp"
    WHERE settlement_point = '{SETTLEMENT_POINT}'
    AND time >= '{start_utc}'
    AND time < '{end_utc}'
    ORDER BY time ASC
    """

    try:
        result = client.query(query)
        df = result.to_pandas()
        if df.empty:
            return []
        # Deduplicate by rounding to 5-minute intervals (scraper may write duplicates)
        df['time_bucket'] = df['time'].dt.floor('5min')
        df = df.drop_duplicates(subset=['time_bucket'], keep='first')
        return df['lmp'].tolist()
    except Exception as e:
        print(f"Error fetching RTM data: {e}")
        return []


def fetch_dam_data(client, date_str: str) -> list:
    """Fetch DAM LMP data for a specific date"""
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
        print(f"Error fetching DAM data: {e}")
        return []


def calculate_stats(prices: list) -> dict:
    """
    Calculate LMP statistics.

    Returns:
        dict with keys: count, over_threshold, over_pct, avg_under_threshold
    """
    count = len(prices)

    if not prices:
        return {
            "count": 0,
            "over_threshold": 0,
            "over_pct": 0,
            "avg_under_threshold": None,
        }

    # Count prices over threshold
    over_threshold = sum(1 for p in prices if p > PRICE_THRESHOLD)
    over_pct = (over_threshold / count) * 100 if count > 0 else 0

    # Calculate average of prices under or equal to threshold
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

    # RTM section (5-min intervals, 288 per day = 24 hours)
    rtm_avg = f"${rtm_stats['avg_under_threshold']:.2f}" if rtm_stats['avg_under_threshold'] else "N/A"
    rtm_icon = "✅" if rtm_stats['count'] >= RTM_EXPECTED_INTERVALS else "❓"
    rtm_data_status = f"{rtm_stats['count']}/{RTM_EXPECTED_INTERVALS} {rtm_icon}"
    rtm_over_hours = rtm_stats['over_threshold'] / 12  # 12 intervals per hour

    # DAM section (hourly, 24 per day)
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
        payload = {
            "chat_id": chat_id,
            "text": message,
        }

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
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date to report on (YYYY-MM-DD). Default: yesterday",
    )
    args = parser.parse_args()

    target_date = args.date or get_yesterday()

    print("=" * 60)
    print("ERCOT LMP Daily Summary Bot")
    print("=" * 60)
    print(f"Date: {target_date}")
    print(f"Settlement Point: {SETTLEMENT_POINT}")
    print()

    # Connect to InfluxDB
    print("Connecting to InfluxDB...")
    client = get_influxdb_client()

    # Fetch RTM data
    print("Fetching RTM data...")
    rtm_prices = fetch_rtm_data(client, target_date)
    print(f"  Found {len(rtm_prices)} RTM records")

    # Fetch DAM data
    print("Fetching DAM data...")
    dam_prices = fetch_dam_data(client, target_date)
    print(f"  Found {len(dam_prices)} DAM records")

    client.close()

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
