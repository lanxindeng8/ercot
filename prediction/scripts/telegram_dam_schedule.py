#!/usr/bin/env python3
"""
Telegram Bot: DAM Price Schedule

Sends next-day DAM prices with on/off recommendations.
Runs daily at 2:30 PM after DAM prices are published.

Usage:
    python scripts/telegram_dam_schedule.py [--date YYYY-MM-DD]
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
MINING_REVENUE = 55.0   # $55/MWh breakeven price
SWITCHING_COST = 40.0   # $40 per on/off cycle

# SQLite database path
SQLITE_DB_PATH = Path("/Users/bot/projects/ercot/scraper/data/ercot_archive.db")


def get_utc_range_for_date(date_str: str) -> tuple:
    """
    Convert Central Time date to UTC range.
    Central Time midnight = 06:00 UTC
    Returns times without timezone suffix for SQLite compatibility.
    """
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    start_utc = date_obj.replace(hour=6, minute=0, second=0)
    end_utc = start_utc + timedelta(days=1)
    return start_utc.strftime("%Y-%m-%dT%H:%M:%S"), end_utc.strftime("%Y-%m-%dT%H:%M:%S")


def get_influxdb_client():
    """Create InfluxDB client from environment variables"""
    try:
        from influxdb_client_3 import InfluxDBClient3
    except ImportError:
        print("Warning: influxdb_client_3 not installed")
        return None

    url = os.environ.get("INFLUXDB_URL")
    token = os.environ.get("INFLUXDB_TOKEN")
    org = os.environ.get("INFLUXDB_ORG")
    database = os.environ.get("INFLUXDB_BUCKET", "ercot")

    if not all([url, token, org]):
        print("Warning: InfluxDB credentials not configured")
        return None

    return InfluxDBClient3(
        host=url.replace("https://", "").replace("http://", ""),
        token=token,
        org=org,
        database=database,
    )


def fetch_dam_prices_sqlite(conn: sqlite3.Connection, date_str: str) -> list:
    """Fetch DAM prices for a specific date from SQLite, return list of (hour, price) tuples"""
    start_utc, end_utc = get_utc_range_for_date(date_str)
    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT time, lmp FROM dam_lmp
            WHERE settlement_point = ?
            AND time >= ? AND time < ?
            ORDER BY time ASC
        """, (SETTLEMENT_POINT, start_utc, end_utc))

        results = []
        for time_str, lmp in cursor.fetchall():
            # Parse time and convert to hour starting (1-24)
            if 'Z' in time_str:
                dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
            elif '+' in time_str:
                dt = datetime.fromisoformat(time_str)
            else:
                dt = datetime.fromisoformat(time_str).replace(tzinfo=timezone.utc)

            # Convert UTC to Central Time hour ending, then to hour starting
            # UTC 06:00 = CT 00:00 = Hour Ending 1 = Hour Starting 0
            utc_hour = dt.hour
            hour_ending = ((utc_hour - 6) % 24) + 1  # 1-24
            hour_starting = hour_ending - 1  # 0-23

            results.append((hour_starting, lmp))

        return results
    except sqlite3.OperationalError as e:
        print(f"SQLite error: {e}")
        return []


def fetch_dam_prices_influxdb(client, date_str: str) -> list:
    """Fetch DAM prices for a specific date from InfluxDB"""
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
        if df.empty:
            return []

        results = []
        for _, row in df.iterrows():
            dt = row['time'].to_pydatetime()
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)

            utc_hour = dt.hour
            hour_ending = ((utc_hour - 6) % 24) + 1
            hour_starting = hour_ending - 1

            results.append((hour_starting, row['lmp']))

        return results
    except Exception as e:
        print(f"InfluxDB error: {e}")
        return []


def calculate_shutdown_schedule(prices: list) -> list:
    """
    Calculate optimal shutdown schedule based on:
    - Mining revenue: $55/MWh (breakeven price)
    - Switching cost: $40 per on/off cycle

    Algorithm:
    1. For each hour, calculate loss = max(0, price - mining_revenue)
    2. Find contiguous shutdown periods where total savings > switching cost
    3. Include breakeven hours (loss=0) in shutdown block if they're between
       high-loss hours to avoid paying extra switching costs

    Returns list of (hour, price, action) tuples where action is "ON" or "OFF"
    """
    if not prices:
        return []

    # Sort by hour
    sorted_prices = sorted(prices, key=lambda x: x[0])
    n = len(sorted_prices)

    # Calculate loss for each hour: max(0, price - mining_revenue)
    losses = []
    for hour, price in sorted_prices:
        loss = max(0, price - MINING_REVENUE)
        losses.append((hour, price, loss))

    # Initialize all hours as ON
    actions = ["ON"] * n

    # Try all possible contiguous shutdown periods [i, j]
    # Use greedy approach: find best period starting at each position
    i = 0
    while i < n:
        # Skip hours with no loss at the start (price <= mining_revenue)
        if losses[i][2] == 0:
            i += 1
            continue

        # Found an hour with loss > 0, find the best shutdown period starting here
        best_end = -1
        best_net_savings = 0

        # Try all possible end points
        current_savings = 0
        for j in range(i, n):
            current_savings += losses[j][2]
            net_savings = current_savings - SWITCHING_COST

            # Update best if this is better
            if net_savings > best_net_savings:
                best_net_savings = net_savings
                best_end = j

        # If we found a profitable period, mark those hours as OFF
        if best_end >= 0 and best_net_savings > 0:
            for k in range(i, best_end + 1):
                actions[k] = "OFF"
            i = best_end + 1
        else:
            i += 1

    # Build result with actions
    result = []
    for idx, (hour, price, loss) in enumerate(losses):
        result.append((hour, price, actions[idx]))

    return result


def format_schedule_message(date_str: str, prices: list) -> str:
    """Format the DAM schedule message"""
    if not prices:
        return f"No DAM data available for {date_str}"

    # Calculate optimal shutdown schedule
    schedule = calculate_shutdown_schedule(prices)

    lines = [
        f"DAM Schedule: {date_str}",
        f"Settlement Point: {SETTLEMENT_POINT}",
        "",
        "Hour | Price  | Action"
    ]

    for hour, price, action in schedule:
        hour_str = f"{hour:02d}:00"
        price_str = f"${price:6.2f}"
        lines.append(f"{hour_str} | {price_str} | {action}")

    # Summary
    on_hours = sum(1 for _, _, a in schedule if a == "ON")
    off_hours = sum(1 for _, _, a in schedule if a == "OFF")
    on_prices = [p for _, p, a in schedule if a == "ON"]
    avg_on_price = sum(on_prices) / len(on_prices) if on_prices else 0

    lines.extend([
        "",
        f"ON: {on_hours}h | OFF: {off_hours}h | Avg(ON): ${avg_on_price:.2f}"
    ])

    return "\n".join(lines)


def send_telegram_message(message: str) -> bool:
    """Send message via Telegram Bot API to all configured chat IDs"""
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_ids_str = os.environ.get("TELEGRAM_CHAT_IDS", os.environ.get("TELEGRAM_CHAT_ID", ""))

    if not bot_token:
        print("Error: TELEGRAM_BOT_TOKEN not set")
        return False

    if not chat_ids_str:
        print("Error: TELEGRAM_CHAT_IDS not set")
        return False

    chat_ids = [cid.strip() for cid in chat_ids_str.split(",") if cid.strip()]
    print(f"Sending to {len(chat_ids)} chat(s): {chat_ids}")

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    success_count = 0

    for chat_id in chat_ids:
        payload = {"chat_id": chat_id, "text": message}
        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                print(f"  ✓ Sent to {chat_id}")
                success_count += 1
            else:
                print(f"  ✗ Error sending to {chat_id}: {response.text}")
        except Exception as e:
            print(f"  ✗ Error sending to {chat_id}: {e}")

    return success_count > 0


def get_tomorrow() -> str:
    """Get tomorrow's date in YYYY-MM-DD format"""
    tomorrow = datetime.now() + timedelta(days=1)
    return tomorrow.strftime("%Y-%m-%d")


def main():
    parser = ArgumentParser(description="Send DAM schedule to Telegram")
    parser.add_argument("--date", type=str, default=None, help="Operation date (YYYY-MM-DD)")
    args = parser.parse_args()

    # Default to tomorrow (next day's operation)
    target_date = args.date or get_tomorrow()

    print("=" * 60)
    print("ERCOT DAM Schedule Bot")
    print("=" * 60)
    print(f"Operation Date: {target_date}")
    print(f"Settlement Point: {SETTLEMENT_POINT}")
    print()

    prices = []

    # Try SQLite first
    if SQLITE_DB_PATH.exists():
        print(f"Trying SQLite: {SQLITE_DB_PATH}")
        conn = sqlite3.connect(str(SQLITE_DB_PATH))
        prices = fetch_dam_prices_sqlite(conn, target_date)
        conn.close()
        print(f"  SQLite DAM: {len(prices)} records")

    # Fallback to InfluxDB
    if len(prices) < 20:
        print("Falling back to InfluxDB...")
        influx_client = get_influxdb_client()
        if influx_client:
            prices_influx = fetch_dam_prices_influxdb(influx_client, target_date)
            print(f"  InfluxDB DAM: {len(prices_influx)} records")
            if len(prices_influx) > len(prices):
                prices = prices_influx
            influx_client.close()

    print()
    print(f"Final data: {len(prices)} hours")

    # Format message
    message = format_schedule_message(target_date, prices)
    print()
    print("Message:")
    print("-" * 40)
    print(message)
    print("-" * 40)
    print()

    # Send to all configured chats
    print("Sending to Telegram...")
    success = send_telegram_message(message)

    print("=" * 60)
    print("Done!" if success else "Failed to send message")
    print("=" * 60)


if __name__ == "__main__":
    main()
