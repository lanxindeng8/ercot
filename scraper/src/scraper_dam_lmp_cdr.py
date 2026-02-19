#!/usr/bin/env python3
"""
DAM LMP CDR Scraper

Scrapes DAM Settlement Point Prices from ERCOT's CDR HTML page:
  https://www.ercot.com/content/cdr/html/dam_spp.html

Saves to a SEPARATE SQLite table (dam_lmp_cdr) so it can be compared
with the API-sourced data in dam_lmp without any overwrites.

Usage:
    python scraper_dam_lmp_cdr.py                  # Scrape only new dates
    python scraper_dam_lmp_cdr.py --backfill       # Scrape all available dates (02/14-02/19 initially)
    python scraper_dam_lmp_cdr.py --date 20260218  # Scrape a specific date
"""

import argparse
import re
import sys
from datetime import datetime, timedelta
from html.parser import HTMLParser
from pathlib import Path
from typing import List, Dict, Any, Optional
import urllib.request

# Same DB as the rest of the project
DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "ercot_archive.db"

BASE_URL = "https://www.ercot.com/content/cdr/html"
CURRENT_PAGE = f"{BASE_URL}/dam_spp.html"

# Column headers in the HTML table (in order)
COLUMNS = [
    "oper_day", "hour_ending",
    "HB_BUSAVG", "HB_HOUSTON", "HB_HUBAVG", "HB_NORTH", "HB_PAN", "HB_SOUTH", "HB_WEST",
    "LZ_AEN", "LZ_CPS", "LZ_HOUSTON", "LZ_LCRA", "LZ_NORTH", "LZ_RAYBN", "LZ_SOUTH", "LZ_WEST",
]


class DamSppParser(HTMLParser):
    """Parse the DAM SPP HTML table into rows of data."""

    def __init__(self):
        super().__init__()
        self.rows: List[List[str]] = []
        self._current_row: List[str] = []
        self._in_td = False
        self._in_th = False
        self._available_dates: List[str] = []
        self._in_option = False
        self._option_value = ""

    def handle_starttag(self, tag, attrs):
        if tag == "td":
            self._in_td = True
            self._current_cell = ""
        elif tag == "th":
            self._in_th = True
        elif tag == "option":
            self._in_option = True
            for name, val in attrs:
                if name == "value":
                    self._option_value = val or ""

    def handle_endtag(self, tag):
        if tag == "td":
            self._in_td = False
            self._current_row.append(self._current_cell.strip())
        elif tag == "th":
            self._in_th = False
        elif tag == "tr":
            if self._current_row:
                self.rows.append(self._current_row)
            self._current_row = []
        elif tag == "option":
            self._in_option = False

    def handle_data(self, data):
        if self._in_td:
            self._current_cell += data


def fetch_page(url: str) -> str:
    """Fetch HTML content from URL."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def parse_available_dates(html: str) -> List[str]:
    """Extract available date codes from the dropdown JS.
    
    The page JS computes dates dynamically. We look for option count
    and compute the dates ourselves based on currentDate.
    """
    # Count "history" options
    history_count = html.count('value="history"')
    
    # Get currentDate
    m = re.search(r'id="currentDate"\s+value="(\d{2}/\d{2}/\d{4})"', html)
    if not m:
        return []
    
    current_date = datetime.strptime(m.group(1), "%m/%d/%Y")
    
    # The dropdown has history_count history entries, then "today", then "tomorrow"
    # total options = history_count + 2
    # dates go from (current - history_count) days ago to current + 1 (tomorrow)
    # But we only need history + today; tomorrow is the current page
    dates = []
    start = current_date - timedelta(days=history_count)
    for i in range(history_count + 2):  # +2 for today and tomorrow
        d = start + timedelta(days=i)
        dates.append(d.strftime("%Y%m%d"))
    
    return dates


def parse_rows(html: str) -> List[Dict[str, Any]]:
    """Parse HTML table into list of record dicts."""
    parser = DamSppParser()
    parser.feed(html)

    records = []
    for row in parser.rows:
        if len(row) != len(COLUMNS):
            continue
        # Skip if first cell doesn't look like a date
        if not re.match(r"\d{2}/\d{2}/\d{4}", row[0]):
            continue

        oper_day = row[0]  # MM/DD/YYYY
        hour_ending = int(row[1].strip())

        # Parse oper_day + hour_ending into a timestamp
        # Hour ending 1 = 00:00-01:00, so the interval starts at hour_ending - 1
        dt = datetime.strptime(oper_day, "%m/%d/%Y")
        dt = dt.replace(hour=hour_ending - 1)

        # Convert CPT to UTC (CST = UTC-6, CDT = UTC-5)
        # For simplicity, use UTC-6 (CST) — matches existing dam_lmp table convention
        dt_utc = dt + timedelta(hours=6)

        for i, col_name in enumerate(COLUMNS[2:], start=2):
            try:
                lmp = float(row[i])
            except (ValueError, IndexError):
                continue

            records.append({
                "time": dt_utc.isoformat(),
                "oper_day": oper_day,
                "hour_ending": hour_ending,
                "settlement_point": col_name,
                "lmp": lmp,
            })

    return records


def init_cdr_table(conn):
    """Create the dam_lmp_cdr table if it doesn't exist."""
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS dam_lmp_cdr (
        time DATETIME NOT NULL,
        oper_day TEXT NOT NULL,
        hour_ending INTEGER NOT NULL,
        settlement_point TEXT NOT NULL,
        lmp REAL NOT NULL,
        scraped_at DATETIME NOT NULL DEFAULT (datetime('now')),
        PRIMARY KEY (time, settlement_point)
    )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_dam_cdr_time ON dam_lmp_cdr(time)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_dam_cdr_sp ON dam_lmp_cdr(settlement_point)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_dam_cdr_oper_day ON dam_lmp_cdr(oper_day)")
    conn.commit()


def write_records(conn, records: List[Dict[str, Any]]) -> int:
    """Write records to dam_lmp_cdr. Returns count written."""
    if not records:
        return 0

    cursor = conn.cursor()
    count = 0
    now = datetime.utcnow().isoformat()

    for r in records:
        try:
            cursor.execute("""
            INSERT OR REPLACE INTO dam_lmp_cdr
            (time, oper_day, hour_ending, settlement_point, lmp, scraped_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (r["time"], r["oper_day"], r["hour_ending"],
                  r["settlement_point"], r["lmp"], now))
            count += 1
        except Exception as e:
            print(f"Error writing record: {e}")
    conn.commit()
    return count


def get_existing_oper_days(conn) -> set:
    """Get set of oper_day values already in the table."""
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT DISTINCT oper_day FROM dam_lmp_cdr")
        return {row[0] for row in cursor.fetchall()}
    except Exception:
        return set()


def main():
    import sqlite3

    parser = argparse.ArgumentParser(description="DAM LMP CDR Scraper")
    parser.add_argument("--backfill", action="store_true",
                        help="Scrape all available dates")
    parser.add_argument("--date", type=str,
                        help="Scrape a specific date (YYYYMMDD)")
    args = parser.parse_args()

    print(f"Starting DAM LMP CDR scraper at {datetime.utcnow().isoformat()}")

    conn = sqlite3.connect(str(DEFAULT_DB_PATH))
    init_cdr_table(conn)

    total = 0

    if args.date:
        # Single date
        url = f"{BASE_URL}/{args.date}_dam_spp.html"
        print(f"Fetching {url}...")
        html = fetch_page(url)
        records = parse_rows(html)
        written = write_records(conn, records)
        print(f"  {args.date}: {written} records")
        total += written

    else:
        # First, get available dates from the current page
        print("Fetching current page to discover available dates...")
        html = fetch_page(CURRENT_PAGE)
        available_dates = parse_available_dates(html)
        print(f"Available dates: {available_dates}")

        if not args.backfill:
            # Only scrape dates we don't have yet
            existing = get_existing_oper_days(conn)
            # Convert YYYYMMDD to MM/DD/YYYY for comparison
            existing_yyyymmdd = set()
            for od in existing:
                try:
                    existing_yyyymmdd.add(datetime.strptime(od, "%m/%d/%Y").strftime("%Y%m%d"))
                except ValueError:
                    pass
            dates_to_scrape = [d for d in available_dates if d not in existing_yyyymmdd]
            if not dates_to_scrape:
                print("No new dates to scrape.")
                conn.close()
                return 0
            print(f"New dates to scrape: {dates_to_scrape}")
        else:
            dates_to_scrape = available_dates

        # The current page (dam_spp.html) shows tomorrow's data
        # Historical dates use {YYYYMMDD}_dam_spp.html
        # Current date page is just dam_spp.html (which shows tomorrow)
        
        # First parse the current page we already fetched (tomorrow's data)
        current_records = parse_rows(html)
        if current_records:
            current_oper = current_records[0]["oper_day"]
            current_yyyymmdd = datetime.strptime(current_oper, "%m/%d/%Y").strftime("%Y%m%d")
            if current_yyyymmdd in dates_to_scrape:
                written = write_records(conn, current_records)
                print(f"  {current_yyyymmdd} (current page): {written} records")
                total += written
                dates_to_scrape = [d for d in dates_to_scrape if d != current_yyyymmdd]

        # Fetch remaining dates
        for date_code in dates_to_scrape:
            url = f"{BASE_URL}/{date_code}_dam_spp.html"
            print(f"Fetching {url}...")
            try:
                page_html = fetch_page(url)
                records = parse_rows(page_html)
                written = write_records(conn, records)
                print(f"  {date_code}: {written} records")
                total += written
            except Exception as e:
                print(f"  {date_code}: ERROR - {e}")

    print(f"\nTotal: {total} records written to dam_lmp_cdr")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
