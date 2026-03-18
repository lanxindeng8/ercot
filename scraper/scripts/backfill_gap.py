#!/usr/bin/env python3
"""
Backfill Gap Script for ERCOT Archive Database

Backfills rtm_lmp_hist and dam_lmp_hist from 2026-02-15 to today using
data from the CDR/API tables (rtm_lmp_cdr, rtm_lmp_api, dam_lmp).

Schema mapping:
  rtm_lmp_cdr (time, settlement_point, lmp)
    → rtm_lmp_hist (delivery_date, delivery_hour, delivery_interval,
                     repeated_hour, settlement_point, settlement_point_type, lmp)

  dam_lmp (time, settlement_point, settlement_point_type, lmp)
    → dam_lmp_hist (delivery_date, hour_ending, repeated_hour,
                     settlement_point, lmp)

Idempotent: uses INSERT OR REPLACE on primary keys.
"""

import sqlite3
import sys
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "ercot_archive.db"

FOCUS_SETTLEMENT_POINTS = [
    "HB_WEST", "HB_NORTH", "HB_SOUTH", "HB_HOUSTON", "HB_BUSAVG",
    "LZ_WEST", "LZ_NORTH", "LZ_SOUTH", "LZ_HOUSTON",
]

# hist tables stop at 2026-02-14
DEFAULT_GAP_START = "2026-02-15"


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Open a read-write connection to the archive database."""
    if not db_path.exists():
        logger.error("Database not found at %s", db_path)
        sys.exit(1)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def ensure_hist_tables(conn: sqlite3.Connection) -> None:
    """Create hist tables if they don't exist (matches import_historical.py schema)."""
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS rtm_lmp_hist (
        delivery_date TEXT NOT NULL,
        delivery_hour INTEGER NOT NULL,
        delivery_interval INTEGER NOT NULL,
        repeated_hour TEXT NOT NULL DEFAULT 'N',
        settlement_point TEXT NOT NULL,
        settlement_point_type TEXT,
        lmp REAL,
        PRIMARY KEY (delivery_date, delivery_hour, delivery_interval,
                     repeated_hour, settlement_point)
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS dam_lmp_hist (
        delivery_date TEXT NOT NULL,
        hour_ending TEXT NOT NULL,
        repeated_hour TEXT NOT NULL DEFAULT 'N',
        settlement_point TEXT NOT NULL,
        lmp REAL,
        PRIMARY KEY (delivery_date, hour_ending, repeated_hour, settlement_point)
    )
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# RTM backfill: CDR/API → hist
# ---------------------------------------------------------------------------

def _utc_to_cpt_components(utc_iso: str) -> dict:
    """
    Convert a UTC ISO timestamp to ERCOT Central Prevailing Time components.

    Returns dict with delivery_date, delivery_hour (1-24), delivery_interval (1-4).
    Assumes CST (UTC-6) — DST is ignored for simplicity as ERCOT uses
    repeated_hour='Y' for the fall-back hour.
    """
    utc_dt = datetime.fromisoformat(utc_iso)
    # Convert UTC → CPT (CST = UTC-6)
    cpt = utc_dt - timedelta(hours=6)
    delivery_date = cpt.strftime("%Y-%m-%d")
    # ERCOT hours are 1-24 (hour ending), so hour 0 → hour ending 24 of prior day
    if cpt.hour == 0:
        delivery_hour = 24
        prev = cpt - timedelta(days=1)
        delivery_date = prev.strftime("%Y-%m-%d")
    else:
        delivery_hour = cpt.hour
    # 15-minute intervals (1-4) derived from 5-minute CDR timestamps
    delivery_interval = (cpt.minute // 15) + 1
    return {
        "delivery_date": delivery_date,
        "delivery_hour": delivery_hour,
        "delivery_interval": delivery_interval,
    }


def backfill_rtm(conn: sqlite3.Connection, start_date: str, end_date: str) -> int:
    """
    Backfill rtm_lmp_hist from rtm_lmp_cdr and rtm_lmp_api.

    Strategy: prefer rtm_lmp_api (has components), fall back to rtm_lmp_cdr.
    For each 15-min interval, take the average LMP of the constituent 5-min records.

    Args:
        conn: Database connection
        start_date: Start date (YYYY-MM-DD) in CPT
        end_date: End date (YYYY-MM-DD) in CPT

    Returns:
        Number of records inserted/updated
    """
    cur = conn.cursor()
    ensure_hist_tables(conn)

    # Convert CPT date range to UTC for querying CDR tables
    start_utc = datetime.fromisoformat(start_date) + timedelta(hours=6)
    end_utc = datetime.fromisoformat(end_date) + timedelta(days=1, hours=6)

    sp_filter = ",".join(f"'{sp}'" for sp in FOCUS_SETTLEMENT_POINTS)

    # Try rtm_lmp_api first (better data), then rtm_lmp_cdr as fallback
    source_table = None
    for candidate in ("rtm_lmp_api", "rtm_lmp_cdr"):
        try:
            cur.execute(
                f'SELECT COUNT(*) FROM "{candidate}" '
                f'WHERE time >= ? AND time < ? '
                f'AND settlement_point IN ({sp_filter})',
                (start_utc.isoformat(), end_utc.isoformat()),
            )
            count = cur.fetchone()[0]
            if count > 0:
                source_table = candidate
                logger.info("Using %s as RTM source (%d records in range)", candidate, count)
                break
        except sqlite3.OperationalError:
            continue

    if not source_table:
        logger.warning("No RTM source data found for %s to %s", start_date, end_date)
        return 0

    # Fetch source data — aggregate 5-min → 15-min by averaging
    # Group by the 15-min interval: floor(minute/15)*15
    logger.info("Querying %s for RTM data...", source_table)
    cur.execute(
        f'SELECT time, settlement_point, lmp FROM "{source_table}" '
        f'WHERE time >= ? AND time < ? '
        f'AND settlement_point IN ({sp_filter}) '
        f'ORDER BY time',
        (start_utc.isoformat(), end_utc.isoformat()),
    )
    rows = cur.fetchall()
    logger.info("Fetched %d source records", len(rows))

    # Aggregate into 15-min buckets
    buckets = {}  # (delivery_date, delivery_hour, delivery_interval, sp) → [lmp]
    for time_val, sp, lmp in rows:
        components = _utc_to_cpt_components(time_val)
        key = (
            components["delivery_date"],
            components["delivery_hour"],
            components["delivery_interval"],
            sp,
        )
        buckets.setdefault(key, []).append(lmp)

    # Insert into rtm_lmp_hist
    inserted = 0
    for (dd, dh, di, sp), lmp_values in buckets.items():
        avg_lmp = sum(lmp_values) / len(lmp_values)
        try:
            cur.execute(
                'INSERT OR REPLACE INTO rtm_lmp_hist '
                '(delivery_date, delivery_hour, delivery_interval, repeated_hour, '
                ' settlement_point, settlement_point_type, lmp) '
                'VALUES (?, ?, ?, ?, ?, ?, ?)',
                (dd, dh, di, "N", sp, None, round(avg_lmp, 2)),
            )
            inserted += 1
        except Exception as e:
            logger.error("Error inserting RTM hist record: %s", e)

    conn.commit()
    logger.info("Inserted/updated %d RTM hist records", inserted)
    return inserted


# ---------------------------------------------------------------------------
# DAM backfill: dam_lmp → dam_lmp_hist
# ---------------------------------------------------------------------------

def backfill_dam(conn: sqlite3.Connection, start_date: str, end_date: str) -> int:
    """
    Backfill dam_lmp_hist from dam_lmp table.

    Schema mapping:
      dam_lmp.time (UTC) → delivery_date + hour_ending in CPT
      dam_lmp.lmp → dam_lmp_hist.lmp

    Args:
        conn: Database connection
        start_date: Start date (YYYY-MM-DD) in CPT
        end_date: End date (YYYY-MM-DD) in CPT

    Returns:
        Number of records inserted/updated
    """
    cur = conn.cursor()
    ensure_hist_tables(conn)

    # Convert CPT date range to UTC
    start_utc = datetime.fromisoformat(start_date) + timedelta(hours=6)
    end_utc = datetime.fromisoformat(end_date) + timedelta(days=1, hours=6)

    sp_filter = ",".join(f"'{sp}'" for sp in FOCUS_SETTLEMENT_POINTS)

    # Query dam_lmp
    logger.info("Querying dam_lmp for DAM data...")
    cur.execute(
        f'SELECT time, settlement_point, lmp FROM dam_lmp '
        f'WHERE time >= ? AND time < ? '
        f'AND settlement_point IN ({sp_filter}) '
        f'ORDER BY time',
        (start_utc.isoformat(), end_utc.isoformat()),
    )
    rows = cur.fetchall()
    logger.info("Fetched %d DAM source records", len(rows))

    if not rows:
        logger.warning("No DAM source data found for %s to %s", start_date, end_date)
        return 0

    inserted = 0
    for time_val, sp, lmp in rows:
        utc_dt = datetime.fromisoformat(time_val)
        cpt = utc_dt - timedelta(hours=6)

        # ERCOT hour_ending format: "HH:00"
        if cpt.hour == 0:
            hour_ending = "24:00"
            prev = cpt - timedelta(days=1)
            delivery_date = prev.strftime("%Y-%m-%d")
        else:
            hour_ending = f"{cpt.hour:02d}:00"
            delivery_date = cpt.strftime("%Y-%m-%d")

        try:
            cur.execute(
                'INSERT OR REPLACE INTO dam_lmp_hist '
                '(delivery_date, hour_ending, repeated_hour, settlement_point, lmp) '
                'VALUES (?, ?, ?, ?, ?)',
                (delivery_date, hour_ending, "N", sp, round(lmp, 2)),
            )
            inserted += 1
        except Exception as e:
            logger.error("Error inserting DAM hist record: %s", e)

    conn.commit()
    logger.info("Inserted/updated %d DAM hist records", inserted)
    return inserted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_backfill(db_path: Path = DB_PATH, start_date: str = None,
                 end_date: str = None) -> dict:
    """
    Run the full backfill process.

    Args:
        db_path: Path to the SQLite database
        start_date: Start date (YYYY-MM-DD). Default: 2026-02-15
        end_date: End date (YYYY-MM-DD). Default: today

    Returns:
        Dict with counts of records backfilled per table
    """
    start = start_date or DEFAULT_GAP_START
    end = end_date or datetime.now().strftime("%Y-%m-%d")

    logger.info("Backfilling from %s to %s", start, end)
    conn = get_connection(db_path)

    results = {}
    results["rtm_lmp_hist"] = backfill_rtm(conn, start, end)
    results["dam_lmp_hist"] = backfill_dam(conn, start, end)

    conn.close()
    logger.info("Backfill complete: %s", results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill historical ERCOT data gaps")
    parser.add_argument("--start-date", default=DEFAULT_GAP_START,
                        help="Start date YYYY-MM-DD (default: 2026-02-15)")
    parser.add_argument("--end-date", default=datetime.now().strftime("%Y-%m-%d"),
                        help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--db-path", default=None,
                        help="Path to SQLite database (default: data/ercot_archive.db)")
    args = parser.parse_args()

    db = Path(args.db_path) if args.db_path else DB_PATH
    results = run_backfill(db, args.start_date, args.end_date)
    print(f"\nBackfill results: {results}")
