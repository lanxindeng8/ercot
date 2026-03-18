#!/usr/bin/env python3
"""
Data Audit Script for ERCOT Archive Database

Detects gaps, reports data quality issues (nulls, duplicates, anomalies),
and shows coverage summaries per settlement point across all tables.
"""

import sqlite3
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

# Setup logging
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


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Open a read-only connection to the archive database."""
    if not db_path.exists():
        logger.error("Database not found at %s", db_path)
        sys.exit(1)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Table overview
# ---------------------------------------------------------------------------

def audit_table_overview(conn: sqlite3.Connection) -> dict:
    """Return basic stats for every known table."""
    tables = {
        "rtm_lmp_hist": {"time_col": "delivery_date", "sp_col": "settlement_point"},
        "dam_lmp_hist": {"time_col": "delivery_date", "sp_col": "settlement_point"},
        "rtm_lmp_cdr":  {"time_col": "time",          "sp_col": "settlement_point"},
        "rtm_lmp_api":  {"time_col": "time",          "sp_col": "settlement_point"},
        "dam_lmp":      {"time_col": "time",          "sp_col": "settlement_point"},
        "fuel_mix_hist": {"time_col": "delivery_date", "sp_col": None},
    }

    overview = {}
    cur = conn.cursor()
    for table, meta in tables.items():
        try:
            cur.execute(f'SELECT COUNT(*) FROM "{table}"')
            count = cur.fetchone()[0]
        except sqlite3.OperationalError:
            logger.warning("Table %s does not exist – skipping", table)
            continue

        time_col = meta["time_col"]
        cur.execute(f'SELECT MIN("{time_col}"), MAX("{time_col}") FROM "{table}"')
        row = cur.fetchone()
        overview[table] = {
            "rows": count,
            "min_time": row[0],
            "max_time": row[1],
            "time_col": time_col,
            "sp_col": meta["sp_col"],
        }
    return overview


# ---------------------------------------------------------------------------
# Gap detection
# ---------------------------------------------------------------------------

def detect_date_gaps_hist(conn: sqlite3.Connection, table: str, time_col: str,
                          sp_col: str) -> dict:
    """Detect missing dates per settlement point in a historical table."""
    cur = conn.cursor()
    placeholders = ",".join("?" for _ in FOCUS_SETTLEMENT_POINTS)
    cur.execute(
        f'SELECT DISTINCT "{time_col}", "{sp_col}" '
        f'FROM "{table}" WHERE "{sp_col}" IN ({placeholders}) '
        f'ORDER BY "{time_col}"',
        FOCUS_SETTLEMENT_POINTS,
    )
    rows = cur.fetchall()

    dates_by_sp = defaultdict(set)
    for row in rows:
        dates_by_sp[row[1]].add(row[0])

    # Find the global date range
    all_dates = set()
    for d in dates_by_sp.values():
        all_dates |= d
    if not all_dates:
        return {}

    min_date = min(all_dates)
    max_date = max(all_dates)

    # Generate expected dates
    expected = set()
    d = datetime.strptime(min_date[:10], "%Y-%m-%d").date()
    end = datetime.strptime(max_date[:10], "%Y-%m-%d").date()
    while d <= end:
        expected.add(d.isoformat())
        d += timedelta(days=1)

    gaps = {}
    for sp in FOCUS_SETTLEMENT_POINTS:
        sp_dates = {dt[:10] for dt in dates_by_sp.get(sp, set())}
        missing = sorted(expected - sp_dates)
        if missing:
            gaps[sp] = missing
    return gaps


def detect_time_gaps_cdr(conn: sqlite3.Connection, table: str,
                         interval_minutes: int = 5) -> dict:
    """Detect missing time intervals per settlement point in CDR/API tables."""
    cur = conn.cursor()
    placeholders = ",".join("?" for _ in FOCUS_SETTLEMENT_POINTS)
    cur.execute(
        f'SELECT DATE(MIN(time)), DATE(MAX(time)) FROM "{table}" '
        f'WHERE settlement_point IN ({placeholders})',
        FOCUS_SETTLEMENT_POINTS,
    )
    row = cur.fetchone()
    if not row[0]:
        return {}
    min_day = datetime.strptime(row[0], "%Y-%m-%d").date()
    max_day = datetime.strptime(row[1], "%Y-%m-%d").date()

    counts_by_sp_day = defaultdict(dict)
    cur.execute(
        f'SELECT DATE(time) as d, settlement_point, COUNT(*) as cnt '
        f'FROM "{table}" WHERE settlement_point IN ({placeholders}) '
        f'GROUP BY d, settlement_point ORDER BY d',
        FOCUS_SETTLEMENT_POINTS,
    )
    for day, sp, cnt in cur.fetchall():
        counts_by_sp_day[sp][day] = cnt

    if interval_minutes == 5:
        expected_per_day = 288
    elif interval_minutes == 60:
        expected_per_day = 24
    else:
        expected_per_day = (24 * 60) // interval_minutes

    gaps = {}
    day = min_day
    expected_days = []
    while day <= max_day:
        expected_days.append(day.isoformat())
        day += timedelta(days=1)

    for sp in FOCUS_SETTLEMENT_POINTS:
        for expected_day in expected_days:
            cnt = counts_by_sp_day.get(sp, {}).get(expected_day, 0)
            if cnt < expected_per_day:
                gaps.setdefault(sp, []).append(
                    {"date": expected_day, "count": cnt, "expected": expected_per_day}
                )
    return gaps


# ---------------------------------------------------------------------------
# Data quality checks
# ---------------------------------------------------------------------------

def check_nulls(conn: sqlite3.Connection, table: str, columns: list) -> dict:
    """Count NULL values in specified columns."""
    cur = conn.cursor()
    results = {}
    for col in columns:
        try:
            cur.execute(f'SELECT COUNT(*) FROM "{table}" WHERE "{col}" IS NULL')
            count = cur.fetchone()[0]
            if count > 0:
                results[col] = count
        except sqlite3.OperationalError:
            pass
    return results


def check_duplicates_hist(conn: sqlite3.Connection, table: str) -> int:
    """Check for duplicate rows in historical tables based on known PKs."""
    cur = conn.cursor()
    if table == "rtm_lmp_hist":
        pk = "delivery_date, delivery_hour, delivery_interval, repeated_hour, settlement_point"
    elif table == "dam_lmp_hist":
        pk = "delivery_date, hour_ending, repeated_hour, settlement_point"
    else:
        return 0

    cur.execute(
        f'SELECT COUNT(*) FROM ('
        f'  SELECT {pk}, COUNT(*) as c FROM "{table}" '
        f'  GROUP BY {pk} HAVING c > 1'
        f')'
    )
    return cur.fetchone()[0]


def check_anomalies(conn: sqlite3.Connection, table: str, lmp_col: str = "lmp") -> dict:
    """Check for anomalous LMP values (negative, extreme, zero)."""
    cur = conn.cursor()
    anomalies = {}

    cur.execute(f'SELECT COUNT(*) FROM "{table}" WHERE "{lmp_col}" < 0')
    neg = cur.fetchone()[0]
    if neg:
        anomalies["negative_lmp"] = neg

    cur.execute(f'SELECT COUNT(*) FROM "{table}" WHERE "{lmp_col}" = 0')
    zero = cur.fetchone()[0]
    if zero:
        anomalies["zero_lmp"] = zero

    # Extreme values (> $5000 or < -$500)
    cur.execute(f'SELECT COUNT(*) FROM "{table}" WHERE "{lmp_col}" > 5000')
    high = cur.fetchone()[0]
    if high:
        anomalies["lmp_above_5000"] = high

    cur.execute(f'SELECT COUNT(*) FROM "{table}" WHERE "{lmp_col}" < -500')
    vlow = cur.fetchone()[0]
    if vlow:
        anomalies["lmp_below_neg500"] = vlow

    cur.execute(f'SELECT MIN("{lmp_col}"), MAX("{lmp_col}"), AVG("{lmp_col}") FROM "{table}"')
    row = cur.fetchone()
    anomalies["min_lmp"] = row[0]
    anomalies["max_lmp"] = row[1]
    anomalies["avg_lmp"] = round(row[2], 2) if row[2] else None

    return anomalies


# ---------------------------------------------------------------------------
# Coverage summary
# ---------------------------------------------------------------------------

def coverage_summary(conn: sqlite3.Connection, table: str, time_col: str,
                     sp_col: str) -> list:
    """Show per-settlement-point coverage for focus points."""
    cur = conn.cursor()
    placeholders = ",".join("?" for _ in FOCUS_SETTLEMENT_POINTS)
    cur.execute(
        f'SELECT "{sp_col}", COUNT(*) as cnt, '
        f'  MIN("{time_col}") as min_t, MAX("{time_col}") as max_t '
        f'FROM "{table}" WHERE "{sp_col}" IN ({placeholders}) '
        f'GROUP BY "{sp_col}" ORDER BY "{sp_col}"',
        FOCUS_SETTLEMENT_POINTS,
    )
    return [dict(row) for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------------

def run_audit(db_path: Path = DB_PATH) -> dict:
    """Run the full data audit and return structured results."""
    conn = get_connection(db_path)
    report = {}

    # 1. Table overview
    logger.info("Gathering table overview...")
    overview = audit_table_overview(conn)
    report["overview"] = overview

    # 2. Gap detection
    logger.info("Detecting gaps...")
    report["gaps"] = {}
    for table, meta in overview.items():
        if table in ("rtm_lmp_hist", "dam_lmp_hist") and meta["sp_col"]:
            gaps = detect_date_gaps_hist(conn, table, meta["time_col"], meta["sp_col"])
            report["gaps"][table] = gaps
        elif table in ("rtm_lmp_cdr", "rtm_lmp_api") and meta["sp_col"]:
            interval = 5
            gaps = detect_time_gaps_cdr(conn, table, interval)
            report["gaps"][table] = gaps
        elif table == "dam_lmp" and meta["sp_col"]:
            gaps = detect_time_gaps_cdr(conn, table, interval_minutes=60)
            report["gaps"][table] = gaps

    # 3. Data quality
    logger.info("Checking data quality...")
    report["quality"] = {}
    for table in overview:
        q = {}
        q["nulls"] = check_nulls(conn, table, ["lmp"] if table != "fuel_mix_hist" else ["generation_mw"])
        if table in ("rtm_lmp_hist", "dam_lmp_hist"):
            q["duplicate_pk_groups"] = check_duplicates_hist(conn, table)
        lmp_col = "lmp" if table != "fuel_mix_hist" else "generation_mw"
        if table != "fuel_mix_hist":
            q["anomalies"] = check_anomalies(conn, table, lmp_col)
        report["quality"][table] = q

    # 4. Coverage per settlement point
    logger.info("Building coverage summaries...")
    report["coverage"] = {}
    for table, meta in overview.items():
        if meta["sp_col"]:
            report["coverage"][table] = coverage_summary(
                conn, table, meta["time_col"], meta["sp_col"]
            )

    conn.close()
    return report


def print_report(report: dict) -> None:
    """Print the audit report in a human-readable format."""
    sep = "=" * 72

    print(f"\n{sep}")
    print("  ERCOT DATA AUDIT REPORT")
    print(f"  Generated: {datetime.now().isoformat()}")
    print(sep)

    # Overview
    print("\n--- TABLE OVERVIEW ---")
    for table, info in report["overview"].items():
        print(f"\n  {table}:")
        print(f"    Rows:  {info['rows']:>12,}")
        print(f"    Range: {info['min_time']}  →  {info['max_time']}")

    # Gaps
    print(f"\n{sep}")
    print("--- GAP DETECTION ---")
    for table, gaps in report["gaps"].items():
        print(f"\n  {table}:")
        if not gaps:
            print("    No gaps detected for focus settlement points.")
            continue
        for sp, gap_info in gaps.items():
            if isinstance(gap_info, list) and gap_info and isinstance(gap_info[0], dict):
                # CDR-style gap info
                print(f"    {sp}: {len(gap_info)} days with incomplete data")
                for g in gap_info[:5]:
                    print(f"      {g['date']}: {g['count']}/{g['expected']} intervals")
                if len(gap_info) > 5:
                    print(f"      ... and {len(gap_info) - 5} more days")
            else:
                # Historical date gaps
                print(f"    {sp}: {len(gap_info)} missing dates")
                for d in gap_info[:5]:
                    print(f"      {d}")
                if len(gap_info) > 5:
                    print(f"      ... and {len(gap_info) - 5} more dates")

    # Quality
    print(f"\n{sep}")
    print("--- DATA QUALITY ---")
    for table, q in report["quality"].items():
        print(f"\n  {table}:")
        if q.get("nulls"):
            print(f"    NULL values: {q['nulls']}")
        else:
            print("    NULL values: none")
        if "duplicate_pk_groups" in q:
            print(f"    Duplicate PK groups: {q['duplicate_pk_groups']}")
        if "anomalies" in q:
            a = q["anomalies"]
            print(f"    LMP range: [{a.get('min_lmp')}, {a.get('max_lmp')}]  avg={a.get('avg_lmp')}")
            if a.get("negative_lmp"):
                print(f"    Negative LMP records: {a['negative_lmp']:,}")
            if a.get("zero_lmp"):
                print(f"    Zero LMP records: {a['zero_lmp']:,}")
            if a.get("lmp_above_5000"):
                print(f"    LMP > $5000 records: {a['lmp_above_5000']:,}")
            if a.get("lmp_below_neg500"):
                print(f"    LMP < -$500 records: {a['lmp_below_neg500']:,}")

    # Coverage
    print(f"\n{sep}")
    print("--- COVERAGE PER SETTLEMENT POINT (focus points) ---")
    for table, rows in report["coverage"].items():
        print(f"\n  {table}:")
        if not rows:
            print("    No data for focus settlement points.")
            continue
        for r in rows:
            sp = r.get("settlement_point", "?")
            cnt = r.get("cnt", 0)
            min_t = r.get("min_t") or r.get("min_time") or "?"
            max_t = r.get("max_t") or r.get("max_time") or "?"
            print(f"    {sp:<15} {cnt:>10,} rows   {min_t} → {max_t}")

    # Check for the hist→CDR gap
    print(f"\n{sep}")
    print("--- HIST→CDR GAP ANALYSIS ---")
    ov = report["overview"]
    if "rtm_lmp_hist" in ov and "rtm_lmp_cdr" in ov:
        hist_end = ov["rtm_lmp_hist"]["max_time"]
        cdr_start = ov["rtm_lmp_cdr"]["min_time"]
        print(f"  rtm_lmp_hist ends:  {hist_end}")
        print(f"  rtm_lmp_cdr starts: {cdr_start}")
        if hist_end and cdr_start:
            h = hist_end[:10] if hist_end else "?"
            c = cdr_start[:10] if cdr_start else "?"
            print(f"  Gap: {h} → {c}")

    if "dam_lmp_hist" in ov and "dam_lmp" in ov:
        hist_end = ov["dam_lmp_hist"]["max_time"]
        cdr_start = ov["dam_lmp"]["min_time"]
        print(f"  dam_lmp_hist ends:  {hist_end}")
        print(f"  dam_lmp starts:     {cdr_start}")

    print(f"\n{sep}\n")


if __name__ == "__main__":
    report = run_audit()
    print_report(report)
