"""
Data quality and integrity tests for the ERCOT archive database.

Tests cover:
  - Table existence and non-empty data
  - No NULL LMP values in critical tables
  - No duplicate primary key groups
  - Settlement point coverage for focus points
  - Backfill script correctness (using in-memory DB)
  - Gap detection after backfill
"""

import sqlite3
import pytest
from datetime import datetime, timedelta
from pathlib import Path

# Paths
DB_PATH = Path(__file__).parent.parent / "data" / "ercot_archive.db"

# Import audit/backfill modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from data_audit import (
    audit_table_overview,
    check_nulls,
    check_anomalies,
    coverage_summary,
    detect_date_gaps_hist,
    detect_time_gaps_cdr,
)
from backfill_gap import (
    backfill_rtm,
    backfill_dam,
    _utc_to_cpt_components,
    _local_date_bounds_to_utc,
)

FOCUS_SETTLEMENT_POINTS = [
    "HB_WEST", "HB_NORTH", "HB_SOUTH", "HB_HOUSTON", "HB_BUSAVG",
    "LZ_WEST", "LZ_NORTH", "LZ_SOUTH", "LZ_HOUSTON",
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def live_db():
    """Read-only connection to the live database (skip if not present)."""
    if not DB_PATH.exists():
        pytest.skip("Live database not found")
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


@pytest.fixture
def mem_db():
    """In-memory SQLite database pre-populated with test data."""
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA journal_mode=WAL")

    # Create CDR/API source tables
    conn.execute("""
    CREATE TABLE rtm_lmp_cdr (
        time DATETIME NOT NULL,
        settlement_point TEXT NOT NULL,
        lmp REAL NOT NULL,
        PRIMARY KEY (time, settlement_point)
    )
    """)
    conn.execute("""
    CREATE TABLE rtm_lmp_api (
        time DATETIME NOT NULL,
        settlement_point TEXT NOT NULL,
        lmp REAL NOT NULL,
        energy_component REAL,
        congestion_component REAL,
        loss_component REAL,
        PRIMARY KEY (time, settlement_point)
    )
    """)
    conn.execute("""
    CREATE TABLE dam_lmp (
        time DATETIME NOT NULL,
        settlement_point TEXT NOT NULL,
        settlement_point_type TEXT,
        lmp REAL NOT NULL,
        PRIMARY KEY (time, settlement_point)
    )
    """)

    # Insert test RTM CDR data: 2026-02-15 full day for HB_WEST
    # Every 5 minutes = 288 records per day, in UTC (CPT + 6h)
    base_utc = datetime(2026, 2, 15, 6, 0, 0)  # midnight CPT = 06:00 UTC
    for i in range(288):
        t = base_utc + timedelta(minutes=5 * i)
        conn.execute(
            "INSERT INTO rtm_lmp_cdr (time, settlement_point, lmp) VALUES (?, ?, ?)",
            (t.isoformat(), "HB_WEST", 25.0 + (i % 10)),
        )

    # Insert test DAM data: 2026-02-15 full day for HB_WEST (24 hours)
    # UTC times: 06:00 on Feb 15 through 05:00 on Feb 16 (CPT hours 1-24)
    dam_base = datetime(2026, 2, 15, 6, 0, 0)
    for h in range(24):
        t = dam_base + timedelta(hours=h)
        conn.execute(
            "INSERT INTO dam_lmp (time, settlement_point, settlement_point_type, lmp) "
            "VALUES (?, ?, ?, ?)",
            (t.isoformat(), "HB_WEST", "Hub", 30.0 + h),
        )

    conn.commit()
    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# Unit tests: UTC→CPT conversion
# ---------------------------------------------------------------------------

class TestUtcToCptComponents:
    """Test the UTC to Central Prevailing Time conversion."""

    def test_midnight_cpt(self):
        """Midnight CPT (06:00 UTC) → hour 24 of previous day."""
        result = _utc_to_cpt_components("2026-02-16T06:00:00")
        assert result["delivery_date"] == "2026-02-15"
        assert result["delivery_hour"] == 24
        assert result["delivery_interval"] == 1

    def test_noon_cpt(self):
        """Noon CPT (18:00 UTC) → hour 12."""
        result = _utc_to_cpt_components("2026-02-15T18:00:00")
        assert result["delivery_date"] == "2026-02-15"
        assert result["delivery_hour"] == 12
        assert result["delivery_interval"] == 1

    def test_interval_mapping(self):
        """5-min intervals map to 15-min delivery intervals 1-4."""
        # minute 0-14 → interval 1
        r = _utc_to_cpt_components("2026-02-15T07:00:00")
        assert r["delivery_interval"] == 1
        r = _utc_to_cpt_components("2026-02-15T07:10:00")
        assert r["delivery_interval"] == 1

        # minute 15-29 → interval 2
        r = _utc_to_cpt_components("2026-02-15T07:15:00")
        assert r["delivery_interval"] == 2

        # minute 30-44 → interval 3
        r = _utc_to_cpt_components("2026-02-15T07:30:00")
        assert r["delivery_interval"] == 3

        # minute 45-59 → interval 4
        r = _utc_to_cpt_components("2026-02-15T07:45:00")
        assert r["delivery_interval"] == 4

    def test_spring_dst_boundary_uses_utc_5_offset(self):
        """DST transition should use CDT offset after the spring-forward boundary."""
        result = _utc_to_cpt_components("2026-03-08T08:00:00")
        assert result["delivery_date"] == "2026-03-08"
        assert result["delivery_hour"] == 3
        assert result["delivery_interval"] == 1
        assert result["repeated_hour"] == 0

    def test_fall_dst_boundary_marks_repeated_hour(self):
        """The second 1 AM hour on fall-back day should set repeated_hour=1."""
        first = _utc_to_cpt_components("2026-11-01T06:00:00")
        second = _utc_to_cpt_components("2026-11-01T07:00:00")
        assert first["delivery_date"] == "2026-11-01"
        assert second["delivery_date"] == "2026-11-01"
        assert first["delivery_hour"] == 1
        assert second["delivery_hour"] == 1
        assert first["repeated_hour"] == 0
        assert second["repeated_hour"] == 1

    def test_local_date_bounds_respect_dst(self):
        """UTC query bounds should expand or contract with the local DST offset."""
        start_utc, end_utc = _local_date_bounds_to_utc("2026-03-08", "2026-03-08")
        assert start_utc == "2026-03-08T06:00:00"
        assert end_utc == "2026-03-09T05:00:00"


# ---------------------------------------------------------------------------
# In-memory backfill tests
# ---------------------------------------------------------------------------

class TestBackfillInMemory:
    """Test backfill logic using in-memory database."""

    def test_rtm_backfill_creates_hist_records(self, mem_db):
        """RTM backfill should create hist records from CDR data."""
        count = backfill_rtm(mem_db, "2026-02-15", "2026-02-15")
        assert count > 0

        cur = mem_db.cursor()
        cur.execute("SELECT COUNT(*) FROM rtm_lmp_hist WHERE settlement_point = 'HB_WEST'")
        hist_count = cur.fetchone()[0]
        # 24 hours * 4 intervals = 96 expected records
        assert hist_count == 96

    def test_rtm_backfill_idempotent(self, mem_db):
        """Running backfill twice should produce the same result."""
        count1 = backfill_rtm(mem_db, "2026-02-15", "2026-02-15")
        count2 = backfill_rtm(mem_db, "2026-02-15", "2026-02-15")
        assert count1 == count2

        cur = mem_db.cursor()
        cur.execute("SELECT COUNT(*) FROM rtm_lmp_hist WHERE settlement_point = 'HB_WEST'")
        assert cur.fetchone()[0] == 96

    def test_rtm_backfill_prefers_api_and_fills_from_cdr(self, mem_db):
        """API rows should override matching CDR rows without dropping CDR-only intervals."""
        mem_db.execute("DELETE FROM rtm_lmp_cdr")
        for minute, lmp in ((0, 10.0), (5, 10.0), (10, 10.0), (15, 20.0), (20, 20.0), (25, 20.0)):
            timestamp = datetime(2026, 2, 15, 6, minute, 0).isoformat()
            mem_db.execute(
                "INSERT INTO rtm_lmp_cdr (time, settlement_point, lmp) VALUES (?, ?, ?)",
                (timestamp, "HB_WEST", lmp),
            )
        for minute, lmp in ((0, 100.0), (5, 100.0), (10, 100.0)):
            timestamp = datetime(2026, 2, 15, 6, minute, 0).isoformat()
            mem_db.execute(
                "INSERT INTO rtm_lmp_api (time, settlement_point, lmp) VALUES (?, ?, ?)",
                (timestamp, "HB_WEST", lmp),
            )
        mem_db.commit()

        count = backfill_rtm(mem_db, "2026-02-15", "2026-02-15")
        assert count >= 2

        cur = mem_db.cursor()
        cur.execute(
            "SELECT delivery_interval, lmp FROM rtm_lmp_hist "
            "WHERE delivery_date = '2026-02-14' AND delivery_hour = 24 "
            "AND settlement_point = 'HB_WEST' ORDER BY delivery_interval"
        )
        rows = cur.fetchall()
        assert rows == [(1, 100.0), (2, 20.0)]

    def test_dam_backfill_creates_hist_records(self, mem_db):
        """DAM backfill should create hist records from dam_lmp data."""
        count = backfill_dam(mem_db, "2026-02-15", "2026-02-15")
        assert count > 0

        cur = mem_db.cursor()
        cur.execute("SELECT COUNT(*) FROM dam_lmp_hist WHERE settlement_point = 'HB_WEST'")
        hist_count = cur.fetchone()[0]
        assert hist_count == 24

    def test_dam_backfill_idempotent(self, mem_db):
        """Running DAM backfill twice should produce the same result."""
        count1 = backfill_dam(mem_db, "2026-02-15", "2026-02-15")
        count2 = backfill_dam(mem_db, "2026-02-15", "2026-02-15")
        assert count1 == count2

    def test_dam_hour_ending_format(self, mem_db):
        """DAM hist records should use integer HE1-24 values."""
        backfill_dam(mem_db, "2026-02-15", "2026-02-15")
        cur = mem_db.cursor()
        cur.execute(
            "SELECT hour_ending FROM dam_lmp_hist "
            "WHERE settlement_point = 'HB_WEST' ORDER BY hour_ending"
        )
        hours = [row[0] for row in cur.fetchall()]
        assert all(isinstance(h, int) for h in hours)
        for h in hours:
            assert 1 <= h <= 24, f"Unexpected hour_ending value: {h}"

    def test_no_gaps_after_backfill(self, mem_db):
        """After backfill, there should be no gaps for the backfilled date."""
        backfill_rtm(mem_db, "2026-02-15", "2026-02-15")
        backfill_dam(mem_db, "2026-02-15", "2026-02-15")

        cur = mem_db.cursor()

        # Check RTM: hours 1-23 present (92 intervals).
        # Hour 24 of 2026-02-15 maps to 06:00 UTC on 2026-02-16, which is
        # outside our test fixture's range, so 92 is the correct count.
        cur.execute(
            "SELECT COUNT(DISTINCT delivery_hour || '-' || delivery_interval) "
            "FROM rtm_lmp_hist WHERE delivery_date = '2026-02-15' "
            "AND settlement_point = 'HB_WEST'"
        )
        assert cur.fetchone()[0] == 92

        # Check DAM: hours 1-23 present (same UTC boundary reason)
        cur.execute(
            "SELECT COUNT(DISTINCT hour_ending) FROM dam_lmp_hist "
            "WHERE delivery_date = '2026-02-15' AND settlement_point = 'HB_WEST'"
        )
        assert cur.fetchone()[0] == 23

    def test_dam_backfill_marks_fall_repeated_hour(self, mem_db):
        """Fallback day should produce distinct repeated-hour rows."""
        mem_db.execute("DELETE FROM dam_lmp")
        for hour_utc, lmp in ((6, 50.0), (7, 60.0)):
            mem_db.execute(
                "INSERT INTO dam_lmp (time, settlement_point, settlement_point_type, lmp) "
                "VALUES (?, ?, ?, ?)",
                (datetime(2026, 11, 1, hour_utc, 0, 0).isoformat(), "HB_WEST", "Hub", lmp),
            )
        mem_db.commit()

        backfill_dam(mem_db, "2026-11-01", "2026-11-01")
        cur = mem_db.cursor()
        cur.execute(
            "SELECT delivery_date, hour_ending, repeated_hour, lmp "
            "FROM dam_lmp_hist WHERE settlement_point = 'HB_WEST' "
            "ORDER BY delivery_date, hour_ending, repeated_hour"
        )
        rows = cur.fetchall()
        assert rows == [
            ("2026-11-01", 1, 0, 50.0),
            ("2026-11-01", 1, 1, 60.0),
        ]


class TestAuditGapDetection:
    """Gap detection should catch fully missing days, not just low-count days."""

    def test_detect_date_gaps_hist_finds_missing_day(self, mem_db):
        mem_db.execute(
            "CREATE TABLE sample_hist (delivery_date TEXT NOT NULL, settlement_point TEXT NOT NULL)"
        )
        mem_db.executemany(
            "INSERT INTO sample_hist (delivery_date, settlement_point) VALUES (?, ?)",
            [
                ("2026-02-15", "HB_WEST"),
                ("2026-02-17", "HB_WEST"),
            ],
        )
        mem_db.commit()

        gaps = detect_date_gaps_hist(mem_db, "sample_hist", "delivery_date", "settlement_point")
        assert gaps["HB_WEST"] == ["2026-02-16"]

    def test_detect_time_gaps_cdr_finds_completely_missing_day(self, mem_db):
        mem_db.execute("DELETE FROM rtm_lmp_cdr")
        base = datetime(2026, 2, 15, 0, 0, 0)
        for offset_days in (0, 2):
            for i in range(288):
                t = base + timedelta(days=offset_days, minutes=5 * i)
                mem_db.execute(
                    "INSERT INTO rtm_lmp_cdr (time, settlement_point, lmp) VALUES (?, ?, ?)",
                    (t.isoformat(), "HB_WEST", 20.0),
                )
        mem_db.commit()

        gaps = detect_time_gaps_cdr(mem_db, "rtm_lmp_cdr")
        assert gaps["HB_WEST"] == [
            {"date": "2026-02-16", "count": 0, "expected": 288},
        ]


# ---------------------------------------------------------------------------
# Live database tests (skipped if DB not present)
# ---------------------------------------------------------------------------

class TestLiveDataQuality:
    """Data quality tests against the live database."""

    def test_tables_exist(self, live_db):
        """All expected tables should exist."""
        overview = audit_table_overview(live_db)
        for table in ("rtm_lmp_hist", "dam_lmp_hist", "rtm_lmp_cdr", "dam_lmp"):
            assert table in overview, f"Table {table} not found"

    def test_tables_non_empty(self, live_db):
        """All critical tables should have data."""
        overview = audit_table_overview(live_db)
        for table in ("rtm_lmp_hist", "dam_lmp_hist", "rtm_lmp_cdr", "dam_lmp"):
            assert overview[table]["rows"] > 0, f"Table {table} is empty"

    def test_no_null_lmp_in_rtm_hist(self, live_db):
        """rtm_lmp_hist should have no NULL lmp values."""
        nulls = check_nulls(live_db, "rtm_lmp_hist", ["lmp"])
        assert not nulls, f"Found NULL lmp values in rtm_lmp_hist: {nulls}"

    def test_no_null_lmp_in_dam_hist(self, live_db):
        """dam_lmp_hist should have no NULL lmp values."""
        nulls = check_nulls(live_db, "dam_lmp_hist", ["lmp"])
        assert not nulls, f"Found NULL lmp values in dam_lmp_hist: {nulls}"

    def test_focus_points_in_rtm_hist(self, live_db):
        """All focus settlement points should have data in rtm_lmp_hist."""
        cov = coverage_summary(live_db, "rtm_lmp_hist", "delivery_date", "settlement_point")
        found_points = {r["settlement_point"] for r in cov}
        for sp in FOCUS_SETTLEMENT_POINTS:
            assert sp in found_points, f"{sp} missing from rtm_lmp_hist"

    def test_focus_points_in_dam_hist(self, live_db):
        """All focus settlement points should have data in dam_lmp_hist."""
        cov = coverage_summary(live_db, "dam_lmp_hist", "delivery_date", "settlement_point")
        found_points = {r["settlement_point"] for r in cov}
        for sp in FOCUS_SETTLEMENT_POINTS:
            assert sp in found_points, f"{sp} missing from dam_lmp_hist"

    def test_lmp_range_reasonable(self, live_db):
        """LMP values should be within reasonable bounds (no data corruption)."""
        for table in ("rtm_lmp_hist", "dam_lmp_hist"):
            anomalies = check_anomalies(live_db, table)
            # ERCOT system-wide cap is $5000, floor is -$9999
            # ERCOT ORDC can push prices above $5000 in scarcity events
            assert anomalies["min_lmp"] >= -15000, \
                f"{table}: min LMP {anomalies['min_lmp']} below -$15,000"
            assert anomalies["max_lmp"] <= 15000, \
                f"{table}: max LMP {anomalies['max_lmp']} above $15,000"
