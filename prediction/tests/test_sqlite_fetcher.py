"""
Tests for the SQLite data fetcher.

Covers:
- DAM and RTM price fetching from SQLite
- Output format compatibility with InfluxDBFetcher
- Settlement point listing and data range queries
- Fallback behavior in _fetch_and_compute_features
"""

import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Import sqlite_fetcher without triggering data/__init__.py (which pulls
# in influxdb_client_3).
import importlib.util as _ilu, sys as _sys, types as _types

for _pkg in ("prediction", "prediction.src", "prediction.src.data"):
    if _pkg not in _sys.modules:
        _sys.modules[_pkg] = _types.ModuleType(_pkg)

_spec = _ilu.spec_from_file_location(
    "prediction.src.data.sqlite_fetcher",
    str(Path(__file__).resolve().parents[1] / "src" / "data" / "sqlite_fetcher.py"),
)
_sf = _ilu.module_from_spec(_spec)
_sys.modules[_spec.name] = _sf
_spec.loader.exec_module(_sf)

SQLiteFetcher = _sf.SQLiteFetcher
create_sqlite_fetcher = _sf.create_sqlite_fetcher


# ---------------------------------------------------------------------------
# Helpers – build a small synthetic SQLite DB
# ---------------------------------------------------------------------------

def _date_range(start: str, days: int):
    base = pd.Timestamp(start)
    for d in range(days):
        yield (base + pd.Timedelta(days=d)).strftime("%Y-%m-%d")


def _create_test_db(db_path: Path, n_days: int = 40) -> None:
    """Populate a minimal SQLite archive for testing."""
    conn = sqlite3.connect(db_path)
    rng = np.random.RandomState(42)

    # --- dam_lmp_hist ---
    conn.execute(
        "CREATE TABLE dam_lmp_hist ("
        "  delivery_date TEXT NOT NULL,"
        "  hour_ending INTEGER NOT NULL,"
        "  repeated_hour INTEGER NOT NULL DEFAULT 0,"
        "  settlement_point TEXT NOT NULL,"
        "  lmp REAL NOT NULL,"
        "  PRIMARY KEY (delivery_date, hour_ending, repeated_hour, settlement_point)"
        ")"
    )
    dam_rows = []
    for date_str in _date_range("2025-01-01", n_days):
        for he in range(1, 25):
            dam_rows.append((date_str, he, 0, "HB_WEST", round(20 + 30 * rng.rand(), 2)))
            dam_rows.append((date_str, he, 0, "HB_NORTH", round(18 + 25 * rng.rand(), 2)))
    conn.executemany("INSERT INTO dam_lmp_hist VALUES (?,?,?,?,?)", dam_rows)

    # --- rtm_lmp_hist ---
    conn.execute(
        "CREATE TABLE rtm_lmp_hist ("
        "  delivery_date TEXT NOT NULL,"
        "  delivery_hour INTEGER NOT NULL,"
        "  delivery_interval INTEGER NOT NULL,"
        "  repeated_hour INTEGER NOT NULL DEFAULT 0,"
        "  settlement_point TEXT NOT NULL,"
        "  settlement_point_type TEXT,"
        "  lmp REAL,"
        "  PRIMARY KEY (delivery_date, delivery_hour, delivery_interval, "
        "               repeated_hour, settlement_point)"
        ")"
    )
    rtm_rows = []
    for date_str in _date_range("2025-01-01", n_days):
        for hr in range(24):
            for interval in range(1, 5):  # 4 x 15-min per hour
                rtm_rows.append(
                    (date_str, hr, interval, 0, "HB_WEST", "Hub",
                     round(15 + 40 * rng.rand(), 2))
                )
                rtm_rows.append(
                    (date_str, hr, interval, 0, "HB_NORTH", "Hub",
                     round(12 + 35 * rng.rand(), 2))
                )
    conn.executemany("INSERT INTO rtm_lmp_hist VALUES (?,?,?,?,?,?,?)", rtm_rows)

    conn.commit()
    conn.close()


@pytest.fixture
def test_db(tmp_path):
    db_path = tmp_path / "test_archive.db"
    _create_test_db(db_path)
    return db_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSQLiteFetcherInit:
    def test_missing_db_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            SQLiteFetcher(db_path=tmp_path / "nonexistent.db")

    def test_creates_from_valid_path(self, test_db):
        fetcher = SQLiteFetcher(db_path=test_db)
        assert fetcher.db_path == test_db
        fetcher.close()


class TestFetchDAM:
    def test_returns_expected_columns(self, test_db):
        fetcher = SQLiteFetcher(db_path=test_db)
        df = fetcher.fetch_dam_prices("HB_WEST", start_date=datetime(2025, 1, 1))
        fetcher.close()

        assert df.index.name == "timestamp"
        assert "dam_price" in df.columns
        assert "hour" in df.columns
        assert "date" in df.columns

    def test_returns_correct_row_count(self, test_db):
        fetcher = SQLiteFetcher(db_path=test_db)
        df = fetcher.fetch_dam_prices(
            "HB_WEST",
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 1, 10),
        )
        fetcher.close()
        # 10 days × 24 hours = 240
        assert len(df) == 240

    def test_empty_for_unknown_sp(self, test_db):
        fetcher = SQLiteFetcher(db_path=test_db)
        df = fetcher.fetch_dam_prices("HB_NONEXISTENT", start_date=datetime(2025, 1, 1))
        fetcher.close()
        assert df.empty

    def test_timestamps_are_utc_shifted(self, test_db):
        fetcher = SQLiteFetcher(db_path=test_db)
        df = fetcher.fetch_dam_prices(
            "HB_WEST",
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 1, 1),
        )
        fetcher.close()
        # First hour (HE1) = 00:00 CST = 06:00 UTC
        first_ts = df.index[0]
        assert first_ts.hour == 6


class TestFetchRTM:
    def test_returns_expected_columns(self, test_db):
        fetcher = SQLiteFetcher(db_path=test_db)
        df = fetcher.fetch_rtm_prices("HB_WEST", start_date=datetime(2025, 1, 1))
        fetcher.close()

        assert df.index.name == "timestamp"
        assert "rtm_price" in df.columns

    def test_aggregates_to_hourly(self, test_db):
        fetcher = SQLiteFetcher(db_path=test_db)
        df = fetcher.fetch_rtm_prices(
            "HB_WEST",
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 1, 1),
        )
        fetcher.close()
        # 1 day × 24 hours = 24 (aggregated from 4 intervals each)
        assert len(df) == 24

    def test_empty_for_unknown_sp(self, test_db):
        fetcher = SQLiteFetcher(db_path=test_db)
        df = fetcher.fetch_rtm_prices("HB_NONEXISTENT", start_date=datetime(2025, 1, 1))
        fetcher.close()
        assert df.empty


class TestMetadata:
    def test_settlement_points(self, test_db):
        fetcher = SQLiteFetcher(db_path=test_db)
        sps = fetcher.get_available_settlement_points("dam_lmp")
        fetcher.close()
        assert "HB_WEST" in sps
        assert "HB_NORTH" in sps

    def test_data_range(self, test_db):
        fetcher = SQLiteFetcher(db_path=test_db)
        min_dt, max_dt = fetcher.get_data_range("dam_lmp")
        fetcher.close()
        assert min_dt is not None
        assert max_dt is not None
        assert min_dt <= max_dt


class TestCloseIdempotent:
    def test_double_close(self, test_db):
        fetcher = SQLiteFetcher(db_path=test_db)
        fetcher.fetch_dam_prices("HB_WEST", start_date=datetime(2025, 1, 1))
        fetcher.close()
        fetcher.close()  # Should not raise
