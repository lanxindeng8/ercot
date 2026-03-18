"""
SQLite Data Fetcher

Fetches historical DAM and RTM price data from the SQLite archive database.
Mirrors the InfluxDBFetcher interface so it can be used as a drop-in replacement.
"""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

DEFAULT_DB = Path(__file__).resolve().parents[3] / "scraper" / "data" / "ercot_archive.db"


class SQLiteFetcher:
    """Fetches ERCOT price data from the SQLite archive database."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB
        if not self.db_path.exists():
            raise FileNotFoundError(f"SQLite database not found: {self.db_path}")
        self._conn: Optional[sqlite3.Connection] = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
        return self._conn

    def close(self):
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # -----------------------------------------------------------------
    # DAM prices
    # -----------------------------------------------------------------

    def fetch_dam_prices(
        self,
        settlement_point: str = "LZ_HOUSTON",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Fetch DAM LMP prices from SQLite.

        Returns DataFrame with columns: timestamp, hour, dam_price, date
        (indexed by timestamp) — same format as InfluxDBFetcher.
        """
        if start_date is None:
            start_date = datetime(2015, 1, 1)
        if end_date is None:
            end_date = datetime.utcnow()

        date_from = start_date.strftime("%Y-%m-%d")
        date_to = (end_date + timedelta(days=1)).strftime("%Y-%m-%d")

        query = (
            "SELECT delivery_date, hour_ending, lmp "
            "FROM dam_lmp_hist "
            "WHERE settlement_point = ? AND repeated_hour = 0 "
            "AND delivery_date >= ? AND delivery_date < ? "
            "ORDER BY delivery_date, hour_ending"
        )

        conn = self._get_conn()
        df = pd.read_sql_query(query, conn, params=[settlement_point, date_from, date_to])

        if df.empty:
            return pd.DataFrame(columns=["timestamp", "hour", "dam_price"])

        # Convert to InfluxDB-compatible format
        # delivery_date + (hour_ending - 1) → Central Time, then +6h to UTC
        df["timestamp"] = pd.to_datetime(df["delivery_date"]) + pd.to_timedelta(
            df["hour_ending"] - 1, unit="h"
        ) + pd.Timedelta(hours=6)
        df = df.rename(columns={"lmp": "dam_price", "hour_ending": "hour"})
        df["date"] = pd.to_datetime(df["delivery_date"]).dt.date
        df = df[["timestamp", "hour", "dam_price", "date"]].set_index("timestamp")
        return df

    # -----------------------------------------------------------------
    # RTM prices
    # -----------------------------------------------------------------

    def fetch_rtm_prices(
        self,
        settlement_point: str = "LZ_HOUSTON",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Fetch RTM LMP prices from SQLite.

        Returns DataFrame with columns: rtm_price
        (indexed by timestamp) — same format as InfluxDBFetcher.
        """
        if start_date is None:
            start_date = datetime.utcnow() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.utcnow()

        date_from = start_date.strftime("%Y-%m-%d")
        date_to = (end_date + timedelta(days=1)).strftime("%Y-%m-%d")

        query = (
            "SELECT delivery_date, delivery_hour, AVG(lmp) AS lmp "
            "FROM rtm_lmp_hist "
            "WHERE settlement_point = ? AND repeated_hour = 0 "
            "AND delivery_date >= ? AND delivery_date < ? "
            "GROUP BY delivery_date, delivery_hour "
            "ORDER BY delivery_date, delivery_hour"
        )

        conn = self._get_conn()
        df = pd.read_sql_query(query, conn, params=[settlement_point, date_from, date_to])

        if df.empty:
            return pd.DataFrame()

        # Convert to InfluxDB-compatible format
        # delivery_hour is 0-23, so timestamp = date + hour * 1h + 6h (CST→UTC)
        df["timestamp"] = pd.to_datetime(df["delivery_date"]) + pd.to_timedelta(
            df["delivery_hour"], unit="h"
        ) + pd.Timedelta(hours=6)
        df = df.rename(columns={"lmp": "rtm_price"})
        df = df[["timestamp", "rtm_price"]].set_index("timestamp")
        return df

    # -----------------------------------------------------------------
    # Metadata
    # -----------------------------------------------------------------

    def get_available_settlement_points(self, measurement: str = "dam_lmp") -> List[str]:
        """Get list of available settlement points."""
        table = "dam_lmp_hist" if "dam" in measurement else "rtm_lmp_hist"
        query = f"SELECT DISTINCT settlement_point FROM {table} ORDER BY settlement_point"
        conn = self._get_conn()
        df = pd.read_sql_query(query, conn)
        return df["settlement_point"].tolist() if not df.empty else []

    def get_data_range(self, measurement: str = "dam_lmp") -> Tuple[Optional[datetime], Optional[datetime]]:
        """Get the time range of available data."""
        table = "dam_lmp_hist" if "dam" in measurement else "rtm_lmp_hist"
        query = f"SELECT MIN(delivery_date) as min_date, MAX(delivery_date) as max_date FROM {table}"
        conn = self._get_conn()
        df = pd.read_sql_query(query, conn)
        if not df.empty and df["min_date"].iloc[0] is not None:
            return (
                pd.to_datetime(df["min_date"].iloc[0]),
                pd.to_datetime(df["max_date"].iloc[0]),
            )
        return None, None


def create_sqlite_fetcher(db_path: Optional[Path] = None) -> SQLiteFetcher:
    """Create a SQLite fetcher, optionally with a custom DB path."""
    return SQLiteFetcher(db_path=db_path)
