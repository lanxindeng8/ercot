"""
SQLite Data Fetcher

Fetches historical DAM and RTM price data from the SQLite archive database.
Mirrors the InfluxDBFetcher interface so it can be used as a drop-in replacement.
"""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

DEFAULT_DB = Path(__file__).resolve().parents[3] / "scraper" / "data" / "ercot_archive.db"
READ_TIMEOUT_SECONDS = 30.0


class SQLiteFetcher:
    """Fetches ERCOT price data from the SQLite archive database."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB
        if not self.db_path.exists():
            raise FileNotFoundError(f"SQLite database not found: {self.db_path}")
        self._conn: Optional[sqlite3.Connection] = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                f"file:{self.db_path}?mode=ro",
                uri=True,
                timeout=READ_TIMEOUT_SECONDS,
                check_same_thread=False,
            )
            self._conn.execute(f"PRAGMA busy_timeout = {int(READ_TIMEOUT_SECONDS * 1000)}")
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

        if not df.empty:
            # Convert to InfluxDB-compatible format
            df["timestamp"] = pd.to_datetime(df["delivery_date"]) + pd.to_timedelta(
                df["hour_ending"] - 1, unit="h"
            ) + pd.Timedelta(hours=6)
            df = df.rename(columns={"lmp": "dam_price", "hour_ending": "hour"})
            df["date"] = pd.to_datetime(df["delivery_date"]).dt.date
            df = df[["timestamp", "hour", "dam_price", "date"]].set_index("timestamp")

        # CDR data (real-time, includes next-day prices)
        try:
            cdr_query = (
                "SELECT time, hour_ending, lmp FROM dam_lmp_cdr "
                "WHERE settlement_point = ? AND oper_day >= ? "
                "ORDER BY oper_day, hour_ending"
            )
            df_cdr = pd.read_sql_query(cdr_query, conn, params=[settlement_point, date_from])
        except Exception:
            df_cdr = pd.DataFrame()

        if not df_cdr.empty:
            df_cdr["timestamp"] = pd.to_datetime(df_cdr["time"])
            df_cdr = df_cdr.rename(columns={"lmp": "dam_price", "hour_ending": "hour"})
            df_cdr["date"] = df_cdr["timestamp"].dt.date
            df_cdr = df_cdr[["timestamp", "hour", "dam_price", "date"]].set_index("timestamp")
            df_cdr = df_cdr[~df_cdr.index.duplicated(keep="last")]

        # Merge: prefer CDR for overlapping timestamps
        if df.empty and df_cdr.empty:
            return pd.DataFrame(columns=["timestamp", "hour", "dam_price"])
        elif df.empty:
            return df_cdr
        elif df_cdr.empty:
            return df
        else:
            combined = pd.concat([df, df_cdr])
            combined = combined[~combined.index.duplicated(keep="last")]
            return combined.sort_index()

    # -----------------------------------------------------------------
    # RTM prices
    # -----------------------------------------------------------------

    def fetch_rtm_prices(
        self,
        settlement_point: str = "LZ_HOUSTON",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Fetch RTM LMP prices from SQLite (hist + CDR for freshness).

        Returns DataFrame with columns: rtm_price
        (indexed by timestamp) — same format as InfluxDBFetcher.
        """
        if start_date is None:
            start_date = datetime.utcnow() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.utcnow()

        date_from = start_date.strftime("%Y-%m-%d")
        date_to = (end_date + timedelta(days=1)).strftime("%Y-%m-%d")
        conn = self._get_conn()

        # Historical data (hourly aggregated)
        hist_query = (
            "SELECT delivery_date, delivery_hour, AVG(lmp) AS lmp "
            "FROM rtm_lmp_hist "
            "WHERE settlement_point = ? AND repeated_hour = 0 "
            "AND delivery_date >= ? AND delivery_date < ? "
            "GROUP BY delivery_date, delivery_hour "
            "ORDER BY delivery_date, delivery_hour"
        )
        df_hist = pd.read_sql_query(hist_query, conn, params=[settlement_point, date_from, date_to])

        if not df_hist.empty:
            df_hist["timestamp"] = pd.to_datetime(df_hist["delivery_date"]) + pd.to_timedelta(
                df_hist["delivery_hour"], unit="h"
            ) + pd.Timedelta(hours=6)
            df_hist = df_hist.rename(columns={"lmp": "rtm_price"})
            df_hist["hour"] = df_hist["delivery_hour"] + 1
            df_hist["date"] = pd.to_datetime(df_hist["delivery_date"]).dt.date
            df_hist = df_hist[["timestamp", "hour", "date", "rtm_price"]].set_index("timestamp")

        # CDR data (real-time, may overlap with hist — CDR wins for recent data)
        try:
            cdr_query = (
                "SELECT time, lmp FROM rtm_lmp_cdr "
                "WHERE settlement_point = ? AND time >= ? "
                "ORDER BY time"
            )
            iso_from = start_date.strftime("%Y-%m-%dT%H:%M:%S")
            df_cdr = pd.read_sql_query(cdr_query, conn, params=[settlement_point, iso_from])
        except Exception:
            df_cdr = pd.DataFrame()

        if not df_cdr.empty:
            df_cdr["timestamp"] = pd.to_datetime(df_cdr["time"])
            # Aggregate to hourly
            df_cdr = df_cdr.set_index("timestamp").resample("1h").agg({"lmp": "mean"}).dropna()
            df_cdr = df_cdr.rename(columns={"lmp": "rtm_price"})
            df_cdr["hour"] = df_cdr.index.hour + 1
            df_cdr["date"] = df_cdr.index.date

        # Merge: prefer CDR for overlapping timestamps
        if df_hist.empty and df_cdr.empty:
            return pd.DataFrame()
        elif df_hist.empty:
            return df_cdr
        elif df_cdr.empty:
            return df_hist
        else:
            combined = pd.concat([df_hist, df_cdr])
            combined = combined[~combined.index.duplicated(keep="last")]
            return combined.sort_index()

    # -----------------------------------------------------------------
    # Metadata
    # -----------------------------------------------------------------

    def get_available_settlement_points(self, measurement: str = "dam_lmp") -> List[str]:
        """Get list of available settlement points."""
        table = self._resolve_table_name(measurement)
        query = f"SELECT DISTINCT settlement_point FROM {table} ORDER BY settlement_point"
        conn = self._get_conn()
        df = pd.read_sql_query(query, conn)
        return df["settlement_point"].tolist() if not df.empty else []

    def get_data_range(self, measurement: str = "dam_lmp") -> Tuple[Optional[datetime], Optional[datetime]]:
        """Get the time range of available data."""
        table = self._resolve_table_name(measurement)
        query = f"SELECT MIN(delivery_date) as min_date, MAX(delivery_date) as max_date FROM {table}"
        conn = self._get_conn()
        df = pd.read_sql_query(query, conn)
        if not df.empty and df["min_date"].iloc[0] is not None:
            return (
                pd.to_datetime(df["min_date"].iloc[0]),
                pd.to_datetime(df["max_date"].iloc[0]),
            )
        return None, None

    @staticmethod
    def _resolve_table_name(measurement: str) -> str:
        measurement_key = measurement.strip().lower()
        table_map = {
            "dam": "dam_lmp_hist",
            "dam_lmp": "dam_lmp_hist",
            "dam_lmp_hist": "dam_lmp_hist",
            "rtm": "rtm_lmp_hist",
            "rtm_lmp": "rtm_lmp_hist",
            "rtm_lmp_hist": "rtm_lmp_hist",
        }
        try:
            return table_map[measurement_key]
        except KeyError as exc:
            raise ValueError(f"Unsupported measurement: {measurement}") from exc


def create_sqlite_fetcher(db_path: Optional[Path] = None) -> SQLiteFetcher:
    """Create a SQLite fetcher, optionally with a custom DB path."""
    return SQLiteFetcher(db_path=db_path)
