"""
Open-Meteo Archive API client for fetching hourly weather data.

Fetches historical weather for ERCOT weather stations and stores
results in a SQLite table (weather_hourly).
"""

import sqlite3
import time
from pathlib import Path

import pandas as pd
import requests
from loguru import logger

from prediction.src.data.weather.stations import WEATHER_STATIONS

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARIABLES = (
    "temperature_2m,wind_speed_10m,wind_direction_10m,"
    "relative_humidity_2m,surface_pressure,dew_point_2m"
)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS weather_hourly (
    station TEXT NOT NULL,
    time TEXT NOT NULL,  -- always stored as UTC ISO 8601
    temperature_2m REAL,
    wind_speed_10m REAL,
    wind_direction_10m REAL,
    relative_humidity_2m REAL,
    surface_pressure REAL,
    dew_point_2m REAL,
    PRIMARY KEY (station, time)
)
"""


def fetch_station_year(station: str, year: int) -> pd.DataFrame:
    """Fetch one station's hourly weather for a single calendar year.

    Returns a DataFrame with columns: station, time, and the 6 weather variables.
    """
    info = WEATHER_STATIONS[station]

    today = pd.Timestamp.now().normalize()
    start_date = f"{year}-01-01"
    end_date = min(pd.Timestamp(f"{year}-12-31"), today - pd.Timedelta(days=1))
    end_date = end_date.strftime("%Y-%m-%d")

    if start_date > end_date:
        logger.warning(f"Skipping {station} {year}: start_date {start_date} > end_date {end_date}")
        return pd.DataFrame()

    params = {
        "latitude": info["lat"],
        "longitude": info["lon"],
        "start_date": start_date,
        "end_date": end_date,
        "hourly": HOURLY_VARIABLES,
        "timezone": "UTC",
    }

    resp = requests.get(ARCHIVE_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()  # noqa: may raise on non-JSON response

    hourly = data.get("hourly", {})
    if not hourly or "time" not in hourly:
        logger.warning(f"No hourly data returned for {station} {year}")
        return pd.DataFrame()

    df = pd.DataFrame(hourly)
    df.insert(0, "station", station)
    return df


def fetch_all_stations(
    start_year: int = 2015,
    end_year: int = 2026,
    db_path: "Path | None" = None,
) -> pd.DataFrame:
    """Fetch all weather stations across a range of years.

    Sleeps 1s between requests to respect Open-Meteo rate limits.
    If *db_path* is provided, saves incrementally after each station.
    Returns a single concatenated DataFrame.
    """
    frames = []
    for station in WEATHER_STATIONS:
        station_frames = []
        for year in range(start_year, end_year + 1):
            logger.info(f"Fetching {station} {year}...")
            try:
                df = fetch_station_year(station, year)
            except Exception:
                logger.warning(f"Failed to fetch {station} {year}, skipping", exc_info=True)
                continue
            if not df.empty:
                logger.info(f"  done ({len(df)} rows)")
                station_frames.append(df)
            time.sleep(1.0)

        if station_frames:
            station_df = pd.concat(station_frames, ignore_index=True)
            frames.append(station_df)
            if db_path is not None:
                save_to_sqlite(station_df, db_path)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def save_to_sqlite(df: pd.DataFrame, db_path: Path) -> None:
    """Write weather data to the weather_hourly table (INSERT OR REPLACE)."""
    if df.empty:
        logger.warning("Empty DataFrame, nothing to save")
        return

    db_path = Path(db_path)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(CREATE_TABLE_SQL)

        cols = [
            "station", "time", "temperature_2m", "wind_speed_10m",
            "wind_direction_10m", "relative_humidity_2m", "surface_pressure",
            "dew_point_2m",
        ]
        placeholders = ", ".join(["?"] * len(cols))
        col_names = ", ".join(cols)
        update_cols = [c for c in cols if c not in ("station", "time")]
        update_clause = ", ".join(f"{c}=excluded.{c}" for c in update_cols)
        sql = (
            f"INSERT INTO weather_hourly ({col_names}) VALUES ({placeholders}) "
            f"ON CONFLICT(station, time) DO UPDATE SET {update_clause}"
        )

        rows = df[cols].values.tolist()
        conn.executemany(sql, rows)
        conn.commit()
        logger.info(f"Saved {len(rows)} rows to {db_path}:weather_hourly")
    finally:
        conn.close()
