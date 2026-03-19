"""
Wind forecast data fetcher for ERCOT NP4-732-CD endpoint.

Fetches STWPF (short-term wind power forecast) vs actual generation data,
used to compute forecast error for spike prediction.
"""

import sqlite3
import time
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from loguru import logger

from scraper.src.ercot_client import ErcotClient


ENDPOINT_WIND_FORECAST = "/np4-732-cd/wpp_hrly_avrg_actl_fcast"

# Mapping from wide-format column prefixes to region names
REGION_COLUMNS = {
    "system": {
        "gen": "genSystemWide",
        "stwpf": "STWPFSystemWide",
        "wgrpp": "WGRPPSystemWide",
        "cop_hsl": "COPHSLSystemWide",
    },
    "south_houston": {
        "gen": "genLoadZoneSouthHouston",
        "stwpf": "STWPFLoadZoneSouthHouston",
        "wgrpp": "WGRPPLoadZoneSouthHouston",
        "cop_hsl": "COPHSLLoadZoneSouthHouston",
    },
    "west": {
        "gen": "genLoadZoneWest",
        "stwpf": "STWPFLoadZoneWest",
        "wgrpp": "WGRPPLoadZoneWest",
        "cop_hsl": "COPHSLLoadZoneWest",
    },
    "north": {
        "gen": "genLoadZoneNorth",
        "stwpf": "STWPFLoadZoneNorth",
        "wgrpp": "WGRPPLoadZoneNorth",
        "cop_hsl": "COPHSLLoadZoneNorth",
    },
}

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS wind_forecast (
    delivery_date TEXT NOT NULL,
    hour_ending INTEGER NOT NULL,
    region TEXT NOT NULL,
    gen_mw REAL,
    stwpf_mw REAL,
    wgrpp_mw REAL,
    cop_hsl_mw REAL,
    PRIMARY KEY (delivery_date, hour_ending, region)
);
"""


def fetch_wind_forecast(
    client: ErcotClient,
    delivery_date_from: str,
    delivery_date_to: str,
) -> pd.DataFrame:
    """Fetch wind forecast data from ERCOT API for a date range.

    Args:
        client: Authenticated ErcotClient instance.
        delivery_date_from: Start date (YYYY-MM-DD).
        delivery_date_to: End date (YYYY-MM-DD).

    Returns:
        DataFrame with all columns from the API response.
    """
    params = {
        "deliveryDateFrom": delivery_date_from,
        "deliveryDateTo": delivery_date_to,
    }

    all_records: List[Dict[str, Any]] = []
    max_retries = 3
    for attempt in range(max_retries):
        try:
            for page_records in client.fetch_paginated_data(ENDPOINT_WIND_FORECAST, params):
                all_records.extend(page_records)
                time.sleep(0.5)
            break  # success
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = 30 * (attempt + 1)
                logger.warning(f"Rate limited, waiting {wait}s before retry ({attempt + 1}/{max_retries})")
                time.sleep(wait)
                all_records.clear()
            else:
                raise

    if not all_records:
        logger.warning(f"No wind forecast data for {delivery_date_from} to {delivery_date_to}")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    logger.info(f"Fetched {len(df)} rows for {delivery_date_from} to {delivery_date_to}")
    return df


def deduplicate_latest(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the latest posted version per (deliveryDate, hourEnding).

    Multiple postedDatetime versions exist for each delivery hour.
    We want the most recent forecast/actual posting.

    Args:
        df: Raw DataFrame with postedDatetime, deliveryDate, hourEnding columns.

    Returns:
        Deduplicated DataFrame.
    """
    if df.empty:
        return df

    df = df.copy()
    df["postedDatetime"] = pd.to_datetime(df["postedDatetime"])
    df = df.sort_values("postedDatetime").groupby(
        ["deliveryDate", "hourEnding"], as_index=False
    ).last()
    return df


def pivot_to_regions(df: pd.DataFrame) -> pd.DataFrame:
    """Transform wide format (all regions in one row) to long format.

    Args:
        df: Wide-format DataFrame with per-region columns.

    Returns:
        Long-format DataFrame with columns:
        delivery_date, hour_ending, region, gen_mw, stwpf_mw, wgrpp_mw, cop_hsl_mw
    """
    if df.empty:
        return pd.DataFrame(
            columns=["delivery_date", "hour_ending", "region",
                     "gen_mw", "stwpf_mw", "wgrpp_mw", "cop_hsl_mw"]
        )

    rows = []
    for _, row in df.iterrows():
        for region, col_map in REGION_COLUMNS.items():
            rows.append({
                "delivery_date": row["deliveryDate"],
                "hour_ending": int(float(row["hourEnding"])),
                "region": region,
                "gen_mw": _to_float(row.get(col_map["gen"])),
                "stwpf_mw": _to_float(row.get(col_map["stwpf"])),
                "wgrpp_mw": _to_float(row.get(col_map["wgrpp"])),
                "cop_hsl_mw": _to_float(row.get(col_map["cop_hsl"])),
            })

    return pd.DataFrame(rows)


def save_to_sqlite(df: pd.DataFrame, db_path: Path) -> None:
    """Write wind forecast data to SQLite, upserting on primary key.

    Args:
        df: Long-format DataFrame from pivot_to_regions.
        db_path: Path to SQLite database file.
    """
    if df.empty:
        logger.warning("No data to save")
        return

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(CREATE_TABLE_SQL)
        conn.executemany(
            """
            INSERT OR REPLACE INTO wind_forecast
                (delivery_date, hour_ending, region, gen_mw, stwpf_mw, wgrpp_mw, cop_hsl_mw)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            df[["delivery_date", "hour_ending", "region",
                "gen_mw", "stwpf_mw", "wgrpp_mw", "cop_hsl_mw"]].values.tolist(),
        )
        conn.commit()
        logger.info(f"Saved {len(df)} rows to {db_path}")
    finally:
        conn.close()


def _to_float(value) -> float | None:
    """Safely convert a value to float, returning None on failure."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None
