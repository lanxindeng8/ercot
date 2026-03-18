#!/usr/bin/env python
"""
Build Load Forecast Features from SQLite

Extracts total system load from the ERCOT archive database (sum of all
fuel types in fuel_mix_hist) and builds training features:
  - Temporal/categorical features (hour, day_of_week, month, etc.)
  - Lag features (1h–168h)
  - Rolling statistics (6h, 12h, 24h, 168h)

Outputs features_augmented.parquet for model training.
"""

from pathlib import Path
import sys

from loguru import logger
import numpy as np
import pandas as pd
import sqlite3


# ---------------------------------------------------------------------------
# ERCOT zone weather coordinates & load weights (from notebook analysis)
# ---------------------------------------------------------------------------
ERCOT_ZONES = {
    "north_central": {"lat": 32.78, "lon": -96.80, "weight": 0.31},
    "coast":         {"lat": 29.76, "lon": -95.37, "weight": 0.28},
    "south_central": {"lat": 29.42, "lon": -98.49, "weight": 0.16},
    "far_west":      {"lat": 31.76, "lon": -106.49, "weight": 0.08},
    "south":         {"lat": 26.20, "lon": -98.23, "weight": 0.08},
    "east":          {"lat": 32.35, "lon": -94.71, "weight": 0.04},
    "west":          {"lat": 31.99, "lon": -102.08, "weight": 0.03},
    "north":         {"lat": 33.91, "lon": -98.49, "weight": 0.02},
}


def extract_total_load(db_path: str, start_date: str = "2015-01-01",
                       end_date: str = "2024-12-31") -> pd.DataFrame:
    """Extract hourly total system load (sum of all fuel types) from SQLite."""
    conn = sqlite3.connect(db_path)

    query = """
        SELECT delivery_date, interval_15min, SUM(generation_mw) as total_load_mw
        FROM fuel_mix_hist
        WHERE delivery_date >= ? AND delivery_date <= ?
        GROUP BY delivery_date, interval_15min
        ORDER BY delivery_date, interval_15min
    """
    df = pd.read_sql_query(query, conn, params=[start_date, end_date])
    conn.close()

    # Build timestamp from delivery_date + interval_15min (1-96)
    df["delivery_date"] = pd.to_datetime(df["delivery_date"])
    df["timestamp"] = df["delivery_date"] + pd.to_timedelta(
        (df["interval_15min"] - 1) * 15, unit="m"
    )

    # Resample to hourly (mean of 15-min intervals)
    df = df.set_index("timestamp").resample("h")["total_load_mw"].mean().reset_index()
    df = df.dropna(subset=["total_load_mw"])

    logger.info(f"Extracted {len(df)} hourly load records")
    logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    logger.info(f"Load range: {df['total_load_mw'].min():.0f} – {df['total_load_mw'].max():.0f} MW")

    return df


def build_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal / categorical features matching notebook indices 8–14."""
    ts = pd.to_datetime(df["timestamp"])

    df["hour_of_day"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek  # 0=Mon … 6=Sun
    df["month"] = ts.dt.month
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    df["is_peak_hour"] = ts.dt.hour.between(7, 22).astype(int)

    # Season: 0=winter, 1=spring, 2=summer, 3=fall
    df["season"] = ts.dt.month.map(
        {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
         6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    )
    df["is_summer"] = (df["season"] == 2).astype(int)

    # US federal holidays (approximate)
    from pandas.tseries.holiday import USFederalHolidayCalendar
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=ts.min(), end=ts.max())
    df["is_holiday"] = ts.dt.normalize().isin(holidays).astype(int)

    return df


def build_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add load lag features (shifted to prevent data leakage)."""
    df = df.sort_values("timestamp").copy()

    lag_hours = [1, 2, 3, 6, 12, 24, 48, 168]
    for lag in lag_hours:
        df[f"load_lag_{lag}h"] = df["total_load_mw"].shift(lag)

    return df


def build_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling statistics (shifted by 1 to prevent leakage)."""
    df = df.sort_values("timestamp").copy()

    windows = [6, 12, 24, 168]
    for window in windows:
        rolled = df["total_load_mw"].rolling(window, min_periods=1)
        df[f"load_roll_{window}h_mean"] = rolled.mean().shift(1)
        df[f"load_roll_{window}h_std"] = rolled.std().shift(1)
        df[f"load_roll_{window}h_min"] = rolled.min().shift(1)
        df[f"load_roll_{window}h_max"] = rolled.max().shift(1)

    return df


def build_change_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add momentum / change features from lagged data."""
    df["load_change_1h"] = df["total_load_mw"].shift(1) - df["total_load_mw"].shift(2)
    df["load_change_24h"] = df["total_load_mw"].shift(1) - df["total_load_mw"].shift(25)
    df["load_roc_1h"] = df["load_change_1h"] / df["total_load_mw"].shift(2).clip(lower=1000)
    return df


# ---------------------------------------------------------------------------
# Categorical feature indices (for CatBoost / LightGBM)
# ---------------------------------------------------------------------------
CATEGORICAL_FEATURES = [
    "hour_of_day", "day_of_week", "month", "is_weekend",
    "is_peak_hour", "season", "is_holiday",
]

TARGET_COL = "total_load_mw"

FEATURE_COLS = None  # set dynamically after build


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return feature columns (everything except timestamp and target)."""
    exclude = {"timestamp", "valid_time", TARGET_COL, "delivery_date", "interval_15min", "is_summer"}
    return [c for c in df.columns if c not in exclude]


def main():
    base_dir = Path(__file__).parent
    db_path = base_dir.parent.parent.parent / "scraper" / "data" / "ercot_archive.db"
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Database: {db_path}")
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        sys.exit(1)

    # Step 1: Extract total load
    logger.info("=" * 60)
    logger.info("Step 1: Extracting total system load from SQLite")
    logger.info("=" * 60)
    load_df = extract_total_load(str(db_path))

    # Save raw load CSV
    raw_path = data_dir / "ercot_load.csv"
    load_df.to_csv(raw_path, index=False)
    logger.info(f"Saved raw load CSV: {raw_path}")

    # Step 2: Temporal features
    logger.info("=" * 60)
    logger.info("Step 2: Adding temporal features")
    logger.info("=" * 60)
    load_df = build_temporal_features(load_df)

    # Step 3: Lag features
    logger.info("=" * 60)
    logger.info("Step 3: Adding lag features")
    logger.info("=" * 60)
    load_df = build_lag_features(load_df)

    # Step 4: Rolling features
    logger.info("=" * 60)
    logger.info("Step 4: Adding rolling statistics")
    logger.info("=" * 60)
    load_df = build_rolling_features(load_df)

    # Step 5: Change features
    logger.info("=" * 60)
    logger.info("Step 5: Adding change / momentum features")
    logger.info("=" * 60)
    load_df = build_change_features(load_df)

    # Rename timestamp → valid_time for pipeline compatibility
    load_df = load_df.rename(columns={"timestamp": "valid_time"})

    # Drop NaN rows from lags
    before = len(load_df)
    load_df = load_df.dropna()
    logger.info(f"Dropped {before - len(load_df)} rows with NaN from lags")

    # Save augmented features
    out_path = data_dir / "features_augmented.parquet"
    load_df.to_parquet(out_path, index=False)
    logger.info(f"Saved: {out_path}  ({load_df.shape})")

    feature_cols = get_feature_columns(load_df)
    logger.info("=" * 60)
    logger.info("FEATURE BUILD COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Date range: {load_df['valid_time'].min()} to {load_df['valid_time'].max()}")
    logger.info(f"Total features: {len(feature_cols)}")
    logger.info(f"Feature columns: {feature_cols}")
    logger.info(f"Categorical: {CATEGORICAL_FEATURES}")


if __name__ == "__main__":
    main()
