"""
Unified Training Data Pipeline for ERCOT prediction models.

Loads RTM, DAM, and fuel-mix history from the SQLite archive, computes
unified features via ``prediction.src.features.unified_features``, splits
by date into train / val / test, and exports per-settlement-point Parquet
files.

Usage
-----
    python -m prediction.src.data.training_pipeline --output ./data/training/

"""

import argparse
import sqlite3
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from prediction.src.features.unified_features import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    aggregate_fuel_mix_hourly,
    compute_features,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_DB = Path(__file__).resolve().parents[3] / "scraper" / "data" / "ercot_archive.db"

SETTLEMENT_POINTS = ["HB_WEST", "HB_NORTH", "HB_SOUTH", "HB_HOUSTON", "HB_BUSAVG"]

# Date-based split boundaries (exclusive upper bound)
TRAIN_END = "2024-01-01"  # train: < 2024
VAL_END = "2025-01-01"    # val:   2024
                           # test:  >= 2025


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def load_dam_hourly(
    db_path: Path,
    settlement_point: str,
    *,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> pd.DataFrame:
    """Load hourly DAM LMP from *dam_lmp_hist*.

    Returns DataFrame with columns: delivery_date, hour_ending, lmp.
    """
    query = (
        "SELECT delivery_date, hour_ending, lmp "
        "FROM dam_lmp_hist "
        "WHERE settlement_point = ? AND repeated_hour = 0"
    )
    params: list = [settlement_point]
    if date_from:
        query += " AND delivery_date >= ?"
        params.append(date_from)
    if date_to:
        query += " AND delivery_date < ?"
        params.append(date_to)
    query += " ORDER BY delivery_date, hour_ending"

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=params)
    return df


def load_rtm_hourly(
    db_path: Path,
    settlement_point: str,
    *,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> pd.DataFrame:
    """Load RTM LMP from *rtm_lmp_hist*, averaged to hourly.

    The source table has 15-min intervals (delivery_interval 1-4 per hour).
    We average to get one LMP per (delivery_date, delivery_hour).

    Returns DataFrame with columns: delivery_date, hour_ending, lmp.
    """
    query = (
        "SELECT delivery_date, delivery_hour, AVG(lmp) AS lmp "
        "FROM rtm_lmp_hist "
        "WHERE settlement_point = ? AND repeated_hour = 0"
    )
    params: list = [settlement_point]
    if date_from:
        query += " AND delivery_date >= ?"
        params.append(date_from)
    if date_to:
        query += " AND delivery_date < ?"
        params.append(date_to)
    query += " GROUP BY delivery_date, delivery_hour ORDER BY delivery_date, delivery_hour"

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=params)

    df = df.rename(columns={"delivery_hour": "hour_ending"})
    return df


def load_fuel_mix(
    db_path: Path,
    *,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> pd.DataFrame:
    """Load raw 15-min fuel-mix from *fuel_mix_hist*.

    Returns DataFrame with columns: delivery_date, fuel, interval_15min,
    generation_mw.
    """
    query = (
        "SELECT delivery_date, fuel, interval_15min, generation_mw "
        "FROM fuel_mix_hist WHERE 1=1"
    )
    params: list = []
    if date_from:
        query += " AND delivery_date >= ?"
        params.append(date_from)
    if date_to:
        query += " AND delivery_date < ?"
        params.append(date_to)

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=params)
    return df


# ---------------------------------------------------------------------------
# Split helper
# ---------------------------------------------------------------------------


def split_by_date(
    df: pd.DataFrame,
    train_end: str = TRAIN_END,
    val_end: str = VAL_END,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split feature DataFrame into train / val / test by delivery_date."""
    train = df[df["delivery_date"] < train_end].copy()
    val = df[(df["delivery_date"] >= train_end) & (df["delivery_date"] < val_end)].copy()
    test = df[df["delivery_date"] >= val_end].copy()
    return train, val, test


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    db_path: Path = DEFAULT_DB,
    output_dir: Path = Path("./data/training"),
    settlement_points: Optional[List[str]] = None,
    verbose: bool = True,
) -> None:
    """Execute the full training-data pipeline.

    For each settlement point: load → features → split → write parquet.
    """
    sps = settlement_points or SETTLEMENT_POINTS
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fuel mix is shared across settlement points – load once.
    if verbose:
        print("Loading fuel mix …")
    fuel_raw = load_fuel_mix(db_path)
    fuel_hourly = aggregate_fuel_mix_hourly(fuel_raw) if not fuel_raw.empty else None
    if verbose and fuel_hourly is not None:
        print(f"  fuel-mix rows (hourly): {len(fuel_hourly):,}")

    for sp in sps:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Settlement point: {sp}")
            print(f"{'='*60}")

        dam = load_dam_hourly(db_path, sp)
        rtm = load_rtm_hourly(db_path, sp)
        if verbose:
            print(f"  DAM rows: {len(dam):,}  RTM rows: {len(rtm):,}")

        if dam.empty or rtm.empty:
            print(f"  ⚠ skipping {sp} – no data")
            continue

        features = compute_features(dam, rtm, fuel_hourly)
        if verbose:
            print(f"  feature rows: {len(features):,}")

        train, val, test = split_by_date(features)
        if verbose:
            print(f"  train: {len(train):,}  val: {len(val):,}  test: {len(test):,}")

        sp_dir = output_dir / sp.lower()
        sp_dir.mkdir(parents=True, exist_ok=True)

        for name, part in [("train", train), ("val", val), ("test", test)]:
            path = sp_dir / f"{name}.parquet"
            part.to_parquet(path, index=False)
            if verbose:
                print(f"  wrote {path}  ({len(part):,} rows)")

    if verbose:
        print("\nDone.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build unified training data for ERCOT prediction models."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./data/training"),
        help="Output directory for parquet files (default: ./data/training/)",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help="Path to SQLite archive (default: scraper/data/ercot_archive.db)",
    )
    parser.add_argument(
        "--settlement-points",
        nargs="+",
        default=None,
        help="Settlement points to process (default: all 5 HB hubs)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args()

    run_pipeline(
        db_path=args.db,
        output_dir=args.output,
        settlement_points=args.settlement_points,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
