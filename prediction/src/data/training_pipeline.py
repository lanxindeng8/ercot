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
    FUEL_GROUPS,
    TARGET_COLUMNS,
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

FLOAT32_COLUMNS = [
    *[c for c in FEATURE_COLUMNS if c not in {"hour_of_day", "day_of_week", "month", "is_weekend", "is_peak_hour", "is_holiday", "is_summer"}],
    *TARGET_COLUMNS,
]
INT8_COLUMNS = [
    "hour_ending",
    "hour_of_day",
    "day_of_week",
    "month",
    "is_weekend",
    "is_peak_hour",
    "is_holiday",
    "is_summer",
]


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def _optimize_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["delivery_date"] = out["delivery_date"].astype("string")
    out["hour_ending"] = out["hour_ending"].astype("int8")
    out["lmp"] = out["lmp"].astype("float32")
    return out


def _coerce_training_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()
    out["delivery_date"] = out["delivery_date"].astype("string")
    for col in INT8_COLUMNS:
        if col in out.columns:
            out[col] = out[col].astype("int8")
    for col in FLOAT32_COLUMNS:
        if col in out.columns:
            out[col] = out[col].astype("float32")
    return out


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
    return _optimize_price_frame(df)


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
    return _optimize_price_frame(df)


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


def load_fuel_mix_hourly(
    db_path: Path,
    *,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> pd.DataFrame:
    """Load hourly fuel-mix percentages without materializing the raw 15-min table."""
    fuel_case = "CASE fuel " + " ".join(
        f"WHEN '{raw}' THEN '{group}'" for raw, group in sorted(FUEL_GROUPS.items())
    ) + " ELSE NULL END"
    query = (
        "SELECT delivery_date, "
        "       CAST(((interval_15min - 1) / 4) AS INTEGER) + 1 AS hour_ending, "
        f"      {fuel_case} AS fuel_group, "
        "       SUM(generation_mw) AS generation_mw "
        "FROM fuel_mix_hist "
        "WHERE 1=1"
    )
    params: list = []
    if date_from:
        query += " AND delivery_date >= ?"
        params.append(date_from)
    if date_to:
        query += " AND delivery_date < ?"
        params.append(date_to)
    query += " GROUP BY delivery_date, hour_ending, fuel_group"

    with sqlite3.connect(db_path) as conn:
        hourly = pd.read_sql_query(query, conn, params=params)

    if hourly.empty:
        return hourly

    hourly = hourly.dropna(subset=["fuel_group"])
    wide = hourly.pivot_table(
        index=["delivery_date", "hour_ending"],
        columns="fuel_group",
        values="generation_mw",
        fill_value=0,
    ).reset_index()
    wide.columns.name = None

    fuel_cols = [c for c in wide.columns if c not in ("delivery_date", "hour_ending")]
    wide["total"] = wide[fuel_cols].sum(axis=1)

    result = wide[["delivery_date", "hour_ending"]].copy()
    result["delivery_date"] = result["delivery_date"].astype("string")
    result["hour_ending"] = result["hour_ending"].astype("int8")
    for grp in ["wind", "solar", "gas", "nuclear", "coal", "hydro"]:
        if grp in wide.columns:
            result[f"{grp}_pct"] = (
                wide[grp].astype("float32") / wide["total"].replace(0, np.nan).astype("float32")
            ) * np.float32(100.0)
        else:
            result[f"{grp}_pct"] = np.float32(np.nan)

    return result


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
    fuel_hourly = load_fuel_mix_hourly(db_path)
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

        features = _coerce_training_schema(compute_features(dam, rtm, fuel_hourly))
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
