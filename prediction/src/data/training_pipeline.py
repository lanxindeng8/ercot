"""
Unified Training Data Pipeline for ERCOT prediction models.

Loads RTM, DAM, fuel-mix, ancillary-service, and RTM-component history
from the SQLite archive, computes unified features (80 total) via
``prediction.src.features.unified_features``, splits by date into
train / val / test, and exports per-settlement-point Parquet files.

Usage
-----
    python -m prediction.src.data.training_pipeline --output ./data/training/

"""

import argparse
import sqlite3
from datetime import timezone
from pathlib import Path
from typing import List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from prediction.src.features.unified_features import (
    FEATURE_COLUMNS,
    FEATURE_COLUMNS_V1,
    FUEL_GROUPS,
    FUEL_GROUPS_EXPANDED,
    TARGET_COLUMNS,
    compute_features,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_DB = Path(__file__).resolve().parents[3] / "scraper" / "data" / "ercot_archive.db"

SETTLEMENT_POINTS = ["HB_WEST", "HB_HOUSTON", "HB_NORTH", "HB_SOUTH", "LZ_LCRA", "LZ_WEST"]

# Date-based split boundaries (exclusive upper bound)
TRAIN_END = "2024-01-01"  # train: < 2024
VAL_END = "2025-01-01"    # val:   2024
                           # test:  >= 2025
ERCOT_TZ = ZoneInfo("America/Chicago")

# Temporal features are int8; everything else is float32.
_TEMPORAL_INT8 = {
    "hour_ending", "hour_of_day", "day_of_week", "month",
    "is_weekend", "is_peak_hour", "is_holiday", "is_summer",
}

FLOAT32_COLUMNS = [
    *[c for c in FEATURE_COLUMNS if c not in _TEMPORAL_INT8],
    *TARGET_COLUMNS,
]
INT8_COLUMNS = list(_TEMPORAL_INT8)


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


def _utc_to_ercot_hourly_components(timestamp_series: pd.Series) -> pd.DataFrame:
    """Convert UTC timestamps to ERCOT delivery_date/hour_ending with DST handling."""
    utc = pd.to_datetime(timestamp_series, utc=True)
    local = utc.dt.tz_convert(ERCOT_TZ)

    delivery_date = local.dt.strftime("%Y-%m-%d")
    hour_ending = local.dt.hour.astype("int16")

    midnight_mask = hour_ending == 0
    if midnight_mask.any():
        delivery_date = delivery_date.mask(
            midnight_mask,
            (local - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d"),
        )
        hour_ending = hour_ending.mask(midnight_mask, 24)

    repeated_hour = local.map(lambda ts: int(ts.fold)).astype("int8")
    return pd.DataFrame(
        {
            "delivery_date": delivery_date.astype("string"),
            "hour_ending": hour_ending.astype("int8"),
            "repeated_hour": repeated_hour,
        }
    )


def _table_exists(db_path: Path, table_name: str) -> bool:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone()
    return row is not None


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
# New data loaders (v2)
# ---------------------------------------------------------------------------


def load_ancillary_hourly(
    db_path: Path,
    *,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> pd.DataFrame:
    """Load hourly ancillary service prices from *dam_asmcpc_hist*.

    Returns DataFrame with columns: delivery_date, hour_ending,
    regdn, regup, rrs, nspin, ecrs.
    """
    if not _table_exists(db_path, "dam_asmcpc_hist"):
        return pd.DataFrame(
            columns=["delivery_date", "hour_ending", "regdn", "regup", "rrs", "nspin", "ecrs"]
        )

    query = (
        "SELECT delivery_date, hour_ending, "
        "       regdn, regup, rrs, nspin, ecrs "
        "FROM dam_asmcpc_hist "
        "WHERE repeated_hour = 0"
    )
    params: list = []
    if date_from:
        query += " AND delivery_date >= ?"
        params.append(date_from)
    if date_to:
        query += " AND delivery_date < ?"
        params.append(date_to)
    query += " ORDER BY delivery_date, hour_ending"

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=params)

    if df.empty:
        return df

    df["delivery_date"] = df["delivery_date"].astype("string")
    df["hour_ending"] = df["hour_ending"].astype("int8")
    for col in ["regdn", "regup", "rrs", "nspin", "ecrs"]:
        df[col] = df[col].astype("float32")
    return df


def load_rtm_components_hourly(
    db_path: Path,
    settlement_point: str,
    *,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> pd.DataFrame:
    """Load hourly RTM price components from *rtm_lmp_api*.

    Aggregates 5-minute data to hourly averages.

    Returns DataFrame with columns: delivery_date, hour_ending,
    lmp, energy_component, congestion_component, loss_component.

    Note: rtm_lmp_api has limited historical coverage (~5 weeks as of
    March 2026). Features will be NaN for dates outside its range.
    """
    if not _table_exists(db_path, "rtm_lmp_api"):
        return pd.DataFrame(
            columns=[
                "delivery_date",
                "hour_ending",
                "lmp",
                "energy_component",
                "congestion_component",
                "loss_component",
            ]
        )

    query = (
        "SELECT time, "
        "       AVG(lmp) AS lmp, "
        "       AVG(energy_component) AS energy_component, "
        "       AVG(congestion_component) AS congestion_component, "
        "       AVG(loss_component) AS loss_component "
        "FROM rtm_lmp_api "
        "WHERE settlement_point = ?"
    )
    params: list = [settlement_point]
    if date_from:
        start_utc = (
            pd.Timestamp(date_from)
            .tz_localize(ERCOT_TZ)
            .tz_convert(timezone.utc)
            .tz_localize(None)
            .isoformat()
        )
        query += " AND time >= ?"
        params.append(start_utc)
    if date_to:
        end_utc = (
            pd.Timestamp(date_to)
            .tz_localize(ERCOT_TZ)
            .tz_convert(timezone.utc)
            .tz_localize(None)
            .isoformat()
        )
        query += " AND time < ?"
        params.append(end_utc)
    query += " GROUP BY time ORDER BY time"

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=params)

    if df.empty:
        return df

    components = _utc_to_ercot_hourly_components(df["time"])
    df = pd.concat([components, df.drop(columns=["time"])], axis=1)
    df = df[df["repeated_hour"] == 0].drop(columns=["repeated_hour"])
    df = (
        df.groupby(["delivery_date", "hour_ending"], as_index=False)[
            ["lmp", "energy_component", "congestion_component", "loss_component"]
        ]
        .mean()
        .sort_values(["delivery_date", "hour_ending"])
        .reset_index(drop=True)
    )
    for col in ["lmp", "energy_component", "congestion_component", "loss_component"]:
        df[col] = df[col].astype("float32")
    return df


def load_fuel_gen_hourly(
    db_path: Path,
    *,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> pd.DataFrame:
    """Load hourly fuel generation in MW with expanded fuel groups.

    Uses ``FUEL_GROUPS_EXPANDED`` to keep Gas-CC separate from simple Gas.

    Returns DataFrame with columns: delivery_date, hour_ending,
    gas_gen_mw, gas_cc_gen_mw, coal_gen_mw, nuclear_gen_mw, solar_gen_mw,
    wind_gen_mw, hydro_gen_mw, biomass_gen_mw, total_gen_mw.
    """
    fuel_case = "CASE fuel " + " ".join(
        f"WHEN '{raw}' THEN '{group}'"
        for raw, group in sorted(FUEL_GROUPS_EXPANDED.items())
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
    # Exclude 'other' from features
    hourly = hourly[hourly["fuel_group"] != "other"]

    wide = hourly.pivot_table(
        index=["delivery_date", "hour_ending"],
        columns="fuel_group",
        values="generation_mw",
        fill_value=0,
    ).reset_index()
    wide.columns.name = None

    result = wide[["delivery_date", "hour_ending"]].copy()
    result["delivery_date"] = result["delivery_date"].astype("string")
    result["hour_ending"] = result["hour_ending"].astype("int8")

    gen_groups = ["gas", "gas_cc", "coal", "nuclear", "solar", "wind", "hydro", "biomass"]
    for grp in gen_groups:
        col_name = f"{grp}_gen_mw"
        if grp in wide.columns:
            result[col_name] = wide[grp].astype("float32")
        else:
            result[col_name] = np.float32(0.0)

    result["total_gen_mw"] = result[[f"{g}_gen_mw" for g in gen_groups]].sum(axis=1).astype("float32")
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

    # Shared data (not per-settlement-point) – load once.
    if verbose:
        print("Loading fuel mix (percentages) …")
    fuel_hourly = load_fuel_mix_hourly(db_path)
    if verbose and fuel_hourly is not None:
        print(f"  fuel-mix rows (hourly): {len(fuel_hourly):,}")

    if verbose:
        print("Loading ancillary service prices …")
    ancillary_hourly = load_ancillary_hourly(db_path)
    if verbose:
        print(f"  ancillary rows: {len(ancillary_hourly):,}")

    if verbose:
        print("Loading expanded fuel generation MW …")
    fuel_gen_hourly = load_fuel_gen_hourly(db_path)
    if verbose:
        print(f"  fuel-gen rows: {len(fuel_gen_hourly):,}")

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

        # RTM components are per-settlement-point and have limited coverage.
        # Load if available; will produce NaN features for dates outside range.
        rtm_comp = load_rtm_components_hourly(db_path, sp)
        if verbose:
            n_comp = len(rtm_comp) if rtm_comp is not None and not rtm_comp.empty else 0
            print(f"  RTM component rows: {n_comp:,}")

        features = _coerce_training_schema(
            compute_features(
                dam, rtm, fuel_hourly,
                ancillary_hourly=ancillary_hourly,
                rtm_components_hourly=rtm_comp if not rtm_comp.empty else None,
                fuel_gen_hourly=fuel_gen_hourly,
            )
        )
        if verbose:
            print(f"  feature rows: {len(features):,}  feature cols: {len([c for c in FEATURE_COLUMNS if c in features.columns])}")

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
