"""
Unified feature engineering for ERCOT training pipeline.

Computes temporal, price-lag, rolling-stat, cross-market, and fuel-mix
features from hourly RTM, DAM, and fuel-mix data aligned by
(delivery_date, hour_ending).
"""

import numpy as np
import pandas as pd
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PEAK_HOURS = set(range(7, 23))  # HE7–HE22
SUMMER_MONTHS = {6, 7, 8, 9}

US_MAJOR_HOLIDAYS = {
    (1, 1), (7, 4), (11, 11), (12, 25), (12, 31),
}

# Canonical fuel groups used as features.  Raw ERCOT names are mapped onto
# these in _normalise_fuel().
FUEL_GROUPS = {
    "Wind": "wind",
    "Wnd": "wind",
    "WSL": "wind",
    "Solar": "solar",
    "Sun": "solar",
    "Gas": "gas",
    "Gas-CC": "gas",
    "Gas_CC": "gas",
    "Gas_GT": "gas",
    "Nuclear": "nuclear",
    "Coal": "coal",
    "Hydro": "hydro",
}

# Ordered list of every feature produced by compute_features().
FEATURE_COLUMNS = [
    # Temporal (7)
    "hour_of_day",
    "day_of_week",
    "month",
    "is_weekend",
    "is_peak_hour",
    "is_holiday",
    "is_summer",
    # DAM price lags (4)
    "dam_lag_1h",
    "dam_lag_4h",
    "dam_lag_24h",
    "dam_lag_168h",
    # RTM price lags (4)
    "rtm_lag_1h",
    "rtm_lag_4h",
    "rtm_lag_24h",
    "rtm_lag_168h",
    # DAM rolling stats – 24 h (4)
    "dam_roll_24h_mean",
    "dam_roll_24h_std",
    "dam_roll_24h_min",
    "dam_roll_24h_max",
    # DAM rolling stats – 168 h (4)
    "dam_roll_168h_mean",
    "dam_roll_168h_std",
    "dam_roll_168h_min",
    "dam_roll_168h_max",
    # RTM rolling stats – 24 h (4)
    "rtm_roll_24h_mean",
    "rtm_roll_24h_std",
    "rtm_roll_24h_min",
    "rtm_roll_24h_max",
    # RTM rolling stats – 168 h (4)
    "rtm_roll_168h_mean",
    "rtm_roll_168h_std",
    "rtm_roll_168h_min",
    "rtm_roll_168h_max",
    # Cross-market (3)
    "dam_rtm_spread",
    "spread_roll_24h_mean",
    "spread_roll_168h_mean",
    # Fuel mix pct (6)
    "wind_pct",
    "solar_pct",
    "gas_pct",
    "nuclear_pct",
    "coal_pct",
    "hydro_pct",
]

# Target columns appended alongside features.
TARGET_COLUMNS = ["dam_lmp", "rtm_lmp"]


def is_us_holiday(dt: pd.Timestamp) -> int:
    """Return 1 if *dt* falls on a simplified US major holiday."""
    if (dt.month, dt.day) in US_MAJOR_HOLIDAYS:
        return 1
    # Thanksgiving: 4th Thursday of November
    if dt.month == 11 and dt.weekday() == 3:
        first_day = pd.Timestamp(dt.year, 11, 1)
        first_thu = first_day + pd.Timedelta(days=(3 - first_day.weekday()) % 7)
        if dt.date() == (first_thu + pd.Timedelta(weeks=3)).date():
            return 1
    # Labor Day: 1st Monday of September
    if dt.month == 9 and dt.weekday() == 0 and dt.day <= 7:
        return 1
    # Memorial Day: last Monday of May
    if dt.month == 5 and dt.weekday() == 0:
        if (dt + pd.Timedelta(weeks=1)).month != 5:
            return 1
    return 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_features(
    dam_hourly: pd.DataFrame,
    rtm_hourly: pd.DataFrame,
    fuel_hourly: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build the unified feature matrix.

    Parameters
    ----------
    dam_hourly : DataFrame
        Columns: delivery_date (str YYYY-MM-DD), hour_ending (int 1-24), lmp.
    rtm_hourly : DataFrame
        Same schema as *dam_hourly* (RTM prices aggregated to hourly).
    fuel_hourly : DataFrame, optional
        Columns: delivery_date, hour_ending, wind_pct, solar_pct, gas_pct,
        nuclear_pct, coal_pct, hydro_pct.  May be ``None`` when fuel-mix
        data is unavailable for the requested date range.

    Returns
    -------
    DataFrame indexed by (delivery_date, hour_ending) with every column in
    ``FEATURE_COLUMNS`` plus ``TARGET_COLUMNS``.  Rows with insufficient
    history for lag/rolling features are dropped.
    """
    # ---- merge DAM + RTM on (delivery_date, hour_ending) ----
    merged = pd.merge(
        dam_hourly.rename(columns={"lmp": "dam_lmp"}),
        rtm_hourly.rename(columns={"lmp": "rtm_lmp"}),
        on=["delivery_date", "hour_ending"],
        how="inner",
        validate="one_to_one",
    )

    if merged.empty:
        return pd.DataFrame(columns=["delivery_date", "hour_ending"] + FEATURE_COLUMNS + TARGET_COLUMNS)

    # Build a proper datetime for sorting / rolling.
    # Reindex to a true hourly timeline so lag/rolling windows stay aligned
    # across any missing delivery hours in the source data.
    merged["dt"] = pd.to_datetime(merged["delivery_date"]) + pd.to_timedelta(
        merged["hour_ending"] - 1, unit="h"
    )
    merged = merged.sort_values("dt").drop_duplicates(subset=["dt"], keep="last")
    merged = merged.set_index("dt").sort_index()
    merged = merged.reindex(pd.date_range(merged.index.min(), merged.index.max(), freq="h"))
    merged.index.name = "dt"
    merged["delivery_date"] = merged.index.strftime("%Y-%m-%d")
    merged["hour_ending"] = merged.index.hour + 1

    # ---- temporal features ----
    merged["hour_of_day"] = merged["hour_ending"]
    merged["day_of_week"] = merged.index.dayofweek
    merged["month"] = merged.index.month
    merged["is_weekend"] = (merged["day_of_week"] >= 5).astype(int)
    merged["is_peak_hour"] = merged["hour_ending"].isin(PEAK_HOURS).astype(int)
    merged["is_holiday"] = pd.Series(merged.index, index=merged.index).apply(is_us_holiday)
    merged["is_summer"] = merged["month"].isin(SUMMER_MONTHS).astype(int)

    # ---- price lags ----
    for prefix, col in [("dam", "dam_lmp"), ("rtm", "rtm_lmp")]:
        merged[f"{prefix}_lag_1h"] = merged[col].shift(1)
        merged[f"{prefix}_lag_4h"] = merged[col].shift(4)
        merged[f"{prefix}_lag_24h"] = merged[col].shift(24)
        merged[f"{prefix}_lag_168h"] = merged[col].shift(168)

    # ---- rolling stats ----
    for prefix, col in [("dam", "dam_lmp"), ("rtm", "rtm_lmp")]:
        for win in [24, 168]:
            r = merged[col].shift(1).rolling(win, min_periods=win)
            merged[f"{prefix}_roll_{win}h_mean"] = r.mean()
            merged[f"{prefix}_roll_{win}h_std"] = r.std()
            merged[f"{prefix}_roll_{win}h_min"] = r.min()
            merged[f"{prefix}_roll_{win}h_max"] = r.max()

    # ---- cross-market spread ----
    merged["dam_rtm_spread"] = merged["dam_lmp"] - merged["rtm_lmp"]
    spread_shifted = merged["dam_rtm_spread"].shift(1)
    merged["spread_roll_24h_mean"] = spread_shifted.rolling(24, min_periods=24).mean()
    merged["spread_roll_168h_mean"] = spread_shifted.rolling(168, min_periods=168).mean()

    # ---- fuel mix ----
    if fuel_hourly is not None and not fuel_hourly.empty:
        fuel_cols = [c for c in fuel_hourly.columns if c.endswith("_pct")]
        fuel_hourly = (
            fuel_hourly.groupby(["delivery_date", "hour_ending"], as_index=False)[fuel_cols]
            .mean()
        )
        merged = pd.merge(
            merged.reset_index(drop=True),
            fuel_hourly,
            on=["delivery_date", "hour_ending"],
            how="left",
        )
    else:
        merged = merged.reset_index(drop=True)
    # Ensure fuel columns exist (fill with NaN when missing)
    for fc in ["wind_pct", "solar_pct", "gas_pct", "nuclear_pct", "coal_pct", "hydro_pct"]:
        if fc not in merged.columns:
            merged[fc] = np.nan

    # ---- select final columns & drop incomplete rows ----
    out = merged[["delivery_date", "hour_ending"] + FEATURE_COLUMNS + TARGET_COLUMNS].copy()
    out = out.dropna(subset=[c for c in FEATURE_COLUMNS if not c.endswith("_pct")])
    out = out.reset_index(drop=True)

    return out


# ---------------------------------------------------------------------------
# Fuel-mix helpers
# ---------------------------------------------------------------------------


def aggregate_fuel_mix_hourly(fuel_raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 15-min fuel-mix data to hourly percentages.

    Parameters
    ----------
    fuel_raw : DataFrame
        Columns: delivery_date, fuel, interval_15min (1-96), generation_mw.

    Returns
    -------
    DataFrame with columns: delivery_date, hour_ending, wind_pct, solar_pct,
    gas_pct, nuclear_pct, coal_pct, hydro_pct.
    """
    df = fuel_raw.copy()
    df["group"] = df["fuel"].map(FUEL_GROUPS)
    # Drop unmapped fuels (Biomass, Other, etc.)
    df = df.dropna(subset=["group"])

    # Map interval 1-96 → hour_ending 1-24
    df["hour_ending"] = ((df["interval_15min"] - 1) // 4) + 1

    # Sum generation per (date, hour, group)
    hourly = (
        df.groupby(["delivery_date", "hour_ending", "group"])["generation_mw"]
        .sum()
        .reset_index()
    )

    # Pivot to wide
    wide = hourly.pivot_table(
        index=["delivery_date", "hour_ending"],
        columns="group",
        values="generation_mw",
        fill_value=0,
    ).reset_index()
    wide.columns.name = None

    # Compute total and percentages
    fuel_cols = [c for c in wide.columns if c not in ("delivery_date", "hour_ending")]
    wide["total"] = wide[fuel_cols].sum(axis=1)

    result = wide[["delivery_date", "hour_ending"]].copy()
    for grp in ["wind", "solar", "gas", "nuclear", "coal", "hydro"]:
        if grp in wide.columns:
            result[f"{grp}_pct"] = (wide[grp] / wide["total"].replace(0, np.nan)) * 100
        else:
            result[f"{grp}_pct"] = np.nan

    return result
