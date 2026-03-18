"""
Unified feature engineering for ERCOT training pipeline.

Computes temporal, price-lag, rolling-stat, cross-market, fuel-mix,
ancillary-service, RTM congestion/loss, and expanded fuel generation
features from hourly data aligned by (delivery_date, hour_ending).

Feature count: 80 (up from 40 in v1).
"""

import numpy as np
import pandas as pd
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PEAK_HOURS = set(range(7, 23))  # HE7–HE22
SUMMER_MONTHS = {6, 7, 8, 9}
ERCOT_TZ = "America/Chicago"

US_MAJOR_HOLIDAYS = {
    (1, 1), (7, 4), (11, 11), (12, 25), (12, 31),
}

# Canonical fuel groups used for percentage features.
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

# Expanded fuel groups — keeps Gas-CC separate for MW-level features.
FUEL_GROUPS_EXPANDED = {
    "Wind": "wind",
    "Wnd": "wind",
    "WSL": "wind",
    "Solar": "solar",
    "Sun": "solar",
    "Gas": "gas",
    "Gas-CC": "gas_cc",
    "Gas_CC": "gas_cc",
    "Gas_GT": "gas",
    "Nuclear": "nuclear",
    "Coal": "coal",
    "Hydro": "hydro",
    "Biomass": "biomass",
    "Oth": "other",
    "Other": "other",
}

# ---------------------------------------------------------------------------
# Feature column registry
# ---------------------------------------------------------------------------

# Original 40 features (v1) ------------------------------------------------

_TEMPORAL_FEATURES = [
    "hour_of_day", "day_of_week", "month",
    "is_weekend", "is_peak_hour", "is_holiday", "is_summer",
]

_DAM_LAG_FEATURES = ["dam_lag_1h", "dam_lag_4h", "dam_lag_24h", "dam_lag_168h"]
_RTM_LAG_FEATURES = ["rtm_lag_1h", "rtm_lag_4h", "rtm_lag_24h", "rtm_lag_168h"]

_DAM_ROLL_FEATURES = [
    f"dam_roll_{w}h_{s}" for w in [24, 168] for s in ["mean", "std", "min", "max"]
]
_RTM_ROLL_FEATURES = [
    f"rtm_roll_{w}h_{s}" for w in [24, 168] for s in ["mean", "std", "min", "max"]
]

_SPREAD_FEATURES = ["dam_rtm_spread", "spread_roll_24h_mean", "spread_roll_168h_mean"]

_FUEL_PCT_FEATURES = [
    "wind_pct", "solar_pct", "gas_pct", "nuclear_pct", "coal_pct", "hydro_pct",
]

# New features (v2) ---------------------------------------------------------

# Ancillary service prices from dam_asmcpc_hist (13 features)
_AS_PRICE_FEATURES = [
    "regdn", "regup", "rrs", "nspin", "ecrs",
]
_AS_DERIVED_FEATURES = [
    "reg_spread",           # regup - regdn
    "total_as_cost",        # sum of all AS prices
]
_AS_LAG_FEATURES = [
    "regup_lag_24h", "rrs_lag_24h", "nspin_lag_24h", "total_as_lag_24h",
]
_AS_ROLL_FEATURES = [
    "total_as_roll_24h_mean", "total_as_roll_24h_std",
]
AS_FEATURES = _AS_PRICE_FEATURES + _AS_DERIVED_FEATURES + _AS_LAG_FEATURES + _AS_ROLL_FEATURES

# RTM congestion/loss components from rtm_lmp_api (6 features)
RTM_COMPONENT_FEATURES = [
    "congestion_pct",           # congestion_component / lmp
    "loss_pct",                 # loss_component / lmp
    "energy_pct",               # energy_component / lmp
    "congestion_ma_4h",         # 4h moving average of congestion_component
    "congestion_volatility_24h",  # 24h rolling std of congestion_component
    "high_congestion_flag",     # binary: |congestion_pct| > 20%
]

# Expanded fuel generation MW + derived (15 features)
_FUEL_GEN_MW_FEATURES = [
    "gas_gen_mw", "gas_cc_gen_mw", "coal_gen_mw", "nuclear_gen_mw",
    "solar_gen_mw", "wind_gen_mw", "hydro_gen_mw", "biomass_gen_mw",
]
_FUEL_DERIVED_FEATURES = [
    "total_gen_mw",             # sum of all generation
    "renewable_ratio",          # (wind + solar) / total
    "thermal_ratio",            # (gas + gas_cc + coal) / total
    "net_load_mw",              # total - wind - solar
]
_FUEL_RAMP_FEATURES = [
    "solar_ramp_1h", "wind_ramp_1h", "gas_ramp_1h",
]
FUEL_GEN_FEATURES = _FUEL_GEN_MW_FEATURES + _FUEL_DERIVED_FEATURES + _FUEL_RAMP_FEATURES

# ---------------------------------------------------------------------------
# Composite feature lists
# ---------------------------------------------------------------------------

# Original v1 features (40) — kept for backward compatibility checks.
FEATURE_COLUMNS_V1 = (
    _TEMPORAL_FEATURES
    + _DAM_LAG_FEATURES + _RTM_LAG_FEATURES
    + _DAM_ROLL_FEATURES + _RTM_ROLL_FEATURES
    + _SPREAD_FEATURES
    + _FUEL_PCT_FEATURES
)

# Full v2 feature set (80).
FEATURE_COLUMNS = (
    FEATURE_COLUMNS_V1
    + AS_FEATURES           # +13
    + RTM_COMPONENT_FEATURES  # +6
    + FUEL_GEN_FEATURES     # +15
)
# => 40 + 13 + 6 + 15 = 74 ... need 6 more to hit 80

# Additional cross-domain features (6 more to reach 80)
CROSS_FEATURES = [
    "dam_as_ratio",             # dam_lag_1h / (total_as_cost + 1)
    "reg_spread_roll_24h_mean", # 24h rolling mean of reg_spread
    "ecrs_lag_24h",             # ECRS lagged 24h
    "gas_cc_share",             # gas_cc_gen_mw / total_gen_mw
    "wind_ramp_4h",             # wind generation change over 4h
    "solar_ramp_4h",            # solar generation change over 4h
]

FEATURE_COLUMNS = FEATURE_COLUMNS + CROSS_FEATURES  # 74 + 6 = 80

# Target columns appended alongside features.
TARGET_COLUMNS = ["dam_lmp", "rtm_lmp"]

# Columns that are allowed to be NaN (optional data sources).
_NULLABLE_SUFFIXES = ("_pct", "_mw", "_ratio", "_flag")
_NULLABLE_FEATURE_SET = set(
    _FUEL_PCT_FEATURES + AS_FEATURES + RTM_COMPONENT_FEATURES
    + FUEL_GEN_FEATURES + CROSS_FEATURES
)


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
    ancillary_hourly: Optional[pd.DataFrame] = None,
    rtm_components_hourly: Optional[pd.DataFrame] = None,
    fuel_gen_hourly: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build the unified feature matrix (80 features).

    Parameters
    ----------
    dam_hourly : DataFrame
        Columns: delivery_date (str YYYY-MM-DD), hour_ending (int 1-24), lmp.
    rtm_hourly : DataFrame
        Same schema as *dam_hourly* (RTM prices aggregated to hourly).
    fuel_hourly : DataFrame, optional
        Columns: delivery_date, hour_ending, wind_pct, solar_pct, gas_pct,
        nuclear_pct, coal_pct, hydro_pct.
    ancillary_hourly : DataFrame, optional
        Columns: delivery_date, hour_ending, regdn, regup, rrs, nspin, ecrs.
    rtm_components_hourly : DataFrame, optional
        Columns: delivery_date, hour_ending, lmp, energy_component,
        congestion_component, loss_component.
    fuel_gen_hourly : DataFrame, optional
        Columns: delivery_date, hour_ending, plus *_gen_mw columns from
        ``aggregate_fuel_gen_hourly()``.

    Returns
    -------
    DataFrame with every column in ``FEATURE_COLUMNS`` plus ``TARGET_COLUMNS``.
    Rows with insufficient history for core lag/rolling features are dropped.
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

    # Build a DST-safe market-hour index and only reindex over valid ERCOT
    # delivery hours. This preserves real data gaps without fabricating the
    # spring-forward hour or the repeated fall-back hour we intentionally drop.
    merged["dt"] = _market_hour_start_index(merged["delivery_date"], merged["hour_ending"])
    merged = merged.dropna(subset=["dt"])
    merged = merged.sort_values("dt").drop_duplicates(subset=["dt"], keep="last")
    market_hours = _market_hour_frame(
        merged["delivery_date"].min(),
        merged["delivery_date"].max(),
    )
    merged = (
        merged.set_index("dt")
        .reindex(market_hours.index)
        .rename_axis("dt")
    )
    merged["delivery_date"] = market_hours["delivery_date"].to_numpy()
    merged["hour_ending"] = market_hours["hour_ending"].to_numpy()

    # ---- temporal features ----
    merged["hour_of_day"] = merged["hour_ending"]
    delivery_dates = pd.to_datetime(merged["delivery_date"])
    merged["day_of_week"] = delivery_dates.dt.dayofweek
    merged["month"] = delivery_dates.dt.month
    merged["is_weekend"] = (merged["day_of_week"] >= 5).astype(int)
    merged["is_peak_hour"] = merged["hour_ending"].isin(PEAK_HOURS).astype(int)
    merged["is_holiday"] = delivery_dates.apply(is_us_holiday)
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
    merged["dam_rtm_spread"] = (merged["dam_lmp"] - merged["rtm_lmp"]).shift(1)
    merged["spread_roll_24h_mean"] = merged["dam_rtm_spread"].rolling(24, min_periods=24).mean()
    merged["spread_roll_168h_mean"] = merged["dam_rtm_spread"].rolling(168, min_periods=168).mean()

    # ---- fuel mix percentages (original v1) ----
    if fuel_hourly is not None and not fuel_hourly.empty:
        fuel_cols = [c for c in fuel_hourly.columns if c.endswith("_pct")]
        fuel_agg = (
            fuel_hourly.groupby(["delivery_date", "hour_ending"], as_index=False)[fuel_cols]
            .mean()
        )
        merged = pd.merge(
            merged.reset_index(drop=True),
            fuel_agg,
            on=["delivery_date", "hour_ending"],
            how="left",
        )
    else:
        merged = merged.reset_index(drop=True)

    for fc in ["wind_pct", "solar_pct", "gas_pct", "nuclear_pct", "coal_pct", "hydro_pct"]:
        if fc not in merged.columns:
            merged[fc] = np.nan

    # ---- ancillary service features ----
    _merge_ancillary(merged, ancillary_hourly)

    # ---- RTM congestion/loss component features ----
    _merge_rtm_components(merged, rtm_components_hourly)

    # ---- expanded fuel generation MW features ----
    _merge_fuel_gen(merged, fuel_gen_hourly)

    # ---- cross-domain features ----
    _build_cross_features(merged)

    # ---- select final columns & drop incomplete rows ----
    # Only require core price lag/rolling features to be non-NaN.
    # New feature groups (AS, congestion, fuel MW) are allowed to be NaN.
    required_cols = [c for c in FEATURE_COLUMNS if c not in _NULLABLE_FEATURE_SET]
    for col in FEATURE_COLUMNS:
        if col not in merged.columns:
            merged[col] = np.nan

    out = merged[["delivery_date", "hour_ending"] + FEATURE_COLUMNS + TARGET_COLUMNS].copy()
    out = out.dropna(subset=required_cols)
    out = out.reset_index(drop=True)

    return out


def _market_hour_start_index(
    delivery_date: pd.Series,
    hour_ending: pd.Series,
) -> pd.Series:
    """Convert ERCOT delivery-date/hour-ending pairs into DST-safe interval starts."""
    delivery_date = pd.to_datetime(delivery_date, format="%Y-%m-%d")
    end_naive = delivery_date + pd.to_timedelta(hour_ending.astype("int64") % 24, unit="h")
    end_local = end_naive.dt.tz_localize(ERCOT_TZ, ambiguous=True, nonexistent="NaT")
    return end_local - pd.Timedelta(hours=1)


def _market_hour_frame(start_date: str, end_date: str) -> pd.DataFrame:
    """Generate the valid ERCOT market-hour sequence between two delivery dates."""
    records: list[dict[str, object]] = []
    for delivery_date in pd.date_range(start_date, end_date, freq="D"):
        delivery_date_str = delivery_date.strftime("%Y-%m-%d")
        for hour_ending in range(1, 25):
            dt = _market_hour_start_index(
                pd.Series([delivery_date_str]),
                pd.Series([hour_ending], dtype="int64"),
            ).iloc[0]
            if pd.isna(dt):
                continue
            records.append(
                {
                    "dt": dt,
                    "delivery_date": delivery_date_str,
                    "hour_ending": hour_ending,
                }
            )
    return pd.DataFrame.from_records(records).set_index("dt")


# ---------------------------------------------------------------------------
# Ancillary service helpers
# ---------------------------------------------------------------------------


def _merge_ancillary(merged: pd.DataFrame, ancillary_hourly: Optional[pd.DataFrame]) -> None:
    """Merge ancillary service features in-place."""
    raw_as: dict[str, pd.Series] = {}
    if ancillary_hourly is not None and not ancillary_hourly.empty:
        as_cols = ["delivery_date", "hour_ending", "regdn", "regup", "rrs", "nspin", "ecrs"]
        as_df = ancillary_hourly[[c for c in as_cols if c in ancillary_hourly.columns]].copy()
        as_df = as_df.drop_duplicates(subset=["delivery_date", "hour_ending"], keep="last")

        # Merge on (delivery_date, hour_ending)
        for col in ["regdn", "regup", "rrs", "nspin", "ecrs"]:
            if col in as_df.columns:
                mapping = as_df.set_index(["delivery_date", "hour_ending"])[col]
                idx = pd.MultiIndex.from_arrays([merged["delivery_date"], merged["hour_ending"]])
                raw_as[col] = pd.Series(idx.map(mapping).values, index=merged.index)

    # Ensure all raw AS columns exist
    for col in _AS_PRICE_FEATURES:
        raw_series = raw_as.get(col, pd.Series(np.nan, index=merged.index))
        merged[col] = raw_series.shift(1)

    # Derived features
    merged["reg_spread"] = merged["regup"] - merged["regdn"]
    merged["total_as_cost"] = (
        merged["regdn"] + merged["regup"] + merged["rrs"]
        + merged["nspin"] + merged["ecrs"]
    )

    # Lags (24h)
    merged["regup_lag_24h"] = raw_as.get("regup", pd.Series(np.nan, index=merged.index)).shift(24)
    merged["rrs_lag_24h"] = raw_as.get("rrs", pd.Series(np.nan, index=merged.index)).shift(24)
    merged["nspin_lag_24h"] = raw_as.get("nspin", pd.Series(np.nan, index=merged.index)).shift(24)
    raw_total_as = sum(
        raw_as.get(col, pd.Series(np.nan, index=merged.index))
        for col in _AS_PRICE_FEATURES
    )
    merged["total_as_lag_24h"] = raw_total_as.shift(24)
    merged["ecrs_lag_24h"] = raw_as.get("ecrs", pd.Series(np.nan, index=merged.index)).shift(24)

    # Rolling stats on total AS cost
    merged["total_as_roll_24h_mean"] = merged["total_as_cost"].rolling(24, min_periods=24).mean()
    merged["total_as_roll_24h_std"] = merged["total_as_cost"].rolling(24, min_periods=24).std()

    # Rolling on reg spread
    merged["reg_spread_roll_24h_mean"] = merged["reg_spread"].rolling(24, min_periods=24).mean()


# ---------------------------------------------------------------------------
# RTM component helpers
# ---------------------------------------------------------------------------


def _merge_rtm_components(
    merged: pd.DataFrame, rtm_components_hourly: Optional[pd.DataFrame]
) -> None:
    """Merge RTM congestion/loss component features in-place."""
    raw_component_cols: dict[str, pd.Series] = {}
    if rtm_components_hourly is not None and not rtm_components_hourly.empty:
        rc = rtm_components_hourly.copy()
        rc = rc.drop_duplicates(subset=["delivery_date", "hour_ending"], keep="last")

        # Compute component percentages
        lmp_safe = rc["lmp"].replace(0, np.nan)
        rc["congestion_pct"] = (rc["congestion_component"] / lmp_safe) * 100
        rc["loss_pct"] = (rc["loss_component"] / lmp_safe) * 100
        rc["energy_pct"] = (rc["energy_component"] / lmp_safe) * 100

        # Merge the pct columns
        rc_merge = rc[["delivery_date", "hour_ending", "congestion_pct", "loss_pct",
                        "energy_pct", "congestion_component"]].copy()
        key_idx = rc_merge.set_index(["delivery_date", "hour_ending"])
        merged_idx = pd.MultiIndex.from_arrays([merged["delivery_date"], merged["hour_ending"]])

        for col in ["congestion_pct", "loss_pct", "energy_pct", "congestion_component"]:
            if col in key_idx.columns:
                raw_component_cols[col] = pd.Series(merged_idx.map(key_idx[col]).values, index=merged.index)

    # Ensure columns exist
    for col in ["congestion_pct", "loss_pct", "energy_pct", "congestion_component"]:
        raw_series = raw_component_cols.get(col, pd.Series(np.nan, index=merged.index))
        if col == "congestion_component":
            merged[col] = raw_series
        else:
            merged[col] = raw_series.shift(1)

    # Derived congestion features
    cong = merged["congestion_component"]
    merged["congestion_ma_4h"] = cong.shift(1).rolling(4, min_periods=4).mean()
    merged["congestion_volatility_24h"] = cong.shift(1).rolling(24, min_periods=24).std()
    merged["high_congestion_flag"] = (merged["congestion_pct"].abs() > 20).astype(float)
    # Convert NaN congestion_pct → NaN flag
    merged.loc[merged["congestion_pct"].isna(), "high_congestion_flag"] = np.nan

    # Drop the intermediate column
    if "congestion_component" in merged.columns:
        merged.drop(columns=["congestion_component"], inplace=True)


# ---------------------------------------------------------------------------
# Expanded fuel generation helpers
# ---------------------------------------------------------------------------


def _merge_fuel_gen(merged: pd.DataFrame, fuel_gen_hourly: Optional[pd.DataFrame]) -> None:
    """Merge expanded fuel generation MW features in-place."""
    gen_cols = [c for c in _FUEL_GEN_MW_FEATURES + ["total_gen_mw"] if True]
    raw_gen: dict[str, pd.Series] = {}

    if fuel_gen_hourly is not None and not fuel_gen_hourly.empty:
        fg = fuel_gen_hourly.drop_duplicates(
            subset=["delivery_date", "hour_ending"], keep="last"
        )
        fg_idx = fg.set_index(["delivery_date", "hour_ending"])
        merged_idx = pd.MultiIndex.from_arrays([merged["delivery_date"], merged["hour_ending"]])

        for col in gen_cols:
            if col in fg_idx.columns:
                raw_gen[col] = pd.Series(merged_idx.map(fg_idx[col]).values, index=merged.index)

    # Ensure all MW columns exist
    for col in _FUEL_GEN_MW_FEATURES:
        merged[col] = raw_gen.get(col, pd.Series(np.nan, index=merged.index)).shift(1)

    # Total generation
    if "total_gen_mw" in raw_gen:
        merged["total_gen_mw"] = raw_gen["total_gen_mw"].shift(1)
    else:
        merged["total_gen_mw"] = (
            merged["gas_gen_mw"] + merged["gas_cc_gen_mw"] + merged["coal_gen_mw"]
            + merged["nuclear_gen_mw"] + merged["solar_gen_mw"]
            + merged["wind_gen_mw"] + merged["hydro_gen_mw"]
            + merged["biomass_gen_mw"]
        )

    total_safe = merged["total_gen_mw"].replace(0, np.nan)

    # Ratios
    merged["renewable_ratio"] = (
        (merged["wind_gen_mw"] + merged["solar_gen_mw"]) / total_safe
    )
    merged["thermal_ratio"] = (
        (merged["gas_gen_mw"] + merged["gas_cc_gen_mw"] + merged["coal_gen_mw"]) / total_safe
    )

    # Net load (total minus wind and solar)
    merged["net_load_mw"] = (
        merged["total_gen_mw"] - merged["wind_gen_mw"] - merged["solar_gen_mw"]
    )

    # Ramps (1h change)
    for fuel, col in [("solar", "solar_gen_mw"), ("wind", "wind_gen_mw"), ("gas", "gas_gen_mw")]:
        merged[f"{fuel}_ramp_1h"] = merged[col] - merged[col].shift(1)

    # 4h ramps
    merged["wind_ramp_4h"] = merged["wind_gen_mw"] - merged["wind_gen_mw"].shift(4)
    merged["solar_ramp_4h"] = merged["solar_gen_mw"] - merged["solar_gen_mw"].shift(4)

    # Gas CC share
    merged["gas_cc_share"] = merged["gas_cc_gen_mw"] / total_safe


def _build_cross_features(merged: pd.DataFrame) -> None:
    """Build cross-domain features in-place."""
    # DAM-to-AS ratio: how does the energy price relate to ancillary costs?
    merged["dam_as_ratio"] = merged["dam_lag_1h"] / (merged["total_as_cost"].abs() + 1)


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


def aggregate_fuel_gen_hourly(fuel_raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 15-min fuel-mix data to hourly MW by expanded fuel type.

    Uses ``FUEL_GROUPS_EXPANDED`` to keep Gas-CC separate from simple-cycle Gas.

    Parameters
    ----------
    fuel_raw : DataFrame
        Columns: delivery_date, fuel, interval_15min (1-96), generation_mw.

    Returns
    -------
    DataFrame with columns: delivery_date, hour_ending, gas_gen_mw,
    gas_cc_gen_mw, coal_gen_mw, nuclear_gen_mw, solar_gen_mw, wind_gen_mw,
    hydro_gen_mw, total_gen_mw.
    """
    df = fuel_raw.copy()
    df["group"] = df["fuel"].map(FUEL_GROUPS_EXPANDED)
    df = df.dropna(subset=["group"])

    # Map interval 1-96 → hour_ending 1-24
    df["hour_ending"] = ((df["interval_15min"] - 1) // 4) + 1

    hourly = (
        df.groupby(["delivery_date", "hour_ending", "group"])["generation_mw"]
        .sum()
        .reset_index()
    )

    wide = hourly.pivot_table(
        index=["delivery_date", "hour_ending"],
        columns="group",
        values="generation_mw",
        fill_value=0,
    ).reset_index()
    wide.columns.name = None

    result = wide[["delivery_date", "hour_ending"]].copy()
    gen_groups = ["gas", "gas_cc", "coal", "nuclear", "solar", "wind", "hydro", "biomass"]
    for grp in gen_groups:
        col_name = f"{grp}_gen_mw"
        result[col_name] = wide[grp].values if grp in wide.columns else 0.0

    result["total_gen_mw"] = result[[f"{g}_gen_mw" for g in gen_groups]].sum(axis=1)

    return result
