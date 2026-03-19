"""
Zone-level spike prediction features built from ERCOT archive SQLite.

Produces a 15-min resolution DataFrame per settlement point with:
- Labels: spike_event, lead_spike_60, regime (from spike_labels table)
- Price, spread, DAM, weather, reserve, wind, and temporal features
- ALL features use shift(1) or larger — no lookahead
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


# ---------------------------------------------------------------------------
# Zone → weather station mapping (inlined to avoid import chain)
# ---------------------------------------------------------------------------

_ZONE_TO_STATION = {
    "LZ_CPS": "san_antonio",
    "LZ_WEST": "midland", "HB_WEST": "midland",
    "LZ_HOUSTON": "houston", "HB_HOUSTON": "houston",
    "HB_NORTH": "dallas", "LZ_NORTH": "dallas",
    "HB_SOUTH": "corpus_christi", "LZ_SOUTH": "corpus_christi",
    "HB_BUSAVG": "austin", "HB_HUBAVG": "austin", "HB_PAN": "austin",
    "LZ_AEN": "austin", "LZ_LCRA": "austin", "LZ_RAYBN": "austin",
}

# 15 settlement points used for training (excludes hub)
SETTLEMENT_POINTS = [
    "HB_BUSAVG", "HB_HOUSTON", "HB_NORTH", "HB_PAN", "HB_SOUTH", "HB_WEST",
    "LZ_AEN", "LZ_CPS", "LZ_HOUSTON", "LZ_LCRA", "LZ_NORTH",
    "LZ_RAYBN", "LZ_SOUTH", "LZ_WEST",
]

# All feature column names produced by build_spike_features.
PRICE_FEATURES = [
    "lmp_lag1", "lmp_lag4", "lmp_lag96",
    "lmp_mean_4", "lmp_mean_96", "lmp_std_96", "lmp_max_96",
    "lmp_ramp_1h", "lmp_ramp_4h",
]
SPREAD_FEATURES = [
    "spread_lag1", "spread_mean_4", "spread_max_96", "spread_ramp_1h",
]
DAM_FEATURES = ["dam_lmp_current_hour", "dam_rtm_gap"]
WEATHER_FEATURES = [
    "temp_2m", "wind_speed_10m", "temp_anomaly", "temp_delta_1h", "wind_chill",
]
RESERVE_FEATURES = [
    "prc_lag1", "prc_mean_1h", "prc_ramp_1h",
    "system_lambda_lag1", "rtordpa_lag1",
]
WIND_FEATURES = ["wind_gen_lag1", "wind_forecast_error"]
TEMPORAL_FEATURES = ["hour_of_day", "day_of_week", "month", "is_weekend"]
LABEL_COLUMNS = ["spike_event", "lead_spike_60", "regime"]

FEATURE_COLUMNS = (
    PRICE_FEATURES + SPREAD_FEATURES + DAM_FEATURES
    + WEATHER_FEATURES + RESERVE_FEATURES + WIND_FEATURES
    + TEMPORAL_FEATURES
)


# ---------------------------------------------------------------------------
# Time conversion helpers
# ---------------------------------------------------------------------------

def _ercot_to_utc(delivery_date: pd.Series, delivery_hour: pd.Series,
                  delivery_interval: pd.Series) -> pd.Series:
    """Convert ERCOT delivery columns to UTC datetime."""
    minutes = (delivery_hour - 1) * 60 + (delivery_interval - 1) * 15
    ct = pd.to_datetime(delivery_date) + pd.to_timedelta(minutes, unit="m")
    ct = ct.dt.tz_localize("America/Chicago", ambiguous="NaT", nonexistent="shift_forward")
    return ct.dt.tz_convert("UTC")


def _ercot_hour_to_utc(delivery_date: pd.Series, hour_ending: pd.Series) -> pd.Series:
    """Convert ERCOT delivery_date + hour_ending to UTC start-of-hour."""
    minutes = (hour_ending - 1) * 60
    ct = pd.to_datetime(delivery_date) + pd.to_timedelta(minutes, unit="m")
    ct = ct.dt.tz_localize("America/Chicago", ambiguous="NaT", nonexistent="shift_forward")
    return ct.dt.tz_convert("UTC")


# ---------------------------------------------------------------------------
# Data loaders (read directly from SQLite)
# ---------------------------------------------------------------------------

def _load_labels(db_path: Path, settlement_point: str,
                 start_date: str, end_date: str) -> pd.DataFrame:
    """Load spike labels for one SP, indexed by UTC time."""
    query = """
        SELECT time, lmp, hub_lmp, spread, spike_event, lead_spike_60, regime
        FROM spike_labels
        WHERE settlement_point = ?
          AND time >= ? AND time <= ?
        ORDER BY time
    """
    with sqlite3.connect(str(db_path)) as conn:
        df = pd.read_sql_query(query, conn, params=[settlement_point, start_date, end_date])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def _load_rtm(db_path: Path, settlement_point: str,
              start_date: str, end_date: str) -> pd.DataFrame:
    """Load 15-min RTM LMP for one SP, indexed by UTC time."""
    query = """
        SELECT delivery_date, delivery_hour, delivery_interval, lmp
        FROM rtm_lmp_hist
        WHERE settlement_point = ?
          AND delivery_date BETWEEN ? AND ?
          AND repeated_hour = 0
        ORDER BY delivery_date, delivery_hour, delivery_interval
    """
    with sqlite3.connect(str(db_path)) as conn:
        df = pd.read_sql_query(query, conn, params=[settlement_point, start_date, end_date])
    df["time"] = _ercot_to_utc(df["delivery_date"], df["delivery_hour"], df["delivery_interval"])
    df = df.dropna(subset=["time"]).set_index("time").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df[["lmp"]]


def _load_dam(db_path: Path, settlement_point: str,
              start_date: str, end_date: str) -> pd.DataFrame:
    """Load hourly DAM LMP for one SP, indexed by UTC hour-start."""
    query = """
        SELECT delivery_date, hour_ending, lmp
        FROM dam_lmp_hist
        WHERE settlement_point = ?
          AND delivery_date BETWEEN ? AND ?
          AND repeated_hour = 0
        ORDER BY delivery_date, hour_ending
    """
    with sqlite3.connect(str(db_path)) as conn:
        df = pd.read_sql_query(query, conn, params=[settlement_point, start_date, end_date])
    df["time"] = _ercot_hour_to_utc(df["delivery_date"], df["hour_ending"])
    df = df.dropna(subset=["time"]).set_index("time").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df[["lmp"]].rename(columns={"lmp": "dam_lmp"})


def _load_weather(db_path: Path, station: str,
                  start_date: str, end_date: str) -> pd.DataFrame:
    """Load hourly weather for one station, indexed by UTC time."""
    query = """
        SELECT time, temperature_2m, wind_speed_10m
        FROM weather_hourly
        WHERE station = ?
          AND time >= ? AND time <= ?
        ORDER BY time
    """
    with sqlite3.connect(str(db_path)) as conn:
        df = pd.read_sql_query(query, conn, params=[station, start_date, end_date])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def _load_reserves(db_path: Path,
                   start_date: str, end_date: str) -> pd.DataFrame:
    """Load RT reserves (~5min), indexed by UTC time."""
    query = """
        SELECT sced_timestamp, system_lambda, prc, rtordpa
        FROM rt_reserves
        WHERE sced_timestamp >= ? AND sced_timestamp <= ?
        ORDER BY sced_timestamp
    """
    with sqlite3.connect(str(db_path)) as conn:
        df = pd.read_sql_query(query, conn, params=[start_date, end_date])
    df["time"] = pd.to_datetime(df["sced_timestamp"], utc=True)
    df = df.set_index("time").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df[["system_lambda", "prc", "rtordpa"]]


def _load_wind(db_path: Path,
               start_date: str, end_date: str) -> pd.DataFrame:
    """Load system-wide wind forecast/actual, indexed by UTC hour-start."""
    query = """
        SELECT delivery_date, hour_ending, gen_mw, stwpf_mw
        FROM wind_forecast
        WHERE region = 'system'
          AND delivery_date BETWEEN ? AND ?
        ORDER BY delivery_date, hour_ending
    """
    with sqlite3.connect(str(db_path)) as conn:
        df = pd.read_sql_query(query, conn, params=[start_date, end_date])
    if df.empty:
        return df
    df["time"] = _ercot_hour_to_utc(df["delivery_date"], df["hour_ending"])
    df = df.dropna(subset=["time"]).set_index("time").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df[["gen_mw", "stwpf_mw"]]


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def _compute_price_features(lmp: pd.Series) -> pd.DataFrame:
    """Compute lagged price features from 15-min LMP series."""
    out = pd.DataFrame(index=lmp.index)
    out["lmp_lag1"] = lmp.shift(1)
    out["lmp_lag4"] = lmp.shift(4)
    out["lmp_lag96"] = lmp.shift(96)

    shifted = lmp.shift(1)
    out["lmp_mean_4"] = shifted.rolling(4, min_periods=1).mean()
    out["lmp_mean_96"] = shifted.rolling(96, min_periods=1).mean()
    out["lmp_std_96"] = shifted.rolling(96, min_periods=1).std()
    out["lmp_max_96"] = shifted.rolling(96, min_periods=1).max()

    out["lmp_ramp_1h"] = shifted - shifted.shift(4)
    out["lmp_ramp_4h"] = shifted - shifted.shift(16)
    return out


def _compute_spread_features(spread: pd.Series) -> pd.DataFrame:
    """Compute lagged spread features."""
    out = pd.DataFrame(index=spread.index)
    out["spread_lag1"] = spread.shift(1)

    shifted = spread.shift(1)
    out["spread_mean_4"] = shifted.rolling(4, min_periods=1).mean()
    out["spread_max_96"] = shifted.rolling(96, min_periods=1).max()
    out["spread_ramp_1h"] = shifted - shifted.shift(4)
    return out


def _compute_dam_features(rtm_15min: pd.DataFrame, dam_hourly: pd.DataFrame) -> pd.DataFrame:
    """Compute DAM features aligned to 15-min index.

    DAM prices are known at 10am day-ahead, so the current hour's DAM price
    is available before the RTM interval — no lookahead.
    """
    out = pd.DataFrame(index=rtm_15min.index)

    # Resample DAM to 15-min by forward-filling the hour's value
    dam_15 = dam_hourly["dam_lmp"].reindex(rtm_15min.index, method="ffill")
    out["dam_lmp_current_hour"] = dam_15

    # DAM - RTM gap at t-1
    rtm_lag1 = rtm_15min["lmp"].shift(1)
    dam_lag1 = dam_15.shift(1)
    out["dam_rtm_gap"] = dam_lag1 - rtm_lag1
    return out


def _compute_weather_features(rtm_index: pd.DatetimeIndex,
                               weather: pd.DataFrame) -> pd.DataFrame:
    """Compute weather features aligned to 15-min RTM index.

    Weather is hourly; shift by 1h to avoid lookahead (weather observation
    for hour H may not be available until after H).
    """
    out = pd.DataFrame(index=rtm_index)

    if weather.empty:
        for col in WEATHER_FEATURES:
            out[col] = np.nan
        return out

    # Shift weather by 1 hour to prevent lookahead, then ffill to 15min
    weather_shifted = weather.shift(1, freq="h")
    temp = weather_shifted["temperature_2m"].reindex(rtm_index, method="ffill")
    wind = weather_shifted["wind_speed_10m"].reindex(rtm_index, method="ffill")

    out["temp_2m"] = temp
    out["wind_speed_10m"] = wind

    # Temperature anomaly: T - 30-day rolling mean
    temp_roll_30d = temp.rolling(96 * 30, min_periods=96).mean()
    out["temp_anomaly"] = temp - temp_roll_30d

    # Temperature delta over 1 hour (4 intervals)
    out["temp_delta_1h"] = temp - temp.shift(4)

    # NWS wind chill: only meaningful when T < 10°C and wind > 4.8 km/h
    # Formula: 13.12 + 0.6215*T - 11.37*V^0.16 + 0.3965*T*V^0.16
    # where T in °C and V in km/h
    v_pow = wind.clip(lower=4.8) ** 0.16
    wc = 13.12 + 0.6215 * temp - 11.37 * v_pow + 0.3965 * temp * v_pow
    # Only apply when cold enough
    mask = (temp < 10) & (wind > 4.8)
    out["wind_chill"] = np.where(mask, wc, temp)

    return out


def _compute_reserve_features(rtm_index: pd.DatetimeIndex,
                               reserves: pd.DataFrame) -> pd.DataFrame:
    """Compute reserve features resampled from ~5min to 15min."""
    out = pd.DataFrame(index=rtm_index)

    if reserves.empty:
        for col in RESERVE_FEATURES:
            out[col] = np.nan
        return out

    # Resample to 15min using last value (floor), then align to RTM index
    res_15 = reserves.resample("15min").last()

    prc = res_15["prc"].reindex(rtm_index, method="ffill")
    sys_lambda = res_15["system_lambda"].reindex(rtm_index, method="ffill")
    rtordpa = res_15["rtordpa"].reindex(rtm_index, method="ffill")

    out["prc_lag1"] = prc.shift(1)
    prc_shifted = prc.shift(1)
    out["prc_mean_1h"] = prc_shifted.rolling(4, min_periods=1).mean()
    out["prc_ramp_1h"] = prc_shifted - prc_shifted.shift(4)

    out["system_lambda_lag1"] = sys_lambda.shift(1)
    out["rtordpa_lag1"] = rtordpa.shift(1)
    return out


def _compute_wind_features(rtm_index: pd.DatetimeIndex,
                            wind: pd.DataFrame) -> pd.DataFrame:
    """Compute wind generation features aligned to 15-min index."""
    out = pd.DataFrame(index=rtm_index)

    if wind.empty:
        out["wind_gen_lag1"] = np.nan
        out["wind_forecast_error"] = np.nan
        return out

    gen = wind["gen_mw"].reindex(rtm_index, method="ffill")
    stwpf = wind["stwpf_mw"].reindex(rtm_index, method="ffill")

    out["wind_gen_lag1"] = gen.shift(1)
    out["wind_forecast_error"] = (gen - stwpf).shift(1)
    return out


def _compute_temporal_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Compute temporal features from UTC time index.

    ERCOT operates in Central time, so we convert for hour_of_day.
    """
    out = pd.DataFrame(index=index)
    ct = index.tz_convert("America/Chicago")
    out["hour_of_day"] = ct.hour
    out["day_of_week"] = ct.dayofweek
    out["month"] = ct.month
    out["is_weekend"] = (ct.dayofweek >= 5).astype(int)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_spike_features(
    db_path: Path,
    settlement_point: str,
    start_date: str = "2015-01-01",
    end_date: str = "2026-03-19",
) -> pd.DataFrame:
    """Build feature matrix for one settlement point.

    Returns DataFrame indexed by UTC time (15min), with label columns
    (spike_event, lead_spike_60, regime) and all feature columns.
    All features use shift(1) or larger — no lookahead.
    """
    db_path = Path(db_path)
    logger.info("Building spike features for {} [{} → {}]", settlement_point, start_date, end_date)

    # --- Load all data sources ---
    labels = _load_labels(db_path, settlement_point, start_date, end_date)
    rtm = _load_rtm(db_path, settlement_point, start_date, end_date)
    dam = _load_dam(db_path, settlement_point, start_date, end_date)

    station = _ZONE_TO_STATION.get(settlement_point, "austin")
    weather = _load_weather(db_path, station, start_date, end_date)
    reserves = _load_reserves(db_path, start_date, end_date)
    wind = _load_wind(db_path, start_date, end_date)

    if rtm.empty:
        logger.warning("No RTM data for {} — returning empty DataFrame", settlement_point)
        return pd.DataFrame()

    logger.info("  RTM: {:,} rows, DAM: {:,} rows, Weather: {:,}, Reserves: {:,}, Wind: {:,}",
                len(rtm), len(dam), len(weather), len(reserves), len(wind))

    # --- Join labels to RTM index ---
    # Labels may have slightly different coverage; use RTM as the base index
    base = rtm[["lmp"]].copy()

    if not labels.empty:
        base = base.join(labels[["spread", "spike_event", "lead_spike_60", "regime"]], how="left")
    else:
        base["spread"] = np.nan
        base["spike_event"] = False
        base["lead_spike_60"] = False
        base["regime"] = "Normal"

    # --- Compute feature groups ---
    price_feats = _compute_price_features(base["lmp"])
    spread_feats = _compute_spread_features(base["spread"])
    dam_feats = _compute_dam_features(base, dam)
    weather_feats = _compute_weather_features(base.index, weather)
    reserve_feats = _compute_reserve_features(base.index, reserves)
    wind_feats = _compute_wind_features(base.index, wind)
    temporal_feats = _compute_temporal_features(base.index)

    # --- Concatenate all features ---
    result = pd.concat([
        base[LABEL_COLUMNS],
        price_feats,
        spread_feats,
        dam_feats,
        weather_feats,
        reserve_feats,
        wind_feats,
        temporal_feats,
    ], axis=1)

    # --- NaN handling ---
    # Weather, reserves, wind may be NaN for years without data.
    # Forward-fill then fill remaining with 0.
    fill_cols = WEATHER_FEATURES + RESERVE_FEATURES + WIND_FEATURES
    result[fill_cols] = result[fill_cols].ffill().fillna(0)

    logger.info("  Final shape: {} rows × {} cols ({} features)",
                len(result), len(result.columns),
                len([c for c in result.columns if c not in LABEL_COLUMNS]))

    return result
