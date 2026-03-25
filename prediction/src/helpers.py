"""Shared helper functions and constants for prediction API endpoints."""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import HTTPException

from .config import SETTLEMENT_POINTS
from .features.unified_features import compute_features

log = logging.getLogger(__name__)

HUB_SETTLEMENT_POINTS = [sp for sp in SETTLEMENT_POINTS if sp.startswith("HB_")]
LOAD_ZONE_SETTLEMENT_POINTS = [sp for sp in SETTLEMENT_POINTS if sp.startswith("LZ_")]

ALLOWED_HORIZONS = {"1h", "4h", "24h"}
PREDICTIONS_DB = Path(__file__).resolve().parents[1] / "data" / "predictions.db"
DELTA_SPREAD_SETTLEMENT_POINTS = {"LZ_WEST", "HB_WEST"}


def fetch_and_compute_features(settlement_point: str) -> pd.DataFrame:
    """
    Fetch DAM + RTM data and compute unified features (80 features).

    Prefers SQLite archive as primary source; falls back to InfluxDB if SQLite
    is unavailable or returns insufficient data.  Also loads ancillary service
    prices, RTM congestion components, and expanded fuel generation MW when
    available from the SQLite archive.

    Returns a DataFrame with all 80 feature columns needed by DAM/RTM/Spike models.
    Raises HTTPException on failure.
    """
    features_df = _try_sqlite_features(settlement_point)
    if features_df is not None:
        return features_df

    # Fallback: InfluxDB
    return _influxdb_features(settlement_point)


def _try_sqlite_features(settlement_point: str) -> Optional[pd.DataFrame]:
    """Attempt to load features from SQLite. Returns None on failure."""
    from .data.sqlite_fetcher import create_sqlite_fetcher
    from .data.training_pipeline import (
        load_ancillary_hourly,
        load_fuel_gen_hourly,
        load_fuel_mix_hourly,
        load_rtm_components_hourly,
    )

    fetcher = None
    db_path = None
    try:
        fetcher = create_sqlite_fetcher()
        db_path = fetcher.db_path
        start = datetime.utcnow() - timedelta(days=30)
        dam_raw = fetcher.fetch_dam_prices(settlement_point=settlement_point, start_date=start)
        rtm_raw = fetcher.fetch_rtm_prices(settlement_point=settlement_point, start_date=start)
    except Exception as e:
        log.warning("SQLite fetch failed, will try InfluxDB: %s", e)
        return None
    finally:
        if fetcher is not None:
            fetcher.close()

    if dam_raw.empty or rtm_raw.empty:
        log.info("SQLite returned no data for %s, will try InfluxDB", settlement_point)
        return None

    latest_rtm_ts = rtm_raw.index.max()
    if latest_rtm_ts is None or pd.isna(latest_rtm_ts):
        log.info("SQLite RTM data has no usable timestamps for %s, will try InfluxDB", settlement_point)
        return None
    latest_rtm_ts = pd.Timestamp(latest_rtm_ts)
    stale_cutoff = pd.Timestamp.utcnow().tz_localize(None) - timedelta(hours=24)
    if latest_rtm_ts.tzinfo is not None:
        latest_rtm_ts = latest_rtm_ts.tz_convert("UTC").tz_localize(None)
    if latest_rtm_ts < stale_cutoff:
        log.info("SQLite RTM data is stale for %s (latest %s), will try InfluxDB", settlement_point, latest_rtm_ts)
        return None

    dam_hourly, rtm_hourly = _raw_to_hourly(dam_raw, rtm_raw)

    # Load additional data sources for 80-feature pipeline
    if db_path is None:
        from .data.training_pipeline import DEFAULT_DB
        db_path = DEFAULT_DB
    start_str = start.strftime("%Y-%m-%d")
    ancillary_hourly = None
    fuel_hourly = None
    rtm_comp = None
    fuel_gen_hourly = None
    try:
        ancillary_hourly = load_ancillary_hourly(db_path, date_from=start_str)
        fuel_hourly = load_fuel_mix_hourly(db_path, date_from=start_str)
        rtm_comp = load_rtm_components_hourly(db_path, settlement_point, date_from=start_str)
        fuel_gen_hourly = load_fuel_gen_hourly(db_path, date_from=start_str)
    except Exception as e:
        log.warning("Failed to load ancillary/fuel/rtm-component data: %s", e)

    features_df = compute_features(
        dam_hourly, rtm_hourly, fuel_hourly,
        ancillary_hourly=ancillary_hourly,
        rtm_components_hourly=rtm_comp if rtm_comp is not None and not rtm_comp.empty else None,
        fuel_gen_hourly=fuel_gen_hourly,
    )

    if features_df.empty:
        log.info("SQLite feature computation returned no rows for %s", settlement_point)
        return None

    return features_df


def _influxdb_features(settlement_point: str) -> pd.DataFrame:
    """Fetch features from InfluxDB (fallback path)."""
    from .data.influxdb_fetcher import create_fetcher_from_env

    fetcher = None
    try:
        fetcher = create_fetcher_from_env()
        start = datetime.utcnow() - timedelta(days=30)
        dam_raw = fetcher.fetch_dam_prices(settlement_point=settlement_point, start_date=start)
        rtm_raw = fetcher.fetch_rtm_prices(settlement_point=settlement_point, start_date=start)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"InfluxDB fetch failed: {e}")
    finally:
        if fetcher is not None:
            fetcher.close()

    if dam_raw.empty or rtm_raw.empty:
        raise HTTPException(status_code=404, detail=f"Insufficient data for {settlement_point}")

    dam_hourly, rtm_hourly = _raw_to_hourly(dam_raw, rtm_raw)
    features_df = compute_features(dam_hourly, rtm_hourly)

    if features_df.empty:
        raise HTTPException(status_code=404, detail="Feature computation returned no rows (insufficient history)")

    return features_df


def _raw_to_hourly(dam_raw: pd.DataFrame, rtm_raw: pd.DataFrame):
    """Convert fetcher output (InfluxDB or SQLite format) to unified hourly frames."""
    # DAM: needs delivery_date, hour_ending, lmp
    dam_hourly = dam_raw.reset_index()
    dam_hourly = dam_hourly.rename(columns={"dam_price": "lmp"})
    if "date" not in dam_hourly.columns:
        dam_hourly["date"] = dam_hourly["timestamp"].dt.date
    dam_hourly["delivery_date"] = dam_hourly["date"].astype(str)
    if "hour" in dam_hourly.columns:
        dam_hourly["hour_ending"] = dam_hourly["hour"]
    else:
        dam_hourly["hour_ending"] = dam_hourly["timestamp"].dt.hour + 1
    dam_hourly = dam_hourly[["delivery_date", "hour_ending", "lmp"]].drop_duplicates(
        subset=["delivery_date", "hour_ending"], keep="last"
    )

    # RTM: needs delivery_date, hour_ending, lmp (aggregate 5-min to hourly)
    rtm_hourly = rtm_raw.reset_index()
    rtm_hourly = rtm_hourly.rename(columns={"rtm_price": "lmp"})
    if "date" in rtm_hourly.columns:
        rtm_hourly["delivery_date"] = pd.to_datetime(rtm_hourly["date"]).dt.strftime("%Y-%m-%d")
    else:
        rtm_hourly["delivery_date"] = rtm_hourly["timestamp"].dt.date.astype(str)
    if "hour" in rtm_hourly.columns:
        rtm_hourly["hour_ending"] = rtm_hourly["hour"]
    else:
        rtm_hourly["hour_ending"] = rtm_hourly["timestamp"].dt.hour + 1
    rtm_hourly = rtm_hourly.groupby(["delivery_date", "hour_ending"], as_index=False)["lmp"].mean()

    return dam_hourly, rtm_hourly


def build_wind_features() -> pd.DataFrame:
    """
    Build synthetic wind features for the next 24 hours.

    In production, replace with real HRRR weather data + historical wind generation
    from InfluxDB. For now, generates temporal + placeholder features matching
    the trained model's feature set.
    """
    from .models.wind_predictor import get_wind_predictor

    history_path = (
        Path(__file__).resolve().parents[1] / "models" / "wind" / "data" / "ercot_wind.csv"
    )

    predictor = get_wind_predictor()
    feature_names = predictor.metadata.get("feature_names", [])
    if not feature_names:
        raise RuntimeError("Wind model metadata missing feature names")
    if history_path.exists():
        history_df = pd.read_csv(history_path)
        history_df["timestamp"] = pd.to_datetime(history_df["timestamp"])
        history_df = history_df.sort_values("timestamp")
        if history_df.empty:
            raise RuntimeError("Wind history is empty")
    else:
        raise RuntimeError(f"Wind history not found: {history_path}")

    generation_history = history_df["wind_generation"].astype(float).tolist()
    proxy_seed = float(np.mean(generation_history[-24:])) if generation_history else 0.0
    now = history_df["timestamp"].iloc[-1]
    records = []

    for h in range(24):
        dt = now + timedelta(hours=h + 1)
        hour = dt.hour
        recent = pd.Series(generation_history, dtype=float)
        record: Dict[str, Any] = {
            "normalized_power_mean": proxy_seed / 40000.0,
            "ws_80m_mean": 3.0 + 9.0 * np.power(np.clip(proxy_seed / 40000.0, 0.001, 0.999), 1.0 / 3.0),
            "ws_80m_std": (3.0 + 9.0 * np.power(np.clip(proxy_seed / 40000.0, 0.001, 0.999), 1.0 / 3.0)) * 0.15,
            "wind_shear_mean": 0.2,
            "power_sensitivity": max(0.0, 3 * ((max(3.0, 3.0 + 9.0 * np.power(np.clip(proxy_seed / 40000.0, 0.001, 0.999), 1.0 / 3.0)) - 3) / 9) ** 2 / 9),
            "hour": hour,
            "hour_of_day": hour,
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "day_of_week": dt.weekday(),
            "dow_sin": np.sin(2 * np.pi * dt.weekday() / 7),
            "dow_cos": np.cos(2 * np.pi * dt.weekday() / 7),
            "month": dt.month,
            "month_sin": np.sin(2 * np.pi * dt.month / 12),
            "month_cos": np.cos(2 * np.pi * dt.month / 12),
            "doy_sin": np.sin(2 * np.pi * dt.timetuple().tm_yday / 365),
            "doy_cos": np.cos(2 * np.pi * dt.timetuple().tm_yday / 365),
            "is_weekend": int(dt.weekday() >= 5),
            "is_peak_hour": int(7 <= hour <= 22),
            "is_ramp_prone_hour": int(hour in (6, 7, 17, 18, 19)),
            "is_morning": int(6 <= hour < 12),
            "is_afternoon": int(12 <= hour < 18),
            "is_evening": int(18 <= hour < 22),
            "is_night": int(hour >= 22 or hour < 6),
            "is_winter": int(dt.month in (12, 1, 2)),
            "is_spring": int(dt.month in (3, 4, 5)),
            "is_summer": int(dt.month in (6, 7, 8)),
            "is_fall": int(dt.month in (9, 10, 11)),
            "is_no_solar_period": int(hour >= 19 or hour < 7),
            "is_evening_peak": int(17 <= hour < 21),
            "hours_to_sunset": float(np.clip(19 - hour, -12, 12)),
            "hours_since_sunset": float(hour - 19 if hour >= 19 else (hour + 5 if hour < 7 else 0)),
        }

        for lag in (1, 2, 3, 6, 12, 24):
            record[f"wind_gen_lag_{lag}h"] = float(generation_history[-lag])

        for window in (6, 12, 24):
            window_values = recent.iloc[-window:]
            record[f"wind_gen_rolling_{window}h_mean"] = float(window_values.mean())
            record[f"wind_gen_rolling_{window}h_std"] = float(window_values.std(ddof=1) if len(window_values) > 1 else 0.0)
            record[f"wind_gen_rolling_{window}h_min"] = float(window_values.min())
            record[f"wind_gen_rolling_{window}h_max"] = float(window_values.max())

        record["wind_gen_change_1h"] = float(generation_history[-1] - generation_history[-2])
        record["wind_gen_change_3h"] = float(generation_history[-1] - generation_history[-4])
        record["wind_gen_change_6h"] = float(generation_history[-1] - generation_history[-7])
        record["wind_gen_roc_1h"] = float(record["wind_gen_change_1h"] / max(generation_history[-2], 100.0))
        record["wind_gen_roc_3h"] = float(record["wind_gen_change_3h"] / max(generation_history[-4], 100.0))
        record["ramp_down_1h"] = float(record["wind_gen_change_1h"] < -1000)
        record["ramp_down_3h"] = float(record["wind_gen_change_3h"] < -2000)
        record["ramp_up_1h"] = float(record["wind_gen_change_1h"] > 1000)

        risk = 0.0
        risk += 0.15 if record["wind_gen_change_1h"] < -1000 else 0.0
        risk += 0.20 if record["wind_gen_change_1h"] < -2000 else 0.0
        risk += 0.15 if record["is_no_solar_period"] else 0.0
        risk += 0.15 if record["is_no_solar_period"] and record["is_evening_peak"] else 0.0
        record["ramp_down_risk_score"] = float(min(max(risk, 0.0), 1.0))

        full_record = {name: record.get(name, 0.0) for name in feature_names}
        records.append(full_record)

        predicted = float(np.clip(
            0.55 * record["wind_gen_lag_1h"]
            + 0.30 * record["wind_gen_rolling_6h_mean"]
            + 0.15 * record["wind_gen_rolling_24h_mean"],
            0.0,
            40000.0,
        ))
        generation_history.append(predicted)
        proxy_seed = predicted

    return pd.DataFrame(records, columns=feature_names)


def build_load_features() -> pd.DataFrame:
    """
    Build load forecast features for the next 24 hours.

    In production, replace with real historical load data from InfluxDB.
    For now, generates temporal features + placeholder lag/rolling features
    matching the trained model's feature set.
    """
    from .models.load_predictor import FEATURE_COLS
    from pandas.tseries.holiday import USFederalHolidayCalendar

    history_path = (
        Path(__file__).resolve().parents[1] / "models" / "load" / "data" / "ercot_load.csv"
    )
    if not history_path.exists():
        raise RuntimeError(f"Load history not found: {history_path}")

    history_df = pd.read_csv(history_path)
    history_df["timestamp"] = pd.to_datetime(history_df["timestamp"])
    history_df = history_df.sort_values("timestamp")
    if history_df.empty:
        raise RuntimeError("Load history is empty")

    load_history = history_df["total_load_mw"].astype(float).tolist()
    calendar = USFederalHolidayCalendar()
    forecast_index = pd.date_range(
        history_df["timestamp"].iloc[-1] + timedelta(hours=1),
        periods=24,
        freq="h",
    )
    holidays = set(calendar.holidays(start=forecast_index.min(), end=forecast_index.max()).normalize())
    records = []

    for dt in forecast_index:
        hour = dt.hour
        recent = pd.Series(load_history, dtype=float)
        record = {
            "hour_of_day": hour,
            "day_of_week": dt.weekday(),
            "month": dt.month,
            "is_weekend": int(dt.weekday() >= 5),
            "is_peak_hour": int(7 <= hour <= 22),
            "season": {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}[dt.month],
            "is_holiday": int(dt.normalize() in holidays),
        }

        for lag in (1, 2, 3, 6, 12, 24, 48, 168):
            record[f"load_lag_{lag}h"] = float(load_history[-lag])

        for window in (6, 12, 24, 168):
            window_values = recent.iloc[-window:]
            record[f"load_roll_{window}h_mean"] = float(window_values.mean())
            record[f"load_roll_{window}h_std"] = float(window_values.std(ddof=1) if len(window_values) > 1 else 0.0)
            record[f"load_roll_{window}h_min"] = float(window_values.min())
            record[f"load_roll_{window}h_max"] = float(window_values.max())

        record["load_change_1h"] = float(load_history[-1] - load_history[-2])
        record["load_change_24h"] = float(load_history[-1] - load_history[-25])
        record["load_roc_1h"] = float(record["load_change_1h"] / max(load_history[-2], 1000.0))
        records.append({col: record.get(col, 0.0) for col in FEATURE_COLS})

        baseline = 0.50 * record["load_lag_24h"] + 0.35 * record["load_lag_1h"] + 0.15 * record["load_roll_24h_mean"]
        if record["is_peak_hour"]:
            baseline += 1200.0
        if record["is_weekend"]:
            baseline -= 900.0
        load_history.append(float(max(baseline, 0.0)))

    return pd.DataFrame(records, columns=FEATURE_COLS)


def generate_delta_features(hours: int) -> pd.DataFrame:
    """
    Generate mock features for Delta Spread prediction.

    In production, replace with real InfluxDB data pipeline.
    """
    now = datetime.utcnow()
    target_date = now + timedelta(days=1)

    records = []
    for h in range(1, min(hours + 1, 25)):
        target_dt = target_date.replace(hour=h % 24, minute=0, second=0, microsecond=0)

        record = {
            "target_dam_price": 35.0 + np.random.randn() * 10,
            "target_hour": int(h),
            "target_dow": int(target_dt.weekday()),
            "target_month": int(target_dt.month),
            "target_day_of_month": int(target_dt.day),
            "target_week": int(target_dt.isocalendar()[1]),
            "target_is_weekend": int(1 if target_dt.weekday() >= 5 else 0),
            "target_is_peak": int(1 if 7 <= h <= 22 else 0),
            "target_is_summer": int(1 if target_dt.month in [6, 7, 8, 9] else 0),
            "spread_mean_7d": -7.0 + np.random.randn() * 3,
            "spread_std_7d": 15.0 + np.random.randn() * 5,
            "spread_max_7d": 50.0 + np.random.randn() * 20,
            "spread_min_7d": -30.0 + np.random.randn() * 10,
            "spread_median_7d": -5.0 + np.random.randn() * 3,
            "spread_mean_24h": -6.0 + np.random.randn() * 5,
            "spread_std_24h": 12.0 + np.random.randn() * 4,
            "rtm_mean_7d": 28.0 + np.random.randn() * 8,
            "rtm_std_7d": 25.0 + np.random.randn() * 10,
            "rtm_max_7d": 150.0 + np.random.randn() * 50,
            "rtm_mean_24h": 27.0 + np.random.randn() * 10,
            "rtm_volatility_24h": 20.0 + np.random.randn() * 8,
            "dam_mean_7d": 35.0 + np.random.randn() * 8,
            "dam_std_7d": 15.0 + np.random.randn() * 5,
            "dam_mean_24h": 34.0 + np.random.randn() * 10,
            "spread_positive_ratio_7d": 0.25 + np.random.rand() * 0.2,
            "spread_positive_ratio_24h": 0.25 + np.random.rand() * 0.3,
            "spike_count_7d": int(np.random.randint(0, 5)),
            "rtm_spike_count_7d": int(np.random.randint(0, 8)),
            "spread_trend_7d": np.random.randn() * 0.5,
            "spread_same_hour_hist": -7.0 + np.random.randn() * 5,
            "spread_same_hour_std": 15.0 + np.random.randn() * 5,
            "rtm_same_hour_hist": 28.0 + np.random.randn() * 10,
            "spread_same_dow_hour": -6.0 + np.random.randn() * 5,
            "dam_vs_7d_mean": np.random.randn() * 0.1,
            "dam_percentile_7d": 0.5 + np.random.randn() * 0.2,
            "spread_same_dow_hour_last": -5.0 + np.random.randn() * 8,
            "dam_price_level": int(np.random.choice([0, 1, 2])),
            "hour_sin": np.sin(2 * np.pi * h / 24),
            "hour_cos": np.cos(2 * np.pi * h / 24),
            "dow_sin": np.sin(2 * np.pi * target_dt.weekday() / 7),
            "dow_cos": np.cos(2 * np.pi * target_dt.weekday() / 7),
            "month_sin": np.sin(2 * np.pi * target_dt.month / 12),
            "month_cos": np.cos(2 * np.pi * target_dt.month / 12),
            "dam_vs_same_hour": np.random.randn() * 5,
            "target_date": target_dt.strftime("%Y-%m-%d"),
        }
        records.append(record)

    return pd.DataFrame(records)


def normalize_settlement_point(settlement_point: str) -> str:
    value = settlement_point.strip().upper()
    if not value:
        raise HTTPException(status_code=400, detail="Settlement point is required")
    allowed = {sp.upper() for sp in SETTLEMENT_POINTS}
    if value not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported settlement point '{settlement_point}'. Available: {sorted(allowed)}",
        )
    return value


def latest_complete_delivery_rows(features_df: pd.DataFrame) -> pd.DataFrame:
    counts = features_df.groupby("delivery_date")["hour_ending"].nunique().sort_index()
    complete_dates = counts[counts >= 24]
    if complete_dates.empty:
        return pd.DataFrame(columns=features_df.columns)
    latest_date = complete_dates.index[-1]
    rows = features_df[features_df["delivery_date"] == latest_date].copy()
    return rows.sort_values("hour_ending").head(24)


def normalize_delta_settlement_point(settlement_point: str) -> str:
    value = settlement_point.strip().upper()
    if value not in DELTA_SPREAD_SETTLEMENT_POINTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported settlement point '{settlement_point}'. Available: {sorted(DELTA_SPREAD_SETTLEMENT_POINTS)}",
        )
    return value


def parse_horizons(horizons: Optional[str]) -> Optional[List[str]]:
    if horizons is None:
        return None

    parsed = [h.strip() for h in horizons.split(",") if h.strip()]
    if not parsed:
        raise HTTPException(status_code=400, detail="At least one horizon must be provided")

    invalid = [h for h in parsed if h not in ALLOWED_HORIZONS]
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported horizons {invalid}. Available: {sorted(ALLOWED_HORIZONS)}",
        )

    seen = set()
    ordered = []
    for horizon in parsed:
        if horizon not in seen:
            ordered.append(horizon)
            seen.add(horizon)
    return ordered
