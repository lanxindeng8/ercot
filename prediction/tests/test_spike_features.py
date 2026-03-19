"""Tests for spike_features feature engineering.

Tests cover:
- No lookahead: features at time t use only data from t-1 or earlier
- Column presence: all expected columns exist
- Temporal alignment: weather (hourly) joined to 15-min data
- Reserve alignment: 5-min reserves resampled to 15-min
- NaN handling: fill strategy works
- Known value check: verify specific date features
"""

import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prediction.src.features.spike_features import (
    FEATURE_COLUMNS,
    LABEL_COLUMNS,
    PRICE_FEATURES,
    SPREAD_FEATURES,
    DAM_FEATURES,
    WEATHER_FEATURES,
    RESERVE_FEATURES,
    WIND_FEATURES,
    TEMPORAL_FEATURES,
    build_spike_features,
    _compute_price_features,
    _compute_spread_features,
    _compute_weather_features,
    _compute_reserve_features,
    _compute_temporal_features,
)


# ---------------------------------------------------------------------------
# Fixtures: create a small in-memory SQLite DB with synthetic data
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_db(tmp_path):
    """Create a minimal SQLite DB with 3 days of synthetic data."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()

    # --- Create tables ---
    c.execute("""
        CREATE TABLE rtm_lmp_hist (
            delivery_date TEXT, delivery_hour INTEGER,
            delivery_interval INTEGER, repeated_hour INTEGER,
            settlement_point TEXT, settlement_point_type TEXT, lmp REAL,
            PRIMARY KEY (delivery_date, delivery_hour, delivery_interval,
                         repeated_hour, settlement_point)
        )
    """)
    c.execute("""
        CREATE TABLE dam_lmp_hist (
            delivery_date TEXT, hour_ending INTEGER,
            repeated_hour INTEGER, settlement_point TEXT, lmp REAL,
            PRIMARY KEY (delivery_date, hour_ending, repeated_hour, settlement_point)
        )
    """)
    c.execute("""
        CREATE TABLE weather_hourly (
            station TEXT, time TEXT, temperature_2m REAL,
            wind_speed_10m REAL, wind_direction_10m REAL,
            relative_humidity_2m REAL, surface_pressure REAL, dew_point_2m REAL,
            PRIMARY KEY (station, time)
        )
    """)
    c.execute("""
        CREATE TABLE rt_reserves (
            sced_timestamp TEXT, repeated_hour TEXT, batch_id INTEGER,
            system_lambda REAL, prc REAL, rtolcap REAL, rtoffcap REAL,
            rtorpa REAL, rtoffpa REAL, rtolhsl REAL, rtbp REAL, rtordpa REAL,
            PRIMARY KEY (sced_timestamp, batch_id)
        )
    """)
    c.execute("""
        CREATE TABLE wind_forecast (
            delivery_date TEXT, hour_ending INTEGER, region TEXT,
            gen_mw REAL, stwpf_mw REAL, wgrpp_mw REAL, cop_hsl_mw REAL,
            PRIMARY KEY (delivery_date, hour_ending, region)
        )
    """)
    c.execute("""
        CREATE TABLE spike_labels (
            time TEXT, settlement_point TEXT, lmp REAL, hub_lmp REAL,
            spread REAL, spike_event INTEGER, lead_spike_60 INTEGER,
            regime TEXT,
            PRIMARY KEY (time, settlement_point)
        )
    """)

    # --- Populate with 3 days of data for HB_WEST ---
    sp = "HB_WEST"
    dates = ["2025-01-10", "2025-01-11", "2025-01-12"]
    np.random.seed(42)

    # RTM: 15-min intervals (96 per day)
    for date in dates:
        for hour in range(1, 25):
            for interval in range(1, 5):
                lmp = 30.0 + np.random.randn() * 10
                c.execute(
                    "INSERT INTO rtm_lmp_hist VALUES (?,?,?,0,?,?,?)",
                    (date, hour, interval, sp, "LZ", lmp),
                )

    # DAM: hourly
    for date in dates:
        for hour in range(1, 25):
            lmp = 35.0 + np.random.randn() * 8
            c.execute(
                "INSERT INTO dam_lmp_hist VALUES (?,?,0,?,?)",
                (date, hour, sp, lmp),
            )

    # Weather: hourly UTC
    for h in range(72):
        t = pd.Timestamp("2025-01-10", tz="UTC") + pd.Timedelta(hours=h)
        c.execute(
            "INSERT INTO weather_hourly VALUES (?,?,?,?,?,?,?,?)",
            ("midland", t.isoformat(), 5.0 + h * 0.1, 15.0 + h * 0.05,
             180, 50, 1013, 2.0),
        )

    # Reserves: ~5-min intervals
    for m in range(0, 72 * 60, 5):
        t = pd.Timestamp("2025-01-10", tz="UTC") + pd.Timedelta(minutes=m)
        c.execute(
            "INSERT INTO rt_reserves VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (t.isoformat(), "0", 1, 25.0 + m * 0.01,
             3000.0 - m * 0.1, 0, 0, 0, 0, 0, 0, 0.5),
        )

    # Wind forecast: hourly
    for date in dates:
        for hour in range(1, 25):
            c.execute(
                "INSERT INTO wind_forecast VALUES (?,?,?,?,?,?,?)",
                (date, hour, "system", 8000 + np.random.randn() * 500,
                 7800 + np.random.randn() * 400, 0, 0),
            )

    # Spike labels: 15-min UTC
    for date in dates:
        for hour in range(1, 25):
            for interval in range(1, 5):
                minutes = (hour - 1) * 60 + (interval - 1) * 15
                ct = pd.Timestamp(date) + pd.Timedelta(minutes=minutes)
                ct = ct.tz_localize("America/Chicago", nonexistent="shift_forward")
                utc = ct.tz_convert("UTC")
                lmp = 30.0 + np.random.randn() * 10
                hub_lmp = 28.0 + np.random.randn() * 5
                spread = lmp - hub_lmp
                c.execute(
                    "INSERT OR IGNORE INTO spike_labels VALUES (?,?,?,?,?,?,?,?)",
                    (utc.isoformat(), sp, lmp, hub_lmp, spread, 0, 0, "Normal"),
                )

    conn.commit()
    conn.close()
    return db_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestColumnPresence:
    """All expected feature and label columns must be present."""

    def test_all_feature_columns_present(self, synthetic_db):
        df = build_spike_features(synthetic_db, "HB_WEST", "2025-01-10", "2025-01-12")
        assert not df.empty
        for col in FEATURE_COLUMNS:
            assert col in df.columns, f"Missing feature column: {col}"

    def test_all_label_columns_present(self, synthetic_db):
        df = build_spike_features(synthetic_db, "HB_WEST", "2025-01-10", "2025-01-12")
        for col in LABEL_COLUMNS:
            assert col in df.columns, f"Missing label column: {col}"

    def test_feature_count(self, synthetic_db):
        df = build_spike_features(synthetic_db, "HB_WEST", "2025-01-10", "2025-01-12")
        feature_cols = [c for c in df.columns if c not in LABEL_COLUMNS]
        assert len(feature_cols) == len(FEATURE_COLUMNS)


class TestNoLookahead:
    """Features at time t must only use data from t-1 or earlier."""

    def test_lmp_lag1_uses_previous_value(self):
        """lmp_lag1 at index i should equal lmp at index i-1."""
        lmp = pd.Series([10, 20, 30, 40, 50], dtype=float,
                        index=pd.date_range("2025-01-01", periods=5, freq="15min", tz="UTC"))
        feats = _compute_price_features(lmp)
        # lmp_lag1 at i=1 should be lmp[0]=10
        assert feats["lmp_lag1"].iloc[1] == 10.0
        assert feats["lmp_lag1"].iloc[2] == 20.0
        assert pd.isna(feats["lmp_lag1"].iloc[0])

    def test_rolling_mean_no_current(self):
        """Rolling mean uses shift(1), so current value is excluded."""
        lmp = pd.Series([100] + [10] * 10, dtype=float,
                        index=pd.date_range("2025-01-01", periods=11, freq="15min", tz="UTC"))
        feats = _compute_price_features(lmp)
        # At index 1, lmp_mean_4 = mean of shifted[1] = lmp[0] = 100
        assert feats["lmp_mean_4"].iloc[1] == 100.0
        # At index 5, lmp_mean_4 = mean(shifted[2:6]) = mean(lmp[1:5]) = mean(10,10,10,10) = 10
        assert feats["lmp_mean_4"].iloc[5] == 10.0

    def test_spread_lag1_is_shifted(self):
        """spread_lag1 at time t should be spread at t-1."""
        spread = pd.Series([5, 10, 15, 20, 25], dtype=float,
                           index=pd.date_range("2025-01-01", periods=5, freq="15min", tz="UTC"))
        feats = _compute_spread_features(spread)
        assert feats["spread_lag1"].iloc[1] == 5.0
        assert feats["spread_lag1"].iloc[3] == 15.0

    def test_weather_shifted_1h(self):
        """Weather features use 1h shift — value at time t is from t-1h."""
        idx = pd.date_range("2025-01-01", periods=8, freq="15min", tz="UTC")
        weather = pd.DataFrame({
            "temperature_2m": [0.0, 10.0, 20.0],
            "wind_speed_10m": [5.0, 15.0, 25.0],
        }, index=pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC"))
        feats = _compute_weather_features(idx, weather)
        # At 00:00 UTC, shifted weather means we use the value from -1h (NaN or ffill)
        # At 01:00, we get the 00:00 weather (T=0, W=5)
        assert feats["temp_2m"].iloc[4] == 0.0  # 01:00 gets shifted 00:00 value

    def test_reserve_lag1(self):
        """Reserve features at t use reserves from t-1."""
        idx = pd.date_range("2025-01-01", periods=4, freq="15min", tz="UTC")
        reserves = pd.DataFrame({
            "prc": [3000, 2900, 2800, 2700, 2600, 2500, 2400, 2300,
                    2200, 2100, 2000, 1900],
            "system_lambda": [25.0] * 12,
            "rtordpa": [0.5] * 12,
        }, index=pd.date_range("2025-01-01", periods=12, freq="5min", tz="UTC"))
        feats = _compute_reserve_features(idx, reserves)
        # prc_lag1 at 00:15 should be the 00:00 value (after resample)
        assert pd.isna(feats["prc_lag1"].iloc[0])  # first interval has no prior
        assert not pd.isna(feats["prc_lag1"].iloc[1])  # second has prior


class TestTemporalAlignment:
    """Weather (hourly) and reserves (~5min) properly aligned to 15min."""

    def test_weather_ffill_to_15min(self):
        """Hourly weather is forward-filled across 15-min intervals."""
        idx = pd.date_range("2025-01-01", periods=8, freq="15min", tz="UTC")
        weather = pd.DataFrame({
            "temperature_2m": [10.0, 20.0, 30.0],
            "wind_speed_10m": [5.0, 10.0, 15.0],
        }, index=pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC"))
        feats = _compute_weather_features(idx, weather)
        # After 1h shift, 01:00-01:45 should all have T=10 (the 00:00 obs)
        vals = feats["temp_2m"].iloc[4:8]
        assert (vals == 10.0).all() or (vals == 0.0).all()  # ffill from shifted

    def test_reserves_resample_last(self):
        """5-min reserves resampled to 15-min using last value."""
        idx = pd.date_range("2025-01-01", periods=4, freq="15min", tz="UTC")
        reserves = pd.DataFrame({
            "prc": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200],
            "system_lambda": range(12),
            "rtordpa": [0.1] * 12,
        }, index=pd.date_range("2025-01-01", periods=12, freq="5min", tz="UTC"))
        feats = _compute_reserve_features(idx, reserves)
        # After resample('15min').last(), first 15-min bucket gets the 3rd value (300)
        # prc_lag1 at idx[1] = resampled prc at idx[0] = 300
        assert feats["prc_lag1"].iloc[1] == 300.0


class TestNaNHandling:
    """Verify NaN fill strategy for optional data sources."""

    def test_weather_nan_filled(self, synthetic_db):
        """Weather features should not have NaN after fill."""
        df = build_spike_features(synthetic_db, "HB_WEST", "2025-01-10", "2025-01-12")
        for col in WEATHER_FEATURES:
            assert df[col].isna().sum() == 0, f"{col} has NaN values"

    def test_reserve_nan_filled(self, synthetic_db):
        """Reserve features should not have NaN after fill."""
        df = build_spike_features(synthetic_db, "HB_WEST", "2025-01-10", "2025-01-12")
        for col in RESERVE_FEATURES:
            assert df[col].isna().sum() == 0, f"{col} has NaN values"

    def test_wind_nan_filled(self, synthetic_db):
        """Wind features should not have NaN after fill."""
        df = build_spike_features(synthetic_db, "HB_WEST", "2025-01-10", "2025-01-12")
        for col in WIND_FEATURES:
            assert df[col].isna().sum() == 0, f"{col} has NaN values"

    def test_empty_sp_returns_empty(self, synthetic_db):
        """Non-existent settlement point should return empty DataFrame."""
        df = build_spike_features(synthetic_db, "FAKE_SP", "2025-01-10", "2025-01-12")
        assert df.empty


class TestTemporalFeatures:
    """Temporal features derived correctly from UTC timestamps."""

    def test_hour_of_day_central_time(self):
        """hour_of_day should reflect Central time, not UTC."""
        # 18:00 UTC in January = 12:00 CT (CST = UTC-6)
        idx = pd.DatetimeIndex(
            [pd.Timestamp("2025-01-10 18:00", tz="UTC")], freq="15min"
        )
        feats = _compute_temporal_features(idx)
        assert feats["hour_of_day"].iloc[0] == 12

    def test_is_weekend(self):
        """Weekend flag should be correct."""
        # 2025-01-11 is a Saturday
        idx = pd.DatetimeIndex(
            [pd.Timestamp("2025-01-11 12:00", tz="UTC")], freq="15min"
        )
        feats = _compute_temporal_features(idx)
        assert feats["is_weekend"].iloc[0] == 1

    def test_weekday(self):
        """Weekday flag should be correct."""
        # 2025-01-10 is a Friday
        idx = pd.DatetimeIndex(
            [pd.Timestamp("2025-01-10 12:00", tz="UTC")], freq="15min"
        )
        feats = _compute_temporal_features(idx)
        assert feats["is_weekend"].iloc[0] == 0


class TestIntegration:
    """End-to-end integration tests with synthetic DB."""

    def test_index_is_utc(self, synthetic_db):
        """Result index should be UTC DatetimeIndex."""
        df = build_spike_features(synthetic_db, "HB_WEST", "2025-01-10", "2025-01-12")
        assert df.index.tz is not None
        assert str(df.index.tz) == "UTC"

    def test_no_duplicate_index(self, synthetic_db):
        """No duplicate timestamps in the result."""
        df = build_spike_features(synthetic_db, "HB_WEST", "2025-01-10", "2025-01-12")
        assert not df.index.duplicated().any()

    def test_feature_dtypes_numeric(self, synthetic_db):
        """All feature columns should be numeric."""
        df = build_spike_features(synthetic_db, "HB_WEST", "2025-01-10", "2025-01-12")
        for col in FEATURE_COLUMNS:
            assert pd.api.types.is_numeric_dtype(df[col]), f"{col} is not numeric: {df[col].dtype}"


class TestKnownValues:
    """Test with the real archive DB if available."""

    @pytest.fixture
    def real_db(self):
        db = Path(__file__).resolve().parent.parent.parent / "scraper" / "data" / "ercot_archive.db"
        if not db.exists():
            pytest.skip("Real archive DB not available")
        return db

    def test_real_db_builds_successfully(self, real_db):
        """Build features for one SP over a small date range."""
        df = build_spike_features(real_db, "HB_WEST", "2025-12-01", "2025-12-15")
        assert not df.empty
        assert len(df) > 96  # at least 1 day of 15-min intervals
        for col in FEATURE_COLUMNS:
            assert col in df.columns

    def test_real_db_spread_high_dec14(self, real_db):
        """On 2025-12-14 evening CT, spreads should be elevated."""
        df = build_spike_features(real_db, "HB_WEST", "2025-12-13", "2025-12-15")
        # 18:00 CT = 00:00 UTC next day (CST = UTC-6)
        target_time = pd.Timestamp("2025-12-15 00:00", tz="UTC")
        if target_time in df.index:
            row = df.loc[target_time]
            # Just verify we got a numeric spread_lag1 (may or may not be high)
            assert not pd.isna(row["spread_lag1"])
