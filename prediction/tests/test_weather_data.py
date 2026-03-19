"""
Tests for the weather data module.

Covers:
- Zone-to-station mapping (all 15 zones)
- Open-Meteo client with mocked HTTP
- Derived weather features (anomaly, delta, wind chill, cold front)
- No-lookahead verification
"""

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from prediction.src.data.weather.stations import WEATHER_STATIONS, get_station_for_zone
from prediction.src.data.weather.zone_weather import compute_weather_features
import prediction.src.data.weather.openmeteo_client as _client_mod


# ---------------------------------------------------------------------------
# Station mapping tests
# ---------------------------------------------------------------------------

class TestStationMapping:
    ALL_ZONES = [
        "LZ_CPS", "LZ_WEST", "HB_WEST", "LZ_HOUSTON", "HB_HOUSTON",
        "HB_NORTH", "LZ_NORTH", "HB_SOUTH", "LZ_SOUTH",
        "HB_BUSAVG", "HB_HUBAVG", "HB_PAN", "LZ_AEN", "LZ_LCRA", "LZ_RAYBN",
    ]

    @pytest.mark.parametrize("zone", ALL_ZONES)
    def test_every_zone_has_station(self, zone):
        station = get_station_for_zone(zone)
        assert station in WEATHER_STATIONS

    def test_specific_mappings(self):
        assert get_station_for_zone("LZ_CPS") == "san_antonio"
        assert get_station_for_zone("HB_WEST") == "midland"
        assert get_station_for_zone("HB_HOUSTON") == "houston"
        assert get_station_for_zone("HB_NORTH") == "dallas"
        assert get_station_for_zone("HB_SOUTH") == "corpus_christi"
        assert get_station_for_zone("LZ_LCRA") == "austin"

    def test_unknown_zone_raises(self):
        with pytest.raises(KeyError):
            get_station_for_zone("NONEXISTENT")

    def test_all_15_zones_covered(self):
        all_zones = set()
        for info in WEATHER_STATIONS.values():
            all_zones.update(info["zones"])
        assert len(all_zones) == 15


# ---------------------------------------------------------------------------
# Open-Meteo client tests (mocked)
# ---------------------------------------------------------------------------

class TestFetchStationYear:
    @patch("prediction.src.data.weather.openmeteo_client.requests.get")
    def test_fetch_returns_dataframe(self, mock_get):
        # Build a mock response with 24 hours of data
        hours = pd.date_range("2023-01-01", periods=24, freq="h")
        mock_json = {
            "hourly": {
                "time": [t.strftime("%Y-%m-%dT%H:%M") for t in hours],
                "temperature_2m": [20.0] * 24,
                "wind_speed_10m": [10.0] * 24,
                "wind_direction_10m": [180.0] * 24,
                "relative_humidity_2m": [50.0] * 24,
                "surface_pressure": [1013.0] * 24,
                "dew_point_2m": [10.0] * 24,
            }
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_json
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        # Import fetch_station_year via importlib to avoid __init__.py
        
        df = _client_mod.fetch_station_year("austin", 2023)

        assert len(df) == 24
        assert "station" in df.columns
        assert df["station"].iloc[0] == "austin"
        assert "temperature_2m" in df.columns

    @patch("prediction.src.data.weather.openmeteo_client.requests.get")
    def test_fetch_passes_correct_params(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"hourly": {"time": [], "temperature_2m": []}}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        
        _client_mod.fetch_station_year("dallas", 2020)

        call_kwargs = mock_get.call_args
        params = call_kwargs.kwargs.get("params") or call_kwargs[1].get("params")
        assert params["latitude"] == 32.78
        assert params["longitude"] == -96.80
        assert params["start_date"] == "2020-01-01"
        assert params["timezone"] == "UTC"


# ---------------------------------------------------------------------------
# SQLite save test
# ---------------------------------------------------------------------------

class TestSaveToSqlite:
    def test_save_and_read_back(self, tmp_path):
        db_path = tmp_path / "test.db"
        df = pd.DataFrame({
            "station": ["austin"] * 3,
            "time": ["2023-01-01T00:00", "2023-01-01T01:00", "2023-01-01T02:00"],
            "temperature_2m": [20.0, 21.0, 22.0],
            "wind_speed_10m": [5.0, 6.0, 7.0],
            "wind_direction_10m": [180.0, 190.0, 200.0],
            "relative_humidity_2m": [50.0, 55.0, 60.0],
            "surface_pressure": [1013.0, 1012.0, 1011.0],
            "dew_point_2m": [10.0, 11.0, 12.0],
        })

        
        _client_mod.save_to_sqlite(df, db_path)

        conn = sqlite3.connect(db_path)
        result = pd.read_sql("SELECT * FROM weather_hourly", conn)
        conn.close()

        assert len(result) == 3
        assert result["station"].iloc[0] == "austin"

    def test_idempotent_upsert(self, tmp_path):
        db_path = tmp_path / "test.db"
        df = pd.DataFrame({
            "station": ["austin"],
            "time": ["2023-01-01T00:00"],
            "temperature_2m": [20.0],
            "wind_speed_10m": [5.0],
            "wind_direction_10m": [180.0],
            "relative_humidity_2m": [50.0],
            "surface_pressure": [1013.0],
            "dew_point_2m": [10.0],
        })

        
        _client_mod.save_to_sqlite(df, db_path)
        # Save again with updated temperature
        df2 = df.copy()
        df2["temperature_2m"] = 25.0
        _client_mod.save_to_sqlite(df2, db_path)

        conn = sqlite3.connect(db_path)
        result = pd.read_sql("SELECT * FROM weather_hourly", conn)
        conn.close()

        assert len(result) == 1
        assert result["temperature_2m"].iloc[0] == 25.0


# ---------------------------------------------------------------------------
# Weather feature tests
# ---------------------------------------------------------------------------

def _make_synthetic_data(n_hours=200):
    """Create synthetic hourly weather data for testing."""
    times = pd.date_range("2023-06-01", periods=n_hours, freq="h", tz="UTC")
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "time": times,
        "temperature_2m": 25.0 + rng.randn(n_hours) * 2,
        "wind_speed_10m": 10.0 + rng.randn(n_hours) * 3,
        "wind_direction_10m": rng.uniform(0, 360, n_hours),
        "relative_humidity_2m": 50.0 + rng.randn(n_hours) * 5,
        "surface_pressure": 1013.0 + rng.randn(n_hours) * 2,
        "dew_point_2m": 15.0 + rng.randn(n_hours) * 2,
    })


class TestWeatherFeatures:
    def test_constant_temperature_anomaly_near_zero(self):
        """For constant temperature, anomaly should be ~0 after warmup."""
        n = 200
        times = pd.date_range("2023-06-01", periods=n, freq="h", tz="UTC")
        df = pd.DataFrame({
            "time": times,
            "temperature_2m": [20.0] * n,
            "wind_speed_10m": [10.0] * n,
            "wind_direction_10m": [180.0] * n,
        })
        result = compute_weather_features(df)
        # After warmup, t_anom should be ~0 for constant temp
        t_anom = result["t_anom"].dropna()
        assert t_anom.abs().max() < 0.01

    def test_delta_t_1h_catches_drop(self):
        """delta_t_1h should detect a temperature drop."""
        n = 50
        times = pd.date_range("2023-06-01", periods=n, freq="h", tz="UTC")
        temps = [25.0] * n
        temps[20] = 15.0  # 10-degree drop at hour 20
        df = pd.DataFrame({
            "time": times,
            "temperature_2m": temps,
            "wind_speed_10m": [10.0] * n,
            "wind_direction_10m": [180.0] * n,
        })
        result = compute_weather_features(df)
        # The drop at hour 20 should show up as -10 in delta_t_1h at hour 21 (shifted)
        delta = result["delta_t_1h"]
        assert delta.min() < -5.0

    def test_wind_chill_formula_in_celsius(self):
        """Wind chill should be in Celsius, less than raw temp at 5°C / 20 km/h."""
        n = 10
        times = pd.date_range("2023-12-01", periods=n, freq="h", tz="UTC")
        df = pd.DataFrame({
            "time": times,
            "temperature_2m": [5.0] * n,
            "wind_speed_10m": [20.0] * n,
            "wind_direction_10m": [180.0] * n,
        })
        result = compute_weather_features(df)
        wc = result["wind_chill"].dropna()
        assert len(wc) > 0
        # Compute expected: NWS formula in F then convert back to C
        t_f = 5.0 * 9.0 / 5.0 + 32.0  # 41 F
        v_mph = 20.0 * 0.621371  # ~12.43 mph
        wc_f = 35.74 + 0.6215 * t_f - 35.75 * v_mph**0.16 + 0.4275 * t_f * v_mph**0.16
        expected_c = (wc_f - 32.0) * 5.0 / 9.0
        np.testing.assert_almost_equal(wc.iloc[0], expected_c, decimal=4)
        # Must be in Celsius range (below raw temp, not in Fahrenheit range)
        assert (wc < 5.0).all()
        assert (wc > -20.0).all()

    def test_wind_chill_equals_temp_when_warm(self):
        """Wind chill = T when conditions not met (T >= 10°C)."""
        n = 10
        times = pd.date_range("2023-07-01", periods=n, freq="h", tz="UTC")
        df = pd.DataFrame({
            "time": times,
            "temperature_2m": [30.0] * n,
            "wind_speed_10m": [20.0] * n,
            "wind_direction_10m": [180.0] * n,
        })
        result = compute_weather_features(df)
        wc = result["wind_chill"].dropna()
        np.testing.assert_array_almost_equal(wc.values, 30.0)

    def test_cold_front_triggers_exact_count(self):
        """Cold front flag should trigger exactly once for our setup."""
        n = 50
        times = pd.date_range("2023-12-01", periods=n, freq="h", tz="UTC")
        temps = [25.0] * n
        winds = [180.0] * n  # South wind
        # Create a cold front at hour 10: 8-degree drop over 3h + north wind
        temps[8] = 22.0
        temps[9] = 18.0
        temps[10] = 14.0  # delta_3h = 14 - 25 = -11
        winds[10] = 350.0  # North wind
        df = pd.DataFrame({
            "time": times,
            "temperature_2m": temps,
            "wind_speed_10m": [15.0] * n,
            "wind_direction_10m": winds,
        })
        result = compute_weather_features(df)
        cf = result["cold_front"]
        assert cf.sum() == 1

    def test_no_lookahead_values(self):
        """Features at time t should use data from t-1, not t."""
        n = 50
        times = pd.date_range("2023-06-01", periods=n, freq="h", tz="UTC")
        temps = [20.0] * n
        temps[10] = 30.0  # spike at hour 10
        df = pd.DataFrame({
            "time": times,
            "temperature_2m": temps,
            "wind_speed_10m": [10.0] * n,
            "wind_direction_10m": [180.0] * n,
        })
        result = compute_weather_features(df)

        # All derived features should be NaN at the first row (due to shift)
        first = result.iloc[0]
        assert pd.isna(first["t_anom"])
        assert pd.isna(first["delta_t_1h"])
        assert pd.isna(first["delta_t_3h"])
        assert pd.isna(first["wind_chill"])
        assert pd.isna(first["cold_front"])

        # At hour 10, delta_t_1h should reflect hour 9→8 change (both 20), not the spike
        t10 = result.iloc[10]
        assert t10["delta_t_1h"] == pytest.approx(0.0, abs=0.01)

        # At hour 11, delta_t_1h should reflect hour 10→9 = 30-20 = +10
        t11 = result.iloc[11]
        assert t11["delta_t_1h"] == pytest.approx(10.0, abs=0.01)

    def test_spring_dst_gap_handled(self):
        """Data with a spring-forward gap should still produce correct features."""
        # Simulate: hours around US spring-forward in UTC (no actual gap in UTC,
        # but test that asfreq handles any missing rows gracefully)
        times = pd.to_datetime([
            "2023-03-12 06:00", "2023-03-12 07:00", "2023-03-12 08:00",
            # skip 09:00 to simulate a missing row
            "2023-03-12 10:00", "2023-03-12 11:00", "2023-03-12 12:00",
        ], utc=True)
        df = pd.DataFrame({
            "time": times,
            "temperature_2m": [10.0, 11.0, 12.0, 14.0, 15.0, 16.0],
            "wind_speed_10m": [5.0] * 6,
            "wind_direction_10m": [180.0] * 6,
        })
        result = compute_weather_features(df)
        # asfreq should have inserted a NaN row at 09:00
        assert len(result) == 7  # 06:00 through 12:00 inclusive
        # delta_t_1h at 11:00 should use 10:00 value (14→NaN gap handled)
        assert "delta_t_1h" in result.columns

    def test_output_columns_present(self):
        df = _make_synthetic_data(50)
        result = compute_weather_features(df)
        for col in ["t_anom", "delta_t_1h", "delta_t_3h", "wind_chill", "cold_front"]:
            assert col in result.columns


# ---------------------------------------------------------------------------
# API error handling tests
# ---------------------------------------------------------------------------

class TestFetchErrorHandling:
    @patch("prediction.src.data.weather.openmeteo_client.requests.get")
    @patch("prediction.src.data.weather.openmeteo_client.time.sleep")
    def test_fetch_all_continues_on_failure(self, mock_sleep, mock_get):
        """fetch_all_stations should skip failed station-years and continue."""
        

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("simulated network failure")
            mock_resp = MagicMock()
            hours = pd.date_range("2023-01-01", periods=24, freq="h")
            mock_resp.json.return_value = {
                "hourly": {
                    "time": [t.strftime("%Y-%m-%dT%H:%M") for t in hours],
                    "temperature_2m": [20.0] * 24,
                    "wind_speed_10m": [10.0] * 24,
                    "wind_direction_10m": [180.0] * 24,
                    "relative_humidity_2m": [50.0] * 24,
                    "surface_pressure": [1013.0] * 24,
                    "dew_point_2m": [10.0] * 24,
                }
            }
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        mock_get.side_effect = side_effect

        # Fetch a single year so the test is fast
        df = _client_mod.fetch_all_stations(start_year=2023, end_year=2023)

        # Should have data despite the first station-year failing
        assert not df.empty
