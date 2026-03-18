"""Targeted tests for the Sprint 3 prediction API endpoints."""

import asyncio
import sys
import types
from datetime import datetime

import pandas as pd
import pytest
from fastapi import HTTPException

from prediction.src import main
from prediction.src.models.spike_predictor import SpikeAlert


def _market_frame(column: str) -> pd.DataFrame:
    index = pd.date_range("2025-01-01 00:00:00", periods=48, freq="h")
    frame = pd.DataFrame(
        {
            column: range(48),
            "date": index.date,
        },
        index=index,
    )
    frame.index.name = "timestamp"
    return frame


class TestValidationHelpers:
    def test_normalize_settlement_point_preserves_load_zone(self):
        assert main._normalize_settlement_point("lz_west") == "LZ_WEST"

    def test_normalize_settlement_point_rejects_unsupported_values(self):
        with pytest.raises(HTTPException) as exc:
            main._normalize_settlement_point("HB_WEST'; DROP TABLE dam_lmp;--")

        assert exc.value.status_code == 400

    def test_parse_horizons_rejects_invalid_values(self):
        with pytest.raises(HTTPException) as exc:
            main._parse_horizons("1h,2h")

        assert exc.value.status_code == 400
        assert "Unsupported horizons" in exc.value.detail


class TestFeatureFetching:
    def test_fetch_and_compute_features_closes_fetcher(self, monkeypatch):
        class FakeFetcher:
            def __init__(self):
                self.closed = False

            def fetch_dam_prices(self, settlement_point, start_date):
                assert settlement_point == "HB_WEST"
                assert isinstance(start_date, datetime)
                return _market_frame("dam_price")

            def fetch_rtm_prices(self, settlement_point, start_date):
                assert settlement_point == "HB_WEST"
                assert isinstance(start_date, datetime)
                return _market_frame("rtm_price")

            def close(self):
                self.closed = True

        fake_fetcher = FakeFetcher()

        monkeypatch.setitem(
            sys.modules,
            "prediction.src.data.influxdb_fetcher",
            types.SimpleNamespace(create_fetcher_from_env=lambda: fake_fetcher),
        )
        monkeypatch.setattr(
            main,
            "compute_features",
            lambda dam_hourly, rtm_hourly: pd.DataFrame(
                {
                    "delivery_date": ["2025-01-02"],
                    "hour_of_day": [1],
                }
            ),
        )

        # Make SQLite path fail so it falls through to InfluxDB
        monkeypatch.setattr(
            main,
            "_try_sqlite_features",
            lambda sp: None,
        )

        features = main._fetch_and_compute_features("HB_WEST")

        assert not features.empty
        assert fake_fetcher.closed is True

    def test_try_sqlite_features_rejects_stale_rtm_data(self, monkeypatch):
        stale_index = pd.date_range("2025-01-01 00:00:00", periods=48, freq="h")
        fresh_index = pd.date_range("2026-03-17 00:00:00", periods=48, freq="h")

        stale_rtm = pd.DataFrame(
            {"rtm_price": range(48), "hour": range(1, 49), "date": stale_index.date},
            index=stale_index,
        )
        stale_rtm.index.name = "timestamp"

        dam_frame = pd.DataFrame(
            {"dam_price": range(48), "hour": range(1, 49), "date": fresh_index.date},
            index=fresh_index,
        )
        dam_frame.index.name = "timestamp"

        class FakeFetcher:
            def fetch_dam_prices(self, settlement_point, start_date):
                return dam_frame

            def fetch_rtm_prices(self, settlement_point, start_date):
                return stale_rtm

            def close(self):
                pass

        monkeypatch.setitem(
            sys.modules,
            "prediction.src.data.sqlite_fetcher",
            types.SimpleNamespace(create_sqlite_fetcher=lambda: FakeFetcher()),
        )

        assert main._try_sqlite_features("HB_WEST") is None

    def test_raw_to_hourly_preserves_sqlite_local_hours(self):
        dam_index = pd.to_datetime(["2025-01-01 06:00:00", "2025-01-01 07:00:00"])
        dam_raw = pd.DataFrame(
            {
                "dam_price": [25.0, 30.0],
                "hour": [1, 2],
                "date": [pd.Timestamp("2025-01-01").date(), pd.Timestamp("2025-01-01").date()],
            },
            index=dam_index,
        )
        dam_raw.index.name = "timestamp"

        rtm_index = pd.to_datetime(["2025-01-01 06:00:00", "2025-01-01 07:00:00"])
        rtm_raw = pd.DataFrame(
            {
                "rtm_price": [15.0, 17.0],
                "hour": [1, 2],
                "date": [pd.Timestamp("2025-01-01").date(), pd.Timestamp("2025-01-01").date()],
            },
            index=rtm_index,
        )
        rtm_raw.index.name = "timestamp"

        dam_hourly, rtm_hourly = main._raw_to_hourly(dam_raw, rtm_raw)

        assert dam_hourly["hour_ending"].tolist() == [1, 2]
        assert rtm_hourly["hour_ending"].tolist() == [1, 2]
        assert dam_hourly["delivery_date"].tolist() == ["2025-01-01", "2025-01-01"]


class TestEndpoints:
    def test_predict_spike_returns_latest_alert(self, monkeypatch):
        class FakePredictor:
            def __init__(self):
                self.features_df = None
                self.settlement_point = None

            def is_ready(self):
                return True
            def has_model(self, sp):
                return True
            def missing_model_message(self, sp):
                return f"No model for {sp}"

            def available_settlement_points(self):
                return ["hb_west"]

            def predict(self, features_df, settlement_point):
                self.features_df = features_df
                self.settlement_point = settlement_point
                return [
                    SpikeAlert("hb_west", 0.1, False, "low", 0.5, datetime.utcnow()),
                    SpikeAlert("hb_west", 0.2, False, "low", 0.5, datetime.utcnow()),
                    SpikeAlert("hb_west", 0.9, True, "high", 0.5, datetime.utcnow()),
                ]

        predictor = FakePredictor()
        feature_frame = pd.DataFrame({"hour_of_day": [1, 2, 3]})

        monkeypatch.setattr(main, "get_spike_predictor", lambda: predictor)
        monkeypatch.setattr(main, "_fetch_and_compute_features", lambda settlement_point: feature_frame)

        result = asyncio.run(main.predict_spike("LZ_WEST"))

        assert result["settlement_point"] == "LZ_WEST"
        assert result["alert"]["spike_probability"] == 0.9
        assert result["alert"]["is_spike"] is True
        assert predictor.features_df is feature_frame
        assert predictor.settlement_point == "lz_west"

    def test_predict_delta_spread_requires_loaded_models(self, monkeypatch):
        class FakePredictor:
            def get_model_info(self):
                return {
                    "regression_loaded": False,
                    "binary_loaded": False,
                    "multiclass_loaded": False,
                }

        monkeypatch.setattr(main, "get_predictor", lambda: FakePredictor())

        with pytest.raises(HTTPException) as exc:
            asyncio.run(main.predict_delta_spread(settlement_point="LZ_WEST"))

        assert exc.value.status_code == 503

    def test_predict_delta_spread_rejects_synthetic_inference(self, monkeypatch):
        class FakePredictor:
            def get_model_info(self):
                return {
                    "regression_loaded": True,
                    "binary_loaded": True,
                    "multiclass_loaded": True,
                }

        monkeypatch.setattr(main, "get_predictor", lambda: FakePredictor())

        with pytest.raises(HTTPException) as exc:
            asyncio.run(main.predict_delta_spread(settlement_point="LZ_WEST"))

        assert exc.value.status_code == 503
        assert "synthetic random features" in exc.value.detail
