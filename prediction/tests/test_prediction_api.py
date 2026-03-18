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
    def test_normalize_settlement_point_maps_load_zone_to_hub(self):
        assert main._normalize_settlement_point("lz_west") == "HB_WEST"

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

        features = main._fetch_and_compute_features("HB_WEST")

        assert not features.empty
        assert fake_fetcher.closed is True


class TestEndpoints:
    def test_predict_spike_returns_latest_alert(self, monkeypatch):
        class FakePredictor:
            def __init__(self):
                self.features_df = None
                self.settlement_point = None

            def is_ready(self):
                return True

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

        assert result["settlement_point"] == "HB_WEST"
        assert result["alert"]["spike_probability"] == 0.9
        assert result["alert"]["is_spike"] is True
        assert predictor.features_df is feature_frame
        assert predictor.settlement_point == "hb_west"

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
