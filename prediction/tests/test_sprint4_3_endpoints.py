"""Tests for Sprint 4.3 — Wind, Load, and BESS prediction endpoints."""

import asyncio
import sys
import types
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException

from prediction.src import helpers
from prediction.src.routers import predictions, models as models_router, system
from prediction.src.models.wind_predictor import WindPrediction
from prediction.src.models.load_predictor import LoadPrediction, FEATURE_COLS
from prediction.src.models.bess_predictor import BessScheduleEntry, BessScheduleResult


# ---------------------------------------------------------------------------
# Wind endpoint tests
# ---------------------------------------------------------------------------

class TestWindEndpoint:
    def test_predict_wind_returns_24_predictions(self, monkeypatch):
        class FakeWindPredictor:
            def is_ready(self):
                return True
            def has_model(self, sp):
                return True
            def missing_model_message(self, sp):
                return f"No model for {sp}"

            @property
            def metadata(self):
                return {"feature_names": ["hour", "hour_sin", "hour_cos"]}

            def predict(self, features_df):
                return [
                    WindPrediction(hour_ending=h + 1, predicted_mw=5000.0 + h * 100,
                                   lower_bound_mw=4000.0 + h * 80,
                                   upper_bound_mw=6000.0 + h * 120)
                    for h in range(24)
                ]

        predictor = FakeWindPredictor()
        monkeypatch.setattr(predictions, "get_wind_predictor", lambda: predictor)
        monkeypatch.setattr(predictions, "build_wind_features", lambda: pd.DataFrame({"hour": range(24)}))

        result = asyncio.run(predictions.predict_wind())

        assert result["status"] == "success"
        assert result["model"] == "Wind GBM Quantile"
        assert len(result["predictions"]) == 24
        assert result["predictions"][0]["predicted_mw"] == 5000.0
        assert result["predictions"][0]["lower_bound_mw"] == 4000.0
        assert result["predictions"][0]["upper_bound_mw"] == 6000.0

    def test_predict_wind_503_when_model_not_loaded(self, monkeypatch):
        class FakeWindPredictor:
            def is_ready(self):
                return False

        monkeypatch.setattr(predictions, "get_wind_predictor", lambda: FakeWindPredictor())

        with pytest.raises(HTTPException) as exc:
            asyncio.run(predictions.predict_wind())
        assert exc.value.status_code == 503

    def test_wind_model_info_endpoint(self, monkeypatch):
        class FakeWindPredictor:
            def is_ready(self):
                return True
            def has_model(self, sp):
                return True
            def missing_model_message(self, sp):
                return f"No model for {sp}"

            def get_model_info(self):
                return {
                    "model_type": "Wind GBM",
                    "quantiles": [0.1, 0.5, 0.9],
                    "feature_count": 57,
                }

        monkeypatch.setattr(models_router, "get_wind_predictor", lambda: FakeWindPredictor())

        result = asyncio.run(models_router.wind_model_info())
        assert result.model_name == "wind"
        assert result.status == "loaded"
        assert result.info["quantiles"] == [0.1, 0.5, 0.9]


# ---------------------------------------------------------------------------
# Load endpoint tests
# ---------------------------------------------------------------------------

class TestLoadEndpoint:
    def test_predict_load_returns_24_predictions(self, monkeypatch):
        class FakeLoadPredictor:
            def is_ready(self):
                return True
            def has_model(self, sp):
                return True
            def missing_model_message(self, sp):
                return f"No model for {sp}"

            def predict(self, features_df):
                return [
                    LoadPrediction(hour_ending=h + 1, predicted_load_mw=45000.0 + h * 200)
                    for h in range(24)
                ]

        predictor = FakeLoadPredictor()
        monkeypatch.setattr(predictions, "get_load_predictor", lambda: predictor)
        monkeypatch.setattr(predictions, "build_load_features", lambda: pd.DataFrame({col: [0.0] * 24 for col in FEATURE_COLS}))

        result = asyncio.run(predictions.predict_load())

        assert result["status"] == "success"
        assert result["model"] == "Load CatBoost+LightGBM Ensemble"
        assert len(result["predictions"]) == 24
        assert result["predictions"][0]["predicted_load_mw"] == 45000.0

    def test_predict_load_503_when_model_not_loaded(self, monkeypatch):
        class FakeLoadPredictor:
            def is_ready(self):
                return False

        monkeypatch.setattr(predictions, "get_load_predictor", lambda: FakeLoadPredictor())

        with pytest.raises(HTTPException) as exc:
            asyncio.run(predictions.predict_load())
        assert exc.value.status_code == 503

    def test_load_model_info_endpoint(self, monkeypatch):
        class FakeLoadPredictor:
            def is_ready(self):
                return True
            def has_model(self, sp):
                return True
            def missing_model_message(self, sp):
                return f"No model for {sp}"

            def get_model_info(self):
                return {
                    "model_type": "Load CatBoost+LightGBM Ensemble",
                    "models_loaded": ["catboost", "lightgbm"],
                    "feature_count": 35,
                }

        monkeypatch.setattr(models_router, "get_load_predictor", lambda: FakeLoadPredictor())

        result = asyncio.run(models_router.load_model_info())
        assert result.model_name == "load"
        assert result.status == "loaded"


# ---------------------------------------------------------------------------
# BESS endpoint tests
# ---------------------------------------------------------------------------

class TestBessEndpoint:
    def _make_bess_mocks(self, monkeypatch):
        """Set up standard mocks for BESS endpoint tests."""

        class FakeDamPredictor:
            def is_ready(self):
                return True
            def has_model(self, sp):
                return True
            def missing_model_message(self, sp):
                return f"No model for {sp}"

            def available_settlement_points(self):
                return ["hb_west"]

            def predict(self, target_rows, sp_key):
                from dataclasses import dataclass

                @dataclass
                class DamPred:
                    hour_ending: int
                    predicted_price: float

                return [DamPred(h + 1, 30.0 + h * 2) for h in range(24)]

            def get_model_info(self):
                return {"models_loaded": ["hb_west"], "settlement_points": ["hb_west"]}

        class FakeBessPredictor:
            def is_ready(self):
                return True
            def has_model(self, sp):
                return True
            def missing_model_message(self, sp):
                return f"No model for {sp}"

            def optimize(self, dam_prices):
                schedule = [
                    BessScheduleEntry(
                        hour_ending=h + 1,
                        action="charge" if dam_prices[h] < 50 else "discharge",
                        power_mw=-2.5 if dam_prices[h] < 50 else 2.5,
                        soc_pct=50.0,
                        dam_price=dam_prices[h],
                    )
                    for h in range(24)
                ]
                return BessScheduleResult(
                    schedule=schedule,
                    total_revenue=150.0,
                    status="Optimal",
                    solve_time=0.05,
                    config={"E_max_mwh": 10.0, "P_max_mw": 2.5},
                )

            def get_model_info(self):
                return {"model_type": "BESS LP", "optimizer_loaded": True}

        features = pd.DataFrame({
            "delivery_date": ["2025-01-02"] * 24,
            "hour_ending": list(range(1, 25)),
            "hour_of_day": list(range(24)),
        })

        monkeypatch.setattr(predictions, "get_dam_v2_predictor", lambda: FakeDamPredictor())
        monkeypatch.setattr(predictions, "get_bess_predictor", lambda: FakeBessPredictor())
        monkeypatch.setattr(predictions, "fetch_and_compute_features", lambda sp: features)

    def test_predict_bess_returns_schedule_with_revenue(self, monkeypatch):
        self._make_bess_mocks(monkeypatch)

        result = asyncio.run(predictions.predict_bess(settlement_point="HB_WEST"))

        assert result["status"] == "success"
        assert result["model"] == "BESS LP Optimizer"
        assert result["settlement_point"] == "HB_WEST"
        assert result["optimization"]["total_revenue"] == 150.0
        assert result["optimization"]["status"] == "Optimal"
        assert len(result["schedule"]) == 24
        assert result["schedule"][0]["action"] in ("charge", "discharge", "idle")

    def test_predict_bess_503_when_dam_not_loaded(self, monkeypatch):
        class FakeDamPredictor:
            def is_ready(self):
                return False

        monkeypatch.setattr(predictions, "get_dam_v2_predictor", lambda: FakeDamPredictor())
        monkeypatch.setattr(predictions, "get_bess_predictor", lambda: type("B", (), {"is_ready": lambda self: True})())

        with pytest.raises(HTTPException) as exc:
            asyncio.run(predictions.predict_bess(settlement_point="HB_WEST"))
        assert exc.value.status_code == 503

    def test_predict_bess_503_when_optimizer_not_loaded(self, monkeypatch):
        class FakeDamPredictor:
            def is_ready(self):
                return True
            def has_model(self, sp):
                return True
            def missing_model_message(self, sp):
                return f"No model for {sp}"

            def available_settlement_points(self):
                return ["hb_west"]

        class FakeBessPredictor:
            def is_ready(self):
                return False

        monkeypatch.setattr(predictions, "get_dam_v2_predictor", lambda: FakeDamPredictor())
        monkeypatch.setattr(predictions, "get_bess_predictor", lambda: FakeBessPredictor())

        with pytest.raises(HTTPException) as exc:
            asyncio.run(predictions.predict_bess(settlement_point="HB_WEST"))
        assert exc.value.status_code == 503

    def test_bess_model_info_endpoint(self, monkeypatch):
        class FakeBessPredictor:
            def is_ready(self):
                return True
            def has_model(self, sp):
                return True
            def missing_model_message(self, sp):
                return f"No model for {sp}"

            def get_model_info(self):
                return {
                    "model_type": "BESS LP Optimizer (PuLP CBC)",
                    "optimizer_loaded": True,
                    "battery_config": {"E_max_mwh": 10.0},
                }

        monkeypatch.setattr(models_router, "get_bess_predictor", lambda: FakeBessPredictor())

        result = asyncio.run(models_router.bess_model_info())
        assert result.model_name == "bess"
        assert result.status == "loaded"

    def test_predict_bess_rejects_unsupported_point(self, monkeypatch):
        class FakeDamPredictor:
            def is_ready(self):
                return True
            def has_model(self, sp):
                return True
            def missing_model_message(self, sp):
                return f"No model for {sp}"

            def available_settlement_points(self):
                return ["hb_west"]

        class FakeBessPredictor:
            def is_ready(self):
                return True
            def has_model(self, sp):
                return True
            def missing_model_message(self, sp):
                return f"No model for {sp}"

        monkeypatch.setattr(predictions, "get_dam_v2_predictor", lambda: FakeDamPredictor())
        monkeypatch.setattr(predictions, "get_bess_predictor", lambda: FakeBessPredictor())

        with pytest.raises(HTTPException) as exc:
            asyncio.run(predictions.predict_bess(settlement_point="FAKE_NODE"))
        assert exc.value.status_code == 400


# ---------------------------------------------------------------------------
# Health endpoint integration
# ---------------------------------------------------------------------------

class TestHealthWithNewModels:
    def test_health_includes_wind_load_bess(self, monkeypatch):
        # Mock all predictors
        class FakePredictor:
            def is_ready(self):
                return True
            def has_model(self, sp):
                return True
            def missing_model_message(self, sp):
                return f"No model for {sp}"

            def get_model_info(self):
                return {
                    "models_loaded": ["test"],
                    "settlement_points": ["hb_west"],
                    "regression_loaded": True,
                    "binary_loaded": True,
                    "multiclass_loaded": True,
                    "feature_count": 10,
                    "quantiles": [0.1, 0.5, 0.9],
                    "optimizer_loaded": True,
                }

        fake = FakePredictor()
        monkeypatch.setattr(system, "get_predictor", lambda: fake)
        monkeypatch.setattr(system, "get_dam_v2_predictor", lambda: fake)
        monkeypatch.setattr(system, "get_rtm_predictor", lambda: fake)
        monkeypatch.setattr(system, "get_spike_predictor", lambda: fake)
        monkeypatch.setattr(system, "get_wind_predictor", lambda: fake)
        monkeypatch.setattr(system, "get_load_predictor", lambda: fake)
        monkeypatch.setattr(system, "get_bess_predictor", lambda: fake)

        result = asyncio.run(system.health_check())

        model_names = [m.name for m in result.models]
        assert "wind" in model_names
        assert "load" in model_names
        assert "bess" in model_names
        assert len(result.models) == 7
        assert result.version == "4.0.0"


# ---------------------------------------------------------------------------
# Feature builder tests
# ---------------------------------------------------------------------------

class TestFeatureBuilders:
    def test_build_wind_features_returns_24_rows(self, monkeypatch):
        class FakeWindPredictor:
            @property
            def metadata(self):
                return {"feature_names": ["hour", "hour_sin", "hour_cos", "is_weekend"]}

        monkeypatch.setattr(
            "prediction.src.models.wind_predictor.get_wind_predictor",
            lambda: FakeWindPredictor(),
        )

        df = helpers.build_wind_features()
        assert len(df) == 24
        assert "hour" in df.columns
        assert "hour_sin" in df.columns

    def test_build_load_features_returns_24_rows_with_all_columns(self):
        df = helpers.build_load_features()
        assert len(df) == 24
        for col in FEATURE_COLS:
            assert col in df.columns, f"Missing column: {col}"

    def test_latest_complete_delivery_rows_prefers_full_day(self):
        df = pd.DataFrame({
            "delivery_date": ["2025-01-01"] * 24 + ["2025-01-02"] * 12,
            "hour_ending": list(range(1, 25)) + list(range(1, 13)),
            "value": list(range(36)),
        })

        result = helpers.latest_complete_delivery_rows(df)
        assert len(result) == 24
        assert result["delivery_date"].nunique() == 1
        assert result["delivery_date"].iloc[0] == "2025-01-01"
