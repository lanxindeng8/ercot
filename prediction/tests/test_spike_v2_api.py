"""Tests for Spike V2 predictor and API endpoints."""

import asyncio
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from prediction.src.models.spike_v2_predictor import (
    SpikeV2Predictor,
    _risk_level,
    get_spike_v2_predictor,
)
from prediction.src import main


# ---------------------------------------------------------------------------
# Risk level classification
# ---------------------------------------------------------------------------


class TestRiskLevel:
    def test_critical(self):
        assert _risk_level(0.85) == "critical"
        assert _risk_level(0.80) == "critical"

    def test_high(self):
        assert _risk_level(0.7) == "high"
        assert _risk_level(0.6) == "high"

    def test_medium(self):
        assert _risk_level(0.5) == "medium"
        assert _risk_level(0.3) == "medium"

    def test_low(self):
        assert _risk_level(0.29) == "low"
        assert _risk_level(0.0) == "low"

    def test_boundary_values(self):
        assert _risk_level(0.8) == "critical"
        assert _risk_level(0.6) == "high"
        assert _risk_level(0.3) == "medium"
        assert _risk_level(0.2999) == "low"


# ---------------------------------------------------------------------------
# Predictor unit tests (mock models)
# ---------------------------------------------------------------------------


def _make_fake_model(prob: float = 0.5):
    """Create a mock LightGBM Booster that returns a fixed probability."""
    model = MagicMock()
    model.predict.return_value = np.array([prob])
    model.feature_name.return_value = ["lmp_lag1", "spread_mean_4", "prc_ramp_1h", "temp_2m"]
    model.feature_importance.return_value = np.array([100.0, 80.0, 60.0, 40.0])
    return model


def _make_fake_features():
    """Create a minimal feature DataFrame matching spike feature columns."""
    from prediction.src.features.spike_features import FEATURE_COLUMNS, LABEL_COLUMNS

    index = pd.date_range("2026-03-19 12:00", periods=4, freq="15min", tz="UTC")
    data = {col: np.random.randn(4) for col in FEATURE_COLUMNS}
    data["spike_event"] = [False, False, True, False]
    data["lead_spike_60"] = [0, 1, 1, 0]
    data["regime"] = ["Normal", "Tight", "Scarcity", "Normal"]
    return pd.DataFrame(data, index=index)


class TestSpikeV2Predictor:
    def test_predict_returns_expected_fields(self):
        predictor = SpikeV2Predictor.__new__(SpikeV2Predictor)
        predictor.models = {"HB_WEST": _make_fake_model(0.75)}
        predictor.metrics = {}
        predictor._feature_cache = {}
        predictor._cache_ttl = 300
        predictor.db_path = Path("/dev/null")

        with patch.object(predictor, "_get_features", return_value=_make_fake_features()):
            result = predictor.predict("HB_WEST")

        assert result["settlement_point"] == "HB_WEST"
        assert result["probability"] == 0.75
        assert result["is_alert"] is True
        assert result["risk_level"] == "high"
        assert result["lead_time_minutes"] == 60
        assert result["model_version"] == "v2_lead60"
        assert isinstance(result["top_drivers"], list)
        assert len(result["top_drivers"]) == 3
        assert "timestamp" in result

    def test_predict_low_probability(self):
        predictor = SpikeV2Predictor.__new__(SpikeV2Predictor)
        predictor.models = {"HB_WEST": _make_fake_model(0.1)}
        predictor.metrics = {}
        predictor._feature_cache = {}
        predictor._cache_ttl = 300
        predictor.db_path = Path("/dev/null")

        with patch.object(predictor, "_get_features", return_value=_make_fake_features()):
            result = predictor.predict("HB_WEST")

        assert result["probability"] == 0.1
        assert result["is_alert"] is False
        assert result["risk_level"] == "low"

    def test_predict_missing_model_returns_error(self):
        predictor = SpikeV2Predictor.__new__(SpikeV2Predictor)
        predictor.models = {}
        predictor.metrics = {}
        predictor._feature_cache = {}
        predictor._cache_ttl = 300
        predictor.db_path = Path("/dev/null")

        result = predictor.predict("NONEXISTENT")
        assert result["error"] is not None
        assert result["probability"] is None
        assert result["risk_level"] == "unknown"

    def test_predict_empty_features_returns_error(self):
        predictor = SpikeV2Predictor.__new__(SpikeV2Predictor)
        predictor.models = {"HB_WEST": _make_fake_model(0.5)}
        predictor.metrics = {}
        predictor._feature_cache = {}
        predictor._cache_ttl = 300
        predictor.db_path = Path("/dev/null")

        with patch.object(predictor, "_get_features", return_value=pd.DataFrame()):
            result = predictor.predict("HB_WEST")

        assert result["error"] is not None
        assert result["probability"] is None

    def test_predict_all_sorted_by_probability(self):
        predictor = SpikeV2Predictor.__new__(SpikeV2Predictor)
        predictor.models = {
            "HB_WEST": _make_fake_model(0.3),
            "LZ_CPS": _make_fake_model(0.9),
            "HB_NORTH": _make_fake_model(0.6),
        }
        predictor.metrics = {}
        predictor._feature_cache = {}
        predictor._cache_ttl = 300
        predictor.db_path = Path("/dev/null")

        with patch.object(predictor, "_get_features", return_value=_make_fake_features()):
            results = predictor.predict_all()

        probs = [r["probability"] for r in results]
        assert probs == sorted(probs, reverse=True)
        assert results[0]["settlement_point"] == "LZ_CPS"
        assert results[0]["probability"] == 0.9

    def test_predict_alerts_filters_below_threshold(self):
        predictor = SpikeV2Predictor.__new__(SpikeV2Predictor)
        predictor.models = {
            "HB_WEST": _make_fake_model(0.1),
            "LZ_CPS": _make_fake_model(0.5),
            "HB_NORTH": _make_fake_model(0.8),
        }
        predictor.metrics = {}
        predictor._feature_cache = {}
        predictor._cache_ttl = 300
        predictor.db_path = Path("/dev/null")

        with patch.object(predictor, "_get_features", return_value=_make_fake_features()):
            alerts = predictor.predict_alerts(threshold=0.3)

        sps = {a["settlement_point"] for a in alerts}
        assert "HB_WEST" not in sps  # 0.1 < 0.3
        assert "LZ_CPS" in sps
        assert "HB_NORTH" in sps

    def test_is_ready_with_models(self):
        predictor = SpikeV2Predictor.__new__(SpikeV2Predictor)
        predictor.models = {"HB_WEST": _make_fake_model()}
        assert predictor.is_ready() is True

    def test_is_ready_without_models(self):
        predictor = SpikeV2Predictor.__new__(SpikeV2Predictor)
        predictor.models = {}
        assert predictor.is_ready() is False

    def test_has_model(self):
        predictor = SpikeV2Predictor.__new__(SpikeV2Predictor)
        predictor.models = {"HB_WEST": _make_fake_model()}
        assert predictor.has_model("HB_WEST") is True
        assert predictor.has_model("hb_west") is True  # .upper() normalizes
        assert predictor.has_model("NONEXISTENT") is False

    def test_top_drivers_returns_feature_names(self):
        predictor = SpikeV2Predictor.__new__(SpikeV2Predictor)
        predictor.models = {"HB_WEST": _make_fake_model()}
        drivers = predictor._top_drivers("HB_WEST", n=3)
        assert len(drivers) == 3
        assert drivers[0] == "lmp_lag1"  # highest importance

    def test_get_model_info(self):
        predictor = SpikeV2Predictor.__new__(SpikeV2Predictor)
        predictor.models = {"HB_WEST": _make_fake_model(), "LZ_CPS": _make_fake_model()}
        predictor.metrics = {}
        info = predictor.get_model_info()
        assert info["models_loaded"] == 2
        assert info["version"] == "v2_lead60"
        assert "HB_WEST" in info["settlement_points"]

    def test_graceful_handling_missing_model_files(self):
        """Predictor should load without errors even when model dir is empty."""
        predictor = SpikeV2Predictor(
            model_dir=Path("/tmp/nonexistent_spike_models"),
            db_path=Path("/dev/null"),
        )
        assert predictor.is_ready() is False
        assert predictor.available_settlement_points() == []


# ---------------------------------------------------------------------------
# API endpoint tests (mock predictor)
# ---------------------------------------------------------------------------


def _make_fake_v2_predictor(predictions=None):
    """Create a mock SpikeV2Predictor for API tests."""
    predictor = MagicMock()
    predictor.is_ready.return_value = True
    predictor.has_model.return_value = True
    predictor.available_settlement_points.return_value = ["HB_WEST", "LZ_CPS"]

    default_pred = {
        "settlement_point": "HB_WEST",
        "probability": 0.75,
        "is_alert": True,
        "risk_level": "high",
        "regime": "Tight",
        "lead_time_minutes": 60,
        "timestamp": "2026-03-19T12:00:00Z",
        "model_version": "v2_lead60",
        "top_drivers": ["lmp_ramp_4h", "spread_mean_4", "prc_ramp_1h"],
    }
    predictor.predict.return_value = predictions or default_pred
    predictor.predict_all.return_value = [
        {**default_pred, "settlement_point": "LZ_CPS", "probability": 0.9, "risk_level": "critical"},
        {**default_pred, "settlement_point": "HB_WEST", "probability": 0.75},
    ]
    predictor.predict_alerts.return_value = [
        {**default_pred, "settlement_point": "LZ_CPS", "probability": 0.9, "risk_level": "critical"},
        {**default_pred, "settlement_point": "HB_WEST", "probability": 0.75},
    ]
    return predictor


class TestSpikeV2Endpoints:
    def test_single_sp_endpoint(self, monkeypatch):
        predictor = _make_fake_v2_predictor()
        monkeypatch.setattr(main, "get_spike_v2_predictor", lambda: predictor)

        result = asyncio.run(main.predict_spike_v2("HB_WEST"))
        assert result["status"] == "success"
        assert result["probability"] == 0.75
        assert result["settlement_point"] == "HB_WEST"
        assert result["risk_level"] == "high"
        assert result["model_version"] == "v2_lead60"
        assert "top_drivers" in result

    def test_all_endpoint_returns_sorted(self, monkeypatch):
        predictor = _make_fake_v2_predictor()
        monkeypatch.setattr(main, "get_spike_v2_predictor", lambda: predictor)

        result = asyncio.run(main.predict_spike_v2_all())
        assert result["status"] == "success"
        assert result["count"] == 2
        preds = result["predictions"]
        assert preds[0]["probability"] >= preds[1]["probability"]

    def test_alerts_endpoint_filters(self, monkeypatch):
        predictor = _make_fake_v2_predictor()
        monkeypatch.setattr(main, "get_spike_v2_predictor", lambda: predictor)

        result = asyncio.run(main.predict_spike_v2_alerts())
        assert result["status"] == "success"
        for alert in result["alerts"]:
            assert alert["probability"] >= 0.3

    def test_single_sp_not_loaded(self, monkeypatch):
        predictor = _make_fake_v2_predictor()
        predictor.is_ready.return_value = False
        monkeypatch.setattr(main, "get_spike_v2_predictor", lambda: predictor)

        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc:
            asyncio.run(main.predict_spike_v2("HB_WEST"))
        assert exc.value.status_code == 503

    def test_single_sp_missing_model(self, monkeypatch):
        predictor = _make_fake_v2_predictor()
        predictor.has_model.return_value = False
        monkeypatch.setattr(main, "get_spike_v2_predictor", lambda: predictor)

        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc:
            asyncio.run(main.predict_spike_v2("HB_WEST"))
        assert exc.value.status_code == 404

    def test_single_sp_data_error(self, monkeypatch):
        predictor = _make_fake_v2_predictor(
            predictions={"settlement_point": "HB_WEST", "error": "No feature data", "probability": None, "is_alert": False, "risk_level": "unknown"}
        )
        monkeypatch.setattr(main, "get_spike_v2_predictor", lambda: predictor)

        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc:
            asyncio.run(main.predict_spike_v2("HB_WEST"))
        assert exc.value.status_code == 502

    def test_invalid_settlement_point(self, monkeypatch):
        predictor = _make_fake_v2_predictor()
        monkeypatch.setattr(main, "get_spike_v2_predictor", lambda: predictor)

        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc:
            asyncio.run(main.predict_spike_v2("INVALID_SP"))
        assert exc.value.status_code == 400

    def test_existing_spike_v1_endpoint_still_works(self, monkeypatch):
        """Ensure the old /predictions/spike endpoint is not broken."""
        from prediction.src.models.spike_predictor import SpikeAlert
        from datetime import datetime

        class FakeV1Predictor:
            def is_ready(self):
                return True
            def has_model(self, sp):
                return True
            def predict(self, features_df, sp):
                return [SpikeAlert(sp, 0.4, False, "low", 0.5, datetime.utcnow())]

        monkeypatch.setattr(main, "get_spike_predictor", lambda: FakeV1Predictor())
        monkeypatch.setattr(main, "_fetch_and_compute_features", lambda sp: pd.DataFrame({"a": [1]}))

        result = asyncio.run(main.predict_spike("HB_WEST"))
        assert result["status"] == "success"
        assert result["model"] == "Spike Detection CatBoost"


# ---------------------------------------------------------------------------
# Prediction runner integration
# ---------------------------------------------------------------------------


class TestSpikeV2Runner:
    def test_run_spike_v2_predictions(self):
        from prediction.scripts.run_predictions import run_spike_v2_predictions

        mock_client = MagicMock()
        mock_conn = MagicMock()
        mock_conn.executemany.return_value = MagicMock(rowcount=3)

        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "status": "success",
                "predictions": [
                    {"settlement_point": "HB_WEST", "probability": 0.8, "is_alert": True,
                     "risk_level": "critical", "regime": "Scarcity", "top_drivers": ["a", "b", "c"],
                     "model_version": "v2_lead60", "timestamp": "2026-03-19T12:00:00Z"},
                    {"settlement_point": "LZ_CPS", "probability": 0.2, "is_alert": False,
                     "risk_level": "low", "regime": "Normal", "top_drivers": ["a", "b", "c"],
                     "model_version": "v2_lead60", "timestamp": "2026-03-19T12:00:00Z"},
                ],
            },
        )
        mock_client.get.return_value.raise_for_status = MagicMock()

        count = run_spike_v2_predictions(mock_client, mock_conn)
        assert count == 2
