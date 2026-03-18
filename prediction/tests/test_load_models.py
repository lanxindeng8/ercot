"""
Tests for load forecast pipeline.

Tests feature building, training script imports, and predictor class.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make load model modules importable
sys.path.insert(0, str(Path(__file__).parent.parent / "models" / "load"))


class _FakeModel:
    """Picklable fake model for testing."""
    def __init__(self, values):
        self._values = values
    def predict(self, X):
        return np.array(self._values)


def _import_load_predictor():
    """Import load_predictor directly to avoid models/__init__.py cascade."""
    spec = importlib.util.spec_from_file_location(
        "load_predictor",
        str(Path(__file__).parent.parent / "src" / "models" / "load_predictor.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Feature building ──────────────────────────────────────────────────────

class TestFeatureBuilding:
    """Test feature extraction functions."""

    def _make_load_df(self, hours: int = 200) -> pd.DataFrame:
        """Create synthetic hourly load data."""
        timestamps = pd.date_range("2024-01-01", periods=hours, freq="h")
        np.random.seed(42)
        base = 40000 + 10000 * np.sin(np.arange(hours) * 2 * np.pi / 24)
        noise = np.random.normal(0, 500, hours)
        return pd.DataFrame({
            "timestamp": timestamps,
            "total_load_mw": base + noise,
        })

    def test_temporal_features(self):
        from build_features_from_sqlite import build_temporal_features

        df = self._make_load_df()
        result = build_temporal_features(df)

        assert "hour_of_day" in result.columns
        assert "day_of_week" in result.columns
        assert "month" in result.columns
        assert "is_weekend" in result.columns
        assert "is_peak_hour" in result.columns
        assert "season" in result.columns
        assert "is_holiday" in result.columns

        assert result["hour_of_day"].between(0, 23).all()
        assert result["day_of_week"].between(0, 6).all()
        assert result["is_weekend"].isin([0, 1]).all()

    def test_lag_features(self):
        from build_features_from_sqlite import build_lag_features

        df = self._make_load_df()
        result = build_lag_features(df)

        for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
            col = f"load_lag_{lag}h"
            assert col in result.columns, f"Missing {col}"

        # First row lag_1h should be NaN (shifted)
        assert pd.isna(result.iloc[0]["load_lag_1h"])
        # Second row lag_1h should equal first row load
        assert result.iloc[1]["load_lag_1h"] == result.iloc[0]["total_load_mw"]

    def test_rolling_features(self):
        from build_features_from_sqlite import build_rolling_features

        df = self._make_load_df()
        result = build_rolling_features(df)

        for window in [6, 12, 24, 168]:
            for stat in ["mean", "std", "min", "max"]:
                col = f"load_roll_{window}h_{stat}"
                assert col in result.columns, f"Missing {col}"

        # First row should be NaN (shifted by 1)
        assert pd.isna(result.iloc[0]["load_roll_6h_mean"])

    def test_change_features(self):
        from build_features_from_sqlite import build_change_features

        df = self._make_load_df()
        result = build_change_features(df)

        assert "load_change_1h" in result.columns
        assert "load_change_24h" in result.columns
        assert "load_roc_1h" in result.columns

    def test_get_feature_columns(self):
        from build_features_from_sqlite import (
            build_temporal_features,
            build_lag_features,
            build_rolling_features,
            build_change_features,
            get_feature_columns,
        )

        df = self._make_load_df()
        df = build_temporal_features(df)
        df = build_lag_features(df)
        df = build_rolling_features(df)
        df = build_change_features(df)
        df = df.rename(columns={"timestamp": "valid_time"})

        feature_cols = get_feature_columns(df)
        assert "total_load_mw" not in feature_cols
        assert "valid_time" not in feature_cols
        assert "hour_of_day" in feature_cols
        assert "load_lag_1h" in feature_cols
        assert len(feature_cols) >= 20


# ── Predictor ─────────────────────────────────────────────────────────────

class TestLoadPredictor:
    """Test the LoadPredictor class."""

    def test_feature_cols_defined(self):
        mod = _import_load_predictor()
        assert len(mod.FEATURE_COLS) > 20
        for cat in mod.CATEGORICAL_FEATURES:
            assert cat in mod.FEATURE_COLS

    def test_predictor_init_no_checkpoints(self, tmp_path):
        mod = _import_load_predictor()
        predictor = mod.LoadPredictor(checkpoint_dir=tmp_path)
        assert not predictor.is_ready()

    def test_predictor_predict_raises_when_not_ready(self, tmp_path):
        mod = _import_load_predictor()
        predictor = mod.LoadPredictor(checkpoint_dir=tmp_path)
        df = pd.DataFrame(np.zeros((1, len(mod.FEATURE_COLS))), columns=mod.FEATURE_COLS)
        with pytest.raises(RuntimeError, match="No load models"):
            predictor.predict(df)

    def test_predictor_missing_columns(self, tmp_path):
        mod = _import_load_predictor()
        import joblib

        joblib.dump(_FakeModel([50000.0]), tmp_path / "load_catboost.joblib")

        predictor = mod.LoadPredictor(checkpoint_dir=tmp_path)
        assert predictor.is_ready()

        bad_df = pd.DataFrame({"foo": [1]})
        with pytest.raises(ValueError, match="Missing load feature"):
            predictor.predict(bad_df)

    def test_predictor_with_mock_model(self, tmp_path):
        mod = _import_load_predictor()
        import joblib

        joblib.dump(_FakeModel([45000.0, 50000.0]), tmp_path / "load_catboost.joblib")

        predictor = mod.LoadPredictor(checkpoint_dir=tmp_path)
        df = pd.DataFrame(np.zeros((2, len(mod.FEATURE_COLS))), columns=mod.FEATURE_COLS)
        df["hour_of_day"] = [14, 15]

        results = predictor.predict(df)
        assert len(results) == 2
        assert results[0].hour_ending == 15  # 14 + 1
        assert results[0].predicted_load_mw == 45000.0

    def test_predictor_uses_metadata_feature_subset(self, tmp_path):
        mod = _import_load_predictor()
        import json
        import joblib

        joblib.dump(_FakeModel([47000.0]), tmp_path / "load_catboost.joblib")
        (tmp_path / "load_catboost_meta.json").write_text(json.dumps({"features": ["hour_of_day", "month"]}))

        predictor = mod.LoadPredictor(checkpoint_dir=tmp_path)
        df = pd.DataFrame({"hour_of_day": [14], "month": [7], "unused": [999]})

        results = predictor.predict(df)
        assert results[0].predicted_load_mw == 47000.0

    def test_get_model_info(self, tmp_path):
        mod = _import_load_predictor()
        predictor = mod.LoadPredictor(checkpoint_dir=tmp_path)
        info = predictor.get_model_info()
        assert "model_type" in info
        assert "feature_count" in info
        assert info["feature_count"] > 0


# ── Evaluation helper ─────────────────────────────────────────────────────

class TestEvaluation:
    """Test evaluation metrics."""

    def test_evaluate(self):
        sys.path.insert(0, str(Path(__file__).parent.parent / "models" / "load"))
        from train_load_models import evaluate

        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 190.0, 310.0])
        metrics = evaluate(y_true, y_pred, "test")

        assert abs(metrics["mae"] - 10.0) < 1e-6
        assert metrics["rmse"] > 0
        assert metrics["mape"] > 0
