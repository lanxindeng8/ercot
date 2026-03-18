"""
Load Forecast Predictor

Production predictor using CatBoost + LightGBM ensemble for
ERCOT total system load forecasting.

Trained with temporal, lag, and rolling features from fuel_mix_hist.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# Features must match build_features_from_sqlite.py output order
TEMPORAL_FEATURES = [
    "hour_of_day", "day_of_week", "month", "is_weekend",
    "is_peak_hour", "season", "is_holiday",
]

LAG_FEATURES = [
    "load_lag_1h", "load_lag_2h", "load_lag_3h", "load_lag_6h",
    "load_lag_12h", "load_lag_24h", "load_lag_48h", "load_lag_168h",
]

ROLLING_FEATURES = [
    "load_roll_6h_mean", "load_roll_6h_std", "load_roll_6h_min", "load_roll_6h_max",
    "load_roll_12h_mean", "load_roll_12h_std", "load_roll_12h_min", "load_roll_12h_max",
    "load_roll_24h_mean", "load_roll_24h_std", "load_roll_24h_min", "load_roll_24h_max",
    "load_roll_168h_mean", "load_roll_168h_std", "load_roll_168h_min", "load_roll_168h_max",
]

CHANGE_FEATURES = [
    "load_change_1h", "load_change_24h", "load_roc_1h",
]

FEATURE_COLS = TEMPORAL_FEATURES + LAG_FEATURES + ROLLING_FEATURES + CHANGE_FEATURES

CATEGORICAL_FEATURES = [
    "hour_of_day", "day_of_week", "month", "is_weekend",
    "is_peak_hour", "season", "is_holiday",
]


@dataclass
class LoadPrediction:
    """Load forecast prediction result."""
    hour_ending: int
    predicted_load_mw: float
    timestamp: Optional[datetime] = None


class LoadPredictor:
    """
    Load Forecast Predictor using CatBoost + LightGBM ensemble.

    Loads checkpoints from disk and averages predictions for the
    final forecast.
    """

    MODEL_FILES = {
        "catboost": "load_catboost.joblib",
        "lightgbm": "load_lightgbm.joblib",
    }

    def __init__(self, checkpoint_dir: Optional[Path] = None):
        if checkpoint_dir is None:
            checkpoint_dir = Path(__file__).parent.parent.parent / "models" / "load" / "checkpoints"

        self.checkpoint_dir = Path(checkpoint_dir)
        self.models: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}
        self._load_models()

    def _load_models(self):
        """Load available model checkpoints."""
        if not self.checkpoint_dir.exists():
            log.warning("Load checkpoint dir not found: %s", self.checkpoint_dir)
            return

        for name, filename in self.MODEL_FILES.items():
            path = self.checkpoint_dir / filename
            if path.exists():
                try:
                    self.models[name] = joblib.load(path)
                    log.info("Loaded load model: %s", name)
                except Exception as e:
                    log.error("Failed to load load model %s: %s", name, e)

            # Load metadata if available
            meta_path = self.checkpoint_dir / f"{Path(filename).stem}_meta.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    self.metadata[name] = json.load(f)

        log.info("Loaded %d/%d load models", len(self.models), len(self.MODEL_FILES))

    def is_ready(self) -> bool:
        return len(self.models) > 0

    def predict(self, features_df: pd.DataFrame) -> List[LoadPrediction]:
        """
        Predict total system load from a feature DataFrame.

        Args:
            features_df: DataFrame with FEATURE_COLS columns, one row per hour.

        Returns:
            List of LoadPrediction.
        """
        if not self.is_ready():
            raise RuntimeError("No load models loaded")

        missing = [col for col in FEATURE_COLS if col not in features_df.columns]
        if missing:
            raise ValueError(f"Missing load feature columns: {missing}")

        X = features_df[FEATURE_COLS].copy()

        # Collect predictions from all loaded models
        preds = []
        for name, model in self.models.items():
            if name == "catboost":
                preds.append(model.predict(X.values))
            else:
                preds.append(model.predict(X.values))

        # Ensemble average
        y_pred = np.mean(preds, axis=0)

        results = []
        for i, load_mw in enumerate(y_pred):
            hour = int(features_df.iloc[i].get("hour_of_day", i))
            results.append(LoadPrediction(
                hour_ending=hour + 1,  # convert 0-indexed hour to 1-indexed
                predicted_load_mw=round(float(load_mw), 1),
                timestamp=datetime.now(tz=None),
            ))
        return results

    def predict_raw(self, features_df: pd.DataFrame) -> np.ndarray:
        """Return raw ensemble prediction array (MW)."""
        if not self.is_ready():
            raise RuntimeError("No load models loaded")

        X = features_df[FEATURE_COLS].values
        preds = [model.predict(X) for model in self.models.values()]
        return np.mean(preds, axis=0)

    def get_model_info(self) -> Dict[str, Any]:
        info = {
            "model_type": "Load CatBoost+LightGBM Ensemble",
            "models_loaded": list(self.models.keys()),
            "feature_count": len(FEATURE_COLS),
            "features": FEATURE_COLS,
            "categorical_features": CATEGORICAL_FEATURES,
        }
        # Attach test metrics if available
        for name, meta in self.metadata.items():
            if "metrics" in meta:
                info[f"{name}_metrics"] = meta["metrics"]
        return info


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_predictor: Optional[LoadPredictor] = None


def get_load_predictor() -> LoadPredictor:
    global _predictor
    if _predictor is None:
        _predictor = LoadPredictor()
    return _predictor
