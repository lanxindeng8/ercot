"""
Wind Generation Forecast Predictor

Production predictor wrapping the GBM wind model for serving
wind generation forecasts via the prediction API.

Uses the trained GBMWindModel checkpoints with quantile outputs
(Q0.10, Q0.50, Q0.90) for uncertainty quantification.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class WindPrediction:
    """Wind generation forecast result."""
    hour_ending: int
    predicted_mw: float
    lower_bound_mw: float  # Q0.10
    upper_bound_mw: float  # Q0.90
    timestamp: Optional[datetime] = None


class WindPredictor:
    """
    Wind Generation Forecast Predictor using GBM quantile models.

    Loads the trained GBMWindModel checkpoint from disk and provides
    a simple predict() interface for the API layer.
    """

    def __init__(self, checkpoint_dir: Optional[Path] = None):
        if checkpoint_dir is None:
            checkpoint_dir = Path(__file__).parent.parent.parent / "models" / "wind" / "checkpoints" / "gbm_model"

        self.checkpoint_dir = Path(checkpoint_dir)
        self.model = None
        self.metadata: Dict[str, Any] = {}
        self._load_model()

    def _load_model(self):
        """Load the GBM wind model from checkpoint."""
        if not self.checkpoint_dir.exists():
            log.warning("Wind checkpoint dir not found: %s", self.checkpoint_dir)
            return

        meta_path = self.checkpoint_dir / "metadata.json"
        if not meta_path.exists():
            log.warning("Wind model metadata not found: %s", meta_path)
            return

        try:
            with open(meta_path) as f:
                self.metadata = json.load(f)

            # Import GBMWindModel and load checkpoint
            import sys
            wind_src = str(Path(__file__).parent.parent.parent / "models" / "wind" / "src")
            if wind_src not in sys.path:
                sys.path.insert(0, wind_src)

            from models.gbm_model import GBMWindModel
            self.model = GBMWindModel.load(str(self.checkpoint_dir))
            log.info("Loaded wind GBM model from %s", self.checkpoint_dir)
        except Exception as e:
            log.error("Failed to load wind model: %s", e)

    def is_ready(self) -> bool:
        return self.model is not None

    def predict(self, features_df) -> List[WindPrediction]:
        """
        Predict wind generation from a feature DataFrame.

        Args:
            features_df: DataFrame with wind feature columns, one row per hour.

        Returns:
            List of WindPrediction with point forecast and uncertainty bounds.
        """
        if not self.is_ready():
            raise RuntimeError("Wind model not loaded")

        import pandas as pd

        # Get quantile predictions
        quantiles = self.model.predict_quantiles(features_df, quantiles=[0.1, 0.5, 0.9])
        q10 = quantiles.get(0.1, np.zeros(len(features_df)))
        q50 = quantiles.get(0.5, np.zeros(len(features_df)))
        q90 = quantiles.get(0.9, np.zeros(len(features_df)))

        results = []
        for i in range(len(features_df)):
            hour = int(features_df.iloc[i].get("hour", i)) if "hour" in features_df.columns else i
            results.append(WindPrediction(
                hour_ending=hour + 1,
                predicted_mw=round(float(q50[i]), 1),
                lower_bound_mw=round(float(q10[i]), 1),
                upper_bound_mw=round(float(q90[i]), 1),
                timestamp=datetime.now(tz=None),
            ))
        return results

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_type": "Wind GBM (LightGBM Quantile Regression)",
            "quantiles": self.metadata.get("quantiles", []),
            "feature_count": len(self.metadata.get("feature_names", [])),
            "features": self.metadata.get("feature_names", []),
            "params": self.metadata.get("params", {}),
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_predictor: Optional[WindPredictor] = None


def get_wind_predictor() -> WindPredictor:
    global _predictor
    if _predictor is None:
        _predictor = WindPredictor()
    return _predictor
