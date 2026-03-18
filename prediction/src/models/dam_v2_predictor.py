"""
DAM Price Predictor V2

Production predictor using per-settlement-point LightGBM models.
Trained with 41 features (temporal, lag, rolling, cross-market, fuel mix).
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

log = logging.getLogger(__name__)

FUEL_COLS = ["wind_pct", "solar_pct", "gas_pct", "nuclear_pct", "coal_pct", "hydro_pct"]

FEATURE_COLS = [
    # Temporal
    "hour_of_day", "day_of_week", "month", "is_weekend", "is_peak_hour",
    "is_holiday", "is_summer",
    # DAM lags
    "dam_lag_1h", "dam_lag_4h", "dam_lag_24h", "dam_lag_168h",
    # RTM lags
    "rtm_lag_1h", "rtm_lag_4h", "rtm_lag_24h", "rtm_lag_168h",
    # DAM rolling
    "dam_roll_24h_mean", "dam_roll_24h_std", "dam_roll_24h_min", "dam_roll_24h_max",
    "dam_roll_168h_mean", "dam_roll_168h_std", "dam_roll_168h_min", "dam_roll_168h_max",
    # RTM rolling
    "rtm_roll_24h_mean", "rtm_roll_24h_std", "rtm_roll_24h_min", "rtm_roll_24h_max",
    "rtm_roll_168h_mean", "rtm_roll_168h_std", "rtm_roll_168h_min", "rtm_roll_168h_max",
    # Cross-market
    "dam_rtm_spread", "spread_roll_24h_mean", "spread_roll_168h_mean",
    # Fuel mix
    *FUEL_COLS,
]

# Settlement points with trained models
SETTLEMENT_POINTS = ["hb_west", "hb_north", "hb_south", "hb_houston", "hb_busavg"]


@dataclass
class DAMV2Prediction:
    """DAM V2 price prediction result"""
    hour_ending: int
    predicted_price: float
    actual_price: Optional[float] = None
    timestamp: Optional[datetime] = None


class DAMV2Predictor:
    """
    DAM Price Predictor V2 using per-settlement-point LightGBM models.

    Each model predicts DAM LMP given 41 features. One model per settlement
    point, covering all 24 hours.
    """

    def __init__(self, checkpoint_dir: Optional[Path] = None):
        if checkpoint_dir is None:
            checkpoint_dir = Path(__file__).parent.parent.parent / "models" / "dam_v2" / "checkpoints"

        self.checkpoint_dir = Path(checkpoint_dir)
        self.models: Dict[str, Any] = {}
        self._load_models()

    def _load_models(self):
        """Load LightGBM models for all available settlement points."""
        if not self.checkpoint_dir.exists():
            log.warning("DAM V2 checkpoint dir not found: %s", self.checkpoint_dir)
            return

        for sp in SETTLEMENT_POINTS:
            path = self.checkpoint_dir / f"{sp}_lightgbm.joblib"
            if path.exists():
                try:
                    self.models[sp] = joblib.load(path)
                    log.info("Loaded DAM V2 model for %s", sp)
                except Exception as e:
                    log.error("Failed to load DAM V2 model %s: %s", sp, e)

        log.info("Loaded %d/%d DAM V2 models", len(self.models), len(SETTLEMENT_POINTS))

    def is_ready(self) -> bool:
        return len(self.models) > 0

    def available_settlement_points(self) -> List[str]:
        return sorted(self.models.keys())

    def predict(self, features_df: pd.DataFrame, settlement_point: str) -> List[DAMV2Prediction]:
        """
        Predict DAM prices from a feature DataFrame.

        Args:
            features_df: DataFrame with FEATURE_COLS columns, one row per hour to predict.
            settlement_point: Settlement point key (e.g. "hb_west").

        Returns:
            List of DAMV2Prediction.
        """
        sp = settlement_point.lower()
        if sp not in self.models:
            raise ValueError(f"No model for settlement point '{sp}'. Available: {self.available_settlement_points()}")

        model = self.models[sp]
        missing = [col for col in FEATURE_COLS if col not in features_df.columns]
        if missing:
            raise ValueError(f"Missing DAM feature columns: {missing}")

        X = features_df[FEATURE_COLS].copy()

        # Fill missing fuel cols with 0 (consistent with training)
        for col in FUEL_COLS:
            if col in X.columns:
                X[col] = X[col].fillna(0)

        preds = model.predict(X)

        results = []
        for i, price in enumerate(preds):
            hour = int(features_df.iloc[i].get("hour_of_day", i + 1))
            results.append(DAMV2Prediction(
                hour_ending=hour,
                predicted_price=round(float(price), 2),
                timestamp=datetime.utcnow(),
            ))
        return results

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_type": "DAM V2 LightGBM (per settlement point)",
            "models_loaded": len(self.models),
            "settlement_points": self.available_settlement_points(),
            "feature_count": len(FEATURE_COLS),
            "features": FEATURE_COLS,
        }


# Singleton
_v2_predictor: Optional[DAMV2Predictor] = None


def get_dam_v2_predictor() -> DAMV2Predictor:
    global _v2_predictor
    if _v2_predictor is None:
        _v2_predictor = DAMV2Predictor()
    return _v2_predictor
