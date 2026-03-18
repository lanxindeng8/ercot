"""
RTM Multi-Horizon Price Predictor

Loads per-horizon LightGBM models (1h, 4h, 24h ahead) for RTM LMP prediction.
Trained with 41 base features + 3 RTM-specific features = 44 total.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

FUEL_COLS = ["wind_pct", "solar_pct", "gas_pct", "nuclear_pct", "coal_pct", "hydro_pct"]

BASE_FEATURE_COLS = [
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

RTM_EXTRA_FEATURES = [
    "rtm_volatility_24h",       # rtm_roll_24h_std / (rtm_roll_24h_mean + 1)
    "rtm_momentum_4h",          # rtm_lag_1h - rtm_lag_4h
    "rtm_mean_revert_signal",   # rtm_lag_1h - rtm_roll_24h_mean
]

FEATURE_COLS = BASE_FEATURE_COLS + RTM_EXTRA_FEATURES

# Horizons trained: name suffix -> hours ahead
HORIZONS = {
    "1h": 1,
    "4h": 4,
    "24h": 24,
}

# Settlement points with trained models
SETTLEMENT_POINTS = ["hb_west"]


@dataclass
class RTMPrediction:
    """RTM price prediction for a single horizon."""
    horizon: str
    hours_ahead: int
    predicted_price: float
    timestamp: datetime


class RTMPredictor:
    """
    RTM multi-horizon predictor using LightGBM models.

    Loads separate models for 1h, 4h, and 24h ahead forecasting.
    Currently trained for hb_west only.
    """

    def __init__(self, checkpoint_dir: Optional[Path] = None):
        if checkpoint_dir is None:
            checkpoint_dir = Path(__file__).parent.parent.parent / "models" / "rtm" / "checkpoints"

        self.checkpoint_dir = Path(checkpoint_dir)
        # models[sp][horizon_key] = model
        self.models: Dict[str, Dict[str, Any]] = {}
        self._load_models()

    def _load_models(self):
        """Load LightGBM models for each settlement point and horizon."""
        if not self.checkpoint_dir.exists():
            log.warning("RTM checkpoint dir not found: %s", self.checkpoint_dir)
            return

        for sp in SETTLEMENT_POINTS:
            self.models[sp] = {}
            for horizon_key, _ in HORIZONS.items():
                path = self.checkpoint_dir / f"{sp}_rtm_lmp_{horizon_key}_lightgbm.joblib"
                if path.exists():
                    try:
                        self.models[sp][horizon_key] = joblib.load(path)
                        log.info("Loaded RTM model %s/%s", sp, horizon_key)
                    except Exception as e:
                        log.error("Failed to load RTM model %s/%s: %s", sp, horizon_key, e)

        total = sum(len(h) for h in self.models.values())
        log.info("Loaded %d RTM models", total)

    def is_ready(self) -> bool:
        return any(len(h) > 0 for h in self.models.values())

    def available_settlement_points(self) -> List[str]:
        return [sp for sp, horizons in self.models.items() if len(horizons) > 0]

    def available_horizons(self, settlement_point: str) -> List[str]:
        sp = settlement_point.lower()
        return sorted(self.models.get(sp, {}).keys())

    def add_rtm_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add RTM-specific engineered features to a DataFrame."""
        out = df.copy()
        out["rtm_volatility_24h"] = out["rtm_roll_24h_std"] / (out["rtm_roll_24h_mean"] + 1)
        out["rtm_momentum_4h"] = out["rtm_lag_1h"] - out["rtm_lag_4h"]
        out["rtm_mean_revert_signal"] = out["rtm_lag_1h"] - out["rtm_roll_24h_mean"]
        return out

    def predict(
        self,
        features_df: pd.DataFrame,
        settlement_point: str,
        horizons: Optional[List[str]] = None,
    ) -> List[RTMPrediction]:
        """
        Predict RTM prices for requested horizons.

        Args:
            features_df: DataFrame with feature columns (single row = current state).
            settlement_point: e.g. "hb_west"
            horizons: List of horizon keys like ["1h", "4h", "24h"]. None = all available.

        Returns:
            List of RTMPrediction.
        """
        sp = settlement_point.lower()
        if sp not in self.models or len(self.models[sp]) == 0:
            raise ValueError(f"No RTM models for '{sp}'. Available: {self.available_settlement_points()}")

        available = self.models[sp]
        if horizons is None:
            horizons = sorted(available.keys())

        # Add RTM extra features if missing
        df = features_df.copy()
        for feat in RTM_EXTRA_FEATURES:
            if feat not in df.columns:
                df = self.add_rtm_features(df)
                break

        # Fill missing fuel cols
        for col in FUEL_COLS:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        X = df[FEATURE_COLS]

        results = []
        now = datetime.utcnow()
        for h_key in horizons:
            if h_key not in available:
                continue
            model = available[h_key]
            pred = model.predict(X)
            # Use the last row prediction (most recent state)
            price = float(pred[-1]) if len(pred) > 1 else float(pred[0])
            results.append(RTMPrediction(
                horizon=h_key,
                hours_ahead=HORIZONS[h_key],
                predicted_price=round(price, 2),
                timestamp=now,
            ))

        return results

    def get_model_info(self) -> Dict[str, Any]:
        model_details = {}
        for sp, horizons in self.models.items():
            model_details[sp] = list(horizons.keys())
        return {
            "model_type": "RTM Multi-Horizon LightGBM",
            "models_loaded": sum(len(h) for h in self.models.values()),
            "settlement_points": model_details,
            "horizons": list(HORIZONS.keys()),
            "feature_count": len(FEATURE_COLS),
        }


# Singleton
_rtm_predictor: Optional[RTMPredictor] = None


def get_rtm_predictor() -> RTMPredictor:
    global _rtm_predictor
    if _rtm_predictor is None:
        _rtm_predictor = RTMPredictor()
    return _rtm_predictor
