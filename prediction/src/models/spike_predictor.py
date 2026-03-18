"""
Spike Detection Predictor

CatBoost binary classifier predicting next-hour RTM price spikes.
Spike = RTM price > max(100, 3x rolling 24h mean).
Trained for hb_west; P=0.886, R=1.0, F1=0.939.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

log = logging.getLogger(__name__)

FUEL_COLS = ["wind_pct", "solar_pct", "gas_pct", "nuclear_pct", "coal_pct", "hydro_pct"]

BASE_FEATURES = [
    "hour_of_day", "day_of_week", "month", "is_weekend", "is_peak_hour",
    "is_holiday", "is_summer",
    "dam_lag_1h", "dam_lag_4h", "dam_lag_24h", "dam_lag_168h",
    "rtm_lag_1h", "rtm_lag_4h", "rtm_lag_24h", "rtm_lag_168h",
    "dam_roll_24h_mean", "dam_roll_24h_std", "dam_roll_24h_min", "dam_roll_24h_max",
    "dam_roll_168h_mean", "dam_roll_168h_std", "dam_roll_168h_min", "dam_roll_168h_max",
    "rtm_roll_24h_mean", "rtm_roll_24h_std", "rtm_roll_24h_min", "rtm_roll_24h_max",
    "rtm_roll_168h_mean", "rtm_roll_168h_std", "rtm_roll_168h_min", "rtm_roll_168h_max",
    "dam_rtm_spread", "spread_roll_24h_mean", "spread_roll_168h_mean",
    *FUEL_COLS,
]

SPIKE_FEATURES = [
    "price_accel",
    "volatility_regime",
    "hour_spike_prob",
    "price_momentum",
    "price_ratio_to_mean",
    "rtm_range_24h",
]

FEATURE_COLS = BASE_FEATURES + SPIKE_FEATURES

SETTLEMENT_POINTS = ["hb_west"]


@dataclass
class SpikeAlert:
    """Spike detection result."""
    settlement_point: str
    spike_probability: float
    is_spike: bool
    confidence: str  # "high", "medium", "low"
    threshold_used: float
    timestamp: datetime


class SpikePredictor:
    """
    Spike detection predictor using CatBoost binary classifier.

    Predicts whether an RTM price spike will occur in the next hour.
    Currently trained for hb_west only.
    """

    def __init__(self, checkpoint_dir: Optional[Path] = None):
        if checkpoint_dir is None:
            checkpoint_dir = Path(__file__).parent.parent.parent / "models" / "spike" / "checkpoints"

        self.checkpoint_dir = Path(checkpoint_dir)
        self.models: Dict[str, CatBoostClassifier] = {}
        self.meta: Dict[str, Dict] = {}
        self._load_models()

    def _load_models(self):
        """Load CatBoost spike models and metadata."""
        if not self.checkpoint_dir.exists():
            log.warning("Spike checkpoint dir not found: %s", self.checkpoint_dir)
            return

        for sp in SETTLEMENT_POINTS:
            model_path = self.checkpoint_dir / f"{sp}_spike_catboost.cbm"
            meta_path = self.checkpoint_dir / f"{sp}_spike_meta.json"

            if model_path.exists():
                try:
                    model = CatBoostClassifier()
                    model.load_model(str(model_path))
                    self.models[sp] = model
                    log.info("Loaded spike model for %s", sp)
                except Exception as e:
                    log.error("Failed to load spike model %s: %s", sp, e)

            if meta_path.exists():
                try:
                    with open(meta_path) as f:
                        self.meta[sp] = json.load(f)
                except Exception as e:
                    log.error("Failed to load spike meta %s: %s", sp, e)

    def is_ready(self) -> bool:
        return len(self.models) > 0

    def available_settlement_points(self) -> List[str]:
        return sorted(self.models.keys())

    def add_spike_features(self, df: pd.DataFrame, settlement_point: str) -> pd.DataFrame:
        """Add spike-specific engineered features."""
        out = df.copy()

        # price_accel: 2nd derivative of RTM price
        if "rtm_lag_1h" in out.columns and "rtm_lag_4h" in out.columns:
            out["price_accel"] = out["rtm_lag_1h"].diff().diff().fillna(0)

        # volatility_regime
        if "rtm_roll_24h_std" in out.columns and "rtm_roll_24h_mean" in out.columns:
            out["volatility_regime"] = out["rtm_roll_24h_std"] / (out["rtm_roll_24h_mean"].abs() + 1e-6)

        # hour_spike_prob from training metadata
        sp = settlement_point.lower()
        hour_probs = {}
        if sp in self.meta and "hour_spike_probs" in self.meta[sp]:
            hour_probs = {int(k): v for k, v in self.meta[sp]["hour_spike_probs"].items()}
        if "hour_of_day" in out.columns:
            out["hour_spike_prob"] = out["hour_of_day"].map(hour_probs).fillna(0.01)

        # price_momentum
        if "rtm_lag_1h" in out.columns and "rtm_roll_24h_mean" in out.columns:
            # Use rtm_lag_1h as proxy for current RTM price
            out["price_momentum"] = out["rtm_lag_1h"] - out["rtm_roll_24h_mean"]

        # price_ratio_to_mean
        if "rtm_lag_1h" in out.columns and "rtm_roll_24h_mean" in out.columns:
            out["price_ratio_to_mean"] = out["rtm_lag_1h"] / (out["rtm_roll_24h_mean"].abs() + 1e-6)

        # rtm_range_24h
        if "rtm_roll_24h_max" in out.columns and "rtm_roll_24h_min" in out.columns:
            out["rtm_range_24h"] = out["rtm_roll_24h_max"] - out["rtm_roll_24h_min"]

        return out

    def predict(
        self,
        features_df: pd.DataFrame,
        settlement_point: str,
    ) -> List[SpikeAlert]:
        """
        Predict spike probability.

        Args:
            features_df: DataFrame with feature columns.
            settlement_point: e.g. "hb_west"

        Returns:
            List of SpikeAlert (one per row in features_df).
        """
        sp = settlement_point.lower()
        if sp not in self.models:
            raise ValueError(f"No spike model for '{sp}'. Available: {self.available_settlement_points()}")

        model = self.models[sp]

        # Add spike features if missing
        df = features_df.copy()
        for feat in SPIKE_FEATURES:
            if feat not in df.columns:
                df = self.add_spike_features(df, sp)
                break

        # Fill missing fuel cols
        for col in FUEL_COLS:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        X = df[FEATURE_COLS]

        # Get prediction threshold from meta (default 0.5)
        threshold = 0.5
        if sp in self.meta:
            threshold = self.meta[sp].get("threshold", 0.5)

        proba = model.predict_proba(X)[:, 1]  # probability of spike class
        predictions = proba >= threshold

        now = datetime.utcnow()
        results = []
        for i in range(len(X)):
            prob = float(proba[i])
            is_spike = bool(predictions[i])

            if prob >= 0.8:
                confidence = "high"
            elif prob >= 0.5:
                confidence = "medium"
            else:
                confidence = "low"

            results.append(SpikeAlert(
                settlement_point=sp,
                spike_probability=round(prob, 4),
                is_spike=is_spike,
                confidence=confidence,
                threshold_used=threshold,
                timestamp=now,
            ))

        return results

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_type": "Spike Detection CatBoost Binary Classifier",
            "models_loaded": len(self.models),
            "settlement_points": self.available_settlement_points(),
            "feature_count": len(FEATURE_COLS),
            "spike_definition": "RTM price > max(100, 3x rolling 24h mean)",
            "lookahead": "1 hour",
        }


# Singleton
_spike_predictor: Optional[SpikePredictor] = None


def get_spike_predictor() -> SpikePredictor:
    global _spike_predictor
    if _spike_predictor is None:
        _spike_predictor = SpikePredictor()
    return _spike_predictor
