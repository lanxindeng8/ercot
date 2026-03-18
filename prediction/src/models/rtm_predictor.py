"""
RTM Multi-Horizon Price Predictor

Loads per-horizon LightGBM models (1h, 4h, 24h ahead) for RTM LMP prediction.
Trained with 80 base features + 3 RTM-specific features = 83 total.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from ..config import RTM_CHECKPOINTS, SETTLEMENT_POINTS

log = logging.getLogger(__name__)

BASE_FEATURE_COLS = [
    # Temporal (7)
    "hour_of_day", "day_of_week", "month", "is_weekend", "is_peak_hour",
    "is_holiday", "is_summer",
    # DAM lags (4)
    "dam_lag_1h", "dam_lag_4h", "dam_lag_24h", "dam_lag_168h",
    # RTM lags (4)
    "rtm_lag_1h", "rtm_lag_4h", "rtm_lag_24h", "rtm_lag_168h",
    # DAM rolling (8)
    "dam_roll_24h_mean", "dam_roll_24h_std", "dam_roll_24h_min", "dam_roll_24h_max",
    "dam_roll_168h_mean", "dam_roll_168h_std", "dam_roll_168h_min", "dam_roll_168h_max",
    # RTM rolling (8)
    "rtm_roll_24h_mean", "rtm_roll_24h_std", "rtm_roll_24h_min", "rtm_roll_24h_max",
    "rtm_roll_168h_mean", "rtm_roll_168h_std", "rtm_roll_168h_min", "rtm_roll_168h_max",
    # Cross-market (3)
    "dam_rtm_spread", "spread_roll_24h_mean", "spread_roll_168h_mean",
    # Fuel mix pct (6)
    "wind_pct", "solar_pct", "gas_pct", "nuclear_pct", "coal_pct", "hydro_pct",
    # Ancillary service (13)
    "regdn", "regup", "rrs", "nspin", "ecrs",
    "reg_spread", "total_as_cost",
    "regup_lag_24h", "rrs_lag_24h", "nspin_lag_24h", "total_as_lag_24h",
    "total_as_roll_24h_mean", "total_as_roll_24h_std",
    # RTM components (6)
    "congestion_pct", "loss_pct", "energy_pct",
    "congestion_ma_4h", "congestion_volatility_24h", "high_congestion_flag",
    # Fuel gen MW (15)
    "gas_gen_mw", "gas_cc_gen_mw", "coal_gen_mw", "nuclear_gen_mw",
    "solar_gen_mw", "wind_gen_mw", "hydro_gen_mw", "biomass_gen_mw",
    "total_gen_mw", "renewable_ratio", "thermal_ratio", "net_load_mw",
    "solar_ramp_1h", "wind_ramp_1h", "gas_ramp_1h",
    # Cross-domain (6)
    "dam_as_ratio", "reg_spread_roll_24h_mean", "ecrs_lag_24h",
    "gas_cc_share", "wind_ramp_4h", "solar_ramp_4h",
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
            checkpoint_dir = RTM_CHECKPOINTS

        self.checkpoint_dir = Path(checkpoint_dir)
        # models[sp][horizon_key] = model
        self.models: Dict[str, Dict[str, Any]] = {}
        self._load_models()

    def _load_models(self):
        """Load LightGBM models for each settlement point and horizon."""
        if not self.checkpoint_dir.exists():
            log.warning("RTM checkpoint dir not found: %s", self.checkpoint_dir)
            return

        for settlement_point in SETTLEMENT_POINTS:
            sp = settlement_point.lower()
            self.models[sp] = {}
            for horizon_key, _ in HORIZONS.items():
                lgbm_path = self.checkpoint_dir / f"{sp}_rtm_lmp_{horizon_key}_lightgbm.joblib"
                cb_path = self.checkpoint_dir / f"{sp}_rtm_lmp_{horizon_key}_catboost.joblib"
                path = lgbm_path if lgbm_path.exists() else cb_path
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
        return sorted(sp for sp, horizons in self.models.items() if len(horizons) > 0)

    def supported_settlement_points(self) -> List[str]:
        return [sp.lower() for sp in SETTLEMENT_POINTS]

    def has_model(self, settlement_point: str) -> bool:
        return len(self.models.get(settlement_point.lower(), {})) > 0

    def missing_model_message(self, settlement_point: str) -> str:
        return (
            f"No RTM checkpoint found for '{settlement_point.upper()}'. "
            f"Loaded models: {[sp.upper() for sp in self.available_settlement_points()]}"
        )

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
            raise ValueError(self.missing_model_message(sp))

        available = self.models[sp]
        if horizons is None:
            horizons = sorted(available.keys())
        else:
            invalid = [h for h in horizons if h not in HORIZONS]
            if invalid:
                raise ValueError(f"Unsupported RTM horizons: {invalid}")

        # Add RTM extra features if missing
        df = features_df.copy()
        for feat in RTM_EXTRA_FEATURES:
            if feat not in df.columns:
                df = self.add_rtm_features(df)
                break

        # Fill NaN in nullable features with 0 (consistent with training)
        df = df.fillna(0)

        missing = [col for col in FEATURE_COLS if col not in df.columns]
        if missing:
            raise ValueError(f"Missing RTM feature columns: {missing}")

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
