"""
DAM Price Predictor V2

Production predictor using per-settlement-point LightGBM models.
Trained with 80 features (temporal, lag, rolling, cross-market, fuel mix,
ancillary services, RTM components, fuel generation MW, cross-domain).
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from ..config import DAM_V2_CHECKPOINTS, SETTLEMENT_POINTS

log = logging.getLogger(__name__)

FEATURE_COLS = [
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

    Each model predicts DAM LMP given 80 features. One model per settlement
    point, covering all 24 hours.
    """

    def __init__(self, checkpoint_dir: Optional[Path] = None):
        if checkpoint_dir is None:
            checkpoint_dir = DAM_V2_CHECKPOINTS

        self.checkpoint_dir = Path(checkpoint_dir)
        self.models: Dict[str, Any] = {}
        self._load_models()

    def _load_models(self):
        """Load LightGBM models for all available settlement points."""
        if not self.checkpoint_dir.exists():
            log.warning("DAM V2 checkpoint dir not found: %s", self.checkpoint_dir)
            return

        for settlement_point in SETTLEMENT_POINTS:
            sp = settlement_point.lower()
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

    def supported_settlement_points(self) -> List[str]:
        return [sp.lower() for sp in SETTLEMENT_POINTS]

    def has_model(self, settlement_point: str) -> bool:
        return settlement_point.lower() in self.models

    def missing_model_message(self, settlement_point: str) -> str:
        return (
            f"No DAM V2 checkpoint found for '{settlement_point.upper()}'. "
            f"Loaded models: {[sp.upper() for sp in self.available_settlement_points()]}"
        )

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
            raise ValueError(self.missing_model_message(sp))

        model = self.models[sp]
        missing = [col for col in FEATURE_COLS if col not in features_df.columns]
        if missing:
            raise ValueError(f"Missing DAM feature columns: {missing}")

        X = features_df[FEATURE_COLS].copy()

        # Fill NaN in nullable features with 0 (consistent with training)
        X = X.fillna(0)

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
