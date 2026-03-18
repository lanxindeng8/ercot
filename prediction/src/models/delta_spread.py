"""Delta Spread prediction model loader and inference."""
import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostRegressor, CatBoostClassifier
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from ..config import DELTA_SPREAD_MODELS


class DeltaSpreadPredictor:
    """Predictor for RTM-DAM spread using trained CatBoost models."""

    # Feature names required for prediction (exact order from trained model)
    FEATURE_NAMES = [
        'target_dam_price', 'target_hour', 'target_dow', 'target_month',
        'target_day_of_month', 'target_week', 'target_is_weekend', 'target_is_peak',
        'target_is_summer', 'spread_mean_7d', 'spread_std_7d', 'spread_max_7d',
        'spread_min_7d', 'spread_median_7d', 'spread_mean_24h', 'spread_std_24h',
        'rtm_mean_7d', 'rtm_std_7d', 'rtm_max_7d', 'rtm_mean_24h', 'rtm_volatility_24h',
        'dam_mean_7d', 'dam_std_7d', 'dam_mean_24h', 'spread_positive_ratio_7d',
        'spread_positive_ratio_24h', 'spike_count_7d', 'rtm_spike_count_7d',
        'spread_trend_7d', 'spread_same_hour_hist', 'spread_same_hour_std',
        'rtm_same_hour_hist', 'spread_same_dow_hour', 'dam_vs_7d_mean',
        'dam_percentile_7d', 'spread_same_dow_hour_last', 'dam_price_level',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
        'dam_vs_same_hour'
    ]

    # Categorical feature indices (as specified during training)
    CAT_FEATURE_INDICES = [1, 2, 3, 4, 5, 6, 7, 8]

    CATEGORICAL_FEATURES = [
        'target_hour', 'target_dow', 'target_month', 'target_day_of_month',
        'target_week', 'target_is_weekend', 'target_is_peak', 'target_is_summer'
    ]

    SPREAD_INTERVALS = {
        0: "< -$20 (Strong Short)",
        1: "-$20 to -$5 (Moderate Short)",
        2: "-$5 to $5 (No Trade)",
        3: "$5 to $20 (Moderate Long)",
        4: "> $20 (Strong Long)"
    }

    def __init__(self):
        self.regression_model = None
        self.binary_model = None
        self.multiclass_model = None
        self._load_models()

    def _load_models(self):
        """Load trained CatBoost models from disk."""
        reg_path = DELTA_SPREAD_MODELS / "regression_model.cbm"
        bin_path = DELTA_SPREAD_MODELS / "binary_model.cbm"
        mc_path = DELTA_SPREAD_MODELS / "multiclass_model.cbm"

        if reg_path.exists():
            self.regression_model = CatBoostRegressor()
            self.regression_model.load_model(str(reg_path))
            print(f"Loaded regression model from {reg_path}")

        if bin_path.exists():
            self.binary_model = CatBoostClassifier()
            self.binary_model.load_model(str(bin_path))
            print(f"Loaded binary model from {bin_path}")

        if mc_path.exists():
            self.multiclass_model = CatBoostClassifier()
            self.multiclass_model.load_model(str(mc_path))
            print(f"Loaded multiclass model from {mc_path}")

    def predict(self, features: pd.DataFrame) -> Dict:
        """
        Make predictions using all three models.

        Args:
            features: DataFrame with required feature columns

        Returns:
            Dictionary with predictions from all models
        """
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "predictions": []
        }

        for idx, row in features.iterrows():
            pred = {
                "target_hour": int(row.get("target_hour", 0)),
                "target_date": str(row.get("target_date", "")),
            }

            # Regression prediction
            if self.regression_model:
                spread_value = float(self.regression_model.predict([row[self.FEATURE_NAMES]])[0])
                pred["spread_value"] = round(spread_value, 2)

            # Binary prediction
            if self.binary_model:
                binary_pred = self.binary_model.predict([row[self.FEATURE_NAMES]])[0]
                binary_prob = self.binary_model.predict_proba([row[self.FEATURE_NAMES]])[0]
                pred["direction"] = "RTM > DAM" if binary_pred == 1 else "RTM < DAM"
                pred["direction_probability"] = round(float(max(binary_prob)), 3)

            # Multiclass prediction
            if self.multiclass_model:
                mc_raw = self.multiclass_model.predict([row[self.FEATURE_NAMES]])[0]
                mc_pred = int(mc_raw) if np.ndim(mc_raw) == 0 else int(mc_raw[0])
                mc_prob = self.multiclass_model.predict_proba([row[self.FEATURE_NAMES]])[0]
                pred["spread_interval"] = mc_pred
                pred["spread_interval_label"] = self.SPREAD_INTERVALS.get(mc_pred, "Unknown")
                pred["interval_probabilities"] = [round(float(p), 3) for p in mc_prob]

            # Trading signal
            pred["signal"] = self._get_trading_signal(pred)

            results["predictions"].append(pred)

        return results

    def _get_trading_signal(self, pred: Dict) -> str:
        """Generate trading signal based on model outputs."""
        spread = pred.get("spread_value", 0)
        prob = pred.get("direction_probability", 0.5)
        interval = pred.get("spread_interval", 2)

        # High confidence signals
        if abs(spread) > 15 and prob > 0.6:
            if spread > 0:
                return "STRONG_LONG"
            else:
                return "STRONG_SHORT"

        # Moderate confidence
        if abs(spread) > 5:
            if spread > 0:
                return "LONG"
            else:
                return "SHORT"

        # Low confidence or no-trade zone
        return "HOLD"

    def get_model_info(self) -> Dict:
        """Get information about loaded models."""
        return {
            "regression_loaded": self.regression_model is not None,
            "binary_loaded": self.binary_model is not None,
            "multiclass_loaded": self.multiclass_model is not None,
            "feature_count": len(self.FEATURE_NAMES),
            "spread_intervals": self.SPREAD_INTERVALS
        }


# Singleton instance
_predictor = None

def get_predictor() -> DeltaSpreadPredictor:
    """Get or create the Delta Spread predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = DeltaSpreadPredictor()
    return _predictor
