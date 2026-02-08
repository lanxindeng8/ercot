"""
DAM Price Predictor

Loads trained DAM price prediction models and generates forecasts.
Supports XGBoost, LightGBM, and CatBoost ensemble.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from ..features.dam_features import DAMFeatureEngineer, FEATURE_NAMES, CATEGORICAL_INDICES


class DAMPredictor:
    """DAM price prediction using ensemble models"""

    def __init__(self, model_dir: str = None):
        """
        Initialize DAM predictor

        Args:
            model_dir: Directory containing trained models
        """
        if model_dir is None:
            model_dir = Path(__file__).parent.parent.parent / "models" / "dam"

        self.model_dir = Path(model_dir)
        self.models = {}
        self.metadata = {}
        self.feature_engineer = DAMFeatureEngineer()
        self._load_models()

    def _load_models(self):
        """Load all available DAM models"""
        if not self.model_dir.exists():
            print(f"Model directory not found: {self.model_dir}")
            return

        # Load XGBoost
        xgb_path = self.model_dir / "xgboost_dam.joblib"
        if xgb_path.exists():
            try:
                self.models['xgboost'] = joblib.load(xgb_path)
                print(f"Loaded XGBoost model from {xgb_path}")
            except Exception as e:
                print(f"Error loading XGBoost model: {e}")

        # Load LightGBM
        lgb_path = self.model_dir / "lightgbm_dam.joblib"
        if lgb_path.exists():
            try:
                self.models['lightgbm'] = joblib.load(lgb_path)
                print(f"Loaded LightGBM model from {lgb_path}")
            except Exception as e:
                print(f"Error loading LightGBM model: {e}")

        # Load CatBoost
        cat_path = self.model_dir / "catboost_dam.cbm"
        if cat_path.exists() and CATBOOST_AVAILABLE:
            try:
                model = CatBoostRegressor()
                model.load_model(str(cat_path))
                self.models['catboost'] = model
                print(f"Loaded CatBoost model from {cat_path}")
            except Exception as e:
                print(f"Error loading CatBoost model: {e}")

        # Load metadata
        metadata_path = self.model_dir / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path) as f:
                    self.metadata = json.load(f)
            except Exception as e:
                print(f"Error loading metadata: {e}")

        print(f"Loaded {len(self.models)} DAM models: {list(self.models.keys())}")

    def predict(
        self,
        features_df: pd.DataFrame,
        use_ensemble: bool = True
    ) -> Dict[str, Any]:
        """
        Generate DAM price predictions

        Args:
            features_df: DataFrame with features (must match FEATURE_NAMES order)
            use_ensemble: Whether to average all model predictions

        Returns:
            Dictionary with predictions and metadata
        """
        if not self.models:
            return {
                "status": "error",
                "message": "No models loaded",
                "predictions": []
            }

        # Ensure feature order
        X = features_df[FEATURE_NAMES].copy()

        predictions = {}
        for name, model in self.models.items():
            try:
                pred = model.predict(X)
                predictions[name] = pred
            except Exception as e:
                print(f"Error predicting with {name}: {e}")

        if not predictions:
            return {
                "status": "error",
                "message": "All models failed to predict",
                "predictions": []
            }

        # Ensemble prediction
        if use_ensemble and len(predictions) > 1:
            ensemble_pred = np.mean(list(predictions.values()), axis=0)
        else:
            ensemble_pred = list(predictions.values())[0]

        # Build response
        result_predictions = []
        for i in range(len(features_df)):
            hour = int(features_df.iloc[i]['hour'])
            pred_value = float(ensemble_pred[i])

            pred_entry = {
                "hour": hour,
                "predicted_price": round(pred_value, 2),
                "confidence": self._calculate_confidence(predictions, i),
            }

            # Add individual model predictions
            for name, pred in predictions.items():
                pred_entry[f"{name}_prediction"] = round(float(pred[i]), 2)

            result_predictions.append(pred_entry)

        return {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "models_used": list(predictions.keys()),
            "ensemble": use_ensemble,
            "predictions": result_predictions
        }

    def _calculate_confidence(self, predictions: Dict[str, np.ndarray], idx: int) -> float:
        """
        Calculate prediction confidence based on model agreement

        Higher agreement = higher confidence
        """
        if len(predictions) < 2:
            return 0.8  # Default for single model

        values = [pred[idx] for pred in predictions.values()]
        mean_val = np.mean(values)
        std_val = np.std(values)

        # Coefficient of variation (lower = more agreement)
        cv = std_val / (abs(mean_val) + 1e-6)

        # Convert to confidence score (0-1)
        confidence = max(0.3, min(0.95, 1 - cv))
        return round(confidence, 2)

    def predict_next_day(
        self,
        price_history: pd.DataFrame,
        settlement_point: str = "LZ_HOUSTON"
    ) -> Dict[str, Any]:
        """
        Generate predictions for the next day

        Args:
            price_history: Historical DAM prices with DatetimeIndex
            settlement_point: ERCOT settlement point

        Returns:
            Predictions for next 24 hours
        """
        # Determine target date (tomorrow)
        last_date = price_history.index.max()
        target_date = last_date + timedelta(days=1)

        # Extract features
        features_df = self.feature_engineer.extract_inference_features(
            price_history,
            target_date,
            target_hours=list(range(1, 25))
        )

        if features_df.empty:
            return {
                "status": "error",
                "message": "Insufficient historical data for feature extraction",
                "predictions": []
            }

        result = self.predict(features_df)
        result["settlement_point"] = settlement_point
        result["target_date"] = target_date.strftime("%Y-%m-%d")
        result["forecast_horizon_hours"] = 24

        return result

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        info = {
            "models_loaded": list(self.models.keys()),
            "model_count": len(self.models),
            "feature_count": len(FEATURE_NAMES),
            "categorical_feature_count": len(CATEGORICAL_INDICES),
        }

        if self.metadata:
            info["training_info"] = self.metadata

        return info

    def is_ready(self) -> bool:
        """Check if predictor has at least one model loaded"""
        return len(self.models) > 0


# Singleton instance
_predictor = None


def get_dam_predictor() -> DAMPredictor:
    """Get singleton DAM predictor instance"""
    global _predictor
    if _predictor is None:
        _predictor = DAMPredictor()
    return _predictor
