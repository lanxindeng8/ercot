"""
DAM Price Predictor V2

Production predictor using 24 per-hour CatBoost models with 35 features.
Based on the proven DAM_Price_Forecast approach.

Expected MAE: ~$8.50 (vs naive baseline ~$10.58)
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

from catboost import CatBoostRegressor

from ..features.dam_features_v2 import (
    DAMFeatureEngineer,
    DAMFeatureConfig,
    ALL_FEATURES,
    is_us_holiday,
)


@dataclass
class DAMV2Prediction:
    """DAM V2 price prediction result"""
    hour_ending: int
    predicted_price: float
    actual_price: Optional[float] = None
    timestamp: Optional[datetime] = None
    model_hour: Optional[int] = None


class DAMV2Predictor:
    """
    DAM Price Predictor V2 using per-hour CatBoost models.

    Uses 35 features and 24 separate models (one per hour) for
    better prediction accuracy.
    """

    def __init__(self, model_dir: Optional[Path] = None):
        """
        Initialize DAM V2 predictor.

        Args:
            model_dir: Directory containing trained models (dam_hour_01.cbm to dam_hour_24.cbm)
        """
        if model_dir is None:
            model_dir = Path(__file__).parent.parent.parent / "models" / "dam_v2"

        self.model_dir = Path(model_dir)
        self.models: Dict[int, CatBoostRegressor] = {}
        self.feature_engineer = DAMFeatureEngineer()
        self.feature_names = ALL_FEATURES

        self._load_models()

    def _load_models(self):
        """Load all 24 per-hour models"""
        if not self.model_dir.exists():
            print(f"Model directory not found: {self.model_dir}")
            return

        loaded = 0
        for hour in range(1, 25):
            model_path = self.model_dir / f"dam_hour_{hour:02d}.cbm"
            if model_path.exists():
                try:
                    model = CatBoostRegressor()
                    model.load_model(str(model_path))
                    self.models[hour] = model
                    loaded += 1
                except Exception as e:
                    print(f"Error loading model for hour {hour}: {e}")

        print(f"Loaded {loaded}/24 DAM V2 models from {self.model_dir}")

    def is_ready(self) -> bool:
        """Check if predictor has models loaded"""
        return len(self.models) > 0

    def predict_next_day(
        self,
        dam_df: pd.DataFrame,
        target_date: Optional[date] = None,
    ) -> List[DAMV2Prediction]:
        """
        Predict DAM prices for the next day.

        Args:
            dam_df: DataFrame with DatetimeIndex and 'dam_price' column
                   (needs at least 28 days of history)
            target_date: Date to predict for (default: tomorrow)

        Returns:
            List of 24 predictions (one per hour)
        """
        if not self.is_ready():
            raise ValueError("No models loaded. Check model directory.")

        # Ensure proper format
        df = dam_df.copy()
        if 'lmp' in df.columns and 'dam_price' not in df.columns:
            df['dam_price'] = df['lmp']

        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')

        df = df.sort_index()

        # Determine target date
        if target_date is None:
            last_date = df.index.max().date()
            target_date = last_date + timedelta(days=1)
        elif isinstance(target_date, datetime):
            target_date = target_date.date()

        # Build features for each hour
        predictions = []
        target_dt = pd.Timestamp(target_date)

        for hour in range(1, 25):
            features = self._build_inference_features(df, target_dt, hour)

            if features is None:
                # Fallback to naive prediction
                pred_price = self._naive_predict(df, hour)
                predictions.append(DAMV2Prediction(
                    hour_ending=hour,
                    predicted_price=pred_price,
                    timestamp=datetime.combine(target_date, datetime.min.time()) + timedelta(hours=hour-1),
                    model_hour=hour,
                ))
                continue

            # Get model for this hour
            if hour in self.models:
                model = self.models[hour]
                X = pd.DataFrame([features])[self.feature_names]
                pred_price = float(model.predict(X)[0])
            else:
                # No model for this hour, use naive
                pred_price = self._naive_predict(df, hour)

            predictions.append(DAMV2Prediction(
                hour_ending=hour,
                predicted_price=pred_price,
                timestamp=datetime.combine(target_date, datetime.min.time()) + timedelta(hours=hour-1),
                model_hour=hour,
            ))

        return predictions

    def _build_inference_features(
        self,
        df: pd.DataFrame,
        target_date: pd.Timestamp,
        target_hour: int,
    ) -> Optional[Dict]:
        """Build features for a single prediction"""
        features = {}

        # Get dates
        d1_date = (target_date - timedelta(days=1)).date()
        d2_date = (target_date - timedelta(days=2)).date()
        d3_date = (target_date - timedelta(days=3)).date()
        d7_date = (target_date - timedelta(days=7)).date()

        # Get price data for each day
        d1_data = df[df.index.date == d1_date]
        d2_data = df[df.index.date == d2_date]
        d3_data = df[df.index.date == d3_date]
        d7_data = df[df.index.date == d7_date]

        # Need at least yesterday's data
        if len(d1_data) < 24:
            return None

        # Get prices as arrays (indexed by hour 0-23)
        d1_prices = self._get_day_prices(d1_data)
        d2_prices = self._get_day_prices(d2_data)
        d3_prices = self._get_day_prices(d3_data)
        d7_prices = self._get_day_prices(d7_data)

        if d1_prices is None:
            return None

        hour_idx = target_hour - 1

        # Time features (10)
        features['hour'] = target_hour
        features['day_of_week'] = target_date.dayofweek
        features['day_of_month'] = target_date.day
        features['month'] = target_date.month
        features['quarter'] = target_date.quarter
        features['week_of_year'] = target_date.isocalendar()[1]
        features['is_weekend'] = int(target_date.dayofweek >= 5)
        features['is_holiday'] = is_us_holiday(target_date)
        features['is_peak_hour'] = int(7 <= target_hour <= 22)
        features['is_summer'] = int(target_date.month in [6, 7, 8, 9])

        # Lag features (12)
        features['dam_lag_24h'] = d1_prices[hour_idx]
        features['dam_lag_24h_prev'] = d1_prices[max(0, hour_idx - 1)]
        features['dam_lag_24h_next'] = d1_prices[min(23, hour_idx + 1)]
        features['dam_lag_48h'] = d2_prices[hour_idx] if d2_prices is not None else features['dam_lag_24h']
        features['dam_lag_168h'] = d7_prices[hour_idx] if d7_prices is not None else features['dam_lag_24h']
        features['dam_d1_mean'] = np.nanmean(d1_prices)
        features['dam_d1_max'] = np.nanmax(d1_prices)
        features['dam_d1_min'] = np.nanmin(d1_prices)
        features['dam_d1_std'] = np.nanstd(d1_prices)

        # 7-day mean and same-hour stats
        prices_7d = []
        same_hour_7d = []
        for j in range(1, 8):
            day_date = (target_date - timedelta(days=j)).date()
            day_data = df[df.index.date == day_date]
            if len(day_data) >= 1:
                day_prices = day_data['dam_price'].values
                prices_7d.extend(day_prices)
                # Same hour value
                hour_mask = day_data.index.hour == (target_hour - 1)
                if hour_mask.any():
                    same_hour_7d.append(day_data.loc[hour_mask, 'dam_price'].iloc[0])

        features['dam_7d_mean'] = np.mean(prices_7d) if prices_7d else features['dam_d1_mean']
        features['dam_7d_same_hour_mean'] = np.mean(same_hour_7d) if same_hour_7d else features['dam_lag_24h']

        # 4-week same day-of-week, same hour
        same_dow_hour = []
        for w in range(1, 5):
            dow_date = (target_date - timedelta(weeks=w)).date()
            dow_data = df[df.index.date == dow_date]
            if len(dow_data) >= 1:
                hour_mask = dow_data.index.hour == (target_hour - 1)
                if hour_mask.any():
                    same_dow_hour.append(dow_data.loc[hour_mask, 'dam_price'].iloc[0])

        features['dam_4w_same_dow_hour_mean'] = np.mean(same_dow_hour) if same_dow_hour else features['dam_lag_168h']

        # Pattern features (8)
        features['dam_d1_hour_ratio'] = features['dam_lag_24h'] / (features['dam_d1_mean'] + 1e-6)
        features['dam_7d_hour_ratio'] = features['dam_7d_same_hour_mean'] / (features['dam_7d_mean'] + 1e-6)

        d2_mean = np.nanmean(d2_prices) if d2_prices is not None else features['dam_d1_mean']
        d7_mean = np.nanmean(d7_prices) if d7_prices is not None else features['dam_d1_mean']
        features['dam_d1_vs_d2'] = features['dam_d1_mean'] - d2_mean
        features['dam_d1_vs_d7'] = features['dam_d1_mean'] - d7_mean

        d3_mean = np.nanmean(d3_prices) if d3_prices is not None else features['dam_d1_mean']
        features['dam_trend_3d'] = (features['dam_d1_mean'] - d3_mean) / 3

        features['dam_7d_cv'] = np.std(prices_7d) / (np.mean(prices_7d) + 1e-6) if prices_7d else 0
        features['dam_d1_range'] = features['dam_d1_max'] - features['dam_d1_min']
        features['dam_7d_same_hour_std'] = np.std(same_hour_7d) if len(same_hour_7d) > 1 else 0

        # Spike features (5)
        threshold = 100.0
        features['dam_d1_spike_count'] = int(np.sum(d1_prices > threshold))
        features['dam_7d_spike_count'] = int(np.sum(np.array(prices_7d) > threshold)) if prices_7d else 0
        features['dam_d1_had_spike'] = int(features['dam_d1_spike_count'] > 0)
        features['dam_lag_24h_is_spike'] = int(features['dam_lag_24h'] > threshold)
        features['dam_d1_max_hour'] = int(np.nanargmax(d1_prices) + 1)

        return features

    def _get_day_prices(self, day_data: pd.DataFrame) -> Optional[np.ndarray]:
        """Get prices as 24-hour array from day's data"""
        if len(day_data) < 24:
            return None

        # Sort by hour and extract prices
        day_data = day_data.copy()
        day_data['hour'] = day_data.index.hour
        day_data = day_data.sort_values('hour')
        return day_data['dam_price'].values[:24]

    def _naive_predict(self, df: pd.DataFrame, hour: int) -> float:
        """Naive prediction: same hour yesterday"""
        yesterday = df.index.max().date()
        yesterday_data = df[
            (df.index.date == yesterday) &
            (df.index.hour == hour - 1)
        ]
        if len(yesterday_data) > 0:
            return float(yesterday_data['dam_price'].iloc[-1])
        return float(df['dam_price'].mean())

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return {
            "model_type": "DAM V2 Per-Hour CatBoost",
            "models_loaded": len(self.models),
            "hours_covered": sorted(self.models.keys()),
            "feature_count": len(self.feature_names),
            "model_dir": str(self.model_dir),
        }


# Singleton instance
_v2_predictor: Optional[DAMV2Predictor] = None


def get_dam_v2_predictor() -> DAMV2Predictor:
    """Get singleton DAM V2 predictor instance"""
    global _v2_predictor
    if _v2_predictor is None:
        _v2_predictor = DAMV2Predictor()
    return _v2_predictor
