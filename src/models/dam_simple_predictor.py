"""
Simple DAM Price Predictor

A lightweight DAM price predictor that works with limited historical data.
Uses basic features: time, recent prices, and rolling statistics.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import joblib
from pathlib import Path

from catboost import CatBoostRegressor


@dataclass
class SimpleDAMPrediction:
    """DAM price prediction result"""
    hour_ending: int
    predicted_price: float
    actual_price: Optional[float] = None
    timestamp: Optional[datetime] = None


class SimpleDAMPredictor:
    """
    Simple DAM price predictor using minimal features.
    Works with as little as 7 days of historical data.
    """

    def __init__(self, model_path: Optional[Path] = None):
        self.model = None
        self.feature_names = [
            'hour', 'day_of_week', 'is_weekend',
            'dam_lag_24h', 'dam_lag_48h',
            'dam_d1_mean', 'dam_d1_max', 'dam_d1_min',
            'dam_3d_same_hour_mean'
        ]
        self.categorical_features = [0, 1, 2]  # hour, day_of_week, is_weekend

        if model_path and model_path.exists():
            self.load(model_path)

    def extract_features(self, dam_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract simple features from DAM price data.

        Args:
            dam_df: DataFrame with DatetimeIndex and 'dam_price' column

        Returns:
            DataFrame with features and target
        """
        df = dam_df.copy()

        # Ensure we have dam_price column
        if 'lmp' in df.columns and 'dam_price' not in df.columns:
            df['dam_price'] = df['lmp']

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            else:
                raise ValueError("DataFrame must have DatetimeIndex or 'timestamp' column")

        df = df.sort_index()

        # Time features
        df['hour'] = df.index.hour + 1  # Hour ending (1-24)
        df['day_of_week'] = df.index.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        # Lag features
        df['dam_lag_24h'] = df['dam_price'].shift(24)
        df['dam_lag_48h'] = df['dam_price'].shift(48)

        # Daily statistics (previous day)
        df['date'] = df.index.date
        daily_stats = df.groupby('date')['dam_price'].agg(['mean', 'max', 'min']).shift(1)
        daily_stats.columns = ['dam_d1_mean', 'dam_d1_max', 'dam_d1_min']
        df = df.join(daily_stats, on='date')

        # Same hour average (last 3 days)
        df['dam_3d_same_hour_mean'] = df.groupby('hour')['dam_price'].transform(
            lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
        )

        # Target: next day's price at same hour
        df['target'] = df['dam_price'].shift(-24)

        # Keep timestamp for reference
        df['timestamp'] = df.index

        # Drop rows with NaN
        feature_cols = self.feature_names + ['target', 'timestamp']
        df = df.dropna(subset=self.feature_names + ['target'])

        return df[feature_cols]

    def train(self, dam_df: pd.DataFrame, test_ratio: float = 0.2) -> Dict:
        """
        Train the model on DAM price data.

        Args:
            dam_df: DataFrame with historical DAM prices
            test_ratio: Fraction of data to use for testing

        Returns:
            Dict with training metrics
        """
        # Extract features
        features_df = self.extract_features(dam_df)

        if len(features_df) < 48:  # Need at least 2 days
            raise ValueError(f"Not enough data for training: {len(features_df)} samples")

        # Split train/test
        n_test = int(len(features_df) * test_ratio)
        train_df = features_df[:-n_test] if n_test > 0 else features_df
        test_df = features_df[-n_test:] if n_test > 0 else None

        X_train = train_df[self.feature_names]
        y_train = train_df['target']

        # Train CatBoost model
        self.model = CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3,
            cat_features=self.categorical_features,
            verbose=100,
            early_stopping_rounds=50,
        )

        if test_df is not None and len(test_df) > 0:
            X_test = test_df[self.feature_names]
            y_test = test_df['target']
            self.model.fit(X_train, y_train, eval_set=(X_test, y_test))

            # Calculate metrics
            predictions = self.model.predict(X_test)
            mae = np.mean(np.abs(predictions - y_test))
            rmse = np.sqrt(np.mean((predictions - y_test) ** 2))

            return {
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'mae': mae,
                'rmse': rmse,
            }
        else:
            self.model.fit(X_train, y_train)
            return {
                'train_samples': len(X_train),
                'test_samples': 0,
            }

    def predict_next_day(
        self,
        dam_df: pd.DataFrame,
        target_date: Optional[datetime] = None
    ) -> List[SimpleDAMPrediction]:
        """
        Predict DAM prices for the next day.

        Args:
            dam_df: DataFrame with recent DAM prices (at least 3 days)
            target_date: Date to predict (default: tomorrow)

        Returns:
            List of predictions for 24 hours
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() or load() first.")

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

        # Get recent data for feature calculation
        recent_df = df.tail(72)  # Last 3 days

        # Calculate daily stats from yesterday
        yesterday = target_date - timedelta(days=1)
        yesterday_data = df[df.index.date == yesterday]

        if len(yesterday_data) == 0:
            # Use most recent complete day
            complete_days = df.groupby(df.index.date).size()
            complete_days = complete_days[complete_days >= 24]
            if len(complete_days) > 0:
                yesterday = complete_days.index[-1]
                yesterday_data = df[df.index.date == yesterday]

        d1_mean = yesterday_data['dam_price'].mean() if len(yesterday_data) > 0 else df['dam_price'].mean()
        d1_max = yesterday_data['dam_price'].max() if len(yesterday_data) > 0 else df['dam_price'].max()
        d1_min = yesterday_data['dam_price'].min() if len(yesterday_data) > 0 else df['dam_price'].min()

        predictions = []

        for hour in range(1, 25):
            # Create feature vector
            features = {}
            features['hour'] = hour
            features['day_of_week'] = pd.Timestamp(target_date).dayofweek
            features['is_weekend'] = 1 if features['day_of_week'] >= 5 else 0

            # Lag features (same hour yesterday and 2 days ago)
            same_hour_yesterday = df[
                (df.index.date == yesterday) &
                (df.index.hour == hour - 1)
            ]
            features['dam_lag_24h'] = same_hour_yesterday['dam_price'].iloc[-1] if len(same_hour_yesterday) > 0 else d1_mean

            two_days_ago = yesterday - timedelta(days=1)
            same_hour_2d = df[
                (df.index.date == two_days_ago) &
                (df.index.hour == hour - 1)
            ]
            features['dam_lag_48h'] = same_hour_2d['dam_price'].iloc[-1] if len(same_hour_2d) > 0 else d1_mean

            features['dam_d1_mean'] = d1_mean
            features['dam_d1_max'] = d1_max
            features['dam_d1_min'] = d1_min

            # 3-day same hour average
            same_hour_data = df[df.index.hour == hour - 1].tail(3)
            features['dam_3d_same_hour_mean'] = same_hour_data['dam_price'].mean() if len(same_hour_data) > 0 else d1_mean

            # Create DataFrame for prediction
            X = pd.DataFrame([features])[self.feature_names]

            # Predict
            predicted_price = float(self.model.predict(X)[0])

            predictions.append(SimpleDAMPrediction(
                hour_ending=hour,
                predicted_price=predicted_price,
                timestamp=datetime.combine(target_date, datetime.min.time()) + timedelta(hours=hour-1)
            ))

        return predictions

    def save(self, model_path: Path):
        """Save model to disk"""
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'categorical_features': self.categorical_features,
        }, model_path)

    def load(self, model_path: Path):
        """Load model from disk"""
        data = joblib.load(model_path)
        self.model = data['model']
        self.feature_names = data['feature_names']
        self.categorical_features = data['categorical_features']

    def is_ready(self) -> bool:
        """Check if model is loaded and ready"""
        return self.model is not None


def train_and_save_simple_model(
    settlement_point: str = "HB_HOUSTON",
    model_dir: Path = None
) -> Dict:
    """
    Train and save a simple DAM predictor.

    Args:
        settlement_point: Settlement point to train for
        model_dir: Directory to save model

    Returns:
        Training metrics
    """
    from data.influxdb_fetcher import create_fetcher_from_env

    if model_dir is None:
        model_dir = Path(__file__).parent.parent.parent / "models"

    # Fetch data
    print(f"Fetching DAM prices for {settlement_point}...")
    fetcher = create_fetcher_from_env()
    df = fetcher.fetch_dam_prices(settlement_point=settlement_point)
    fetcher.close()

    print(f"Loaded {len(df)} records")
    print(f"Date range: {df.index.min()} to {df.index.max()}")

    # Train model
    predictor = SimpleDAMPredictor()
    metrics = predictor.train(df)

    # Save model
    model_path = model_dir / f"dam_simple_{settlement_point.lower()}.joblib"
    predictor.save(model_path)
    print(f"Model saved to {model_path}")

    return metrics


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from dotenv import load_dotenv
    load_dotenv()

    metrics = train_and_save_simple_model("HB_HOUSTON")
    print(f"\nTraining complete!")
    print(f"Metrics: {metrics}")
