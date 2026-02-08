"""
DAM Feature Engineering

Extracts features from DAM price data for model training and inference.
Based on DAM_feature_extraction.ipynb logic.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta


# US Major Holidays (simplified)
US_MAJOR_HOLIDAYS = {
    (1, 1),    # New Year's Day
    (7, 4),    # Independence Day
    (11, 11),  # Veterans Day
    (12, 25),  # Christmas Day
    (12, 31),  # New Year's Eve
}


def is_us_holiday(dt: pd.Timestamp) -> int:
    """Check if date is a US holiday"""
    if (dt.month, dt.day) in US_MAJOR_HOLIDAYS:
        return 1

    # Thanksgiving: 4th Thursday of November
    if dt.month == 11 and dt.weekday() == 3:
        first_day = pd.Timestamp(dt.year, 11, 1)
        first_thursday = first_day + pd.Timedelta(days=(3 - first_day.weekday()) % 7)
        fourth_thursday = first_thursday + pd.Timedelta(weeks=3)
        if dt.date() == fourth_thursday.date():
            return 1

    # Labor Day: 1st Monday of September
    if dt.month == 9 and dt.weekday() == 0 and dt.day <= 7:
        return 1

    # Memorial Day: Last Monday of May
    if dt.month == 5 and dt.weekday() == 0:
        next_monday = dt + pd.Timedelta(weeks=1)
        if next_monday.month != 5:
            return 1

    return 0


@dataclass
class DAMFeatureConfig:
    """Configuration for DAM feature extraction"""
    hours: List[int] = field(default_factory=lambda: list(range(1, 25)))
    spike_threshold: float = 100.0
    peak_hours: List[int] = field(default_factory=lambda: list(range(7, 23)))
    summer_months: List[int] = field(default_factory=lambda: [6, 7, 8, 9])
    min_history_days: int = 28


# Feature names in order (35 features)
FEATURE_NAMES = [
    'hour', 'day_of_week', 'day_of_month', 'month', 'quarter',
    'week_of_year', 'is_weekend', 'is_holiday', 'is_peak_hour', 'is_summer',
    'dam_lag_24h', 'dam_lag_24h_prev', 'dam_lag_24h_next', 'dam_lag_48h',
    'dam_lag_168h', 'dam_d1_mean', 'dam_d1_max', 'dam_d1_min', 'dam_d1_std',
    'dam_7d_mean', 'dam_7d_same_hour_mean', 'dam_4w_same_dow_hour_mean',
    'dam_d1_hour_ratio', 'dam_7d_hour_ratio', 'dam_d1_vs_d2',
    'dam_d1_vs_d7', 'dam_trend_3d', 'dam_7d_cv', 'dam_d1_range',
    'dam_7d_same_hour_std', 'dam_d1_spike_count', 'dam_7d_spike_count',
    'dam_d1_had_spike', 'dam_lag_24h_is_spike', 'dam_d1_max_hour',
]

# Categorical feature indices
CATEGORICAL_INDICES = [0, 1, 2, 3, 4, 6, 7, 8, 9, 32, 33]


class DAMFeatureEngineer:
    """Extract features from DAM price data"""

    def __init__(self, config: DAMFeatureConfig = None):
        self.config = config or DAMFeatureConfig()
        self.feature_names = FEATURE_NAMES.copy()
        self.categorical_indices = CATEGORICAL_INDICES.copy()

    def extract_features(
        self,
        dam_df: pd.DataFrame,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Extract features from DAM price data

        Args:
            dam_df: DataFrame with DatetimeIndex and 'dam_price' column
            verbose: Print progress info

        Returns:
            DataFrame with 35 features + target + timestamp
        """
        if verbose:
            print("=" * 60)
            print("DAM Feature Extraction")
            print("=" * 60)

        dam_df = dam_df.copy()

        # Ensure DatetimeIndex
        if not isinstance(dam_df.index, pd.DatetimeIndex):
            raise ValueError("dam_df must have DatetimeIndex")

        # Add date and hour columns
        dam_df['date'] = dam_df.index.date
        dam_df['hour'] = dam_df.index.hour + 1  # HE1-HE24

        # Create pivot table (date x hour)
        price_pivot = dam_df.pivot_table(
            index='date',
            columns='hour',
            values='dam_price',
            aggfunc='first'
        )

        dates = price_pivot.index.tolist()
        n_dates = len(dates)

        if verbose:
            print(f"Date range: {dates[0]} ~ {dates[-1]}")
            print(f"Total days: {n_dates}")

        # Extract features
        records = []
        start_idx = self.config.min_history_days

        if verbose:
            print(f"Valid prediction days: {n_dates - start_idx}")

        for i in range(start_idx, n_dates):
            target_date = dates[i]

            for target_hour in self.config.hours:
                if target_hour not in price_pivot.columns:
                    continue

                target_price = price_pivot.loc[target_date, target_hour]
                if pd.isna(target_price):
                    continue

                features = self._build_features(
                    price_pivot, dates, i, target_hour
                )

                if features is None:
                    continue

                features['target'] = target_price
                d1_date = dates[i - 1]
                features['timestamp'] = pd.Timestamp(d1_date) + pd.Timedelta(hours=6)

                records.append(features)

        df = pd.DataFrame(records)
        col_order = self.feature_names + ['target', 'timestamp']
        df = df[col_order]

        if verbose:
            print(f"Total records: {len(df):,}")
            print(f"Features: {len(self.feature_names)}")

        return df

    def _build_features(
        self,
        price_pivot: pd.DataFrame,
        dates: List,
        target_date_idx: int,
        target_hour: int
    ) -> Optional[Dict]:
        """Build features for a single prediction task"""
        features = {}

        target_date = dates[target_date_idx]
        d1_date = dates[target_date_idx - 1]
        d2_date = dates[target_date_idx - 2]
        d3_date = dates[target_date_idx - 3]
        d7_date = dates[target_date_idx - 7]

        try:
            d1_prices = price_pivot.loc[d1_date].values
            d2_prices = price_pivot.loc[d2_date].values
            d3_prices = price_pivot.loc[d3_date].values
            d7_prices = price_pivot.loc[d7_date].values
        except KeyError:
            return None

        if np.any(pd.isna(d1_prices)) or np.any(pd.isna(d7_prices)):
            return None

        hour_idx = target_hour - 1
        target_dt = pd.Timestamp(target_date)

        # Time features (indices 0-9)
        features['hour'] = target_hour
        features['day_of_week'] = target_dt.dayofweek
        features['day_of_month'] = target_dt.day
        features['month'] = target_dt.month
        features['quarter'] = target_dt.quarter
        features['week_of_year'] = target_dt.isocalendar()[1]
        features['is_weekend'] = int(target_dt.dayofweek >= 5)
        features['is_holiday'] = is_us_holiday(target_dt)
        features['is_peak_hour'] = int(target_hour in self.config.peak_hours)
        features['is_summer'] = int(target_dt.month in self.config.summer_months)

        # Lag features (indices 10-21)
        features['dam_lag_24h'] = d1_prices[hour_idx]
        features['dam_lag_24h_prev'] = d1_prices[max(0, hour_idx - 1)]
        features['dam_lag_24h_next'] = d1_prices[min(23, hour_idx + 1)]
        features['dam_lag_48h'] = d2_prices[hour_idx]
        features['dam_lag_168h'] = d7_prices[hour_idx]
        features['dam_d1_mean'] = np.nanmean(d1_prices)
        features['dam_d1_max'] = np.nanmax(d1_prices)
        features['dam_d1_min'] = np.nanmin(d1_prices)
        features['dam_d1_std'] = np.nanstd(d1_prices)

        # 7-day stats
        prices_7d = []
        for j in range(1, 8):
            if target_date_idx - j >= 0:
                day_prices = price_pivot.loc[dates[target_date_idx - j]].values
                prices_7d.extend(day_prices[~pd.isna(day_prices)])

        features['dam_7d_mean'] = np.mean(prices_7d) if prices_7d else features['dam_d1_mean']

        # Same hour 7d
        same_hour_7d = []
        for j in range(1, 8):
            if target_date_idx - j >= 0:
                val = price_pivot.loc[dates[target_date_idx - j], target_hour]
                if not pd.isna(val):
                    same_hour_7d.append(val)
        features['dam_7d_same_hour_mean'] = np.mean(same_hour_7d) if same_hour_7d else features['dam_lag_24h']

        # Same dow/hour 4w
        same_dow_hour = []
        for w in range(1, 5):
            idx = target_date_idx - 7 * w
            if idx >= 0:
                val = price_pivot.loc[dates[idx], target_hour]
                if not pd.isna(val):
                    same_dow_hour.append(val)
        features['dam_4w_same_dow_hour_mean'] = np.mean(same_dow_hour) if same_dow_hour else features['dam_lag_168h']

        # Pattern features (indices 22-29)
        features['dam_d1_hour_ratio'] = features['dam_lag_24h'] / (features['dam_d1_mean'] + 1e-6)
        features['dam_7d_hour_ratio'] = features['dam_7d_same_hour_mean'] / (features['dam_7d_mean'] + 1e-6)

        d2_mean = np.nanmean(d2_prices)
        d7_mean = np.nanmean(d7_prices)
        features['dam_d1_vs_d2'] = features['dam_d1_mean'] - d2_mean
        features['dam_d1_vs_d7'] = features['dam_d1_mean'] - d7_mean

        d3_mean = np.nanmean(d3_prices)
        features['dam_trend_3d'] = (features['dam_d1_mean'] - d3_mean) / 3

        features['dam_7d_cv'] = np.std(prices_7d) / (np.mean(prices_7d) + 1e-6) if prices_7d else 0
        features['dam_d1_range'] = features['dam_d1_max'] - features['dam_d1_min']
        features['dam_7d_same_hour_std'] = np.std(same_hour_7d) if len(same_hour_7d) > 1 else 0

        # Spike features (indices 30-34)
        threshold = self.config.spike_threshold
        features['dam_d1_spike_count'] = int(np.sum(d1_prices > threshold))
        features['dam_7d_spike_count'] = int(np.sum(np.array(prices_7d) > threshold)) if prices_7d else 0
        features['dam_d1_had_spike'] = int(features['dam_d1_spike_count'] > 0)
        features['dam_lag_24h_is_spike'] = int(features['dam_lag_24h'] > threshold)
        features['dam_d1_max_hour'] = int(np.nanargmax(d1_prices) + 1)

        return features

    def extract_inference_features(
        self,
        price_history: pd.DataFrame,
        target_date: datetime,
        target_hours: List[int] = None
    ) -> pd.DataFrame:
        """
        Extract features for real-time inference

        Args:
            price_history: Historical DAM prices (at least 28 days)
            target_date: The date to predict (D)
            target_hours: Hours to predict (default: 1-24)

        Returns:
            DataFrame with features for each hour
        """
        if target_hours is None:
            target_hours = list(range(1, 25))

        # Create pivot table
        price_history = price_history.copy()
        price_history['date'] = price_history.index.date
        price_history['hour'] = price_history.index.hour + 1

        price_pivot = price_history.pivot_table(
            index='date',
            columns='hour',
            values='dam_price',
            aggfunc='first'
        )

        dates = sorted(price_pivot.index.tolist())

        # Find the target date index (or use last date + 1)
        target_date_obj = target_date.date() if hasattr(target_date, 'date') else target_date

        # Build features for each hour
        records = []
        for target_hour in target_hours:
            # Use last available date as D-1
            features = self._build_inference_features(
                price_pivot, dates, target_date_obj, target_hour
            )
            if features:
                records.append(features)

        return pd.DataFrame(records)

    def _build_inference_features(
        self,
        price_pivot: pd.DataFrame,
        dates: List,
        target_date,
        target_hour: int
    ) -> Optional[Dict]:
        """Build features for inference (no target available)"""
        # D-1 is the last date in history
        if len(dates) < 28:
            return None

        d1_idx = len(dates) - 1  # Last date is D-1
        d2_idx = d1_idx - 1
        d3_idx = d1_idx - 2
        d7_idx = d1_idx - 6

        try:
            d1_prices = price_pivot.loc[dates[d1_idx]].values
            d2_prices = price_pivot.loc[dates[d2_idx]].values
            d3_prices = price_pivot.loc[dates[d3_idx]].values
            d7_prices = price_pivot.loc[dates[d7_idx]].values
        except (KeyError, IndexError):
            return None

        hour_idx = target_hour - 1
        target_dt = pd.Timestamp(target_date)

        features = {}

        # Time features
        features['hour'] = target_hour
        features['day_of_week'] = target_dt.dayofweek
        features['day_of_month'] = target_dt.day
        features['month'] = target_dt.month
        features['quarter'] = target_dt.quarter
        features['week_of_year'] = target_dt.isocalendar()[1]
        features['is_weekend'] = int(target_dt.dayofweek >= 5)
        features['is_holiday'] = is_us_holiday(target_dt)
        features['is_peak_hour'] = int(target_hour in self.config.peak_hours)
        features['is_summer'] = int(target_dt.month in self.config.summer_months)

        # Lag features
        features['dam_lag_24h'] = d1_prices[hour_idx] if not pd.isna(d1_prices[hour_idx]) else 0
        features['dam_lag_24h_prev'] = d1_prices[max(0, hour_idx - 1)]
        features['dam_lag_24h_next'] = d1_prices[min(23, hour_idx + 1)]
        features['dam_lag_48h'] = d2_prices[hour_idx] if not pd.isna(d2_prices[hour_idx]) else 0
        features['dam_lag_168h'] = d7_prices[hour_idx] if not pd.isna(d7_prices[hour_idx]) else 0
        features['dam_d1_mean'] = np.nanmean(d1_prices)
        features['dam_d1_max'] = np.nanmax(d1_prices)
        features['dam_d1_min'] = np.nanmin(d1_prices)
        features['dam_d1_std'] = np.nanstd(d1_prices)

        # 7-day stats
        prices_7d = []
        for j in range(7):
            if d1_idx - j >= 0:
                day_prices = price_pivot.loc[dates[d1_idx - j]].values
                prices_7d.extend(day_prices[~pd.isna(day_prices)])

        features['dam_7d_mean'] = np.mean(prices_7d) if prices_7d else features['dam_d1_mean']

        same_hour_7d = []
        for j in range(7):
            if d1_idx - j >= 0:
                val = price_pivot.loc[dates[d1_idx - j], target_hour]
                if not pd.isna(val):
                    same_hour_7d.append(val)
        features['dam_7d_same_hour_mean'] = np.mean(same_hour_7d) if same_hour_7d else features['dam_lag_24h']

        same_dow_hour = []
        for w in range(4):
            idx = d1_idx - 7 * w
            if idx >= 0:
                val = price_pivot.loc[dates[idx], target_hour]
                if not pd.isna(val):
                    same_dow_hour.append(val)
        features['dam_4w_same_dow_hour_mean'] = np.mean(same_dow_hour) if same_dow_hour else features['dam_lag_168h']

        # Pattern features
        features['dam_d1_hour_ratio'] = features['dam_lag_24h'] / (features['dam_d1_mean'] + 1e-6)
        features['dam_7d_hour_ratio'] = features['dam_7d_same_hour_mean'] / (features['dam_7d_mean'] + 1e-6)

        d2_mean = np.nanmean(d2_prices)
        d7_mean = np.nanmean(d7_prices)
        features['dam_d1_vs_d2'] = features['dam_d1_mean'] - d2_mean
        features['dam_d1_vs_d7'] = features['dam_d1_mean'] - d7_mean

        d3_mean = np.nanmean(d3_prices)
        features['dam_trend_3d'] = (features['dam_d1_mean'] - d3_mean) / 3

        features['dam_7d_cv'] = np.std(prices_7d) / (np.mean(prices_7d) + 1e-6) if prices_7d else 0
        features['dam_d1_range'] = features['dam_d1_max'] - features['dam_d1_min']
        features['dam_7d_same_hour_std'] = np.std(same_hour_7d) if len(same_hour_7d) > 1 else 0

        # Spike features
        threshold = self.config.spike_threshold
        features['dam_d1_spike_count'] = int(np.sum(d1_prices > threshold))
        features['dam_7d_spike_count'] = int(np.sum(np.array(prices_7d) > threshold)) if prices_7d else 0
        features['dam_d1_had_spike'] = int(features['dam_d1_spike_count'] > 0)
        features['dam_lag_24h_is_spike'] = int(features['dam_lag_24h'] > threshold)
        features['dam_d1_max_hour'] = int(np.nanargmax(d1_prices) + 1)

        return features

    def get_feature_names(self) -> List[str]:
        return self.feature_names

    def get_categorical_indices(self) -> List[int]:
        return self.categorical_indices
