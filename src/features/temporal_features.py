"""
Temporal Feature Engineering

Time-based features for wind forecasting.
"""

from datetime import datetime
from typing import Tuple
import numpy as np
import pandas as pd


class TemporalFeatureEngineer:
    """
    Compute time-based features.

    Features:
    1. Hour of day (cyclical encoding)
    2. Day of year (cyclical encoding)
    3. Day of week
    4. ERCOT peak hours
    5. Historical ramp-prone hours
    """

    # ERCOT peak demand hours (typical)
    PEAK_HOURS = [14, 15, 16, 17, 18, 19, 20]

    # Hours historically prone to wind ramps in Texas
    # Morning: nocturnal low-level jet decay
    # Evening: surface heating decay, frontal passages
    RAMP_PRONE_HOURS = [5, 6, 7, 8, 17, 18, 19, 20, 21]

    @staticmethod
    def encode_cyclical(
        value: np.ndarray,
        period: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode cyclical features using sin/cos transformation.

        This preserves the circular nature of time features
        (e.g., hour 23 is close to hour 0).

        Args:
            value: Values to encode (e.g., hour 0-23)
            period: Full cycle period (e.g., 24 for hours)

        Returns:
            Tuple of (sin_encoded, cos_encoded) arrays
        """
        radians = 2 * np.pi * value / period
        return np.sin(radians), np.cos(radians)

    def compute_features(
        self,
        timestamps: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """
        Compute temporal features.

        Args:
            timestamps: DatetimeIndex of timestamps

        Returns:
            DataFrame with temporal features
        """
        features = {}

        # Extract components
        hours = timestamps.hour.values
        days_of_week = timestamps.dayofweek.values
        days_of_year = timestamps.dayofyear.values
        months = timestamps.month.values

        # Hour of day (cyclical)
        hour_sin, hour_cos = self.encode_cyclical(hours, 24)
        features['hour_sin'] = hour_sin
        features['hour_cos'] = hour_cos
        features['hour'] = hours  # Also keep raw hour for tree models

        # Day of week (cyclical)
        dow_sin, dow_cos = self.encode_cyclical(days_of_week, 7)
        features['dow_sin'] = dow_sin
        features['dow_cos'] = dow_cos
        features['day_of_week'] = days_of_week

        # Day of year (cyclical) - captures seasonal patterns
        doy_sin, doy_cos = self.encode_cyclical(days_of_year, 365.25)
        features['doy_sin'] = doy_sin
        features['doy_cos'] = doy_cos

        # Month (cyclical)
        month_sin, month_cos = self.encode_cyclical(months, 12)
        features['month_sin'] = month_sin
        features['month_cos'] = month_cos
        features['month'] = months

        # Binary flags
        features['is_weekend'] = np.isin(days_of_week, [5, 6]).astype(float)
        features['is_peak_hour'] = np.isin(hours, self.PEAK_HOURS).astype(float)
        features['is_ramp_prone_hour'] = np.isin(hours, self.RAMP_PRONE_HOURS).astype(float)

        # Time of day categories
        features['is_morning'] = ((hours >= 5) & (hours < 12)).astype(float)
        features['is_afternoon'] = ((hours >= 12) & (hours < 17)).astype(float)
        features['is_evening'] = ((hours >= 17) & (hours < 22)).astype(float)
        features['is_night'] = ((hours >= 22) | (hours < 5)).astype(float)

        # Season (for tree models)
        # Winter: Dec, Jan, Feb (12, 1, 2)
        # Spring: Mar, Apr, May (3, 4, 5)
        # Summer: Jun, Jul, Aug (6, 7, 8)
        # Fall: Sep, Oct, Nov (9, 10, 11)
        features['is_winter'] = np.isin(months, [12, 1, 2]).astype(float)
        features['is_spring'] = np.isin(months, [3, 4, 5]).astype(float)
        features['is_summer'] = np.isin(months, [6, 7, 8]).astype(float)
        features['is_fall'] = np.isin(months, [9, 10, 11]).astype(float)

        return pd.DataFrame(features, index=timestamps)

    @staticmethod
    def compute_lag_features(
        series: pd.Series,
        lags: list = [1, 2, 3, 6, 12, 24],
    ) -> pd.DataFrame:
        """
        Compute lagged features from a time series.

        Args:
            series: Time series to lag
            lags: List of lag periods

        Returns:
            DataFrame with lagged features
        """
        features = {}
        name = series.name or 'value'

        for lag in lags:
            features[f'{name}_lag_{lag}'] = series.shift(lag)

        return pd.DataFrame(features, index=series.index)

    @staticmethod
    def compute_rolling_features(
        series: pd.Series,
        windows: list = [6, 12, 24],
    ) -> pd.DataFrame:
        """
        Compute rolling statistics.

        Args:
            series: Time series
            windows: Rolling window sizes

        Returns:
            DataFrame with rolling features
        """
        features = {}
        name = series.name or 'value'

        for window in windows:
            features[f'{name}_rolling_mean_{window}'] = series.rolling(window).mean()
            features[f'{name}_rolling_std_{window}'] = series.rolling(window).std()
            features[f'{name}_rolling_min_{window}'] = series.rolling(window).min()
            features[f'{name}_rolling_max_{window}'] = series.rolling(window).max()

        return pd.DataFrame(features, index=series.index)
