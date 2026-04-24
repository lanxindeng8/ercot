"""
Feature Engineering Module

Implements feature computation for ERCOT RTM LMP Spike prediction
Including: price structure, supply-demand balance, weather-driven, and temporal features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class PriceStructureFeatures:
    """Price structure feature computation

    Objective: Capture regional scarcity and congestion signals
    """

    @staticmethod
    def calculate(df: pd.DataFrame, zones: List[str] = ['CPS', 'West', 'Houston']) -> pd.DataFrame:
        """Compute price structure features

        Args:
            df: DataFrame containing price data
                Required columns: P_CPS, P_West, P_Houston, P_Hub, P_CPS_DA, P_West_DA, P_Houston_DA
            zones: List of zones

        Returns:
            DataFrame containing price structure features
        """
        features = pd.DataFrame(index=df.index)

        for zone in zones:
            # 1. Zone-system spread
            features[f'spread_{zone}_hub'] = df[f'P_{zone}'] - df['P_Hub']

            # 2. Real-time to day-ahead premium
            if f'P_{zone}_DA' in df.columns:
                features[f'spread_rt_da_{zone}'] = df[f'P_{zone}'] - df[f'P_{zone}_DA']

            # 3. Price slope (5-minute)
            # Assumes data is at 5-minute intervals
            features[f'price_ramp_5m_{zone}'] = df[f'P_{zone}'].diff(1) / 5  # $/MWh/min

            # 4. Price slope (15-minute)
            features[f'price_ramp_15m_{zone}'] = df[f'P_{zone}'].diff(3) / 15  # Assumes 3 five-minute steps

            # 5. Price acceleration
            features[f'price_accel_{zone}'] = features[f'price_ramp_5m_{zone}'].diff(1)

        # 6. Cross-zone spread (special focus on CPS-Houston)
        if 'CPS' in zones and 'Houston' in zones:
            features['spread_CPS_Houston'] = df['P_CPS'] - df['P_Houston']
            features['spread_CPS_Houston_ramp'] = features['spread_CPS_Houston'].diff(1)

        return features


class SupplyDemandFeatures:
    """Supply-demand balance feature computation

    Objective: Capture system/regional stress conditions
    """

    @staticmethod
    def calculate(df: pd.DataFrame, lookback_days: int = 30) -> pd.DataFrame:
        """Compute supply-demand balance features

        Args:
            df: DataFrame containing system data
                Required columns: Load, Wind, Solar, Gas, Coal, ESR
            lookback_days: Rolling window in days (for computing anomalies)

        Returns:
            DataFrame containing supply-demand balance features
        """
        features = pd.DataFrame(index=df.index)

        # 1. Net load
        features['net_load'] = df['Load'] - df['Wind'] - df['Solar']

        # 2. Net load ramp rate (MW/5min)
        features['net_load_ramp_5m'] = features['net_load'].diff(1)
        features['net_load_ramp_15m'] = features['net_load'].diff(3)

        # 3. Net load acceleration
        features['net_load_accel'] = features['net_load_ramp_5m'].diff(1)

        # 4. Wind features
        # Rolling mean and standard deviation (for computing anomalies)
        window_size = lookback_days * 24 * 12  # Assumes 5-minute data
        wind_rolling_mean = df['Wind'].rolling(window=window_size, min_periods=1).mean()
        wind_rolling_std = df['Wind'].rolling(window=window_size, min_periods=1).std()

        # Wind anomaly (standardized deviation)
        features['wind_anomaly'] = (df['Wind'] - wind_rolling_mean) / (wind_rolling_std + 1e-6)

        # Wind ramp
        features['wind_ramp'] = df['Wind'].diff(1)
        features['wind_ramp_15m'] = df['Wind'].diff(3)

        # 5. Gas generation saturation
        # Using 7-day rolling 95th percentile as reference capacity
        gas_window = 7 * 24 * 12
        gas_p95 = df['Gas'].rolling(window=gas_window, min_periods=1).quantile(0.95)
        features['gas_saturation'] = df['Gas'] / (gas_p95 + 1e-6)

        # 6. Coal stress (nighttime ramp-up flag)
        # Coal change
        coal_diff = df['Coal'].diff(1)
        # Nighttime hours (0-5)
        is_night = df.index.hour.isin(range(0, 6))
        # Nighttime coal ramp-up
        features['coal_stress'] = ((coal_diff > 0) & is_night).astype(int)
        features['coal_ramp'] = coal_diff

        # 7. Energy storage system net output
        features['esr_net_output'] = df['ESR']
        features['esr_is_charging'] = (df['ESR'] < 0).astype(int)
        features['esr_is_discharging'] = (df['ESR'] > 0).astype(int)

        # 8. Solar ramp
        features['solar_ramp'] = df['Solar'].diff(1)
        features['solar_ramp_15m'] = df['Solar'].diff(3)

        return features


class WeatherFeatures:
    """Weather-driven feature computation (Zone-level)

    Objective: Capture demand-side shock signals
    """

    @staticmethod
    def calculate(df: pd.DataFrame, zones: List[str] = ['CPS', 'West', 'Houston'],
                  lookback_days: int = 30) -> pd.DataFrame:
        """Compute weather-driven features

        Args:
            df: DataFrame containing weather data
                Required columns: T_{zone}, WindSpeed_{zone}, WindDir_{zone} for each zone
            zones: List of zones
            lookback_days: Rolling window in days

        Returns:
            DataFrame containing weather features
        """
        features = pd.DataFrame(index=df.index)

        for zone in zones:
            temp_col = f'T_{zone}'
            wind_speed_col = f'WindSpeed_{zone}'
            wind_dir_col = f'WindDir_{zone}'

            if temp_col not in df.columns:
                continue

            # 1. Temperature anomaly (relative to historical same-hour mean)
            # Compute rolling mean grouped by hour
            hourly_temp_mean = df.groupby(df.index.hour)[temp_col].transform(
                lambda x: x.rolling(window=lookback_days, min_periods=1).mean()
            )
            features[f'T_anomaly_{zone}'] = df[temp_col] - hourly_temp_mean

            # 2. Cooling rate (degrees F/hour)
            # Assumes data is at 15-minute intervals, 12 steps = 1 hour
            features[f'T_ramp_{zone}'] = df[temp_col].diff(12) / 1  # degrees F/hr

            # 3. Wind Chill Index
            if wind_speed_col in df.columns:
                T = df[temp_col]
                v = df[wind_speed_col]
                # Wind Chill formula (applicable for T <= 50 degrees F and v >= 3 mph)
                features[f'WindChill_{zone}'] = (
                    35.74 + 0.6215 * T - 35.75 * (v ** 0.16) + 0.4275 * T * (v ** 0.16)
                )
                # For inapplicable conditions, use actual temperature
                mask = (T > 50) | (v < 3)
                features.loc[mask, f'WindChill_{zone}'] = T[mask]

            # 4. Cold front flag
            if wind_dir_col in df.columns:
                # North wind: wind direction between 315-45 degrees
                wind_to_north = (df[wind_dir_col] > 315) | (df[wind_dir_col] < 45)
                # Rapid cooling + north wind = cold front
                features[f'ColdFront_{zone}'] = (
                    (features[f'T_ramp_{zone}'] < -5) & wind_to_north
                ).astype(int)
            else:
                features[f'ColdFront_{zone}'] = 0

        return features


class TemporalFeatures:
    """Temporal feature computation

    Objective: Capture intraday patterns and solar recovery windows
    """

    @staticmethod
    def calculate(df: pd.DataFrame, latitude: float = 29.76) -> pd.DataFrame:
        """Compute temporal features

        Args:
            df: DataFrame with datetime index
            latitude: Latitude (for computing sunrise time, San Antonio ~29.76 degrees N)

        Returns:
            DataFrame containing temporal features
        """
        features = pd.DataFrame(index=df.index)

        # 1. Basic temporal features
        features['hour'] = df.index.hour
        features['day_of_week'] = df.index.dayofweek
        features['month'] = df.index.month

        # 2. Evening peak flag
        features['is_evening_peak'] = df.index.hour.isin(range(17, 23)).astype(int)

        # 3. Sunrise time estimation (simplified version)
        # Simplified formula used here; in production use the ephem or astral library
        day_of_year = df.index.dayofyear
        # Simplified sunrise time calculation (hours, local time)
        # This is an approximation; use accurate astronomical algorithms in production
        declination = 23.45 * np.sin(np.radians((360/365) * (day_of_year - 81)))
        sunrise_hour = 12 - (1/15) * np.degrees(
            np.arccos(-np.tan(np.radians(latitude)) * np.tan(np.radians(declination)))
        )

        # Minutes to sunrise
        current_hour = df.index.hour + df.index.minute / 60
        minutes_to_sunrise = (sunrise_hour - current_hour) * 60
        # Handle day-crossing cases
        minutes_to_sunrise = np.where(minutes_to_sunrise < -720, minutes_to_sunrise + 1440, minutes_to_sunrise)
        features['minutes_to_sunrise'] = minutes_to_sunrise

        # 4. Pre/post sunrise flags
        features['is_pre_sunrise'] = (minutes_to_sunrise > 0) & (minutes_to_sunrise < 120)
        features['is_post_sunrise'] = (minutes_to_sunrise < 0) & (minutes_to_sunrise > -120)

        # 5. Expected solar ramp (if Solar data is available)
        if 'Solar' in df.columns:
            features['solar_ramp_expected'] = df['Solar'].diff(3)  # 15-minute change

        return features


class FeatureEngineer:
    """Feature engineering main class

    Integrates all feature computation modules
    """

    def __init__(self, zones: List[str] = ['CPS', 'West', 'Houston'],
                 lookback_days: int = 30):
        """Initialize

        Args:
            zones: List of zones for which to compute features
            lookback_days: Rolling window in days
        """
        self.zones = zones
        self.lookback_days = lookback_days

    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all features

        Args:
            df: Raw data DataFrame

        Returns:
            DataFrame containing all features
        """
        print("Computing price structure features...")
        price_features = PriceStructureFeatures.calculate(df, self.zones)

        print("Computing supply-demand balance features...")
        supply_demand_features = SupplyDemandFeatures.calculate(df, self.lookback_days)

        print("Computing weather-driven features...")
        weather_features = WeatherFeatures.calculate(df, self.zones, self.lookback_days)

        print("Computing temporal features...")
        temporal_features = TemporalFeatures.calculate(df)

        # Merge all features
        all_features = pd.concat([
            df,  # Preserve original data
            price_features,
            supply_demand_features,
            weather_features,
            temporal_features
        ], axis=1)

        print(f"Feature computation complete! Total features: {len(all_features.columns)}")

        return all_features

    def get_feature_names(self, feature_type: Optional[str] = None) -> List[str]:
        """Get feature name list

        Args:
            feature_type: Feature type ('price', 'supply_demand', 'weather', 'temporal', None=all)

        Returns:
            List of feature names
        """
        if feature_type == 'price':
            features = []
            for zone in self.zones:
                features.extend([
                    f'spread_{zone}_hub',
                    f'spread_rt_da_{zone}',
                    f'price_ramp_5m_{zone}',
                    f'price_ramp_15m_{zone}',
                    f'price_accel_{zone}',
                ])
            features.append('spread_CPS_Houston')
            features.append('spread_CPS_Houston_ramp')
            return features

        elif feature_type == 'supply_demand':
            return [
                'net_load', 'net_load_ramp_5m', 'net_load_ramp_15m', 'net_load_accel',
                'wind_anomaly', 'wind_ramp', 'wind_ramp_15m',
                'gas_saturation',
                'coal_stress', 'coal_ramp',
                'esr_net_output', 'esr_is_charging', 'esr_is_discharging',
                'solar_ramp', 'solar_ramp_15m',
            ]

        elif feature_type == 'weather':
            features = []
            for zone in self.zones:
                features.extend([
                    f'T_anomaly_{zone}',
                    f'T_ramp_{zone}',
                    f'WindChill_{zone}',
                    f'ColdFront_{zone}',
                ])
            return features

        elif feature_type == 'temporal':
            return [
                'hour', 'day_of_week', 'month',
                'is_evening_peak',
                'minutes_to_sunrise', 'is_pre_sunrise', 'is_post_sunrise',
                'solar_ramp_expected',
            ]

        else:
            # Return all features
            return (
                self.get_feature_names('price') +
                self.get_feature_names('supply_demand') +
                self.get_feature_names('weather') +
                self.get_feature_names('temporal')
            )


if __name__ == '__main__':
    # Test code
    print("Feature engineering module loaded successfully!")
    print("\nSupported feature types:")
    print("1. Price Structure Features")
    print("2. Supply-Demand Balance Features")
    print("3. Weather-Driven Features")
    print("4. Temporal Features")
