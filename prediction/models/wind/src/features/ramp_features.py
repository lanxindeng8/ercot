"""
Ramp Detection Feature Engineering

Features designed to detect and quantify wind power ramps.
CRITICAL for ERCOT operations.

Key Focus: Ramp-Down + No-Solar Period
=========================================
When wind power drops rapidly during evening/night (no solar backup),
the system must rely on gas generation, often causing price spikes.
This is the highest-priority scenario for early warning.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

try:
    from astral import LocationInfo
    from astral.sun import sun
    ASTRAL_AVAILABLE = True
except ImportError:
    ASTRAL_AVAILABLE = False


@dataclass
class RampDefinition:
    """Definition of a wind power ramp event."""
    magnitude_mw: float      # Absolute change threshold
    duration_minutes: int    # Time window
    direction: str           # 'up', 'down', 'both'


# ERCOT-relevant ramp thresholds
RAMP_DEFINITIONS = {
    'small_15m': RampDefinition(500, 15, 'both'),
    'medium_1h': RampDefinition(2000, 60, 'both'),
    'large_3h': RampDefinition(5000, 180, 'both'),
    'severe_3h': RampDefinition(8000, 180, 'both'),
}

# Texas approximate location for sunset/sunrise
TEXAS_LOCATION = {
    'name': 'Texas',
    'region': 'USA',
    'timezone': 'America/Chicago',
    'latitude': 31.5,    # Central Texas
    'longitude': -99.5,
}


class RampFeatureEngineer:
    """
    Compute ramp-focused features.

    PRIORITY: Ramp-Down Detection
    ----------------------------
    Ramp-down events (wind power dropping) are more critical than ramp-up
    because they require gas/thermal ramping which is expensive and slow.

    Features:
    1. Wind speed/power change rates (d/dt)
    2. Acceleration (d^2/dt^2)
    3. Power curve sensitivity zones
    4. Frontal passage indicators
    5. No-solar period risk scoring
    """

    def __init__(
        self,
        ramp_thresholds: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize ramp feature engineer.

        Args:
            ramp_thresholds: Dict of threshold names to MW values
        """
        self.ramp_thresholds = ramp_thresholds or {
            'small': 1000,
            'medium': 2000,
            'large': 3000,
        }

        # Initialize astral for sunset/sunrise
        if ASTRAL_AVAILABLE:
            self.location = LocationInfo(
                TEXAS_LOCATION['name'],
                TEXAS_LOCATION['region'],
                TEXAS_LOCATION['timezone'],
                TEXAS_LOCATION['latitude'],
                TEXAS_LOCATION['longitude'],
            )
        else:
            self.location = None

    @staticmethod
    def compute_change_rate(
        values: np.ndarray,
        time_hours: np.ndarray,
    ) -> np.ndarray:
        """
        Compute rate of change.

        Args:
            values: Time series values (e.g., wind speed or power)
            time_hours: Time in hours

        Returns:
            Rate of change (units per hour)
        """
        return np.gradient(values, time_hours)

    @staticmethod
    def compute_acceleration(
        values: np.ndarray,
        time_hours: np.ndarray,
    ) -> np.ndarray:
        """
        Compute acceleration (second derivative).

        Args:
            values: Time series values
            time_hours: Time in hours

        Returns:
            Acceleration (units per hour^2)
        """
        first_deriv = np.gradient(values, time_hours)
        return np.gradient(first_deriv, time_hours)

    def compute_ramp_over_horizon(
        self,
        values: np.ndarray,
        horizons: List[int] = [1, 3, 6, 12],
    ) -> Dict[str, np.ndarray]:
        """
        Compute ramp (change) over multiple horizons.

        Args:
            values: Time series values
            horizons: List of horizon steps

        Returns:
            Dict mapping horizon to change values
        """
        results = {}
        for h in horizons:
            if len(values) > h:
                # Forward-looking change
                change = np.zeros_like(values)
                change[:-h] = values[h:] - values[:-h]
                results[f'change_{h}h'] = change
            else:
                results[f'change_{h}h'] = np.zeros_like(values)
        return results

    def is_no_solar_period(
        self,
        timestamp: datetime,
    ) -> bool:
        """
        Check if timestamp is in no-solar period (after sunset, before sunrise).

        Args:
            timestamp: Datetime to check

        Returns:
            True if no solar generation expected
        """
        if self.location is None or not ASTRAL_AVAILABLE:
            # Fallback: simple hour-based check
            hour = timestamp.hour
            return hour >= 19 or hour < 7

        try:
            s = sun(self.location.observer, date=timestamp.date())
            sunrise = s['sunrise'].replace(tzinfo=None)
            sunset = s['sunset'].replace(tzinfo=None)

            ts_naive = timestamp.replace(tzinfo=None) if timestamp.tzinfo else timestamp
            return ts_naive < sunrise or ts_naive > sunset
        except Exception:
            # Fallback
            hour = timestamp.hour
            return hour >= 19 or hour < 7

    def compute_solar_timing_features(
        self,
        timestamp: datetime,
    ) -> Dict[str, float]:
        """
        Compute features related to solar timing.

        Args:
            timestamp: Current timestamp

        Returns:
            Dict with solar timing features
        """
        features = {}

        hour = timestamp.hour
        features['is_no_solar_period'] = float(self.is_no_solar_period(timestamp))
        features['is_evening_peak'] = float(17 <= hour <= 21)
        features['is_morning_ramp'] = float(5 <= hour <= 9)
        features['is_night'] = float(hour >= 22 or hour < 5)

        if self.location is not None and ASTRAL_AVAILABLE:
            try:
                s = sun(self.location.observer, date=timestamp.date())
                sunset = s['sunset'].replace(tzinfo=None)
                sunrise = s['sunrise'].replace(tzinfo=None)
                ts_naive = timestamp.replace(tzinfo=None) if timestamp.tzinfo else timestamp

                # Minutes to/from sunset
                if ts_naive < sunset:
                    features['minutes_to_sunset'] = (sunset - ts_naive).total_seconds() / 60
                    features['minutes_since_sunset'] = 0
                else:
                    features['minutes_to_sunset'] = 0
                    features['minutes_since_sunset'] = (ts_naive - sunset).total_seconds() / 60

                # Minutes to/from sunrise
                if ts_naive < sunrise:
                    features['minutes_to_sunrise'] = (sunrise - ts_naive).total_seconds() / 60
                else:
                    # Next day sunrise
                    features['minutes_to_sunrise'] = 0

            except Exception:
                features['minutes_to_sunset'] = 0
                features['minutes_since_sunset'] = 0
                features['minutes_to_sunrise'] = 0
        else:
            features['minutes_to_sunset'] = 0
            features['minutes_since_sunset'] = 0
            features['minutes_to_sunrise'] = 0

        return features

    def compute_ramp_down_risk(
        self,
        wind_change_mw: float,
        current_hour: int,
        is_no_solar: bool,
    ) -> float:
        """
        Compute wind ramp-down risk score (0-1).

        HIGH RISK CONDITIONS:
        1. Wind power forecast to drop significantly (> 2000 MW)
        2. No solar period (after sunset / before sunrise)
        3. Evening demand peak (17:00 - 21:00)

        Args:
            wind_change_mw: Forecast wind power change (negative = drop)
            current_hour: Hour of day (0-23)
            is_no_solar: Whether currently in no-solar period

        Returns:
            Risk score 0.0 to 1.0
        """
        risk = 0.0

        # Wind drop severity
        if wind_change_mw < -1000:
            risk += 0.15
        if wind_change_mw < -2000:
            risk += 0.20
        if wind_change_mw < -3000:
            risk += 0.20
        if wind_change_mw < -5000:
            risk += 0.15

        # No solar backup
        if is_no_solar:
            risk += 0.15

        # Evening peak + no solar = MOST DANGEROUS
        is_evening_peak = 17 <= current_hour <= 21
        if is_evening_peak and is_no_solar:
            risk += 0.15

        return min(risk, 1.0)

    def compute_frontal_indicator(
        self,
        temperature_change: float,
        wind_dir_change: float,
        pressure_change: float,
    ) -> float:
        """
        Detect weather front passage probability.

        Fronts cause rapid wind changes. Cold fronts are especially important.

        Cold Front Signature:
        - Temperature drops
        - Wind direction shifts (often northerly after passage)
        - Pressure rises

        Args:
            temperature_change: Temperature change (K/hour)
            wind_dir_change: Wind direction change (degrees/hour)
            pressure_change: Pressure change (Pa/hour)

        Returns:
            Front probability 0.0 to 1.0
        """
        score = 0.0

        # Temperature drop (cold front)
        if temperature_change < -2:  # K/hour
            score += 0.3
        elif temperature_change < -1:
            score += 0.15

        # Wind direction shift
        if abs(wind_dir_change) > 45:  # degrees/hour
            score += 0.3
        elif abs(wind_dir_change) > 20:
            score += 0.15

        # Pressure rise (post-frontal)
        if pressure_change > 100:  # Pa/hour
            score += 0.3
        elif pressure_change > 50:
            score += 0.15

        # Strong combined signal
        if temperature_change < -1 and abs(wind_dir_change) > 30 and pressure_change > 50:
            score += 0.1

        return min(score, 1.0)

    def compute_features(
        self,
        timestamps: pd.DatetimeIndex,
        wind_power_forecast: np.ndarray,
        wind_speed: Optional[np.ndarray] = None,
        temperature: Optional[np.ndarray] = None,
        wind_direction: Optional[np.ndarray] = None,
        pressure: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Compute all ramp features.

        Args:
            timestamps: Time index
            wind_power_forecast: Forecast wind power (MW)
            wind_speed: Wind speed (m/s, optional)
            temperature: Temperature (K, optional)
            wind_direction: Wind direction (degrees, optional)
            pressure: Pressure (Pa, optional)

        Returns:
            DataFrame with ramp features
        """
        n = len(timestamps)
        features = {}

        # Time in hours for derivatives
        time_hours = np.arange(n, dtype=float)

        # Power change rates
        if len(wind_power_forecast) > 1:
            features['power_change_rate'] = self.compute_change_rate(
                wind_power_forecast, time_hours
            )
            features['power_acceleration'] = self.compute_acceleration(
                wind_power_forecast, time_hours
            )
        else:
            features['power_change_rate'] = np.zeros(n)
            features['power_acceleration'] = np.zeros(n)

        # Ramp over horizons
        ramp_horizons = self.compute_ramp_over_horizon(
            wind_power_forecast, [1, 3, 6]
        )
        features.update({
            'ramp_1h': ramp_horizons.get('change_1h', np.zeros(n)),
            'ramp_3h': ramp_horizons.get('change_3h', np.zeros(n)),
            'ramp_6h': ramp_horizons.get('change_6h', np.zeros(n)),
        })

        # Ramp DOWN specific (negative values = drop)
        features['ramp_down_1h'] = np.minimum(ramp_horizons.get('change_1h', np.zeros(n)), 0)
        features['ramp_down_3h'] = np.minimum(ramp_horizons.get('change_3h', np.zeros(n)), 0)

        # Solar timing features (per timestamp)
        solar_features = {
            'is_no_solar_period': [],
            'is_evening_peak': [],
            'minutes_to_sunset': [],
            'minutes_since_sunset': [],
        }
        for ts in timestamps:
            sf = self.compute_solar_timing_features(ts)
            for key in solar_features:
                solar_features[key].append(sf.get(key, 0))

        features.update({k: np.array(v) for k, v in solar_features.items()})

        # Ramp-down risk score
        risk_scores = []
        for i, ts in enumerate(timestamps):
            wind_change = features['ramp_3h'][i] if i < len(features['ramp_3h']) else 0
            risk = self.compute_ramp_down_risk(
                wind_change,
                ts.hour,
                bool(features['is_no_solar_period'][i]),
            )
            risk_scores.append(risk)
        features['ramp_down_no_solar_risk'] = np.array(risk_scores)

        # Wind speed features (if provided)
        if wind_speed is not None and len(wind_speed) > 1:
            features['ws_change_rate'] = self.compute_change_rate(wind_speed, time_hours)

        # Frontal indicator (if weather data provided)
        if all(x is not None for x in [temperature, wind_direction, pressure]):
            frontal_scores = []
            for i in range(n):
                if i == 0:
                    frontal_scores.append(0.0)
                else:
                    temp_change = temperature[i] - temperature[i-1]
                    wd_change = wind_direction[i] - wind_direction[i-1]
                    # Handle wind direction wrap-around
                    if wd_change > 180:
                        wd_change -= 360
                    elif wd_change < -180:
                        wd_change += 360
                    pres_change = pressure[i] - pressure[i-1]
                    frontal_scores.append(
                        self.compute_frontal_indicator(temp_change, wd_change, pres_change)
                    )
            features['frontal_indicator'] = np.array(frontal_scores)

        return pd.DataFrame(features, index=timestamps)
