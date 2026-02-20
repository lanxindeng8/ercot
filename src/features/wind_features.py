"""
Wind Feature Engineering

Converts raw HRRR U/V wind components to operationally relevant features
for wind power forecasting.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import xarray as xr


class WindFeatureEngineer:
    """
    Compute wind power features from HRRR forecast data.

    Key Features:
    1. Wind speed at hub height (80m)
    2. Wind power density
    3. Wind direction (for farm orientation)
    4. Wind shear (difference 10m vs 80m)
    5. Normalized power output via power curve
    """

    # Default turbine parameters (generic utility-scale)
    DEFAULT_CUT_IN = 3.0      # m/s - turbine starts generating
    DEFAULT_RATED = 12.0      # m/s - turbine reaches rated power
    DEFAULT_CUT_OUT = 25.0    # m/s - turbine shuts down for safety

    def __init__(
        self,
        cut_in_speed: float = DEFAULT_CUT_IN,
        rated_speed: float = DEFAULT_RATED,
        cut_out_speed: float = DEFAULT_CUT_OUT,
    ):
        """
        Initialize with turbine parameters.

        Args:
            cut_in_speed: Cut-in wind speed (m/s)
            rated_speed: Rated wind speed (m/s)
            cut_out_speed: Cut-out wind speed (m/s)
        """
        self.cut_in = cut_in_speed
        self.rated = rated_speed
        self.cut_out = cut_out_speed

    @staticmethod
    def compute_wind_speed(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Compute wind speed from U/V components.

        Args:
            u: U (eastward) wind component (m/s)
            v: V (northward) wind component (m/s)

        Returns:
            Wind speed (m/s)
        """
        return np.sqrt(u**2 + v**2)

    @staticmethod
    def compute_wind_direction(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Compute meteorological wind direction.

        Args:
            u: U wind component
            v: V wind component

        Returns:
            Wind direction in degrees (0-360, from which wind is blowing)
        """
        direction = (270 - np.degrees(np.arctan2(v, u))) % 360
        return direction

    @staticmethod
    def compute_power_density(
        wind_speed: np.ndarray,
        temperature_k: Optional[np.ndarray] = None,
        pressure_pa: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute wind power density (W/m^2).

        P = 0.5 * rho * v^3

        Args:
            wind_speed: Wind speed (m/s)
            temperature_k: Air temperature in Kelvin (optional)
            pressure_pa: Surface pressure in Pa (optional)

        Returns:
            Power density (W/m^2)
        """
        # Calculate air density if T and P provided
        if temperature_k is not None and pressure_pa is not None:
            # Ideal gas law: rho = P / (R * T)
            R_air = 287.05  # J/(kg·K), specific gas constant for dry air
            air_density = pressure_pa / (R_air * temperature_k)
        else:
            air_density = 1.225  # kg/m^3, standard sea level

        return 0.5 * air_density * wind_speed**3

    @staticmethod
    def compute_wind_shear(
        ws_lower: np.ndarray,
        ws_upper: np.ndarray,
        z_lower: float = 10.0,
        z_upper: float = 80.0,
    ) -> np.ndarray:
        """
        Compute wind shear exponent using power law.

        ws(z2) / ws(z1) = (z2 / z1)^alpha
        alpha = ln(ws2/ws1) / ln(z2/z1)

        Args:
            ws_lower: Wind speed at lower height (m/s)
            ws_upper: Wind speed at upper height (m/s)
            z_lower: Lower height (m)
            z_upper: Upper height (m)

        Returns:
            Wind shear exponent (typical range: 0.1 - 0.3)
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            alpha = np.log(ws_upper / ws_lower) / np.log(z_upper / z_lower)
            # Replace invalid values with typical neutral stability value
            alpha = np.where(np.isfinite(alpha), alpha, 0.143)
            # Clip to reasonable range
            alpha = np.clip(alpha, 0.0, 0.5)
        return alpha

    def apply_power_curve(
        self,
        wind_speed: np.ndarray,
    ) -> np.ndarray:
        """
        Apply generic turbine power curve.

        Regions:
        - Below cut-in: 0
        - Cut-in to rated: cubic increase
        - Rated to cut-out: 1.0 (rated power)
        - Above cut-out: 0

        Args:
            wind_speed: Wind speed (m/s)

        Returns:
            Normalized power output [0, 1]
        """
        power = np.zeros_like(wind_speed, dtype=float)

        # Cubic region (cut-in to rated)
        cubic_mask = (wind_speed >= self.cut_in) & (wind_speed < self.rated)
        power[cubic_mask] = (
            (wind_speed[cubic_mask] - self.cut_in) / (self.rated - self.cut_in)
        ) ** 3

        # Rated region
        rated_mask = (wind_speed >= self.rated) & (wind_speed < self.cut_out)
        power[rated_mask] = 1.0

        # Cut-out (high wind shutdown)
        power[wind_speed >= self.cut_out] = 0.0

        return power

    def compute_power_sensitivity(self, wind_speed: np.ndarray) -> np.ndarray:
        """
        Compute power curve sensitivity (d_power/d_wind).

        High sensitivity in cubic region (3-12 m/s) means small wind changes
        cause large power changes - important for ramp detection.

        Args:
            wind_speed: Wind speed (m/s)

        Returns:
            Sensitivity coefficient (unitless, 0-1 normalized)
        """
        sensitivity = np.zeros_like(wind_speed, dtype=float)

        # In cubic region, derivative is 3 * ((ws - cut_in) / (rated - cut_in))^2
        cubic_mask = (wind_speed >= self.cut_in) & (wind_speed < self.rated)
        normalized_ws = (wind_speed[cubic_mask] - self.cut_in) / (self.rated - self.cut_in)
        sensitivity[cubic_mask] = 3 * normalized_ws**2

        # Normalize to 0-1 range (max sensitivity at rated speed)
        sensitivity = sensitivity / 3.0

        return sensitivity

    def compute_features_from_hrrr(
        self,
        data: xr.DataArray,
        region_name: str = 'SYSTEM',
    ) -> pd.DataFrame:
        """
        Compute all wind features from HRRR data array.

        Args:
            data: HRRR xarray with variables u10m, v10m, u80m, v80m, t2m, sp
            region_name: Name for output (used in column naming)

        Returns:
            DataFrame with wind features indexed by time
        """
        # Extract variables (handle both regional and full grid data)
        try:
            u10m = data.sel(variable='u10m').values
            v10m = data.sel(variable='v10m').values
            u80m = data.sel(variable='u80m').values
            v80m = data.sel(variable='v80m').values
        except KeyError as e:
            raise ValueError(f"Missing required variable: {e}")

        # Optional variables
        try:
            t2m = data.sel(variable='t2m').values
            sp = data.sel(variable='sp').values
        except KeyError:
            t2m = None
            sp = None

        # Compute wind speeds
        ws_10m = self.compute_wind_speed(u10m, v10m)
        ws_80m = self.compute_wind_speed(u80m, v80m)

        # Wind direction at hub height
        wd_80m = self.compute_wind_direction(u80m, v80m)

        # Power density
        power_density = self.compute_power_density(ws_80m, t2m, sp)

        # Wind shear
        shear = self.compute_wind_shear(ws_10m, ws_80m)

        # Normalized power (via power curve)
        normalized_power = self.apply_power_curve(ws_80m)

        # Power sensitivity
        power_sensitivity = self.compute_power_sensitivity(ws_80m)

        # Build feature dictionary
        # If data has spatial dims, take mean; otherwise use as-is
        def spatial_agg(arr, agg_func='mean'):
            """Aggregate over spatial dimensions if present."""
            if arr.ndim > 2:  # Has spatial dims
                if agg_func == 'mean':
                    return np.nanmean(arr, axis=(-2, -1))
                elif agg_func == 'std':
                    return np.nanstd(arr, axis=(-2, -1))
                elif agg_func == 'max':
                    return np.nanmax(arr, axis=(-2, -1))
                elif agg_func == 'min':
                    return np.nanmin(arr, axis=(-2, -1))
            return arr

        features = {
            f'ws_80m_mean': spatial_agg(ws_80m, 'mean'),
            f'ws_80m_std': spatial_agg(ws_80m, 'std'),
            f'ws_80m_max': spatial_agg(ws_80m, 'max'),
            f'ws_80m_min': spatial_agg(ws_80m, 'min'),
            f'ws_10m_mean': spatial_agg(ws_10m, 'mean'),
            f'wd_80m_mean': spatial_agg(wd_80m, 'mean'),  # Note: should use circular mean
            f'power_density_mean': spatial_agg(power_density, 'mean'),
            f'shear_mean': spatial_agg(shear, 'mean'),
            f'normalized_power_mean': spatial_agg(normalized_power, 'mean'),
            f'power_sensitivity_mean': spatial_agg(power_sensitivity, 'mean'),
        }

        # Get time coordinates
        if 'time' in data.coords:
            time_index = pd.to_datetime(data.coords['time'].values)
        else:
            time_index = None

        # Build DataFrame
        df = pd.DataFrame(features)
        if time_index is not None:
            if len(time_index) == 1:
                # Expand for lead times
                if 'lead_time' in data.coords:
                    lead_times = data.coords['lead_time'].values
                    # Create valid times
                    valid_times = [time_index[0] + lt for lt in lead_times]
                    df.index = pd.DatetimeIndex(valid_times)
            else:
                df.index = time_index

        return df


def compute_wind_features_batch(
    hrrr_data: Dict[str, xr.DataArray],
    engineer: Optional[WindFeatureEngineer] = None,
) -> pd.DataFrame:
    """
    Compute wind features for multiple regions.

    Args:
        hrrr_data: Dict mapping region name to HRRR data
        engineer: WindFeatureEngineer instance (optional)

    Returns:
        DataFrame with features for all regions
    """
    if engineer is None:
        engineer = WindFeatureEngineer()

    all_features = []
    for region_name, data in hrrr_data.items():
        features = engineer.compute_features_from_hrrr(data, region_name)
        # Add region prefix to columns
        features = features.add_prefix(f'{region_name}_')
        all_features.append(features)

    if all_features:
        return pd.concat(all_features, axis=1)
    return pd.DataFrame()
