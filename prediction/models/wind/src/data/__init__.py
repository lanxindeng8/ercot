"""
Data Module for Wind Forecasting

Data clients for HRRR weather data and ERCOT wind generation.
"""

from .hrrr_client import (
    HRRRWindClient,
    WIND_VARIABLES,
    compute_wind_speed,
    compute_wind_direction,
)
from .texas_regions import (
    TEXAS_BOUNDS,
    ERCOT_WIND_REGIONS,
    WindRegion,
    create_texas_mask,
    create_region_mask,
    create_all_region_masks,
)

__all__ = [
    # HRRR client
    'HRRRWindClient',
    'WIND_VARIABLES',
    'compute_wind_speed',
    'compute_wind_direction',
    # Texas regions
    'TEXAS_BOUNDS',
    'ERCOT_WIND_REGIONS',
    'WindRegion',
    'create_texas_mask',
    'create_region_mask',
    'create_all_region_masks',
]
