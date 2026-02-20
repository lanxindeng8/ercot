"""
Texas/ERCOT Wind Region Definitions

Defines the major wind generation regions in ERCOT and provides
utilities for mapping HRRR grid coordinates to these regions.
"""

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np


@dataclass
class WindRegion:
    """Definition of an ERCOT wind region."""
    name: str
    lat_bounds: Tuple[float, float]  # (min_lat, max_lat)
    lon_bounds: Tuple[float, float]  # (min_lon, max_lon) in negative degrees
    installed_capacity_mw: float     # Approximate installed capacity
    hub_height_m: float = 80.0       # Typical turbine hub height


# Major ERCOT wind regions
# Reference: ERCOT Generation Reports and Wind Integration Reports
ERCOT_WIND_REGIONS: Dict[str, WindRegion] = {
    'PANHANDLE': WindRegion(
        name='Panhandle',
        lat_bounds=(34.0, 36.5),
        lon_bounds=(-103.0, -100.0),
        installed_capacity_mw=8000,
        hub_height_m=80.0,
    ),
    'WEST': WindRegion(
        name='West Texas',
        lat_bounds=(30.5, 34.0),
        lon_bounds=(-104.5, -100.5),
        installed_capacity_mw=15000,
        hub_height_m=80.0,
    ),
    'COASTAL': WindRegion(
        name='Gulf Coast',
        lat_bounds=(26.5, 30.0),
        lon_bounds=(-97.5, -95.0),
        installed_capacity_mw=5000,
        hub_height_m=80.0,
    ),
    'SOUTH': WindRegion(
        name='South Texas',
        lat_bounds=(26.0, 29.5),
        lon_bounds=(-100.5, -97.5),
        installed_capacity_mw=3000,
        hub_height_m=80.0,
    ),
}

# Texas overall bounds for HRRR subsetting
TEXAS_BOUNDS = {
    'lat_min': 25.5,
    'lat_max': 36.5,
    'lon_min': -106.5,
    'lon_max': -93.5,
}

# Total ERCOT wind capacity (approximate)
TOTAL_WIND_CAPACITY_MW = sum(r.installed_capacity_mw for r in ERCOT_WIND_REGIONS.values())


def create_region_mask(
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    region: WindRegion,
) -> np.ndarray:
    """
    Create a boolean mask for a specific wind region on the HRRR grid.

    Args:
        lat_grid: 2D array of latitudes (HRRR grid)
        lon_grid: 2D array of longitudes (HRRR grid, 0-360 format)
        region: WindRegion definition

    Returns:
        Boolean mask array with same shape as lat_grid
    """
    # Convert HRRR longitudes from 0-360 to -180 to 180 if needed
    lon_converted = np.where(lon_grid > 180, lon_grid - 360, lon_grid)

    lat_mask = (lat_grid >= region.lat_bounds[0]) & (lat_grid <= region.lat_bounds[1])
    lon_mask = (lon_converted >= region.lon_bounds[0]) & (lon_converted <= region.lon_bounds[1])

    return lat_mask & lon_mask


def create_texas_mask(
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
) -> np.ndarray:
    """
    Create a boolean mask for the entire Texas region.

    Args:
        lat_grid: 2D array of latitudes
        lon_grid: 2D array of longitudes (0-360 format)

    Returns:
        Boolean mask for Texas
    """
    lon_converted = np.where(lon_grid > 180, lon_grid - 360, lon_grid)

    lat_mask = (lat_grid >= TEXAS_BOUNDS['lat_min']) & (lat_grid <= TEXAS_BOUNDS['lat_max'])
    lon_mask = (lon_converted >= TEXAS_BOUNDS['lon_min']) & (lon_converted <= TEXAS_BOUNDS['lon_max'])

    return lat_mask & lon_mask


def create_all_region_masks(
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Create masks for all ERCOT wind regions.

    Args:
        lat_grid: 2D array of latitudes
        lon_grid: 2D array of longitudes

    Returns:
        Dictionary mapping region name to boolean mask
    """
    masks = {}
    for name, region in ERCOT_WIND_REGIONS.items():
        masks[name] = create_region_mask(lat_grid, lon_grid, region)

    # Also add full Texas mask
    masks['TEXAS'] = create_texas_mask(lat_grid, lon_grid)

    return masks


def get_region_weights() -> Dict[str, float]:
    """
    Get capacity-based weights for each region.

    Returns:
        Dictionary mapping region name to weight (sums to 1.0)
    """
    total = sum(r.installed_capacity_mw for r in ERCOT_WIND_REGIONS.values())
    return {
        name: region.installed_capacity_mw / total
        for name, region in ERCOT_WIND_REGIONS.items()
    }
