"""
HRRR Wind Data Client

Fetches HRRR forecast data via Earth2Studio for Texas/ERCOT region.
Extracts wind components at hub height (80m) and surface (10m).
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import xarray as xr
from loguru import logger

from .texas_regions import (
    TEXAS_BOUNDS,
    ERCOT_WIND_REGIONS,
    create_texas_mask,
    create_all_region_masks,
)


# Key variables to fetch from HRRR for wind forecasting
WIND_VARIABLES = [
    'u10m',   # 10m U wind component (m/s)
    'v10m',   # 10m V wind component (m/s)
    'u80m',   # 80m U wind component (hub height)
    'v80m',   # 80m V wind component (hub height)
    't2m',    # 2m temperature (K)
    'sp',     # Surface pressure (Pa)
]


class HRRRWindClient:
    """
    Client for fetching HRRR wind forecast data.

    Uses Earth2Studio's HRRR_FX class for data access.
    Provides Texas-specific subsetting and regional aggregation.
    """

    def __init__(
        self,
        source: str = 'aws',
        cache: bool = True,
        max_workers: int = 12,
        verbose: bool = True,
    ):
        """
        Initialize HRRR client.

        Args:
            source: Data source ('aws', 'google', 'nomads')
            cache: Whether to cache downloaded data
            max_workers: Number of parallel download workers
            verbose: Whether to show download progress
        """
        self.source = source
        self.cache = cache
        self.max_workers = max_workers
        self.verbose = verbose

        self._hrrr_fx = None
        self._lat_grid = None
        self._lon_grid = None
        self._region_masks = None

    def _init_hrrr(self):
        """Lazy initialization of HRRR client."""
        if self._hrrr_fx is None:
            try:
                from earth2studio.data import HRRR_FX
                self._hrrr_fx = HRRR_FX(
                    source=self.source,
                    cache=self.cache,
                    max_workers=self.max_workers,
                    verbose=self.verbose,
                )
                logger.info(f"Initialized HRRR_FX client with source={self.source}")
            except ImportError:
                raise ImportError(
                    "earth2studio is required. Install with: pip install earth2studio"
                )

    def _init_grid(self, data: xr.DataArray):
        """Initialize lat/lon grid from HRRR data."""
        if self._lat_grid is None:
            self._lat_grid = data.coords['lat'].values
            self._lon_grid = data.coords['lon'].values
            self._region_masks = create_all_region_masks(
                self._lat_grid, self._lon_grid
            )
            logger.info(f"Initialized HRRR grid: {self._lat_grid.shape}")

    def fetch_forecast(
        self,
        init_time: datetime,
        lead_times: Optional[List[int]] = None,
        variables: Optional[List[str]] = None,
    ) -> xr.DataArray:
        """
        Fetch HRRR forecast data.

        Args:
            init_time: Forecast initialization time (UTC)
            lead_times: Lead times in hours (default: 0-12)
            variables: HRRR variable names (default: WIND_VARIABLES)

        Returns:
            xr.DataArray with dims [time, lead_time, variable, hrrr_y, hrrr_x]
        """
        self._init_hrrr()

        if lead_times is None:
            lead_times = list(range(0, 13))  # 0-12 hours

        if variables is None:
            variables = WIND_VARIABLES

        # Convert lead times to timedelta
        lead_time_td = [timedelta(hours=h) for h in lead_times]

        logger.info(
            f"Fetching HRRR forecast: init={init_time}, "
            f"lead_times={lead_times}, variables={variables}"
        )

        data = self._hrrr_fx(
            time=init_time,
            lead_time=lead_time_td,
            variable=variables,
        )

        # Initialize grid info
        self._init_grid(data)

        return data

    def fetch_forecast_texas(
        self,
        init_time: datetime,
        lead_times: Optional[List[int]] = None,
        variables: Optional[List[str]] = None,
    ) -> xr.DataArray:
        """
        Fetch HRRR forecast data subsetted to Texas region.

        This is more memory efficient than fetching the full CONUS grid.

        Args:
            init_time: Forecast initialization time (UTC)
            lead_times: Lead times in hours
            variables: HRRR variable names

        Returns:
            xr.DataArray subsetted to Texas bounding box
        """
        # Fetch full data first
        data = self.fetch_forecast(init_time, lead_times, variables)

        # Subset to Texas
        return self.subset_texas(data)

    def subset_texas(self, data: xr.DataArray) -> xr.DataArray:
        """
        Extract Texas region from full HRRR grid.

        Uses bounding box subsetting for efficiency.

        Args:
            data: Full HRRR data array

        Returns:
            Subsetted data array
        """
        if self._region_masks is None:
            self._init_grid(data)

        texas_mask = self._region_masks['TEXAS']

        # Find bounding indices
        y_indices = np.where(texas_mask.any(axis=1))[0]
        x_indices = np.where(texas_mask.any(axis=0))[0]

        if len(y_indices) == 0 or len(x_indices) == 0:
            logger.warning("No Texas data found in grid!")
            return data

        y_slice = slice(y_indices[0], y_indices[-1] + 1)
        x_slice = slice(x_indices[0], x_indices[-1] + 1)

        return data.isel(hrrr_y=y_slice, hrrr_x=x_slice)

    def aggregate_to_regions(
        self,
        data: xr.DataArray,
        regions: Optional[List[str]] = None,
    ) -> Dict[str, xr.DataArray]:
        """
        Aggregate grid data to ERCOT regions.

        Computes mean values over each region's grid points.

        Args:
            data: HRRR data array
            regions: List of region names (default: all regions)

        Returns:
            Dictionary mapping region name to aggregated data
        """
        if self._region_masks is None:
            self._init_grid(data)

        if regions is None:
            regions = list(ERCOT_WIND_REGIONS.keys())

        results = {}
        for region_name in regions:
            if region_name not in self._region_masks:
                logger.warning(f"Unknown region: {region_name}")
                continue

            mask = self._region_masks[region_name]

            # Apply mask and compute regional mean
            # Note: This is a simplified approach; could weight by capacity
            masked_data = data.where(mask, drop=False)
            regional_mean = masked_data.mean(dim=['hrrr_y', 'hrrr_x'])

            results[region_name] = regional_mean

        return results

    def compute_regional_statistics(
        self,
        data: xr.DataArray,
        regions: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, xr.DataArray]]:
        """
        Compute statistics (mean, std, min, max) for each region.

        Args:
            data: HRRR data array
            regions: List of region names

        Returns:
            Nested dict: region -> statistic -> data
        """
        if self._region_masks is None:
            self._init_grid(data)

        if regions is None:
            regions = list(ERCOT_WIND_REGIONS.keys())

        results = {}
        for region_name in regions:
            if region_name not in self._region_masks:
                continue

            mask = self._region_masks[region_name]
            masked_data = data.where(mask, drop=False)

            results[region_name] = {
                'mean': masked_data.mean(dim=['hrrr_y', 'hrrr_x']),
                'std': masked_data.std(dim=['hrrr_y', 'hrrr_x']),
                'min': masked_data.min(dim=['hrrr_y', 'hrrr_x']),
                'max': masked_data.max(dim=['hrrr_y', 'hrrr_x']),
            }

        return results


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


def compute_wind_direction(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Compute meteorological wind direction from U/V components.

    Args:
        u: U (eastward) wind component
        v: V (northward) wind component

    Returns:
        Wind direction in degrees (0-360, from which wind is blowing)
    """
    # Meteorological convention: direction wind is coming FROM
    direction = (270 - np.degrees(np.arctan2(v, u))) % 360
    return direction
