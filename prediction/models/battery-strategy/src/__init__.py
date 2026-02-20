"""
Battery Energy Storage Strategy Package

A Python library for optimizing battery energy storage system operations
based on electricity price arbitrage using linear programming.
"""

from .battery_config import BatteryConfig, create_default_battery
from .optimizer import (
    BatteryOptimizer,
    OptimizationResult,
    interpolate_hourly_to_5min,
    create_sample_prices
)
from .visualizer import BatteryVisualizer
from .price_utils import (
    load_dam_prices_from_csv,
    resample_to_hourly,
    get_next_day_prices,
    create_synthetic_prices,
    calculate_price_statistics
)

__version__ = "0.1.0"

__all__ = [
    "BatteryConfig",
    "create_default_battery",
    "BatteryOptimizer",
    "OptimizationResult",
    "BatteryVisualizer",
    "interpolate_hourly_to_5min",
    "create_sample_prices",
    "load_dam_prices_from_csv",
    "resample_to_hourly",
    "get_next_day_prices",
    "create_synthetic_prices",
    "calculate_price_statistics",
]
