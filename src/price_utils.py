"""
Utility functions for processing electricity price data
"""

import numpy as np
import pandas as pd
from typing import Union, Optional
from datetime import datetime, timedelta


def load_dam_prices_from_csv(
    file_path: str,
    date_column: str = "date",
    price_column: str = "price"
) -> pd.DataFrame:
    """
    Load Day-Ahead Market (DAM) prices from CSV file

    Args:
        file_path: Path to CSV file
        date_column: Name of date/time column
        price_column: Name of price column

    Returns:
        DataFrame with datetime index and price column
    """
    df = pd.read_csv(file_path)
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.set_index(date_column)
    df = df[[price_column]].rename(columns={price_column: "price"})
    return df


def resample_to_hourly(
    df: pd.DataFrame,
    method: str = "mean"
) -> pd.DataFrame:
    """
    Resample price data to hourly resolution

    Args:
        df: DataFrame with datetime index and price column
        method: Resampling method ("mean", "first", "last", "max", "min")

    Returns:
        DataFrame with hourly prices
    """
    if method == "mean":
        return df.resample("H").mean()
    elif method == "first":
        return df.resample("H").first()
    elif method == "last":
        return df.resample("H").last()
    elif method == "max":
        return df.resample("H").max()
    elif method == "min":
        return df.resample("H").min()
    else:
        raise ValueError(f"Unknown resampling method: {method}")


def get_next_day_prices(
    df: pd.DataFrame,
    target_date: Union[str, datetime]
) -> np.ndarray:
    """
    Extract 24-hour prices for a specific date

    Args:
        df: DataFrame with datetime index and price column
        target_date: Target date (string or datetime)

    Returns:
        Array of 24 hourly prices

    Raises:
        ValueError: If data for target date is incomplete
    """
    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date)

    # Get prices for the full day
    start = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(hours=23)

    day_prices = df.loc[start:end, "price"].values

    if len(day_prices) != 24:
        raise ValueError(
            f"Expected 24 hourly prices for {target_date.date()}, "
            f"got {len(day_prices)}"
        )

    return day_prices


def create_synthetic_prices(
    base_price: float = 50.0,
    peak_hours: tuple = (8, 20),
    peak_premium: float = 30.0,
    noise_std: float = 5.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Create synthetic 24-hour price pattern

    Args:
        base_price: Base price level ($/MWh)
        peak_hours: Tuple of (start_hour, end_hour) for peak pricing
        peak_premium: Additional price during peak hours ($/MWh)
        noise_std: Standard deviation of random noise ($/MWh)
        seed: Random seed for reproducibility

    Returns:
        Array of 24 hourly prices
    """
    if seed is not None:
        np.random.seed(seed)

    prices = np.full(24, base_price)

    # Add peak premium
    start_hour, end_hour = peak_hours
    prices[start_hour:end_hour] += peak_premium

    # Add random noise
    if noise_std > 0:
        prices += np.random.normal(0, noise_std, 24)

    # Ensure non-negative prices
    prices = np.maximum(prices, 0.0)

    return prices


def calculate_price_statistics(prices: np.ndarray) -> dict:
    """
    Calculate statistics for price array

    Args:
        prices: Array of prices

    Returns:
        Dictionary with price statistics
    """
    return {
        "mean": float(np.mean(prices)),
        "std": float(np.std(prices)),
        "min": float(np.min(prices)),
        "max": float(np.max(prices)),
        "median": float(np.median(prices)),
        "range": float(np.max(prices) - np.min(prices)),
        "q25": float(np.percentile(prices, 25)),
        "q75": float(np.percentile(prices, 75))
    }


def detect_arbitrage_opportunities(
    prices: np.ndarray,
    min_spread: float = 10.0
) -> list:
    """
    Detect potential arbitrage opportunities in price series

    Args:
        prices: Array of prices
        min_spread: Minimum price spread to consider ($/MWh)

    Returns:
        List of (buy_idx, sell_idx, spread) tuples
    """
    opportunities = []

    for i in range(len(prices)):
        for j in range(i + 1, len(prices)):
            spread = prices[j] - prices[i]
            if spread >= min_spread:
                opportunities.append((i, j, spread))

    # Sort by spread (descending)
    opportunities.sort(key=lambda x: x[2], reverse=True)

    return opportunities


def format_price_schedule(
    prices: np.ndarray,
    resolution: str = "5min"
) -> pd.DataFrame:
    """
    Format price array as DataFrame with time index

    Args:
        prices: Array of prices
        resolution: Time resolution ("5min", "hourly")

    Returns:
        DataFrame with datetime index and price column
    """
    if resolution == "5min":
        # 288 intervals (24 hours * 12 per hour)
        freq = "5min"
        periods = len(prices)
    elif resolution == "hourly":
        # 24 intervals
        freq = "H"
        periods = len(prices)
    else:
        raise ValueError(f"Unknown resolution: {resolution}")

    # Create datetime index starting at midnight
    start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    time_index = pd.date_range(start=start_time, periods=periods, freq=freq)

    df = pd.DataFrame({
        "price": prices
    }, index=time_index)

    return df
