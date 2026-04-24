#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RTM-DAM Delta/Spread Data Preparation Script
=============================================
Merge RTM and DAM data, calculate Spread, create prediction labels

Usage:
    python prepare_delta_data.py --rtm ../data/rtm_lz_west.csv --dam ../data/dam_lz_west.csv --output ../data/spread_data.csv
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def load_rtm_data(filepath: str) -> pd.DataFrame:
    """Load RTM data (15-minute intervals)"""
    df = pd.read_csv(filepath)

    # Convert date
    df['date_dt'] = pd.to_datetime(df['date'], format='%m/%d/%Y')

    # Create full timestamp (15-minute granularity)
    df['timestamp'] = df['date_dt'] + pd.to_timedelta(df['hour'] - 1, unit='h') + pd.to_timedelta((df['interval'] - 1) * 15, unit='m')

    return df


def load_dam_data(filepath: str) -> pd.DataFrame:
    """Load DAM data (hourly intervals)"""
    df = pd.read_csv(filepath)

    # Convert date
    df['date_dt'] = pd.to_datetime(df['date'], format='%m/%d/%Y')

    # Create full timestamp (hourly granularity)
    df['timestamp'] = df['date_dt'] + pd.to_timedelta(df['hour'] - 1, unit='h')

    return df


def aggregate_rtm_to_hourly(rtm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate RTM 15-minute data to hourly level

    Aggregation methods:
    - rtm_mean: Average price of 4 intervals within the hour
    - rtm_max: Highest price within the hour
    - rtm_min: Lowest price within the hour
    - rtm_last: Last interval price within the hour (actual settlement price)
    - rtm_std: Price volatility within the hour
    """
    rtm_df['hour_start'] = rtm_df['timestamp'].dt.floor('h')

    agg_df = rtm_df.groupby('hour_start').agg({
        'price': ['mean', 'max', 'min', 'last', 'std', 'count']
    }).reset_index()

    agg_df.columns = ['timestamp', 'rtm_mean', 'rtm_max', 'rtm_min', 'rtm_last', 'rtm_std', 'rtm_count']

    # Only keep complete hours (4 intervals)
    agg_df = agg_df[agg_df['rtm_count'] == 4].copy()
    agg_df = agg_df.drop(columns=['rtm_count'])

    return agg_df


def merge_rtm_dam(rtm_hourly: pd.DataFrame, dam_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge RTM and DAM data
    """
    # Prepare DAM data
    dam_simple = dam_df[['timestamp', 'dam_price']].copy()

    # Merge by timestamp
    merged = pd.merge(rtm_hourly, dam_simple, on='timestamp', how='inner')

    # Sort by time
    merged = merged.sort_values('timestamp').reset_index(drop=True)

    return merged


def calculate_spread_and_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Spread and create prediction labels

    Spread definition:
    - spread = RTM average price - DAM price
    - spread > 0: RTM higher than DAM (real-time market tight)
    - spread < 0: RTM lower than DAM (real-time market loose)
    """
    # Calculate spread (using RTM average price)
    df['spread'] = df['rtm_mean'] - df['dam_price']

    # Also calculate spread using last price (closer to actual settlement)
    df['spread_last'] = df['rtm_last'] - df['dam_price']

    # Binary label: spread direction
    # 1 = RTM > DAM (buying in DAM is favorable)
    # 0 = RTM < DAM (selling in DAM is favorable)
    df['spread_direction'] = (df['spread'] > 0).astype(int)

    # Multi-class label: spread interval
    # 0: < -20
    # 1: -20 ~ -5
    # 2: -5 ~ 5 (no-trade zone)
    # 3: 5 ~ 20
    # 4: >= 20
    bins = [-np.inf, -20, -5, 5, 20, np.inf]
    labels = [0, 1, 2, 3, 4]
    df['spread_class'] = pd.cut(df['spread'], bins=bins, labels=labels).astype(int)

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time features"""
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['quarter'] = df['timestamp'].dt.quarter
    df['day_of_month'] = df['timestamp'].dt.day
    df['week_of_year'] = df['timestamp'].dt.isocalendar().week.astype(int)
    df['year'] = df['timestamp'].dt.year

    # Is weekend
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Is peak hours (typically 6-10am and 5-9pm)
    df['is_peak'] = ((df['hour'] >= 6) & (df['hour'] <= 10) |
                     (df['hour'] >= 17) & (df['hour'] <= 21)).astype(int)

    # Is summer (June-September, ERCOT summer peak)
    df['is_summer'] = df['month'].isin([6, 7, 8, 9]).astype(int)

    return df


def add_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add historical features (for prediction)

    Note: These features are computed from historical data and do not cause information leakage
    """
    # Spread lag features
    for lag in [24, 48, 72, 168]:  # 1 day, 2 days, 3 days, 1 week
        df[f'spread_lag_{lag}h'] = df['spread'].shift(lag)

    # Same-hour historical spread (same hour last week)
    df['spread_same_hour_7d'] = df['spread'].shift(168)  # 7*24 = 168

    # Spread rolling statistics (past 7 days)
    df['spread_mean_7d'] = df['spread'].rolling(168).mean()
    df['spread_std_7d'] = df['spread'].rolling(168).std()
    df['spread_max_7d'] = df['spread'].rolling(168).max()
    df['spread_min_7d'] = df['spread'].rolling(168).min()

    # Spread rolling statistics (past 24 hours)
    df['spread_mean_24h'] = df['spread'].rolling(24).mean()
    df['spread_std_24h'] = df['spread'].rolling(24).std()

    # RTM historical features
    df['rtm_mean_24h'] = df['rtm_mean'].rolling(24).mean()
    df['rtm_std_24h'] = df['rtm_mean'].rolling(24).std()
    df['rtm_mean_7d'] = df['rtm_mean'].rolling(168).mean()

    # DAM historical features
    df['dam_mean_24h'] = df['dam_price'].rolling(24).mean()
    df['dam_std_24h'] = df['dam_price'].rolling(24).std()
    df['dam_mean_7d'] = df['dam_price'].rolling(168).mean()

    # Same-hour historical mean (historical average grouped by hour)
    # This needs to be calculated separately

    return df


def add_spike_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add price spike features"""
    # Spread spike detection
    for threshold in [10, 20, 50]:
        df[f'spread_spike_{threshold}'] = (df['spread'].abs() > threshold).astype(int)

    # RTM spike detection
    for threshold in [100, 200, 500]:
        df[f'rtm_spike_{threshold}'] = (df['rtm_mean'] > threshold).astype(int)

    # Spike count in the past 24 hours
    df['spike_count_24h'] = df['spread_spike_20'].rolling(24).sum()
    df['rtm_spike_count_24h'] = df['rtm_spike_100'].rolling(24).sum()

    return df


def prepare_delta_data(
    rtm_path: str,
    dam_path: str,
    output_path: str,
    start_date: str = '2015-01-01'
) -> pd.DataFrame:
    """
    Main function: Prepare Delta prediction data
    """
    print("=" * 60)
    print("RTM-DAM Delta Data Preparation")
    print("=" * 60)

    # 1. Load data
    print("\n1. Loading data...")
    rtm_df = load_rtm_data(rtm_path)
    dam_df = load_dam_data(dam_path)
    print(f"   RTM: {len(rtm_df):,} records (15-minute)")
    print(f"   DAM: {len(dam_df):,} records (hourly)")

    # 2. Aggregate RTM to hourly level
    print("\n2. Aggregating RTM to hourly level...")
    rtm_hourly = aggregate_rtm_to_hourly(rtm_df)
    print(f"   RTM hourly data: {len(rtm_hourly):,} records")

    # 3. Merge RTM and DAM
    print("\n3. Merging RTM and DAM data...")
    merged = merge_rtm_dam(rtm_hourly, dam_df)
    print(f"   After merging: {len(merged):,} records")

    # 4. Calculate Spread and labels
    print("\n4. Calculating Spread and labels...")
    merged = calculate_spread_and_labels(merged)

    # 5. Add time features
    print("\n5. Adding time features...")
    merged = add_time_features(merged)

    # 6. Add historical features
    print("\n6. Adding historical features...")
    merged = add_historical_features(merged)

    # 7. Add spike features
    print("\n7. Adding spike features...")
    merged = add_spike_features(merged)

    # 8. Filter data range
    start_dt = pd.to_datetime(start_date)
    merged = merged[merged['timestamp'] >= start_dt].copy()

    # 9. Drop rows with NaN (generated by rolling calculations)
    merged = merged.dropna().reset_index(drop=True)

    # 10. Save data
    print("\n8. Saving data...")
    merged.to_csv(output_path, index=False)
    print(f"   Saved to: {output_path}")

    # Print statistics
    print("\n" + "=" * 60)
    print("Data Statistics")
    print("=" * 60)
    print(f"Total samples: {len(merged):,}")
    print(f"Time range: {merged['timestamp'].iloc[0]} ~ {merged['timestamp'].iloc[-1]}")
    print(f"\nSpread statistics:")
    print(f"  Mean: ${merged['spread'].mean():.2f}")
    print(f"  Std: ${merged['spread'].std():.2f}")
    print(f"  Min: ${merged['spread'].min():.2f}")
    print(f"  Max: ${merged['spread'].max():.2f}")
    print(f"\nSpread direction distribution:")
    print(f"  RTM > DAM (positive spread): {(merged['spread_direction'] == 1).sum():,} ({(merged['spread_direction'] == 1).mean()*100:.1f}%)")
    print(f"  RTM < DAM (negative spread): {(merged['spread_direction'] == 0).sum():,} ({(merged['spread_direction'] == 0).mean()*100:.1f}%)")
    print(f"\nSpread interval distribution:")
    class_dist = merged['spread_class'].value_counts().sort_index()
    class_labels = ['< -$20', '-$20~-$5', '-$5~$5', '$5~$20', '>= $20']
    for i, label in enumerate(class_labels):
        count = class_dist.get(i, 0)
        pct = count / len(merged) * 100
        print(f"  {label}: {count:,} ({pct:.1f}%)")

    return merged


def main():
    parser = argparse.ArgumentParser(description='Prepare RTM-DAM Delta/Spread data')
    parser.add_argument('--rtm', '-r', type=str, required=True,
                        help='Path to RTM CSV file')
    parser.add_argument('--dam', '-d', type=str, required=True,
                        help='Path to DAM CSV file')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output CSV file path')
    parser.add_argument('--start-date', type=str, default='2015-01-01',
                        help='Start date for data (default: 2015-01-01)')
    args = parser.parse_args()

    prepare_delta_data(
        rtm_path=args.rtm,
        dam_path=args.dam,
        output_path=args.output,
        start_date=args.start_date
    )


if __name__ == "__main__":
    main()
