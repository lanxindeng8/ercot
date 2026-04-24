#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RTM-DAM Delta Feature Engineering Script (40-hour Prediction)
==============================================================
Create features for day-ahead arbitrage strategy

Prediction Scenario:
    - Prediction time: Day D-1, 10:00 AM
    - Target time: Day D, 00:00 ~ 23:00 (24 hours)
    - Prediction lead time: 14~38 hours

Feature Design Principles:
    - Only use historical data available before the prediction time
    - DAM price is known at prediction time (DAM clears on D-1)
    - Time features for the target hour are known

Usage:
    python delta_feature_extraction.py --input ../data/spread_data.csv --output ../data/train_features.csv
"""

import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def create_training_samples(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create training samples

    Each sample represents:
    - At 10:00 AM on Day D-1 (or latest available time)
    - Predicting the spread for a specific hour on Day D

    Features:
    - Historical spread/RTM/DAM statistics (up to the latest data on D-1)
    - DAM price for the target hour (known)
    - Time features for the target hour
    """
    print("Creating training samples...")

    # Group by date
    df['date'] = df['timestamp'].dt.date

    # Get unique dates
    dates = sorted(df['date'].unique())

    samples = []

    for i in tqdm(range(1, len(dates)), desc="Processing daily data"):
        # Day D-1 (historical data day)
        hist_date = dates[i-1]
        # Day D (target prediction day)
        target_date = dates[i]

        # Get D-1 data (for computing historical features)
        hist_data = df[df['date'] <= hist_date].copy()
        if len(hist_data) < 168:  # Need at least 7 days of history
            continue

        # Get Day D data (for extracting targets)
        target_data = df[df['date'] == target_date].copy()
        if len(target_data) == 0:
            continue

        # Historical statistical features (based on all data up to and including D-1)
        recent_168h = hist_data.tail(168)  # Most recent 7 days
        recent_24h = hist_data.tail(24)    # Most recent 1 day

        hist_features = {
            # Recent spread statistics
            'spread_mean_7d': recent_168h['spread'].mean(),
            'spread_std_7d': recent_168h['spread'].std(),
            'spread_max_7d': recent_168h['spread'].max(),
            'spread_min_7d': recent_168h['spread'].min(),
            'spread_median_7d': recent_168h['spread'].median(),

            'spread_mean_24h': recent_24h['spread'].mean(),
            'spread_std_24h': recent_24h['spread'].std(),

            # Recent RTM statistics
            'rtm_mean_7d': recent_168h['rtm_mean'].mean(),
            'rtm_std_7d': recent_168h['rtm_mean'].std(),
            'rtm_max_7d': recent_168h['rtm_max'].max(),

            'rtm_mean_24h': recent_24h['rtm_mean'].mean(),
            'rtm_volatility_24h': recent_24h['rtm_std'].mean(),

            # Recent DAM statistics
            'dam_mean_7d': recent_168h['dam_price'].mean(),
            'dam_std_7d': recent_168h['dam_price'].std(),
            'dam_mean_24h': recent_24h['dam_price'].mean(),

            # Spread direction statistics
            'spread_positive_ratio_7d': (recent_168h['spread'] > 0).mean(),
            'spread_positive_ratio_24h': (recent_24h['spread'] > 0).mean(),

            # Spike counts
            'spike_count_7d': (recent_168h['spread'].abs() > 20).sum(),
            'rtm_spike_count_7d': (recent_168h['rtm_mean'] > 100).sum(),

            # Trend features
            'spread_trend_7d': recent_168h['spread'].iloc[-24:].mean() - recent_168h['spread'].iloc[:24].mean(),
        }

        # Hourly grouped historical means (for same-hour prediction)
        hourly_hist = hist_data.groupby('hour').agg({
            'spread': ['mean', 'std'],
            'rtm_mean': 'mean',
            'dam_price': 'mean'
        })
        hourly_hist.columns = ['spread_by_hour_mean', 'spread_by_hour_std', 'rtm_by_hour_mean', 'dam_by_hour_mean']

        # Group by hour + day of week
        hist_data['dow'] = pd.to_datetime(hist_data['date']).dt.dayofweek
        dow_hour_hist = hist_data.groupby(['dow', 'hour'])['spread'].mean().to_dict()

        # Create a sample for each hour of Day D
        target_dow = pd.to_datetime(target_date).dayofweek
        target_month = pd.to_datetime(target_date).month
        target_day_of_month = pd.to_datetime(target_date).day
        target_week = pd.to_datetime(target_date).isocalendar().week

        for _, row in target_data.iterrows():
            target_hour = row['hour']

            sample = {
                # Time ID
                'timestamp': row['timestamp'],
                'date': target_date,

                # Target variables
                'target_spread': row['spread'],
                'target_spread_last': row['spread_last'],
                'target_direction': row['spread_direction'],
                'target_class': row['spread_class'],

                # Target hour DAM price (known)
                'target_dam_price': row['dam_price'],

                # Target hour time features
                'target_hour': target_hour,
                'target_dow': target_dow,
                'target_month': target_month,
                'target_day_of_month': target_day_of_month,
                'target_week': target_week,
                'target_is_weekend': int(target_dow >= 5),
                'target_is_peak': int((6 <= target_hour <= 10) or (17 <= target_hour <= 21)),
                'target_is_summer': int(target_month in [6, 7, 8, 9]),

                # Historical statistical features
                **hist_features,

                # Same-hour historical features
                'spread_same_hour_hist': hourly_hist.loc[target_hour, 'spread_by_hour_mean'] if target_hour in hourly_hist.index else np.nan,
                'spread_same_hour_std': hourly_hist.loc[target_hour, 'spread_by_hour_std'] if target_hour in hourly_hist.index else np.nan,
                'rtm_same_hour_hist': hourly_hist.loc[target_hour, 'rtm_by_hour_mean'] if target_hour in hourly_hist.index else np.nan,

                # Same hour + same day-of-week historical
                'spread_same_dow_hour': dow_hour_hist.get((target_dow, target_hour), np.nan),

                # DAM relative features
                'dam_vs_7d_mean': row['dam_price'] - hist_features['dam_mean_7d'],
                'dam_percentile_7d': (recent_168h['dam_price'] < row['dam_price']).mean(),
            }

            # Actual spread from the same hour last week (if available)
            same_hour_last_week = hist_data[(hist_data['hour'] == target_hour) &
                                            (hist_data['dow'] == target_dow)].tail(1)
            if len(same_hour_last_week) > 0:
                sample['spread_same_dow_hour_last'] = same_hour_last_week['spread'].values[0]
            else:
                sample['spread_same_dow_hour_last'] = np.nan

            samples.append(sample)

    result_df = pd.DataFrame(samples)
    return result_df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features"""
    # DAM price level
    df['dam_price_level'] = pd.cut(df['target_dam_price'],
                                    bins=[0, 20, 40, 60, 100, 200, np.inf],
                                    labels=[0, 1, 2, 3, 4, 5]).astype(float)

    # Cyclical time encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['target_hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['target_hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['target_dow'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['target_dow'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['target_month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['target_month'] / 12)

    # Prediction error baseline (naive prediction: use historical same-hour mean)
    df['naive_pred'] = df['spread_same_hour_hist']

    # Relative features
    df['dam_vs_same_hour'] = df['target_dam_price'] - df['rtm_same_hour_hist'].fillna(df['rtm_mean_7d'])

    return df


def select_features(df: pd.DataFrame) -> tuple:
    """
    Select feature columns

    Returns:
    - feature_cols: list of feature column names
    - target_cols: dictionary of target column names
    """
    # Exclude non-feature columns
    exclude_cols = [
        'timestamp', 'date',
        'target_spread', 'target_spread_last', 'target_direction', 'target_class',
        'naive_pred'
    ]

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    target_cols = {
        'regression': 'target_spread',
        'binary': 'target_direction',
        'multiclass': 'target_class'
    }

    return feature_cols, target_cols


def extract_delta_features(
    input_path: str,
    output_path: str
) -> pd.DataFrame:
    """
    Main function: Extract Delta prediction features
    """
    print("=" * 60)
    print("RTM-DAM Delta Feature Engineering (40-hour Prediction)")
    print("=" * 60)

    # 1. Load data
    print("\n1. Loading Spread data...")
    df = pd.read_csv(input_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"   Raw sample count: {len(df):,}")

    # 2. Create training samples
    print("\n2. Creating training samples...")
    samples_df = create_training_samples(df)
    print(f"   Training sample count: {len(samples_df):,}")

    # 3. Add derived features
    print("\n3. Adding derived features...")
    samples_df = add_derived_features(samples_df)

    # 4. Handle missing values
    print("\n4. Handling missing values...")
    samples_df = samples_df.dropna()
    print(f"   Valid sample count: {len(samples_df):,}")

    # 5. Save
    print("\n5. Saving feature data...")
    samples_df.to_csv(output_path, index=False)
    print(f"   Saved to: {output_path}")

    # Print feature information
    feature_cols, target_cols = select_features(samples_df)
    print(f"\nFeature count: {len(feature_cols)}")
    print(f"Feature list:")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i:2d}. {col}")

    # Print target variable statistics
    print("\nTarget variable statistics:")
    print(f"  Regression target (spread): mean={samples_df['target_spread'].mean():.2f}, std={samples_df['target_spread'].std():.2f}")
    print(f"  Binary target: RTM>DAM={samples_df['target_direction'].mean()*100:.1f}%")
    print(f"  Multi-class target distribution:")
    for c in range(5):
        pct = (samples_df['target_class'] == c).mean() * 100
        print(f"    Class {c}: {pct:.1f}%")

    return samples_df


def main():
    parser = argparse.ArgumentParser(description='Extract features for Delta prediction')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input spread data CSV')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output feature CSV')
    args = parser.parse_args()

    extract_delta_features(
        input_path=args.input,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
