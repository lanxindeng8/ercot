"""
Feature Engineering Example Script

Demonstrates how to use the feature computation and label generation modules
"""

import sys
sys.path.append('..')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.data.feature_engineering import FeatureEngineer
from src.utils.labels import LabelGenerator


def create_sample_data(n_days=7):
    """Create sample data for testing

    Args:
        n_days: Number of days

    Returns:
        Sample DataFrame
    """
    # Generate time index (5-minute intervals)
    start_time = datetime(2025, 12, 10, 0, 0)
    periods = n_days * 24 * 12  # 5-minute intervals
    timestamps = pd.date_range(start=start_time, periods=periods, freq='5min')

    # Create sample data
    np.random.seed(42)

    # Base price (simulating intraday pattern)
    hour = timestamps.hour + timestamps.minute / 60
    base_price = 50 + 30 * np.sin((hour - 6) * np.pi / 12)  # Intraday fluctuation

    # Add random noise
    noise = np.random.normal(0, 10, len(timestamps))

    # Simulate spike events (price surges during certain periods)
    spike_mask = (timestamps.day == 14) & (timestamps.hour >= 20) & (timestamps.hour <= 22)
    spike_boost = np.where(spike_mask, 500, 0)

    df = pd.DataFrame({
        # Price data
        'P_CPS': base_price + noise + spike_boost + np.random.normal(20, 5, len(timestamps)),
        'P_West': base_price + noise + spike_boost * 0.7 + np.random.normal(15, 5, len(timestamps)),
        'P_Houston': base_price + noise + spike_boost * 0.3 + np.random.normal(0, 5, len(timestamps)),
        'P_Hub': base_price + noise,

        # Day-ahead prices
        'P_CPS_DA': base_price + np.random.normal(0, 5, len(timestamps)),
        'P_West_DA': base_price + np.random.normal(0, 5, len(timestamps)),
        'P_Houston_DA': base_price + np.random.normal(0, 5, len(timestamps)),

        # System data
        'Load': 40000 + 10000 * np.sin((hour - 12) * np.pi / 12) + np.random.normal(0, 500, len(timestamps)),
        'Wind': 8000 + 3000 * np.sin(hour * np.pi / 24) + np.random.normal(0, 500, len(timestamps)),
        'Solar': np.maximum(0, 6000 * np.sin((hour - 6) * np.pi / 12)) + np.random.normal(0, 300, len(timestamps)),
        'Gas': 25000 + 5000 * np.sin((hour - 14) * np.pi / 12) + np.random.normal(0, 300, len(timestamps)),
        'Coal': 8000 + np.random.normal(0, 200, len(timestamps)),
        'ESR': np.random.normal(0, 1000, len(timestamps)),  # ESR net output

        # Weather data
        'T_CPS': 65 + 15 * np.sin((hour - 15) * np.pi / 12) + np.random.normal(0, 2, len(timestamps)),
        'T_West': 63 + 16 * np.sin((hour - 15) * np.pi / 12) + np.random.normal(0, 2, len(timestamps)),
        'T_Houston': 68 + 12 * np.sin((hour - 15) * np.pi / 12) + np.random.normal(0, 2, len(timestamps)),

        'WindSpeed_CPS': 10 + 5 * np.random.random(len(timestamps)),
        'WindSpeed_West': 12 + 5 * np.random.random(len(timestamps)),
        'WindSpeed_Houston': 8 + 4 * np.random.random(len(timestamps)),

        'WindDir_CPS': np.random.uniform(0, 360, len(timestamps)),
        'WindDir_West': np.random.uniform(0, 360, len(timestamps)),
        'WindDir_Houston': np.random.uniform(0, 360, len(timestamps)),
    }, index=timestamps)

    # Simulate cold front event (12/14)
    cold_front_mask = (timestamps.day == 14) & (timestamps.hour >= 18)
    df.loc[cold_front_mask, 'T_CPS'] -= 10
    df.loc[cold_front_mask, 'WindDir_CPS'] = 0  # North wind

    return df


def main():
    """Main function"""
    print("=" * 80)
    print("ERCOT RTM LMP Spike Prediction - Feature Engineering Example")
    print("=" * 80)

    # 1. Create sample data
    print("\nStep 1: Creating sample data...")
    df = create_sample_data(n_days=7)
    print(f"Data dimensions: {df.shape}")
    print(f"Time range: {df.index[0]} to {df.index[-1]}")
    print(f"\nRaw data columns:\n{df.columns.tolist()}")

    # 2. Compute features
    print("\n" + "=" * 80)
    print("Step 2: Computing features...")
    print("=" * 80)

    feature_engineer = FeatureEngineer(
        zones=['CPS', 'West', 'Houston'],
        lookback_days=30
    )

    df_with_features = feature_engineer.calculate_all_features(df)
    print(f"\nData dimensions after adding features: {df_with_features.shape}")

    # Display feature names
    price_features = feature_engineer.get_feature_names('price')
    supply_demand_features = feature_engineer.get_feature_names('supply_demand')
    weather_features = feature_engineer.get_feature_names('weather')
    temporal_features = feature_engineer.get_feature_names('temporal')

    print(f"\nPrice structure features ({len(price_features)}):")
    print(f"  {price_features[:5]}... (showing first 5)")

    print(f"\nSupply-demand balance features ({len(supply_demand_features)}):")
    print(f"  {supply_demand_features[:5]}... (showing first 5)")

    print(f"\nWeather-driven features ({len(weather_features)}):")
    print(f"  {weather_features[:5]}... (showing first 5)")

    print(f"\nTemporal features ({len(temporal_features)}):")
    print(f"  {temporal_features}")

    # 3. Generate labels
    print("\n" + "=" * 80)
    print("Step 3: Generating labels...")
    print("=" * 80)

    label_generator = LabelGenerator(
        zones=['CPS', 'West', 'Houston'],
        P_hi=400,      # Spike price threshold
        S_hi=50,       # Spike spread threshold
        S_cross_hi=80, # Cross-zone spread threshold
        P_mid=150,     # Tight price threshold
        S_mid=20,      # Tight spread threshold
        m=3,           # Duration threshold (3 x 5-minute = 15 minutes)
        H=60,          # Lead Spike warning window (60 minutes)
        dt=5           # Data time interval
    )

    labels = label_generator.generate_all_labels(df_with_features)
    print(f"\nLabel dimensions: {labels.shape}")
    print(f"Label columns: {labels.columns.tolist()}")

    # 4. Merge data
    print("\n" + "=" * 80)
    print("Step 4: Merging features and labels...")
    print("=" * 80)

    final_df = pd.concat([df_with_features, labels], axis=1)
    print(f"\nFinal data dimensions: {final_df.shape}")
    print(f"Total columns: {len(final_df.columns)}")

    # 5. Identify Spike events
    print("\n" + "=" * 80)
    print("Step 5: Identifying independent Spike events...")
    print("=" * 80)

    for zone in ['CPS', 'West', 'Houston']:
        spike_col = f'SpikeEvent_{zone}'
        if spike_col in labels.columns:
            events = label_generator.identify_spike_events(labels[spike_col])
            print(f"\n{zone} zone: found {len(events)} independent Spike event(s):")
            for i, event in enumerate(events, 1):
                print(f"  Event {i}:")
                print(f"    Start: {event['start']}")
                print(f"    End: {event['end']}")
                print(f"    Duration: {event['duration']} time steps ({event['duration'] * 5} minutes)")

    # 6. Data quality check
    print("\n" + "=" * 80)
    print("Step 6: Data quality check...")
    print("=" * 80)

    # Check for missing values
    missing = final_df.isnull().sum()
    if missing.sum() > 0:
        print(f"\nMissing values found:")
        print(missing[missing > 0])
    else:
        print("\nNo missing values")

    # Check for infinite values
    inf_count = np.isinf(final_df.select_dtypes(include=[np.number])).sum()
    if inf_count.sum() > 0:
        print(f"\nInfinite values found:")
        print(inf_count[inf_count > 0])
    else:
        print("No infinite values")

    # 7. Save sample data
    print("\n" + "=" * 80)
    print("Step 7: Saving sample data...")
    print("=" * 80)

    output_path = '../data/processed/sample_features_labels.csv'
    final_df.to_csv(output_path)
    print(f"Data saved to: {output_path}")

    # 8. Display data during key moments (Spike period)
    print("\n" + "=" * 80)
    print("Step 8: Viewing data during Spike period...")
    print("=" * 80)

    spike_mask = labels['SpikeEvent_CPS'] == 1
    if spike_mask.any():
        print(f"\nSpike period data sample (first 5 rows):")
        spike_data = final_df[spike_mask].head()
        display_cols = ['P_CPS', 'P_Hub', 'spread_CPS_hub', 'net_load',
                       'wind_anomaly', 'gas_saturation', 'T_anomaly_CPS',
                       'Regime_CPS']
        print(spike_data[display_cols])

    print("\n" + "=" * 80)
    print("Feature engineering example complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Replace sample data with real ERCOT data")
    print("2. Perform feature analysis and visualization")
    print("3. Train prediction models")
    print("=" * 80)


if __name__ == '__main__':
    main()
