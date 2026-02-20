#!/usr/bin/env python3
"""
Process ERCOT Fuel Mix data to extract hourly wind generation.
Converts 15-minute interval data to hourly averages.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta


def parse_time_column(col_name):
    """Parse time column like '0:15', '12:30', '0:00' to hours and minutes."""
    col_str = str(col_name)
    if ':' not in col_str:
        return None
    # Handle DST columns like '1:15 (DST)'
    if '(DST)' in col_str:
        return None  # Skip DST columns to avoid duplicates
    parts = col_str.split(':')
    try:
        hours = int(parts[0])
        minutes = int(parts[1].split()[0])  # Handle any trailing text
        return hours, minutes
    except ValueError:
        return None


def process_excel_file(file_path, months=None):
    """
    Process ERCOT fuel mix Excel file.

    Args:
        file_path: Path to Excel file
        months: List of month names to process (e.g., ['Jul', 'Aug'])
                If None, process all month sheets

    Returns:
        DataFrame with timestamp and wind_generation columns
    """
    xlsx = pd.ExcelFile(file_path)

    # Default month sheets
    all_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    if months is None:
        months = [m for m in all_months if m in xlsx.sheet_names]

    all_data = []

    for month in months:
        if month not in xlsx.sheet_names:
            print(f"  Warning: {month} sheet not found, skipping")
            continue

        print(f"  Processing {month}...")
        df = pd.read_excel(xlsx, sheet_name=month)

        # Filter to Wind fuel type
        wind_df = df[df['Fuel'] == 'Wind'].copy()

        if wind_df.empty:
            print(f"  Warning: No Wind data found in {month}")
            continue

        # Get time columns (format: '0:15', '0:30', etc.)
        time_cols = [c for c in df.columns if parse_time_column(c) is not None]

        for _, row in wind_df.iterrows():
            date = pd.to_datetime(row['Date']).date()

            for col in time_cols:
                hours, minutes = parse_time_column(col)

                # Handle midnight (0:00 is end of day, not start)
                if hours == 0 and minutes == 0:
                    # This is actually 24:00, which is midnight of next day
                    timestamp = datetime.combine(date, datetime.min.time()) + timedelta(days=1)
                else:
                    timestamp = datetime.combine(date, datetime.min.time()) + timedelta(hours=hours, minutes=minutes)

                value = row[col]
                if pd.notna(value):
                    all_data.append({
                        'timestamp': timestamp,
                        'wind_generation': float(value)
                    })

    result_df = pd.DataFrame(all_data)
    result_df = result_df.sort_values('timestamp').reset_index(drop=True)

    return result_df


def aggregate_to_hourly(df):
    """
    Aggregate 15-minute data to hourly averages.

    Args:
        df: DataFrame with timestamp and wind_generation columns

    Returns:
        DataFrame with hourly timestamps and averaged wind_generation
    """
    df = df.copy()
    df['hour'] = df['timestamp'].dt.floor('H')

    hourly = df.groupby('hour').agg({
        'wind_generation': 'mean'
    }).reset_index()

    hourly = hourly.rename(columns={'hour': 'timestamp'})

    return hourly


def main():
    # Data source: ercot-scraper historical data
    base_path = Path(__file__).parent.parent.parent / 'ercot-scraper/historical_data/raw/fuel_mix/extracted'
    # Output to ercot-scraper historical_data (symlinked from wind-generation-forecast/data/)
    output_path = Path(__file__).parent.parent.parent / 'ercot-scraper/historical_data/processed/ercot_wind.csv'

    print("Processing ERCOT Wind Generation Data")
    print("=" * 50)

    all_data = []

    # Process 2024 data (Jul-Dec for our date range)
    file_2024 = base_path / 'IntGenbyFuel2024.xlsx'
    if file_2024.exists():
        print(f"\nProcessing 2024 file...")
        df_2024 = process_excel_file(file_2024, months=['Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        all_data.append(df_2024)
        print(f"  Found {len(df_2024)} 15-min records")

    # Combine all data
    if not all_data:
        print("Error: No data processed!")
        return

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.sort_values('timestamp').reset_index(drop=True)

    print(f"\nTotal 15-min records: {len(combined)}")
    print(f"Date range: {combined['timestamp'].min()} to {combined['timestamp'].max()}")

    # Aggregate to hourly
    print("\nAggregating to hourly data...")
    hourly = aggregate_to_hourly(combined)

    print(f"Hourly records: {len(hourly)}")
    print(f"Date range: {hourly['timestamp'].min()} to {hourly['timestamp'].max()}")

    # Filter to our target range (2024-07-01 to 2025-01-22)
    # Note: 2024 file only has data through Dec 2024
    start_date = '2024-07-01'
    end_date = '2024-12-31'  # End of 2024 file

    hourly_filtered = hourly[
        (hourly['timestamp'] >= start_date) &
        (hourly['timestamp'] <= end_date)
    ].copy()

    print(f"\nFiltered to {start_date} - {end_date}: {len(hourly_filtered)} records")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hourly_filtered.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    # Show sample
    print("\nSample data:")
    print(hourly_filtered.head(10).to_string(index=False))

    # Statistics
    print("\nStatistics:")
    print(f"  Mean wind generation: {hourly_filtered['wind_generation'].mean():.1f} MW")
    print(f"  Max wind generation:  {hourly_filtered['wind_generation'].max():.1f} MW")
    print(f"  Min wind generation:  {hourly_filtered['wind_generation'].min():.1f} MW")


if __name__ == '__main__':
    main()
