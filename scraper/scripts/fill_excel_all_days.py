#!/usr/bin/env python3
"""
Fill feb2026_lmp_west.xlsx with all February 2026 data.
Creates one sheet per day, maintaining the original 24-hour format.
"""

import pandas as pd
from pathlib import Path

# Paths
CSV_PATH = Path(__file__).parent.parent / "data" / "feb2026_dam_rtm_west.csv"
XLSX_PATH = Path(__file__).parent.parent / "data" / "feb2026_lmp_west.xlsx"

def create_day_sheet(df_day, date_str):
    """Create a 24-row sheet for a single day in the original format"""
    result = []

    for hour in range(1, 25):
        # Get data for this hour
        hour_data = df_day[df_day['hour_ending'] == hour]

        # Create hour label (0:00-1:00, 1:00-2:00, etc.)
        hour_label = f"{hour-1}:00-{hour}:00"

        row = {'Hour': hour_label}

        # Extract data for each settlement point
        for _, record in hour_data.iterrows():
            sp = record['settlement_point']
            dam_price = record['dam_price'] if pd.notna(record['dam_price']) and record['dam_price'] != '' else None
            rtm_price = record['rtm_price'] if pd.notna(record['rtm_price']) and record['rtm_price'] != '' else None

            if sp == 'HB_WEST':
                row['DAM HB_WEST'] = dam_price
                row['RTM HB_WEST'] = rtm_price
            elif sp == 'LZ_WEST':
                row['DAM LZ_WEST'] = dam_price
                row['RTM LZ_WEST'] = rtm_price

        # Add Status column (None/NaN)
        row['Status'] = None

        result.append(row)

    # Create DataFrame with original column order
    day_df = pd.DataFrame(result)
    day_df = day_df[['Hour', 'DAM HB_WEST', 'DAM LZ_WEST', 'RTM HB_WEST', 'RTM LZ_WEST', 'Status']]

    return day_df

def main():
    print("Reading CSV data...")
    df = pd.read_csv(CSV_PATH)

    # Filter out hour 25 and only February 2026
    df = df[
        (df['hour_ending'] >= 1) &
        (df['hour_ending'] <= 24) &
        (df['oper_date'].str.startswith('2026-02'))
    ].copy()

    print(f"Found {len(df)} records for February 2026")

    # Get all unique dates
    dates = sorted(df['oper_date'].unique())
    print(f"Dates: {len(dates)} days from {dates[0]} to {dates[-1]}")

    # Create Excel writer
    print(f"\nWriting to {XLSX_PATH}...")
    with pd.ExcelWriter(XLSX_PATH, engine='openpyxl') as writer:
        for date in dates:
            # Get data for this date
            df_day = df[df['oper_date'] == date]

            # Create sheet for this day
            day_df = create_day_sheet(df_day, date)

            # Sheet name: use format like "Feb 01", "Feb 02", etc.
            day = int(date.split('-')[2])
            sheet_name = f"Feb {day:02d}"

            # Write to Excel
            day_df.to_excel(writer, sheet_name=sheet_name, index=False)

            print(f"  Created sheet: {sheet_name}")

    print(f"\nSuccessfully created Excel file with {len(dates)} sheets!")

    # Print summary
    print("\nSummary:")
    print(f"  Total sheets: {len(dates)}")
    print(f"  Date range: {dates[0]} to {dates[-1]}")
    print(f"  Each sheet: 24 rows (0:00-1:00 to 23:00-24:00)")
    print(f"  Columns: Hour, DAM HB_WEST, DAM LZ_WEST, RTM HB_WEST, RTM LZ_WEST, Status")

if __name__ == "__main__":
    main()
