#!/usr/bin/env python3
"""
Restore the original feb2026_lmp_west.xlsx format (single day, 24 hours).
"""

import pandas as pd
from pathlib import Path

# Paths
CSV_PATH = Path(__file__).parent.parent / "data" / "feb2026_dam_rtm_west.csv"
XLSX_PATH = Path(__file__).parent.parent / "data" / "feb2026_lmp_west.xlsx"

def main():
    print("Reading CSV data...")
    df = pd.read_csv(CSV_PATH)

    # Get only Feb 1, 2026 data, hours 1-24
    df_feb1 = df[
        (df['oper_date'] == '2026-02-01') &
        (df['hour_ending'] >= 1) &
        (df['hour_ending'] <= 24)
    ].copy()

    print(f"Found {len(df_feb1)} records for Feb 1")

    # Create the original format
    result = []

    for hour in range(1, 25):
        # Get data for this hour
        hour_data = df_feb1[df_feb1['hour_ending'] == hour]

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

        # Add Status column (NaN)
        row['Status'] = None

        result.append(row)

    # Create DataFrame with original column order
    excel_df = pd.DataFrame(result)
    excel_df = excel_df[['Hour', 'DAM HB_WEST', 'DAM LZ_WEST', 'RTM HB_WEST', 'RTM LZ_WEST', 'Status']]

    print(f"\nCreated {len(excel_df)} rows (original format)")

    # Write to Excel
    print(f"Writing to {XLSX_PATH}...")
    excel_df.to_excel(XLSX_PATH, sheet_name='Sheet1', index=False)

    print("Successfully restored original Excel file!")

    # Show the data
    print("\nRestored data (first 10 rows):")
    print(excel_df.head(10).to_string())

if __name__ == "__main__":
    main()
