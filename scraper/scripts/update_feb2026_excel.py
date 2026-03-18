#!/usr/bin/env python3
"""
Update feb2026_lmp_west.xlsx with data from CSV file.
Converts the CSV format to Excel format with separate columns for each settlement point.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

# Paths
CSV_PATH = Path(__file__).parent.parent / "data" / "feb2026_dam_rtm_west.csv"
XLSX_PATH = Path(__file__).parent.parent / "data" / "feb2026_lmp_west.xlsx"

def main():
    print("Reading CSV data...")
    df = pd.read_csv(CSV_PATH)

    # Filter out hour 25 (abnormal hours)
    df = df[df['hour_ending'] <= 24]

    # Filter only February 2026 data
    df = df[df['oper_date'].str.startswith('2026-02')]

    print(f"Found {len(df)} records")

    # Create a pivot structure: for each date and hour, we need DAM and RTM for both HB_WEST and LZ_WEST
    # Group by date and hour
    result = []

    for date in sorted(df['oper_date'].unique()):
        for hour in range(1, 25):
            # Get data for this date and hour
            mask = (df['oper_date'] == date) & (df['hour_ending'] == hour)
            day_hour_data = df[mask]

            if len(day_hour_data) == 0:
                continue

            # Create hour label (0:00-1:00, 1:00-2:00, etc.)
            hour_label = f"{hour-1}:00-{hour}:00"

            row = {
                'Date': date,
                'Hour': hour_label
            }

            # Extract data for each settlement point
            for _, record in day_hour_data.iterrows():
                sp = record['settlement_point']
                dam_price = record['dam_price'] if pd.notna(record['dam_price']) and record['dam_price'] != '' else None
                rtm_price = record['rtm_price'] if pd.notna(record['rtm_price']) and record['rtm_price'] != '' else None

                if sp == 'HB_WEST':
                    row['DAM HB_WEST'] = dam_price
                    row['RTM HB_WEST'] = rtm_price
                elif sp == 'LZ_WEST':
                    row['DAM LZ_WEST'] = dam_price
                    row['RTM LZ_WEST'] = rtm_price

            result.append(row)

    # Create DataFrame
    excel_df = pd.DataFrame(result)

    # Reorder columns
    excel_df = excel_df[['Date', 'Hour', 'DAM HB_WEST', 'DAM LZ_WEST', 'RTM HB_WEST', 'RTM LZ_WEST']]

    print(f"\nCreated {len(excel_df)} rows for Excel")
    print(f"Date range: {excel_df['Date'].min()} to {excel_df['Date'].max()}")

    # Write to Excel
    print(f"\nWriting to {XLSX_PATH}...")

    # Create Excel writer with some formatting
    with pd.ExcelWriter(XLSX_PATH, engine='openpyxl') as writer:
        excel_df.to_excel(writer, sheet_name='February 2026', index=False)

        # Get the worksheet
        worksheet = writer.sheets['February 2026']

        # Auto-adjust column widths
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width

    print(f"Successfully wrote Excel file!")

    # Print summary statistics
    print("\nSummary:")
    print(f"  Total rows: {len(excel_df)}")
    print(f"  Missing DAM HB_WEST: {excel_df['DAM HB_WEST'].isna().sum()}")
    print(f"  Missing DAM LZ_WEST: {excel_df['DAM LZ_WEST'].isna().sum()}")
    print(f"  Missing RTM HB_WEST: {excel_df['RTM HB_WEST'].isna().sum()}")
    print(f"  Missing RTM LZ_WEST: {excel_df['RTM LZ_WEST'].isna().sum()}")

    # Show first few rows
    print("\nFirst 5 rows:")
    print(excel_df.head().to_string())

    print("\nLast 5 rows:")
    print(excel_df.tail().to_string())

if __name__ == "__main__":
    main()
