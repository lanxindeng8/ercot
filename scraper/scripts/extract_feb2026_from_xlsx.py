#!/usr/bin/env python3
"""
Extract February 2026 data from Excel file and merge with existing CSV.
"""

import pandas as pd
from pathlib import Path

# Paths
XLSX_PATH = Path(__file__).parent.parent / "data" / "feb2026_lmp_west.xlsx"
CSV_PATH = Path(__file__).parent.parent / "data" / "feb2026_dam_rtm_west.csv"

def main():
    print(f"Reading Excel file: {XLSX_PATH}")

    if not XLSX_PATH.exists():
        print(f"ERROR: Excel file not found: {XLSX_PATH}")
        return

    # Read Excel file
    df_xlsx = pd.read_excel(XLSX_PATH, sheet_name=0)
    print(f"Excel file has {len(df_xlsx)} rows")
    print(f"Columns: {list(df_xlsx.columns)}")
    print("\nFirst few rows:")
    print(df_xlsx.head(10))

    print("\nLast few rows:")
    print(df_xlsx.tail(10))

    # Show date range
    if 'oper_date' in df_xlsx.columns:
        dates = df_xlsx['oper_date'].unique()
        print(f"\nDates in Excel: {sorted(dates)}")

    print(f"\nExcel file analysis complete")

if __name__ == "__main__":
    main()
