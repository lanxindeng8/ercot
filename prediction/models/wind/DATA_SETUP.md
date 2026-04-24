# Data Setup Guide

## Overview

This project requires two types of data:
1. **HRRR Weather Data** (2GB, not in the git repository)
2. **ERCOT Wind Generation Data** (needs to be provided by the user)

---

## 1. HRRR Weather Data

### Data Description
- **Source**: NOAA HRRR (High-Resolution Rapid Refresh)
- **Resolution**: 3km
- **Region**: Texas
- **Variables**: u10m, v10m, u80m, v80m, t2m, sp
- **Time Range**: 2024-07-01 to 2025-01-22
- **Size**: ~2GB (130 zarr files)

### Download Steps

Activate the virtual environment and install dependencies:
```bash
cd /path/to/wind-generation-forecast
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Install earth2studio (if needed)
# PIP_CONFIG_FILE=/dev/null pip install -e /path/to/earth2studio
```

Download HRRR data (example):
```bash
# Download all initialization times for a single date
python scripts/fetch_hrrr_data.py --date 2024-07-01 --hours 0 6 12 18

# Batch download (recommended to run as background tasks)
for date in 2024-07-01 2024-07-08 2024-07-15; do
    python scripts/fetch_hrrr_data.py --date $date --hours 0 6 12 18 &
done
```

### Data File Structure
```
data/hrrr/
├── hrrr_20240701_00.zarr/
├── hrrr_20240701_06.zarr/
├── hrrr_20240701_12.zarr/
├── hrrr_20240701_18.zarr/
├── ...
└── hrrr_20250122_18.zarr/
```

Each zarr file contains:
- Shape: (1, 7, 6, 424, 445)
  - 1 initialization time
  - 7 forecast lead times (0, 1, 2, 3, 6, 9, 12 hours)
  - 6 meteorological variables
  - 424x445 grid points covering Texas

### List of Downloaded Dates
Refer to the currently downloaded dates in the repository:
- 2024-07: 01, 08, 15, 22, 29 (4 initialization times per day)
- 2024-08: 01, 08, 15, 22, 29
- 2024-09: 01, 08, 15, 22, 29
- 2024-10: 01, 08, 15, 22, 29
- 2024-11: 01, 08, 15, 22, 29
- 2024-12: 01, 08, 15, 22, 28, 29
- 2025-01: 01, 05, 10, 15, 18, 19, 20, 21, 22

---

## 2. ERCOT Wind Generation Data

### Data Requirements

**Format**: CSV or Parquet
**Required Fields**:
- `timestamp` or `valid_time`: datetime format
- `wind_generation`: wind power generation (MW)

**Time Requirements**:
- Time range: 2024-07-01 to 2025-01-22
- Frequency: hourly data
- Timezone: UTC or clearly labeled

### Example Data Format

CSV:
```csv
timestamp,wind_generation
2024-07-01 00:00:00,15234.5
2024-07-01 01:00:00,16012.3
2024-07-01 02:00:00,14890.1
...
```

Or Parquet format stored at `data/ercot/wind_generation.parquet`

### Data Sources
- ERCOT official website: http://www.ercot.com/gridinfo/generation
- Specific page: Look for "Wind Power Production" or "Actual System Load by Fuel Type"

---

## 3. Data Alignment and Feature Construction

After obtaining both types of data, run the feature construction script:

```bash
python scripts/build_features.py
```

This script will:
1. Read HRRR weather data
2. Read ERCOT wind power data
3. Align timestamps
4. Compute wind speed and wind power features
5. Compute ramp features (rate of change, acceleration, etc.)
6. Compute temporal features (hour, day of week, whether nighttime, etc.)
7. Save to `data/processed/features.parquet`

---

## 4. Train Models

```bash
# Train with LightGBM
python scripts/train_models.py --model gbm

# Train with LSTM (requires more data)
python scripts/train_models.py --model lstm

# Train with ensemble (combined models)
python scripts/train_models.py --model ensemble
```

---

## 5. FAQ

### Q: What if downloading HRRR data is too slow?
A: Use parallel downloads, downloading 4-5 dates at a time. The NOAA server may throttle speeds, so be patient.

### Q: How to verify HRRR data integrity?
A: Run the following command to check:
```bash
ls data/hrrr/*.zarr | wc -l  # Should have ~130 files
```

### Q: What if the ERCOT data timezone is inconsistent?
A: Convert everything to UTC:
```python
import pandas as pd
df = pd.read_csv('ercot_data.csv')
df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize('America/Chicago').dt.tz_convert('UTC')
```

### Q: What if there is insufficient training data?
A: You can download more historical HRRR data (years 2023-2024). Refer to `scripts/fetch_hrrr_data.py` and modify the date range.

---

## Contact

If you have questions, please submit a GitHub Issue or contact the project maintainers.
