# ERCOT RTM LMP Spike Forecasting

ERCOT Real-Time Market LMP Spike Prediction Algorithm and Energy Storage Strategy Optimization System

## Project Overview

This project aims to predict regional LMP Spike events in the ERCOT Real-Time Market (RTM) and optimize Battery Energy Storage System (BESS) dispatch strategies to avoid "premature discharge" errors and maximize revenue during extreme price windows.

**Core Capabilities**:
- Zone-level Spike event prediction (60-90 minutes in advance)
- System state recognition (Normal -> Tight -> Scarcity)
- Hold-SOC strategy optimization
- Estimated revenue improvement: 30-60%

## Project Structure

```
spike-forecast/
├── data/
│   ├── raw/              # Raw data
│   ├── processed/        # Processed data
│   └── features/         # Feature data
├── src/
│   ├── data/
│   │   ├── feature_engineering.py  # Feature computation module
│   │   ├── ingestion.py            # Data ingestion (to be implemented)
│   │   └── preprocessing.py        # Preprocessing (to be implemented)
│   ├── models/
│   │   ├── baseline.py             # XGBoost/LightGBM (to be implemented)
│   │   └── cfc_model.py            # CfC/LTC (to be implemented)
│   ├── strategy/
│   │   ├── rules.py                # Hard rules (to be implemented)
│   │   ├── mpc.py                  # MPC optimization (to be implemented)
│   │   └── backtest.py             # Backtesting framework (to be implemented)
│   └── utils/
│       ├── labels.py               # Label generation module
│       └── metrics.py              # Evaluation metrics (to be implemented)
├── notebooks/
│   └── 01_feature_engineering_example.py  # Feature engineering example
├── configs/              # Configuration files
├── tests/                # Tests
├── docs/
│   ├── DESIGN.md         # Algorithm design document
│   └── ...               # Reference documents
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/powerA-ai/spike-forecast.git
cd spike-forecast

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Feature Engineering Example

```bash
cd notebooks
python 01_feature_engineering_example.py
```

This will:
- Create sample data (simulating 7 days of ERCOT market data)
- Compute all features (~40-50 dimensions)
- Generate labels (SpikeEvent, LeadSpike, Regime)
- Save the processed data

### 3. View the Design Document

The detailed algorithm design document is located at [DESIGN.md](DESIGN.md), covering:
- System architecture
- Detailed feature engineering description
- Prediction model design
- Strategy optimization framework
- Implementation roadmap

## Feature Engineering

### Feature Groups (~40-50 dimensions)

#### 1. Price Structure Features
Capture regional scarcity and congestion signals
- Zone-system spread (`spread_zone_hub`)
- Real-time to day-ahead premium (`spread_rt_da`)
- Price slope and acceleration

#### 2. Supply-Demand Balance Features
Capture system/regional stress conditions
- Net load and its ramp rate, acceleration
- Wind anomaly (`wind_anomaly`)
- Gas generation saturation (`gas_saturation`)
- Nighttime coal ramp-up (`coal_stress`)
- ESR net output

#### 3. Weather-Driven Features (Zone-level)
Capture demand-side shock signals
- Temperature anomaly (`T_anomaly_zone`)
- Cooling rate (`T_ramp_zone`)
- Wind chill index (`WindChill_zone`)
- Cold front flag (`ColdFront_zone`)

#### 4. Temporal Features
Capture intraday patterns and solar recovery windows
- Hour, day of week, month
- Evening peak flag
- Minutes to sunrise (`minutes_to_sunrise`)
- Expected solar ramp

### Usage Example

```python
from src.data.feature_engineering import FeatureEngineer

# Initialize
feature_engineer = FeatureEngineer(
    zones=['CPS', 'West', 'Houston'],
    lookback_days=30
)

# Compute all features
df_with_features = feature_engineer.calculate_all_features(raw_data)

# Get feature names
price_features = feature_engineer.get_feature_names('price')
```

## Label Generation

### Label Types

1. **SpikeEvent**: Spike event indicator (0/1)
2. **LeadSpike**: Early warning label (0/1, 60 minutes in advance)
3. **Regime**: System state (Normal / Tight / Scarcity)

### Usage Example

```python
from src.utils.labels import LabelGenerator

# Initialize
label_generator = LabelGenerator(
    zones=['CPS', 'West', 'Houston'],
    P_hi=400,      # Spike price threshold
    S_hi=50,       # Spike spread threshold
    H=60,          # Lead Spike warning window (minutes)
)

# Generate all labels
labels = label_generator.generate_all_labels(df)

# Identify independent Spike events
events = label_generator.identify_spike_events(labels['SpikeEvent_CPS'])
```

## Data Requirements

### Required Data

#### Market Data (5-minute/15-minute)
- RT LMP (Real-Time Market Prices): P_CPS, P_West, P_Houston, P_Hub
- DA LMP (Day-Ahead Market Prices): P_CPS_DA, P_West_DA, P_Houston_DA

#### System Data (5-minute)
- Load
- Wind (Wind Generation)
- Solar (Solar Generation)
- Gas (Natural Gas Generation)
- Coal (Coal Generation)
- ESR (ESR Net Output)

#### Weather Data (15-minute, Zone-level)
- T_{zone} (Temperature)
- WindSpeed_{zone} (Wind Speed)
- WindDir_{zone} (Wind Direction)

### Data Sources

- **ERCOT Data**: http://www.ercot.com/
  - Real-Time Market (RTM) prices
  - Day-Ahead Market (DAM) prices
  - Fuel Mix data

- **Weather Data**: NOAA / Weather API
  - San Antonio (CPS)
  - West Texas
  - Houston

## Implementation Roadmap

See [DESIGN.md](DESIGN.md) Section 6 for details.

**Core Strategy**:
1. **Phase 1**: Historical data preparation (data collection, feature engineering, label generation)
2. **Phase 2**: Model training and validation (XGBoost baseline + CfC advanced model)
3. **Phase 3**: Strategy backtesting (hard rules + MPC optimization)
4. **Phase 4**: 12/14-15 Case Study (demonstrating PowerA technical capabilities)
5. **Phase 5-6**: Real-time system planning and deployment (future)

## Key Results

### 12/14-15 Case Study Objectives

Case analysis targeting the 2025-12-14 evening peak Spike event:

**Problem**:
- ESR discharged prematurely during 16:30-20:00 ($100-$325 range)
- Missed the 20:00-22:00 extreme price window ($400-$686)

**PowerA Solution**:
- 60-minute advance warning (predicting 20:00-22:00 spike at 16:00)
- Hold-SOC strategy (limiting discharge during Tight state)
- Release SOC during the extreme price window

**Expected Revenue Improvement**: 30-60%

## Technology Stack

- **Data**: Pandas, NumPy
- **ML Baseline**: XGBoost, LightGBM, Scikit-learn
- **Deep Learning**: PyTorch, ncps (CfC/LTC)
- **Optimization**: SciPy, CVXPY
- **Visualization**: Matplotlib, Seaborn, Plotly

## Contributing

This project is developed by the PowerA AI Team.

## License

Private Repository - PowerA AI

## Contact

- **Organization**: powerA-ai
- **Repository**: https://github.com/powerA-ai/spike-forecast

---

**Last Updated**: 2025-12-29
