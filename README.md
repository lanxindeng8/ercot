# ERCOT Wind Generation Forecast

0-12 hour wind power generation forecasting system for ERCOT/Texas using HRRR weather data.

## Features

- **Quantile Forecasting**: P10/P50/P90 predictions for uncertainty quantification
- **Ramp Detection**: Special focus on wind ramp-down events during no-solar periods
- **Multiple Models**: LightGBM (baseline), LSTM (sequence), and Ensemble options
- **HRRR Integration**: Uses Earth2Studio for high-resolution weather data

## Project Structure

```
├── configs/
│   └── default.yaml              # Configuration
├── scripts/
│   ├── fetch_hrrr_data.py        # Download HRRR data
│   ├── build_features.py         # Feature engineering
│   ├── train_models.py           # Model training
│   └── demo.py                   # Demo with synthetic data
├── src/
│   ├── data/                     # Data clients
│   ├── features/                 # Feature engineering
│   ├── models/                   # ML models
│   ├── evaluation/               # Metrics
│   └── utils/                    # Configuration
├── tests/
│   └── test_basic.py             # Unit tests
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

**📖 For detailed data setup instructions (HRRR download + ERCOT data), see [DATA_SETUP.md](DATA_SETUP.md)**

## Quick Start

### Run Demo (No HRRR data required)

```bash
python scripts/demo.py
```

### Full Pipeline

```bash
# 1. Fetch HRRR data
python scripts/fetch_hrrr_data.py --date 2025-01-20 --hours 0 6 12 18

# 2. Build features
python scripts/build_features.py

# 3. Train model
python scripts/train_models.py --model gbm
```

### Run Tests

```bash
python tests/test_basic.py
```

## Key Components

### Models

- **GBMWindModel**: LightGBM quantile regression (fast, good baseline)
- **LSTMWindModel**: LSTM with multi-quantile output
- **EnsembleWindModel**: Weighted combination of models

### Evaluation

- Standard metrics: MAE, RMSE, NMAE, Skill Score
- Quantile metrics: Coverage, Sharpness, Winkler Score
- **Ramp metrics**: POD, FAR, CSI
- **Critical**: `evaluate_ramp_down_in_no_solar()` for high-risk scenarios

## Critical Scenario: Ramp-Down + No Solar

The most important scenario for ERCOT is wind generation dropping rapidly during evening/night hours when solar cannot compensate. This leads to:
- Gas becoming marginal
- Price spikes
- Grid stress

The system specifically tracks and evaluates this scenario with dedicated metrics.

## License

MIT
