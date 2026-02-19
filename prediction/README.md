# TrueFlux Prediction Service

FastAPI service for serving electricity price predictions.

## Features

- **Delta Spread Prediction**: RTM-DAM spread prediction using trained CatBoost models
- **DAM V2 Price Forecast**: Day-ahead market price predictions using 24 per-hour CatBoost models with 35 features
- **DAM Price Forecast**: Legacy DAM predictions (pending model serialization)
- **RTM Price Forecast**: Real-time market price predictions (pending model serialization)

## Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the service
./run.sh
```

## API Endpoints

### Health Check
```
GET /health
```

### DAM V2 Predictions (Recommended)
```
GET /predictions/dam-v2/next-day?settlement_point=LZ_WEST
```

Returns next-day DAM price predictions using 24 per-hour CatBoost models with 35 features.

**Example Response:**
```json
{
  "status": "success",
  "model": "DAM V2 (24 per-hour CatBoost)",
  "settlement_point": "LZ_WEST",
  "delivery_date": "2026-02-09",
  "predictions": [
    {"hour_ending": "01:00", "predicted_price": 15.04},
    {"hour_ending": "02:00", "predicted_price": 16.10},
    ...
  ]
}
```

### Delta Spread Predictions
```
GET /predictions/delta-spread?settlement_point=LZ_WEST&hours_ahead=24
```

### DAM Predictions (Legacy)
```
GET /predictions/dam/next-day?settlement_point=HB_HOUSTON
```

### RTM Predictions (Pending)
```
GET /predictions/rtm?settlement_point=LZ_HOUSTON&horizon_type=short
```

### Model Information
```
GET /models/dam-v2/info
GET /models/dam/info
GET /models/delta-spread/info
```

## Models

### DAM V2 (Production-Ready)
- **Location**: `models/dam_v2/dam_hour_01.cbm` to `dam_hour_24.cbm`
- **Architecture**: 24 per-hour CatBoost models
- **Features**: 35 features (time, lag, pattern, spike detection)
- **Training Data**: 96,889 records (2015-01-01 to 2026-02-09)
- **MAE**: $51.96 (28.7% improvement over naive baseline)

### Delta Spread (Production-Ready)
- **regression_model.cbm**: Predicts spread value (MAE: $8.98)
- **binary_model.cbm**: Predicts direction (AUC: 0.689)
- **multiclass_model.cbm**: Predicts 5 spread intervals

### DAM Forecast (Legacy)
- XGBoost, LightGBM, CatBoost ensemble
- 24 hours ahead, hourly granularity
- Best MAE: $8.54

### RTM Forecast (Pending Serialization)
- Short-term: 15/30/45/60 minute horizons
- Medium-term: 1-24 hour horizons
- 8-27% improvement over baseline

## Training DAM V2 Models

```bash
# Train models from combined CSV data
python scripts/train_dam_v2.py \
  --input /path/to/dam_lz_west_combined.csv \
  --output-dir models/dam_v2 \
  --train-start 2022-01-01 \
  --test-start 2026-01-01
```

## Project Structure

```
prediction-service/
├── src/
│   ├── main.py                    # FastAPI application
│   ├── config.py                  # Configuration
│   ├── data/
│   │   └── influxdb_fetcher.py    # InfluxDB data fetcher
│   ├── features/
│   │   ├── dam_features.py        # Legacy DAM features (9 features)
│   │   └── dam_features_v2.py     # V2 DAM features (35 features)
│   └── models/
│       ├── delta_spread.py        # Delta Spread predictor
│       ├── dam_predictor.py       # Legacy DAM predictor
│       ├── dam_simple_predictor.py # Simple DAM predictor
│       └── dam_v2_predictor.py    # V2 DAM predictor (24 per-hour models)
├── scripts/
│   ├── train_dam_models.py        # Legacy training script
│   └── train_dam_v2.py            # V2 training script
├── models/
│   ├── dam_v2/                    # V2 per-hour models
│   │   ├── dam_hour_01.cbm
│   │   ├── ...
│   │   └── dam_hour_24.cbm
│   └── delta_spread/              # Delta spread models
└── requirements.txt
```

## Integration

The service is designed to integrate with the trueflux-frontend Next.js application.

Frontend can fetch predictions from:
- `http://localhost:8001/predictions/dam-v2/next-day` (recommended)
- `http://localhost:8001/predictions/delta-spread`
- `http://localhost:8001/predictions/dam/next-day`
- `http://localhost:8001/predictions/rtm`
