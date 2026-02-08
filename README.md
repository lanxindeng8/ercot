# TrueFlux Prediction Service

FastAPI service for serving electricity price predictions.

## Features

- **Delta Spread Prediction**: RTM-DAM spread prediction using trained CatBoost models
- **DAM Price Forecast**: Day-ahead market price predictions (pending model serialization)
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

### Delta Spread Predictions
```
GET /predictions/delta-spread?settlement_point=LZ_WEST&hours_ahead=24
```

### DAM Predictions (pending)
```
GET /predictions/dam?settlement_point=LZ_HOUSTON
```

### RTM Predictions (pending)
```
GET /predictions/rtm?settlement_point=LZ_HOUSTON&horizon_type=short
```

## Models

### Delta Spread (Production-Ready)
- **regression_model.cbm**: Predicts spread value (MAE: $8.98)
- **binary_model.cbm**: Predicts direction (AUC: 0.689)
- **multiclass_model.cbm**: Predicts 5 spread intervals

### DAM Forecast (Pending Serialization)
- XGBoost, LightGBM, CatBoost ensemble
- 24 hours ahead, hourly granularity
- Best MAE: $8.54

### RTM Forecast (Pending Serialization)
- Short-term: 15/30/45/60 minute horizons
- Medium-term: 1-24 hour horizons
- 8-27% improvement over baseline

## Integration

The service is designed to integrate with the trueflux-frontend Next.js application.

Frontend can fetch predictions from:
- `http://localhost:8001/predictions/delta-spread`
- `http://localhost:8001/predictions/dam`
- `http://localhost:8001/predictions/rtm`
