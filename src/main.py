"""
TrueFlux Prediction Service

FastAPI service for serving electricity price predictions.
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from .config import SETTLEMENT_POINTS, DAM_HORIZONS, RTM_SHORT_HORIZONS
from .models.delta_spread import get_predictor, DeltaSpreadPredictor
from .models.dam_predictor import get_dam_predictor, DAMPredictor
from .models.dam_simple_predictor import SimpleDAMPredictor
from pathlib import Path

# Simple DAM predictor instance (lazy loaded)
_simple_dam_predictor = None

def get_simple_dam_predictor(settlement_point: str = "HB_HOUSTON") -> SimpleDAMPredictor:
    """Get or create simple DAM predictor."""
    global _simple_dam_predictor
    model_path = Path(__file__).parent.parent / "models" / f"dam_simple_{settlement_point.lower()}.joblib"

    if _simple_dam_predictor is None or not _simple_dam_predictor.is_ready():
        _simple_dam_predictor = SimpleDAMPredictor(model_path)

    return _simple_dam_predictor

app = FastAPI(
    title="TrueFlux Prediction Service",
    description="API for RTM & DAM electricity price predictions",
    version="1.0.0"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3002", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Response models
class PredictionResponse(BaseModel):
    status: str
    timestamp: str
    predictions: List[Dict[str, Any]]


class ModelInfoResponse(BaseModel):
    model_name: str
    status: str
    info: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    models: Dict[str, bool]


# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check service health and model status."""
    delta_predictor = get_predictor()
    delta_info = delta_predictor.get_model_info()

    dam_predictor = get_dam_predictor()
    dam_info = dam_predictor.get_model_info()

    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        models={
            "delta_spread_regression": delta_info["regression_loaded"],
            "delta_spread_binary": delta_info["binary_loaded"],
            "delta_spread_multiclass": delta_info["multiclass_loaded"],
            "dam_xgboost": "xgboost" in dam_info["models_loaded"],
            "dam_lightgbm": "lightgbm" in dam_info["models_loaded"],
            "dam_catboost": "catboost" in dam_info["models_loaded"],
        }
    )


@app.get("/models/delta-spread/info", response_model=ModelInfoResponse)
async def delta_spread_info():
    """Get Delta Spread model information."""
    predictor = get_predictor()
    return ModelInfoResponse(
        model_name="delta_spread",
        status="loaded",
        info=predictor.get_model_info()
    )


@app.get("/predictions/delta-spread")
async def predict_delta_spread(
    settlement_point: str = Query(default="LZ_WEST", description="ERCOT settlement point"),
    hours_ahead: int = Query(default=24, ge=1, le=48, description="Hours to forecast"),
):
    """
    Get Delta Spread (RTM-DAM) predictions.

    This endpoint predicts the spread between Real-Time Market and Day-Ahead Market prices.
    Positive spread means RTM > DAM, negative means RTM < DAM.
    """
    try:
        predictor = get_predictor()

        # Generate mock features for demonstration
        # In production, this would fetch real data from InfluxDB
        features = _generate_mock_features(hours_ahead)

        predictions = predictor.predict(features)
        predictions["settlement_point"] = settlement_point
        predictions["forecast_horizon_hours"] = hours_ahead

        return predictions

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predictions/dam")
async def predict_dam(
    settlement_point: str = Query(default="LZ_HOUSTON", description="ERCOT settlement point"),
    hours: List[int] = Query(default=DAM_HORIZONS, description="Hours to forecast (1-24)"),
):
    """
    Get Day-Ahead Market price predictions.

    Returns predicted DAM prices for specified hours ahead.
    Uses ensemble of XGBoost, LightGBM, and CatBoost models.
    """
    try:
        dam_predictor = get_dam_predictor()

        if not dam_predictor.is_ready():
            # Models not yet trained - return placeholder
            return {
                "status": "pending",
                "message": "DAM models not yet trained. Run train_dam_models.py first.",
                "settlement_point": settlement_point,
                "forecast_hours": hours,
                "predictions": [
                    {
                        "hour": h,
                        "predicted_price": None,
                        "confidence": None,
                        "model_status": "pending_training"
                    }
                    for h in hours
                ]
            }

        # Generate features for prediction
        features_df = _generate_dam_features(hours)
        result = dam_predictor.predict(features_df)

        result["settlement_point"] = settlement_point
        result["target_date"] = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/dam/info", response_model=ModelInfoResponse)
async def dam_model_info():
    """Get DAM model information."""
    dam_predictor = get_dam_predictor()
    return ModelInfoResponse(
        model_name="dam",
        status="loaded" if dam_predictor.is_ready() else "not_loaded",
        info=dam_predictor.get_model_info()
    )


@app.get("/predictions/dam/next-day")
async def predict_dam_next_day(
    settlement_point: str = Query(default="HB_HOUSTON", description="ERCOT settlement point"),
    target_date: Optional[str] = Query(default=None, description="Target date (YYYY-MM-DD), default tomorrow"),
):
    """
    Get Day-Ahead Market price predictions for next day.

    Uses the simple DAM predictor with real InfluxDB data.
    Returns predictions for all 24 hours.
    """
    from .data.influxdb_fetcher import create_fetcher_from_env
    from datetime import datetime

    try:
        # Load predictor
        predictor = get_simple_dam_predictor(settlement_point)

        if not predictor.is_ready():
            raise HTTPException(
                status_code=503,
                detail=f"DAM model not trained for {settlement_point}. Run training first."
            )

        # Fetch recent DAM data from InfluxDB
        fetcher = create_fetcher_from_env()
        dam_df = fetcher.fetch_dam_prices(settlement_point=settlement_point)
        fetcher.close()

        if dam_df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No DAM data found for {settlement_point}"
            )

        # Parse target date
        target_dt = None
        if target_date:
            target_dt = datetime.strptime(target_date, "%Y-%m-%d")

        # Generate predictions
        predictions = predictor.predict_next_day(dam_df, target_dt)

        # Format response
        delivery_date = predictions[0].timestamp.date() if predictions else None

        return {
            "status": "success",
            "settlement_point": settlement_point,
            "delivery_date": str(delivery_date) if delivery_date else None,
            "generated_at": datetime.utcnow().isoformat(),
            "predictions": [
                {
                    "hour_ending": f"{p.hour_ending:02d}:00",
                    "predicted_price": round(p.predicted_price, 2),
                    "timestamp": p.timestamp.isoformat() if p.timestamp else None,
                }
                for p in predictions
            ]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predictions/rtm")
async def predict_rtm(
    settlement_point: str = Query(default="LZ_HOUSTON", description="ERCOT settlement point"),
    horizon_type: str = Query(default="short", enum=["short", "medium"], description="Forecast horizon type"),
):
    """
    Get Real-Time Market price predictions.

    Short-term: 15/30/45/60 minute horizons
    Medium-term: 1-24 hour horizons
    Note: Full RTM model integration pending.
    """
    horizons = RTM_SHORT_HORIZONS if horizon_type == "short" else [1, 2, 4, 6, 12, 24]

    # Placeholder - RTM models need to be serialized from notebooks
    return {
        "status": "partial",
        "message": "RTM model integration in progress",
        "settlement_point": settlement_point,
        "horizon_type": horizon_type,
        "predictions": [
            {
                "horizon": h,
                "horizon_unit": "intervals" if horizon_type == "short" else "hours",
                "predicted_delta": None,
                "confidence": None,
                "model_status": "pending_serialization"
            }
            for h in horizons
        ]
    }


@app.get("/settlement-points")
async def list_settlement_points():
    """List available ERCOT settlement points."""
    return {
        "hubs": ["HB_HOUSTON", "HB_NORTH", "HB_SOUTH", "HB_WEST"],
        "load_zones": ["LZ_HOUSTON", "LZ_NORTH", "LZ_SOUTH", "LZ_WEST"],
        "all": SETTLEMENT_POINTS
    }


def _generate_dam_features(hours: List[int]) -> pd.DataFrame:
    """
    Generate DAM features for prediction.

    In production, this would fetch real historical data from InfluxDB
    and compute the actual features. Currently uses mock data.
    """
    from .features.dam_features import FEATURE_NAMES

    now = datetime.utcnow()
    target_date = now + timedelta(days=1)  # D+1 prediction

    records = []
    for h in hours:
        target_dt = target_date.replace(hour=h - 1, minute=0, second=0, microsecond=0)

        # Generate mock features matching FEATURE_NAMES order
        record = {
            'hour': int(h),
            'day_of_week': int(target_dt.weekday()),
            'day_of_month': int(target_dt.day),
            'month': int(target_dt.month),
            'quarter': int((target_dt.month - 1) // 3 + 1),
            'week_of_year': int(target_dt.isocalendar()[1]),
            'is_weekend': int(1 if target_dt.weekday() >= 5 else 0),
            'is_holiday': 0,
            'is_peak_hour': int(1 if 7 <= h <= 22 else 0),
            'is_summer': int(1 if target_dt.month in [6, 7, 8, 9] else 0),
            'dam_lag_24h': 35.0 + np.random.randn() * 10,
            'dam_lag_24h_prev': 34.0 + np.random.randn() * 10,
            'dam_lag_24h_next': 36.0 + np.random.randn() * 10,
            'dam_lag_48h': 33.0 + np.random.randn() * 10,
            'dam_lag_168h': 34.0 + np.random.randn() * 10,
            'dam_d1_mean': 35.0 + np.random.randn() * 5,
            'dam_d1_max': 55.0 + np.random.randn() * 15,
            'dam_d1_min': 20.0 + np.random.randn() * 5,
            'dam_d1_std': 10.0 + np.random.randn() * 3,
            'dam_7d_mean': 34.0 + np.random.randn() * 5,
            'dam_7d_same_hour_mean': 35.0 + np.random.randn() * 8,
            'dam_4w_same_dow_hour_mean': 34.0 + np.random.randn() * 8,
            'dam_d1_hour_ratio': 1.0 + np.random.randn() * 0.2,
            'dam_7d_hour_ratio': 1.0 + np.random.randn() * 0.15,
            'dam_d1_vs_d2': np.random.randn() * 3,
            'dam_d1_vs_d7': np.random.randn() * 5,
            'dam_trend_3d': np.random.randn() * 0.5,
            'dam_7d_cv': 0.3 + np.random.randn() * 0.1,
            'dam_d1_range': 35.0 + np.random.randn() * 10,
            'dam_7d_same_hour_std': 8.0 + np.random.randn() * 3,
            'dam_d1_spike_count': int(np.random.choice([0, 0, 0, 1])),
            'dam_7d_spike_count': int(np.random.randint(0, 5)),
            'dam_d1_had_spike': int(np.random.choice([0, 0, 0, 1])),
            'dam_lag_24h_is_spike': 0,
            'dam_d1_max_hour': int(np.random.randint(7, 21)),
        }
        records.append(record)

    return pd.DataFrame(records)


def _generate_mock_features(hours: int) -> pd.DataFrame:
    """
    Generate mock features for demonstration.

    In production, this would fetch real historical data from InfluxDB
    and compute the required features.

    Feature order must match exactly what the model expects.
    """
    now = datetime.utcnow()
    target_date = now + timedelta(days=1)  # D+1 prediction

    records = []
    for h in range(1, min(hours + 1, 25)):
        target_dt = target_date.replace(hour=h, minute=0, second=0, microsecond=0)

        # Features in exact order expected by trained model
        record = {
            # Feature 0: target_dam_price (float)
            "target_dam_price": 35.0 + np.random.randn() * 10,
            # Features 1-8: Categorical (must be integers)
            "target_hour": int(h),
            "target_dow": int(target_dt.weekday()),
            "target_month": int(target_dt.month),
            "target_day_of_month": int(target_dt.day),
            "target_week": int(target_dt.isocalendar()[1]),
            "target_is_weekend": int(1 if target_dt.weekday() >= 5 else 0),
            "target_is_peak": int(1 if 7 <= h <= 22 else 0),
            "target_is_summer": int(1 if target_dt.month in [6, 7, 8, 9] else 0),
            # Spread features (floats)
            "spread_mean_7d": -7.0 + np.random.randn() * 3,
            "spread_std_7d": 15.0 + np.random.randn() * 5,
            "spread_max_7d": 50.0 + np.random.randn() * 20,
            "spread_min_7d": -30.0 + np.random.randn() * 10,
            "spread_median_7d": -5.0 + np.random.randn() * 3,
            "spread_mean_24h": -6.0 + np.random.randn() * 5,
            "spread_std_24h": 12.0 + np.random.randn() * 4,
            # RTM features (floats)
            "rtm_mean_7d": 28.0 + np.random.randn() * 8,
            "rtm_std_7d": 25.0 + np.random.randn() * 10,
            "rtm_max_7d": 150.0 + np.random.randn() * 50,
            "rtm_mean_24h": 27.0 + np.random.randn() * 10,
            "rtm_volatility_24h": 20.0 + np.random.randn() * 8,
            # DAM features (floats)
            "dam_mean_7d": 35.0 + np.random.randn() * 8,
            "dam_std_7d": 15.0 + np.random.randn() * 5,
            "dam_mean_24h": 34.0 + np.random.randn() * 10,
            # Ratio features
            "spread_positive_ratio_7d": 0.25 + np.random.rand() * 0.2,
            "spread_positive_ratio_24h": 0.25 + np.random.rand() * 0.3,
            # Spike counts (integers)
            "spike_count_7d": int(np.random.randint(0, 5)),
            "rtm_spike_count_7d": int(np.random.randint(0, 8)),
            # Trend and historical
            "spread_trend_7d": np.random.randn() * 0.5,
            "spread_same_hour_hist": -7.0 + np.random.randn() * 5,
            "spread_same_hour_std": 15.0 + np.random.randn() * 5,
            "rtm_same_hour_hist": 28.0 + np.random.randn() * 10,
            "spread_same_dow_hour": -6.0 + np.random.randn() * 5,
            "dam_vs_7d_mean": np.random.randn() * 0.1,
            "dam_percentile_7d": 0.5 + np.random.randn() * 0.2,
            "spread_same_dow_hour_last": -5.0 + np.random.randn() * 8,
            "dam_price_level": int(np.random.choice([0, 1, 2])),
            # Trigonometric features (floats)
            "hour_sin": np.sin(2 * np.pi * h / 24),
            "hour_cos": np.cos(2 * np.pi * h / 24),
            "dow_sin": np.sin(2 * np.pi * target_dt.weekday() / 7),
            "dow_cos": np.cos(2 * np.pi * target_dt.weekday() / 7),
            "month_sin": np.sin(2 * np.pi * target_dt.month / 12),
            "month_cos": np.cos(2 * np.pi * target_dt.month / 12),
            # Extra feature
            "dam_vs_same_hour": np.random.randn() * 5,
            # Metadata (not used for prediction)
            "target_date": target_dt.strftime("%Y-%m-%d"),
        }
        records.append(record)

    return pd.DataFrame(records)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
