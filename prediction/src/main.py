"""
TrueFlux Prediction Service

FastAPI service for serving electricity price predictions:
- DAM V2: Next-day price prediction (LightGBM, per settlement point)
- RTM: Multi-horizon 1h/4h/24h forecasting (LightGBM)
- Delta Spread: RTM-DAM spread predictions (CatBoost regression + classification)
- Spike Detection: Next-hour spike alerts (CatBoost binary classifier)
"""

import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import SETTLEMENT_POINTS
from .models.delta_spread import get_predictor
from .models.dam_v2_predictor import get_dam_v2_predictor
from .models.rtm_predictor import get_rtm_predictor
from .models.spike_predictor import get_spike_predictor
from .features.unified_features import compute_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


app = FastAPI(
    title="TrueFlux Prediction Service",
    description=(
        "API for ERCOT electricity price predictions.\n\n"
        "**Models:**\n"
        "- **DAM V2** — Next-day DAM price (LightGBM, 5 settlement points)\n"
        "- **RTM** — Multi-horizon RTM price (1h/4h/24h, LightGBM)\n"
        "- **Delta Spread** — RTM-DAM spread with trading signals (CatBoost)\n"
        "- **Spike Detection** — Next-hour spike alerts (CatBoost, F1=0.939)\n"
    ),
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3002", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class ModelStatus(BaseModel):
    name: str
    loaded: bool
    details: Dict[str, Any] = {}


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    models: List[ModelStatus]


class PredictionResponse(BaseModel):
    status: str
    model: str
    settlement_point: str
    generated_at: str
    predictions: List[Dict[str, Any]]


class ModelInfoResponse(BaseModel):
    model_name: str
    status: str
    info: Dict[str, Any]


# ---------------------------------------------------------------------------
# Health & info
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Service health check with status of all loaded models."""
    delta = get_predictor()
    delta_info = delta.get_model_info()

    dam = get_dam_v2_predictor()
    dam_info = dam.get_model_info()

    rtm = get_rtm_predictor()
    rtm_info = rtm.get_model_info()

    spike = get_spike_predictor()
    spike_info = spike.get_model_info()

    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version="3.0.0",
        models=[
            ModelStatus(
                name="dam_v2",
                loaded=dam.is_ready(),
                details={"models_loaded": dam_info["models_loaded"], "settlement_points": dam_info["settlement_points"]},
            ),
            ModelStatus(
                name="rtm",
                loaded=rtm.is_ready(),
                details={"models_loaded": rtm_info["models_loaded"], "settlement_points": rtm_info["settlement_points"]},
            ),
            ModelStatus(
                name="delta_spread",
                loaded=delta_info["regression_loaded"],
                details={k: delta_info[k] for k in ("regression_loaded", "binary_loaded", "multiclass_loaded")},
            ),
            ModelStatus(
                name="spike",
                loaded=spike.is_ready(),
                details={"models_loaded": spike_info["models_loaded"], "settlement_points": spike_info["settlement_points"]},
            ),
        ],
    )


@app.get("/settlement-points", tags=["System"])
async def list_settlement_points():
    """List available ERCOT settlement points."""
    return {
        "hubs": ["HB_HOUSTON", "HB_NORTH", "HB_SOUTH", "HB_WEST"],
        "load_zones": ["LZ_HOUSTON", "LZ_NORTH", "LZ_SOUTH", "LZ_WEST"],
        "all": SETTLEMENT_POINTS,
    }


# ---------------------------------------------------------------------------
# Model info endpoints
# ---------------------------------------------------------------------------

@app.get("/models/dam/info", response_model=ModelInfoResponse, tags=["Models"])
async def dam_model_info():
    """DAM V2 model metadata."""
    p = get_dam_v2_predictor()
    return ModelInfoResponse(
        model_name="dam_v2",
        status="loaded" if p.is_ready() else "not_loaded",
        info=p.get_model_info(),
    )


@app.get("/models/rtm/info", response_model=ModelInfoResponse, tags=["Models"])
async def rtm_model_info():
    """RTM multi-horizon model metadata."""
    p = get_rtm_predictor()
    return ModelInfoResponse(
        model_name="rtm",
        status="loaded" if p.is_ready() else "not_loaded",
        info=p.get_model_info(),
    )


@app.get("/models/delta-spread/info", response_model=ModelInfoResponse, tags=["Models"])
async def delta_spread_info():
    """Delta Spread model metadata."""
    p = get_predictor()
    return ModelInfoResponse(
        model_name="delta_spread",
        status="loaded",
        info=p.get_model_info(),
    )


@app.get("/models/spike/info", response_model=ModelInfoResponse, tags=["Models"])
async def spike_model_info():
    """Spike detection model metadata."""
    p = get_spike_predictor()
    return ModelInfoResponse(
        model_name="spike",
        status="loaded" if p.is_ready() else "not_loaded",
        info=p.get_model_info(),
    )


# ---------------------------------------------------------------------------
# DAM predictions
# ---------------------------------------------------------------------------

@app.get("/predictions/dam/next-day", tags=["Predictions"])
async def predict_dam_next_day(
    settlement_point: str = Query(default="HB_WEST", description="Settlement point (e.g. HB_WEST, HB_HOUSTON)"),
    target_date: Optional[str] = Query(default=None, description="Target date YYYY-MM-DD (default: tomorrow)"),
):
    """
    Next-day DAM price predictions (24 hours).

    Uses LightGBM models trained per settlement point with 41 features.
    Fetches recent DAM + RTM data from InfluxDB, computes features, and predicts.

    **Available settlement points**: hb_west, hb_north, hb_south, hb_houston, hb_busavg
    """
    predictor = get_dam_v2_predictor()
    if not predictor.is_ready():
        raise HTTPException(status_code=503, detail="DAM V2 models not loaded")

    sp_key = settlement_point.lower().replace("lz_", "hb_")
    if sp_key not in predictor.available_settlement_points():
        raise HTTPException(
            status_code=400,
            detail=f"No model for '{settlement_point}'. Available: {predictor.available_settlement_points()}",
        )

    target_dt = None
    if target_date:
        try:
            target_dt = datetime.strptime(target_date, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    try:
        features_df = _fetch_and_compute_features(settlement_point)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Data pipeline failed: {e}")

    # Filter to the target date (or last available day for "tomorrow" prediction)
    if target_dt:
        target_rows = features_df[features_df["delivery_date"] == str(target_dt)]
    else:
        # Use the last full day of features as a proxy for tomorrow's prediction
        last_date = features_df["delivery_date"].max()
        target_rows = features_df[features_df["delivery_date"] == last_date]

    if target_rows.empty:
        raise HTTPException(status_code=404, detail="Insufficient data to generate predictions for target date")

    try:
        predictions = predictor.predict(target_rows, sp_key)
        delivery_date = target_rows["delivery_date"].iloc[0]
        return {
            "status": "success",
            "model": "DAM V2 LightGBM",
            "settlement_point": settlement_point,
            "delivery_date": delivery_date,
            "generated_at": datetime.utcnow().isoformat(),
            "predictions": [
                {
                    "hour_ending": f"{p.hour_ending:02d}:00",
                    "predicted_price": p.predicted_price,
                }
                for p in predictions
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# RTM predictions
# ---------------------------------------------------------------------------

@app.get("/predictions/rtm", tags=["Predictions"])
async def predict_rtm(
    settlement_point: str = Query(default="HB_WEST", description="Settlement point"),
    horizons: Optional[str] = Query(default=None, description="Comma-separated horizons: 1h,4h,24h (default: all)"),
):
    """
    Real-Time Market price predictions at multiple horizons.

    Returns predicted RTM LMP price for 1-hour, 4-hour, and 24-hour ahead.
    Fetches recent market data, computes features, and runs inference.

    **Available**: hb_west (more settlement points coming)
    """
    predictor = get_rtm_predictor()
    if not predictor.is_ready():
        raise HTTPException(status_code=503, detail="RTM models not loaded")

    sp_key = settlement_point.lower().replace("lz_", "hb_")
    if sp_key not in predictor.available_settlement_points():
        raise HTTPException(
            status_code=400,
            detail=f"No RTM model for '{settlement_point}'. Available: {predictor.available_settlement_points()}",
        )

    horizon_list = None
    if horizons:
        horizon_list = [h.strip() for h in horizons.split(",")]

    try:
        features_df = _fetch_and_compute_features(settlement_point)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Data pipeline failed: {e}")

    # Use the most recent rows for prediction
    latest = features_df.tail(1)

    try:
        results = predictor.predict(latest, sp_key, horizon_list)
        return {
            "status": "success",
            "model": "RTM Multi-Horizon LightGBM",
            "settlement_point": settlement_point,
            "generated_at": datetime.utcnow().isoformat(),
            "predictions": [
                {
                    "horizon": r.horizon,
                    "hours_ahead": r.hours_ahead,
                    "predicted_price": r.predicted_price,
                }
                for r in results
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Delta Spread predictions
# ---------------------------------------------------------------------------

@app.get("/predictions/delta-spread", tags=["Predictions"])
async def predict_delta_spread(
    settlement_point: str = Query(default="LZ_WEST", description="ERCOT settlement point"),
    hours_ahead: int = Query(default=24, ge=1, le=48, description="Hours to forecast"),
):
    """
    Delta Spread (RTM - DAM) predictions with trading signals.

    Predicts the spread between Real-Time and Day-Ahead Market prices.
    Returns spread value, direction, interval classification, and trading signal.

    **Signals**: STRONG_LONG, LONG, HOLD, SHORT, STRONG_SHORT
    """
    try:
        predictor = get_predictor()
        features = _generate_delta_features(hours_ahead)
        predictions = predictor.predict(features)
        predictions["settlement_point"] = settlement_point
        predictions["forecast_horizon_hours"] = hours_ahead
        predictions["model"] = "Delta Spread CatBoost (regression + binary + multiclass)"
        predictions["status"] = "success"
        return predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Spike detection
# ---------------------------------------------------------------------------

@app.get("/predictions/spike", tags=["Predictions"])
async def predict_spike(
    settlement_point: str = Query(default="HB_WEST", description="Settlement point"),
):
    """
    Spike detection alert for the next hour.

    Predicts whether RTM price will spike (> max($100, 3x rolling 24h mean))
    in the next hour. Returns probability and confidence level.

    **Performance**: Precision=0.886, Recall=1.0, F1=0.939

    **Available**: hb_west (more settlement points coming)
    """
    predictor = get_spike_predictor()
    if not predictor.is_ready():
        raise HTTPException(status_code=503, detail="Spike models not loaded")

    sp_key = settlement_point.lower().replace("lz_", "hb_")
    if sp_key not in predictor.available_settlement_points():
        raise HTTPException(
            status_code=400,
            detail=f"No spike model for '{settlement_point}'. Available: {predictor.available_settlement_points()}",
        )

    try:
        features_df = _fetch_and_compute_features(settlement_point)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Data pipeline failed: {e}")

    # Use the most recent row for prediction
    latest = features_df.tail(1)

    try:
        alerts = predictor.predict(latest, sp_key)
        alert = alerts[0]

        return {
            "status": "success",
            "model": "Spike Detection CatBoost",
            "settlement_point": settlement_point,
            "generated_at": datetime.utcnow().isoformat(),
            "alert": {
                "is_spike": alert.is_spike,
                "spike_probability": alert.spike_probability,
                "confidence": alert.confidence,
                "threshold": alert.threshold_used,
                "lookahead": "1 hour",
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch_and_compute_features(settlement_point: str) -> pd.DataFrame:
    """
    Fetch DAM + RTM data from InfluxDB and compute unified features.

    Returns a DataFrame with all 41 feature columns needed by DAM/RTM/Spike models.
    Raises HTTPException on failure.
    """
    from .data.influxdb_fetcher import create_fetcher_from_env

    try:
        fetcher = create_fetcher_from_env()
        # Fetch last 30 days (enough for 168h rolling windows + buffer)
        start = datetime.utcnow() - timedelta(days=30)
        dam_raw = fetcher.fetch_dam_prices(settlement_point=settlement_point, start_date=start)
        rtm_raw = fetcher.fetch_rtm_prices(settlement_point=settlement_point, start_date=start)
        fetcher.close()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"InfluxDB fetch failed: {e}")

    if dam_raw.empty or rtm_raw.empty:
        raise HTTPException(status_code=404, detail=f"Insufficient data for {settlement_point}")

    # Convert InfluxDB format to unified_features format
    # DAM: needs delivery_date, hour_ending, lmp
    dam_hourly = dam_raw.reset_index()
    dam_hourly = dam_hourly.rename(columns={"dam_price": "lmp"})
    if "date" not in dam_hourly.columns:
        dam_hourly["date"] = dam_hourly["timestamp"].dt.date
    dam_hourly["delivery_date"] = dam_hourly["date"].astype(str)
    dam_hourly["hour_ending"] = dam_hourly["timestamp"].dt.hour + 1
    dam_hourly = dam_hourly[["delivery_date", "hour_ending", "lmp"]].drop_duplicates(
        subset=["delivery_date", "hour_ending"], keep="last"
    )

    # RTM: needs delivery_date, hour_ending, lmp (aggregate 5-min to hourly)
    rtm_hourly = rtm_raw.reset_index()
    rtm_hourly = rtm_hourly.rename(columns={"rtm_price": "lmp"})
    rtm_hourly["delivery_date"] = rtm_hourly["timestamp"].dt.date.astype(str)
    rtm_hourly["hour_ending"] = rtm_hourly["timestamp"].dt.hour + 1
    rtm_hourly = rtm_hourly.groupby(["delivery_date", "hour_ending"], as_index=False)["lmp"].mean()

    features_df = compute_features(dam_hourly, rtm_hourly)

    if features_df.empty:
        raise HTTPException(status_code=404, detail="Feature computation returned no rows (insufficient history)")

    return features_df


def _generate_delta_features(hours: int) -> pd.DataFrame:
    """
    Generate mock features for Delta Spread prediction.

    In production, replace with real InfluxDB data pipeline.
    """
    now = datetime.utcnow()
    target_date = now + timedelta(days=1)

    records = []
    for h in range(1, min(hours + 1, 25)):
        target_dt = target_date.replace(hour=h % 24, minute=0, second=0, microsecond=0)

        record = {
            "target_dam_price": 35.0 + np.random.randn() * 10,
            "target_hour": int(h),
            "target_dow": int(target_dt.weekday()),
            "target_month": int(target_dt.month),
            "target_day_of_month": int(target_dt.day),
            "target_week": int(target_dt.isocalendar()[1]),
            "target_is_weekend": int(1 if target_dt.weekday() >= 5 else 0),
            "target_is_peak": int(1 if 7 <= h <= 22 else 0),
            "target_is_summer": int(1 if target_dt.month in [6, 7, 8, 9] else 0),
            "spread_mean_7d": -7.0 + np.random.randn() * 3,
            "spread_std_7d": 15.0 + np.random.randn() * 5,
            "spread_max_7d": 50.0 + np.random.randn() * 20,
            "spread_min_7d": -30.0 + np.random.randn() * 10,
            "spread_median_7d": -5.0 + np.random.randn() * 3,
            "spread_mean_24h": -6.0 + np.random.randn() * 5,
            "spread_std_24h": 12.0 + np.random.randn() * 4,
            "rtm_mean_7d": 28.0 + np.random.randn() * 8,
            "rtm_std_7d": 25.0 + np.random.randn() * 10,
            "rtm_max_7d": 150.0 + np.random.randn() * 50,
            "rtm_mean_24h": 27.0 + np.random.randn() * 10,
            "rtm_volatility_24h": 20.0 + np.random.randn() * 8,
            "dam_mean_7d": 35.0 + np.random.randn() * 8,
            "dam_std_7d": 15.0 + np.random.randn() * 5,
            "dam_mean_24h": 34.0 + np.random.randn() * 10,
            "spread_positive_ratio_7d": 0.25 + np.random.rand() * 0.2,
            "spread_positive_ratio_24h": 0.25 + np.random.rand() * 0.3,
            "spike_count_7d": int(np.random.randint(0, 5)),
            "rtm_spike_count_7d": int(np.random.randint(0, 8)),
            "spread_trend_7d": np.random.randn() * 0.5,
            "spread_same_hour_hist": -7.0 + np.random.randn() * 5,
            "spread_same_hour_std": 15.0 + np.random.randn() * 5,
            "rtm_same_hour_hist": 28.0 + np.random.randn() * 10,
            "spread_same_dow_hour": -6.0 + np.random.randn() * 5,
            "dam_vs_7d_mean": np.random.randn() * 0.1,
            "dam_percentile_7d": 0.5 + np.random.randn() * 0.2,
            "spread_same_dow_hour_last": -5.0 + np.random.randn() * 8,
            "dam_price_level": int(np.random.choice([0, 1, 2])),
            "hour_sin": np.sin(2 * np.pi * h / 24),
            "hour_cos": np.cos(2 * np.pi * h / 24),
            "dow_sin": np.sin(2 * np.pi * target_dt.weekday() / 7),
            "dow_cos": np.cos(2 * np.pi * target_dt.weekday() / 7),
            "month_sin": np.sin(2 * np.pi * target_dt.month / 12),
            "month_cos": np.cos(2 * np.pi * target_dt.month / 12),
            "dam_vs_same_hour": np.random.randn() * 5,
            "target_date": target_dt.strftime("%Y-%m-%d"),
        }
        records.append(record)

    return pd.DataFrame(records)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8011)
