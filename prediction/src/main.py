"""
TrueFlux Prediction Service

FastAPI service for serving electricity price predictions:
- DAM V2: Next-day price prediction (LightGBM, per settlement point)
- RTM: Multi-horizon 1h/4h/24h forecasting (LightGBM)
- Delta Spread: RTM-DAM spread predictions (CatBoost regression + classification)
- Spike Detection: Next-hour spike alerts (CatBoost binary classifier)
- Wind: Wind generation forecast (LightGBM quantile regression)
- Load: Total system load forecast (CatBoost + LightGBM ensemble)
- BESS: Battery storage optimal charge/discharge schedule (LP optimizer)
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import SETTLEMENT_POINTS
from .models.delta_spread import get_predictor
from .models.dam_v2_predictor import get_dam_v2_predictor
from .models.rtm_predictor import get_rtm_predictor
from .models.spike_predictor import get_spike_predictor
from .models.wind_predictor import get_wind_predictor
from .models.load_predictor import get_load_predictor
from .models.bess_predictor import get_bess_predictor
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
        "- **Wind** — Wind generation forecast (LightGBM quantile, Q10/Q50/Q90)\n"
        "- **Load** — Total system load forecast (CatBoost + LightGBM ensemble)\n"
        "- **BESS** — Battery storage optimal schedule (LP optimizer)\n"
    ),
    version="4.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3002"],
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
    details: Dict[str, Any] = Field(default_factory=dict)


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


ALLOWED_HORIZONS = {"1h", "4h", "24h"}
DELTA_SPREAD_SETTLEMENT_POINTS = {"LZ_WEST", "HB_WEST"}


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

    wind = get_wind_predictor()
    wind_info = wind.get_model_info()

    load = get_load_predictor()
    load_info = load.get_model_info()

    bess = get_bess_predictor()
    bess_info = bess.get_model_info()

    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version="4.0.0",
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
            ModelStatus(
                name="wind",
                loaded=wind.is_ready(),
                details={"feature_count": wind_info.get("feature_count", 0), "quantiles": wind_info.get("quantiles", [])},
            ),
            ModelStatus(
                name="load",
                loaded=load.is_ready(),
                details={"models_loaded": load_info.get("models_loaded", [])},
            ),
            ModelStatus(
                name="bess",
                loaded=bess.is_ready(),
                details={"optimizer_loaded": bess_info.get("optimizer_loaded", False)},
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


@app.get("/models/wind/info", response_model=ModelInfoResponse, tags=["Models"])
async def wind_model_info():
    """Wind generation forecast model metadata."""
    p = get_wind_predictor()
    return ModelInfoResponse(
        model_name="wind",
        status="loaded" if p.is_ready() else "not_loaded",
        info=p.get_model_info(),
    )


@app.get("/models/load/info", response_model=ModelInfoResponse, tags=["Models"])
async def load_model_info():
    """Load forecast model metadata."""
    p = get_load_predictor()
    return ModelInfoResponse(
        model_name="load",
        status="loaded" if p.is_ready() else "not_loaded",
        info=p.get_model_info(),
    )


@app.get("/models/bess/info", response_model=ModelInfoResponse, tags=["Models"])
async def bess_model_info():
    """BESS optimizer metadata."""
    p = get_bess_predictor()
    return ModelInfoResponse(
        model_name="bess",
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
    normalized_sp = _normalize_settlement_point(settlement_point)

    predictor = get_dam_v2_predictor()
    if not predictor.is_ready():
        raise HTTPException(status_code=503, detail="DAM V2 models not loaded")

    sp_key = normalized_sp.lower()
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
        features_df = _fetch_and_compute_features(normalized_sp)
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
            "settlement_point": normalized_sp,
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
    normalized_sp = _normalize_settlement_point(settlement_point)

    predictor = get_rtm_predictor()
    if not predictor.is_ready():
        raise HTTPException(status_code=503, detail="RTM models not loaded")

    sp_key = normalized_sp.lower()
    if sp_key not in predictor.available_settlement_points():
        raise HTTPException(
            status_code=400,
            detail=f"No RTM model for '{settlement_point}'. Available: {predictor.available_settlement_points()}",
        )

    horizon_list = _parse_horizons(horizons)

    try:
        features_df = _fetch_and_compute_features(normalized_sp)
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
            "settlement_point": normalized_sp,
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
        model_info = predictor.get_model_info()
        if not any(model_info[key] for key in ("regression_loaded", "binary_loaded", "multiclass_loaded")):
            raise HTTPException(status_code=503, detail="Delta spread models not loaded")
        normalized_sp = _normalize_delta_settlement_point(settlement_point)
        raise HTTPException(
            status_code=503,
            detail=(
                f"Delta spread live inference is unavailable for {normalized_sp}. "
                "The endpoint previously returned synthetic random features, which has been disabled until a real pipeline is implemented."
            ),
        )
    except HTTPException:
        raise
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
    normalized_sp = _normalize_settlement_point(settlement_point)

    predictor = get_spike_predictor()
    if not predictor.is_ready():
        raise HTTPException(status_code=503, detail="Spike models not loaded")

    sp_key = normalized_sp.lower()
    if sp_key not in predictor.available_settlement_points():
        raise HTTPException(
            status_code=400,
            detail=f"No spike model for '{settlement_point}'. Available: {predictor.available_settlement_points()}",
        )

    try:
        features_df = _fetch_and_compute_features(normalized_sp)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Data pipeline failed: {e}")

    # Use the most recent row for prediction
    try:
        alerts = predictor.predict(features_df, sp_key)
        alert = alerts[-1]

        return {
            "status": "success",
            "model": "Spike Detection CatBoost",
            "settlement_point": normalized_sp,
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
# Wind predictions
# ---------------------------------------------------------------------------

@app.get("/predictions/wind", tags=["Predictions"])
async def predict_wind():
    """
    Wind generation forecast for next 24 hours.

    Returns point forecast (Q0.50) with uncertainty bounds (Q0.10, Q0.90)
    from a LightGBM quantile regression model trained on ERCOT wind data.
    """
    predictor = get_wind_predictor()
    if not predictor.is_ready():
        raise HTTPException(status_code=503, detail="Wind model not loaded")

    try:
        features_df = _build_wind_features()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Wind feature pipeline failed: {e}")

    try:
        predictions = predictor.predict(features_df)
        return {
            "status": "success",
            "model": "Wind GBM Quantile",
            "generated_at": datetime.utcnow().isoformat(),
            "predictions": [
                {
                    "hour_ending": f"{p.hour_ending:02d}:00",
                    "predicted_mw": p.predicted_mw,
                    "lower_bound_mw": p.lower_bound_mw,
                    "upper_bound_mw": p.upper_bound_mw,
                }
                for p in predictions
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Load predictions
# ---------------------------------------------------------------------------

@app.get("/predictions/load", tags=["Predictions"])
async def predict_load():
    """
    Total system load forecast for next 24 hours.

    Uses CatBoost + LightGBM ensemble trained on ERCOT fuel mix historical data.
    Returns predicted load in MW for each hour.
    """
    predictor = get_load_predictor()
    if not predictor.is_ready():
        raise HTTPException(status_code=503, detail="Load models not loaded")

    try:
        features_df = _build_load_features()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Load feature pipeline failed: {e}")

    try:
        predictions = predictor.predict(features_df)
        return {
            "status": "success",
            "model": "Load CatBoost+LightGBM Ensemble",
            "generated_at": datetime.utcnow().isoformat(),
            "predictions": [
                {
                    "hour_ending": f"{p.hour_ending:02d}:00",
                    "predicted_load_mw": p.predicted_load_mw,
                }
                for p in predictions
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# BESS predictions
# ---------------------------------------------------------------------------

@app.get("/predictions/bess", tags=["Predictions"])
async def predict_bess(
    settlement_point: str = Query(default="HB_WEST", description="Settlement point for DAM prices"),
):
    """
    BESS optimal charge/discharge schedule.

    Takes DAM price forecasts and runs LP optimization to find the
    optimal battery schedule for maximum arbitrage revenue.

    Returns hourly charge/discharge actions, SoC trajectory, and
    estimated revenue.
    """
    normalized_sp = _normalize_settlement_point(settlement_point)

    # Get DAM predictions to use as input prices
    dam_predictor = get_dam_v2_predictor()
    if not dam_predictor.is_ready():
        raise HTTPException(status_code=503, detail="DAM models not loaded (required for BESS)")

    bess = get_bess_predictor()
    if not bess.is_ready():
        raise HTTPException(status_code=503, detail="BESS optimizer not loaded")

    try:
        features_df = _fetch_and_compute_features(normalized_sp)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Data pipeline failed: {e}")

    # Get last day's features for DAM prediction
    last_date = features_df["delivery_date"].max()
    target_rows = features_df[features_df["delivery_date"] == last_date]

    if target_rows.empty:
        raise HTTPException(status_code=404, detail="Insufficient data for BESS optimization")

    try:
        sp_key = normalized_sp.lower()
        dam_predictions = dam_predictor.predict(target_rows, sp_key)

        # Extract 24 hourly prices
        dam_prices = [p.predicted_price for p in dam_predictions[:24]]
        if len(dam_prices) < 24:
            raise HTTPException(status_code=500, detail=f"Only got {len(dam_prices)} DAM prices, need 24")

        result = bess.optimize(dam_prices)

        return {
            "status": "success",
            "model": "BESS LP Optimizer",
            "settlement_point": normalized_sp,
            "generated_at": datetime.utcnow().isoformat(),
            "optimization": {
                "status": result.status,
                "total_revenue": result.total_revenue,
                "solve_time_s": result.solve_time,
                "battery_config": result.config,
            },
            "schedule": [
                {
                    "hour_ending": f"{e.hour_ending:02d}:00",
                    "action": e.action,
                    "power_mw": e.power_mw,
                    "soc_pct": e.soc_pct,
                    "dam_price": e.dam_price,
                }
                for e in result.schedule
            ],
        }
    except HTTPException:
        raise
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

    fetcher = None
    try:
        fetcher = create_fetcher_from_env()
        # Fetch last 30 days (enough for 168h rolling windows + buffer)
        start = datetime.utcnow() - timedelta(days=30)
        dam_raw = fetcher.fetch_dam_prices(settlement_point=settlement_point, start_date=start)
        rtm_raw = fetcher.fetch_rtm_prices(settlement_point=settlement_point, start_date=start)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"InfluxDB fetch failed: {e}")
    finally:
        if fetcher is not None:
            fetcher.close()

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


def _build_wind_features() -> pd.DataFrame:
    """
    Build synthetic wind features for the next 24 hours.

    In production, replace with real HRRR weather data + historical wind generation
    from InfluxDB. For now, generates temporal + placeholder features matching
    the trained model's feature set.
    """
    from .models.wind_predictor import get_wind_predictor

    predictor = get_wind_predictor()
    feature_names = predictor.metadata.get("feature_names", [])
    if not feature_names:
        raise RuntimeError("Wind model metadata missing feature names")

    now = datetime.utcnow()
    records = []
    for h in range(24):
        hour = (now.hour + h + 1) % 24
        dt = now + timedelta(hours=h + 1)
        record: Dict[str, Any] = {}

        for feat in feature_names:
            if feat == "hour" or feat == "hour_of_day":
                record[feat] = hour
            elif feat == "hour_sin":
                record[feat] = np.sin(2 * np.pi * hour / 24)
            elif feat == "hour_cos":
                record[feat] = np.cos(2 * np.pi * hour / 24)
            elif feat == "day_of_week":
                record[feat] = dt.weekday()
            elif feat == "dow_sin":
                record[feat] = np.sin(2 * np.pi * dt.weekday() / 7)
            elif feat == "dow_cos":
                record[feat] = np.cos(2 * np.pi * dt.weekday() / 7)
            elif feat == "month":
                record[feat] = dt.month
            elif feat == "month_sin":
                record[feat] = np.sin(2 * np.pi * dt.month / 12)
            elif feat == "month_cos":
                record[feat] = np.cos(2 * np.pi * dt.month / 12)
            elif feat == "doy_sin":
                record[feat] = np.sin(2 * np.pi * dt.timetuple().tm_yday / 365)
            elif feat == "doy_cos":
                record[feat] = np.cos(2 * np.pi * dt.timetuple().tm_yday / 365)
            elif feat == "is_weekend":
                record[feat] = 1 if dt.weekday() >= 5 else 0
            elif feat == "is_peak_hour":
                record[feat] = 1 if 7 <= hour <= 22 else 0
            elif feat == "is_ramp_prone_hour":
                record[feat] = 1 if hour in (6, 7, 17, 18, 19) else 0
            elif feat == "is_morning":
                record[feat] = 1 if 6 <= hour < 12 else 0
            elif feat == "is_afternoon":
                record[feat] = 1 if 12 <= hour < 18 else 0
            elif feat == "is_evening":
                record[feat] = 1 if 18 <= hour < 22 else 0
            elif feat == "is_night":
                record[feat] = 1 if hour >= 22 or hour < 6 else 0
            elif feat == "is_winter":
                record[feat] = 1 if dt.month in (12, 1, 2) else 0
            elif feat == "is_spring":
                record[feat] = 1 if dt.month in (3, 4, 5) else 0
            elif feat == "is_summer":
                record[feat] = 1 if dt.month in (6, 7, 8) else 0
            elif feat == "is_fall":
                record[feat] = 1 if dt.month in (9, 10, 11) else 0
            elif feat == "is_no_solar_period":
                record[feat] = 1 if hour >= 20 or hour <= 5 else 0
            elif feat == "is_evening_peak":
                record[feat] = 1 if 17 <= hour <= 20 else 0
            elif feat == "hours_to_sunset":
                record[feat] = max(0, 19 - hour)
            elif feat == "hours_since_sunset":
                record[feat] = max(0, hour - 19) if hour >= 19 else max(0, hour + 5)
            else:
                # Lag, rolling, ramp, weather features — use reasonable defaults
                record[feat] = 0.0

        records.append(record)

    return pd.DataFrame(records)


def _build_load_features() -> pd.DataFrame:
    """
    Build load forecast features for the next 24 hours.

    In production, replace with real historical load data from InfluxDB.
    For now, generates temporal features + placeholder lag/rolling features
    matching the trained model's feature set.
    """
    from .models.load_predictor import FEATURE_COLS

    now = datetime.utcnow()
    records = []
    for h in range(24):
        hour = (now.hour + h + 1) % 24
        dt = now + timedelta(hours=h + 1)

        record = {
            "hour_of_day": hour,
            "day_of_week": dt.weekday(),
            "month": dt.month,
            "is_weekend": 1 if dt.weekday() >= 5 else 0,
            "is_peak_hour": 1 if 7 <= hour <= 22 else 0,
            "season": (dt.month % 12) // 3,  # 0=winter, 1=spring, 2=summer, 3=fall
            "is_holiday": 0,
            # Placeholder lag/rolling features (in production, fetch real data)
            "load_lag_1h": 45000.0,
            "load_lag_2h": 44500.0,
            "load_lag_3h": 44000.0,
            "load_lag_6h": 42000.0,
            "load_lag_12h": 40000.0,
            "load_lag_24h": 44000.0,
            "load_lag_48h": 43000.0,
            "load_lag_168h": 44500.0,
            "load_roll_6h_mean": 43000.0,
            "load_roll_6h_std": 1500.0,
            "load_roll_6h_min": 40000.0,
            "load_roll_6h_max": 46000.0,
            "load_roll_12h_mean": 42500.0,
            "load_roll_12h_std": 2000.0,
            "load_roll_12h_min": 38000.0,
            "load_roll_12h_max": 47000.0,
            "load_roll_24h_mean": 43000.0,
            "load_roll_24h_std": 2500.0,
            "load_roll_24h_min": 37000.0,
            "load_roll_24h_max": 48000.0,
            "load_roll_168h_mean": 43500.0,
            "load_roll_168h_std": 3000.0,
            "load_roll_168h_min": 35000.0,
            "load_roll_168h_max": 50000.0,
            "load_change_1h": 500.0,
            "load_change_24h": 0.0,
            "load_roc_1h": 0.01,
        }
        records.append(record)

    df = pd.DataFrame(records)

    # Ensure all required columns exist
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0

    return df[FEATURE_COLS]


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


def _normalize_settlement_point(settlement_point: str) -> str:
    value = settlement_point.strip().upper()
    if not value:
        raise HTTPException(status_code=400, detail="Settlement point is required")
    value = value.replace("LZ_", "HB_")
    allowed = {sp.upper() for sp in SETTLEMENT_POINTS if sp.startswith("HB_")}
    if value not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported settlement point '{settlement_point}'")
    return value


def _normalize_delta_settlement_point(settlement_point: str) -> str:
    value = settlement_point.strip().upper()
    if value not in DELTA_SPREAD_SETTLEMENT_POINTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported settlement point '{settlement_point}'. Available: {sorted(DELTA_SPREAD_SETTLEMENT_POINTS)}",
        )
    return value


def _parse_horizons(horizons: Optional[str]) -> Optional[List[str]]:
    if horizons is None:
        return None

    parsed = [h.strip() for h in horizons.split(",") if h.strip()]
    if not parsed:
        raise HTTPException(status_code=400, detail="At least one horizon must be provided")

    invalid = [h for h in parsed if h not in ALLOWED_HORIZONS]
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported horizons {invalid}. Available: {sorted(ALLOWED_HORIZONS)}",
        )

    seen = set()
    ordered = []
    for horizon in parsed:
        if horizon not in seen:
            ordered.append(horizon)
            seen.add(horizon)
    return ordered


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8011)
