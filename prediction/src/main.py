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
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .config import SETTLEMENT_POINTS
from .models.delta_spread import get_predictor
from .models.dam_v2_predictor import get_dam_v2_predictor
from .models.rtm_predictor import get_rtm_predictor
from .models.spike_predictor import get_spike_predictor
from .models.wind_predictor import get_wind_predictor
from .models.load_predictor import get_load_predictor
from .models.bess_predictor import get_bess_predictor
from .dispatch.mining_dispatch import compute_dispatch, load_config as load_dispatch_config
from .dispatch.alert_service import get_alert_service
from .dispatch.bess_signals import (
    generate_daily_signals,
    record_daily_pnl,
    get_rolling_pnl,
    compute_risk_metrics,
    DailySignals,
)
from .features.unified_features import compute_features
from .auth.api_keys import get_key_manager, APIKey

import sqlite3

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
# Auth configuration
# ---------------------------------------------------------------------------

ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")
SKIP_AUTH_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}


@app.middleware("http")
async def api_key_auth_middleware(request: Request, call_next):
    """Authenticate requests via X-API-Key header. Skip auth for health/docs."""
    path = request.url.path

    # Skip auth for health, docs, and admin endpoints (admin has its own auth)
    if path in SKIP_AUTH_PATHS or path.startswith("/admin"):
        response = await call_next(request)
        return response

    api_key_header = request.headers.get("X-API-Key")
    if not api_key_header:
        return JSONResponse(status_code=401, content={"detail": "Missing X-API-Key header"})

    manager = get_key_manager()
    api_key = manager.validate_key(api_key_header)
    if api_key is None:
        return JSONResponse(status_code=401, content={"detail": "Invalid or revoked API key"})

    # Rate limit check
    if not manager.check_rate_limit(api_key):
        return JSONResponse(
            status_code=429,
            content={"detail": f"Rate limit exceeded for tier '{api_key.tier}'"},
        )

    # Attach key to request state for downstream use
    request.state.api_key = api_key

    response = await call_next(request)

    # Log usage asynchronously (best-effort)
    try:
        manager.record_usage(api_key, path, response.status_code)
    except Exception:
        log.warning("Failed to record usage for key %s", api_key.key_prefix, exc_info=True)

    return response


def _require_admin(x_admin_token: str = Header(...)):
    """Dependency that enforces ADMIN_TOKEN for /admin/* endpoints."""
    if not ADMIN_TOKEN:
        raise HTTPException(status_code=503, detail="Admin token not configured (set ADMIN_TOKEN env var)")
    if x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid admin token")


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

PREDICTIONS_DB = Path(__file__).resolve().parents[1] / "data" / "predictions.db"
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

    Uses LightGBM models trained per settlement point with 80 features.
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
        target_rows = _latest_complete_delivery_rows(features_df)

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
    if normalized_sp.lower() not in dam_predictor.available_settlement_points():
        raise HTTPException(
            status_code=400,
            detail=f"No DAM model for '{settlement_point}'. Available: {dam_predictor.available_settlement_points()}",
        )

    bess = get_bess_predictor()
    if not bess.is_ready():
        raise HTTPException(status_code=503, detail="BESS optimizer not loaded")

    try:
        features_df = _fetch_and_compute_features(normalized_sp)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Data pipeline failed: {e}")

    # Get the latest complete day of features for DAM prediction
    target_rows = _latest_complete_delivery_rows(features_df)

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
# Accuracy
# ---------------------------------------------------------------------------

@app.get("/accuracy", tags=["Accuracy"])
async def get_accuracy(
    model: str = Query(default="dam", description="Model name: dam or rtm"),
    days: int = Query(default=7, ge=1, le=90, description="Number of days to look back"),
):
    """
    Prediction accuracy statistics for recent predictions.

    Returns MAE, RMSE, MAPE, count, directional accuracy, and per-hour breakdown.
    """
    if model not in ("dam", "rtm"):
        raise HTTPException(status_code=400, detail=f"Model must be 'dam' or 'rtm', got '{model}'")

    if not PREDICTIONS_DB.exists():
        raise HTTPException(status_code=404, detail="Predictions database not found")

    conn = sqlite3.connect(str(PREDICTIONS_DB))
    try:
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

        # Overall metrics
        rows = conn.execute(
            """SELECT pa.error, pa.pct_error
               FROM prediction_accuracy pa
               JOIN predictions p ON p.id = pa.prediction_id
               WHERE p.model = ? AND pa.computed_at >= ?""",
            (model, cutoff),
        ).fetchall()

        if not rows:
            return {
                "status": "success",
                "model": model,
                "days": days,
                "metrics": {"mae": None, "rmse": None, "mape": None, "directional_accuracy": None, "count": 0},
                "hourly": {},
                "recent_comparisons": [],
            }

        errors = [r[0] for r in rows]
        pct_errors = [r[1] for r in rows]
        n = len(errors)
        abs_errors = [abs(e) for e in errors]
        mae = sum(abs_errors) / n
        rmse = float(np.sqrt(np.mean([e ** 2 for e in errors])))
        valid_pct = [abs(p) for p in pct_errors if p is not None]
        mape = sum(valid_pct) / len(valid_pct) if valid_pct else None

        # Directional accuracy
        direction_rows = conn.execute(
            """SELECT p.predicted_value, a.actual_value
               FROM prediction_accuracy pa
               JOIN predictions p ON p.id = pa.prediction_id
               JOIN actuals a ON a.id = pa.actual_id
               WHERE p.model = ? AND pa.computed_at >= ?
               ORDER BY p.target_time""",
            (model, cutoff),
        ).fetchall()

        dir_acc = None
        if len(direction_rows) > 1:
            correct = sum(
                1 for i in range(1, len(direction_rows))
                if (direction_rows[i][0] - direction_rows[i - 1][1]) *
                   (direction_rows[i][1] - direction_rows[i - 1][1]) > 0
            )
            dir_acc = round(correct / (len(direction_rows) - 1), 4)

        metrics = {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "mape": round(mape, 4) if mape is not None else None,
            "directional_accuracy": dir_acc,
            "count": n,
        }

        # Per-hour breakdown
        hour_rows = conn.execute(
            """SELECT
                   CAST(SUBSTR(p.target_time, 12, 2) AS INTEGER) as hour,
                   pa.error,
                   pa.pct_error
               FROM prediction_accuracy pa
               JOIN predictions p ON p.id = pa.prediction_id
               WHERE p.model = ? AND pa.computed_at >= ?""",
            (model, cutoff),
        ).fetchall()

        hourly: Dict[str, Any] = {}
        hour_buckets: Dict[int, Dict[str, list]] = {}
        for hour, error, pct_error in hour_rows:
            if hour not in hour_buckets:
                hour_buckets[hour] = {"errors": [], "pct_errors": []}
            hour_buckets[hour]["errors"].append(error)
            hour_buckets[hour]["pct_errors"].append(pct_error)

        for hour in sorted(hour_buckets.keys()):
            h_errors = hour_buckets[hour]["errors"]
            h_pct = hour_buckets[hour]["pct_errors"]
            h_n = len(h_errors)
            h_abs = [abs(e) for e in h_errors]
            h_valid_pct = [abs(p) for p in h_pct if p is not None]
            hourly[str(hour)] = {
                "mae": round(sum(h_abs) / h_n, 4),
                "rmse": round(float(np.sqrt(np.mean([e ** 2 for e in h_errors]))), 4),
                "mape": round(sum(h_valid_pct) / len(h_valid_pct), 4) if h_valid_pct else None,
                "count": h_n,
            }

        # Recent comparisons
        recent = conn.execute(
            """SELECT p.target_time, p.settlement_point, p.predicted_value,
                      a.actual_value, pa.error, pa.abs_error, pa.pct_error
               FROM prediction_accuracy pa
               JOIN predictions p ON p.id = pa.prediction_id
               JOIN actuals a ON a.id = pa.actual_id
               WHERE p.model = ? AND pa.computed_at >= ?
               ORDER BY p.target_time DESC
               LIMIT 48""",
            (model, cutoff),
        ).fetchall()

        recent_comparisons = [
            {
                "target_time": r[0],
                "settlement_point": r[1],
                "predicted": round(r[2], 2),
                "actual": round(r[3], 2),
                "error": round(r[4], 2),
                "abs_error": round(r[5], 2),
                "pct_error": round(r[6], 2) if r[6] is not None else None,
            }
            for r in recent
        ]

        return {
            "status": "success",
            "model": model,
            "days": days,
            "metrics": metrics,
            "hourly": hourly,
            "recent_comparisons": recent_comparisons,
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch_and_compute_features(settlement_point: str) -> pd.DataFrame:
    """
    Fetch DAM + RTM data and compute unified features (80 features).

    Prefers SQLite archive as primary source; falls back to InfluxDB if SQLite
    is unavailable or returns insufficient data.  Also loads ancillary service
    prices, RTM congestion components, and expanded fuel generation MW when
    available from the SQLite archive.

    Returns a DataFrame with all 80 feature columns needed by DAM/RTM/Spike models.
    Raises HTTPException on failure.
    """
    features_df = _try_sqlite_features(settlement_point)
    if features_df is not None:
        return features_df

    # Fallback: InfluxDB
    return _influxdb_features(settlement_point)


def _try_sqlite_features(settlement_point: str) -> Optional[pd.DataFrame]:
    """Attempt to load features from SQLite. Returns None on failure."""
    from .data.sqlite_fetcher import create_sqlite_fetcher
    from .data.training_pipeline import (
        load_ancillary_hourly,
        load_fuel_gen_hourly,
        load_fuel_mix_hourly,
        load_rtm_components_hourly,
    )

    fetcher = None
    db_path = None
    try:
        fetcher = create_sqlite_fetcher()
        db_path = fetcher.db_path
        start = datetime.utcnow() - timedelta(days=30)
        dam_raw = fetcher.fetch_dam_prices(settlement_point=settlement_point, start_date=start)
        rtm_raw = fetcher.fetch_rtm_prices(settlement_point=settlement_point, start_date=start)
    except Exception as e:
        log.warning("SQLite fetch failed, will try InfluxDB: %s", e)
        return None
    finally:
        if fetcher is not None:
            fetcher.close()

    if dam_raw.empty or rtm_raw.empty:
        log.info("SQLite returned no data for %s, will try InfluxDB", settlement_point)
        return None

    latest_rtm_ts = rtm_raw.index.max()
    if latest_rtm_ts is None or pd.isna(latest_rtm_ts):
        log.info("SQLite RTM data has no usable timestamps for %s, will try InfluxDB", settlement_point)
        return None
    latest_rtm_ts = pd.Timestamp(latest_rtm_ts)
    stale_cutoff = pd.Timestamp.utcnow().tz_localize(None) - timedelta(hours=24)
    if latest_rtm_ts.tzinfo is not None:
        latest_rtm_ts = latest_rtm_ts.tz_convert("UTC").tz_localize(None)
    if latest_rtm_ts < stale_cutoff:
        log.info("SQLite RTM data is stale for %s (latest %s), will try InfluxDB", settlement_point, latest_rtm_ts)
        return None

    dam_hourly, rtm_hourly = _raw_to_hourly(dam_raw, rtm_raw)

    # Load additional data sources for 80-feature pipeline
    if db_path is None:
        from .data.training_pipeline import DEFAULT_DB
        db_path = DEFAULT_DB
    start_str = start.strftime("%Y-%m-%d")
    ancillary_hourly = None
    fuel_hourly = None
    rtm_comp = None
    fuel_gen_hourly = None
    try:
        ancillary_hourly = load_ancillary_hourly(db_path, date_from=start_str)
        fuel_hourly = load_fuel_mix_hourly(db_path, date_from=start_str)
        rtm_comp = load_rtm_components_hourly(db_path, settlement_point, date_from=start_str)
        fuel_gen_hourly = load_fuel_gen_hourly(db_path, date_from=start_str)
    except Exception as e:
        log.warning("Failed to load ancillary/fuel/rtm-component data: %s", e)

    features_df = compute_features(
        dam_hourly, rtm_hourly, fuel_hourly,
        ancillary_hourly=ancillary_hourly,
        rtm_components_hourly=rtm_comp if rtm_comp is not None and not rtm_comp.empty else None,
        fuel_gen_hourly=fuel_gen_hourly,
    )

    if features_df.empty:
        log.info("SQLite feature computation returned no rows for %s", settlement_point)
        return None

    return features_df


def _influxdb_features(settlement_point: str) -> pd.DataFrame:
    """Fetch features from InfluxDB (fallback path)."""
    from .data.influxdb_fetcher import create_fetcher_from_env

    fetcher = None
    try:
        fetcher = create_fetcher_from_env()
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

    dam_hourly, rtm_hourly = _raw_to_hourly(dam_raw, rtm_raw)
    features_df = compute_features(dam_hourly, rtm_hourly)

    if features_df.empty:
        raise HTTPException(status_code=404, detail="Feature computation returned no rows (insufficient history)")

    return features_df


def _raw_to_hourly(dam_raw: pd.DataFrame, rtm_raw: pd.DataFrame):
    """Convert fetcher output (InfluxDB or SQLite format) to unified hourly frames."""
    # DAM: needs delivery_date, hour_ending, lmp
    dam_hourly = dam_raw.reset_index()
    dam_hourly = dam_hourly.rename(columns={"dam_price": "lmp"})
    if "date" not in dam_hourly.columns:
        dam_hourly["date"] = dam_hourly["timestamp"].dt.date
    dam_hourly["delivery_date"] = dam_hourly["date"].astype(str)
    if "hour" in dam_hourly.columns:
        dam_hourly["hour_ending"] = dam_hourly["hour"]
    else:
        dam_hourly["hour_ending"] = dam_hourly["timestamp"].dt.hour + 1
    dam_hourly = dam_hourly[["delivery_date", "hour_ending", "lmp"]].drop_duplicates(
        subset=["delivery_date", "hour_ending"], keep="last"
    )

    # RTM: needs delivery_date, hour_ending, lmp (aggregate 5-min to hourly)
    rtm_hourly = rtm_raw.reset_index()
    rtm_hourly = rtm_hourly.rename(columns={"rtm_price": "lmp"})
    if "date" in rtm_hourly.columns:
        rtm_hourly["delivery_date"] = pd.to_datetime(rtm_hourly["date"]).dt.strftime("%Y-%m-%d")
    else:
        rtm_hourly["delivery_date"] = rtm_hourly["timestamp"].dt.date.astype(str)
    if "hour" in rtm_hourly.columns:
        rtm_hourly["hour_ending"] = rtm_hourly["hour"]
    else:
        rtm_hourly["hour_ending"] = rtm_hourly["timestamp"].dt.hour + 1
    rtm_hourly = rtm_hourly.groupby(["delivery_date", "hour_ending"], as_index=False)["lmp"].mean()

    return dam_hourly, rtm_hourly


def _build_wind_features() -> pd.DataFrame:
    """
    Build synthetic wind features for the next 24 hours.

    In production, replace with real HRRR weather data + historical wind generation
    from InfluxDB. For now, generates temporal + placeholder features matching
    the trained model's feature set.
    """
    from .models.wind_predictor import get_wind_predictor

    history_path = (
        Path(__file__).resolve().parents[1] / "models" / "wind" / "data" / "ercot_wind.csv"
    )

    predictor = get_wind_predictor()
    feature_names = predictor.metadata.get("feature_names", [])
    if not feature_names:
        raise RuntimeError("Wind model metadata missing feature names")
    if history_path.exists():
        history_df = pd.read_csv(history_path)
        history_df["timestamp"] = pd.to_datetime(history_df["timestamp"])
        history_df = history_df.sort_values("timestamp")
        if history_df.empty:
            raise RuntimeError("Wind history is empty")
    else:
        raise RuntimeError(f"Wind history not found: {history_path}")

    generation_history = history_df["wind_generation"].astype(float).tolist()
    proxy_seed = float(np.mean(generation_history[-24:])) if generation_history else 0.0
    now = history_df["timestamp"].iloc[-1]
    records = []

    for h in range(24):
        dt = now + timedelta(hours=h + 1)
        hour = dt.hour
        recent = pd.Series(generation_history, dtype=float)
        record: Dict[str, Any] = {
            "normalized_power_mean": proxy_seed / 40000.0,
            "ws_80m_mean": 3.0 + 9.0 * np.power(np.clip(proxy_seed / 40000.0, 0.001, 0.999), 1.0 / 3.0),
            "ws_80m_std": (3.0 + 9.0 * np.power(np.clip(proxy_seed / 40000.0, 0.001, 0.999), 1.0 / 3.0)) * 0.15,
            "wind_shear_mean": 0.2,
            "power_sensitivity": max(0.0, 3 * ((max(3.0, 3.0 + 9.0 * np.power(np.clip(proxy_seed / 40000.0, 0.001, 0.999), 1.0 / 3.0)) - 3) / 9) ** 2 / 9),
            "hour": hour,
            "hour_of_day": hour,
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "day_of_week": dt.weekday(),
            "dow_sin": np.sin(2 * np.pi * dt.weekday() / 7),
            "dow_cos": np.cos(2 * np.pi * dt.weekday() / 7),
            "month": dt.month,
            "month_sin": np.sin(2 * np.pi * dt.month / 12),
            "month_cos": np.cos(2 * np.pi * dt.month / 12),
            "doy_sin": np.sin(2 * np.pi * dt.timetuple().tm_yday / 365),
            "doy_cos": np.cos(2 * np.pi * dt.timetuple().tm_yday / 365),
            "is_weekend": int(dt.weekday() >= 5),
            "is_peak_hour": int(7 <= hour <= 22),
            "is_ramp_prone_hour": int(hour in (6, 7, 17, 18, 19)),
            "is_morning": int(6 <= hour < 12),
            "is_afternoon": int(12 <= hour < 18),
            "is_evening": int(18 <= hour < 22),
            "is_night": int(hour >= 22 or hour < 6),
            "is_winter": int(dt.month in (12, 1, 2)),
            "is_spring": int(dt.month in (3, 4, 5)),
            "is_summer": int(dt.month in (6, 7, 8)),
            "is_fall": int(dt.month in (9, 10, 11)),
            "is_no_solar_period": int(hour >= 19 or hour < 7),
            "is_evening_peak": int(17 <= hour < 21),
            "hours_to_sunset": float(np.clip(19 - hour, -12, 12)),
            "hours_since_sunset": float(hour - 19 if hour >= 19 else (hour + 5 if hour < 7 else 0)),
        }

        for lag in (1, 2, 3, 6, 12, 24):
            record[f"wind_gen_lag_{lag}h"] = float(generation_history[-lag])

        for window in (6, 12, 24):
            window_values = recent.iloc[-window:]
            record[f"wind_gen_rolling_{window}h_mean"] = float(window_values.mean())
            record[f"wind_gen_rolling_{window}h_std"] = float(window_values.std(ddof=1) if len(window_values) > 1 else 0.0)
            record[f"wind_gen_rolling_{window}h_min"] = float(window_values.min())
            record[f"wind_gen_rolling_{window}h_max"] = float(window_values.max())

        record["wind_gen_change_1h"] = float(generation_history[-1] - generation_history[-2])
        record["wind_gen_change_3h"] = float(generation_history[-1] - generation_history[-4])
        record["wind_gen_change_6h"] = float(generation_history[-1] - generation_history[-7])
        record["wind_gen_roc_1h"] = float(record["wind_gen_change_1h"] / max(generation_history[-2], 100.0))
        record["wind_gen_roc_3h"] = float(record["wind_gen_change_3h"] / max(generation_history[-4], 100.0))
        record["ramp_down_1h"] = float(record["wind_gen_change_1h"] < -1000)
        record["ramp_down_3h"] = float(record["wind_gen_change_3h"] < -2000)
        record["ramp_up_1h"] = float(record["wind_gen_change_1h"] > 1000)

        risk = 0.0
        risk += 0.15 if record["wind_gen_change_1h"] < -1000 else 0.0
        risk += 0.20 if record["wind_gen_change_1h"] < -2000 else 0.0
        risk += 0.15 if record["is_no_solar_period"] else 0.0
        risk += 0.15 if record["is_no_solar_period"] and record["is_evening_peak"] else 0.0
        record["ramp_down_risk_score"] = float(min(max(risk, 0.0), 1.0))

        full_record = {name: record.get(name, 0.0) for name in feature_names}
        records.append(full_record)

        predicted = float(np.clip(
            0.55 * record["wind_gen_lag_1h"]
            + 0.30 * record["wind_gen_rolling_6h_mean"]
            + 0.15 * record["wind_gen_rolling_24h_mean"],
            0.0,
            40000.0,
        ))
        generation_history.append(predicted)
        proxy_seed = predicted

    return pd.DataFrame(records, columns=feature_names)


def _build_load_features() -> pd.DataFrame:
    """
    Build load forecast features for the next 24 hours.

    In production, replace with real historical load data from InfluxDB.
    For now, generates temporal features + placeholder lag/rolling features
    matching the trained model's feature set.
    """
    from .models.load_predictor import FEATURE_COLS
    from pandas.tseries.holiday import USFederalHolidayCalendar

    history_path = (
        Path(__file__).resolve().parents[1] / "models" / "load" / "data" / "ercot_load.csv"
    )
    if not history_path.exists():
        raise RuntimeError(f"Load history not found: {history_path}")

    history_df = pd.read_csv(history_path)
    history_df["timestamp"] = pd.to_datetime(history_df["timestamp"])
    history_df = history_df.sort_values("timestamp")
    if history_df.empty:
        raise RuntimeError("Load history is empty")

    load_history = history_df["total_load_mw"].astype(float).tolist()
    calendar = USFederalHolidayCalendar()
    forecast_index = pd.date_range(
        history_df["timestamp"].iloc[-1] + timedelta(hours=1),
        periods=24,
        freq="h",
    )
    holidays = set(calendar.holidays(start=forecast_index.min(), end=forecast_index.max()).normalize())
    records = []

    for dt in forecast_index:
        hour = dt.hour
        recent = pd.Series(load_history, dtype=float)
        record = {
            "hour_of_day": hour,
            "day_of_week": dt.weekday(),
            "month": dt.month,
            "is_weekend": int(dt.weekday() >= 5),
            "is_peak_hour": int(7 <= hour <= 22),
            "season": {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}[dt.month],
            "is_holiday": int(dt.normalize() in holidays),
        }

        for lag in (1, 2, 3, 6, 12, 24, 48, 168):
            record[f"load_lag_{lag}h"] = float(load_history[-lag])

        for window in (6, 12, 24, 168):
            window_values = recent.iloc[-window:]
            record[f"load_roll_{window}h_mean"] = float(window_values.mean())
            record[f"load_roll_{window}h_std"] = float(window_values.std(ddof=1) if len(window_values) > 1 else 0.0)
            record[f"load_roll_{window}h_min"] = float(window_values.min())
            record[f"load_roll_{window}h_max"] = float(window_values.max())

        record["load_change_1h"] = float(load_history[-1] - load_history[-2])
        record["load_change_24h"] = float(load_history[-1] - load_history[-25])
        record["load_roc_1h"] = float(record["load_change_1h"] / max(load_history[-2], 1000.0))
        records.append({col: record.get(col, 0.0) for col in FEATURE_COLS})

        baseline = 0.50 * record["load_lag_24h"] + 0.35 * record["load_lag_1h"] + 0.15 * record["load_roll_24h_mean"]
        if record["is_peak_hour"]:
            baseline += 1200.0
        if record["is_weekend"]:
            baseline -= 900.0
        load_history.append(float(max(baseline, 0.0)))

    return pd.DataFrame(records, columns=FEATURE_COLS)


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


def _latest_complete_delivery_rows(features_df: pd.DataFrame) -> pd.DataFrame:
    counts = features_df.groupby("delivery_date")["hour_ending"].nunique().sort_index()
    complete_dates = counts[counts >= 24]
    if complete_dates.empty:
        return pd.DataFrame(columns=features_df.columns)
    latest_date = complete_dates.index[-1]
    rows = features_df[features_df["delivery_date"] == latest_date].copy()
    return rows.sort_values("hour_ending").head(24)


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


# ---------------------------------------------------------------------------
# Mining Dispatch
# ---------------------------------------------------------------------------

@app.get("/dispatch/mining/schedule", tags=["Dispatch"])
async def dispatch_mining_schedule(
    settlement_point: str = Query(default="HB_WEST", description="Settlement point for DAM prices"),
):
    """
    Today's mining ON/OFF dispatch schedule.

    Combines DAM predictions, spike alerts, and BESS schedule to compute
    optimal hours to run or curtail mining operations.
    """
    normalized_sp = _normalize_settlement_point(settlement_point)

    dam_predictor = get_dam_v2_predictor()
    if not dam_predictor.is_ready():
        raise HTTPException(status_code=503, detail="DAM models not loaded")
    if normalized_sp.lower() not in dam_predictor.available_settlement_points():
        raise HTTPException(
            status_code=400,
            detail=f"No DAM model for '{settlement_point}'. Available: {dam_predictor.available_settlement_points()}",
        )

    try:
        features_df = _fetch_and_compute_features(normalized_sp)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Data pipeline failed: {e}")

    target_rows = _latest_complete_delivery_rows(features_df)
    if target_rows.empty:
        raise HTTPException(status_code=404, detail="Insufficient data for dispatch")

    try:
        sp_key = normalized_sp.lower()

        # DAM predictions
        dam_predictions = dam_predictor.predict(target_rows, sp_key)
        dam_prices = [
            {"hour_ending": p.hour_ending, "predicted_price": p.predicted_price}
            for p in dam_predictions[:24]
        ]

        # Spike alerts (best-effort)
        spike_alerts = None
        try:
            spike_pred = get_spike_predictor()
            if spike_pred.is_ready():
                alerts = spike_pred.predict(target_rows, sp_key)
                spike_alerts = [
                    {
                        "hour_ending": i + 1,
                        "spike_probability": a.spike_probability,
                        "is_spike": a.is_spike,
                    }
                    for i, a in enumerate(alerts)
                ]
        except Exception as exc:
            log.warning("Spike predictor unavailable for dispatch: %s", exc)

        # BESS schedule (best-effort)
        bess_schedule = None
        try:
            bess = get_bess_predictor()
            if bess.is_ready():
                bess_prices = [p["predicted_price"] for p in dam_prices]
                if len(bess_prices) == 24:
                    bess_result = bess.optimize(bess_prices)
                    bess_schedule = [
                        {"hour_ending": e.hour_ending, "action": e.action}
                        for e in bess_result.schedule
                    ]
        except Exception as exc:
            log.warning("BESS optimizer unavailable for dispatch: %s", exc)

        config = load_dispatch_config()
        config.setdefault("mining", {})["settlement_point"] = normalized_sp
        schedule = compute_dispatch(dam_prices, spike_alerts, bess_schedule, config)

        return {
            "status": "success",
            "settlement_point": normalized_sp,
            "generated_at": schedule.generated_at,
            "date": schedule.date,
            "schedule": [
                {
                    "hour_ending": f"{ha.hour_ending:02d}:00",
                    "dam_price": ha.dam_price,
                    "action": ha.action,
                    "reason": ha.reason,
                    "spike_probability": ha.spike_probability,
                    "bess_action": ha.bess_action,
                }
                for ha in schedule.hours
            ],
            "summary": {
                "hours_to_run": schedule.hours_to_run,
                "hours_to_curtail": schedule.hours_to_curtail,
                "expected_cost_savings": schedule.expected_cost_savings,
                "always_on_cost": schedule.always_on_cost,
                "dispatch_cost": schedule.dispatch_cost,
                "peak_price": schedule.peak_price,
                "avg_on_price": schedule.avg_on_price,
                "spike_hours": schedule.spike_hours,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dispatch/mining/savings", tags=["Dispatch"])
async def dispatch_mining_savings(
    settlement_point: str = Query(default="HB_WEST", description="Settlement point"),
):
    """
    Estimated savings from dispatch vs always-on mining.

    Returns cost comparison and savings breakdown.
    """
    result = await dispatch_mining_schedule(settlement_point)
    summary = result["summary"]
    return {
        "status": "success",
        "settlement_point": result["settlement_point"],
        "date": result["date"],
        "always_on_cost": summary["always_on_cost"],
        "dispatch_cost": summary["dispatch_cost"],
        "expected_cost_savings": summary["expected_cost_savings"],
        "hours_to_run": summary["hours_to_run"],
        "hours_to_curtail": summary["hours_to_curtail"],
        "peak_price": summary["peak_price"],
        "avg_on_price": summary["avg_on_price"],
        "savings_pct": round(
            summary["expected_cost_savings"] / summary["always_on_cost"] * 100, 1
        ) if summary["always_on_cost"] > 0 else 0.0,
    }


class AlertConfigRequest(BaseModel):
    chat_ids: Optional[List[str]] = None
    spike_alert_threshold: Optional[float] = None
    spike_cooldown_minutes: Optional[int] = None
    bot_token: Optional[str] = None


@app.post("/dispatch/alerts/config", tags=["Dispatch"])
async def configure_alerts(req: AlertConfigRequest):
    """
    Configure alert preferences for the Telegram alert service.

    Update chat IDs, spike thresholds, cooldown periods, or bot token.
    """
    svc = get_alert_service()
    updated = svc.update_config(
        chat_ids=req.chat_ids,
        spike_alert_threshold=req.spike_alert_threshold,
        spike_cooldown_minutes=req.spike_cooldown_minutes,
        bot_token=req.bot_token,
    )
    return {"status": "success", "config": updated}


# ---------------------------------------------------------------------------
# BESS Dispatch Signals
# ---------------------------------------------------------------------------

@app.get("/dispatch/bess/daily-signals", tags=["Dispatch"])
async def dispatch_bess_daily_signals(
    settlement_point: str = Query(default="HB_WEST", description="Settlement point"),
):
    """
    Today's BESS charge/discharge recommendations.

    Combines DAM predictions, BESS LP optimizer, spike alerts, RTM volatility,
    and mining dispatch to produce risk-adjusted arbitrage signals.
    """
    normalized_sp = _normalize_settlement_point(settlement_point)

    dam_predictor = get_dam_v2_predictor()
    if not dam_predictor.is_ready():
        raise HTTPException(status_code=503, detail="DAM models not loaded")
    if normalized_sp.lower() not in dam_predictor.available_settlement_points():
        raise HTTPException(
            status_code=400,
            detail=f"No DAM model for '{settlement_point}'.",
        )

    bess = get_bess_predictor()
    if not bess.is_ready():
        raise HTTPException(status_code=503, detail="BESS optimizer not loaded")

    try:
        features_df = _fetch_and_compute_features(normalized_sp)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Data pipeline failed: {e}")

    target_rows = _latest_complete_delivery_rows(features_df)
    if target_rows.empty:
        raise HTTPException(status_code=404, detail="Insufficient data for BESS signals")

    try:
        sp_key = normalized_sp.lower()

        # DAM predictions
        dam_predictions = dam_predictor.predict(target_rows, sp_key)
        dam_prices = [
            {"hour_ending": p.hour_ending, "predicted_price": p.predicted_price}
            for p in dam_predictions[:24]
        ]

        # BESS LP schedule
        bess_prices = [p["predicted_price"] for p in dam_prices]
        if len(bess_prices) < 24:
            raise HTTPException(status_code=500, detail=f"Only {len(bess_prices)} DAM prices, need 24")
        bess_result = bess.optimize(bess_prices)
        bess_schedule = [
            {
                "hour_ending": e.hour_ending,
                "action": e.action,
                "power_mw": e.power_mw,
                "soc_pct": e.soc_pct,
                "dam_price": e.dam_price,
            }
            for e in bess_result.schedule
        ]

        # Spike alerts (best-effort)
        spike_alerts = None
        try:
            spike_pred = get_spike_predictor()
            if spike_pred.is_ready():
                alerts = spike_pred.predict(target_rows, sp_key)
                spike_alerts = [
                    {"hour_ending": i + 1, "spike_probability": a.spike_probability, "is_spike": a.is_spike}
                    for i, a in enumerate(alerts)
                ]
        except Exception as exc:
            log.warning("Spike predictor unavailable for BESS signals: %s", exc)

        # Mining dispatch (best-effort, for coordination)
        mining_schedule = None
        try:
            config = load_dispatch_config()
            config.setdefault("mining", {})["settlement_point"] = normalized_sp
            mining_result = compute_dispatch(dam_prices, spike_alerts, bess_schedule=[
                {"hour_ending": b["hour_ending"], "action": b["action"]} for b in bess_schedule
            ], config=config)
            mining_schedule = [
                {"hour_ending": ha.hour_ending, "action": ha.action}
                for ha in mining_result.hours
            ]
        except Exception as exc:
            log.warning("Mining dispatch unavailable for BESS signals: %s", exc)

        signals = generate_daily_signals(
            dam_prices=dam_prices,
            bess_schedule=bess_schedule,
            spike_alerts=spike_alerts,
            mining_schedule=mining_schedule,
            settlement_point=normalized_sp,
        )

        # Record PnL
        try:
            record_daily_pnl(signals)
        except Exception as exc:
            log.warning("Failed to record BESS PnL: %s", exc)

        return {
            "status": "success",
            "settlement_point": normalized_sp,
            "date": signals.date,
            "generated_at": signals.generated_at,
            "summary": {
                "total_revenue_estimate": signals.total_revenue_estimate,
                "risk_adjusted_revenue": signals.risk_adjusted_revenue,
                "charge_hours": signals.charge_hours,
                "discharge_hours": signals.discharge_hours,
                "idle_hours": signals.idle_hours,
                "peak_discharge_price": signals.peak_discharge_price,
                "avg_charge_price": signals.avg_charge_price,
                "spike_hold_hours": signals.spike_hold_hours,
            },
            "signals": [
                {
                    "hour_ending": f"{s.hour_ending:02d}:00",
                    "action": s.action,
                    "power_mw": s.power_mw,
                    "soc_pct": s.soc_pct,
                    "dam_price": s.dam_price,
                    "rtm_volatility": s.rtm_volatility,
                    "spike_probability": s.spike_probability,
                    "revenue_estimate": s.revenue_estimate,
                    "risk_flag": s.risk_flag,
                    "mining_curtail": s.mining_curtail,
                }
                for s in signals.signals
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dispatch/bess/pnl", tags=["Dispatch"])
async def dispatch_bess_pnl(
    days: int = Query(default=7, ge=1, le=90, description="Number of days to look back"),
    settlement_point: str = Query(default="HB_WEST", description="Settlement point"),
):
    """
    Rolling BESS PnL with daily breakdown.

    Returns per-day charge cost, discharge revenue, cycles, degradation,
    and net PnL for the requested window.
    """
    records = get_rolling_pnl(days=days, settlement_point=settlement_point)

    total_pnl = sum(r.net_pnl for r in records)
    total_cycles = sum(r.cycles for r in records)
    total_charge_cost = sum(r.charge_cost for r in records)
    total_discharge_rev = sum(r.discharge_revenue for r in records)

    return {
        "status": "success",
        "settlement_point": settlement_point,
        "days_requested": days,
        "days_available": len(records),
        "summary": {
            "total_pnl": round(total_pnl, 2),
            "total_cycles": round(total_cycles, 3),
            "total_charge_cost": round(total_charge_cost, 2),
            "total_discharge_revenue": round(total_discharge_rev, 2),
            "avg_daily_pnl": round(total_pnl / len(records), 2) if records else 0.0,
        },
        "daily": [
            {
                "date": r.date,
                "projected_revenue": r.projected_revenue,
                "actual_revenue": r.actual_revenue,
                "charge_cost": r.charge_cost,
                "discharge_revenue": r.discharge_revenue,
                "cycles": r.cycles,
                "degradation_cost": r.degradation_cost,
                "net_pnl": r.net_pnl,
            }
            for r in records
        ],
    }


@app.get("/dispatch/bess/risk", tags=["Dispatch"])
async def dispatch_bess_risk(
    days: int = Query(default=30, ge=7, le=365, description="Risk window in days"),
    settlement_point: str = Query(default="HB_WEST", description="Settlement point"),
):
    """
    BESS risk metrics: VaR, max drawdown, Sharpe ratio, win rate.

    Computed from historical PnL over the requested window.
    """
    metrics = compute_risk_metrics(days=days, settlement_point=settlement_point)
    return {
        "status": "success",
        "settlement_point": settlement_point,
        "risk": {
            "days": metrics.days,
            "total_pnl": metrics.total_pnl,
            "avg_daily_pnl": metrics.avg_daily_pnl,
            "var_95": metrics.var_95,
            "max_drawdown": metrics.max_drawdown,
            "win_rate": metrics.win_rate,
            "sharpe_ratio": metrics.sharpe_ratio,
            "best_day": metrics.best_day,
            "worst_day": metrics.worst_day,
            "volatility": metrics.volatility,
        },
    }


# ---------------------------------------------------------------------------
# Admin endpoints — API key management
# ---------------------------------------------------------------------------


class CreateKeyRequest(BaseModel):
    name: str
    tier: str = "free"


class KeyResponse(BaseModel):
    id: int
    key_prefix: str
    name: str
    tier: str
    active: bool
    created_at: str
    last_used: Optional[str]
    request_count: int


@app.post("/admin/keys", tags=["Admin"], dependencies=[Depends(_require_admin)])
async def create_api_key(body: CreateKeyRequest):
    """Create a new API key. The raw key is returned only once."""
    manager = get_key_manager()
    try:
        raw_key = manager.create_key(body.name, body.tier)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "created", "api_key": raw_key, "name": body.name, "tier": body.tier}


@app.get("/admin/keys", tags=["Admin"], dependencies=[Depends(_require_admin)])
async def list_api_keys():
    """List all API keys with usage stats."""
    manager = get_key_manager()
    keys = manager.list_keys()
    return {
        "status": "success",
        "keys": [
            KeyResponse(
                id=k.id, key_prefix=k.key_prefix, name=k.name, tier=k.tier,
                active=k.active, created_at=k.created_at, last_used=k.last_used,
                request_count=k.request_count,
            ).model_dump()
            for k in keys
        ],
    }


@app.delete("/admin/keys/{key_id}", tags=["Admin"], dependencies=[Depends(_require_admin)])
async def revoke_api_key(key_id: int):
    """Revoke an API key."""
    manager = get_key_manager()
    if not manager.revoke_key(key_id):
        raise HTTPException(status_code=404, detail="Key not found")
    return {"status": "revoked", "key_id": key_id}


@app.get("/admin/usage", tags=["Admin"], dependencies=[Depends(_require_admin)])
async def get_usage_analytics(
    key_id: Optional[int] = Query(None, description="Filter by key ID"),
    days: int = Query(7, ge=1, le=90, description="Number of days to look back"),
):
    """Usage analytics for API keys."""
    manager = get_key_manager()
    usage = manager.get_usage(key_id=key_id, days=days)
    return {"status": "success", "days": days, "total_requests": len(usage), "usage": usage}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8011)
