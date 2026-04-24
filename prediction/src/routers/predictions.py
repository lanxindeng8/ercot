"""Prediction endpoints: DAM, RTM, Delta Spread, Spike, Wind, Load, BESS."""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from ..models.delta_spread import get_predictor
from ..models.dam_v2_predictor import get_dam_v2_predictor
from ..models.rtm_predictor import get_rtm_predictor
from ..models.spike_predictor import get_spike_predictor
from ..models.spike_v2_predictor import get_spike_v2_predictor
from ..models.wind_predictor import get_wind_predictor
from ..models.load_predictor import get_load_predictor
from ..models.bess_predictor import get_bess_predictor
from ..helpers import (
    fetch_and_compute_features,
    normalize_settlement_point,
    normalize_delta_settlement_point,
    latest_complete_delivery_rows,
    parse_horizons,
    build_wind_features,
    build_load_features,
)

router = APIRouter(tags=["Predictions"])


# ---------------------------------------------------------------------------
# DAM predictions
# ---------------------------------------------------------------------------

@router.get("/predictions/dam/next-day")
async def predict_dam_next_day(
    settlement_point: str = Query(default="HB_WEST", description="Settlement point (e.g. HB_WEST, HB_HOUSTON)"),
    target_date: Optional[str] = Query(default=None, description="Target date YYYY-MM-DD (default: tomorrow)"),
):
    """
    Next-day DAM price predictions (24 hours).

    Uses LightGBM models trained per settlement point with 80 features.
    Fetches recent DAM + RTM data from InfluxDB, computes features, and predicts.

    Accepts any configured ERCOT settlement point. If a checkpoint has not been trained
    for the requested point yet, the endpoint returns 404.
    """
    normalized_sp = normalize_settlement_point(settlement_point)

    predictor = get_dam_v2_predictor()
    if not predictor.is_ready():
        raise HTTPException(status_code=503, detail="DAM V2 models not loaded")

    sp_key = normalized_sp.lower()
    if not predictor.has_model(sp_key):
        raise HTTPException(
            status_code=404,
            detail=predictor.missing_model_message(normalized_sp),
        )

    target_dt = None
    if target_date:
        try:
            target_dt = datetime.strptime(target_date, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    try:
        features_df = fetch_and_compute_features(normalized_sp)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Data pipeline failed: {e}")

    # Filter to the target date (or last available day for "tomorrow" prediction)
    if target_dt:
        target_rows = features_df[features_df["delivery_date"] == str(target_dt)]
    else:
        # Use the last full day of features as a proxy for tomorrow's prediction
        target_rows = latest_complete_delivery_rows(features_df)

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
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# RTM predictions
# ---------------------------------------------------------------------------

@router.get("/predictions/rtm/next-day")
async def predict_rtm_next_day(
    settlement_point: str = Query(default="HB_WEST", description="Settlement point (e.g. HB_WEST, LZ_WEST)"),
    target_date: Optional[str] = Query(default=None, description="Target date YYYY-MM-DD (default: tomorrow)"),
):
    """
    Next-day RTM price predictions (24 hours).

    Uses the 1h-ahead LightGBM model on each hour's features to produce
    hourly RTM price forecasts, mirroring the DAM next-day endpoint.
    """
    normalized_sp = normalize_settlement_point(settlement_point)

    predictor = get_rtm_predictor()
    if not predictor.is_ready():
        raise HTTPException(status_code=503, detail="RTM models not loaded")

    sp_key = normalized_sp.lower()
    if not predictor.has_model(sp_key):
        raise HTTPException(
            status_code=404,
            detail=predictor.missing_model_message(normalized_sp),
        )

    target_dt = None
    if target_date:
        try:
            target_dt = datetime.strptime(target_date, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    try:
        features_df = fetch_and_compute_features(normalized_sp)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Data pipeline failed: {e}")

    if target_dt:
        target_rows = features_df[features_df["delivery_date"] == str(target_dt)]
    else:
        target_rows = latest_complete_delivery_rows(features_df)

    if target_rows.empty:
        raise HTTPException(status_code=404, detail="Insufficient data to generate predictions for target date")

    try:
        # Add RTM extra features and run 1h model on each hour
        df = predictor.add_rtm_features(target_rows)
        df = df.fillna(0)

        from ..models.rtm_predictor import FEATURE_COLS as RTM_FEATURE_COLS
        missing = [col for col in RTM_FEATURE_COLS if col not in df.columns]
        if missing:
            raise ValueError(f"Missing RTM feature columns: {missing}")

        model = predictor.models[sp_key].get("1h")
        if model is None:
            raise HTTPException(status_code=404, detail=f"No 1h RTM model for {normalized_sp}")

        X = df[RTM_FEATURE_COLS]
        preds = model.predict(X)
        delivery_date = target_rows["delivery_date"].iloc[0]

        return {
            "status": "success",
            "model": "RTM 1h-ahead LightGBM",
            "settlement_point": normalized_sp,
            "delivery_date": delivery_date,
            "generated_at": datetime.utcnow().isoformat(),
            "predictions": [
                {
                    "hour_ending": f"{int(df.iloc[i].get('hour_of_day', i + 1)):02d}:00",
                    "predicted_price": round(float(preds[i]), 2),
                }
                for i in range(len(preds))
            ],
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/predictions/rtm")
async def predict_rtm(
    settlement_point: str = Query(default="HB_WEST", description="Settlement point"),
    horizons: Optional[str] = Query(default=None, description="Comma-separated horizons: 1h,4h,24h (default: all)"),
):
    """
    Real-Time Market price predictions at multiple horizons.

    Returns predicted RTM LMP price for 1-hour, 4-hour, and 24-hour ahead.
    Fetches recent market data, computes features, and runs inference.

    Accepts any configured ERCOT settlement point. Missing checkpoints return 404.
    """
    normalized_sp = normalize_settlement_point(settlement_point)

    predictor = get_rtm_predictor()
    if not predictor.is_ready():
        raise HTTPException(status_code=503, detail="RTM models not loaded")

    sp_key = normalized_sp.lower()
    if not predictor.has_model(sp_key):
        raise HTTPException(
            status_code=404,
            detail=predictor.missing_model_message(normalized_sp),
        )

    horizon_list = parse_horizons(horizons)

    try:
        features_df = fetch_and_compute_features(normalized_sp)
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
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Delta Spread predictions
# ---------------------------------------------------------------------------

@router.get("/predictions/delta-spread")
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
        normalized_sp = normalize_delta_settlement_point(settlement_point)
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

@router.get("/predictions/spike")
async def predict_spike(
    settlement_point: str = Query(default="HB_WEST", description="Settlement point"),
):
    """
    Spike detection alert for the next hour.

    Predicts whether RTM price will spike (> max($100, 3x rolling 24h mean))
    in the next hour. Returns probability and confidence level.

    **Performance**: Precision=0.886, Recall=1.0, F1=0.939

    Accepts any configured ERCOT settlement point. Missing checkpoints return 404.
    """
    normalized_sp = normalize_settlement_point(settlement_point)

    predictor = get_spike_predictor()
    if not predictor.is_ready():
        raise HTTPException(status_code=503, detail="Spike models not loaded")

    sp_key = normalized_sp.lower()
    if not predictor.has_model(sp_key):
        raise HTTPException(
            status_code=404,
            detail=predictor.missing_model_message(normalized_sp),
        )

    try:
        features_df = fetch_and_compute_features(normalized_sp)
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
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Spike V2 — LightGBM zone-level spike detection (14 SPs)
# ---------------------------------------------------------------------------

@router.get("/predict/spike/v2/all")
async def predict_spike_v2_all():
    """
    Spike V2 predictions for all 14 settlement points, ranked by probability.
    """
    predictor = get_spike_v2_predictor()
    if not predictor.is_ready():
        raise HTTPException(status_code=503, detail="Spike V2 models not loaded")
    results = predictor.predict_all()
    return {
        "status": "success",
        "model_version": "v2_lead60",
        "count": len(results),
        "predictions": results,
    }


@router.get("/predict/spike/v2/alerts")
async def predict_spike_v2_alerts():
    """
    Spike V2 alerts — only settlement points with probability >= 0.3 (medium+ risk).
    """
    predictor = get_spike_v2_predictor()
    if not predictor.is_ready():
        raise HTTPException(status_code=503, detail="Spike V2 models not loaded")
    alerts = predictor.predict_alerts(threshold=0.3)
    return {
        "status": "success",
        "model_version": "v2_lead60",
        "count": len(alerts),
        "alerts": alerts,
    }


@router.get("/predict/spike/v2/{sp}")
async def predict_spike_v2(sp: str):
    """
    Spike V2 prediction for a single settlement point.

    Uses LightGBM models trained per-zone on lead_spike_60 target.
    Returns probability, risk level, regime, and top feature drivers.
    """
    normalized_sp = normalize_settlement_point(sp)
    predictor = get_spike_v2_predictor()
    if not predictor.is_ready():
        raise HTTPException(status_code=503, detail="Spike V2 models not loaded")
    if not predictor.has_model(normalized_sp):
        raise HTTPException(
            status_code=404,
            detail=f"No spike V2 model for '{normalized_sp}'. Available: {predictor.available_settlement_points()}",
        )
    result = predictor.predict(normalized_sp)
    if result.get("error"):
        raise HTTPException(status_code=502, detail=result["error"])
    return {"status": "success", **result}


# ---------------------------------------------------------------------------
# Wind predictions
# ---------------------------------------------------------------------------

@router.get("/predictions/wind")
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
        features_df = build_wind_features()
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

@router.get("/predictions/load")
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
        features_df = build_load_features()
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

@router.get("/predictions/bess")
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
    normalized_sp = normalize_settlement_point(settlement_point)

    # Get DAM predictions to use as input prices
    dam_predictor = get_dam_v2_predictor()
    if not dam_predictor.is_ready():
        raise HTTPException(status_code=503, detail="DAM models not loaded (required for BESS)")
    if not dam_predictor.has_model(normalized_sp):
        raise HTTPException(
            status_code=404,
            detail=dam_predictor.missing_model_message(normalized_sp),
        )

    bess = get_bess_predictor()
    if not bess.is_ready():
        raise HTTPException(status_code=503, detail="BESS optimizer not loaded")

    try:
        features_df = fetch_and_compute_features(normalized_sp)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Data pipeline failed: {e}")

    # Get the latest complete day of features for DAM prediction
    target_rows = latest_complete_delivery_rows(features_df)

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
