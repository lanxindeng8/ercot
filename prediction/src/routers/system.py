"""System endpoints: health check, settlement points."""

from datetime import datetime

from fastapi import APIRouter

from ..helpers import HUB_SETTLEMENT_POINTS, LOAD_ZONE_SETTLEMENT_POINTS
from ..config import SETTLEMENT_POINTS
from ..models.delta_spread import get_predictor
from ..models.dam_v2_predictor import get_dam_v2_predictor
from ..models.rtm_predictor import get_rtm_predictor
from ..models.spike_predictor import get_spike_predictor
from ..models.wind_predictor import get_wind_predictor
from ..models.load_predictor import get_load_predictor
from ..models.bess_predictor import get_bess_predictor
from ..schemas import HealthResponse, ModelStatus

router = APIRouter(tags=["System"])


@router.get("/health", response_model=HealthResponse)
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


@router.get("/settlement-points")
async def list_settlement_points():
    """List available ERCOT settlement points."""
    return {
        "hubs": HUB_SETTLEMENT_POINTS,
        "load_zones": LOAD_ZONE_SETTLEMENT_POINTS,
        "all": SETTLEMENT_POINTS,
    }
