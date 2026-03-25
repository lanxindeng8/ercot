"""Model info endpoints."""

from fastapi import APIRouter

from ..models.delta_spread import get_predictor
from ..models.dam_v2_predictor import get_dam_v2_predictor
from ..models.rtm_predictor import get_rtm_predictor
from ..models.spike_predictor import get_spike_predictor
from ..models.wind_predictor import get_wind_predictor
from ..models.load_predictor import get_load_predictor
from ..models.bess_predictor import get_bess_predictor
from ..schemas import ModelInfoResponse

router = APIRouter(tags=["Models"])


@router.get("/models/dam/info", response_model=ModelInfoResponse)
async def dam_model_info():
    """DAM V2 model metadata."""
    p = get_dam_v2_predictor()
    return ModelInfoResponse(
        model_name="dam_v2",
        status="loaded" if p.is_ready() else "not_loaded",
        info=p.get_model_info(),
    )


@router.get("/models/rtm/info", response_model=ModelInfoResponse)
async def rtm_model_info():
    """RTM multi-horizon model metadata."""
    p = get_rtm_predictor()
    return ModelInfoResponse(
        model_name="rtm",
        status="loaded" if p.is_ready() else "not_loaded",
        info=p.get_model_info(),
    )


@router.get("/models/delta-spread/info", response_model=ModelInfoResponse)
async def delta_spread_info():
    """Delta Spread model metadata."""
    p = get_predictor()
    return ModelInfoResponse(
        model_name="delta_spread",
        status="loaded",
        info=p.get_model_info(),
    )


@router.get("/models/spike/info", response_model=ModelInfoResponse)
async def spike_model_info():
    """Spike detection model metadata."""
    p = get_spike_predictor()
    return ModelInfoResponse(
        model_name="spike",
        status="loaded" if p.is_ready() else "not_loaded",
        info=p.get_model_info(),
    )


@router.get("/models/wind/info", response_model=ModelInfoResponse)
async def wind_model_info():
    """Wind generation forecast model metadata."""
    p = get_wind_predictor()
    return ModelInfoResponse(
        model_name="wind",
        status="loaded" if p.is_ready() else "not_loaded",
        info=p.get_model_info(),
    )


@router.get("/models/load/info", response_model=ModelInfoResponse)
async def load_model_info():
    """Load forecast model metadata."""
    p = get_load_predictor()
    return ModelInfoResponse(
        model_name="load",
        status="loaded" if p.is_ready() else "not_loaded",
        info=p.get_model_info(),
    )


@router.get("/models/bess/info", response_model=ModelInfoResponse)
async def bess_model_info():
    """BESS optimizer metadata."""
    p = get_bess_predictor()
    return ModelInfoResponse(
        model_name="bess",
        status="loaded" if p.is_ready() else "not_loaded",
        info=p.get_model_info(),
    )
