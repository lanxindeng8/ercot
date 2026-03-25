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

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .auth.api_keys import get_key_manager
from .routers import system, models, predictions, dispatch, admin, accuracy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


app = FastAPI(
    title="TrueFlux Prediction Service",
    description=(
        "API for ERCOT electricity price predictions.\n\n"
        "**Models:**\n"
        "- **DAM V2** — Next-day DAM price (LightGBM, per-settlement-point checkpoints)\n"
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
# Auth middleware
# ---------------------------------------------------------------------------

ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")
SKIP_AUTH_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}


@app.middleware("http")
async def api_key_auth_middleware(request: Request, call_next):
    """Authenticate requests via X-API-Key header. Skip auth for health/docs."""
    path = request.url.path

    # Skip auth for health, docs, admin, and localhost requests (internal runner)
    if path in SKIP_AUTH_PATHS or path.startswith("/admin"):
        response = await call_next(request)
        return response

    client_host = request.client.host if request.client else None
    if client_host in ("127.0.0.1", "::1", "localhost"):
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


# ---------------------------------------------------------------------------
# Include routers
# ---------------------------------------------------------------------------

app.include_router(system.router)
app.include_router(models.router)
app.include_router(predictions.router)
app.include_router(dispatch.router)
app.include_router(admin.router)
app.include_router(accuracy.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8011)
