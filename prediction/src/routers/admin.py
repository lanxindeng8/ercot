"""Admin endpoints: API key management and usage analytics."""

import os
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query

from ..auth.api_keys import get_key_manager
from ..schemas import CreateKeyRequest, KeyResponse

ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")


def _require_admin(x_admin_token: str = Header(...)):
    """Dependency that enforces ADMIN_TOKEN for /admin/* endpoints."""
    if not ADMIN_TOKEN:
        raise HTTPException(status_code=503, detail="Admin token not configured (set ADMIN_TOKEN env var)")
    if x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid admin token")


router = APIRouter(tags=["Admin"])


@router.post("/admin/keys", dependencies=[Depends(_require_admin)])
async def create_api_key(body: CreateKeyRequest):
    """Create a new API key. The raw key is returned only once."""
    manager = get_key_manager()
    try:
        raw_key = manager.create_key(body.name, body.tier)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "created", "api_key": raw_key, "name": body.name, "tier": body.tier}


@router.get("/admin/keys", dependencies=[Depends(_require_admin)])
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


@router.delete("/admin/keys/{key_id}", dependencies=[Depends(_require_admin)])
async def revoke_api_key(key_id: int):
    """Revoke an API key."""
    manager = get_key_manager()
    if not manager.revoke_key(key_id):
        raise HTTPException(status_code=404, detail="Key not found")
    return {"status": "revoked", "key_id": key_id}


@router.get("/admin/usage", dependencies=[Depends(_require_admin)])
async def get_usage_analytics(
    key_id: Optional[int] = Query(None, description="Filter by key ID"),
    days: int = Query(7, ge=1, le=90, description="Number of days to look back"),
):
    """Usage analytics for API keys."""
    manager = get_key_manager()
    usage = manager.get_usage(key_id=key_id, days=days)
    return {"status": "success", "days": days, "total_requests": len(usage), "usage": usage}
