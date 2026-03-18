"""Tests for API key management, auth middleware, and rate limiting."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from prediction.src.auth.api_keys import APIKeyManager, TIER_LIMITS, reset_key_manager


@pytest.fixture
def manager(tmp_path):
    """Fresh APIKeyManager with temp DB."""
    db_path = tmp_path / "test_keys.db"
    return APIKeyManager(db_path)


@pytest.fixture
def client(tmp_path):
    """TestClient with auth configured."""
    reset_key_manager()
    db_path = tmp_path / "test_keys.db"
    mgr = APIKeyManager(db_path)

    with patch.dict(os.environ, {"ADMIN_TOKEN": "test-admin-secret"}):
        with patch("prediction.src.main.get_key_manager", return_value=mgr):
            with patch("prediction.src.main.ADMIN_TOKEN", "test-admin-secret"):
                from prediction.src.main import app
                yield TestClient(app), mgr


# ---------------------------------------------------------------------------
# APIKeyManager unit tests
# ---------------------------------------------------------------------------


class TestAPIKeyManager:
    def test_create_and_validate_key(self, manager):
        raw = manager.create_key("test-app", "free")
        assert raw.startswith("tf_")
        key = manager.validate_key(raw)
        assert key is not None
        assert key.name == "test-app"
        assert key.tier == "free"
        assert key.active is True

    def test_invalid_key_returns_none(self, manager):
        assert manager.validate_key("tf_bogus") is None

    def test_invalid_tier_raises(self, manager):
        with pytest.raises(ValueError, match="Invalid tier"):
            manager.create_key("bad", "platinum")

    def test_revoke_key(self, manager):
        raw = manager.create_key("revokeme", "pro")
        key = manager.validate_key(raw)
        assert key is not None
        assert manager.revoke_key(key.id) is True
        assert manager.validate_key(raw) is None

    def test_revoke_nonexistent(self, manager):
        assert manager.revoke_key(9999) is False

    def test_list_keys(self, manager):
        manager.create_key("a", "free")
        manager.create_key("b", "pro")
        keys = manager.list_keys()
        assert len(keys) == 2
        names = {k.name for k in keys}
        assert names == {"a", "b"}

    def test_rate_limit_free_tier(self, manager):
        raw = manager.create_key("rate-test", "free")
        key = manager.validate_key(raw)
        # Free tier: 100/day
        for _ in range(100):
            assert manager.check_rate_limit(key) is True
        assert manager.check_rate_limit(key) is False

    def test_rate_limit_enterprise_unlimited(self, manager):
        raw = manager.create_key("enterprise-test", "enterprise")
        key = manager.validate_key(raw)
        for _ in range(200):
            assert manager.check_rate_limit(key) is True

    def test_usage_tracking(self, manager):
        raw = manager.create_key("usage-test", "free")
        key = manager.validate_key(raw)
        manager.record_usage(key, "/predict/dam", 200)
        manager.record_usage(key, "/predict/rtm", 200)

        usage = manager.get_usage(key_id=key.id, days=1)
        assert len(usage) == 2

        # Check request count updated
        updated = manager.validate_key(raw)
        assert updated.request_count == 2

    def test_usage_all_keys(self, manager):
        r1 = manager.create_key("a", "free")
        r2 = manager.create_key("b", "free")
        k1 = manager.validate_key(r1)
        k2 = manager.validate_key(r2)
        manager.record_usage(k1, "/predict/dam", 200)
        manager.record_usage(k2, "/predict/rtm", 200)

        usage = manager.get_usage(days=1)
        assert len(usage) == 2


# ---------------------------------------------------------------------------
# Middleware / endpoint integration tests
# ---------------------------------------------------------------------------


class TestAuthMiddleware:
    def test_health_no_auth_required(self, client):
        c, _ = client
        resp = c.get("/health")
        # May fail model loading but should NOT be 401
        assert resp.status_code != 401

    def test_docs_no_auth_required(self, client):
        c, _ = client
        resp = c.get("/docs")
        assert resp.status_code != 401

    def test_missing_api_key_returns_401(self, client):
        c, _ = client
        resp = c.get("/predict/dam")
        assert resp.status_code == 401
        assert "Missing X-API-Key" in resp.json()["detail"]

    def test_invalid_api_key_returns_401(self, client):
        c, _ = client
        resp = c.get("/predict/dam", headers={"X-API-Key": "tf_invalid"})
        assert resp.status_code == 401
        assert "Invalid or revoked" in resp.json()["detail"]

    def test_valid_api_key_passes_auth(self, client):
        c, mgr = client
        raw = mgr.create_key("integration", "pro")
        resp = c.get("/predict/dam", headers={"X-API-Key": raw})
        # Should pass auth (may get 422/500 from missing params, but not 401)
        assert resp.status_code != 401

    def test_revoked_key_rejected(self, client):
        c, mgr = client
        raw = mgr.create_key("revoked", "free")
        key = mgr.validate_key(raw)
        mgr.revoke_key(key.id)
        resp = c.get("/predict/dam", headers={"X-API-Key": raw})
        assert resp.status_code == 401

    def test_rate_limited_returns_429(self, client):
        c, mgr = client
        raw = mgr.create_key("limited", "free")
        # Exhaust free tier (100/day)
        key = mgr.validate_key(raw)
        for _ in range(100):
            mgr.check_rate_limit(key)
        resp = c.get("/predict/dam", headers={"X-API-Key": raw})
        assert resp.status_code == 429
        assert "Rate limit exceeded" in resp.json()["detail"]


class TestAdminEndpoints:
    def test_create_key_no_admin_token_403(self, client):
        c, _ = client
        resp = c.post("/admin/keys", json={"name": "test"})
        # Missing header → 422 (FastAPI validation), not auth pass-through
        assert resp.status_code == 422

    def test_create_key_wrong_admin_token(self, client):
        c, _ = client
        resp = c.post(
            "/admin/keys",
            json={"name": "test"},
            headers={"X-Admin-Token": "wrong"},
        )
        assert resp.status_code == 403

    def test_create_key_success(self, client):
        c, _ = client
        resp = c.post(
            "/admin/keys",
            json={"name": "my-app", "tier": "pro"},
            headers={"X-Admin-Token": "test-admin-secret"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "created"
        assert data["api_key"].startswith("tf_")
        assert data["tier"] == "pro"

    def test_list_keys(self, client):
        c, mgr = client
        mgr.create_key("k1", "free")
        mgr.create_key("k2", "pro")
        resp = c.get("/admin/keys", headers={"X-Admin-Token": "test-admin-secret"})
        assert resp.status_code == 200
        assert len(resp.json()["keys"]) == 2

    def test_revoke_key(self, client):
        c, mgr = client
        raw = mgr.create_key("to-revoke", "free")
        key = mgr.validate_key(raw)
        resp = c.delete(
            f"/admin/keys/{key.id}",
            headers={"X-Admin-Token": "test-admin-secret"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "revoked"

    def test_revoke_nonexistent_404(self, client):
        c, _ = client
        resp = c.delete(
            "/admin/keys/9999",
            headers={"X-Admin-Token": "test-admin-secret"},
        )
        assert resp.status_code == 404

    def test_usage_analytics(self, client):
        c, mgr = client
        raw = mgr.create_key("usage-key", "free")
        key = mgr.validate_key(raw)
        mgr.record_usage(key, "/predict/dam", 200)
        resp = c.get(
            "/admin/usage",
            params={"key_id": key.id, "days": 7},
            headers={"X-Admin-Token": "test-admin-secret"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_requests"] == 1

    def test_create_key_invalid_tier(self, client):
        c, _ = client
        resp = c.post(
            "/admin/keys",
            json={"name": "bad-tier", "tier": "platinum"},
            headers={"X-Admin-Token": "test-admin-secret"},
        )
        assert resp.status_code == 400
