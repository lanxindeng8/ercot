"""
API Key management with SQLite-backed storage.

Supports key generation, validation, per-key rate limiting, usage tracking,
and tiered access (free / pro / enterprise).
"""

import hashlib
import logging
import os
import sqlite3
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "api_keys.db"

TIER_LIMITS: Dict[str, int] = {
    "free": 100,         # requests per day
    "pro": 10_000,       # requests per day
    "enterprise": 0,     # 0 = unlimited
}


@dataclass
class APIKey:
    id: int
    key_prefix: str
    key_hash: str
    name: str
    tier: str
    active: bool
    created_at: str
    last_used: Optional[str]
    request_count: int


def _hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


class APIKeyManager:
    """SQLite-backed API key store with rate limiting."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

        # In-memory rate-limit tracking: key_hash -> list of request timestamps
        self._rate_windows: Dict[str, list] = defaultdict(list)
        self._rate_lock = Lock()

    def _init_db(self):
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key_prefix TEXT NOT NULL,
                    key_hash TEXT NOT NULL UNIQUE,
                    name TEXT NOT NULL,
                    tier TEXT NOT NULL DEFAULT 'free',
                    active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    last_used TEXT,
                    request_count INTEGER NOT NULL DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS usage_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key_hash TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    status_code INTEGER NOT NULL DEFAULT 200
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_usage_key_ts
                ON usage_log (key_hash, timestamp)
            """)
            conn.commit()

    # ------------------------------------------------------------------
    # Key CRUD
    # ------------------------------------------------------------------

    def create_key(self, name: str, tier: str = "free") -> str:
        """Create a new API key. Returns the raw key (only shown once)."""
        if tier not in TIER_LIMITS:
            raise ValueError(f"Invalid tier: {tier}. Must be one of {list(TIER_LIMITS.keys())}")

        raw_key = f"tf_{uuid.uuid4().hex}"
        key_hash = _hash_key(raw_key)
        key_prefix = raw_key[:10]
        now = datetime.now(UTC).isoformat()

        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                "INSERT INTO api_keys (key_prefix, key_hash, name, tier, created_at) VALUES (?, ?, ?, ?, ?)",
                (key_prefix, key_hash, name, tier, now),
            )
            conn.commit()

        log.info("Created API key '%s' (tier=%s, prefix=%s)", name, tier, key_prefix)
        return raw_key

    def validate_key(self, raw_key: str) -> Optional[APIKey]:
        """Validate a raw API key. Returns APIKey if valid and active, else None."""
        key_hash = _hash_key(raw_key)
        with sqlite3.connect(str(self.db_path)) as conn:
            row = conn.execute(
                "SELECT id, key_prefix, key_hash, name, tier, active, created_at, last_used, request_count "
                "FROM api_keys WHERE key_hash = ?",
                (key_hash,),
            ).fetchone()

        if row is None or not row[5]:  # not found or inactive
            return None

        return APIKey(
            id=row[0], key_prefix=row[1], key_hash=row[2], name=row[3],
            tier=row[4], active=bool(row[5]), created_at=row[6],
            last_used=row[7], request_count=row[8],
        )

    def list_keys(self) -> List[APIKey]:
        with sqlite3.connect(str(self.db_path)) as conn:
            rows = conn.execute(
                "SELECT id, key_prefix, key_hash, name, tier, active, created_at, last_used, request_count "
                "FROM api_keys ORDER BY created_at DESC"
            ).fetchall()
        return [
            APIKey(id=r[0], key_prefix=r[1], key_hash=r[2], name=r[3],
                   tier=r[4], active=bool(r[5]), created_at=r[6],
                   last_used=r[7], request_count=r[8])
            for r in rows
        ]

    def revoke_key(self, key_id: int) -> bool:
        with sqlite3.connect(str(self.db_path)) as conn:
            cur = conn.execute("UPDATE api_keys SET active = 0 WHERE id = ?", (key_id,))
            conn.commit()
            return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def check_rate_limit(self, api_key: APIKey) -> bool:
        """Return True if request is within rate limit, False if exceeded."""
        daily_limit = TIER_LIMITS.get(api_key.tier, 100)
        if daily_limit == 0:  # unlimited
            return True

        now = time.time()
        window_start = now - 86400  # 24-hour rolling window

        with self._rate_lock:
            timestamps = self._rate_windows[api_key.key_hash]
            # Prune old entries
            self._rate_windows[api_key.key_hash] = [t for t in timestamps if t > window_start]
            if len(self._rate_windows[api_key.key_hash]) >= daily_limit:
                return False
            self._rate_windows[api_key.key_hash].append(now)
            return True

    # ------------------------------------------------------------------
    # Usage tracking
    # ------------------------------------------------------------------

    def record_usage(self, api_key: APIKey, endpoint: str, status_code: int = 200):
        now = datetime.now(UTC).isoformat()
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                "INSERT INTO usage_log (key_hash, endpoint, timestamp, status_code) VALUES (?, ?, ?, ?)",
                (api_key.key_hash, endpoint, now, status_code),
            )
            conn.execute(
                "UPDATE api_keys SET last_used = ?, request_count = request_count + 1 WHERE id = ?",
                (now, api_key.id),
            )
            conn.commit()

    def get_usage(self, key_id: Optional[int] = None, days: int = 7) -> List[Dict]:
        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            if key_id is not None:
                row = conn.execute("SELECT key_hash FROM api_keys WHERE id = ?", (key_id,)).fetchone()
                if not row:
                    return []
                rows = conn.execute(
                    "SELECT endpoint, timestamp, status_code FROM usage_log "
                    "WHERE key_hash = ? AND timestamp > ? ORDER BY timestamp DESC",
                    (row["key_hash"], cutoff),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT u.endpoint, u.timestamp, u.status_code, k.name as key_name "
                    "FROM usage_log u JOIN api_keys k ON u.key_hash = k.key_hash "
                    "WHERE u.timestamp > ? ORDER BY u.timestamp DESC",
                    (cutoff,),
                ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_manager: Optional[APIKeyManager] = None


def get_key_manager(db_path: Optional[Path] = None) -> APIKeyManager:
    global _manager
    if _manager is None:
        _manager = APIKeyManager(db_path)
    return _manager


def reset_key_manager():
    """Reset singleton (for testing)."""
    global _manager
    _manager = None
