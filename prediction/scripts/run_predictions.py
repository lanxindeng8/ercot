#!/usr/bin/env python3
"""
Real-time prediction runner for TrueFlux Prediction Service.

Scheduling logic (called every 5 minutes by LaunchAgent):
- Every 5 min: RTM price + spike detection
- Every hour (minute 0): wind + load forecasts
- Every day at 09:00 CT: DAM next-day for all settlement points

Writes all predictions to SQLite for tracking and comparison with actuals.
"""

import json
import logging
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE = "http://127.0.0.1:8011"
DB_PATH = Path(__file__).resolve().parents[1] / "data" / "predictions.db"
REQUEST_TIMEOUT = 60.0

DAM_SETTLEMENT_POINTS = ["HB_WEST", "HB_NORTH", "HB_SOUTH", "HB_HOUSTON", "HB_BUSAVG"]
RTM_SETTLEMENT_POINTS = ["HB_WEST"]
SPIKE_SETTLEMENT_POINTS = ["HB_WEST"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("prediction-runner")


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def init_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Initialize SQLite database with predictions and actuals tables."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT NOT NULL,
            settlement_point TEXT,
            target_time TEXT NOT NULL,
            horizon TEXT,
            predicted_value REAL NOT NULL,
            unit TEXT NOT NULL DEFAULT 'USD/MWh',
            metadata TEXT,
            generated_at TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS actuals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market TEXT NOT NULL,
            settlement_point TEXT,
            target_time TEXT NOT NULL,
            actual_value REAL NOT NULL,
            unit TEXT NOT NULL DEFAULT 'USD/MWh',
            fetched_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS prediction_accuracy (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id INTEGER NOT NULL REFERENCES predictions(id),
            actual_id INTEGER NOT NULL REFERENCES actuals(id),
            error REAL NOT NULL,
            abs_error REAL NOT NULL,
            pct_error REAL,
            computed_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_pred_model_time
            ON predictions(model, target_time);
        CREATE INDEX IF NOT EXISTS idx_pred_generated
            ON predictions(generated_at);
        CREATE INDEX IF NOT EXISTS idx_actual_market_time
            ON actuals(market, target_time);
        CREATE INDEX IF NOT EXISTS idx_accuracy_pred
            ON prediction_accuracy(prediction_id);
    """)
    conn.commit()
    return conn


def store_prediction(
    conn: sqlite3.Connection,
    model: str,
    settlement_point: Optional[str],
    target_time: str,
    predicted_value: float,
    unit: str = "USD/MWh",
    horizon: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    generated_at: Optional[str] = None,
) -> int:
    """Insert a prediction row and return its ID."""
    cur = conn.execute(
        """INSERT INTO predictions
           (model, settlement_point, target_time, horizon, predicted_value, unit, metadata, generated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            model,
            settlement_point,
            target_time,
            horizon,
            predicted_value,
            unit,
            json.dumps(metadata) if metadata else None,
            generated_at or datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    return cur.lastrowid


def store_predictions_batch(
    conn: sqlite3.Connection,
    rows: List[tuple],
) -> int:
    """Insert multiple prediction rows. Each row is a tuple matching the INSERT columns."""
    conn.executemany(
        """INSERT INTO predictions
           (model, settlement_point, target_time, horizon, predicted_value, unit, metadata, generated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()
    return len(rows)


def compute_accuracy(conn: sqlite3.Connection) -> int:
    """Match unscored predictions with actuals and compute accuracy metrics."""
    rows = conn.execute("""
        SELECT p.id, a.id, p.predicted_value, a.actual_value
        FROM predictions p
        JOIN actuals a
            ON p.model LIKE '%' || a.market || '%'
            AND p.settlement_point = a.settlement_point
            AND p.target_time = a.target_time
        LEFT JOIN prediction_accuracy pa ON pa.prediction_id = p.id AND pa.actual_id = a.id
        WHERE pa.id IS NULL
    """).fetchall()

    if not rows:
        return 0

    accuracy_rows = []
    for pred_id, actual_id, predicted, actual in rows:
        error = predicted - actual
        abs_error = abs(error)
        pct_error = (error / actual * 100) if actual != 0 else None
        accuracy_rows.append((pred_id, actual_id, error, abs_error, pct_error))

    conn.executemany(
        """INSERT INTO prediction_accuracy
           (prediction_id, actual_id, error, abs_error, pct_error)
           VALUES (?, ?, ?, ?, ?)""",
        accuracy_rows,
    )
    conn.commit()
    return len(accuracy_rows)


# ---------------------------------------------------------------------------
# API callers
# ---------------------------------------------------------------------------

def call_api(client: httpx.Client, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
    """Call prediction API endpoint and return JSON response."""
    url = f"{API_BASE}{endpoint}"
    try:
        resp = client.get(url, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        log.warning("API %s returned %d: %s", endpoint, e.response.status_code, e.response.text[:200])
        return None
    except Exception as e:
        log.error("API %s failed: %s", endpoint, e)
        return None


def run_rtm_predictions(client: httpx.Client, conn: sqlite3.Connection) -> int:
    """Fetch RTM predictions and store them."""
    count = 0
    now = datetime.now(timezone.utc)
    for sp in RTM_SETTLEMENT_POINTS:
        data = call_api(client, "/predictions/rtm", {"settlement_point": sp})
        if not data or data.get("status") != "success":
            continue
        generated_at = data.get("generated_at", now.isoformat())
        rows = []
        for pred in data.get("predictions", []):
            horizon = pred.get("horizon", "")
            hours_ahead = pred.get("hours_ahead", 0)
            target_time = now.replace(minute=0, second=0, microsecond=0)
            from datetime import timedelta
            target_time = (target_time + timedelta(hours=hours_ahead)).isoformat()
            rows.append((
                "rtm",
                sp,
                target_time,
                horizon,
                pred["predicted_price"],
                "USD/MWh",
                None,
                generated_at,
            ))
        if rows:
            count += store_predictions_batch(conn, rows)
            log.info("RTM: stored %d predictions for %s", len(rows), sp)
    return count


def run_spike_predictions(client: httpx.Client, conn: sqlite3.Connection) -> int:
    """Fetch spike predictions and store them."""
    count = 0
    now = datetime.now(timezone.utc)
    for sp in SPIKE_SETTLEMENT_POINTS:
        data = call_api(client, "/predictions/spike", {"settlement_point": sp})
        if not data or data.get("status") != "success":
            continue
        alert = data.get("alert", {})
        generated_at = data.get("generated_at", now.isoformat())
        from datetime import timedelta
        target_time = (now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)).isoformat()
        meta = {
            "is_spike": alert.get("is_spike"),
            "confidence": alert.get("confidence"),
            "threshold": alert.get("threshold"),
        }
        store_prediction(
            conn,
            model="spike",
            settlement_point=sp,
            target_time=target_time,
            predicted_value=alert.get("spike_probability", 0.0),
            unit="probability",
            horizon="1h",
            metadata=meta,
            generated_at=generated_at,
        )
        count += 1
        log.info("Spike: %s prob=%.3f is_spike=%s confidence=%s",
                 sp, alert.get("spike_probability", 0), alert.get("is_spike"), alert.get("confidence"))
    return count


def run_wind_predictions(client: httpx.Client, conn: sqlite3.Connection) -> int:
    """Fetch wind generation predictions and store them."""
    now = datetime.now(timezone.utc)
    data = call_api(client, "/predictions/wind")
    if not data or data.get("status") != "success":
        return 0
    generated_at = data.get("generated_at", now.isoformat())
    base_date = now.strftime("%Y-%m-%d")
    rows = []
    for pred in data.get("predictions", []):
        hour_str = pred.get("hour_ending", "01:00")
        target_time = f"{base_date}T{hour_str}:00+00:00"
        meta = {
            "lower_bound_mw": pred.get("lower_bound_mw"),
            "upper_bound_mw": pred.get("upper_bound_mw"),
        }
        rows.append((
            "wind",
            None,
            target_time,
            None,
            pred["predicted_mw"],
            "MW",
            json.dumps(meta),
            generated_at,
        ))
    if rows:
        store_predictions_batch(conn, rows)
        log.info("Wind: stored %d hourly forecasts", len(rows))
    return len(rows)


def run_load_predictions(client: httpx.Client, conn: sqlite3.Connection) -> int:
    """Fetch load predictions and store them."""
    now = datetime.now(timezone.utc)
    data = call_api(client, "/predictions/load")
    if not data or data.get("status") != "success":
        return 0
    generated_at = data.get("generated_at", now.isoformat())
    base_date = now.strftime("%Y-%m-%d")
    rows = []
    for pred in data.get("predictions", []):
        hour_str = pred.get("hour_ending", "01:00")
        target_time = f"{base_date}T{hour_str}:00+00:00"
        rows.append((
            "load",
            None,
            target_time,
            None,
            pred["predicted_load_mw"],
            "MW",
            None,
            generated_at,
        ))
    if rows:
        store_predictions_batch(conn, rows)
        log.info("Load: stored %d hourly forecasts", len(rows))
    return len(rows)


def run_dam_predictions(client: httpx.Client, conn: sqlite3.Connection) -> int:
    """Fetch DAM next-day predictions for all settlement points."""
    count = 0
    now = datetime.now(timezone.utc)
    for sp in DAM_SETTLEMENT_POINTS:
        data = call_api(client, "/predictions/dam/next-day", {"settlement_point": sp})
        if not data or data.get("status") != "success":
            continue
        generated_at = data.get("generated_at", now.isoformat())
        delivery_date = data.get("delivery_date", "")
        rows = []
        for pred in data.get("predictions", []):
            hour_str = pred.get("hour_ending", "01:00")
            target_time = f"{delivery_date}T{hour_str}:00"
            rows.append((
                "dam",
                sp,
                target_time,
                "next-day",
                pred["predicted_price"],
                "USD/MWh",
                None,
                generated_at,
            ))
        if rows:
            count += store_predictions_batch(conn, rows)
            log.info("DAM: stored %d predictions for %s (delivery %s)", len(rows), sp, delivery_date)
    return count


# ---------------------------------------------------------------------------
# Main scheduler logic
# ---------------------------------------------------------------------------

def run(db_path: Path = DB_PATH, api_base: str = API_BASE) -> Dict[str, int]:
    """
    Execute prediction tasks based on current time.

    Returns dict of model -> count of predictions stored.
    """
    global API_BASE
    API_BASE = api_base

    now = datetime.now(timezone.utc)
    minute = now.minute
    hour = now.hour
    results: Dict[str, int] = {}

    conn = init_db(db_path)
    client = httpx.Client(base_url=api_base, timeout=REQUEST_TIMEOUT)

    try:
        # Every 5 min: RTM + spike
        log.info("Running 5-min tasks (RTM + spike)...")
        results["rtm"] = run_rtm_predictions(client, conn)
        results["spike"] = run_spike_predictions(client, conn)

        # Every hour (when minute < 5, i.e. first run of the hour): wind + load
        if minute < 5:
            log.info("Running hourly tasks (wind + load)...")
            results["wind"] = run_wind_predictions(client, conn)
            results["load"] = run_load_predictions(client, conn)

        # Daily at 09:00 UTC (minute < 5): DAM next-day for all SPs
        if hour == 9 and minute < 5:
            log.info("Running daily DAM next-day predictions...")
            results["dam"] = run_dam_predictions(client, conn)

        # Try to match predictions with actuals
        matched = compute_accuracy(conn)
        if matched:
            log.info("Accuracy: scored %d prediction-actual pairs", matched)
            results["accuracy_matched"] = matched

    finally:
        client.close()
        conn.close()

    total = sum(v for k, v in results.items() if k != "accuracy_matched")
    log.info("Run complete: %d predictions stored %s", total, results)
    return results


if __name__ == "__main__":
    run()
