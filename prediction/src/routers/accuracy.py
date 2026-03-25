"""Accuracy endpoint: prediction accuracy statistics."""

import sqlite3
from datetime import datetime, timedelta
from typing import Any, Dict

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from ..helpers import PREDICTIONS_DB

router = APIRouter(tags=["Accuracy"])


@router.get("/accuracy")
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
