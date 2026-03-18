#!/usr/bin/env python3
"""
Prediction accuracy scorer for TrueFlux.

Reads stored predictions from predictions.db, fetches actual prices from
ercot_archive.db, computes error metrics, and updates the prediction_accuracy table.

Usage:
    python score_predictions.py [--days 7] [--report]
"""

import json
import logging
import math
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("score-predictions")

PREDICTIONS_DB = Path(__file__).resolve().parents[1] / "data" / "predictions.db"
ARCHIVE_DB = Path(__file__).resolve().parents[2] / "scraper" / "data" / "ercot_archive.db"


# ---------------------------------------------------------------------------
# Fetching actuals from the archive
# ---------------------------------------------------------------------------

def fetch_dam_actuals(
    archive_conn: sqlite3.Connection,
    settlement_point: str,
    start_date: str,
    end_date: str,
) -> List[Tuple[str, float]]:
    """Fetch actual DAM LMP prices. Returns list of (target_time, actual_value)."""
    rows = archive_conn.execute(
        """SELECT delivery_date, hour_ending, lmp
           FROM dam_lmp_hist
           WHERE settlement_point = ?
             AND delivery_date >= ? AND delivery_date <= ?
           ORDER BY delivery_date, hour_ending""",
        (settlement_point, start_date, end_date),
    ).fetchall()

    results = []
    for delivery_date, hour_ending, lmp in rows:
        target_time = f"{delivery_date}T{hour_ending:02d}:00:00"
        results.append((target_time, lmp))
    return results


def fetch_rtm_actuals(
    archive_conn: sqlite3.Connection,
    settlement_point: str,
    start_date: str,
    end_date: str,
) -> List[Tuple[str, float]]:
    """Fetch actual RTM LMP prices (averaged per hour). Returns list of (target_time, actual_value)."""
    rows = archive_conn.execute(
        """SELECT delivery_date, delivery_hour, AVG(lmp) as avg_lmp
           FROM rtm_lmp_hist
           WHERE settlement_point = ?
             AND delivery_date >= ? AND delivery_date <= ?
           GROUP BY delivery_date, delivery_hour
           ORDER BY delivery_date, delivery_hour""",
        (settlement_point, start_date, end_date),
    ).fetchall()

    results = []
    for delivery_date, delivery_hour, avg_lmp in rows:
        # RTM uses delivery_hour (0-23); predictions store hour_ending (1-24 style)
        # Normalize to match prediction target_time format
        hour = delivery_hour + 1  # convert 0-based hour to hour_ending
        target_time = f"{delivery_date}T{hour:02d}:00:00"
        results.append((target_time, avg_lmp))
    return results


# ---------------------------------------------------------------------------
# Syncing actuals into predictions.db
# ---------------------------------------------------------------------------

def sync_actuals(
    pred_conn: sqlite3.Connection,
    archive_conn: sqlite3.Connection,
    days: int = 7,
) -> int:
    """
    Pull actual prices from archive DB into predictions.db actuals table.

    Only fetches actuals for dates that have predictions but no matching actuals yet.
    """
    end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")

    # Find distinct (model, settlement_point) combos from predictions in the date range
    pred_combos = pred_conn.execute(
        """SELECT DISTINCT model, settlement_point
           FROM predictions
           WHERE target_time >= ? AND model IN ('dam', 'rtm')""",
        (start_date,),
    ).fetchall()

    inserted = 0
    for model, sp in pred_combos:
        if not sp:
            continue

        market = model  # 'dam' or 'rtm'

        if market == "dam":
            actuals = fetch_dam_actuals(archive_conn, sp, start_date, end_date)
        elif market == "rtm":
            actuals = fetch_rtm_actuals(archive_conn, sp, start_date, end_date)
        else:
            continue

        for target_time, actual_value in actuals:
            # Upsert: skip if already exists
            existing = pred_conn.execute(
                """SELECT id FROM actuals
                   WHERE market = ? AND settlement_point = ? AND target_time = ?""",
                (market, sp, target_time),
            ).fetchone()

            if existing is None:
                pred_conn.execute(
                    """INSERT INTO actuals (market, settlement_point, target_time, actual_value, unit)
                       VALUES (?, ?, ?, ?, 'USD/MWh')""",
                    (market, sp, target_time, actual_value),
                )
                inserted += 1

    pred_conn.commit()
    log.info("Synced %d actuals from archive", inserted)
    return inserted


# ---------------------------------------------------------------------------
# Scoring: match predictions to actuals and compute errors
# ---------------------------------------------------------------------------

def score_predictions(pred_conn: sqlite3.Connection) -> int:
    """
    Match unscored predictions with actuals and insert accuracy rows.

    Uses the existing compute_accuracy logic from run_predictions.py
    but is idempotent — skips already-scored pairs.
    """
    rows = pred_conn.execute("""
        SELECT p.id, a.id, p.predicted_value, a.actual_value
        FROM predictions p
        JOIN actuals a
            ON p.model = a.market
            AND COALESCE(p.settlement_point, '') = COALESCE(a.settlement_point, '')
            AND p.target_time = a.target_time
        LEFT JOIN prediction_accuracy pa
            ON pa.prediction_id = p.id AND pa.actual_id = a.id
        WHERE pa.id IS NULL
          AND p.model IN ('dam', 'rtm')
    """).fetchall()

    if not rows:
        log.info("No unscored predictions to match")
        return 0

    accuracy_rows = []
    for pred_id, actual_id, predicted, actual in rows:
        error = predicted - actual
        abs_error = abs(error)
        pct_error = (error / actual * 100) if actual != 0 else None
        accuracy_rows.append((pred_id, actual_id, error, abs_error, pct_error))

    pred_conn.executemany(
        """INSERT INTO prediction_accuracy
           (prediction_id, actual_id, error, abs_error, pct_error)
           VALUES (?, ?, ?, ?, ?)""",
        accuracy_rows,
    )
    pred_conn.commit()
    log.info("Scored %d prediction-actual pairs", len(accuracy_rows))
    return len(accuracy_rows)


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def compute_metrics(errors: List[float], pct_errors: List[Optional[float]]) -> Dict[str, Any]:
    """Compute MAE, RMSE, MAPE, and directional accuracy from error lists."""
    if not errors:
        return {"mae": None, "rmse": None, "mape": None, "directional_accuracy": None, "count": 0}

    n = len(errors)
    abs_errors = [abs(e) for e in errors]
    mae = sum(abs_errors) / n
    rmse = math.sqrt(sum(e ** 2 for e in errors) / n)

    valid_pct = [abs(p) for p in pct_errors if p is not None]
    mape = sum(valid_pct) / len(valid_pct) if valid_pct else None

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape": round(mape, 4) if mape is not None else None,
        "count": n,
    }


def generate_accuracy_report(
    pred_conn: sqlite3.Connection,
    days: int = 7,
) -> Dict[str, Any]:
    """
    Generate a comprehensive accuracy report for recent predictions.

    Returns a dict with per-model and per-hour breakdowns.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    # Per-model summary
    models = pred_conn.execute(
        """SELECT DISTINCT p.model
           FROM prediction_accuracy pa
           JOIN predictions p ON p.id = pa.prediction_id
           WHERE pa.computed_at >= ?""",
        (cutoff,),
    ).fetchall()

    report: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "days": days,
        "models": {},
    }

    for (model,) in models:
        # Overall metrics for this model
        rows = pred_conn.execute(
            """SELECT pa.error, pa.pct_error
               FROM prediction_accuracy pa
               JOIN predictions p ON p.id = pa.prediction_id
               WHERE p.model = ? AND pa.computed_at >= ?""",
            (model, cutoff),
        ).fetchall()

        errors = [r[0] for r in rows]
        pct_errors = [r[1] for r in rows]
        metrics = compute_metrics(errors, pct_errors)

        # Directional accuracy: % of predictions where sign of change was correct
        direction_rows = pred_conn.execute(
            """SELECT p.predicted_value, a.actual_value, p.target_time
               FROM prediction_accuracy pa
               JOIN predictions p ON p.id = pa.prediction_id
               JOIN actuals a ON a.id = pa.actual_id
               WHERE p.model = ? AND pa.computed_at >= ?
               ORDER BY p.target_time""",
            (model, cutoff),
        ).fetchall()

        if len(direction_rows) > 1:
            correct_dir = 0
            total_dir = 0
            for i in range(1, len(direction_rows)):
                prev_actual = direction_rows[i - 1][1]
                curr_pred = direction_rows[i][0]
                curr_actual = direction_rows[i][1]
                pred_dir = curr_pred - prev_actual
                actual_dir = curr_actual - prev_actual
                if pred_dir * actual_dir > 0:
                    correct_dir += 1
                total_dir += 1
            metrics["directional_accuracy"] = round(correct_dir / total_dir, 4) if total_dir > 0 else None
        else:
            metrics["directional_accuracy"] = None

        # Per-hour breakdown
        hour_rows = pred_conn.execute(
            """SELECT
                   CAST(SUBSTR(p.target_time, 12, 2) AS INTEGER) as hour,
                   pa.error,
                   pa.pct_error
               FROM prediction_accuracy pa
               JOIN predictions p ON p.id = pa.prediction_id
               WHERE p.model = ? AND pa.computed_at >= ?""",
            (model, cutoff),
        ).fetchall()

        hourly: Dict[int, Dict[str, list]] = {}
        for hour, error, pct_error in hour_rows:
            if hour not in hourly:
                hourly[hour] = {"errors": [], "pct_errors": []}
            hourly[hour]["errors"].append(error)
            hourly[hour]["pct_errors"].append(pct_error)

        hourly_metrics = {}
        for hour in sorted(hourly.keys()):
            hourly_metrics[str(hour)] = compute_metrics(
                hourly[hour]["errors"], hourly[hour]["pct_errors"]
            )

        # Recent comparisons
        recent = pred_conn.execute(
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

        report["models"][model] = {
            "metrics": metrics,
            "hourly": hourly_metrics,
            "recent_comparisons": recent_comparisons,
        }

    return report


def print_report(report: Dict[str, Any]) -> None:
    """Print a human-readable summary of the accuracy report."""
    print(f"\n{'='*60}")
    print(f"  TrueFlux Prediction Accuracy Report")
    print(f"  Period: last {report['days']} days")
    print(f"  Generated: {report['generated_at']}")
    print(f"{'='*60}")

    for model, data in report["models"].items():
        metrics = data["metrics"]
        print(f"\n  Model: {model.upper()}")
        print(f"  {'─'*40}")
        print(f"  Predictions scored: {metrics['count']}")
        print(f"  MAE:  {metrics['mae']:.2f} $/MWh" if metrics["mae"] else "  MAE:  N/A")
        print(f"  RMSE: {metrics['rmse']:.2f} $/MWh" if metrics["rmse"] else "  RMSE: N/A")
        print(f"  MAPE: {metrics['mape']:.2f}%" if metrics["mape"] else "  MAPE: N/A")
        if metrics.get("directional_accuracy") is not None:
            print(f"  Directional Accuracy: {metrics['directional_accuracy']:.1%}")

        if data["recent_comparisons"]:
            print(f"\n  Latest comparisons (most recent first):")
            print(f"  {'Target Time':<22} {'Pred':>8} {'Actual':>8} {'Error':>8}")
            for c in data["recent_comparisons"][:10]:
                print(f"  {c['target_time']:<22} {c['predicted']:>8.2f} {c['actual']:>8.2f} {c['error']:>+8.2f}")

    print(f"\n{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    days: int = 7,
    report: bool = True,
    predictions_db: Optional[Path] = None,
    archive_db: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run the full scoring pipeline: sync actuals, score, generate report."""
    pred_path = predictions_db or PREDICTIONS_DB
    arch_path = archive_db or ARCHIVE_DB

    if not pred_path.exists():
        log.error("Predictions DB not found: %s", pred_path)
        return {"error": "predictions_db_not_found"}

    pred_conn = sqlite3.connect(str(pred_path))
    pred_conn.execute("PRAGMA journal_mode=WAL")
    pred_conn.execute("PRAGMA foreign_keys=ON")

    archive_conn = None
    if arch_path.exists():
        archive_conn = sqlite3.connect(str(arch_path))
        try:
            synced = sync_actuals(pred_conn, archive_conn, days=days)
        except Exception as e:
            log.warning("Failed to sync actuals: %s", e)
            synced = 0
    else:
        log.warning("Archive DB not found: %s — skipping actuals sync", arch_path)
        synced = 0

    scored = score_predictions(pred_conn)

    accuracy_report = generate_accuracy_report(pred_conn, days=days)
    accuracy_report["synced_actuals"] = synced
    accuracy_report["newly_scored"] = scored

    if report:
        print_report(accuracy_report)

        # Write JSON report
        report_path = pred_path.parent / "accuracy_report.json"
        with open(report_path, "w") as f:
            json.dump(accuracy_report, f, indent=2)
        log.info("Report written to %s", report_path)

    pred_conn.close()
    if archive_conn:
        archive_conn.close()

    return accuracy_report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Score predictions against actuals")
    parser.add_argument("--days", type=int, default=7, help="Number of days to look back (default: 7)")
    parser.add_argument("--no-report", action="store_true", help="Skip printing report")
    args = parser.parse_args()

    run(days=args.days, report=not args.no_report)
