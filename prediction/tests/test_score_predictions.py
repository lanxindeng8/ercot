"""Tests for the prediction scoring pipeline."""

import json
import math
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from score_predictions import (
    compute_metrics,
    fetch_dam_actuals,
    fetch_rtm_actuals,
    generate_accuracy_report,
    score_predictions,
    sync_actuals,
)
from run_predictions import init_db


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pred_db(tmp_path):
    """Create a predictions DB with schema."""
    db_path = tmp_path / "predictions.db"
    conn = init_db(db_path)
    yield conn
    conn.close()


@pytest.fixture
def archive_db(tmp_path):
    """Create a mock archive DB with dam_lmp_hist and rtm_lmp_hist tables."""
    db_path = tmp_path / "archive.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE dam_lmp_hist (
            delivery_date TEXT NOT NULL,
            hour_ending INTEGER NOT NULL,
            repeated_hour INTEGER NOT NULL DEFAULT 0,
            settlement_point TEXT NOT NULL,
            lmp REAL NOT NULL,
            PRIMARY KEY (delivery_date, hour_ending, repeated_hour, settlement_point)
        );
        CREATE TABLE rtm_lmp_hist (
            delivery_date TEXT NOT NULL,
            delivery_hour INTEGER NOT NULL,
            delivery_interval INTEGER NOT NULL,
            repeated_hour INTEGER NOT NULL DEFAULT 0,
            settlement_point TEXT NOT NULL,
            settlement_point_type TEXT,
            lmp REAL NOT NULL,
            PRIMARY KEY (delivery_date, delivery_hour, delivery_interval, repeated_hour, settlement_point)
        );
    """)
    conn.commit()
    yield conn
    conn.close()


def _insert_dam_actual(conn, date, hour, sp, lmp):
    conn.execute(
        "INSERT INTO dam_lmp_hist VALUES (?, ?, 0, ?, ?)",
        (date, hour, sp, lmp),
    )


def _insert_rtm_actual(conn, date, hour, interval, sp, lmp):
    conn.execute(
        "INSERT INTO rtm_lmp_hist VALUES (?, ?, ?, 0, ?, NULL, ?)",
        (date, hour, interval, sp, lmp),
    )


def _insert_prediction(conn, model, sp, target_time, value, generated_at="2026-03-18T09:00:00"):
    conn.execute(
        """INSERT INTO predictions (model, settlement_point, target_time, predicted_value, unit, generated_at)
           VALUES (?, ?, ?, ?, 'USD/MWh', ?)""",
        (model, sp, target_time, value, generated_at),
    )


# ---------------------------------------------------------------------------
# compute_metrics tests
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_empty_errors(self):
        result = compute_metrics([], [])
        assert result["count"] == 0
        assert result["mae"] is None
        assert result["rmse"] is None

    def test_basic_errors(self):
        errors = [2.0, -3.0, 1.0, -1.0]
        pct_errors = [5.0, -8.0, 2.5, -2.0]
        result = compute_metrics(errors, pct_errors)
        assert result["count"] == 4
        assert result["mae"] == pytest.approx(1.75, abs=0.01)
        expected_rmse = math.sqrt((4 + 9 + 1 + 1) / 4)
        assert result["rmse"] == pytest.approx(expected_rmse, abs=0.01)
        # MAPE = mean of abs(pct_errors)
        expected_mape = (5 + 8 + 2.5 + 2) / 4
        assert result["mape"] == pytest.approx(expected_mape, abs=0.01)

    def test_with_none_pct_errors(self):
        errors = [2.0, -3.0]
        pct_errors = [5.0, None]
        result = compute_metrics(errors, pct_errors)
        assert result["mape"] == pytest.approx(5.0, abs=0.01)

    def test_all_none_pct_errors(self):
        errors = [2.0]
        pct_errors = [None]
        result = compute_metrics(errors, pct_errors)
        assert result["mape"] is None


# ---------------------------------------------------------------------------
# fetch_actuals tests
# ---------------------------------------------------------------------------

class TestFetchActuals:
    def test_fetch_dam_actuals(self, archive_db):
        _insert_dam_actual(archive_db, "2026-03-18", 10, "HB_WEST", 35.50)
        _insert_dam_actual(archive_db, "2026-03-18", 11, "HB_WEST", 37.25)
        _insert_dam_actual(archive_db, "2026-03-18", 10, "HB_NORTH", 33.00)
        archive_db.commit()

        results = fetch_dam_actuals(archive_db, "HB_WEST", "2026-03-18", "2026-03-18")
        assert len(results) == 2
        assert results[0] == ("2026-03-18T10:00:00", 35.50)
        assert results[1] == ("2026-03-18T11:00:00", 37.25)

    def test_fetch_rtm_actuals_averages_intervals(self, archive_db):
        # 4 intervals for hour 10
        for interval in range(1, 5):
            _insert_rtm_actual(archive_db, "2026-03-18", 10, interval, "HB_WEST", 40.0 + interval)
        archive_db.commit()

        results = fetch_rtm_actuals(archive_db, "HB_WEST", "2026-03-18", "2026-03-18")
        assert len(results) == 1
        # RTM hour 10 -> hour_ending 11
        assert results[0][0] == "2026-03-18T11:00:00"
        # Average of 41, 42, 43, 44 = 42.5
        assert results[0][1] == pytest.approx(42.5)

    def test_fetch_dam_no_data(self, archive_db):
        results = fetch_dam_actuals(archive_db, "HB_WEST", "2026-01-01", "2026-01-01")
        assert results == []


# ---------------------------------------------------------------------------
# sync_actuals tests
# ---------------------------------------------------------------------------

class TestSyncActuals:
    def test_syncs_dam_actuals(self, pred_db, archive_db):
        # Add a DAM prediction
        _insert_prediction(pred_db, "dam", "HB_WEST", "2026-03-18T10:00:00", 36.0)
        pred_db.commit()

        # Add matching actual in archive
        _insert_dam_actual(archive_db, "2026-03-18", 10, "HB_WEST", 35.50)
        archive_db.commit()

        synced = sync_actuals(pred_db, archive_db, days=7)
        assert synced >= 1

        actuals = pred_db.execute("SELECT * FROM actuals").fetchall()
        assert len(actuals) >= 1
        # Check market is 'dam'
        assert actuals[0][1] == "dam"

    def test_no_duplicate_sync(self, pred_db, archive_db):
        _insert_prediction(pred_db, "dam", "HB_WEST", "2026-03-18T10:00:00", 36.0)
        pred_db.commit()
        _insert_dam_actual(archive_db, "2026-03-18", 10, "HB_WEST", 35.50)
        archive_db.commit()

        first = sync_actuals(pred_db, archive_db, days=7)
        second = sync_actuals(pred_db, archive_db, days=7)
        assert first >= 1
        assert second == 0


# ---------------------------------------------------------------------------
# score_predictions tests
# ---------------------------------------------------------------------------

class TestScorePredictions:
    def test_scores_matching_pairs(self, pred_db):
        _insert_prediction(pred_db, "dam", "HB_WEST", "2026-03-18T10:00:00", 36.0)
        pred_db.execute(
            """INSERT INTO actuals (market, settlement_point, target_time, actual_value, unit)
               VALUES ('dam', 'HB_WEST', '2026-03-18T10:00:00', 34.0, 'USD/MWh')"""
        )
        pred_db.commit()

        scored = score_predictions(pred_db)
        assert scored == 1

        acc = pred_db.execute("SELECT error, abs_error, pct_error FROM prediction_accuracy").fetchone()
        assert acc[0] == pytest.approx(2.0)  # 36 - 34
        assert acc[1] == pytest.approx(2.0)
        assert acc[2] == pytest.approx(2.0 / 34.0 * 100, abs=0.1)

    def test_no_double_scoring(self, pred_db):
        _insert_prediction(pred_db, "dam", "HB_WEST", "2026-03-18T10:00:00", 36.0)
        pred_db.execute(
            """INSERT INTO actuals (market, settlement_point, target_time, actual_value, unit)
               VALUES ('dam', 'HB_WEST', '2026-03-18T10:00:00', 34.0, 'USD/MWh')"""
        )
        pred_db.commit()

        first = score_predictions(pred_db)
        second = score_predictions(pred_db)
        assert first == 1
        assert second == 0

    def test_no_match_returns_zero(self, pred_db):
        _insert_prediction(pred_db, "dam", "HB_WEST", "2026-03-18T10:00:00", 36.0)
        pred_db.commit()
        assert score_predictions(pred_db) == 0

    def test_rtm_model_matching(self, pred_db):
        """RTM predictions should match with rtm actuals."""
        _insert_prediction(pred_db, "rtm", "HB_WEST", "2026-03-18T15:00:00", 45.0)
        pred_db.execute(
            """INSERT INTO actuals (market, settlement_point, target_time, actual_value, unit)
               VALUES ('rtm', 'HB_WEST', '2026-03-18T15:00:00', 42.0, 'USD/MWh')"""
        )
        pred_db.commit()

        scored = score_predictions(pred_db)
        assert scored == 1
        acc = pred_db.execute("SELECT error FROM prediction_accuracy").fetchone()
        assert acc[0] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# generate_accuracy_report tests
# ---------------------------------------------------------------------------

class TestGenerateReport:
    def test_empty_report(self, pred_db):
        report = generate_accuracy_report(pred_db, days=7)
        assert report["days"] == 7
        assert report["models"] == {}

    def test_report_with_data(self, pred_db):
        # Insert several scored predictions
        for hour in range(10, 14):
            pred_id = hour  # will be auto-assigned
            _insert_prediction(pred_db, "dam", "HB_WEST", f"2026-03-18T{hour:02d}:00:00", 35.0 + hour)
            pred_db.execute(
                """INSERT INTO actuals (market, settlement_point, target_time, actual_value, unit)
                   VALUES ('dam', 'HB_WEST', ?, ?, 'USD/MWh')""",
                (f"2026-03-18T{hour:02d}:00:00", 33.0 + hour),
            )
        pred_db.commit()
        score_predictions(pred_db)

        report = generate_accuracy_report(pred_db, days=7)
        assert "dam" in report["models"]
        dam = report["models"]["dam"]
        assert dam["metrics"]["count"] == 4
        assert dam["metrics"]["mae"] == pytest.approx(2.0, abs=0.01)
        assert len(dam["recent_comparisons"]) == 4
        assert len(dam["hourly"]) == 4

    def test_report_per_hour_breakdown(self, pred_db):
        # Two predictions at same hour
        _insert_prediction(pred_db, "dam", "HB_WEST", "2026-03-17T10:00:00", 36.0)
        pred_db.execute(
            "INSERT INTO actuals (market, settlement_point, target_time, actual_value, unit) VALUES ('dam', 'HB_WEST', '2026-03-17T10:00:00', 34.0, 'USD/MWh')"
        )
        _insert_prediction(pred_db, "dam", "HB_WEST", "2026-03-18T10:00:00", 38.0)
        pred_db.execute(
            "INSERT INTO actuals (market, settlement_point, target_time, actual_value, unit) VALUES ('dam', 'HB_WEST', '2026-03-18T10:00:00', 35.0, 'USD/MWh')"
        )
        pred_db.commit()
        score_predictions(pred_db)

        report = generate_accuracy_report(pred_db, days=7)
        hourly = report["models"]["dam"]["hourly"]
        assert "10" in hourly
        assert hourly["10"]["count"] == 2
        # errors: 2.0 and 3.0, MAE = 2.5
        assert hourly["10"]["mae"] == pytest.approx(2.5, abs=0.01)
