"""Tests for the prediction runner script."""

import json
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add prediction scripts to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_predictions import (
    _hour_ending_target_time,
    init_db,
    store_prediction,
    store_predictions_batch,
    compute_accuracy,
    run_rtm_predictions,
    run_spike_predictions,
    run_wind_predictions,
    run_load_predictions,
    run_dam_predictions,
    run,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test_predictions.db"


@pytest.fixture
def db_conn(db_path):
    conn = init_db(db_path)
    yield conn
    conn.close()


@pytest.fixture
def mock_client():
    return MagicMock()


# ---------------------------------------------------------------------------
# Database tests
# ---------------------------------------------------------------------------

class TestInitDB:
    def test_creates_tables(self, db_path):
        conn = init_db(db_path)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = [t[0] for t in tables]
        assert "predictions" in table_names
        assert "actuals" in table_names
        assert "prediction_accuracy" in table_names
        conn.close()

    def test_idempotent(self, db_path):
        conn1 = init_db(db_path)
        conn1.close()
        conn2 = init_db(db_path)
        tables = conn2.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        assert len([t for t in tables if t[0] in ("predictions", "actuals", "prediction_accuracy")]) == 3
        conn2.close()

    def test_creates_parent_dirs(self, tmp_path):
        deep_path = tmp_path / "a" / "b" / "c" / "test.db"
        conn = init_db(deep_path)
        assert deep_path.exists()
        conn.close()


class TestStorePrediction:
    def test_store_single(self, db_conn):
        pid = store_prediction(
            db_conn,
            model="rtm",
            settlement_point="HB_WEST",
            target_time="2026-03-18T15:00:00",
            predicted_value=42.50,
            unit="USD/MWh",
            horizon="1h",
        )
        assert pid > 0
        row = db_conn.execute("SELECT * FROM predictions WHERE id = ?", (pid,)).fetchone()
        assert row is not None
        assert row[1] == "rtm"  # model
        assert row[2] == "HB_WEST"  # settlement_point
        assert row[5] == 42.50  # predicted_value

    def test_store_with_metadata(self, db_conn):
        meta = {"is_spike": True, "confidence": "high"}
        pid = store_prediction(
            db_conn,
            model="spike",
            settlement_point="HB_WEST",
            target_time="2026-03-18T15:00:00",
            predicted_value=0.85,
            unit="probability",
            metadata=meta,
        )
        row = db_conn.execute("SELECT metadata FROM predictions WHERE id = ?", (pid,)).fetchone()
        assert json.loads(row[0]) == meta

    def test_store_batch(self, db_conn):
        rows = [
            ("dam", "HB_WEST", "2026-03-19T01:00:00", "next-day", 35.0, "USD/MWh", None, "2026-03-18T09:00:00"),
            ("dam", "HB_WEST", "2026-03-19T02:00:00", "next-day", 32.0, "USD/MWh", None, "2026-03-18T09:00:00"),
            ("dam", "HB_WEST", "2026-03-19T03:00:00", "next-day", 30.0, "USD/MWh", None, "2026-03-18T09:00:00"),
        ]
        count = store_predictions_batch(db_conn, rows)
        assert count == 3
        total = db_conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        assert total == 3

    def test_upserts_existing_prediction(self, db_conn):
        first_id = store_prediction(
            db_conn,
            model="rtm",
            settlement_point="HB_WEST",
            target_time="2026-03-18T15:00:00+00:00",
            predicted_value=42.50,
            unit="USD/MWh",
            horizon="1h",
            generated_at="2026-03-18T14:00:00+00:00",
        )
        second_id = store_prediction(
            db_conn,
            model="rtm",
            settlement_point="HB_WEST",
            target_time="2026-03-18T15:00:00+00:00",
            predicted_value=43.25,
            unit="USD/MWh",
            horizon="1h",
            generated_at="2026-03-18T14:05:00+00:00",
        )

        row = db_conn.execute(
            "SELECT id, predicted_value, generated_at FROM predictions WHERE model = 'rtm'"
        ).fetchone()
        assert row[0] == first_id == second_id
        assert row[1] == 43.25
        assert row[2] == "2026-03-18T14:05:00+00:00"


class TestComputeAccuracy:
    def test_matches_predictions_with_actuals(self, db_conn):
        # Insert a prediction
        db_conn.execute(
            """INSERT INTO predictions (model, settlement_point, target_time, predicted_value, unit, generated_at)
               VALUES ('dam', 'HB_WEST', '2026-03-19T01:00:00', 35.0, 'USD/MWh', '2026-03-18T09:00:00')"""
        )
        # Insert matching actual
        db_conn.execute(
            """INSERT INTO actuals (market, settlement_point, target_time, actual_value, unit)
               VALUES ('dam', 'HB_WEST', '2026-03-19T01:00:00', 33.0, 'USD/MWh')"""
        )
        db_conn.commit()

        matched = compute_accuracy(db_conn)
        assert matched == 1

        acc = db_conn.execute("SELECT error, abs_error, pct_error FROM prediction_accuracy").fetchone()
        assert abs(acc[0] - 2.0) < 0.01  # error = 35 - 33
        assert abs(acc[1] - 2.0) < 0.01  # abs_error
        assert abs(acc[2] - (2.0 / 33.0 * 100)) < 0.1  # pct_error

    def test_no_double_scoring(self, db_conn):
        db_conn.execute(
            """INSERT INTO predictions (model, settlement_point, target_time, predicted_value, unit, generated_at)
               VALUES ('dam', 'HB_WEST', '2026-03-19T01:00:00', 35.0, 'USD/MWh', '2026-03-18T09:00:00')"""
        )
        db_conn.execute(
            """INSERT INTO actuals (market, settlement_point, target_time, actual_value, unit)
               VALUES ('dam', 'HB_WEST', '2026-03-19T01:00:00', 33.0, 'USD/MWh')"""
        )
        db_conn.commit()

        first = compute_accuracy(db_conn)
        second = compute_accuracy(db_conn)
        assert first == 1
        assert second == 0

    def test_no_actuals_returns_zero(self, db_conn):
        db_conn.execute(
            """INSERT INTO predictions (model, settlement_point, target_time, predicted_value, unit, generated_at)
               VALUES ('rtm', 'HB_WEST', '2026-03-18T15:00:00', 42.0, 'USD/MWh', '2026-03-18T14:00:00')"""
        )
        db_conn.commit()
        assert compute_accuracy(db_conn) == 0


# ---------------------------------------------------------------------------
# API caller tests
# ---------------------------------------------------------------------------

def _make_mock_client(response_data):
    """Create a mock httpx.Client that returns given data for GET calls."""
    mock = MagicMock()
    resp = MagicMock()
    resp.json.return_value = response_data
    resp.raise_for_status.return_value = None
    mock.get.return_value = resp
    return mock


class TestRunRTMPredictions:
    def test_stores_predictions(self, db_path):
        conn = init_db(db_path)
        client = _make_mock_client({
            "status": "success",
            "generated_at": "2026-03-18T14:00:00",
            "predictions": [
                {"horizon": "1h", "hours_ahead": 1, "predicted_price": 42.15},
                {"horizon": "4h", "hours_ahead": 4, "predicted_price": 38.92},
                {"horizon": "24h", "hours_ahead": 24, "predicted_price": 35.67},
            ],
        })
        count = run_rtm_predictions(client, conn)
        assert count == 3
        rows = conn.execute("SELECT model, predicted_value FROM predictions ORDER BY predicted_value").fetchall()
        assert len(rows) == 3
        assert rows[0][0] == "rtm"
        conn.close()

    def test_handles_api_failure(self, db_path):
        conn = init_db(db_path)
        client = _make_mock_client(None)
        client.get.side_effect = Exception("connection refused")
        count = run_rtm_predictions(client, conn)
        assert count == 0
        conn.close()


class TestRunSpikePredictions:
    def test_stores_spike_alert(self, db_path):
        conn = init_db(db_path)
        client = _make_mock_client({
            "status": "success",
            "generated_at": "2026-03-18T14:00:00",
            "alert": {
                "is_spike": True,
                "spike_probability": 0.823,
                "confidence": "high",
                "threshold": 0.707,
            },
        })
        count = run_spike_predictions(client, conn)
        assert count == 1
        row = conn.execute("SELECT model, predicted_value, unit, metadata FROM predictions").fetchone()
        assert row[0] == "spike"
        assert abs(row[1] - 0.823) < 0.001
        assert row[2] == "probability"
        meta = json.loads(row[3])
        assert meta["is_spike"] is True
        conn.close()


class TestRunWindPredictions:
    def test_stores_24_hours(self, db_path):
        conn = init_db(db_path)
        preds = [
            {"hour_ending": f"{h:02d}:00", "predicted_mw": 10000 + h * 100,
             "lower_bound_mw": 8000 + h * 80, "upper_bound_mw": 12000 + h * 120}
            for h in range(1, 25)
        ]
        client = _make_mock_client({
            "status": "success",
            "generated_at": "2026-03-18T14:00:00",
            "predictions": preds,
        })
        count = run_wind_predictions(client, conn)
        assert count == 24
        rows = conn.execute("SELECT unit FROM predictions LIMIT 1").fetchone()
        assert rows[0] == "MW"
        conn.close()


class TestRunLoadPredictions:
    def test_stores_24_hours(self, db_path):
        conn = init_db(db_path)
        preds = [
            {"hour_ending": f"{h:02d}:00", "predicted_load_mw": 40000 + h * 500}
            for h in range(1, 25)
        ]
        client = _make_mock_client({
            "status": "success",
            "generated_at": "2026-03-18T14:00:00",
            "predictions": preds,
        })
        count = run_load_predictions(client, conn)
        assert count == 24
        conn.close()


class TestRunDAMPredictions:
    def test_stores_all_settlement_points(self, db_path):
        conn = init_db(db_path)
        preds = [
            {"hour_ending": f"{h:02d}:00", "predicted_price": 30 + h}
            for h in range(1, 25)
        ]
        client = _make_mock_client({
            "status": "success",
            "delivery_date": "2026-03-19",
            "generated_at": "2026-03-18T09:00:00",
            "predictions": preds,
        })
        count = run_dam_predictions(client, conn)
        # 5 settlement points * 24 hours = 120
        assert count == 120
        sps = conn.execute("SELECT DISTINCT settlement_point FROM predictions").fetchall()
        assert len(sps) == 5
        conn.close()


class TestHourEndingConversion:
    def test_rolls_he24_to_next_day_midnight(self):
        assert _hour_ending_target_time("2026-03-19", "24:00") == "2026-03-20T00:00:00"


# ---------------------------------------------------------------------------
# Integration: run() scheduler
# ---------------------------------------------------------------------------

class TestRunScheduler:
    @patch("run_predictions.datetime")
    def test_five_min_run(self, mock_dt, db_path):
        """Non-hour, non-9am run should only do RTM + spike."""
        mock_dt.now.return_value = datetime(2026, 3, 18, 14, 15, 0, tzinfo=timezone.utc)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        with patch("run_predictions.httpx.Client") as MockClient:
            mock_client = MagicMock()
            # Return failure for all to keep test simple
            resp = MagicMock()
            resp.json.return_value = {"status": "error"}
            resp.raise_for_status.return_value = None
            mock_client.get.return_value = resp
            MockClient.return_value = mock_client

            results = run(db_path=db_path, api_base="http://127.0.0.1:8011")

        # RTM and spike should be attempted, wind/load/dam should not
        call_urls = [call.args[0] for call in mock_client.get.call_args_list]
        assert any("/predictions/rtm" in u for u in call_urls)
        assert any("/predictions/spike" in u for u in call_urls)
        assert not any("/predictions/wind" in u for u in call_urls)
        assert not any("/predictions/load" in u for u in call_urls)
        assert not any("/predictions/dam" in u for u in call_urls)

    @patch("run_predictions.datetime")
    def test_hourly_run(self, mock_dt, db_path):
        """Hour boundary (minute=0) should also run wind + load."""
        mock_dt.now.return_value = datetime(2026, 3, 18, 14, 0, 0, tzinfo=timezone.utc)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        with patch("run_predictions.httpx.Client") as MockClient:
            mock_client = MagicMock()
            resp = MagicMock()
            resp.json.return_value = {"status": "error"}
            resp.raise_for_status.return_value = None
            mock_client.get.return_value = resp
            MockClient.return_value = mock_client

            results = run(db_path=db_path, api_base="http://127.0.0.1:8011")

        call_urls = [call.args[0] for call in mock_client.get.call_args_list]
        assert any("/predictions/wind" in u for u in call_urls)
        assert any("/predictions/load" in u for u in call_urls)

    @patch("run_predictions.datetime")
    def test_daily_dam_run(self, mock_dt, db_path):
        """09:00 CT run should trigger DAM predictions."""
        mock_dt.now.return_value = datetime(2026, 3, 18, 14, 0, 0, tzinfo=timezone.utc)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        with patch("run_predictions.httpx.Client") as MockClient:
            mock_client = MagicMock()
            resp = MagicMock()
            resp.json.return_value = {"status": "error"}
            resp.raise_for_status.return_value = None
            mock_client.get.return_value = resp
            MockClient.return_value = mock_client

            results = run(db_path=db_path, api_base="http://127.0.0.1:8011")

        call_urls = [call.args[0] for call in mock_client.get.call_args_list]
        assert any("/predictions/dam" in u for u in call_urls)

    @patch("run_predictions.datetime")
    def test_nine_utc_no_longer_triggers_dam_outside_central_9am(self, mock_dt, db_path):
        mock_dt.now.return_value = datetime(2026, 3, 18, 9, 0, 0, tzinfo=timezone.utc)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        with patch("run_predictions.httpx.Client") as MockClient:
            mock_client = MagicMock()
            resp = MagicMock()
            resp.json.return_value = {"status": "error"}
            resp.raise_for_status.return_value = None
            mock_client.get.return_value = resp
            MockClient.return_value = mock_client

            run(db_path=db_path, api_base="http://127.0.0.1:8011")

        call_urls = [call.args[0] for call in mock_client.get.call_args_list]
        assert not any("/predictions/dam" in u for u in call_urls)
