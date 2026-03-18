"""Tests for Sprint 6.2 — BESS Arbitrage Signal Service."""

import asyncio
import sqlite3
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import List
from unittest.mock import patch

import pytest

from prediction.src.dispatch.bess_signals import (
    generate_daily_signals,
    record_daily_pnl,
    get_rolling_pnl,
    compute_risk_metrics,
    DailySignals,
    DailyPnL,
    RiskMetrics,
    HourSignal,
    _compute_rtm_volatility,
    _ensure_pnl_db,
    BESS_DB,
)
from prediction.src import main


# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------

def _make_dam_prices(prices: List[float]) -> List[dict]:
    return [
        {"hour_ending": i + 1, "predicted_price": p}
        for i, p in enumerate(prices)
    ]


def _make_bess_schedule(actions: List[str], powers: List[float] = None, socs: List[float] = None) -> List[dict]:
    if powers is None:
        powers = [2.5 if a != "idle" else 0.0 for a in actions]
    if socs is None:
        socs = [50.0] * len(actions)
    return [
        {
            "hour_ending": i + 1,
            "action": a,
            "power_mw": powers[i],
            "soc_pct": socs[i],
            "dam_price": 50.0,
        }
        for i, a in enumerate(actions)
    ]


def _make_spike_alerts(probs: List[float]) -> List[dict]:
    return [
        {"hour_ending": i + 1, "spike_probability": p, "is_spike": p >= 0.7}
        for i, p in enumerate(probs)
    ]


def _make_mining_schedule(actions: List[str]) -> List[dict]:
    return [
        {"hour_ending": i + 1, "action": a}
        for i, a in enumerate(actions)
    ]


# Typical day: cheap overnight, expensive afternoon
TYPICAL_DAM = (
    [25.0] * 6 + [45.0, 55.0, 60.0, 65.0, 70.0, 75.0]
    + [80.0, 78.0, 76.0, 80.0, 85.0, 90.0]
    + [85.0, 70.0, 55.0, 45.0, 38.0, 32.0]
)

TYPICAL_BESS = (
    ["charge"] * 6 + ["idle"] * 6 + ["discharge"] * 6 + ["idle"] * 6
)


# ---------------------------------------------------------------------------
# Signal Generation
# ---------------------------------------------------------------------------

class TestGenerateDailySignals:
    def test_basic_signal_generation(self):
        """Signals should be generated for all 24 hours."""
        dam = _make_dam_prices(TYPICAL_DAM)
        bess = _make_bess_schedule(TYPICAL_BESS)
        signals = generate_daily_signals(dam, bess)

        assert len(signals.signals) == 24
        assert signals.charge_hours + signals.discharge_hours + signals.idle_hours == 24
        assert signals.settlement_point == "HB_WEST"

    def test_charge_during_low_prices(self):
        """Charge actions should occur during low-price hours."""
        dam = _make_dam_prices(TYPICAL_DAM)
        bess = _make_bess_schedule(TYPICAL_BESS)
        signals = generate_daily_signals(dam, bess)

        charge_signals = [s for s in signals.signals if s.action == "charge"]
        assert len(charge_signals) > 0
        avg_charge = sum(s.dam_price for s in charge_signals) / len(charge_signals)
        assert avg_charge < 50.0  # charge prices should be low

    def test_discharge_during_high_prices(self):
        """Discharge actions should occur during high-price hours."""
        dam = _make_dam_prices(TYPICAL_DAM)
        bess = _make_bess_schedule(TYPICAL_BESS)
        signals = generate_daily_signals(dam, bess)

        discharge_signals = [s for s in signals.signals if s.action == "discharge"]
        assert len(discharge_signals) > 0
        avg_dis = sum(s.dam_price for s in discharge_signals) / len(discharge_signals)
        assert avg_dis > 50.0  # discharge prices should be high

    def test_revenue_positive_with_spread(self):
        """Total revenue should be positive when there's a price spread."""
        dam = _make_dam_prices(TYPICAL_DAM)
        bess = _make_bess_schedule(TYPICAL_BESS)
        signals = generate_daily_signals(dam, bess)

        assert signals.total_revenue_estimate > 0

    def test_spike_hold_defers_discharge(self):
        """When spike_prob is high, discharge should be held for later."""
        dam = _make_dam_prices(TYPICAL_DAM)
        # Discharge at hours 13-18
        bess = _make_bess_schedule(TYPICAL_BESS)
        # High spike prob at hour 13, with future spike at hour 16
        spikes = [0.0] * 12 + [0.8, 0.0, 0.0, 0.8] + [0.0] * 8
        alerts = _make_spike_alerts(spikes)

        signals = generate_daily_signals(dam, bess, spike_alerts=alerts)

        # Hour 13 should be held (spike_hold) because hour 16 also has spike
        sig_13 = signals.signals[12]  # 0-indexed
        assert sig_13.risk_flag == "spike_hold"
        assert sig_13.action == "idle"
        assert 13 in signals.spike_hold_hours

    def test_no_spike_hold_without_future_spikes(self):
        """Don't hold if there are no future spike hours."""
        dam = _make_dam_prices(TYPICAL_DAM)
        bess = _make_bess_schedule(TYPICAL_BESS)
        # Spike only at last discharge hour (18), no future spikes
        spikes = [0.0] * 17 + [0.8] + [0.0] * 6
        alerts = _make_spike_alerts(spikes)

        signals = generate_daily_signals(dam, bess, spike_alerts=alerts)

        # Hour 18 should NOT be spike_hold because there's no future spike
        sig_18 = signals.signals[17]
        assert sig_18.risk_flag != "spike_hold"

    def test_mining_curtail_flag(self):
        """mining_curtail should be True when mining schedule says OFF."""
        dam = _make_dam_prices(TYPICAL_DAM)
        bess = _make_bess_schedule(TYPICAL_BESS)
        mining = _make_mining_schedule(["ON"] * 12 + ["OFF"] * 6 + ["ON"] * 6)

        signals = generate_daily_signals(dam, bess, mining_schedule=mining)

        for sig in signals.signals:
            if 13 <= sig.hour_ending <= 18:
                assert sig.mining_curtail is True
            else:
                assert sig.mining_curtail is False

    def test_risk_adjusted_revenue_lower(self):
        """Risk-adjusted revenue should be <= raw revenue."""
        dam = _make_dam_prices(TYPICAL_DAM)
        bess = _make_bess_schedule(TYPICAL_BESS)
        signals = generate_daily_signals(dam, bess)

        assert signals.risk_adjusted_revenue <= signals.total_revenue_estimate

    def test_all_idle_no_revenue(self):
        """All idle hours should produce zero revenue."""
        dam = _make_dam_prices([50.0] * 24)
        bess = _make_bess_schedule(["idle"] * 24)
        signals = generate_daily_signals(dam, bess)

        assert signals.total_revenue_estimate == 0.0
        assert signals.charge_hours == 0
        assert signals.discharge_hours == 0
        assert signals.idle_hours == 24

    def test_summary_statistics(self):
        """Summary fields should be consistent with signals."""
        dam = _make_dam_prices(TYPICAL_DAM)
        bess = _make_bess_schedule(TYPICAL_BESS)
        signals = generate_daily_signals(dam, bess)

        assert signals.peak_discharge_price > 0
        assert signals.avg_charge_price > 0
        assert signals.peak_discharge_price >= signals.avg_charge_price


# ---------------------------------------------------------------------------
# RTM Volatility
# ---------------------------------------------------------------------------

class TestRtmVolatility:
    def test_no_prices_returns_zero(self):
        assert _compute_rtm_volatility(None) == 0.0
        assert _compute_rtm_volatility([]) == 0.0

    def test_single_price_returns_zero(self):
        assert _compute_rtm_volatility([{"price": 50.0}]) == 0.0

    def test_constant_prices_zero_vol(self):
        prices = [{"price": 50.0}] * 12
        assert _compute_rtm_volatility(prices) == 0.0

    def test_volatile_prices_positive(self):
        prices = [{"price": p} for p in [30, 80, 25, 90, 40, 85, 35, 75, 45, 70, 50, 60]]
        vol = _compute_rtm_volatility(prices)
        assert vol > 0


# ---------------------------------------------------------------------------
# PnL Tracking
# ---------------------------------------------------------------------------

class TestPnLTracking:
    @pytest.fixture(autouse=True)
    def _use_temp_db(self, tmp_path, monkeypatch):
        """Point BESS_DB to a temp file for each test."""
        temp_db = tmp_path / "bess_pnl.db"
        monkeypatch.setattr("prediction.src.dispatch.bess_signals.BESS_DB", temp_db)
        monkeypatch.setattr("prediction.src.dispatch.bess_signals.DATA_DIR", tmp_path)

    def test_record_and_retrieve_pnl(self):
        dam = _make_dam_prices(TYPICAL_DAM)
        bess = _make_bess_schedule(TYPICAL_BESS)
        signals = generate_daily_signals(dam, bess)

        pnl = record_daily_pnl(signals)
        assert isinstance(pnl, DailyPnL)
        assert pnl.date == signals.date
        assert pnl.charge_cost >= 0
        assert pnl.discharge_revenue >= 0

        # Retrieve
        records = get_rolling_pnl(days=7)
        assert len(records) == 1
        assert records[0].date == pnl.date
        assert records[0].net_pnl == pnl.net_pnl

    def test_pnl_upsert(self):
        """Recording same date should update, not duplicate."""
        dam = _make_dam_prices(TYPICAL_DAM)
        bess = _make_bess_schedule(TYPICAL_BESS)
        signals = generate_daily_signals(dam, bess)

        record_daily_pnl(signals)
        record_daily_pnl(signals)

        records = get_rolling_pnl(days=7)
        assert len(records) == 1

    def test_pnl_with_actuals(self):
        dam = _make_dam_prices(TYPICAL_DAM)
        bess = _make_bess_schedule(TYPICAL_BESS)
        signals = generate_daily_signals(dam, bess)

        actuals = [{"hour_ending": i + 1, "price": TYPICAL_DAM[i] + 5.0} for i in range(24)]
        pnl = record_daily_pnl(signals, actual_prices=actuals)

        assert pnl.actual_revenue is not None

    def test_pnl_cycles(self):
        """Cycles should reflect energy throughput."""
        dam = _make_dam_prices(TYPICAL_DAM)
        bess = _make_bess_schedule(TYPICAL_BESS, powers=[2.5] * 24)
        signals = generate_daily_signals(dam, bess)

        pnl = record_daily_pnl(signals, battery_config={"E_max_mwh": 10.0})
        assert pnl.cycles > 0

    def test_empty_pnl(self):
        """Empty history should return empty list."""
        records = get_rolling_pnl(days=7)
        assert records == []


# ---------------------------------------------------------------------------
# Risk Metrics
# ---------------------------------------------------------------------------

class TestRiskMetrics:
    @pytest.fixture(autouse=True)
    def _use_temp_db(self, tmp_path, monkeypatch):
        temp_db = tmp_path / "bess_pnl.db"
        monkeypatch.setattr("prediction.src.dispatch.bess_signals.BESS_DB", temp_db)
        monkeypatch.setattr("prediction.src.dispatch.bess_signals.DATA_DIR", tmp_path)

    def test_empty_risk_metrics(self):
        """No PnL data should return zeroed metrics."""
        metrics = compute_risk_metrics(days=30)
        assert isinstance(metrics, RiskMetrics)
        assert metrics.total_pnl == 0.0
        assert metrics.win_rate == 0.0
        assert metrics.sharpe_ratio == 0.0

    def test_risk_with_data(self):
        """Metrics should be computed from PnL history."""
        from datetime import date, timedelta

        conn = _ensure_pnl_db()
        try:
            for i in range(10):
                d = (date.today() - timedelta(days=9 - i)).isoformat()
                pnl = 50.0 if i % 2 == 0 else -20.0
                conn.execute(
                    """INSERT INTO bess_daily_pnl
                       (date, settlement_point, projected_revenue, actual_revenue,
                        charge_cost, discharge_revenue, cycles, degradation_cost, net_pnl)
                       VALUES (?, 'HB_WEST', ?, NULL, 100.0, 150.0, 1.0, 5.0, ?)""",
                    (d, pnl, pnl),
                )
            conn.commit()
        finally:
            conn.close()

        metrics = compute_risk_metrics(days=30)
        assert metrics.days == 10
        assert metrics.total_pnl == 5 * 50.0 + 5 * (-20.0)  # 150.0
        assert metrics.win_rate == 0.5
        assert metrics.volatility > 0
        assert metrics.var_95 < 0  # 5th percentile should be negative
        assert metrics.max_drawdown <= 0

    def test_sharpe_positive_when_profitable(self):
        """Sharpe ratio should be positive for consistently profitable days."""
        from datetime import date, timedelta

        conn = _ensure_pnl_db()
        try:
            for i in range(20):
                d = (date.today() - timedelta(days=19 - i)).isoformat()
                pnl = 100.0 + (i % 3) * 10
                conn.execute(
                    """INSERT INTO bess_daily_pnl
                       (date, settlement_point, projected_revenue, actual_revenue,
                        charge_cost, discharge_revenue, cycles, degradation_cost, net_pnl)
                       VALUES (?, 'HB_WEST', ?, NULL, 50.0, 200.0, 1.5, 3.0, ?)""",
                    (d, pnl, pnl),
                )
            conn.commit()
        finally:
            conn.close()

        metrics = compute_risk_metrics(days=30)
        assert metrics.sharpe_ratio > 0
        assert metrics.win_rate == 1.0
        assert metrics.max_drawdown == 0.0  # never drops from peak


# ---------------------------------------------------------------------------
# API Endpoint Tests
# ---------------------------------------------------------------------------

from dataclasses import dataclass


@dataclass
class FakeDAMPrediction:
    hour_ending: int
    predicted_price: float


@dataclass
class FakeSpikeAlert:
    spike_probability: float
    is_spike: bool
    confidence: str
    threshold_used: float
    settlement_point: str
    timestamp: str


class TestBessDispatchEndpoints:
    def _setup_fakes(self, monkeypatch, tmp_path):
        import pandas as pd
        from prediction.src.models.bess_predictor import BessScheduleEntry, BessScheduleResult

        dam_prices = TYPICAL_DAM

        class FakeDAMPredictor:
            def is_ready(self):
                return True
            def has_model(self, sp):
                return sp.lower() in ["hb_west"]
            def missing_model_message(self, sp):
                return f"No model for {sp}"
            def available_settlement_points(self):
                return ["hb_west"]
            def predict(self, df, sp):
                return [FakeDAMPrediction(h + 1, dam_prices[h]) for h in range(24)]

        class FakeSpikePredictor:
            def is_ready(self):
                return True
            def has_model(self, sp):
                return True
            def missing_model_message(self, sp):
                return f"No model for {sp}"
            def predict(self, df, sp):
                return [
                    FakeSpikeAlert(0.1, False, "low", 100.0, "HB_WEST", "")
                    for _ in range(24)
                ]

        class FakeBESSPredictor:
            def is_ready(self):
                return True
            def has_model(self, sp):
                return True
            def missing_model_message(self, sp):
                return f"No model for {sp}"
            def optimize(self, prices):
                sched = []
                for h in range(24):
                    if h < 6:
                        action, power = "charge", 2.5
                    elif h < 12:
                        action, power = "idle", 0.0
                    elif h < 18:
                        action, power = "discharge", 2.5
                    else:
                        action, power = "idle", 0.0
                    sched.append(BessScheduleEntry(h + 1, action, power, 50.0, prices[h]))
                return BessScheduleResult(sched, 250.0, "Optimal", 0.02, {})

        monkeypatch.setattr(main, "get_dam_v2_predictor", lambda: FakeDAMPredictor())
        monkeypatch.setattr(main, "get_spike_predictor", lambda: FakeSpikePredictor())
        monkeypatch.setattr(main, "get_bess_predictor", lambda: FakeBESSPredictor())
        monkeypatch.setattr(main, "_fetch_and_compute_features", lambda sp: pd.DataFrame({
            "delivery_date": ["2025-01-01"] * 24,
            "hour_ending": list(range(1, 25)),
        }))
        monkeypatch.setattr(main, "_latest_complete_delivery_rows", lambda df: df)

        # Point PnL DB to temp
        temp_db = tmp_path / "bess_pnl.db"
        monkeypatch.setattr("prediction.src.dispatch.bess_signals.BESS_DB", temp_db)
        monkeypatch.setattr("prediction.src.dispatch.bess_signals.DATA_DIR", tmp_path)

    def test_daily_signals_endpoint(self, monkeypatch, tmp_path):
        self._setup_fakes(monkeypatch, tmp_path)
        result = asyncio.run(main.dispatch_bess_daily_signals(settlement_point="HB_WEST"))

        assert result["status"] == "success"
        assert len(result["signals"]) == 24
        assert "total_revenue_estimate" in result["summary"]
        assert "risk_adjusted_revenue" in result["summary"]
        assert result["summary"]["charge_hours"] + result["summary"]["discharge_hours"] + result["summary"]["idle_hours"] == 24

    def test_daily_signals_has_risk_flags(self, monkeypatch, tmp_path):
        self._setup_fakes(monkeypatch, tmp_path)
        result = asyncio.run(main.dispatch_bess_daily_signals(settlement_point="HB_WEST"))

        for sig in result["signals"]:
            assert "risk_flag" in sig
            assert sig["risk_flag"] in ("normal", "spike_hold", "high_volatility")
            assert "mining_curtail" in sig

    def test_pnl_endpoint(self, monkeypatch, tmp_path):
        self._setup_fakes(monkeypatch, tmp_path)

        # First generate signals to record PnL
        asyncio.run(main.dispatch_bess_daily_signals(settlement_point="HB_WEST"))

        result = asyncio.run(main.dispatch_bess_pnl(days=7, settlement_point="HB_WEST"))
        assert result["status"] == "success"
        assert "summary" in result
        assert "daily" in result
        assert result["days_available"] >= 1

    def test_risk_endpoint(self, monkeypatch, tmp_path):
        self._setup_fakes(monkeypatch, tmp_path)

        result = asyncio.run(main.dispatch_bess_risk(days=30, settlement_point="HB_WEST"))
        assert result["status"] == "success"
        assert "risk" in result
        assert "var_95" in result["risk"]
        assert "max_drawdown" in result["risk"]
        assert "sharpe_ratio" in result["risk"]
        assert "win_rate" in result["risk"]

    def test_pnl_recorded_after_signals(self, monkeypatch, tmp_path):
        """Calling daily-signals should auto-record PnL."""
        self._setup_fakes(monkeypatch, tmp_path)
        asyncio.run(main.dispatch_bess_daily_signals(settlement_point="HB_WEST"))

        pnl_result = asyncio.run(main.dispatch_bess_pnl(days=7, settlement_point="HB_WEST"))
        assert pnl_result["days_available"] >= 1
        assert pnl_result["summary"]["total_pnl"] != 0
