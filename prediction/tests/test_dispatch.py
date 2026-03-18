"""Tests for Sprint 6.1 — Mining Dispatch and Alert Service."""

import asyncio
from datetime import datetime
from typing import List
from dataclasses import dataclass

import pytest

from prediction.src.dispatch.mining_dispatch import (
    compute_dispatch,
    DispatchSchedule,
    HourAction,
    _parse_hour_ending,
)
from prediction.src.dispatch.alert_service import AlertService, get_alert_service
from prediction.src import main


# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------

def _make_dam_prices(prices: List[float]) -> List[dict]:
    return [
        {"hour_ending": i + 1, "predicted_price": p}
        for i, p in enumerate(prices)
    ]


def _make_spike_alerts(probs: List[float]) -> List[dict]:
    return [
        {"hour_ending": i + 1, "spike_probability": p, "is_spike": p >= 0.7}
        for i, p in enumerate(probs)
    ]


def _make_bess_schedule(actions: List[str]) -> List[dict]:
    return [
        {"hour_ending": i + 1, "action": a}
        for i, a in enumerate(actions)
    ]


DEFAULT_CONFIG = {
    "mining": {
        "breakeven_price": 55.0,
        "switching_cost": 40.0,
        "min_curtail_hours": 2,
        "spike_curtail_threshold": 0.7,
        "settlement_point": "HB_WEST",
        "load_mw": 10.0,
    },
    "bess": {"coordinate": True},
}


# ---------------------------------------------------------------------------
# Mining Dispatch Logic
# ---------------------------------------------------------------------------

class TestComputeDispatch:
    def test_all_cheap_hours_stay_on(self):
        """When all prices below breakeven, all hours should be ON."""
        prices = _make_dam_prices([30.0] * 24)
        schedule = compute_dispatch(prices, config=DEFAULT_CONFIG)
        assert schedule.hours_to_run == 24
        assert schedule.hours_to_curtail == 0
        assert all(h.action == "ON" for h in schedule.hours)

    def test_expensive_hours_curtailed(self):
        """Hours well above breakeven in a block should be curtailed."""
        # 10 cheap hours, 14 expensive hours (enough to exceed switching cost)
        prices_list = [30.0] * 10 + [120.0] * 14
        prices = _make_dam_prices(prices_list)
        schedule = compute_dispatch(prices, config=DEFAULT_CONFIG)
        assert schedule.hours_to_curtail > 0
        # Expensive hours (11-24) should mostly be OFF
        for ha in schedule.hours:
            if ha.hour_ending > 10:
                assert ha.action == "OFF", f"HE{ha.hour_ending} should be OFF at $120"

    def test_spike_forces_curtailment(self):
        """Spike alerts above threshold force hours OFF regardless of price."""
        prices = _make_dam_prices([40.0] * 24)  # all below breakeven
        spikes = _make_spike_alerts([0.0] * 4 + [0.9] * 2 + [0.0] * 18)
        schedule = compute_dispatch(prices, spikes, config=DEFAULT_CONFIG)
        # Hours 5-6 have spike prob 0.9 — should be OFF
        for ha in schedule.hours:
            if ha.hour_ending in (5, 6):
                assert ha.action == "OFF", f"HE{ha.hour_ending} should be OFF due to spike"
                assert ha.reason == "spike_alert"

    def test_bess_discharge_coordination(self):
        """BESS discharge hours with above-breakeven prices should curtail mining."""
        prices_list = [60.0] * 24  # all slightly above breakeven
        prices = _make_dam_prices(prices_list)
        bess = _make_bess_schedule(
            ["idle"] * 5 + ["discharge"] * 4 + ["idle"] * 15
        )
        schedule = compute_dispatch(prices, bess_schedule=bess, config=DEFAULT_CONFIG)
        # Hours 6-9 have BESS discharge + above breakeven
        for ha in schedule.hours:
            if ha.hour_ending in (6, 7, 8, 9):
                assert ha.action == "OFF", f"HE{ha.hour_ending} should be OFF (BESS discharge)"

    def test_savings_calculation(self):
        """Savings should be positive when curtailment avoids expensive hours."""
        prices_list = [30.0] * 12 + [150.0] * 12
        prices = _make_dam_prices(prices_list)
        schedule = compute_dispatch(prices, config=DEFAULT_CONFIG)
        assert schedule.expected_cost_savings > 0
        assert schedule.dispatch_cost < schedule.always_on_cost

    def test_min_curtail_hours_respected(self):
        """Single expensive hour shouldn't trigger curtailment if below min_curtail."""
        config = {**DEFAULT_CONFIG, "mining": {**DEFAULT_CONFIG["mining"], "min_curtail_hours": 3}}
        prices_list = [30.0] * 23 + [200.0]  # only 1 expensive hour
        prices = _make_dam_prices(prices_list)
        schedule = compute_dispatch(prices, config=config)
        # A single hour block can't meet min_curtail=3
        assert schedule.hours[23].action == "ON"

    def test_schedule_has_24_hours(self):
        prices = _make_dam_prices([50.0] * 24)
        schedule = compute_dispatch(prices, config=DEFAULT_CONFIG)
        assert len(schedule.hours) == 24
        assert schedule.hours_to_run + schedule.hours_to_curtail == 24

    def test_spike_hours_tracked(self):
        prices = _make_dam_prices([40.0] * 24)
        spikes = _make_spike_alerts([0.0] * 14 + [0.8, 0.85] + [0.0] * 8)
        schedule = compute_dispatch(prices, spikes, config=DEFAULT_CONFIG)
        assert 15 in schedule.spike_hours
        assert 16 in schedule.spike_hours


class TestParseHourEnding:
    def test_int(self):
        assert _parse_hour_ending(5) == 5

    def test_string_colon(self):
        assert _parse_hour_ending("14:00") == 14

    def test_string_plain(self):
        assert _parse_hour_ending("7") == 7


# ---------------------------------------------------------------------------
# Alert Service Formatting
# ---------------------------------------------------------------------------

class TestAlertFormatting:
    def _make_schedule(self) -> DispatchSchedule:
        prices = _make_dam_prices([30.0] * 12 + [120.0] * 12)
        return compute_dispatch(prices, config=DEFAULT_CONFIG)

    def test_format_schedule_message(self):
        schedule = self._make_schedule()
        msg = AlertService.format_schedule_message(schedule)
        assert "MINING DISPATCH" in msg
        assert "RUN:" in msg
        assert "CURTAIL:" in msg
        assert "Savings:" in msg

    def test_format_spike_alert(self):
        msg = AlertService.format_spike_alert(
            hour_ending=15, probability=0.92, dam_price=185.50,
        )
        assert "SPIKE ALERT" in msg
        assert "15:00" in msg
        assert "92%" in msg
        assert "$185.50" in msg
        assert "CURTAIL" in msg

    def test_format_pnl_summary(self):
        schedule = self._make_schedule()
        msg = AlertService.format_pnl_summary(schedule)
        assert "PnL SUMMARY" in msg
        assert "Always-on cost" in msg
        assert "Hours ON" in msg

    def test_pnl_with_actuals(self):
        schedule = self._make_schedule()
        actuals = [{"price": 30.0, "load_mw": 10.0, "action": "ON"}] * 12
        msg = AlertService.format_pnl_summary(schedule, actuals)
        assert "Actual cost" in msg


# ---------------------------------------------------------------------------
# Alert Service Config
# ---------------------------------------------------------------------------

class TestAlertService:
    def test_not_configured_by_default(self):
        svc = AlertService(config={"alerts": {}})
        assert not svc.is_configured

    def test_update_config(self):
        svc = AlertService(config={"alerts": {}})
        result = svc.update_config(
            chat_ids=["123", "456"],
            spike_alert_threshold=0.8,
            spike_cooldown_minutes=15,
        )
        assert result["chat_ids_count"] == 2
        assert result["spike_alert_threshold"] == 0.8
        assert result["spike_cooldown_minutes"] == 15

    def test_spike_below_threshold_not_sent(self):
        svc = AlertService(config={"alerts": {"spike_alert_threshold": 0.7}})
        result = svc.send_spike_alert(
            hour_ending=10, probability=0.3, dam_price=50.0,
        )
        assert result["sent"] == 0
        assert any("below threshold" in e for e in result["errors"])

    def test_send_message_not_configured(self):
        svc = AlertService(config={"alerts": {}})
        result = svc.send_message("test")
        assert result["sent"] == 0
        assert any("not configured" in e for e in result["errors"])


# ---------------------------------------------------------------------------
# API Endpoint Tests (monkeypatched)
# ---------------------------------------------------------------------------

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


class TestDispatchEndpoints:
    def _setup_fakes(self, monkeypatch, dam_prices=None):
        if dam_prices is None:
            dam_prices = [30.0] * 12 + [120.0] * 12

        class FakeDAMPredictor:
            def is_ready(self):
                return True
            def has_model(self, sp):
                return True
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
                from prediction.src.models.bess_predictor import BessScheduleEntry, BessScheduleResult
                sched = [
                    BessScheduleEntry(h + 1, "idle", 0.0, 50.0, prices[h])
                    for h in range(24)
                ]
                return BessScheduleResult(sched, 100.0, "Optimal", 0.01, {})

        import pandas as pd
        monkeypatch.setattr(main, "get_dam_v2_predictor", lambda: FakeDAMPredictor())
        monkeypatch.setattr(main, "get_spike_predictor", lambda: FakeSpikePredictor())
        monkeypatch.setattr(main, "get_bess_predictor", lambda: FakeBESSPredictor())
        monkeypatch.setattr(main, "_fetch_and_compute_features", lambda sp: pd.DataFrame({
            "delivery_date": ["2025-01-01"] * 24,
            "hour_ending": list(range(1, 25)),
        }))
        monkeypatch.setattr(main, "_latest_complete_delivery_rows", lambda df: df)

    def test_mining_schedule_endpoint(self, monkeypatch):
        self._setup_fakes(monkeypatch)
        result = asyncio.run(main.dispatch_mining_schedule(settlement_point="HB_WEST"))
        assert result["status"] == "success"
        assert len(result["schedule"]) == 24
        assert "hours_to_run" in result["summary"]
        assert "hours_to_curtail" in result["summary"]
        assert result["summary"]["hours_to_run"] + result["summary"]["hours_to_curtail"] == 24

    def test_mining_savings_endpoint(self, monkeypatch):
        self._setup_fakes(monkeypatch)
        result = asyncio.run(main.dispatch_mining_savings(settlement_point="HB_WEST"))
        assert result["status"] == "success"
        assert "expected_cost_savings" in result
        assert "savings_pct" in result
        assert result["always_on_cost"] > 0

    def test_alert_config_endpoint(self, monkeypatch):
        req = main.AlertConfigRequest(
            chat_ids=["12345"],
            spike_alert_threshold=0.85,
        )
        result = asyncio.run(main.configure_alerts(req))
        assert result["status"] == "success"
        assert result["config"]["chat_ids_count"] == 1
        assert result["config"]["spike_alert_threshold"] == 0.85

    def test_schedule_all_cheap(self, monkeypatch):
        """When all prices are cheap, all hours should be ON."""
        self._setup_fakes(monkeypatch, dam_prices=[25.0] * 24)
        result = asyncio.run(main.dispatch_mining_schedule(settlement_point="HB_WEST"))
        assert result["summary"]["hours_to_run"] == 24
        assert result["summary"]["hours_to_curtail"] == 0
