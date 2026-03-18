"""
Telegram Alert Service for Mining Dispatch

Sends:
- Daily dispatch schedule (09:00 CT, after DAM predictions)
- Urgent spike alerts (within 1 min of detection)
- Daily PnL summary (22:00 CT)

Uses python-telegram-bot for async message delivery.
"""

import logging
import os
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import yaml

from .mining_dispatch import DispatchSchedule, HourAction, load_config

log = logging.getLogger(__name__)

CT = timezone(timedelta(hours=-6))  # Central Time (CST; CDT would be -5)


class AlertService:
    """Telegram alert sender for mining dispatch signals."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = load_config()
        alert_cfg = config.get("alerts", {})

        self.bot_token: str = (
            alert_cfg.get("bot_token", "")
            or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        )
        chat_ids_raw: str = (
            alert_cfg.get("chat_ids", "")
            or os.environ.get("TELEGRAM_CHAT_IDS", "")
            or os.environ.get("TELEGRAM_CHAT_ID", "")
        )
        self.chat_ids: List[str] = [
            cid.strip() for cid in str(chat_ids_raw).split(",") if cid.strip()
        ]
        self.spike_alert_threshold: float = alert_cfg.get("spike_alert_threshold", 0.7)
        self.spike_cooldown_minutes: int = alert_cfg.get("spike_cooldown_minutes", 30)
        self._last_spike_alert: float = 0.0  # epoch timestamp

    @property
    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_ids)

    # ------------------------------------------------------------------
    # Message formatting
    # ------------------------------------------------------------------

    @staticmethod
    def format_schedule_message(schedule: DispatchSchedule) -> str:
        """Format the daily dispatch schedule for Telegram."""
        lines = [
            f"MINING DISPATCH — {schedule.date}",
            f"Settlement: {schedule.settlement_point}",
            "",
            "Hour | Price   | Action | Reason",
            "—" * 36,
        ]

        for ha in schedule.hours:
            hour_str = f"{ha.hour_ending:02d}:00"
            price_str = f"${ha.dam_price:7.2f}"
            flag = ""
            if ha.spike_probability >= 0.5:
                flag = " ⚡"
            lines.append(f"{hour_str} | {price_str} | {ha.action:3s}  | {ha.reason}{flag}")

        lines.append("")
        lines.append(
            f"RUN: {schedule.hours_to_run}h | CURTAIL: {schedule.hours_to_curtail}h"
        )
        lines.append(f"Avg ON price: ${schedule.avg_on_price:.2f}")
        lines.append(f"Peak price:   ${schedule.peak_price:.2f}")
        lines.append(f"Savings:      ${schedule.expected_cost_savings:.2f}")

        if schedule.spike_hours:
            spike_str = ", ".join(f"HE{h}" for h in schedule.spike_hours)
            lines.append(f"Spike hours:  {spike_str}")

        return "\n".join(lines)

    @staticmethod
    def format_spike_alert(
        hour_ending: int,
        probability: float,
        dam_price: float,
        settlement_point: str = "HB_WEST",
    ) -> str:
        """Format an urgent spike alert message."""
        now_ct = datetime.now(CT).strftime("%H:%M CT")
        return (
            f"⚡ SPIKE ALERT — {settlement_point}\n"
            f"Hour Ending: {hour_ending:02d}:00\n"
            f"Probability: {probability:.0%}\n"
            f"DAM Price:   ${dam_price:.2f}\n"
            f"Action:      CURTAIL IMMEDIATELY\n"
            f"Detected at: {now_ct}"
        )

    @staticmethod
    def format_pnl_summary(
        schedule: DispatchSchedule,
        actual_prices: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Format the daily PnL summary message."""
        lines = [
            f"DAILY PnL SUMMARY — {schedule.date}",
            f"Settlement: {schedule.settlement_point}",
            "",
        ]

        if actual_prices:
            actual_cost = sum(
                float(p.get("price", 0)) * float(p.get("load_mw", 10.0))
                for p in actual_prices
                if p.get("action") == "ON"
            )
            lines.append(f"Actual cost:    ${actual_cost:,.2f}")
        else:
            lines.append(f"Projected cost: ${schedule.dispatch_cost:,.2f}")

        lines.extend([
            f"Always-on cost: ${schedule.always_on_cost:,.2f}",
            f"Estimated savings: ${schedule.expected_cost_savings:,.2f}",
            "",
            f"Hours ON:  {schedule.hours_to_run}",
            f"Hours OFF: {schedule.hours_to_curtail}",
            f"Avg ON price: ${schedule.avg_on_price:.2f}",
        ])

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Sending
    # ------------------------------------------------------------------

    def send_message(self, text: str) -> Dict[str, Any]:
        """
        Send a message to all configured Telegram chats.

        Returns dict with 'sent' count and 'errors' list.
        """
        import requests

        if not self.is_configured:
            return {"sent": 0, "errors": ["AlertService not configured (missing bot_token or chat_ids)"]}

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        sent = 0
        errors = []

        for chat_id in self.chat_ids:
            try:
                resp = requests.post(
                    url,
                    json={"chat_id": chat_id, "text": text},
                    timeout=10,
                )
                if resp.status_code == 200:
                    sent += 1
                else:
                    errors.append(f"chat {chat_id}: HTTP {resp.status_code}")
            except Exception as exc:
                errors.append(f"chat {chat_id}: {exc}")

        return {"sent": sent, "errors": errors}

    def send_schedule(self, schedule: DispatchSchedule) -> Dict[str, Any]:
        """Format and send the daily dispatch schedule."""
        text = self.format_schedule_message(schedule)
        return self.send_message(text)

    def send_spike_alert(
        self,
        hour_ending: int,
        probability: float,
        dam_price: float,
        settlement_point: str = "HB_WEST",
    ) -> Dict[str, Any]:
        """Send an urgent spike alert, respecting cooldown."""
        now = time.time()
        elapsed = (now - self._last_spike_alert) / 60.0
        if elapsed < self.spike_cooldown_minutes and self._last_spike_alert > 0:
            return {"sent": 0, "errors": [f"cooldown: {self.spike_cooldown_minutes - elapsed:.0f}min remaining"]}

        if probability < self.spike_alert_threshold:
            return {"sent": 0, "errors": [f"probability {probability:.2f} below threshold {self.spike_alert_threshold}"]}

        text = self.format_spike_alert(hour_ending, probability, dam_price, settlement_point)
        result = self.send_message(text)
        if result["sent"] > 0:
            self._last_spike_alert = now
        return result

    def send_pnl_summary(
        self,
        schedule: DispatchSchedule,
        actual_prices: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Format and send the daily PnL summary."""
        text = self.format_pnl_summary(schedule, actual_prices)
        return self.send_message(text)

    def get_config_summary(self) -> Dict[str, Any]:
        """Return current alert configuration (safe for API response)."""
        return {
            "configured": self.is_configured,
            "chat_ids_count": len(self.chat_ids),
            "spike_alert_threshold": self.spike_alert_threshold,
            "spike_cooldown_minutes": self.spike_cooldown_minutes,
        }

    def update_config(
        self,
        chat_ids: Optional[List[str]] = None,
        spike_alert_threshold: Optional[float] = None,
        spike_cooldown_minutes: Optional[int] = None,
        bot_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update alert preferences at runtime."""
        if chat_ids is not None:
            self.chat_ids = [cid.strip() for cid in chat_ids if cid.strip()]
        if spike_alert_threshold is not None:
            self.spike_alert_threshold = spike_alert_threshold
        if spike_cooldown_minutes is not None:
            self.spike_cooldown_minutes = spike_cooldown_minutes
        if bot_token is not None:
            self.bot_token = bot_token
        return self.get_config_summary()


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_service: Optional[AlertService] = None


def get_alert_service() -> AlertService:
    global _service
    if _service is None:
        _service = AlertService()
    return _service
