"""
BESS (Battery Energy Storage System) Predictor

Wraps the LP-based battery optimizer to provide optimal charge/discharge
schedules given DAM price forecasts.

Uses DAM predictions as input, runs the battery optimizer, and returns
the optimal schedule with revenue estimates.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class BessScheduleEntry:
    """Single hour entry in a BESS schedule."""
    hour_ending: int
    action: str           # "charge", "discharge", or "idle"
    power_mw: float       # positive = discharge, negative = charge
    soc_pct: float        # state of charge percentage
    dam_price: float      # DAM price for this hour


@dataclass
class BessScheduleResult:
    """Full BESS optimization result."""
    schedule: List[BessScheduleEntry]
    total_revenue: float
    status: str
    solve_time: float
    config: Dict[str, Any]


class BessPredictor:
    """
    BESS Schedule Optimizer.

    Takes DAM price forecasts and runs LP optimization to find the
    optimal charge/discharge schedule for maximum arbitrage revenue.
    """

    def __init__(self):
        self._optimizer = None
        self._config = None
        self._load_optimizer()

    def _load_optimizer(self):
        """Load the battery optimizer and default config."""
        try:
            import sys
            bess_src = str(Path(__file__).parent.parent.parent / "models" / "battery-strategy" / "src")
            if bess_src not in sys.path:
                sys.path.insert(0, bess_src)

            from battery_config import create_default_battery
            from optimizer import BatteryOptimizer

            self._config = create_default_battery()
            # Use hourly resolution for DAM-based optimization
            self._config.delta_t = 1.0  # 1 hour
            self._optimizer = BatteryOptimizer(self._config)
            log.info("BESS optimizer loaded with default config")
        except Exception as e:
            log.error("Failed to load BESS optimizer: %s", e)

    def is_ready(self) -> bool:
        return self._optimizer is not None

    def optimize(self, dam_prices: List[float]) -> BessScheduleResult:
        """
        Optimize battery schedule given DAM prices.

        Args:
            dam_prices: List of 24 hourly DAM prices ($/MWh).

        Returns:
            BessScheduleResult with optimal schedule and revenue.
        """
        if not self.is_ready():
            raise RuntimeError("BESS optimizer not loaded")

        prices = np.array(dam_prices, dtype=float)
        if len(prices) != 24:
            raise ValueError(f"Expected 24 hourly prices, got {len(prices)}")

        result = self._optimizer.optimize(prices)

        schedule = []
        for h in range(24):
            p_ch = float(result.P_ch[h]) if not np.isnan(result.P_ch[h]) else 0.0
            p_dis = float(result.P_dis[h]) if not np.isnan(result.P_dis[h]) else 0.0
            soc = float(result.SoC[h]) if not np.isnan(result.SoC[h]) else 0.0
            net_power = p_dis - p_ch

            if abs(net_power) < 0.001:
                action = "idle"
            elif net_power > 0:
                action = "discharge"
            else:
                action = "charge"

            schedule.append(BessScheduleEntry(
                hour_ending=h + 1,
                action=action,
                power_mw=round(net_power, 3),
                soc_pct=round(soc * 100, 1),
                dam_price=round(float(prices[h]), 2),
            ))

        return BessScheduleResult(
            schedule=schedule,
            total_revenue=round(float(result.total_revenue), 2) if not np.isnan(result.total_revenue) else 0.0,
            status=result.status,
            solve_time=round(result.solve_time, 4),
            config={
                "E_max_mwh": self._config.E_max,
                "P_max_mw": self._config.P_ch_max,
                "eta_ch": self._config.eta_ch,
                "eta_dis": self._config.eta_dis,
                "SoC_min_pct": self._config.SoC_min * 100,
                "SoC_max_pct": self._config.SoC_max * 100,
                "c_deg": self._config.c_deg,
            },
        )

    def get_model_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "model_type": "BESS LP Optimizer (PuLP CBC)",
            "optimizer_loaded": self.is_ready(),
        }
        if self._config:
            info["battery_config"] = {
                "E_max_mwh": self._config.E_max,
                "P_ch_max_mw": self._config.P_ch_max,
                "P_dis_max_mw": self._config.P_dis_max,
                "eta_ch": self._config.eta_ch,
                "eta_dis": self._config.eta_dis,
                "SoC_range": f"{self._config.SoC_min*100:.0f}%-{self._config.SoC_max*100:.0f}%",
                "degradation_cost": self._config.c_deg,
            }
        return info


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_predictor: Optional[BessPredictor] = None


def get_bess_predictor() -> BessPredictor:
    global _predictor
    if _predictor is None:
        _predictor = BessPredictor()
    return _predictor
