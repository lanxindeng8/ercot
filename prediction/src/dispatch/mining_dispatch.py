"""
Mining Dispatch Engine

Computes optimal ON/OFF schedule for mining operations based on:
- DAM price predictions (next 24h)
- Spike alerts (force curtail during predicted spikes)
- BESS discharge schedule (coordinate to avoid high-price hours)

Returns hourly actions, run/curtail hours, and estimated cost savings.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

log = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "dispatch.yaml"


@dataclass
class HourAction:
    """Single-hour dispatch decision."""
    hour_ending: int          # 1-24
    dam_price: float          # $/MWh
    action: str               # "ON" or "OFF"
    reason: str               # why this action was chosen
    spike_probability: float = 0.0
    bess_action: str = ""     # charge/discharge/idle from BESS


@dataclass
class DispatchSchedule:
    """Full 24-hour mining dispatch schedule."""
    date: str
    settlement_point: str
    hours: List[HourAction]
    hours_to_run: int
    hours_to_curtail: int
    expected_cost_savings: float   # $ vs always-on
    always_on_cost: float          # total cost if always on
    dispatch_cost: float           # total cost with schedule
    peak_price: float
    avg_on_price: float
    spike_hours: List[int]         # hours with spike alerts
    generated_at: str = ""

    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.utcnow().isoformat()


def load_config() -> Dict[str, Any]:
    """Load dispatch configuration from YAML."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f)
    log.warning("Dispatch config not found at %s, using defaults", CONFIG_PATH)
    return {
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


def compute_dispatch(
    dam_prices: List[Dict[str, Any]],
    spike_alerts: Optional[List[Dict[str, Any]]] = None,
    bess_schedule: Optional[List[Dict[str, Any]]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> DispatchSchedule:
    """
    Compute optimal mining ON/OFF schedule.

    Args:
        dam_prices: List of dicts with 'hour_ending' (int 1-24) and 'predicted_price'.
        spike_alerts: Optional list with 'hour_ending', 'spike_probability', 'is_spike'.
        bess_schedule: Optional list with 'hour_ending', 'action' (charge/discharge/idle).
        config: Override config (uses dispatch.yaml if None).

    Returns:
        DispatchSchedule with hourly actions and savings summary.
    """
    if config is None:
        config = load_config()

    mining_cfg = config.get("mining", {})
    bess_cfg = config.get("bess", {})

    breakeven = mining_cfg.get("breakeven_price", 55.0)
    switching_cost = mining_cfg.get("switching_cost", 40.0)
    min_curtail = mining_cfg.get("min_curtail_hours", 2)
    spike_threshold = mining_cfg.get("spike_curtail_threshold", 0.7)
    settlement_point = mining_cfg.get("settlement_point", "HB_WEST")
    load_mw = mining_cfg.get("load_mw", 10.0)
    coordinate_bess = bess_cfg.get("coordinate", True)

    # Index prices by hour_ending (1-24)
    price_by_hour: Dict[int, float] = {}
    for p in dam_prices:
        he = _parse_hour_ending(p.get("hour_ending", 0))
        price_by_hour[he] = float(p.get("predicted_price", 0.0))

    # Index spike alerts by hour
    spike_by_hour: Dict[int, Dict[str, Any]] = {}
    if spike_alerts:
        for s in spike_alerts:
            he = _parse_hour_ending(s.get("hour_ending", 0))
            spike_by_hour[he] = s

    # Index BESS schedule by hour
    bess_by_hour: Dict[int, str] = {}
    if bess_schedule and coordinate_bess:
        for b in bess_schedule:
            he = _parse_hour_ending(b.get("hour_ending", 0))
            bess_by_hour[he] = b.get("action", "idle")

    # --- Step 1: compute per-hour loss (cost above breakeven) ---
    hours = sorted(price_by_hour.keys())
    if not hours:
        hours = list(range(1, 25))

    losses: Dict[int, float] = {}  # positive = losing money running
    for h in hours:
        price = price_by_hour.get(h, 0.0)
        losses[h] = max(0.0, price - breakeven)

    # --- Step 2: force-curtail spike hours ---
    forced_off: set = set()
    spike_hours: List[int] = []
    for h in hours:
        sa = spike_by_hour.get(h, {})
        prob = float(sa.get("spike_probability", 0.0))
        if prob >= spike_threshold or sa.get("is_spike", False):
            forced_off.add(h)
            spike_hours.append(h)

    # --- Step 3: avoid BESS discharge hours (high-value, let battery sell) ---
    for h in hours:
        if bess_by_hour.get(h) == "discharge":
            # If price also above breakeven, prefer curtail
            if losses[h] > 0:
                forced_off.add(h)

    # --- Step 4: optimal curtailment via greedy contiguous blocks ---
    # Mark forced-off hours, then find additional profitable curtail windows
    actions: Dict[int, str] = {h: "ON" for h in hours}
    reasons: Dict[int, str] = {h: "price_below_breakeven" for h in hours}

    for h in forced_off:
        actions[h] = "OFF"
        if h in spike_hours:
            reasons[h] = "spike_alert"
        elif bess_by_hour.get(h) == "discharge":
            reasons[h] = "bess_discharge_coordination"
        else:
            reasons[h] = "forced_curtail"

    # Find contiguous OFF blocks among non-forced hours
    remaining = [h for h in hours if h not in forced_off]
    _apply_greedy_curtailment(
        remaining, losses, actions, reasons,
        switching_cost=switching_cost,
        min_curtail=min_curtail,
    )

    # --- Step 5: build schedule ---
    hour_actions: List[HourAction] = []
    for h in hours:
        price = price_by_hour.get(h, 0.0)
        sa = spike_by_hour.get(h, {})
        hour_actions.append(HourAction(
            hour_ending=h,
            dam_price=round(price, 2),
            action=actions[h],
            reason=reasons[h],
            spike_probability=round(float(sa.get("spike_probability", 0.0)), 3),
            bess_action=bess_by_hour.get(h, ""),
        ))

    # --- Step 6: compute savings ---
    on_hours = [ha for ha in hour_actions if ha.action == "ON"]
    off_hours = [ha for ha in hour_actions if ha.action == "OFF"]

    always_on_cost = sum(price_by_hour.get(h, 0.0) for h in hours) * load_mw
    dispatch_cost = sum(ha.dam_price for ha in on_hours) * load_mw

    # Add switching costs (count ON->OFF and OFF->ON transitions)
    transitions = 0
    prev_action = "ON"
    for ha in hour_actions:
        if ha.action != prev_action:
            transitions += 1
        prev_action = ha.action
    dispatch_cost += (transitions / 2) * switching_cost  # each cycle = 2 transitions

    savings = always_on_cost - dispatch_cost
    on_prices = [ha.dam_price for ha in on_hours]
    avg_on = sum(on_prices) / len(on_prices) if on_prices else 0.0
    all_prices = [price_by_hour.get(h, 0.0) for h in hours]
    peak = max(all_prices) if all_prices else 0.0

    today = date.today().isoformat()

    return DispatchSchedule(
        date=today,
        settlement_point=settlement_point,
        hours=hour_actions,
        hours_to_run=len(on_hours),
        hours_to_curtail=len(off_hours),
        expected_cost_savings=round(savings, 2),
        always_on_cost=round(always_on_cost, 2),
        dispatch_cost=round(dispatch_cost, 2),
        peak_price=round(peak, 2),
        avg_on_price=round(avg_on, 2),
        spike_hours=spike_hours,
    )


def _parse_hour_ending(val: Any) -> int:
    """Parse hour_ending from int or string like '05:00'."""
    if isinstance(val, int):
        return val
    s = str(val)
    if ":" in s:
        return int(s.split(":")[0])
    return int(s)


def _apply_greedy_curtailment(
    candidate_hours: List[int],
    losses: Dict[int, float],
    actions: Dict[int, str],
    reasons: Dict[int, str],
    switching_cost: float,
    min_curtail: int,
) -> None:
    """
    Greedy algorithm: find contiguous blocks where total savings exceed
    switching cost, respecting minimum curtailment duration.
    """
    if not candidate_hours:
        return

    sorted_hours = sorted(candidate_hours)
    n = len(sorted_hours)
    i = 0

    while i < n:
        h = sorted_hours[i]
        if losses[h] <= 0:
            i += 1
            continue

        # Found an hour with loss > 0; try to build a contiguous block
        best_end = -1
        best_savings = 0.0
        cumulative = 0.0

        for j in range(i, n):
            # Only extend contiguously (hours must be adjacent)
            if j > i and sorted_hours[j] != sorted_hours[j - 1] + 1:
                break
            cumulative += losses[sorted_hours[j]]
            net = cumulative - switching_cost
            block_len = j - i + 1
            if net > best_savings and block_len >= min_curtail:
                best_savings = net
                best_end = j

        if best_end >= 0:
            for k in range(i, best_end + 1):
                hk = sorted_hours[k]
                actions[hk] = "OFF"
                reasons[hk] = "price_above_breakeven"
            i = best_end + 1
        else:
            i += 1
