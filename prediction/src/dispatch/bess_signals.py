"""
BESS Arbitrage Signal Service

Generates daily charge/discharge recommendations by combining:
- DAM price predictions (primary arbitrage windows)
- RTM volatility (risk adjustment)
- Spike predictions (hold charge for predicted spikes)
- Mining dispatch coordination (discharge during curtailment)

Tracks rolling PnL and computes risk metrics (VaR, max drawdown).
"""

import json
import logging
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
BESS_DB = DATA_DIR / "bess_pnl.db"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class HourSignal:
    """Single-hour BESS dispatch recommendation."""
    hour_ending: int          # 1-24
    action: str               # charge / discharge / idle
    power_mw: float           # MW (positive)
    soc_pct: float            # state of charge after action (%)
    dam_price: float          # $/MWh
    rtm_volatility: float     # std-dev of recent RTM prices ($/MWh)
    spike_probability: float  # 0-1
    revenue_estimate: float   # $ expected from this hour
    risk_flag: str            # "normal", "spike_hold", "high_volatility"
    mining_curtail: bool      # True if mining should curtail this hour


@dataclass
class DailySignals:
    """Full day of BESS arbitrage signals."""
    date: str
    settlement_point: str
    signals: List[HourSignal]
    total_revenue_estimate: float
    charge_hours: int
    discharge_hours: int
    idle_hours: int
    peak_discharge_price: float
    avg_charge_price: float
    spike_hold_hours: List[int]
    risk_adjusted_revenue: float
    generated_at: str = ""

    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.utcnow().isoformat()


@dataclass
class DailyPnL:
    """Single day PnL record."""
    date: str
    settlement_point: str
    projected_revenue: float
    actual_revenue: Optional[float]
    charge_cost: float
    discharge_revenue: float
    cycles: float
    degradation_cost: float
    net_pnl: float


@dataclass
class RiskMetrics:
    """Portfolio risk metrics over a window."""
    days: int
    total_pnl: float
    avg_daily_pnl: float
    var_95: float             # 95% Value at Risk (daily loss threshold)
    max_drawdown: float       # worst peak-to-trough
    win_rate: float           # fraction of profitable days
    sharpe_ratio: float       # risk-adjusted return (annualized)
    best_day: float
    worst_day: float
    volatility: float         # daily PnL std dev


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def generate_daily_signals(
    dam_prices: List[Dict[str, Any]],
    bess_schedule: List[Dict[str, Any]],
    spike_alerts: Optional[List[Dict[str, Any]]] = None,
    rtm_prices: Optional[List[Dict[str, Any]]] = None,
    mining_schedule: Optional[List[Dict[str, Any]]] = None,
    config: Optional[Dict[str, Any]] = None,
    settlement_point: str = "HB_WEST",
) -> DailySignals:
    """
    Generate daily BESS arbitrage signals combining all prediction sources.

    Args:
        dam_prices: 24 dicts with 'hour_ending' and 'predicted_price'.
        bess_schedule: 24 dicts from LP optimizer with 'hour_ending', 'action',
                       'power_mw', 'soc_pct', 'dam_price'.
        spike_alerts: Optional per-hour spike probabilities.
        rtm_prices: Optional recent RTM prices for volatility.
        mining_schedule: Optional mining dispatch with 'hour_ending', 'action'.
        config: BESS signal config overrides.
        settlement_point: Settlement point name.

    Returns:
        DailySignals with per-hour recommendations and summary.
    """
    cfg = config or {}
    bess_cfg = cfg.get("bess_signals", {})
    spike_hold_threshold = bess_cfg.get("spike_hold_threshold", 0.6)
    volatility_premium = bess_cfg.get("volatility_premium", 0.15)
    battery_efficiency = bess_cfg.get("round_trip_efficiency", 0.90)
    degradation_per_mwh = bess_cfg.get("degradation_cost_per_mwh", 2.0)

    # Index inputs by hour
    dam_by_hour: Dict[int, float] = {}
    for p in dam_prices:
        he = _parse_he(p.get("hour_ending", 0))
        dam_by_hour[he] = float(p.get("predicted_price", 0.0))

    bess_by_hour: Dict[int, Dict[str, Any]] = {}
    for b in bess_schedule:
        he = _parse_he(b.get("hour_ending", 0))
        bess_by_hour[he] = b

    spike_by_hour: Dict[int, float] = {}
    if spike_alerts:
        for s in spike_alerts:
            he = _parse_he(s.get("hour_ending", 0))
            spike_by_hour[he] = float(s.get("spike_probability", 0.0))

    mining_by_hour: Dict[int, str] = {}
    if mining_schedule:
        for m in mining_schedule:
            he = _parse_he(m.get("hour_ending", 0))
            mining_by_hour[he] = m.get("action", "ON")

    # Compute RTM volatility (rolling window std-dev)
    rtm_vol = _compute_rtm_volatility(rtm_prices) if rtm_prices else 0.0

    # Build per-hour signals
    signals: List[HourSignal] = []
    spike_hold_hours: List[int] = []
    total_revenue = 0.0

    for he in range(1, 25):
        bess_entry = bess_by_hour.get(he, {})
        action = bess_entry.get("action", "idle")
        power_mw = float(bess_entry.get("power_mw", 0.0))
        soc_pct = float(bess_entry.get("soc_pct", 50.0))
        dam_price = dam_by_hour.get(he, 0.0)
        spike_prob = spike_by_hour.get(he, 0.0)
        mining_action = mining_by_hour.get(he, "ON")

        # --- Spike-aware SoC management ---
        risk_flag = "normal"
        if spike_prob >= spike_hold_threshold and action == "discharge":
            # Hold charge — don't discharge now, spike coming means higher price
            # Only hold if a future hour has higher spike probability
            future_spikes = [spike_by_hour.get(h, 0.0) for h in range(he + 1, 25)]
            if any(sp >= spike_hold_threshold for sp in future_spikes):
                risk_flag = "spike_hold"
                spike_hold_hours.append(he)
                action = "idle"
                power_mw = 0.0

        # --- High volatility flag ---
        if rtm_vol > dam_price * volatility_premium and risk_flag == "normal":
            risk_flag = "high_volatility"

        # --- Revenue estimate ---
        revenue = 0.0
        if action == "discharge":
            revenue = dam_price * power_mw * 1.0  # 1 hour
            revenue -= degradation_per_mwh * power_mw
        elif action == "charge":
            revenue = -dam_price * power_mw * 1.0

        # Risk-adjust: reduce estimated revenue by volatility factor
        risk_factor = 1.0
        if risk_flag == "high_volatility":
            risk_factor = 1.0 - volatility_premium

        total_revenue += revenue * risk_factor

        # Mining coordination: discharge during mining curtailment
        mining_curtail = mining_action == "OFF"

        signals.append(HourSignal(
            hour_ending=he,
            action=action,
            power_mw=round(power_mw, 3),
            soc_pct=round(soc_pct, 1),
            dam_price=round(dam_price, 2),
            rtm_volatility=round(rtm_vol, 2),
            spike_probability=round(spike_prob, 3),
            revenue_estimate=round(revenue, 2),
            risk_flag=risk_flag,
            mining_curtail=mining_curtail,
        ))

    # Summary stats
    charge_signals = [s for s in signals if s.action == "charge"]
    discharge_signals = [s for s in signals if s.action == "discharge"]
    idle_signals = [s for s in signals if s.action == "idle"]

    peak_dis_price = max((s.dam_price for s in discharge_signals), default=0.0)
    avg_ch_price = (
        sum(s.dam_price for s in charge_signals) / len(charge_signals)
        if charge_signals else 0.0
    )

    raw_revenue = sum(s.revenue_estimate for s in signals)
    risk_adj = raw_revenue * (1.0 - volatility_premium * min(rtm_vol / 50.0, 1.0))

    today = date.today().isoformat()

    return DailySignals(
        date=today,
        settlement_point=settlement_point,
        signals=signals,
        total_revenue_estimate=round(raw_revenue, 2),
        charge_hours=len(charge_signals),
        discharge_hours=len(discharge_signals),
        idle_hours=len(idle_signals),
        peak_discharge_price=round(peak_dis_price, 2),
        avg_charge_price=round(avg_ch_price, 2),
        spike_hold_hours=spike_hold_hours,
        risk_adjusted_revenue=round(risk_adj, 2),
    )


# ---------------------------------------------------------------------------
# PnL tracking (SQLite persistence)
# ---------------------------------------------------------------------------

def _ensure_pnl_db() -> sqlite3.Connection:
    """Create / open the BESS PnL database and ensure schema exists."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(BESS_DB))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bess_daily_pnl (
            date TEXT NOT NULL,
            settlement_point TEXT NOT NULL,
            projected_revenue REAL,
            actual_revenue REAL,
            charge_cost REAL,
            discharge_revenue REAL,
            cycles REAL,
            degradation_cost REAL,
            net_pnl REAL,
            PRIMARY KEY (date, settlement_point)
        )
    """)
    conn.commit()
    return conn


def record_daily_pnl(
    signals: DailySignals,
    actual_prices: Optional[List[Dict[str, Any]]] = None,
    battery_config: Optional[Dict[str, Any]] = None,
) -> DailyPnL:
    """
    Compute and persist daily PnL from signals and (optional) actual prices.

    If actual_prices provided, computes realized PnL; otherwise uses projected.
    """
    cfg = battery_config or {}
    degradation_per_mwh = cfg.get("degradation_cost_per_mwh", 2.0)
    efficiency = cfg.get("round_trip_efficiency", 0.90)

    charge_cost = 0.0
    discharge_revenue = 0.0
    energy_charged = 0.0
    energy_discharged = 0.0

    for sig in signals.signals:
        price = sig.dam_price
        if actual_prices:
            # Use actual price if available
            for ap in actual_prices:
                if _parse_he(ap.get("hour_ending", 0)) == sig.hour_ending:
                    price = float(ap.get("price", sig.dam_price))
                    break

        if sig.action == "charge":
            charge_cost += price * sig.power_mw
            energy_charged += sig.power_mw
        elif sig.action == "discharge":
            discharge_revenue += price * sig.power_mw
            energy_discharged += sig.power_mw

    # Full cycles = min(charged, discharged) / E_max (approximate)
    e_max = cfg.get("E_max_mwh", 10.0)
    cycles = min(energy_charged, energy_discharged) / e_max if e_max > 0 else 0.0
    deg_cost = degradation_per_mwh * energy_discharged

    actual_rev = discharge_revenue * efficiency - charge_cost if actual_prices else None
    projected_rev = signals.total_revenue_estimate
    net = actual_rev if actual_rev is not None else projected_rev - deg_cost

    pnl = DailyPnL(
        date=signals.date,
        settlement_point=signals.settlement_point,
        projected_revenue=round(projected_rev, 2),
        actual_revenue=round(actual_rev, 2) if actual_rev is not None else None,
        charge_cost=round(charge_cost, 2),
        discharge_revenue=round(discharge_revenue, 2),
        cycles=round(cycles, 3),
        degradation_cost=round(deg_cost, 2),
        net_pnl=round(net, 2),
    )

    # Persist
    conn = _ensure_pnl_db()
    try:
        conn.execute(
            """INSERT OR REPLACE INTO bess_daily_pnl
               (date, settlement_point, projected_revenue, actual_revenue,
                charge_cost, discharge_revenue, cycles, degradation_cost, net_pnl)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (pnl.date, pnl.settlement_point, pnl.projected_revenue,
             pnl.actual_revenue, pnl.charge_cost, pnl.discharge_revenue,
             pnl.cycles, pnl.degradation_cost, pnl.net_pnl),
        )
        conn.commit()
    finally:
        conn.close()

    return pnl


def get_rolling_pnl(
    days: int = 7,
    settlement_point: str = "HB_WEST",
) -> List[DailyPnL]:
    """Fetch rolling PnL for the last N days."""
    conn = _ensure_pnl_db()
    try:
        cutoff = (date.today() - timedelta(days=days)).isoformat()
        rows = conn.execute(
            """SELECT date, settlement_point, projected_revenue, actual_revenue,
                      charge_cost, discharge_revenue, cycles, degradation_cost, net_pnl
               FROM bess_daily_pnl
               WHERE settlement_point = ? AND date >= ?
               ORDER BY date""",
            (settlement_point, cutoff),
        ).fetchall()
    finally:
        conn.close()

    return [
        DailyPnL(
            date=r[0], settlement_point=r[1], projected_revenue=r[2],
            actual_revenue=r[3], charge_cost=r[4], discharge_revenue=r[5],
            cycles=r[6], degradation_cost=r[7], net_pnl=r[8],
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------

def compute_risk_metrics(
    days: int = 30,
    settlement_point: str = "HB_WEST",
) -> RiskMetrics:
    """
    Compute risk metrics from PnL history.

    Returns VaR (95%), max drawdown, Sharpe ratio, win rate, etc.
    """
    pnl_records = get_rolling_pnl(days=days, settlement_point=settlement_point)

    if not pnl_records:
        return RiskMetrics(
            days=days, total_pnl=0.0, avg_daily_pnl=0.0, var_95=0.0,
            max_drawdown=0.0, win_rate=0.0, sharpe_ratio=0.0,
            best_day=0.0, worst_day=0.0, volatility=0.0,
        )

    daily_pnls = [r.net_pnl for r in pnl_records]
    n = len(daily_pnls)

    total = sum(daily_pnls)
    avg = total / n
    best = max(daily_pnls)
    worst = min(daily_pnls)
    wins = sum(1 for p in daily_pnls if p > 0)
    win_rate = wins / n

    # Volatility (std dev of daily PnL)
    if n > 1:
        vol = float(np.std(daily_pnls, ddof=1))
    else:
        vol = 0.0

    # Value at Risk — 5th percentile of daily PnL (loss threshold)
    var_95 = float(np.percentile(daily_pnls, 5)) if n >= 2 else 0.0

    # Max drawdown — worst peak-to-trough in cumulative PnL
    cum = np.cumsum(daily_pnls)
    running_max = np.maximum.accumulate(cum)
    drawdowns = cum - running_max
    max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

    # Sharpe ratio (annualized, assuming 365 trading days)
    sharpe = (avg / vol * np.sqrt(365)) if vol > 0 else 0.0

    return RiskMetrics(
        days=n,
        total_pnl=round(total, 2),
        avg_daily_pnl=round(avg, 2),
        var_95=round(var_95, 2),
        max_drawdown=round(max_dd, 2),
        win_rate=round(win_rate, 4),
        sharpe_ratio=round(sharpe, 2),
        best_day=round(best, 2),
        worst_day=round(worst, 2),
        volatility=round(vol, 2),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_he(val: Any) -> int:
    """Parse hour_ending from int or '05:00' format."""
    if isinstance(val, int):
        return val
    s = str(val)
    if ":" in s:
        return int(s.split(":")[0])
    return int(s)


def _compute_rtm_volatility(
    rtm_prices: Optional[List[Dict[str, Any]]],
    window: int = 12,
) -> float:
    """Compute std-dev of most recent RTM prices."""
    if not rtm_prices:
        return 0.0
    prices = [float(p.get("price", p.get("lmp", 0.0))) for p in rtm_prices]
    if len(prices) < 2:
        return 0.0
    recent = prices[-window:]
    return float(np.std(recent, ddof=1))
