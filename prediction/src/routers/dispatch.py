"""Dispatch endpoints: mining schedule/savings, alerts config, BESS signals/pnl/risk."""

import logging

from fastapi import APIRouter, HTTPException, Query

from ..models.dam_v2_predictor import get_dam_v2_predictor
from ..models.spike_predictor import get_spike_predictor
from ..models.bess_predictor import get_bess_predictor
from ..dispatch.mining_dispatch import compute_dispatch, load_config as load_dispatch_config
from ..dispatch.alert_service import get_alert_service
from ..dispatch.bess_signals import (
    generate_daily_signals,
    record_daily_pnl,
    get_rolling_pnl,
    compute_risk_metrics,
)
from ..helpers import (
    fetch_and_compute_features,
    normalize_settlement_point,
    latest_complete_delivery_rows,
)
from ..schemas import AlertConfigRequest

log = logging.getLogger(__name__)

router = APIRouter(tags=["Dispatch"])


# ---------------------------------------------------------------------------
# Mining Dispatch
# ---------------------------------------------------------------------------

@router.get("/dispatch/mining/schedule")
async def dispatch_mining_schedule(
    settlement_point: str = Query(default="HB_WEST", description="Settlement point for DAM prices"),
):
    """
    Today's mining ON/OFF dispatch schedule.

    Combines DAM predictions, spike alerts, and BESS schedule to compute
    optimal hours to run or curtail mining operations.
    """
    normalized_sp = normalize_settlement_point(settlement_point)

    dam_predictor = get_dam_v2_predictor()
    if not dam_predictor.is_ready():
        raise HTTPException(status_code=503, detail="DAM models not loaded")
    if not dam_predictor.has_model(normalized_sp):
        raise HTTPException(
            status_code=404,
            detail=dam_predictor.missing_model_message(normalized_sp),
        )

    try:
        features_df = fetch_and_compute_features(normalized_sp)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Data pipeline failed: {e}")

    target_rows = latest_complete_delivery_rows(features_df)
    if target_rows.empty:
        raise HTTPException(status_code=404, detail="Insufficient data for dispatch")

    try:
        sp_key = normalized_sp.lower()

        # DAM predictions
        dam_predictions = dam_predictor.predict(target_rows, sp_key)
        dam_prices = [
            {"hour_ending": p.hour_ending, "predicted_price": p.predicted_price}
            for p in dam_predictions[:24]
        ]

        # Spike alerts (best-effort)
        spike_alerts = None
        try:
            spike_pred = get_spike_predictor()
            if spike_pred.is_ready():
                alerts = spike_pred.predict(target_rows, sp_key)
                spike_alerts = [
                    {
                        "hour_ending": i + 1,
                        "spike_probability": a.spike_probability,
                        "is_spike": a.is_spike,
                    }
                    for i, a in enumerate(alerts)
                ]
        except Exception as exc:
            log.warning("Spike predictor unavailable for dispatch: %s", exc)

        # BESS schedule (best-effort)
        bess_schedule = None
        try:
            bess = get_bess_predictor()
            if bess.is_ready():
                bess_prices = [p["predicted_price"] for p in dam_prices]
                if len(bess_prices) == 24:
                    bess_result = bess.optimize(bess_prices)
                    bess_schedule = [
                        {"hour_ending": e.hour_ending, "action": e.action}
                        for e in bess_result.schedule
                    ]
        except Exception as exc:
            log.warning("BESS optimizer unavailable for dispatch: %s", exc)

        config = load_dispatch_config()
        config.setdefault("mining", {})["settlement_point"] = normalized_sp
        schedule = compute_dispatch(dam_prices, spike_alerts, bess_schedule, config)

        return {
            "status": "success",
            "settlement_point": normalized_sp,
            "generated_at": schedule.generated_at,
            "date": schedule.date,
            "schedule": [
                {
                    "hour_ending": f"{ha.hour_ending:02d}:00",
                    "dam_price": ha.dam_price,
                    "action": ha.action,
                    "reason": ha.reason,
                    "spike_probability": ha.spike_probability,
                    "bess_action": ha.bess_action,
                }
                for ha in schedule.hours
            ],
            "summary": {
                "hours_to_run": schedule.hours_to_run,
                "hours_to_curtail": schedule.hours_to_curtail,
                "expected_cost_savings": schedule.expected_cost_savings,
                "always_on_cost": schedule.always_on_cost,
                "dispatch_cost": schedule.dispatch_cost,
                "peak_price": schedule.peak_price,
                "avg_on_price": schedule.avg_on_price,
                "spike_hours": schedule.spike_hours,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dispatch/mining/savings")
async def dispatch_mining_savings(
    settlement_point: str = Query(default="HB_WEST", description="Settlement point"),
):
    """
    Estimated savings from dispatch vs always-on mining.

    Returns cost comparison and savings breakdown.
    """
    result = await dispatch_mining_schedule(settlement_point)
    summary = result["summary"]
    return {
        "status": "success",
        "settlement_point": result["settlement_point"],
        "date": result["date"],
        "always_on_cost": summary["always_on_cost"],
        "dispatch_cost": summary["dispatch_cost"],
        "expected_cost_savings": summary["expected_cost_savings"],
        "hours_to_run": summary["hours_to_run"],
        "hours_to_curtail": summary["hours_to_curtail"],
        "peak_price": summary["peak_price"],
        "avg_on_price": summary["avg_on_price"],
        "savings_pct": round(
            summary["expected_cost_savings"] / summary["always_on_cost"] * 100, 1
        ) if summary["always_on_cost"] > 0 else 0.0,
    }


@router.post("/dispatch/alerts/config")
async def configure_alerts(req: AlertConfigRequest):
    """
    Configure alert preferences for the Telegram alert service.

    Update chat IDs, spike thresholds, cooldown periods, or bot token.
    """
    svc = get_alert_service()
    updated = svc.update_config(
        chat_ids=req.chat_ids,
        spike_alert_threshold=req.spike_alert_threshold,
        spike_cooldown_minutes=req.spike_cooldown_minutes,
        bot_token=req.bot_token,
    )
    return {"status": "success", "config": updated}


# ---------------------------------------------------------------------------
# BESS Dispatch Signals
# ---------------------------------------------------------------------------

@router.get("/dispatch/bess/daily-signals")
async def dispatch_bess_daily_signals(
    settlement_point: str = Query(default="HB_WEST", description="Settlement point"),
):
    """
    Today's BESS charge/discharge recommendations.

    Combines DAM predictions, BESS LP optimizer, spike alerts, RTM volatility,
    and mining dispatch to produce risk-adjusted arbitrage signals.
    """
    normalized_sp = normalize_settlement_point(settlement_point)

    dam_predictor = get_dam_v2_predictor()
    if not dam_predictor.is_ready():
        raise HTTPException(status_code=503, detail="DAM models not loaded")
    if not dam_predictor.has_model(normalized_sp):
        raise HTTPException(
            status_code=404,
            detail=dam_predictor.missing_model_message(normalized_sp),
        )

    bess = get_bess_predictor()
    if not bess.is_ready():
        raise HTTPException(status_code=503, detail="BESS optimizer not loaded")

    try:
        features_df = fetch_and_compute_features(normalized_sp)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Data pipeline failed: {e}")

    target_rows = latest_complete_delivery_rows(features_df)
    if target_rows.empty:
        raise HTTPException(status_code=404, detail="Insufficient data for BESS signals")

    try:
        sp_key = normalized_sp.lower()

        # DAM predictions
        dam_predictions = dam_predictor.predict(target_rows, sp_key)
        dam_prices = [
            {"hour_ending": p.hour_ending, "predicted_price": p.predicted_price}
            for p in dam_predictions[:24]
        ]

        # BESS LP schedule
        bess_prices = [p["predicted_price"] for p in dam_prices]
        if len(bess_prices) < 24:
            raise HTTPException(status_code=500, detail=f"Only {len(bess_prices)} DAM prices, need 24")
        bess_result = bess.optimize(bess_prices)
        bess_schedule = [
            {
                "hour_ending": e.hour_ending,
                "action": e.action,
                "power_mw": e.power_mw,
                "soc_pct": e.soc_pct,
                "dam_price": e.dam_price,
            }
            for e in bess_result.schedule
        ]

        # Spike alerts (best-effort)
        spike_alerts = None
        try:
            spike_pred = get_spike_predictor()
            if spike_pred.is_ready():
                alerts = spike_pred.predict(target_rows, sp_key)
                spike_alerts = [
                    {"hour_ending": i + 1, "spike_probability": a.spike_probability, "is_spike": a.is_spike}
                    for i, a in enumerate(alerts)
                ]
        except Exception as exc:
            log.warning("Spike predictor unavailable for BESS signals: %s", exc)

        # Mining dispatch (best-effort, for coordination)
        mining_schedule = None
        try:
            config = load_dispatch_config()
            config.setdefault("mining", {})["settlement_point"] = normalized_sp
            mining_result = compute_dispatch(dam_prices, spike_alerts, bess_schedule=[
                {"hour_ending": b["hour_ending"], "action": b["action"]} for b in bess_schedule
            ], config=config)
            mining_schedule = [
                {"hour_ending": ha.hour_ending, "action": ha.action}
                for ha in mining_result.hours
            ]
        except Exception as exc:
            log.warning("Mining dispatch unavailable for BESS signals: %s", exc)

        signals = generate_daily_signals(
            dam_prices=dam_prices,
            bess_schedule=bess_schedule,
            spike_alerts=spike_alerts,
            mining_schedule=mining_schedule,
            settlement_point=normalized_sp,
        )

        # Record PnL
        try:
            record_daily_pnl(signals)
        except Exception as exc:
            log.warning("Failed to record BESS PnL: %s", exc)

        return {
            "status": "success",
            "settlement_point": normalized_sp,
            "date": signals.date,
            "generated_at": signals.generated_at,
            "summary": {
                "total_revenue_estimate": signals.total_revenue_estimate,
                "risk_adjusted_revenue": signals.risk_adjusted_revenue,
                "charge_hours": signals.charge_hours,
                "discharge_hours": signals.discharge_hours,
                "idle_hours": signals.idle_hours,
                "peak_discharge_price": signals.peak_discharge_price,
                "avg_charge_price": signals.avg_charge_price,
                "spike_hold_hours": signals.spike_hold_hours,
            },
            "signals": [
                {
                    "hour_ending": f"{s.hour_ending:02d}:00",
                    "action": s.action,
                    "power_mw": s.power_mw,
                    "soc_pct": s.soc_pct,
                    "dam_price": s.dam_price,
                    "rtm_volatility": s.rtm_volatility,
                    "spike_probability": s.spike_probability,
                    "revenue_estimate": s.revenue_estimate,
                    "risk_flag": s.risk_flag,
                    "mining_curtail": s.mining_curtail,
                }
                for s in signals.signals
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dispatch/bess/pnl")
async def dispatch_bess_pnl(
    days: int = Query(default=7, ge=1, le=90, description="Number of days to look back"),
    settlement_point: str = Query(default="HB_WEST", description="Settlement point"),
):
    """
    Rolling BESS PnL with daily breakdown.

    Returns per-day charge cost, discharge revenue, cycles, degradation,
    and net PnL for the requested window.
    """
    records = get_rolling_pnl(days=days, settlement_point=settlement_point)

    total_pnl = sum(r.net_pnl for r in records)
    total_cycles = sum(r.cycles for r in records)
    total_charge_cost = sum(r.charge_cost for r in records)
    total_discharge_rev = sum(r.discharge_revenue for r in records)

    return {
        "status": "success",
        "settlement_point": settlement_point,
        "days_requested": days,
        "days_available": len(records),
        "summary": {
            "total_pnl": round(total_pnl, 2),
            "total_cycles": round(total_cycles, 3),
            "total_charge_cost": round(total_charge_cost, 2),
            "total_discharge_revenue": round(total_discharge_rev, 2),
            "avg_daily_pnl": round(total_pnl / len(records), 2) if records else 0.0,
        },
        "daily": [
            {
                "date": r.date,
                "projected_revenue": r.projected_revenue,
                "actual_revenue": r.actual_revenue,
                "charge_cost": r.charge_cost,
                "discharge_revenue": r.discharge_revenue,
                "cycles": r.cycles,
                "degradation_cost": r.degradation_cost,
                "net_pnl": r.net_pnl,
            }
            for r in records
        ],
    }


@router.get("/dispatch/bess/risk")
async def dispatch_bess_risk(
    days: int = Query(default=30, ge=7, le=365, description="Risk window in days"),
    settlement_point: str = Query(default="HB_WEST", description="Settlement point"),
):
    """
    BESS risk metrics: VaR, max drawdown, Sharpe ratio, win rate.

    Computed from historical PnL over the requested window.
    """
    metrics = compute_risk_metrics(days=days, settlement_point=settlement_point)
    return {
        "status": "success",
        "settlement_point": settlement_point,
        "risk": {
            "days": metrics.days,
            "total_pnl": metrics.total_pnl,
            "avg_daily_pnl": metrics.avg_daily_pnl,
            "var_95": metrics.var_95,
            "max_drawdown": metrics.max_drawdown,
            "win_rate": metrics.win_rate,
            "sharpe_ratio": metrics.sharpe_ratio,
            "best_day": metrics.best_day,
            "worst_day": metrics.worst_day,
            "volatility": metrics.volatility,
        },
    }
