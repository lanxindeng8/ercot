#!/usr/bin/env python3
"""Sprint 2 Task 3: Delta-spread backtesting & validation report.

Loads model predictions on test data (2025-2026), simulates trading strategies
with realistic costs, and outputs PnL metrics as JSON + summary table.

Usage:
    # First run train_and_eval.py to generate predictions, then:
    python prediction/models/delta-spread/backtest_report.py [--settlement-point hb_west]
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Trading assumptions
TRANSACTION_COST = 0.50  # USD/MWh round trip
POSITION_SIZE_MWH = 1.0  # Normalize to 1 MWh for per-unit metrics
ANNUAL_TRADING_HOURS = 8760


@dataclass
class TradeLog:
    """Accumulates trade-level PnL for a single strategy."""

    name: str
    pnls: list = field(default_factory=list)
    signals: list = field(default_factory=list)
    timestamps: list = field(default_factory=list)

    def add(self, timestamp, signal: int, actual_spread: float):
        """Record a trade. signal: +1 = long spread (bet DAM>RTM), -1 = short."""
        gross_pnl = signal * actual_spread * POSITION_SIZE_MWH
        net_pnl = gross_pnl - TRANSACTION_COST * POSITION_SIZE_MWH
        self.pnls.append(net_pnl)
        self.signals.append(signal)
        self.timestamps.append(timestamp)

    def metrics(self) -> dict:
        if not self.pnls:
            return {
                "strategy": self.name,
                "num_trades": 0,
                "total_pnl": 0.0,
                "avg_pnl_per_trade": 0.0,
                "win_rate": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "max_drawdown_pct": 0.0,
                "avg_trade_size": 0.0,
            }

        pnls = np.array(self.pnls)
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        peak_val = running_max[np.argmax(drawdown)] if len(drawdown) > 0 else 0.0

        # Annualize by the observed trade cadence instead of always assuming
        # hourly trading. This keeps sparse strategies from being overstated.
        trades_per_year = 0.0
        if len(self.timestamps) > 1:
            ts = pd.to_datetime(pd.Series(self.timestamps), errors="coerce").dropna().sort_values()
            if len(ts) > 1:
                elapsed_hours = (ts.iloc[-1] - ts.iloc[0]).total_seconds() / 3600
            else:
                elapsed_hours = 0
            if elapsed_hours > 0:
                trades_per_year = ((len(pnls) - 1) / elapsed_hours) * ANNUAL_TRADING_HOURS
        mean_pnl = pnls.mean()
        std_pnl = pnls.std() if len(pnls) > 1 else 1.0
        sharpe = (mean_pnl / std_pnl) * np.sqrt(trades_per_year) if std_pnl > 0 else 0.0

        return {
            "strategy": self.name,
            "num_trades": len(pnls),
            "total_pnl": float(cumulative[-1]),
            "avg_pnl_per_trade": float(mean_pnl),
            "win_rate": float((pnls > 0).mean()),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(drawdown.max()),
            "max_drawdown_pct": float(drawdown.max() / peak_val * 100) if peak_val > 0 else 0.0,
            "avg_trade_size": float(np.abs(pnls).mean()),
        }


def run_strategies(df: pd.DataFrame) -> list[dict]:
    """Run all trading strategies. No look-ahead: signals use only prediction columns."""
    strategies = []

    # Strategy 1: CatBoost regression — trade when |predicted spread| > threshold
    for threshold in [0, 3, 5, 10, 15]:
        log = TradeLog(name=f"cb_regression_thresh_{threshold}")
        for _, row in df.iterrows():
            pred = row["cb_pred_spread"]
            if abs(pred) > threshold:
                signal = 1 if pred > 0 else -1
                log.add(row["delivery_date"], signal, row["actual_spread"])
        strategies.append(log.metrics())

    # Strategy 2: CatBoost classifier — trade based on predicted direction + probability
    for prob_threshold in [0.5, 0.55, 0.6, 0.65, 0.7]:
        log = TradeLog(name=f"cb_classifier_prob_{prob_threshold}")
        for _, row in df.iterrows():
            p = row["cb_pred_proba"]
            if p > prob_threshold:
                log.add(row["delivery_date"], 1, row["actual_spread"])
            elif (1 - p) > prob_threshold:
                log.add(row["delivery_date"], -1, row["actual_spread"])
        strategies.append(log.metrics())

    # Strategy 3: LightGBM regression — trade when |predicted spread| > threshold
    for threshold in [0, 5, 10, 15]:
        log = TradeLog(name=f"lgb_regression_thresh_{threshold}")
        for _, row in df.iterrows():
            pred = row["lgb_pred_spread"]
            if abs(pred) > threshold:
                signal = 1 if pred > 0 else -1
                log.add(row["delivery_date"], signal, row["actual_spread"])
        strategies.append(log.metrics())

    # Strategy 4: Ensemble — both models agree on direction
    log = TradeLog(name="ensemble_agree")
    for _, row in df.iterrows():
        cb_dir = 1 if row["cb_pred_spread"] > 0 else -1
        lgb_dir = 1 if row["lgb_pred_spread"] > 0 else -1
        cls_dir = 1 if row["cb_pred_proba"] > 0.5 else -1
        if cb_dir == lgb_dir == cls_dir:
            log.add(row["delivery_date"], cb_dir, row["actual_spread"])
    strategies.append(log.metrics())

    # Strategy 5: Ensemble high-confidence — all agree + regression |pred| > 5
    log = TradeLog(name="ensemble_high_conf")
    for _, row in df.iterrows():
        cb_dir = 1 if row["cb_pred_spread"] > 0 else -1
        lgb_dir = 1 if row["lgb_pred_spread"] > 0 else -1
        if cb_dir == lgb_dir and abs(row["cb_pred_spread"]) > 5:
            log.add(row["delivery_date"], cb_dir, row["actual_spread"])
    strategies.append(log.metrics())

    # Baseline: always short spread (bet DAM < RTM, i.e. RTM will be lower)
    log = TradeLog(name="baseline_always_short")
    for _, row in df.iterrows():
        log.add(row["delivery_date"], -1, row["actual_spread"])
    strategies.append(log.metrics())

    # Baseline: always long spread
    log = TradeLog(name="baseline_always_long")
    for _, row in df.iterrows():
        log.add(row["delivery_date"], 1, row["actual_spread"])
    strategies.append(log.metrics())

    return strategies


def print_summary(results: list[dict]):
    """Print formatted summary table."""
    print("\n" + "=" * 110)
    print(f"{'Strategy':<30} {'Trades':>7} {'Total PnL':>12} {'Avg PnL':>10} {'Win Rate':>9} {'Sharpe':>8} {'Max DD':>10}")
    print("-" * 110)
    for r in results:
        print(
            f"{r['strategy']:<30} "
            f"{r['num_trades']:>7d} "
            f"${r['total_pnl']:>10,.0f} "
            f"${r['avg_pnl_per_trade']:>8.2f} "
            f"{r['win_rate']:>8.1%} "
            f"{r['sharpe_ratio']:>8.2f} "
            f"${r['max_drawdown']:>8,.0f}"
        )
    print("=" * 110)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--settlement-point", default="hb_west")
    args = parser.parse_args()
    sp = args.settlement_point

    pred_path = CHECKPOINT_DIR / f"{sp}_predictions.parquet"
    if not pred_path.exists():
        print(f"ERROR: Predictions not found at {pred_path}")
        print("Run train_and_eval.py first.")
        return 1

    print(f"=== Backtest Report: {sp.upper()} ===\n")
    df = pd.read_parquet(pred_path)
    print(f"Test data: {len(df)} hours, {df['delivery_date'].min()} to {df['delivery_date'].max()}")
    print(f"Actual spread — mean: ${df['actual_spread'].mean():.2f}, std: ${df['actual_spread'].std():.2f}")
    print(f"Transaction cost: ${TRANSACTION_COST}/MWh round trip\n")

    results = run_strategies(df)
    print_summary(results)

    # Save JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / f"{sp}_backtest_report.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "settlement_point": sp,
                "test_period": {
                    "start": str(df["delivery_date"].min()),
                    "end": str(df["delivery_date"].max()),
                },
                "num_test_hours": len(df),
                "transaction_cost_per_mwh": TRANSACTION_COST,
                "strategies": results,
            },
            f,
            indent=2,
        )
    print(f"\nJSON report saved to {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
