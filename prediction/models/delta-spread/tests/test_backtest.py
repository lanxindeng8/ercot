"""Tests for delta-spread backtest logic using small synthetic data."""

import numpy as np
import pandas as pd
import pytest

# Import the backtest module — add parent to path
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest_report import TRANSACTION_COST, TradeLog, run_strategies


class TestTradeLog:
    """Test TradeLog PnL accounting."""

    def test_empty_log(self):
        log = TradeLog(name="empty")
        m = log.metrics()
        assert m["num_trades"] == 0
        assert m["total_pnl"] == 0.0
        assert m["win_rate"] == 0.0

    def test_single_winning_long_trade(self):
        """Long spread when DAM > RTM → profit = spread - cost."""
        log = TradeLog(name="single_win")
        # actual_spread = 10 (DAM > RTM by $10), signal = +1 (long)
        log.add("2025-01-01", signal=1, actual_spread=10.0)
        m = log.metrics()
        assert m["num_trades"] == 1
        expected_pnl = 10.0 - TRANSACTION_COST
        assert abs(m["total_pnl"] - expected_pnl) < 1e-6
        assert m["win_rate"] == 1.0

    def test_single_losing_long_trade(self):
        """Long spread when DAM < RTM → loss."""
        log = TradeLog(name="single_loss")
        log.add("2025-01-01", signal=1, actual_spread=-10.0)
        m = log.metrics()
        expected_pnl = -10.0 - TRANSACTION_COST
        assert abs(m["total_pnl"] - expected_pnl) < 1e-6
        assert m["win_rate"] == 0.0

    def test_short_trade(self):
        """Short spread when DAM < RTM → profit = -spread - cost."""
        log = TradeLog(name="short_win")
        # actual_spread = -10 (RTM > DAM), signal = -1 (short spread)
        log.add("2025-01-01", signal=-1, actual_spread=-10.0)
        m = log.metrics()
        expected_pnl = (-1) * (-10.0) - TRANSACTION_COST  # = 10 - 0.5 = 9.5
        assert abs(m["total_pnl"] - expected_pnl) < 1e-6
        assert m["win_rate"] == 1.0

    def test_transaction_cost_kills_small_spread(self):
        """A tiny spread should result in a loss after transaction costs."""
        log = TradeLog(name="small_spread")
        log.add("2025-01-01", signal=1, actual_spread=0.30)
        m = log.metrics()
        # 0.30 - 0.50 = -0.20
        assert m["total_pnl"] < 0
        assert m["win_rate"] == 0.0

    def test_multiple_trades_pnl(self):
        """Verify total PnL across multiple trades."""
        log = TradeLog(name="multi")
        spreads = [10.0, -5.0, 20.0, -3.0, 8.0]
        signals = [1, -1, 1, -1, 1]
        expected_total = 0.0
        for sig, spread in zip(signals, spreads):
            log.add("2025-01-01", signal=sig, actual_spread=spread)
            expected_total += sig * spread - TRANSACTION_COST
        m = log.metrics()
        assert abs(m["total_pnl"] - expected_total) < 1e-6
        assert m["num_trades"] == 5

    def test_max_drawdown(self):
        """Max drawdown should capture the largest peak-to-trough decline."""
        log = TradeLog(name="drawdown")
        # Cumulative PnL sequence: +9.5, +19, -11, -1.5
        # After tx costs: [9.5, 9.5, -30.5, 9.5]
        log.add("d1", 1, 10.0)   # net +9.5, cum 9.5
        log.add("d2", 1, 10.0)   # net +9.5, cum 19.0
        log.add("d3", 1, -30.0)  # net -30.5, cum -11.5
        log.add("d4", 1, 10.0)   # net +9.5, cum -2.0
        m = log.metrics()
        # Peak was 19.0, trough was -11.5 → drawdown = 30.5
        assert abs(m["max_drawdown"] - 30.5) < 1e-6

    def test_win_rate_calculation(self):
        """Win rate = fraction of trades with positive net PnL."""
        log = TradeLog(name="winrate")
        log.add("d1", 1, 5.0)   # net 4.5 > 0 → win
        log.add("d2", 1, 0.3)   # net -0.2 < 0 → loss
        log.add("d3", -1, -5.0) # net 4.5 > 0 → win
        log.add("d4", 1, -5.0)  # net -5.5 < 0 → loss
        m = log.metrics()
        assert abs(m["win_rate"] - 0.5) < 1e-6

    def test_sharpe_annualization_uses_trade_cadence(self):
        """Sharpe should scale with observed trade frequency, not a fixed hourly factor."""
        log = TradeLog(name="cadence")
        spreads = [2.5, 0.5, 2.5, 0.5]  # net pnls = [2, 0, 2, 0]
        timestamps = pd.to_datetime([
            "2025-01-01 00:00:00",
            "2025-01-31 00:00:00",
            "2025-03-02 00:00:00",
            "2025-04-01 00:00:00",
        ])

        for ts, spread in zip(timestamps, spreads):
            log.add(ts, 1, spread)

        m = log.metrics()
        elapsed_hours = (timestamps[-1] - timestamps[0]).total_seconds() / 3600
        expected_trades_per_year = ((len(spreads) - 1) / elapsed_hours) * 8760
        expected_sharpe = np.array([2.0, 0.0, 2.0, 0.0]).mean() / np.array([2.0, 0.0, 2.0, 0.0]).std()
        expected_sharpe *= np.sqrt(expected_trades_per_year)

        assert m["sharpe_ratio"] == pytest.approx(expected_sharpe)
        assert m["sharpe_ratio"] < np.sqrt(8760)


class TestRunStrategies:
    """Test strategy logic on synthetic prediction data."""

    @pytest.fixture
    def synthetic_predictions(self):
        """Create small synthetic prediction DataFrame."""
        np.random.seed(42)
        n = 100
        actual_spread = np.random.normal(3.0, 15.0, n)
        # Make regression predictions correlated with actual
        cb_pred = actual_spread * 0.5 + np.random.normal(0, 5, n)
        lgb_pred = actual_spread * 0.4 + np.random.normal(0, 6, n)
        cb_proba = 1.0 / (1.0 + np.exp(-cb_pred / 10))  # Sigmoid of prediction

        return pd.DataFrame({
            "delivery_date": pd.date_range("2025-06-01", periods=n, freq="h"),
            "hour_ending": np.tile(np.arange(1, 25), n // 24 + 1)[:n],
            "dam_lmp": 30 + actual_spread / 2,
            "rtm_lmp": 30 - actual_spread / 2,
            "actual_spread": actual_spread,
            "actual_direction": (actual_spread > 0).astype(int),
            "cb_pred_spread": cb_pred,
            "cb_pred_direction": (cb_pred > 0).astype(int),
            "cb_pred_proba": cb_proba,
            "lgb_pred_spread": lgb_pred,
        })

    def test_returns_all_strategies(self, synthetic_predictions):
        results = run_strategies(synthetic_predictions)
        assert len(results) > 0
        names = {r["strategy"] for r in results}
        assert "baseline_always_long" in names
        assert "baseline_always_short" in names
        assert "ensemble_agree" in names

    def test_baselines_trade_every_hour(self, synthetic_predictions):
        results = run_strategies(synthetic_predictions)
        n = len(synthetic_predictions)
        for r in results:
            if r["strategy"].startswith("baseline_always"):
                assert r["num_trades"] == n

    def test_threshold_reduces_trades(self, synthetic_predictions):
        results = run_strategies(synthetic_predictions)
        by_name = {r["strategy"]: r for r in results}
        # Higher threshold → fewer trades
        assert by_name["cb_regression_thresh_0"]["num_trades"] >= by_name["cb_regression_thresh_5"]["num_trades"]
        assert by_name["cb_regression_thresh_5"]["num_trades"] >= by_name["cb_regression_thresh_15"]["num_trades"]

    def test_baselines_are_mirror(self, synthetic_predictions):
        """Always-long and always-short PnL should be negatives (offset by 2x tx costs)."""
        results = run_strategies(synthetic_predictions)
        by_name = {r["strategy"]: r for r in results}
        long_pnl = by_name["baseline_always_long"]["total_pnl"]
        short_pnl = by_name["baseline_always_short"]["total_pnl"]
        n = len(synthetic_predictions)
        # long + short = -2 * n * TRANSACTION_COST (both pay costs)
        expected_sum = -2 * n * TRANSACTION_COST
        assert abs(long_pnl + short_pnl - expected_sum) < 1e-4

    def test_no_look_ahead_bias(self, synthetic_predictions):
        """Verify strategies only use prediction columns, not actual values.

        We scramble actuals — if strategies used actuals, results would change
        for signal generation (they shouldn't since signals come from pred columns).
        """
        df1 = synthetic_predictions.copy()
        df2 = synthetic_predictions.copy()
        # Keep predictions the same but shuffle actual spreads
        shuffled_actual = df2["actual_spread"].values.copy()
        np.random.shuffle(shuffled_actual)
        df2["actual_spread"] = shuffled_actual
        df2["actual_direction"] = (shuffled_actual > 0).astype(int)

        results1 = run_strategies(df1)
        results2 = run_strategies(df2)

        # Same number of trades for every strategy (signals don't depend on actuals)
        for r1, r2 in zip(results1, results2):
            assert r1["num_trades"] == r2["num_trades"], (
                f"Strategy {r1['strategy']}: trade count changed when actuals shuffled"
            )

    def test_all_metrics_present(self, synthetic_predictions):
        results = run_strategies(synthetic_predictions)
        required_keys = {
            "strategy", "num_trades", "total_pnl", "avg_pnl_per_trade",
            "win_rate", "sharpe_ratio", "max_drawdown", "max_drawdown_pct",
            "avg_trade_size",
        }
        for r in results:
            assert required_keys.issubset(r.keys()), f"Missing keys in {r['strategy']}"
