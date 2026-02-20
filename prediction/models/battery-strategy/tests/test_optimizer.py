"""
Unit tests for optimizer module
"""

import sys
from pathlib import Path

# Ensure project root on sys.path for direct execution
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import numpy as np
import csv
from src.battery_config import BatteryConfig, create_default_battery
from src.optimizer import (
    BatteryOptimizer,
    interpolate_hourly_to_5min,
    create_sample_prices
)
from src.visualizer import BatteryVisualizer


class TestBatteryOptimizer:
    """Test BatteryOptimizer class"""

    @pytest.fixture
    def config(self):
        """Create test battery configuration"""
        return BatteryConfig(
            E_max=10.0,
            P_ch_max=2.5,
            P_dis_max=2.5,
            eta_ch=0.95,
            eta_dis=0.95,
            SoC_min=0.1,
            SoC_max=0.9,
            SoC_0=0.5,
            SoC_T_target=0.5
        )

    @pytest.fixture
    def prices(self):
        """Create test price array"""
        hourly = create_sample_prices("peak")
        return interpolate_hourly_to_5min(hourly)

    def test_optimization_runs(self, config, prices):
        """Test that optimization runs without error"""
        optimizer = BatteryOptimizer(config)
        result = optimizer.optimize(prices)

        assert result is not None
        assert len(result.P_ch) == len(prices)
        assert len(result.P_dis) == len(prices)
        assert len(result.SoC) == len(prices)

    def test_power_constraints(self, config, prices):
        """Test that power constraints are satisfied"""
        optimizer = BatteryOptimizer(config)
        result = optimizer.optimize(prices)

        # Check charging power limits
        assert np.all(result.P_ch >= 0)
        assert np.all(result.P_ch <= config.P_ch_max + 1e-6)

        # Check discharging power limits
        assert np.all(result.P_dis >= 0)
        assert np.all(result.P_dis <= config.P_dis_max + 1e-6)

    def test_soc_constraints(self, config, prices):
        """Test that SoC constraints are satisfied"""
        optimizer = BatteryOptimizer(config)
        result = optimizer.optimize(prices)

        # Check SoC bounds
        assert np.all(result.SoC >= config.SoC_min - 1e-6)
        assert np.all(result.SoC <= config.SoC_max + 1e-6)

    def test_terminal_soc(self, config, prices):
        """Test that terminal SoC constraint is satisfied"""
        optimizer = BatteryOptimizer(config)
        result = optimizer.optimize(prices)

        # Check terminal SoC
        assert abs(result.SoC[-1] - config.SoC_T_target) < 1e-4

    def test_flat_prices_no_arbitrage(self, config):
        """Test that flat prices result in no trading"""
        flat_prices = np.full(288, 50.0)
        optimizer = BatteryOptimizer(config)
        result = optimizer.optimize(flat_prices)

        # With flat prices and round-trip losses, should not trade
        # (or trade very minimally)
        total_trading = np.sum(result.P_ch) + np.sum(result.P_dis)
        assert total_trading < 1.0  # Very small or zero

    def test_positive_revenue_with_price_spread(self, config, prices):
        """Test that positive revenue is achieved with price spread"""
        optimizer = BatteryOptimizer(config)
        result = optimizer.optimize(prices)

        # With significant price spread, should achieve positive revenue
        assert result.total_revenue > 0

    def test_charges_at_low_prices(self, config):
        """Test that battery charges during low price periods"""
        # Create prices: low then high
        hourly = np.concatenate([
            np.full(12, 20.0),  # Low prices first 12 hours
            np.full(12, 80.0)   # High prices last 12 hours
        ])
        prices = interpolate_hourly_to_5min(hourly)

        optimizer = BatteryOptimizer(config)
        result = optimizer.optimize(prices)

        # Should charge more in first half (low prices)
        first_half_charge = np.sum(result.P_ch[:144])
        second_half_charge = np.sum(result.P_ch[144:])
        assert first_half_charge > second_half_charge

    def test_discharges_at_high_prices(self, config):
        """Test that battery discharges during high price periods"""
        # Create prices: low then high
        hourly = np.concatenate([
            np.full(12, 20.0),  # Low prices first 12 hours
            np.full(12, 80.0)   # High prices last 12 hours
        ])
        prices = interpolate_hourly_to_5min(hourly)

        optimizer = BatteryOptimizer(config)
        result = optimizer.optimize(prices)

        # Should discharge more in second half (high prices)
        first_half_discharge = np.sum(result.P_dis[:144])
        second_half_discharge = np.sum(result.P_dis[144:])
        assert second_half_discharge > first_half_discharge

    def test_hourly_csv_to_5min_runs(self, config):
        """Test optimize with hourly DAM CSV (interpolated to 5min)"""
        data_path = Path(__file__).resolve().parent / "input" / "data.csv"
        hourly = np.loadtxt(data_path, skiprows=1)
        prices = interpolate_hourly_to_5min(hourly)

        optimizer = BatteryOptimizer(config)
        result = optimizer.optimize(prices)

        assert result.status == "Optimal"
        assert not np.isnan(result.P_ch).all()
        assert len(result.P_ch) == len(prices) == 288
        # SOC bounds still respected
        assert np.nanmin(result.SoC) >= config.SoC_min - 1e-6
        assert np.nanmax(result.SoC) <= config.SoC_max + 1e-6


class TestInterpolateHourlyTo5Min:
    """Test price interpolation function"""

    def test_interpolation_length(self):
        """Test that interpolation produces correct length"""
        hourly = create_sample_prices("peak")
        five_min = interpolate_hourly_to_5min(hourly)

        assert len(five_min) == 288  # 24 * 12

    def test_interpolation_endpoints(self):
        """Test that interpolation preserves endpoints"""
        hourly = create_sample_prices("peak")
        five_min = interpolate_hourly_to_5min(hourly)

        # First value should be close to first hourly value
        assert abs(five_min[0] - hourly[0]) < 1e-6

    def test_invalid_length(self):
        """Test that invalid length raises error"""
        with pytest.raises(ValueError, match="Expected 24 hourly prices"):
            interpolate_hourly_to_5min(np.array([1, 2, 3]))


class TestCreateSamplePrices:
    """Test sample price generation"""

    def test_peak_pattern(self):
        """Test peak price pattern"""
        prices = create_sample_prices("peak")

        assert len(prices) == 24
        # Peak hours should have higher prices
        peak_avg = np.mean(prices[8:20])
        night_avg = np.mean(prices[0:6])
        assert peak_avg > night_avg

    def test_flat_pattern(self):
        """Test flat price pattern"""
        prices = create_sample_prices("flat")

        assert len(prices) == 24
        # All prices should be the same
        assert np.std(prices) < 1e-6

    def test_invalid_pattern(self):
        """Test that invalid pattern raises error"""
        with pytest.raises(ValueError, match="Unknown pattern"):
            create_sample_prices("invalid")


def run_example_with_input_csv():
    """
    Run optimization on tests/input/data.csv and write outputs to tests/output.

    This uses the same algorithm as the src modules and produces:
      - PNG figures in tests/output/
      - A CSV with 288 points containing price and decision variables.
    """
    # Resolve project root (same ROOT as above)
    input_path = ROOT / "tests" / "input" / "data.csv"
    output_dir = ROOT / "tests" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load 24-hour hourly prices
    hourly = np.loadtxt(input_path, skiprows=1)
    prices_5min = interpolate_hourly_to_5min(hourly)

    # Use default battery config for consistency with examples
    config = create_default_battery()
    optimizer = BatteryOptimizer(config)
    result = optimizer.optimize(prices_5min, verbose=False)

    # Visualization
    visualizer = BatteryVisualizer(config)
    visualizer.plot_optimization_results(
        result,
        prices_5min,
        save_path=str(output_dir / "example1_results.png"),
    )
    visualizer.plot_power_vs_price(
        result,
        prices_5min,
        save_path=str(output_dir / "example1_power_vs_price.png"),
    )

    # Write 288-point time series CSV with price and decision variables
    csv_path = output_dir / "optimization_timeseries.csv"
    T = len(prices_5min)
    time_hours = np.arange(T) * config.delta_t

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "time_index",
                "time_hours",
                "price",
                "P_ch",
                "P_dis",
                "SoC",
                "ramp_up",
                "ramp_down",
            ]
        )
        for t in range(T):
            writer.writerow(
                [
                    t,
                    time_hours[t],
                    prices_5min[t],
                    result.P_ch[t],
                    result.P_dis[t],
                    result.SoC[t],
                    result.ramp_up[t],
                    result.ramp_down[t],
                ]
            )


def _main():
    """
    When run directly:
      1. Execute pytest on this file.
      2. Run the example using tests/input/data.csv and write outputs.
    """
    import pytest as _pytest

    exit_code = _pytest.main([__file__])

    try:
        run_example_with_input_csv()
    except Exception as exc:
        # Do not change test exit code if example fails; just report.
        print(f"Error while running example with input CSV: {exc}")

    raise SystemExit(exit_code)


if __name__ == "__main__":
    _main()
