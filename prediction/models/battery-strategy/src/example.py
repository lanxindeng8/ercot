"""
Example usage of battery energy storage optimization

This script demonstrates how to use the battery optimization system
for electricity price arbitrage.
"""

import numpy as np
import sys
import os

# Support both direct execution and module import
if __name__ == "__main__":
    # When run directly from src/ directory
    from battery_config import BatteryConfig, create_default_battery
    from optimizer import BatteryOptimizer, interpolate_hourly_to_5min, create_sample_prices
    from visualizer import BatteryVisualizer
else:
    # When imported as a module
    from .battery_config import BatteryConfig, create_default_battery
    from .optimizer import BatteryOptimizer, interpolate_hourly_to_5min, create_sample_prices
    from .visualizer import BatteryVisualizer


def example_basic_optimization():
    """
    Basic example: optimize battery for a day with peak pricing pattern
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Optimization with Peak Pricing Pattern")
    print("="*70 + "\n")

    # 1. Create battery configuration
    config = create_default_battery()
    print("Battery Configuration:")
    print(f"  Capacity: {config.E_max} MWh")
    print(f"  Max Charge/Discharge Power: {config.P_ch_max}/{config.P_dis_max} MW")
    print(f"  Charging/Discharging Efficiency: {config.eta_ch:.0%}/{config.eta_dis:.0%}")
    print(f"  SoC Range: [{config.SoC_min:.0%}, {config.SoC_max:.0%}]")
    print(f"  Initial SoC: {config.SoC_0:.0%}\n")

    # 2. Create sample hourly prices (24 hours)
    hourly_prices = create_sample_prices(pattern="peak")
    print(f"Hourly prices (24 hours): min=${hourly_prices.min():.2f}, "
          f"max=${hourly_prices.max():.2f}, avg=${hourly_prices.mean():.2f}\n")

    # 3. Interpolate to 5-minute resolution (288 intervals)
    prices_5min = interpolate_hourly_to_5min(hourly_prices)
    print(f"Interpolated to 5-minute resolution: {len(prices_5min)} intervals\n")

    # 4. Run optimization
    optimizer = BatteryOptimizer(config)
    print("Running LP optimization...")
    result = optimizer.optimize(prices_5min, verbose=False)
    print(f"Optimization completed: {result.status}\n")

    # 5. Display results
    visualizer = BatteryVisualizer(config)
    visualizer.print_summary(result, prices_5min)

    # 6. Create visualizations
    print("\nGenerating visualizations...")
    visualizer.plot_optimization_results(
        result,
        prices_5min,
        save_path="../docs/example1_results.png"
    )
    visualizer.plot_power_vs_price(
        result,
        prices_5min,
        save_path="../docs/example1_power_vs_price.png"
    )


def example_different_price_patterns():
    """
    Compare optimization results across different price patterns
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Comparison Across Different Price Patterns")
    print("="*70 + "\n")

    config = create_default_battery()
    optimizer = BatteryOptimizer(config)
    visualizer = BatteryVisualizer(config)

    patterns = ["peak", "valley", "volatile", "flat"]
    results = {}

    for pattern in patterns:
        print(f"\nOptimizing for '{pattern}' price pattern...")
        hourly_prices = create_sample_prices(pattern=pattern)
        prices_5min = interpolate_hourly_to_5min(hourly_prices)

        result = optimizer.optimize(prices_5min, verbose=False)
        results[pattern] = {
            "result": result,
            "prices": prices_5min,
            "revenue": result.total_revenue
        }

        print(f"  Revenue: ${result.total_revenue:.2f}")

    # Summary comparison
    print("\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)
    for pattern in patterns:
        print(f"{pattern:12s}: ${results[pattern]['revenue']:8.2f}")
    print("="*70)


def example_custom_battery():
    """
    Example with custom battery configuration
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Custom Battery Configuration")
    print("="*70 + "\n")

    # Create larger battery with different parameters
    config = BatteryConfig(
        E_max=50.0,  # 50 MWh - larger capacity
        P_ch_max=10.0,  # 10 MW charging
        P_dis_max=10.0,  # 10 MW discharging
        eta_ch=0.92,  # Slightly lower efficiency
        eta_dis=0.92,
        SoC_min=0.2,  # Higher minimum (battery health)
        SoC_max=0.8,  # Lower maximum (battery health)
        SoC_0=0.5,
        SoC_T_target=0.5,
        delta_t=5.0 / 60.0
    )

    print("Custom Battery Configuration:")
    print(f"  Capacity: {config.E_max} MWh")
    print(f"  Max Power: {config.P_ch_max} MW")
    print(f"  Efficiency: {config.eta_ch:.1%}")
    print(f"  SoC Range: [{config.SoC_min:.0%}, {config.SoC_max:.0%}]\n")

    # Use volatile price pattern
    hourly_prices = create_sample_prices(pattern="volatile")
    prices_5min = interpolate_hourly_to_5min(hourly_prices)

    optimizer = BatteryOptimizer(config)
    result = optimizer.optimize(prices_5min, verbose=False)

    visualizer = BatteryVisualizer(config)
    visualizer.print_summary(result, prices_5min)


def example_no_end_constraint():
    """
    Example without terminal SoC constraint
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Optimization Without End SoC Constraint")
    print("="*70 + "\n")

    # Battery without end constraint (can end at any SoC)
    config = BatteryConfig(
        E_max=10.0,
        P_ch_max=2.5,
        P_dis_max=2.5,
        eta_ch=0.95,
        eta_dis=0.95,
        SoC_min=0.1,
        SoC_max=0.9,
        SoC_0=0.5,
        SoC_T_target=None,  # No end constraint
        delta_t=5.0 / 60.0
    )

    print("Battery without terminal SoC constraint\n")

    hourly_prices = create_sample_prices(pattern="peak")
    prices_5min = interpolate_hourly_to_5min(hourly_prices)

    optimizer = BatteryOptimizer(config)
    result = optimizer.optimize(prices_5min, verbose=False)

    visualizer = BatteryVisualizer(config)
    visualizer.print_summary(result, prices_5min)

    print("\nNote: Without end constraint, battery may end at different SoC")
    print(f"Final SoC: {result.SoC[-1]:.1%} (Initial: {config.SoC_0:.1%})")


def main():
    """
    Run all examples
    """
    print("\n" + "="*70)
    print("BATTERY ENERGY STORAGE OPTIMIZATION EXAMPLES")
    print("="*70)

    # Run examples
    example_basic_optimization()
    example_different_price_patterns()
    example_custom_battery()
    example_no_end_constraint()

    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
