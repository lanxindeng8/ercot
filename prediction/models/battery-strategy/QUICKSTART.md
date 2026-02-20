# Quick Start Guide

## Installation

```bash
# 1. Clone or navigate to the project directory
cd battery-storage-strategy

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

## Running Examples

```bash
# Run all examples
python run_example.py

# Run tests
pytest tests/ -v
```

## Basic Usage Example

```python
from src.battery_config import BatteryConfig
from src.optimizer import BatteryOptimizer, interpolate_hourly_to_5min, create_sample_prices
from src.visualizer import BatteryVisualizer

# Configure battery
config = BatteryConfig(
    E_max=10.0,        # 10 MWh capacity
    P_ch_max=2.5,      # 2.5 MW charging
    P_dis_max=2.5,     # 2.5 MW discharging
    eta_ch=0.95,       # 95% efficiency
    eta_dis=0.95,
    SoC_min=0.1,       # 10% minimum SoC
    SoC_max=0.9,       # 90% maximum SoC
    SoC_0=0.5,         # Start at 50%
    SoC_T_target=0.5   # End at 50%
)

# Prepare prices (24 hourly values)
hourly_prices = create_sample_prices("peak")
prices_5min = interpolate_hourly_to_5min(hourly_prices)

# Optimize
optimizer = BatteryOptimizer(config)
result = optimizer.optimize(prices_5min)

# View results
visualizer = BatteryVisualizer(config)
visualizer.print_summary(result, prices_5min)
visualizer.plot_optimization_results(result, prices_5min)
```

## Understanding the Output

The optimization will show:

1. **Total Revenue**: Profit from arbitrage ($)
2. **Energy Charged/Discharged**: Total MWh moved
3. **Round-Trip Efficiency**: Actual efficiency achieved
4. **Average Prices**: Avg price when charging vs discharging
5. **Price Spread**: Difference captured by strategy
6. **SoC Statistics**: Battery state throughout the day

### Example Output

```
Total Revenue: $407.56
Energy Charged: 9.34 MWh
Energy Discharged: 8.43 MWh
Round-Trip Efficiency: 90.25%
Average Charging Price: $28.97/MWh
Average Discharging Price: $80.42/MWh
Price Spread Captured: $51.46/MWh
```

## Key Insights from Examples

### Example 1: Peak Pattern
- **Revenue**: $407.56
- **Strategy**: Charge during night (low prices), discharge during day (high prices)
- **Efficiency**: 90.25% round-trip

### Example 2: Pattern Comparison
- **Volatile prices**: Highest revenue ($440.49) - more arbitrage opportunities
- **Peak prices**: Good revenue ($407.56) - predictable pattern
- **Valley prices**: Moderate revenue ($293.39) - inverse pattern
- **Flat prices**: Zero revenue ($0.00) - no arbitrage opportunity

### Example 3: Custom Large Battery (50 MWh)
- **Revenue**: $1,415.22
- **Higher capacity** = more energy throughput = more revenue
- **Efficiency**: 84.64% (lower due to conservative SoC bounds)

### Example 4: No End Constraint
- **Revenue**: $546.24 (higher than constrained case)
- **Freedom to end at any SoC** allows more aggressive strategy
- **Final SoC**: 10% (started at 50%)
- **Effective efficiency**: 164% (used initial stored energy)

## Tips

1. **Price Spread Matters**: The system only profits when there's sufficient price difference to overcome round-trip losses
2. **Efficiency Impact**: At 95% round-trip efficiency (90.25% effective), you need >11% price spread to break even
3. **Constraints**: Terminal SoC constraints reduce revenue but ensure battery is ready for next day
4. **Volatile Markets**: More volatile prices create more arbitrage opportunities

## Next Steps

- Load your own price data (see README for CSV import)
- Experiment with different battery configurations
- Analyze different market conditions
- Add degradation costs (future enhancement)
