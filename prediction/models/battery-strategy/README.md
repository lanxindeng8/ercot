# Battery Storage Strategy

A Python-based optimization system for battery energy storage charging/discharging strategy based on electricity price arbitrage.

## Overview

This project implements a **Linear Programming (LP)** based optimization algorithm for maximizing revenue from battery energy storage systems (BESS) participating in electricity markets. The system optimizes charging and discharging schedules based on Day-Ahead Market (DAM) prices, accounting for:

- Battery capacity and power constraints
- Charging/discharging efficiency
- State of Charge (SoC) bounds
- Terminal SoC constraints

### Key Features

- **Linear Programming Optimization**: Uses PuLP for efficient LP solving
- **5-minute Resolution**: Operates at 5-minute intervals (288 periods per day)
- **Price Interpolation**: Automatically interpolates hourly DAM prices to 5-minute RTM resolution
- **Comprehensive Visualization**: Generates detailed plots and analysis
- **Flexible Configuration**: Easily customize battery parameters

## Mathematical Formulation

The optimization maximizes revenue from price arbitrage:

```
maximize: Σ [p(t) · P_dis(t) - p(t) · P_ch(t)] · Δt
```

Subject to:
- **SoC dynamics**: `SoC(t) = SoC(t-1) + (η_ch · P_ch(t) · Δt)/E_max - (P_dis(t) · Δt)/(η_dis · E_max)`
- **SoC bounds**: `SoC_min ≤ SoC(t) ≤ SoC_max`
- **Power constraints**: `0 ≤ P_ch(t) ≤ P_ch_max`, `0 ≤ P_dis(t) ≤ P_dis_max`
- **Terminal constraint**: `SoC(T) = SoC_target` (optional)

## Project Structure

```
battery-storage-strategy/
├── src/
│   ├── __init__.py           # Package initialization
│   ├── battery_config.py     # Battery configuration and parameters
│   ├── optimizer.py          # LP optimization solver
│   ├── visualizer.py         # Result visualization
│   ├── price_utils.py        # Price data processing utilities
│   └── example.py            # Example usage scripts
├── tests/
│   ├── __init__.py
│   ├── test_battery_config.py
│   └── test_optimizer.py
├── data/                     # Data files (price data)
├── docs/                     # Generated documentation and figures
└── requirements.txt          # Python dependencies
```

## Setup

### 1. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `numpy` - Numerical computing
- `pandas` - Data processing
- `matplotlib` - Visualization
- `pulp` - Linear programming solver
- `pytest` - Unit testing

## Usage

### Quick Start

Run the example script to see the optimization in action:

```bash
python run_example.py
```

This will run multiple examples demonstrating different price patterns and battery configurations.

### Basic Usage

```python
from src.battery_config import BatteryConfig
from src.optimizer import BatteryOptimizer, interpolate_hourly_to_5min, create_sample_prices
from src.visualizer import BatteryVisualizer

# 1. Create battery configuration
config = BatteryConfig(
    E_max=10.0,           # 10 MWh capacity
    P_ch_max=2.5,         # 2.5 MW charging power
    P_dis_max=2.5,        # 2.5 MW discharging power
    eta_ch=0.95,          # 95% charging efficiency
    eta_dis=0.95,         # 95% discharging efficiency
    SoC_min=0.1,          # 10% minimum SoC
    SoC_max=0.9,          # 90% maximum SoC
    SoC_0=0.5,            # Start at 50% SoC
    SoC_T_target=0.5      # End at 50% SoC
)

# 2. Prepare price data
hourly_prices = create_sample_prices(pattern="peak")  # 24 hourly prices
prices_5min = interpolate_hourly_to_5min(hourly_prices)  # Interpolate to 5-min

# 3. Run optimization
optimizer = BatteryOptimizer(config)
result = optimizer.optimize(prices_5min)

# 4. Visualize results
visualizer = BatteryVisualizer(config)
visualizer.print_summary(result, prices_5min)
visualizer.plot_optimization_results(result, prices_5min)
```

### Using Custom Price Data

```python
import numpy as np
from src.price_utils import load_dam_prices_from_csv, get_next_day_prices

# Load from CSV
df = load_dam_prices_from_csv("data/prices.csv", date_column="timestamp", price_column="lmp")

# Get prices for specific date
target_date = "2025-01-15"
hourly_prices = get_next_day_prices(df, target_date)

# Convert to 5-minute resolution and optimize
prices_5min = interpolate_hourly_to_5min(hourly_prices)
result = optimizer.optimize(prices_5min)
```

### Customizing Battery Parameters

```python
# Large utility-scale battery
large_battery = BatteryConfig(
    E_max=100.0,          # 100 MWh
    P_ch_max=25.0,        # 25 MW
    P_dis_max=25.0,       # 25 MW
    eta_ch=0.92,
    eta_dis=0.92,
    SoC_min=0.2,          # Conservative bounds for longevity
    SoC_max=0.8,
    SoC_0=0.5,
    SoC_T_target=None     # No end constraint
)
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_optimizer.py
```

## Output and Visualization

The system generates:

1. **Comprehensive plots** showing:
   - Electricity prices over 24 hours
   - Charging/discharging power schedule
   - State of Charge evolution
   - Cumulative revenue

2. **Text summary** including:
   - Total revenue
   - Energy charged/discharged
   - Round-trip efficiency
   - Average charging/discharging prices
   - Number of cycles

3. **Power vs. Price scatter plots** showing optimization behavior

## Performance

- **Optimization time**: Typically < 1 second for 288 time periods
- **Solver**: CBC (open-source) or GUROBI/CPLEX (commercial, faster)
- **Scalability**: Can handle multiple days with minor modifications

## Theory and Algorithm

The system uses Linear Programming to find the optimal charging/discharging schedule. Key advantages:

- **Global optimum**: LP guarantees finding the best solution
- **Fast solving**: Modern LP solvers are highly efficient
- **No simultaneous charge/discharge**: Natural outcome due to efficiency losses
- **Price-responsive**: Automatically charges at low prices, discharges at high prices

The interpolation from hourly DAM prices to 5-minute RTM prices uses linear interpolation, providing smooth price transitions.

## Limitations and Future Work

Current limitations:
- Perfect price foresight (day-ahead prices assumed known)
- No battery degradation modeling
- No minimum up/down time constraints
- No ramp rate constraints

Potential extensions:
- Add battery degradation costs
- Implement stochastic optimization for price uncertainty
- Include ancillary service revenue streams
- Add mixed-integer constraints for minimum run times

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Contact

For questions or collaboration, please open an issue on GitHub.
