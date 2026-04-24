# ERCOT Wind Power Prediction System Implementation Plan

## Overview

Build a 0-12 hour wind power generation forecasting system for ERCOT/Texas based on HRRR weather forecast data.

**Core Objectives:**
- System-level wind power generation forecast (MW)
- Quantile prediction (p10/p50/p90) for uncertainty quantification
- Wind power ramp detection and early warning

> **Key Scenario: Wind Power Decline + No Solar Period**
>
> When a rapid wind power decline occurs after sunset / before sunrise (no solar),
> the system needs to rely on gas-fired generation,
> and prices tend to spike. This is the scenario most in need of advance warning.

---

## Project Structure

```
trueflux/wind-generation-forecast/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── hrrr_client.py        # HRRR data retrieval (Earth2Studio)
│   │   ├── ercot_wind_client.py  # ERCOT wind data retrieval
│   │   └── texas_regions.py      # Texas region definitions
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── wind_features.py      # Wind speed/power features
│   │   ├── ramp_features.py      # Ramp detection features
│   │   └── temporal_features.py  # Temporal features
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py               # Model base class interface
│   │   ├── gbm_model.py          # LightGBM baseline
│   │   ├── lstm_model.py         # LSTM sequence model
│   │   └── ensemble.py           # Ensemble model
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py            # MAE, RMSE, skill scores
│   │   └── ramp_metrics.py       # Ramp detection metrics (POD, FAR, CSI)
│   │
│   └── utils/
│       ├── __init__.py
│       └── config.py             # Configuration management
│
├── scripts/
│   ├── fetch_hrrr_data.py        # Download HRRR historical data
│   ├── build_features.py         # Build training features
│   └── train_models.py           # Train models
│
├── configs/
│   └── default.yaml              # Configuration file
│
├── notebooks/
│   └── exploration.ipynb         # Data exploration
│
└── requirements.txt
```

---

## Implementation Steps

### Phase 1: Data Layer (src/data/)

**1.1 hrrr_client.py - HRRR Data Retrieval**

```python
# Key functionality
class HRRRWindClient:
    """Retrieve HRRR forecast data via Earth2Studio"""

    WIND_VARIABLES = ['u10m', 'v10m', 'u80m', 'v80m', 't2m', 'sp']

    def fetch_forecast(
        self,
        init_time: datetime,
        lead_times: List[int],  # [0, 1, 2, ..., 12] hours
    ) -> xr.DataArray:
        """Retrieve HRRR forecast for the Texas region"""

    def subset_texas(self, data: xr.DataArray) -> xr.DataArray:
        """Extract Texas subregion"""
```

**1.2 ercot_wind_client.py - ERCOT Wind Data**

Reuse the `ercot-scraper` pattern to retrieve historical wind generation data for training.

**1.3 texas_regions.py - Region Definitions**

```python
ERCOT_WIND_REGIONS = {
    'PANHANDLE': {'lat': (34.0, 36.5), 'lon': (-103.0, -100.0)},
    'WEST': {'lat': (30.5, 34.0), 'lon': (-104.0, -100.5)},
    'COASTAL': {'lat': (26.5, 30.0), 'lon': (-97.5, -95.0)},
}
```

---

### Phase 2: Feature Engineering (src/features/)

**2.1 wind_features.py - Wind Power Features**

Reference the `spike-forecast/src/data/feature_engineering.py` pattern:

```python
class WindFeatureEngineer:
    """Wind power feature computation"""

    @staticmethod
    def compute_wind_speed(u, v) -> np.ndarray:
        """Compute wind speed from U/V components"""
        return np.sqrt(u**2 + v**2)

    @staticmethod
    def compute_power_curve(wind_speed, cut_in=3.0, rated=12.0, cut_out=25.0):
        """Apply turbine power curve"""

    @staticmethod
    def compute_wind_shear(ws_10m, ws_80m):
        """Compute wind shear exponent"""
```

**Output Features:**
- `ws_80m_mean`: 80m mean wind speed
- `ws_80m_std`: Wind speed spatial variability
- `power_density`: Wind power density
- `normalized_power`: Power curve normalized output

**2.2 ramp_features.py - Ramp Features**

> **Focus: Ramp-Down + No Solar Combined Risk**

```python
class RampFeatureEngineer:
    """Ramp detection features - focus on ramp-down"""

    def compute_wind_change_rate(self, wind_speed, time_hours):
        """Wind speed rate of change (m/s/h)"""

    def compute_power_sensitivity(self, wind_speed):
        """Power curve sensitive range (3-12 m/s is most sensitive)"""

    def compute_frontal_indicator(self, temp_change, wind_dir_change):
        """Weather front indicator"""

    def compute_ramp_down_risk(
        self,
        wind_change: float,      # Predicted wind power change (MW)
        current_hour: int,       # Current hour (0-23)
    ) -> float:
        """
        Compute wind power decline risk score (0-1)

        High-risk combinations:
        1. Forecast wind power decline > 2000 MW
        2. No solar period (18:00 - 07:00)
        3. Evening demand peak (17:00 - 21:00)
        """
        risk = 0.0
        is_no_solar = (current_hour >= 18) or (current_hour < 7)
        is_evening_peak = 17 <= current_hour <= 21

        if wind_change < -1000: risk += 0.2
        if wind_change < -2000: risk += 0.2
        if wind_change < -3000: risk += 0.2
        if is_no_solar: risk += 0.2
        if is_evening_peak and is_no_solar: risk += 0.2  # Most dangerous combination

        return min(risk, 1.0)
```

**Output Features:**
- `ws_change_1h`, `ws_change_3h`: Wind speed change
- `ramp_down_1h`, `ramp_down_3h`: Wind power decline (MW, negative values indicate decline)
- `power_sensitivity`: Power curve sensitivity
- `frontal_indicator`: Front probability
- **`is_no_solar_period`**: Whether it is a no-solar period (18:00-07:00)
- **`is_evening_peak`**: Whether it is the evening peak (17:00-21:00)
- **`ramp_down_no_solar_risk`**: Combined risk score (0-1)
- **`minutes_to_sunset`**: Minutes until sunset
- **`minutes_since_sunset`**: Minutes since sunset

**2.3 temporal_features.py - Temporal Features**

```python
class TemporalFeatureEngineer:
    """Temporal features"""

    @staticmethod
    def encode_cyclical(value, period):
        """Cyclical encoding (sin/cos)"""
        return np.sin(2*np.pi*value/period), np.cos(2*np.pi*value/period)
```

**Output Features:**
- `hour_sin`, `hour_cos`: Hour cycle
- `doy_sin`, `doy_cos`: Day-of-year cycle

---

### Phase 3: Model Layer (src/models/)

**3.1 base.py - Model Interface**

```python
class BaseWindForecastModel(ABC):
    """Wind power forecast model base class"""

    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model"""

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """Point prediction (p50)"""

    @abstractmethod
    def predict_quantiles(self, X, quantiles=[0.1, 0.5, 0.9]):
        """Quantile prediction"""

    def predict_ramp(self, X, current_gen, threshold=2000):
        """Ramp prediction"""
```

**3.2 gbm_model.py - LightGBM Baseline**

Reference `RTM_LMP_Price_Forecast/src/rtm_short_term_forecast.py`:

```python
class GBMWindModel(BaseWindForecastModel):
    """LightGBM quantile regression"""

    def __init__(self, quantiles=[0.1, 0.5, 0.9], use_gpu=True):
        self.models = {}  # One model per quantile

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        for q in self.quantiles:
            model = lgb.LGBMRegressor(
                objective='quantile',
                alpha=q,
                n_estimators=1000,
                learning_rate=0.02,
                max_depth=8,
            )
            model.fit(X_train, y_train)
            self.models[q] = model
```

**3.3 lstm_model.py - LSTM Model**

Reference `Load-_forecast/Load_Forecast/src/lstm_forecast.py`:

```python
class LSTMWindModel(nn.Module):
    """LSTM sequence model"""

    def __init__(self, n_features, hidden_dim=256, num_layers=3):
        self.lstm = nn.LSTM(n_features, hidden_dim, num_layers)
        self.quantile_heads = nn.ModuleList([...])  # Multi-head output for quantiles
```

**3.4 ensemble.py - Ensemble Model**

```python
class EnsembleWindModel(BaseWindForecastModel):
    """Model ensemble"""

    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights or [1/len(models)] * len(models)

    def predict_quantiles(self, X, quantiles=None):
        """Weighted average of quantiles from each model"""
```

---

### Phase 4: Evaluation (src/evaluation/)

**4.1 metrics.py - Standard Metrics**

```python
def mae(y_true, y_pred): ...
def rmse(y_true, y_pred): ...
def nmae(y_true, y_pred, capacity): ...  # Normalized MAE
def skill_score(y_true, y_pred, y_baseline): ...
```

**4.2 ramp_metrics.py - Ramp Metrics**

> **Focus on evaluating Ramp-Down detection capability**

```python
def detect_ramps(values, threshold, window, direction='both'):
    """
    Detect ramp events

    direction: 'both', 'up', 'down'
    """

def compute_ramp_metrics(y_true, y_pred, threshold=2000, window=12):
    """Compute ramp detection metrics"""
    # POD: Probability of Detection = hits / (hits + misses)
    # FAR: False Alarm Rate = false_alarms / (hits + false_alarms)
    # CSI: Critical Success Index = hits / (hits + misses + false_alarms)

def evaluate_ramp_down_in_no_solar(
    y_true, y_pred, timestamps, threshold=-2000
):
    """
    Specifically evaluate wind power decline detection during no-solar periods

    This is the most critical scenario:
    - Only evaluate the 18:00-07:00 period
    - Only evaluate decline events (change < threshold)
    - Compute advance warning lead time
    """
```

**Key Evaluation Metrics:**
- **Ramp-Down POD**: Wind power decline event hit rate (target > 0.8)
- **No-Solar POD**: Decline event hit rate during no-solar periods (most critical)
- **Lead Time**: Average advance warning time (target > 2 hours)
- **Miss Rate**: Missed detection rate (must be < 0.2)

---

## Dependencies (requirements.txt)

```
# Data
earth2studio>=0.12.0
xarray
pandas
numpy

# ML
lightgbm
torch
scikit-learn

# Utils
pyyaml
python-dateutil
loguru
```

---

## Validation Plan

1. **Data Layer Validation:**
   ```bash
   python scripts/fetch_hrrr_data.py --date 2025-01-20 --hours 12
   # Verify output xarray shape and variables
   ```

2. **Feature Validation:**
   ```bash
   python scripts/build_features.py --start 2024-01-01 --end 2024-12-31
   # Check feature DataFrame for NaN/Inf
   ```

3. **Model Training Validation:**
   ```bash
   python scripts/train_models.py
   # Output MAE, RMSE, CSI metrics
   ```

4. **Ramp Detection Validation:**
   - Test POD on historical large ramp events (>3000 MW/h)
   - Target: POD > 0.7, FAR < 0.4

---

## Implementation Priority

| Priority | Component | Description |
|----------|-----------|-------------|
| P0 | hrrr_client.py | Data retrieval foundation |
| P0 | wind_features.py | Core features |
| P0 | gbm_model.py | Baseline model |
| P1 | ramp_features.py | Ramp features |
| P1 | lstm_model.py | Sequence model |
| P1 | ramp_metrics.py | Ramp evaluation |
| P2 | ensemble.py | Model ensemble |
| P2 | Real-time inference | Production deployment |

---

## Key Reference Files

1. `/home/lanxin/projects/weather-forecast/earth2studio/earth2studio/data/hrrr.py` - HRRR data retrieval pattern
2. `/home/lanxin/projects/trueflux/spike-forecast/src/data/feature_engineering.py` - Feature engineering pattern
3. `/home/lanxin/projects/trueflux/Load-_forecast/Load_Forecast/src/lstm_forecast.py` - LSTM implementation
4. `/home/lanxin/projects/trueflux/RTM_LMP_Price_Forecast/src/rtm_short_term_forecast.py` - Multi-model training
