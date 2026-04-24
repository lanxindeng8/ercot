# ERCOT RTM LMP Spike Prediction Algorithm Design Document

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Data Architecture](#data-architecture)
3. [Feature Engineering](#feature-engineering)
4. [Prediction Model](#prediction-model)
5. [Strategy Optimization](#strategy-optimization)
6. [Implementation Roadmap](#implementation-roadmap)

---

## 1. System Architecture

### 1.1 Overall Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                     Data Ingestion Layer (Data Layer)        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │ Market   │ │ System   │ │ Weather  │ │Constraint│       │
│  │ Data     │ │ Data     │ │ Data     │ │ Data     │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   Feature Engineering Layer (Feature Layer)  │
│  ┌────────────────────────────────────────────────────┐     │
│  │ Zone-level Features (CPS, West, Houston, Hub)      │     │
│  │ • Price Structure  • Supply-Demand  • Weather-Driven│    │
│  └────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   Prediction Model Layer (Model Layer)       │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │ Spike Event  │ │ Regime State │ │ Price Quantile│       │
│  │ Prediction   │ │ Recognition  │ │ Forecast      │       │
│  └──────────────┘ └──────────────┘ └──────────────┘        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   Strategy Optimization Layer (Strategy Layer)│
│  ┌────────────────────────────────────────────────────┐     │
│  │ BESS Dispatch Optimizer                            │     │
│  │ • Hold-SOC Rules • MPC Optimization • Risk Control │     │
│  └────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Core Modules
- **Data Ingestion Module**: Real-time data acquisition, preprocessing, quality control
- **Feature Computation Module**: Real-time feature computation, rolling window statistics
- **Prediction Engine**: Zone-level spike alerting, state recognition
- **Strategy Engine**: SOC management, power dispatch, risk control

---

## 2. Data Architecture

### 2.1 Data Source Inventory

#### Market Data (5-min/15-min)
| Data Item | Field Name | Unit | Update Frequency | Data Source |
|-----------|------------|------|------------------|-------------|
| RT LMP - LZ CPS | `P_CPS` | $/MWh | 5-min | ERCOT RTM |
| RT LMP - LZ West | `P_West` | $/MWh | 5-min | ERCOT RTM |
| RT LMP - LZ Houston | `P_Houston` | $/MWh | 5-min | ERCOT RTM |
| RT Hub Average | `P_Hub` | $/MWh | 5-min | ERCOT RTM |
| DA LMP (corresponding interval) | `P_DA_*` | $/MWh | hourly | ERCOT DAM |

#### System Data
| Data Item | Field Name | Unit | Update Frequency | Data Source |
|-----------|------------|------|------------------|-------------|
| System Load | `Load` | MW | 5-min | ERCOT |
| Wind Generation | `Wind` | MW | 5-min | ERCOT Fuel Mix |
| Solar Generation | `Solar` | MW | 5-min | ERCOT Fuel Mix |
| Natural Gas Generation | `Gas` | MW | 5-min | ERCOT Fuel Mix |
| Coal Generation | `Coal` | MW | 5-min | ERCOT Fuel Mix |
| ESR Net Output | `ESR` | MW | 5-min | ERCOT Fuel Mix |
| Net Load | `NetLoad` | MW | 5-min | Calculated |

#### Weather Data (Zone-level)
| Data Item | Field Name | Unit | Update Frequency | Data Source |
|-----------|------------|------|------------------|-------------|
| Temperature | `T_{zone}` | °F | 15-min | NOAA/Weather API |
| Wind Speed | `WindSpeed_{zone}` | mph | 15-min | NOAA/Weather API |
| Wind Direction | `WindDir_{zone}` | degree | 15-min | NOAA/Weather API |
| Wind Chill | `WindChill_{zone}` | °F | 15-min | Calculated |

#### Constraint Data (Optional, Enhanced Version)
| Data Item | Field Name | Unit | Update Frequency | Data Source |
|-----------|------------|------|------------------|-------------|
| Binding Constraint List | `BindingConstraints` | - | 5-min | ERCOT |
| Interface Shadow Price | `ShadowPrice_*` | $/MW | 5-min | ERCOT |

### 2.2 Data Storage Plan

#### Time Series Database (Recommended: InfluxDB/TimescaleDB)
```
measurement: ercot_rtm
tags:
  - zone (CPS, West, Houston, Hub)
  - data_type (price, fuel_mix, weather)
fields:
  - price (float)
  - load (float)
  - wind (float)
  - solar (float)
  - gas (float)
  - coal (float)
  - esr (float)
  - temperature (float)
  - wind_speed (float)
  - ...
timestamp: UTC with 5-min resolution
```

#### Feature Store
```
measurement: spike_features
tags:
  - zone
  - feature_group (price_structure, supply_demand, weather)
fields:
  - spread_zone_hub (float)
  - spread_rt_da (float)
  - price_ramp (float)
  - net_load_ramp (float)
  - wind_anomaly (float)
  - gas_saturation (float)
  - temperature_anomaly (float)
  - ...
timestamp: UTC with 5-min resolution
```

---

## 3. Feature Engineering

### 3.1 Feature Groups

#### Group 1: Price Structure Features
**Objective**: Capture regional scarcity and congestion signals

| Feature Name | Formula | Description | Threshold Example |
|-------------|---------|-------------|-------------------|
| `Spread_zone_hub` | `P_zone(t) - P_Hub(t)` | Zone-system spread | >50 $/MWh |
| `Spread_CPS_Houston` | `P_CPS(t) - P_Houston(t)` | CPS-Houston spread | >80 $/MWh |
| `Spread_RT_DA` | `P_zone^RT(t) - P_zone^DA(t)` | Real-time to day-ahead premium | >100 $/MWh |
| `PriceRamp_5m` | `(P(t) - P(t-5)) / 5` | 5-minute price slope | >10 $/MWh/min |
| `PriceRamp_15m` | `(P(t) - P(t-15)) / 15` | 15-minute price slope | >5 $/MWh/min |
| `PriceAccel` | `PriceRamp_5m(t) - PriceRamp_5m(t-5)` | Price acceleration | >2 $/MWh/min² |

**Implementation Example**:
```python
def calc_price_features(df, zone='CPS'):
    features = {}

    # Spread features
    features[f'spread_{zone}_hub'] = df[f'P_{zone}'] - df['P_Hub']
    features[f'spread_rt_da'] = df[f'P_{zone}'] - df[f'P_{zone}_DA']

    # Slope features
    features[f'price_ramp_5m'] = df[f'P_{zone}'].diff(1) / 5  # 5-min data
    features[f'price_ramp_15m'] = df[f'P_{zone}'].diff(3) / 15

    # Acceleration
    features[f'price_accel'] = features[f'price_ramp_5m'].diff(1)

    return pd.DataFrame(features)
```

#### Group 2: Supply-Demand Balance Features
**Objective**: Capture system/regional stress conditions

| Feature Name | Formula | Description | Threshold Example |
|-------------|---------|-------------|-------------------|
| `NetLoad` | `Load(t) - Wind(t) - Solar(t)` | Net load | - |
| `NetLoadRamp_5m` | `dNetLoad/dt` | Net load ramp rate | >200 MW/5min |
| `NetLoadAccel` | `d²NetLoad/dt²` | Net load acceleration | >50 MW/5min² |
| `WindAnomaly` | `(Wind(t) - Wind_MA30d(t)) / Wind_Std30d(t)` | Wind anomaly | <-1.5 σ |
| `WindRamp` | `dWind/dt` | Wind change rate | <-100 MW/5min |
| `GasSaturation` | `Gas(t) / Gas_P95_7d` | Gas generation saturation | >0.95 |
| `CoalStress` | `I(Coal(t) > Coal(t-1)) & I(hour ∈ [0,5])` | Nighttime coal ramp-up | 1 (True) |
| `ESRNetOutput` | `ESR(t)` | ESR net output | <-1000 MW (charging) |

**Implementation Example**:
```python
def calc_supply_demand_features(df):
    features = {}

    # Net load
    features['net_load'] = df['Load'] - df['Wind'] - df['Solar']
    features['net_load_ramp_5m'] = features['net_load'].diff(1)
    features['net_load_accel'] = features['net_load_ramp_5m'].diff(1)

    # Wind anomaly
    wind_ma = df['Wind'].rolling(window=30*24*12, min_periods=1).mean()  # 30-day rolling mean
    wind_std = df['Wind'].rolling(window=30*24*12, min_periods=1).std()
    features['wind_anomaly'] = (df['Wind'] - wind_ma) / wind_std
    features['wind_ramp'] = df['Wind'].diff(1)

    # Gas generation saturation
    gas_p95 = df['Gas'].rolling(window=7*24*12, min_periods=1).quantile(0.95)
    features['gas_saturation'] = df['Gas'] / gas_p95

    # Coal stress
    features['coal_stress'] = (
        (df['Coal'].diff(1) > 0) &
        (df.index.hour.isin(range(0, 6)))
    ).astype(int)

    # Energy storage
    features['esr_net_output'] = df['ESR']

    return pd.DataFrame(features)
```

#### Group 3: Weather-Driven Features (Zone-level)
**Objective**: Capture demand-side shock signals

| Feature Name | Formula | Description | Threshold Example |
|-------------|---------|-------------|-------------------|
| `T_anomaly_zone` | `T(t) - T_MA30d(hour)` | Temperature anomaly | <-10 °F |
| `T_ramp_zone` | `dT/dt` | Cooling rate | <-5 °F/hr |
| `WindChill_zone` | `35.74 + 0.6215T - 35.75v^0.16 + 0.4275Tv^0.16` | Wind chill index | <20 °F |
| `ColdFront_zone` | `I(ΔT < -5 & WindShift to N)` | Cold front flag | 1 (True) |

**Implementation Example**:
```python
def calc_weather_features(df, zone='CPS'):
    features = {}

    # Temperature anomaly (relative to historical same-hour mean)
    t_hourly_mean = df.groupby(df.index.hour)[f'T_{zone}'].transform(
        lambda x: x.rolling(window=30*24, min_periods=1).mean()
    )
    features[f'T_anomaly_{zone}'] = df[f'T_{zone}'] - t_hourly_mean

    # Cooling rate
    features[f'T_ramp_{zone}'] = df[f'T_{zone}'].diff(12) / 1  # 12 five-min intervals = 1 hour

    # Wind chill index
    T = df[f'T_{zone}']
    v = df[f'WindSpeed_{zone}']
    features[f'WindChill_{zone}'] = (
        35.74 + 0.6215*T - 35.75*(v**0.16) + 0.4275*T*(v**0.16)
    )

    # Cold front flag
    wind_to_north = (df[f'WindDir_{zone}'] > 315) | (df[f'WindDir_{zone}'] < 45)
    features[f'ColdFront_{zone}'] = (
        (features[f'T_ramp_{zone}'] < -5) & wind_to_north
    ).astype(int)

    return pd.DataFrame(features)
```

#### Group 4: Temporal Features
**Objective**: Capture intraday patterns and solar recovery windows

| Feature Name | Formula | Description |
|-------------|---------|-------------|
| `hour` | `hour of day` | Hour (0-23) |
| `is_evening_peak` | `I(hour ∈ [17, 22])` | Evening peak flag |
| `minutes_to_sunrise` | `sunrise_time - current_time` | Minutes to sunrise |
| `solar_ramp_expected` | `dSolar_forecast/dt` | Expected solar ramp |

### 3.2 Feature Importance (Based on Document Analysis)

**Tier 1 (Strong Causal/Trigger Signals)**:
1. `Spread_zone_hub` - Regional spread expansion
2. `NetLoadAccel` - Net load acceleration
3. `WindAnomaly` - Wind buffer collapse
4. `T_anomaly_zone` - Temperature anomaly
5. `PriceAccel` - Price acceleration

**Tier 2 (Confirmation/Amplification Signals)**:
6. `GasSaturation` - Gas generation saturation
7. `CoalStress` - Nighttime coal ramp-up
8. `ESRNetOutput` - ESR charging amplification
9. `ColdFront_zone` - Cold front event
10. `T_ramp_zone` - Rapid cooling

**Tier 3 (Recovery/Termination Signals)**:
11. `minutes_to_sunrise` - Solar recovery countdown
12. `solar_ramp_expected` - Expected solar ramp

---

## 4. Prediction Model

### 4.1 Label Generation

#### 4.1.1 SpikeEvent Label
**Definition**: Whether zone z is in a spike state at time t

**Rules**:
```python
def generate_spike_label(df, zone='CPS', P_hi=400, S_hi=50, S_cross_hi=80, m=3):
    """
    Generate SpikeEvent labels

    Args:
        df: DataFrame containing price and spread fields
        zone: Zone name
        P_hi: Price threshold ($/MWh)
        S_hi: Zone-hub spread threshold ($/MWh)
        S_cross_hi: Zone-Houston spread threshold ($/MWh)
        m: Duration threshold (number of time steps)

    Returns:
        SpikeEvent_{zone}: 0/1 label
    """
    # Condition A: High price
    cond_price = df[f'P_{zone}'] >= P_hi

    # Condition B: Large spread (constraint-driven)
    spread_zh = df[f'P_{zone}'] - df['P_Hub']
    spread_cross = df[f'P_{zone}'] - df['P_Houston']
    cond_spread = (spread_zh >= S_hi) | (spread_cross >= S_cross_hi)

    # Condition C: Duration
    raw_spike = cond_price & cond_spread
    sustained_spike = raw_spike.rolling(window=m).sum() >= m

    return sustained_spike.fillna(False).astype(int)
```

#### 4.1.2 LeadSpike Label
**Definition**: Whether a spike will occur within the next H minutes

```python
def generate_lead_spike_label(df, zone='CPS', H=60, dt=5):
    """
    Generate LeadSpike label (early warning)

    Args:
        H: Warning time window (minutes)
        dt: Data time resolution (minutes)

    Returns:
        LeadSpike_{zone}_{H}m: 0/1 label
    """
    spike_event = df[f'SpikeEvent_{zone}']
    k = int(H / dt)  # Window size

    # Reverse -> rolling max -> reverse back
    lead_spike = spike_event[::-1].rolling(window=k).max()[::-1]

    return lead_spike.fillna(0).astype(int)
```

#### 4.1.3 Regime Label
**Definition**: System state (Normal / Tight / Scarcity)

```python
def generate_regime_label(df, zone='CPS', P_mid=150, S_mid=20):
    """
    Generate Regime state label

    Returns:
        Regime_{zone}: 'Normal' / 'Tight' / 'Scarcity'
    """
    spread_zh = df[f'P_{zone}'] - df['P_Hub']
    spike_event = df[f'SpikeEvent_{zone}']

    regime = pd.Series('Normal', index=df.index)

    # Tight state
    tight_cond = (df[f'P_{zone}'] >= P_mid) | (spread_zh >= S_mid)
    regime[tight_cond] = 'Tight'

    # Scarcity state (highest priority)
    regime[spike_event == 1] = 'Scarcity'

    return regime
```

### 4.2 Model Architecture

#### 4.2.1 Model Selection
**Recommended Approach**: Hierarchical modeling + ensemble

```
┌─────────────────────────────────────────────────────┐
│ Layer 1: Baseline (XGBoost/LightGBM)               │
│ - Rapidly validate feature effectiveness            │
│ - Provide interpretable feature importance          │
│ - Serve as part of the ensemble model               │
└─────────────────────────────────────────────────────┘
                        +
┌─────────────────────────────────────────────────────┐
│ Layer 2: Continuous-Time Model (CfC/LTC)           │
│ - Handle irregular time intervals                   │
│ - Capture state transition dynamics                 │
│ - Adapt to event-driven characteristics             │
└─────────────────────────────────────────────────────┘
                        =
┌─────────────────────────────────────────────────────┐
│ Ensemble Prediction                                 │
│ p_hat(t) = α·p_baseline(t) + (1-α)·p_cfc(t)        │
└─────────────────────────────────────────────────────┘
```

#### 4.2.2 Input/Output Design

**Input** (at each time step t):
```python
X(t) = {
    # Price structure (6 dims × 3 zones = 18 dims)
    'spread_zone_hub', 'spread_rt_da', 'price_ramp_5m',
    'price_ramp_15m', 'price_accel', ...

    # Supply-demand balance (8 dims)
    'net_load', 'net_load_ramp', 'net_load_accel',
    'wind_anomaly', 'wind_ramp', 'gas_saturation',
    'coal_stress', 'esr_net_output',

    # Weather-driven (4 dims × 3 zones = 12 dims)
    'T_anomaly_CPS', 'T_ramp_CPS', 'WindChill_CPS', 'ColdFront_CPS',
    'T_anomaly_West', ...

    # Temporal features (4 dims)
    'hour', 'is_evening_peak', 'minutes_to_sunrise', 'solar_ramp_expected',

    # Historical prices (sliding window)
    'P_CPS_lag_1', 'P_CPS_lag_3', 'P_CPS_lag_12',  # 5/15/60 minutes ago
}
Total dimensions: ~40-50
```

**Output** (multi-task):
```python
Y(t) = {
    # Primary task: Spike early warning
    'LeadSpike_CPS_60m': P(SpikeEvent_CPS(t+60) = 1),  # [0, 1]
    'LeadSpike_West_60m': P(SpikeEvent_West(t+60) = 1),
    'LeadSpike_Houston_60m': P(SpikeEvent_Houston(t+60) = 1),

    # Auxiliary task: Regime recognition
    'Regime_prob': {
        'Normal': [0, 1],
        'Tight': [0, 1],
        'Scarcity': [0, 1]
    },  # 3-class probability

    # Optional task: Price quantile
    'P_CPS_P90_60m': 90th percentile price forecast,  # For risk control
}
```

#### 4.2.3 Model Implementation Framework

**XGBoost Baseline**:
```python
import xgboost as xgb

# Multi-task training
def train_baseline_model(X_train, y_train):
    models = {}

    for zone in ['CPS', 'West', 'Houston']:
        # Spike warning model
        model_spike = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            max_depth=6,
            learning_rate=0.05,
            n_estimators=500,
            scale_pos_weight=10,  # Handle class imbalance
        )
        model_spike.fit(
            X_train,
            y_train[f'LeadSpike_{zone}_60m'],
            eval_set=[(X_val, y_val[f'LeadSpike_{zone}_60m'])],
            early_stopping_rounds=50,
        )
        models[f'spike_{zone}'] = model_spike

        # Regime recognition model
        model_regime = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            max_depth=5,
            learning_rate=0.05,
            n_estimators=300,
        )
        model_regime.fit(
            X_train,
            y_train[f'Regime_{zone}'].map({'Normal': 0, 'Tight': 1, 'Scarcity': 2}),
        )
        models[f'regime_{zone}'] = model_regime

    return models
```

**CfC/LTC (Continuous-Time) Model**:
```python
# Using the ncps library (Neural Circuit Policies)
from ncps.torch import CfC
import torch
import torch.nn as nn

class SpikeForecastCfC(nn.Module):
    def __init__(self, input_size, hidden_size, num_zones=3):
        super().__init__()

        # CfC core
        self.cfc = CfC(
            input_size=input_size,
            units=hidden_size,
            mode='default',
        )

        # Multi-task output heads
        self.spike_heads = nn.ModuleDict({
            zone: nn.Linear(hidden_size, 1)
            for zone in ['CPS', 'West', 'Houston']
        })

        self.regime_head = nn.Linear(hidden_size, 3)  # 3-class

    def forward(self, x, time_deltas):
        """
        x: [batch, seq_len, features]
        time_deltas: [batch, seq_len] - time intervals (seconds)
        """
        # CfC handles irregular time series
        h, _ = self.cfc(x, time_deltas)

        # Multi-task output
        outputs = {}
        for zone in ['CPS', 'West', 'Houston']:
            outputs[f'spike_{zone}'] = torch.sigmoid(
                self.spike_heads[zone](h[:, -1, :])
            )

        outputs['regime'] = torch.softmax(
            self.regime_head(h[:, -1, :]),
            dim=-1
        )

        return outputs
```

### 4.3 Training Strategy

#### 4.3.1 Data Split
```python
# Time-based split (avoid data leakage)
train_end = '2025-11-30'
val_start = '2025-12-01'
val_end = '2025-12-10'
test_start = '2025-12-11'  # Includes 12-14/12-15 events

train_data = df[df.index < train_end]
val_data = df[(df.index >= val_start) & (df.index < val_end)]
test_data = df[df.index >= test_start]
```

#### 4.3.2 Loss Function
```python
# Multi-task loss
def multi_task_loss(pred, target, alpha=0.7, beta=0.3):
    """
    alpha: Spike warning task weight
    beta: Regime recognition task weight
    """
    # Spike warning: Focal Loss (handle class imbalance)
    loss_spike = focal_loss(
        pred['spike_CPS'],
        target['LeadSpike_CPS_60m'],
        gamma=2.0,  # Focus on hard examples
    )

    # Regime recognition: Cross-Entropy
    loss_regime = nn.CrossEntropyLoss()(
        pred['regime'],
        target['Regime'],
    )

    return alpha * loss_spike + beta * loss_regime
```

#### 4.3.3 Evaluation Metrics
```python
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

def evaluate_model(y_true, y_pred):
    metrics = {}

    # 1. Spike warning performance
    metrics['AUC'] = roc_auc_score(y_true['LeadSpike'], y_pred['spike_prob'])

    precision, recall, _ = precision_recall_curve(
        y_true['LeadSpike'],
        y_pred['spike_prob']
    )
    metrics['PR-AUC'] = auc(recall, precision)

    # 2. Event-level Recall (critical!)
    # Whether at least one warning was issued within 60 minutes before a real spike event
    events = identify_spike_events(y_true['SpikeEvent'])
    metrics['Event_Recall'] = compute_event_recall(events, y_pred, lead_time=60)

    # 3. Regime recognition accuracy
    metrics['Regime_Accuracy'] = accuracy_score(
        y_true['Regime'],
        y_pred['regime_pred']
    )

    # 4. False positive rate (costly metric)
    metrics['False_Positive_Rate'] = compute_fpr(y_true, y_pred, threshold=0.5)

    return metrics
```

---

## 5. Strategy Optimization

### 5.1 Strategy Framework

```
Input:
  - p_hat(t): Spike warning probability
  - regime(t): System state
  - SOC(t): Current state of charge
  - P_max: Maximum power
  - market_data(t): Real-time market information

Output:
  - P_dispatch(t): Dispatch power (positive=discharge, negative=charge)
  - action: 'charge' / 'discharge' / 'hold'
```

### 5.2 Hard Rule Layer (Safety Layer)

#### Rule A: Locational Charge Prohibition Rule
```python
def rule_anti_local_charging(market_data, zone='CPS', threshold_spread=50):
    """
    Prevent charging during regional high prices
    """
    P_zone = market_data[f'P_{zone}']
    P_hub = market_data['P_Hub']
    spread = P_zone - P_hub

    # Condition 1: High regional price
    cond1 = P_zone > 200

    # Condition 2: Rapidly expanding spread
    spread_ramp = spread - spread_prev
    cond2 = (spread > threshold_spread) or (spread_ramp > 20)

    if cond1 and cond2:
        return 'FORBID_CHARGE'
    else:
        return 'ALLOW'
```

#### Rule B: SOC Reservation Rule
```python
def rule_soc_reservation(p_hat, regime, SOC, SOC_min=0.2, SOC_reserve=0.6):
    """
    Reserve SOC during the warning phase
    """
    # Entering Tight state or high warning probability
    if regime == 'Tight' or p_hat > 0.4:
        if SOC < SOC_reserve:
            return 'FORBID_DISCHARGE'
        else:
            return 'LIMIT_DISCHARGE', 0.3  # Limit to 30% power

    # Entering Scarcity state -> allow full discharge
    if regime == 'Scarcity' or p_hat > 0.8:
        return 'ALLOW_FULL_DISCHARGE'

    return 'NORMAL'
```

#### Rule C: Solar Recovery Countdown
```python
def rule_solar_repair_countdown(market_data, minutes_to_sunrise, solar_ramp_forecast):
    """
    Take profit near sunrise
    """
    # Less than 30 minutes to sunrise and solar starts ramping
    if minutes_to_sunrise < 30 and solar_ramp_forecast > 100:
        return 'FAST_PROFIT_TAKING'

    # Less than 10 minutes to sunrise -> force profit-taking/recharging
    if minutes_to_sunrise < 10:
        return 'FORCE_PROFIT_TAKING'

    return 'NORMAL'
```

### 5.3 MPC Optimization Layer

#### 5.3.1 Objective Function
```python
"""
Optimization Objective:
  Maximize: Revenue - Risk_Penalty

  Revenue = Σ [P_zone(t) · P_discharge(t) - P_zone(t) · P_charge(t)] · Δt

  Risk_Penalty = λ₁·R_spread(t) + λ₂·R_time_decay(t) + λ₃·R_congestion(t)

Constraints:
  1. SOC dynamics: SOC(t+1) = SOC(t) - P_discharge(t)·Δt/E_cap + P_charge(t)·Δt·η/E_cap
  2. SOC bounds: SOC_min ≤ SOC(t) ≤ SOC_max
  3. Power bounds: -P_max ≤ P(t) ≤ P_max
  4. Hard rule constraints: Logic constraints from Rule A/B/C
"""

from scipy.optimize import minimize

def mpc_optimization(
    state,
    forecast,
    horizon=12,  # 60 minutes (12 × 5min)
    SOC_current=0.7,
    E_cap=100,  # MWh
    P_max=25,   # MW
):
    """
    MPC rolling optimization
    """
    # Decision variables: P(t), t=0,1,...,horizon-1
    def objective(P):
        revenue = 0
        risk = 0
        SOC = SOC_current

        for t in range(horizon):
            # Revenue
            if P[t] > 0:  # Discharge
                revenue += forecast['P_zone'][t] * P[t] * (5/60)
            else:  # Charge
                revenue += forecast['P_zone'][t] * P[t] * (5/60)  # P<0, so negative contribution

            # Risk terms
            # R1: Spread volatility risk
            spread_vol = np.std(forecast['spread_zone_hub'][t:t+3])
            risk += 0.1 * spread_vol * abs(P[t])

            # R2: SOC time-value decay
            time_to_sunrise = forecast['minutes_to_sunrise'][t]
            if time_to_sunrise < 60:
                decay_penalty = (60 - time_to_sunrise) / 60
                risk += 0.5 * decay_penalty * SOC

            # R3: Congestion uncertainty
            if forecast['regime'][t] == 'Tight':
                risk += 0.2 * abs(P[t])

            # Update SOC
            SOC -= P[t] * (5/60) / E_cap  # Simplified, efficiency not considered

        return -(revenue - risk)  # Minimize negative revenue

    # Constraints
    constraints = []

    # SOC constraints
    def soc_constraint_min(P):
        SOC = SOC_current
        min_soc = SOC_current
        for t in range(horizon):
            SOC -= P[t] * (5/60) / E_cap
            min_soc = min(min_soc, SOC)
        return min_soc - 0.1  # SOC >= 10%

    def soc_constraint_max(P):
        SOC = SOC_current
        max_soc = SOC_current
        for t in range(horizon):
            SOC -= P[t] * (5/60) / E_cap
            max_soc = max(max_soc, SOC)
        return 0.95 - max_soc  # SOC <= 95%

    constraints.append({'type': 'ineq', 'fun': soc_constraint_min})
    constraints.append({'type': 'ineq', 'fun': soc_constraint_max})

    # Hard rule constraints (example: charge prohibition)
    for t in range(horizon):
        if forecast['forbid_charge'][t]:
            # P(t) >= 0 (charging not allowed)
            constraints.append({
                'type': 'ineq',
                'fun': lambda P, t=t: P[t]
            })

    # Bounds
    bounds = [(-P_max, P_max) for _ in range(horizon)]

    # Solve
    result = minimize(
        objective,
        x0=np.zeros(horizon),  # Initial guess
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
    )

    # Return first-step decision
    return result.x[0]
```

### 5.4 Strategy Integration

```python
class BESSDispatchStrategy:
    def __init__(self, model, mpc_config):
        self.model = model
        self.mpc_config = mpc_config

    def dispatch(self, t, state, market_data):
        """
        Real-time dispatch decision
        """
        # 1. Model prediction
        X_t = self.extract_features(market_data, t)
        prediction = self.model.predict(X_t)

        p_hat = prediction['spike_prob_CPS']
        regime = prediction['regime']

        # 2. Hard rule checks
        rule_a = rule_anti_local_charging(market_data, zone='CPS')
        rule_b = rule_soc_reservation(p_hat, regime, state['SOC'])
        rule_c = rule_solar_repair_countdown(
            market_data,
            state['minutes_to_sunrise'],
            prediction['solar_ramp_forecast']
        )

        # 3. Decision logic
        if rule_a == 'FORBID_CHARGE':
            P_dispatch = max(0, self.mpc_optimization(...))  # Only allow discharge or hold

        elif rule_b == 'FORBID_DISCHARGE':
            P_dispatch = min(0, self.mpc_optimization(...))  # Only allow charge or hold

        elif rule_b[0] == 'LIMIT_DISCHARGE':
            P_max_temp = rule_b[1] * self.P_max
            P_dispatch = self.mpc_optimization(..., P_max=P_max_temp)

        elif rule_c == 'FAST_PROFIT_TAKING':
            # Fast profit-taking: if discharging, continue; if not, do not enter
            if state['last_action'] == 'discharge':
                P_dispatch = self.P_max  # Full discharge
            else:
                P_dispatch = 0

        elif rule_c == 'FORCE_PROFIT_TAKING':
            # Forced profit-taking: stop discharge, prepare to recharge
            P_dispatch = -self.P_max * 0.5  # Low-power recharging

        else:
            # Normal MPC optimization
            P_dispatch = self.mpc_optimization(state, prediction)

        return P_dispatch
```

### 5.5 Backtesting Framework

```python
def backtest_strategy(strategy, historical_data, initial_SOC=0.7):
    """
    Strategy backtesting
    """
    results = []
    state = {'SOC': initial_SOC, 'last_action': None}

    for t in historical_data.index:
        # Get current market data
        market_data = historical_data.loc[:t]

        # Strategy decision
        P_dispatch = strategy.dispatch(t, state, market_data)

        # Execute and update state
        revenue_t = historical_data.loc[t, 'P_CPS'] * P_dispatch * (5/60)
        state['SOC'] -= P_dispatch * (5/60) / E_cap
        state['last_action'] = 'discharge' if P_dispatch > 0 else 'charge'

        # Record
        results.append({
            'timestamp': t,
            'P_dispatch': P_dispatch,
            'SOC': state['SOC'],
            'revenue': revenue_t,
            'price': historical_data.loc[t, 'P_CPS'],
        })

    df_results = pd.DataFrame(results)

    # Evaluation
    total_revenue = df_results['revenue'].sum()
    max_revenue_window = df_results[
        df_results['price'] > 400
    ]['revenue'].sum()

    print(f"Total Revenue: ${total_revenue:,.2f}")
    print(f"Revenue in Spike Window (P>400): ${max_revenue_window:,.2f}")

    return df_results
```

---

## 6. Implementation Roadmap

**Core Strategy**: First train + backtest with historical data, then demonstrate PowerA capabilities through the 12/14-12/15 Case Study, and finally plan real-time system deployment

### 6.1 Phase 1: Historical Data Preparation
**Objective**: Collect and process historical data, establish training/testing datasets

- [ ] Data Collection (In Progress)
  - Download ERCOT historical data (2024-2025 RTM, DAM, Fuel Mix)
  - Collect historical weather data (NOAA/Weather API)
  - Identify historical spike events (build event database)
  - Data quality checks and cleaning

- [x] Feature Engineering Implementation (Completed 2025-12-29)
  - Implemented feature computation module (price structure, supply-demand balance, weather-driven)
  - Generated labels (SpikeEvent, LeadSpike, Regime)
  - Feature-label alignment validation
  - Feature statistical analysis and visualization

- [ ] Dataset Split
  - Training set: 2024-01 to 2025-11
  - Validation set: 2025-12-01 to 2025-12-10
  - Test set: 2025-12-11 to 2025-12-20 (includes 12/14-15 events)

**Deliverables**:
- Historical datasets (CSV/Parquet format)
- Feature computation codebase ([feature_engineering.py](src/data/feature_engineering.py))
- Label generation scripts ([labels.py](src/utils/labels.py))
- Test suite (24 tests, all passing)
- Data exploration analysis report

### 6.2 Phase 2: Model Training and Validation
**Objective**: Train prediction models, validate performance on historical spike events

- [ ] Baseline Model Development
  - Train XGBoost/LightGBM
  - Feature importance analysis
  - Hyperparameter tuning
  - Validation set performance evaluation

- [ ] Advanced Model Development (Optional)
  - Implement CfC/LTC architecture
  - Multi-task learning training
  - Model ensemble

- [ ] Model Evaluation
  - Evaluate on test set (AUC, PR-AUC, Event Recall)
  - 12/14-15 event prediction analysis
  - False positive/false negative case analysis
  - Model interpretability analysis

**Deliverables**:
- Trained model files
- Model performance evaluation report
- Feature importance ranking
- Error case analysis

### 6.3 Phase 3: Strategy Backtesting
**Objective**: Implement BESS strategy and backtest on historical data, quantify revenue improvement

- [ ] Strategy Implementation
  - Implement three hard rules (Rule A/B/C)
  - Implement MPC optimization framework (optional, rules-first approach)
  - Integrate prediction model with strategy decisions

- [ ] Backtesting Framework
  - Implement backtesting engine
  - Define evaluation metrics (revenue, SOC utilization, risk metrics)
  - Strategy parameter sensitivity analysis

- [ ] Comparative Analysis
  - Baseline strategy: Traditional valley-charge, peak-discharge
  - New strategy: Prediction-driven + Hold-SOC
  - Compare revenue on historical spike events

**Deliverables**:
- Strategy codebase
- Backtesting engine
- Historical revenue comparison report
- Strategy optimization recommendations

### 6.4 Phase 4: 12/14-15 Case Study
**Objective**: Conduct in-depth case analysis of the 12/14-15 event, demonstrating PowerA technical capabilities

- [ ] Event Reconstruction
  - Reconstruct 12/14-15 price curves and market states
  - Visualize key feature evolution
  - Analyze system state transitions on an interval-by-interval basis

- [ ] Prediction Capability Demonstration
  - Model warning probability at 12/14 16:00
  - Regime state recognition timeline
  - Comparison with actual spike occurrence time

- [ ] Strategy Comparison
  - Reconstruct actual ESR behavior (16:30-20:00 early discharge)
  - PowerA strategy simulation (Hold-SOC until 20:00-22:00)
  - Revenue difference quantification (estimated 30-60% improvement)

- [ ] Visualization Report
  - Timeline visualization (price, features, predictions, strategy)
  - Revenue comparison charts
  - Key decision point annotations

**Deliverables**:
- 12/14-15 Case Study report (PDF/PPT)
- Visualization Dashboard (Jupyter Notebook / Streamlit)
- Demo video/animation
- Technical white paper

### 6.5 Phase 5: Real-Time System Planning (Future)
**Objective**: Design real-time data flow and online prediction system architecture

- [ ] Real-Time Data Architecture
  - Real-time data ingestion plan (ERCOT API/WebSocket)
  - Time series database selection and deployment
  - Data stream processing (Kafka/Spark Streaming)

- [ ] Model Serving
  - Model deployment plan (FastAPI/TorchServe)
  - Real-time feature computation optimization
  - Low-latency inference (<1s)

- [ ] Monitoring and Alerting
  - Prediction monitoring Dashboard
  - Spike warning notification system
  - Model performance monitoring

**Deliverables**:
- Real-time system architecture design document
- Technology selection report
- POC prototype (optional)

### 6.6 Phase 6: Production Deployment (Future)
**Objective**: Actual deployment and operations

- [ ] Shadow run
- [ ] Canary release
- [ ] Full rollout
- [ ] Continuous optimization

---

## Appendix

### A. Recommended Technology Stack

**Data Layer**:
- Time Series Database: InfluxDB 2.x / TimescaleDB
- Feature Store: Feast / Tecton
- Data Quality: Great Expectations

**Model Layer**:
- Baseline: XGBoost, LightGBM
- Deep Learning: PyTorch + ncps (CfC/LTC)
- Experiment Tracking: MLflow / Weights & Biases

**Strategy Layer**:
- Optimizer: scipy.optimize / cvxpy (convex optimization)
- Backtesting: Backtrader / custom-built

**Deployment Layer**:
- Inference Service: FastAPI + uvicorn
- Containerization: Docker
- Orchestration: Kubernetes (optional)
- Monitoring: Prometheus + Grafana

### B. Key File Structure
```
spike-forecast/
├── data/
│   ├── raw/              # Raw data
│   ├── processed/        # Processed data
│   └── features/         # Feature data
├── src/
│   ├── data/
│   │   ├── ingestion.py      # Data ingestion
│   │   ├── preprocessing.py  # Preprocessing
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── baseline.py       # XGBoost/LightGBM
│   │   ├── cfc_model.py      # CfC/LTC
│   │   └── ensemble.py       # Ensemble
│   ├── strategy/
│   │   ├── rules.py          # Hard rules
│   │   ├── mpc.py            # MPC optimization
│   │   └── backtest.py       # Backtesting framework
│   └── utils/
│       ├── labels.py         # Label generation
│       └── metrics.py        # Evaluation metrics
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory data analysis
│   ├── 02_feature_analysis.ipynb
│   └── 03_model_evaluation.ipynb
├── configs/
│   ├── data_config.yaml
│   ├── model_config.yaml
│   └── strategy_config.yaml
├── tests/
├── docs/
│   └── DESIGN.md (this document)
├── requirements.txt
└── README.md
```

---

## Progress Log

### 2025-12-29
**Completed**: Phase 1 - Feature Engineering Implementation

- Project structure created
- Feature engineering module implemented ([feature_engineering.py](src/data/feature_engineering.py))
  - Price structure features (6 features x 3 zones)
  - Supply-demand balance features (15 features)
  - Weather-driven features (4 features x 3 zones)
  - Temporal features (8 features)
  - Total: ~50 features
- Label generation module implemented ([labels.py](src/utils/labels.py))
  - SpikeEvent label generation
  - LeadSpike early warning label (60-minute lead time)
  - Regime state classification (Normal/Tight/Scarcity)
  - Support for both fixed threshold and rolling quantile modes
- Test suite created
  - test_feature_engineering.py: 13 tests
  - test_labels.py: 11 tests
  - All tests passing
- Example script created ([01_feature_engineering_example.py](notebooks/01_feature_engineering_example.py))
- Project documentation created (README.md, requirements.txt)

**Next Steps**: Wait for data download to complete, then proceed with feature analysis and visualization

---

**Document Version**: v1.1
**Last Updated**: 2025-12-29
