# ERCOT RTM LMP Spike 预测算法设计文档

## 目录
1. [系统架构](#系统架构)
2. [数据架构](#数据架构)
3. [特征工程](#特征工程)
4. [预测模型](#预测模型)
5. [策略优化](#策略优化)
6. [实施路线](#实施路线)

---

## 1. 系统架构

### 1.1 总体架构
```
┌─────────────────────────────────────────────────────────────┐
│                     数据采集层 (Data Layer)                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │ 市场数据 │ │ 系统数据 │ │ 天气数据 │ │ 约束数据 │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   特征工程层 (Feature Layer)                  │
│  ┌────────────────────────────────────────────────────┐     │
│  │ Zone-level Features (CPS, West, Houston, Hub)      │     │
│  │ • 价格结构特征 • 供需平衡特征 • 天气驱动特征       │     │
│  └────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   预测模型层 (Model Layer)                    │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │ Spike Event  │ │ Regime State │ │ Price Quantile│       │
│  │ Prediction   │ │ Recognition  │ │ Forecast      │       │
│  └──────────────┘ └──────────────┘ └──────────────┘        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   策略优化层 (Strategy Layer)                 │
│  ┌────────────────────────────────────────────────────┐     │
│  │ BESS Dispatch Optimizer                            │     │
│  │ • Hold-SOC Rules • MPC Optimization • Risk Control │     │
│  └────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 核心模块
- **数据采集模块**：实时数据获取、预处理、质量控制
- **特征计算模块**：实时特征计算、滚动窗口统计
- **预测引擎**：Zone-level spike 预警、状态识别
- **策略引擎**：SOC 管理、功率调度、风险控制

---

## 2. 数据架构

### 2.1 数据源清单

#### 市场数据 (5-min/15-min)
| 数据项 | 字段名 | 单位 | 更新频率 | 数据源 |
|--------|--------|------|----------|--------|
| RT LMP - LZ CPS | `P_CPS` | $/MWh | 5-min | ERCOT RTM |
| RT LMP - LZ West | `P_West` | $/MWh | 5-min | ERCOT RTM |
| RT LMP - LZ Houston | `P_Houston` | $/MWh | 5-min | ERCOT RTM |
| RT Hub Average | `P_Hub` | $/MWh | 5-min | ERCOT RTM |
| DA LMP (对应时段) | `P_DA_*` | $/MWh | hourly | ERCOT DAM |

#### 系统数据
| 数据项 | 字段名 | 单位 | 更新频率 | 数据源 |
|--------|--------|------|----------|--------|
| 系统负荷 | `Load` | MW | 5-min | ERCOT |
| 风电出力 | `Wind` | MW | 5-min | ERCOT Fuel Mix |
| 光伏出力 | `Solar` | MW | 5-min | ERCOT Fuel Mix |
| 天然气出力 | `Gas` | MW | 5-min | ERCOT Fuel Mix |
| 煤电出力 | `Coal` | MW | 5-min | ERCOT Fuel Mix |
| 储能净出力 | `ESR` | MW | 5-min | ERCOT Fuel Mix |
| 净负荷 | `NetLoad` | MW | 5-min | 计算得出 |

#### 天气数据 (Zone-level)
| 数据项 | 字段名 | 单位 | 更新频率 | 数据源 |
|--------|--------|------|----------|--------|
| 温度 | `T_{zone}` | °F | 15-min | NOAA/Weather API |
| 风速 | `WindSpeed_{zone}` | mph | 15-min | NOAA/Weather API |
| 风向 | `WindDir_{zone}` | degree | 15-min | NOAA/Weather API |
| 体感温度 | `WindChill_{zone}` | °F | 15-min | 计算得出 |

#### 约束数据 (可选，增强版)
| 数据项 | 字段名 | 单位 | 更新频率 | 数据源 |
|--------|--------|------|----------|--------|
| 绑定约束列表 | `BindingConstraints` | - | 5-min | ERCOT |
| 断面影子价格 | `ShadowPrice_*` | $/MW | 5-min | ERCOT |

### 2.2 数据存储方案

#### 时序数据库 (推荐 InfluxDB/TimescaleDB)
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

#### 特征存储 (Feature Store)
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

## 3. 特征工程

### 3.1 特征分组

#### 组 1: 价格结构特征 (Price Structure)
**目标**: 捕捉区域性稀缺与拥塞信号

| 特征名 | 计算公式 | 含义 | 阈值示例 |
|--------|----------|------|----------|
| `Spread_zone_hub` | `P_zone(t) - P_Hub(t)` | 区域-系统价差 | >50 $/MWh |
| `Spread_CPS_Houston` | `P_CPS(t) - P_Houston(t)` | CPS-Houston 价差 | >80 $/MWh |
| `Spread_RT_DA` | `P_zone^RT(t) - P_zone^DA(t)` | 实时-日前溢价 | >100 $/MWh |
| `PriceRamp_5m` | `(P(t) - P(t-5)) / 5` | 5分钟价格斜率 | >10 $/MWh/min |
| `PriceRamp_15m` | `(P(t) - P(t-15)) / 15` | 15分钟价格斜率 | >5 $/MWh/min |
| `PriceAccel` | `PriceRamp_5m(t) - PriceRamp_5m(t-5)` | 价格加速度 | >2 $/MWh/min² |

**实现示例**:
```python
def calc_price_features(df, zone='CPS'):
    features = {}

    # 价差特征
    features[f'spread_{zone}_hub'] = df[f'P_{zone}'] - df['P_Hub']
    features[f'spread_rt_da'] = df[f'P_{zone}'] - df[f'P_{zone}_DA']

    # 斜率特征
    features[f'price_ramp_5m'] = df[f'P_{zone}'].diff(1) / 5  # 5-min data
    features[f'price_ramp_15m'] = df[f'P_{zone}'].diff(3) / 15

    # 加速度
    features[f'price_accel'] = features[f'price_ramp_5m'].diff(1)

    return pd.DataFrame(features)
```

#### 组 2: 供需平衡特征 (Supply-Demand Balance)
**目标**: 捕捉系统/区域紧张状态

| 特征名 | 计算公式 | 含义 | 阈值示例 |
|--------|----------|------|----------|
| `NetLoad` | `Load(t) - Wind(t) - Solar(t)` | 净负荷 | - |
| `NetLoadRamp_5m` | `dNetLoad/dt` | 净负荷爬坡速度 | >200 MW/5min |
| `NetLoadAccel` | `d²NetLoad/dt²` | 净负荷加速度 | >50 MW/5min² |
| `WindAnomaly` | `(Wind(t) - Wind_MA30d(t)) / Wind_Std30d(t)` | 风电异常 | <-1.5 σ |
| `WindRamp` | `dWind/dt` | 风电变化率 | <-100 MW/5min |
| `GasSaturation` | `Gas(t) / Gas_P95_7d` | 气电饱和度 | >0.95 |
| `CoalStress` | `I(Coal(t) > Coal(t-1)) & I(hour ∈ [0,5])` | 夜间煤电上行 | 1 (True) |
| `ESRNetOutput` | `ESR(t)` | 储能净出力 | <-1000 MW (充电) |

**实现示例**:
```python
def calc_supply_demand_features(df):
    features = {}

    # 净负荷
    features['net_load'] = df['Load'] - df['Wind'] - df['Solar']
    features['net_load_ramp_5m'] = features['net_load'].diff(1)
    features['net_load_accel'] = features['net_load_ramp_5m'].diff(1)

    # 风电异常
    wind_ma = df['Wind'].rolling(window=30*24*12, min_periods=1).mean()  # 30天滚动均值
    wind_std = df['Wind'].rolling(window=30*24*12, min_periods=1).std()
    features['wind_anomaly'] = (df['Wind'] - wind_ma) / wind_std
    features['wind_ramp'] = df['Wind'].diff(1)

    # 气电饱和度
    gas_p95 = df['Gas'].rolling(window=7*24*12, min_periods=1).quantile(0.95)
    features['gas_saturation'] = df['Gas'] / gas_p95

    # 煤电压力
    features['coal_stress'] = (
        (df['Coal'].diff(1) > 0) &
        (df.index.hour.isin(range(0, 6)))
    ).astype(int)

    # 储能
    features['esr_net_output'] = df['ESR']

    return pd.DataFrame(features)
```

#### 组 3: 天气驱动特征 (Weather-Driven, Zone-level)
**目标**: 捕捉需求侧冲击信号

| 特征名 | 计算公式 | 含义 | 阈值示例 |
|--------|----------|------|----------|
| `T_anomaly_zone` | `T(t) - T_MA30d(hour)` | 温度异常 | <-10 °F |
| `T_ramp_zone` | `dT/dt` | 降温速度 | <-5 °F/hr |
| `WindChill_zone` | `35.74 + 0.6215T - 35.75v^0.16 + 0.4275Tv^0.16` | 风寒指数 | <20 °F |
| `ColdFront_zone` | `I(ΔT < -5 & WindShift to N)` | 冷锋标志 | 1 (True) |

**实现示例**:
```python
def calc_weather_features(df, zone='CPS'):
    features = {}

    # 温度异常 (相对于历史同小时均值)
    t_hourly_mean = df.groupby(df.index.hour)[f'T_{zone}'].transform(
        lambda x: x.rolling(window=30*24, min_periods=1).mean()
    )
    features[f'T_anomaly_{zone}'] = df[f'T_{zone}'] - t_hourly_mean

    # 降温速度
    features[f'T_ramp_{zone}'] = df[f'T_{zone}'].diff(12) / 1  # 12个5-min = 1小时

    # 风寒指数
    T = df[f'T_{zone}']
    v = df[f'WindSpeed_{zone}']
    features[f'WindChill_{zone}'] = (
        35.74 + 0.6215*T - 35.75*(v**0.16) + 0.4275*T*(v**0.16)
    )

    # 冷锋标志
    wind_to_north = (df[f'WindDir_{zone}'] > 315) | (df[f'WindDir_{zone}'] < 45)
    features[f'ColdFront_{zone}'] = (
        (features[f'T_ramp_{zone}'] < -5) & wind_to_north
    ).astype(int)

    return pd.DataFrame(features)
```

#### 组 4: 时间特征 (Temporal)
**目标**: 捕捉日内模式与光伏修复窗口

| 特征名 | 计算公式 | 含义 |
|--------|----------|------|
| `hour` | `hour of day` | 小时 (0-23) |
| `is_evening_peak` | `I(hour ∈ [17, 22])` | 晚高峰标志 |
| `minutes_to_sunrise` | `sunrise_time - current_time` | 距日出时间 (分钟) |
| `solar_ramp_expected` | `dSolar_forecast/dt` | 预期光伏爬坡 |

### 3.2 特征重要性（基于文档分析）

**Tier 1 (强因果/触发信号)**:
1. `Spread_zone_hub` - 区域价差扩张
2. `NetLoadAccel` - 净负荷加速
3. `WindAnomaly` - 风电缓冲塌陷
4. `T_anomaly_zone` - 温度异常
5. `PriceAccel` - 价格加速度

**Tier 2 (确认/放大信号)**:
6. `GasSaturation` - 气电饱和
7. `CoalStress` - 煤电夜间上行
8. `ESRNetOutput` - 储能充电放大
9. `ColdFront_zone` - 冷锋事件
10. `T_ramp_zone` - 快速降温

**Tier 3 (修复/终止信号)**:
11. `minutes_to_sunrise` - 光伏修复倒计时
12. `solar_ramp_expected` - 光伏爬坡预期

---

## 4. 预测模型

### 4.1 标签生成 (Label Generation)

#### 4.1.1 SpikeEvent 标签
**定义**: Zone z 在时刻 t 是否处于 spike 状态

**规则**:
```python
def generate_spike_label(df, zone='CPS', P_hi=400, S_hi=50, S_cross_hi=80, m=3):
    """
    生成 SpikeEvent 标签

    参数:
        df: 数据框，包含价格和价差字段
        zone: 区域名称
        P_hi: 价格阈值 ($/MWh)
        S_hi: zone-hub 价差阈值 ($/MWh)
        S_cross_hi: zone-houston 价差阈值 ($/MWh)
        m: 持续时间阈值 (时间步数)

    返回:
        SpikeEvent_{zone}: 0/1 标签
    """
    # 条件 A: 价格高
    cond_price = df[f'P_{zone}'] >= P_hi

    # 条件 B: 价差大 (约束主导)
    spread_zh = df[f'P_{zone}'] - df['P_Hub']
    spread_cross = df[f'P_{zone}'] - df['P_Houston']
    cond_spread = (spread_zh >= S_hi) | (spread_cross >= S_cross_hi)

    # 条件 C: 持续时间
    raw_spike = cond_price & cond_spread
    sustained_spike = raw_spike.rolling(window=m).sum() >= m

    return sustained_spike.fillna(False).astype(int)
```

#### 4.1.2 LeadSpike 标签
**定义**: 未来 H 分钟内是否会发生 spike

```python
def generate_lead_spike_label(df, zone='CPS', H=60, dt=5):
    """
    生成 LeadSpike 标签 (提前预警)

    参数:
        H: 预警时间窗口 (分钟)
        dt: 数据时间分辨率 (分钟)

    返回:
        LeadSpike_{zone}_{H}m: 0/1 标签
    """
    spike_event = df[f'SpikeEvent_{zone}']
    k = int(H / dt)  # 窗口大小

    # 反转 -> 滚动最大值 -> 再反转
    lead_spike = spike_event[::-1].rolling(window=k).max()[::-1]

    return lead_spike.fillna(0).astype(int)
```

#### 4.1.3 Regime 标签
**定义**: 系统状态 (Normal / Tight / Scarcity)

```python
def generate_regime_label(df, zone='CPS', P_mid=150, S_mid=20):
    """
    生成 Regime 状态标签

    返回:
        Regime_{zone}: 'Normal' / 'Tight' / 'Scarcity'
    """
    spread_zh = df[f'P_{zone}'] - df['P_Hub']
    spike_event = df[f'SpikeEvent_{zone}']

    regime = pd.Series('Normal', index=df.index)

    # Tight 状态
    tight_cond = (df[f'P_{zone}'] >= P_mid) | (spread_zh >= S_mid)
    regime[tight_cond] = 'Tight'

    # Scarcity 状态 (优先级最高)
    regime[spike_event == 1] = 'Scarcity'

    return regime
```

### 4.2 模型架构

#### 4.2.1 模型选择
**推荐方案**: 分层建模 + 集成

```
┌─────────────────────────────────────────────────────┐
│ Layer 1: Baseline (XGBoost/LightGBM)               │
│ - 快速验证特征有效性                                │
│ - 提供可解释的特征重要性                            │
│ - 作为集成模型的一部分                              │
└─────────────────────────────────────────────────────┘
                        +
┌─────────────────────────────────────────────────────┐
│ Layer 2: Continuous-Time Model (CfC/LTC)           │
│ - 处理不规则时间间隔                                │
│ - 捕捉状态跃迁动态                                  │
│ - 适配事件驱动特性                                  │
└─────────────────────────────────────────────────────┘
                        =
┌─────────────────────────────────────────────────────┐
│ Ensemble Prediction                                 │
│ p_hat(t) = α·p_baseline(t) + (1-α)·p_cfc(t)        │
└─────────────────────────────────────────────────────┘
```

#### 4.2.2 输入输出设计

**输入** (每个时刻 t):
```python
X(t) = {
    # 价格结构 (6 维 × 3 zones = 18 维)
    'spread_zone_hub', 'spread_rt_da', 'price_ramp_5m',
    'price_ramp_15m', 'price_accel', ...

    # 供需平衡 (8 维)
    'net_load', 'net_load_ramp', 'net_load_accel',
    'wind_anomaly', 'wind_ramp', 'gas_saturation',
    'coal_stress', 'esr_net_output',

    # 天气驱动 (4 维 × 3 zones = 12 维)
    'T_anomaly_CPS', 'T_ramp_CPS', 'WindChill_CPS', 'ColdFront_CPS',
    'T_anomaly_West', ...

    # 时间特征 (4 维)
    'hour', 'is_evening_peak', 'minutes_to_sunrise', 'solar_ramp_expected',

    # 历史价格 (滑动窗口)
    'P_CPS_lag_1', 'P_CPS_lag_3', 'P_CPS_lag_12',  # 5/15/60 分钟前
}
总维度: ~40-50 维
```

**输出** (多任务):
```python
Y(t) = {
    # 主任务: Spike 预警
    'LeadSpike_CPS_60m': P(SpikeEvent_CPS(t+60) = 1),  # [0, 1]
    'LeadSpike_West_60m': P(SpikeEvent_West(t+60) = 1),
    'LeadSpike_Houston_60m': P(SpikeEvent_Houston(t+60) = 1),

    # 辅助任务: Regime 识别
    'Regime_prob': {
        'Normal': [0, 1],
        'Tight': [0, 1],
        'Scarcity': [0, 1]
    },  # 三分类概率

    # 可选任务: 价格分位数
    'P_CPS_P90_60m': 第90百分位价格预测,  # 用于风控
}
```

#### 4.2.3 模型实现框架

**XGBoost Baseline**:
```python
import xgboost as xgb

# 多任务训练
def train_baseline_model(X_train, y_train):
    models = {}

    for zone in ['CPS', 'West', 'Houston']:
        # Spike 预警模型
        model_spike = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            max_depth=6,
            learning_rate=0.05,
            n_estimators=500,
            scale_pos_weight=10,  # 处理类别不平衡
        )
        model_spike.fit(
            X_train,
            y_train[f'LeadSpike_{zone}_60m'],
            eval_set=[(X_val, y_val[f'LeadSpike_{zone}_60m'])],
            early_stopping_rounds=50,
        )
        models[f'spike_{zone}'] = model_spike

        # Regime 识别模型
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
# 使用 ncps 库 (Neural Circuit Policies)
from ncps.torch import CfC
import torch
import torch.nn as nn

class SpikeForecastCfC(nn.Module):
    def __init__(self, input_size, hidden_size, num_zones=3):
        super().__init__()

        # CfC 核心
        self.cfc = CfC(
            input_size=input_size,
            units=hidden_size,
            mode='default',
        )

        # 多任务输出头
        self.spike_heads = nn.ModuleDict({
            zone: nn.Linear(hidden_size, 1)
            for zone in ['CPS', 'West', 'Houston']
        })

        self.regime_head = nn.Linear(hidden_size, 3)  # 3-class

    def forward(self, x, time_deltas):
        """
        x: [batch, seq_len, features]
        time_deltas: [batch, seq_len] - 时间间隔 (秒)
        """
        # CfC 处理不规则时间序列
        h, _ = self.cfc(x, time_deltas)

        # 多任务输出
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

### 4.3 训练策略

#### 4.3.1 数据划分
```python
# 时序划分 (避免数据泄漏)
train_end = '2025-11-30'
val_start = '2025-12-01'
val_end = '2025-12-10'
test_start = '2025-12-11'  # 包含 12-14/12-15 事件

train_data = df[df.index < train_end]
val_data = df[(df.index >= val_start) & (df.index < val_end)]
test_data = df[df.index >= test_start]
```

#### 4.3.2 损失函数
```python
# 多任务损失
def multi_task_loss(pred, target, alpha=0.7, beta=0.3):
    """
    alpha: spike 预警任务权重
    beta: regime 识别任务权重
    """
    # Spike 预警: Focal Loss (处理类别不平衡)
    loss_spike = focal_loss(
        pred['spike_CPS'],
        target['LeadSpike_CPS_60m'],
        gamma=2.0,  # 聚焦难样本
    )

    # Regime 识别: Cross-Entropy
    loss_regime = nn.CrossEntropyLoss()(
        pred['regime'],
        target['Regime'],
    )

    return alpha * loss_spike + beta * loss_regime
```

#### 4.3.3 评估指标
```python
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

def evaluate_model(y_true, y_pred):
    metrics = {}

    # 1. Spike 预警性能
    metrics['AUC'] = roc_auc_score(y_true['LeadSpike'], y_pred['spike_prob'])

    precision, recall, _ = precision_recall_curve(
        y_true['LeadSpike'],
        y_pred['spike_prob']
    )
    metrics['PR-AUC'] = auc(recall, precision)

    # 2. 事件级 Recall (关键!)
    # 在真实 spike 事件前 60 分钟内，是否至少预警一次
    events = identify_spike_events(y_true['SpikeEvent'])
    metrics['Event_Recall'] = compute_event_recall(events, y_pred, lead_time=60)

    # 3. Regime 识别准确率
    metrics['Regime_Accuracy'] = accuracy_score(
        y_true['Regime'],
        y_pred['regime_pred']
    )

    # 4. 假阳率 (代价高的指标)
    metrics['False_Positive_Rate'] = compute_fpr(y_true, y_pred, threshold=0.5)

    return metrics
```

---

## 5. 策略优化

### 5.1 策略框架

```
Input:
  - p_hat(t): Spike 预警概率
  - regime(t): 系统状态
  - SOC(t): 当前电量
  - P_max: 最大功率
  - market_data(t): 实时市场信息

Output:
  - P_dispatch(t): 调度功率 (正=放电, 负=充电)
  - action: 'charge' / 'discharge' / 'hold'
```

### 5.2 硬规则层 (Safety Layer)

#### Rule A: Locational 禁充规则
```python
def rule_anti_local_charging(market_data, zone='CPS', threshold_spread=50):
    """
    防止在区域高价时充电
    """
    P_zone = market_data[f'P_{zone}']
    P_hub = market_data['P_Hub']
    spread = P_zone - P_hub

    # 条件 1: 区域价格高
    cond1 = P_zone > 200

    # 条件 2: 价差快速扩张
    spread_ramp = spread - spread_prev
    cond2 = (spread > threshold_spread) or (spread_ramp > 20)

    if cond1 and cond2:
        return 'FORBID_CHARGE'
    else:
        return 'ALLOW'
```

#### Rule B: SOC 保留规则
```python
def rule_soc_reservation(p_hat, regime, SOC, SOC_min=0.2, SOC_reserve=0.6):
    """
    在预警阶段保留 SOC
    """
    # 进入 Tight 状态 或 预警概率高
    if regime == 'Tight' or p_hat > 0.4:
        if SOC < SOC_reserve:
            return 'FORBID_DISCHARGE'
        else:
            return 'LIMIT_DISCHARGE', 0.3  # 限制为 30% 功率

    # 进入 Scarcity 状态 → 允许全力放电
    if regime == 'Scarcity' or p_hat > 0.8:
        return 'ALLOW_FULL_DISCHARGE'

    return 'NORMAL'
```

#### Rule C: 光伏修复倒计时
```python
def rule_solar_repair_countdown(market_data, minutes_to_sunrise, solar_ramp_forecast):
    """
    接近日出时止盈
    """
    # 距日出 < 30 分钟 且 光伏开始爬坡
    if minutes_to_sunrise < 30 and solar_ramp_forecast > 100:
        return 'FAST_PROFIT_TAKING'

    # 距日出 < 10 分钟 → 强制止盈/回充
    if minutes_to_sunrise < 10:
        return 'FORCE_PROFIT_TAKING'

    return 'NORMAL'
```

### 5.3 MPC 优化层

#### 5.3.1 目标函数
```python
"""
优化目标:
  Maximize: Revenue - Risk_Penalty

  Revenue = Σ [P_zone(t) · P_discharge(t) - P_zone(t) · P_charge(t)] · Δt

  Risk_Penalty = λ₁·R_spread(t) + λ₂·R_time_decay(t) + λ₃·R_congestion(t)

约束:
  1. SOC 动态: SOC(t+1) = SOC(t) - P_discharge(t)·Δt/E_cap + P_charge(t)·Δt·η/E_cap
  2. SOC 边界: SOC_min ≤ SOC(t) ≤ SOC_max
  3. 功率边界: -P_max ≤ P(t) ≤ P_max
  4. 硬规则约束: Rule A/B/C 的逻辑约束
"""

from scipy.optimize import minimize

def mpc_optimization(
    state,
    forecast,
    horizon=12,  # 60 分钟 (12 × 5min)
    SOC_current=0.7,
    E_cap=100,  # MWh
    P_max=25,   # MW
):
    """
    MPC 滚动优化
    """
    # 决策变量: P(t), t=0,1,...,horizon-1
    def objective(P):
        revenue = 0
        risk = 0
        SOC = SOC_current

        for t in range(horizon):
            # 收益
            if P[t] > 0:  # 放电
                revenue += forecast['P_zone'][t] * P[t] * (5/60)
            else:  # 充电
                revenue += forecast['P_zone'][t] * P[t] * (5/60)  # P<0, 所以是负贡献

            # 风险项
            # R1: 价差变化风险
            spread_vol = np.std(forecast['spread_zone_hub'][t:t+3])
            risk += 0.1 * spread_vol * abs(P[t])

            # R2: SOC 时间价值衰减
            time_to_sunrise = forecast['minutes_to_sunrise'][t]
            if time_to_sunrise < 60:
                decay_penalty = (60 - time_to_sunrise) / 60
                risk += 0.5 * decay_penalty * SOC

            # R3: 拥塞不确定性
            if forecast['regime'][t] == 'Tight':
                risk += 0.2 * abs(P[t])

            # 更新 SOC
            SOC -= P[t] * (5/60) / E_cap  # 简化,未考虑效率

        return -(revenue - risk)  # minimize 负收益

    # 约束
    constraints = []

    # SOC 约束
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

    # 硬规则约束 (示例: 禁充)
    for t in range(horizon):
        if forecast['forbid_charge'][t]:
            # P(t) >= 0 (不允许充电)
            constraints.append({
                'type': 'ineq',
                'fun': lambda P, t=t: P[t]
            })

    # 边界
    bounds = [(-P_max, P_max) for _ in range(horizon)]

    # 求解
    result = minimize(
        objective,
        x0=np.zeros(horizon),  # 初始猜测
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
    )

    # 返回第一步决策
    return result.x[0]
```

### 5.4 策略集成

```python
class BESSDispatchStrategy:
    def __init__(self, model, mpc_config):
        self.model = model
        self.mpc_config = mpc_config

    def dispatch(self, t, state, market_data):
        """
        实时调度决策
        """
        # 1. 模型预测
        X_t = self.extract_features(market_data, t)
        prediction = self.model.predict(X_t)

        p_hat = prediction['spike_prob_CPS']
        regime = prediction['regime']

        # 2. 硬规则检查
        rule_a = rule_anti_local_charging(market_data, zone='CPS')
        rule_b = rule_soc_reservation(p_hat, regime, state['SOC'])
        rule_c = rule_solar_repair_countdown(
            market_data,
            state['minutes_to_sunrise'],
            prediction['solar_ramp_forecast']
        )

        # 3. 决策逻辑
        if rule_a == 'FORBID_CHARGE':
            P_dispatch = max(0, self.mpc_optimization(...))  # 只允许放电或持有

        elif rule_b == 'FORBID_DISCHARGE':
            P_dispatch = min(0, self.mpc_optimization(...))  # 只允许充电或持有

        elif rule_b[0] == 'LIMIT_DISCHARGE':
            P_max_temp = rule_b[1] * self.P_max
            P_dispatch = self.mpc_optimization(..., P_max=P_max_temp)

        elif rule_c == 'FAST_PROFIT_TAKING':
            # 快速止盈: 如果在放电,继续;如果未放,不再进入
            if state['last_action'] == 'discharge':
                P_dispatch = self.P_max  # 全力放电
            else:
                P_dispatch = 0

        elif rule_c == 'FORCE_PROFIT_TAKING':
            # 强制止盈: 停止放电,准备回充
            P_dispatch = -self.P_max * 0.5  # 小功率回充

        else:
            # 常规 MPC 优化
            P_dispatch = self.mpc_optimization(state, prediction)

        return P_dispatch
```

### 5.5 回测框架

```python
def backtest_strategy(strategy, historical_data, initial_SOC=0.7):
    """
    策略回测
    """
    results = []
    state = {'SOC': initial_SOC, 'last_action': None}

    for t in historical_data.index:
        # 获取当前市场数据
        market_data = historical_data.loc[:t]

        # 策略决策
        P_dispatch = strategy.dispatch(t, state, market_data)

        # 执行并更新状态
        revenue_t = historical_data.loc[t, 'P_CPS'] * P_dispatch * (5/60)
        state['SOC'] -= P_dispatch * (5/60) / E_cap
        state['last_action'] = 'discharge' if P_dispatch > 0 else 'charge'

        # 记录
        results.append({
            'timestamp': t,
            'P_dispatch': P_dispatch,
            'SOC': state['SOC'],
            'revenue': revenue_t,
            'price': historical_data.loc[t, 'P_CPS'],
        })

    df_results = pd.DataFrame(results)

    # 评估
    total_revenue = df_results['revenue'].sum()
    max_revenue_window = df_results[
        df_results['price'] > 400
    ]['revenue'].sum()

    print(f"Total Revenue: ${total_revenue:,.2f}")
    print(f"Revenue in Spike Window (P>400): ${max_revenue_window:,.2f}")

    return df_results
```

---

## 6. 实施路线

**核心策略**: 先用历史数据训练+回测，再通过 12/14-12/15 Case Study 展示 PowerA 能力，最后规划实时系统上线

### 6.1 Phase 1: 历史数据准备
**目标**: 收集并处理历史数据，建立训练/测试数据集

- [ ] 数据收集 ⏳ 进行中
  - 下载 ERCOT 历史数据 (2024-2025 RTM, DAM, Fuel Mix)
  - 收集历史天气数据 (NOAA/气象 API)
  - 识别历史 spike 事件（建立事件库）
  - 数据质量检查与清洗

- [x] 特征工程实现 ✅ 已完成 (2025-12-29)
  - ✅ 实现特征计算模块（价格结构、供需平衡、天气驱动）
  - ✅ 生成标签（SpikeEvent, LeadSpike, Regime）
  - ✅ 特征-标签对齐验证
  - 特征统计分析与可视化

- [ ] 数据集划分
  - 训练集：2024-01 至 2025-11
  - 验证集：2025-12-01 至 2025-12-10
  - 测试集：2025-12-11 至 2025-12-20（包含 12/14-15 事件）

**交付物**:
- 历史数据集（CSV/Parquet 格式）
- ✅ 特征计算代码库 ([feature_engineering.py](src/data/feature_engineering.py))
- ✅ 标签生成脚本 ([labels.py](src/utils/labels.py))
- ✅ 测试套件 (24 个测试，全部通过)
- 数据探索分析报告

### 6.2 Phase 2: 模型训练与验证
**目标**: 训练预测模型，验证在历史 spike 事件上的性能

- [ ] Baseline 模型开发
  - 训练 XGBoost/LightGBM
  - 特征重要性分析
  - 超参数调优
  - 验证集性能评估

- [ ] 高级模型开发（可选）
  - 实现 CfC/LTC 架构
  - 多任务学习训练
  - 模型集成

- [ ] 模型评估
  - 在测试集上评估（AUC, PR-AUC, Event Recall）
  - 12/14-15 事件预测分析
  - 假阳/假阴案例分析
  - 模型可解释性分析

**交付物**:
- 训练好的模型文件
- 模型性能评估报告
- 特征重要性排序
- 错误案例分析

### 6.3 Phase 3: 策略回测
**目标**: 实现 BESS 策略并在历史数据上回测，量化收益提升

- [ ] 策略实现
  - 实现三大硬规则（Rule A/B/C）
  - 实现 MPC 优化框架（可选，可先用规则）
  - 集成预测模型与策略决策

- [ ] 回测框架
  - 实现回测引擎
  - 定义评估指标（收益、SOC 利用率、风险指标）
  - 策略参数敏感性分析

- [ ] 对比分析
  - Baseline 策略：传统谷充峰放
  - 新策略：预测驱动 + Hold-SOC
  - 在历史 spike 事件上对比收益

**交付物**:
- 策略代码库
- 回测引擎
- 历史收益对比报告
- 策略优化建议

### 6.4 Phase 4: 12/14-15 Case Study
**目标**: 针对 12/14-15 事件做深度案例分析，展示 PowerA 技术能力

- [ ] 事件复现
  - 重现 12/14-15 价格曲线与市场状态
  - 可视化关键特征演变
  - 逐时段分析系统状态切换

- [ ] 预测能力展示
  - 模型在 12/14 16:00 时的预警概率
  - Regime 状态识别时间线
  - 与实际 spike 发生时间对比

- [ ] 策略对比
  - 实际储能行为复现（16:30-20:00 提前放电）
  - PowerA 策略模拟（Hold-SOC 至 20:00-22:00）
  - 收益差异量化（预计提升 30-60%）

- [ ] 可视化报告
  - 时间轴可视化（价格、特征、预测、策略）
  - 收益对比图表
  - 关键决策点标注

**交付物**:
- 12/14-15 Case Study 报告（PDF/PPT）
- 可视化 Dashboard（Jupyter Notebook / Streamlit）
- 演示视频/动画
- 技术白皮书

### 6.5 Phase 5: 实时系统规划（后续）
**目标**: 设计实时数据流与在线预测系统架构

- [ ] 实时数据架构
  - 实时数据接入方案（ERCOT API/WebSocket）
  - 时序数据库选型与部署
  - 数据流处理（Kafka/Spark Streaming）

- [ ] 模型服务化
  - 模型部署方案（FastAPI/TorchServe）
  - 实时特征计算优化
  - 低延迟推理（<1s）

- [ ] 监控与告警
  - 预测监控 Dashboard
  - Spike 预警通知系统
  - 模型性能监控

**交付物**:
- 实时系统架构设计文档
- 技术选型报告
- POC 原型（可选）

### 6.6 Phase 6: 生产上线（未来）
**目标**: 实际部署与运营

- [ ] 影子运行
- [ ] 灰度发布
- [ ] 全量上线
- [ ] 持续优化

---

## 附录

### A. 技术栈推荐

**数据层**:
- 时序数据库: InfluxDB 2.x / TimescaleDB
- Feature Store: Feast / Tecton
- 数据质量: Great Expectations

**模型层**:
- Baseline: XGBoost, LightGBM
- Deep Learning: PyTorch + ncps (CfC/LTC)
- 实验跟踪: MLflow / Weights & Biases

**策略层**:
- 优化器: scipy.optimize / cvxpy (凸优化)
- 回测: Backtrader / 自研

**部署层**:
- 推理服务: FastAPI + uvicorn
- 容器化: Docker
- 编排: Kubernetes (可选)
- 监控: Prometheus + Grafana

### B. 关键文件结构
```
spike-forecast/
├── data/
│   ├── raw/              # 原始数据
│   ├── processed/        # 处理后数据
│   └── features/         # 特征数据
├── src/
│   ├── data/
│   │   ├── ingestion.py      # 数据采集
│   │   ├── preprocessing.py  # 预处理
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── baseline.py       # XGBoost/LightGBM
│   │   ├── cfc_model.py      # CfC/LTC
│   │   └── ensemble.py       # 集成
│   ├── strategy/
│   │   ├── rules.py          # 硬规则
│   │   ├── mpc.py            # MPC 优化
│   │   └── backtest.py       # 回测框架
│   └── utils/
│       ├── labels.py         # 标签生成
│       └── metrics.py        # 评估指标
├── notebooks/
│   ├── 01_eda.ipynb          # 探索性分析
│   ├── 02_feature_analysis.ipynb
│   └── 03_model_evaluation.ipynb
├── configs/
│   ├── data_config.yaml
│   ├── model_config.yaml
│   └── strategy_config.yaml
├── tests/
├── docs/
│   └── DESIGN.md (本文档)
├── requirements.txt
└── README.md
```

---

## 进度日志

### 2025-12-29
**完成**: Phase 1 - 特征工程实现

- ✅ 创建项目结构
- ✅ 实现特征工程模块 ([feature_engineering.py](src/data/feature_engineering.py))
  - 价格结构特征 (6 个特征 × 3 zones)
  - 供需平衡特征 (15 个特征)
  - 天气驱动特征 (4 个特征 × 3 zones)
  - 时间特征 (8 个特征)
  - 总计: ~50 个特征
- ✅ 实现标签生成模块 ([labels.py](src/utils/labels.py))
  - SpikeEvent 标签生成
  - LeadSpike 预警标签 (60分钟提前)
  - Regime 状态分类 (Normal/Tight/Scarcity)
  - 支持固定阈值和滚动分位数两种模式
- ✅ 创建测试套件
  - test_feature_engineering.py: 13 个测试
  - test_labels.py: 11 个测试
  - 所有测试通过 ✓
- ✅ 创建示例脚本 ([01_feature_engineering_example.py](notebooks/01_feature_engineering_example.py))
- ✅ 创建项目文档 (README.md, requirements.txt)

**下一步**: 等待数据下载完成，进行特征分析和可视化

---

**文档版本**: v1.1
**最后更新**: 2025-12-29
