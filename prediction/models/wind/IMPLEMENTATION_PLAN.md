# ERCOT 风电预测系统实现计划

## 概述

基于 HRRR 天气预报数据，为 ERCOT/Texas 构建 0-12 小时风电发电量预测系统。

**核心目标：**
- 系统级风电发电量预测 (MW)
- 分位数预测 (p10/p50/p90) 用于不确定性量化
- 风电爬坡检测和预警

> **重点场景：风电下降 + 无光伏时段**
>
> 当风电快速下降发生在日落后/日出前（无光伏），系统需要依靠 Gas 发电，
> 价格容易飙升。这是最需要提前预警的场景。

---

## 项目结构

```
trueflux/wind-generation-forecast/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── hrrr_client.py        # HRRR 数据获取 (Earth2Studio)
│   │   ├── ercot_wind_client.py  # ERCOT 风电数据获取
│   │   └── texas_regions.py      # Texas 区域定义
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── wind_features.py      # 风速/功率特征
│   │   ├── ramp_features.py      # 爬坡检测特征
│   │   └── temporal_features.py  # 时间特征
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py               # 模型基类接口
│   │   ├── gbm_model.py          # LightGBM 基线
│   │   ├── lstm_model.py         # LSTM 序列模型
│   │   └── ensemble.py           # 集成模型
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py            # MAE, RMSE, 技能分数
│   │   └── ramp_metrics.py       # 爬坡检测指标 (POD, FAR, CSI)
│   │
│   └── utils/
│       ├── __init__.py
│       └── config.py             # 配置管理
│
├── scripts/
│   ├── fetch_hrrr_data.py        # 下载 HRRR 历史数据
│   ├── build_features.py         # 构建训练特征
│   └── train_models.py           # 训练模型
│
├── configs/
│   └── default.yaml              # 配置文件
│
├── notebooks/
│   └── exploration.ipynb         # 数据探索
│
└── requirements.txt
```

---

## 实现步骤

### Phase 1: 数据层 (src/data/)

**1.1 hrrr_client.py - HRRR 数据获取**

```python
# 关键功能
class HRRRWindClient:
    """通过 Earth2Studio 获取 HRRR 预报数据"""

    WIND_VARIABLES = ['u10m', 'v10m', 'u80m', 'v80m', 't2m', 'sp']

    def fetch_forecast(
        self,
        init_time: datetime,
        lead_times: List[int],  # [0, 1, 2, ..., 12] 小时
    ) -> xr.DataArray:
        """获取 Texas 区域 HRRR 预报"""

    def subset_texas(self, data: xr.DataArray) -> xr.DataArray:
        """提取 Texas 子区域"""
```

**1.2 ercot_wind_client.py - ERCOT 风电数据**

复用 `ercot-scraper` 模式，获取历史风电发电量用于训练。

**1.3 texas_regions.py - 区域定义**

```python
ERCOT_WIND_REGIONS = {
    'PANHANDLE': {'lat': (34.0, 36.5), 'lon': (-103.0, -100.0)},
    'WEST': {'lat': (30.5, 34.0), 'lon': (-104.0, -100.5)},
    'COASTAL': {'lat': (26.5, 30.0), 'lon': (-97.5, -95.0)},
}
```

---

### Phase 2: 特征工程 (src/features/)

**2.1 wind_features.py - 风电特征**

参考 `spike-forecast/src/data/feature_engineering.py` 模式：

```python
class WindFeatureEngineer:
    """风电特征计算"""

    @staticmethod
    def compute_wind_speed(u, v) -> np.ndarray:
        """从 U/V 分量计算风速"""
        return np.sqrt(u**2 + v**2)

    @staticmethod
    def compute_power_curve(wind_speed, cut_in=3.0, rated=12.0, cut_out=25.0):
        """应用风机功率曲线"""

    @staticmethod
    def compute_wind_shear(ws_10m, ws_80m):
        """计算风切变指数"""
```

**输出特征：**
- `ws_80m_mean`: 80m平均风速
- `ws_80m_std`: 风速空间变异
- `power_density`: 风功率密度
- `normalized_power`: 功率曲线归一化输出

**2.2 ramp_features.py - 爬坡特征**

> **重点：Ramp-Down + 无光伏组合风险**

```python
class RampFeatureEngineer:
    """爬坡检测特征 - 重点关注 ramp-down"""

    def compute_wind_change_rate(self, wind_speed, time_hours):
        """风速变化率 (m/s/h)"""

    def compute_power_sensitivity(self, wind_speed):
        """功率曲线敏感区间 (3-12 m/s 最敏感)"""

    def compute_frontal_indicator(self, temp_change, wind_dir_change):
        """天气锋面指示器"""

    def compute_ramp_down_risk(
        self,
        wind_change: float,      # 预测风电变化 (MW)
        current_hour: int,       # 当前小时 (0-23)
    ) -> float:
        """
        计算风电下降风险分数 (0-1)

        高风险组合:
        1. 风电预报下降 > 2000 MW
        2. 无光伏时段 (18:00 - 07:00)
        3. 傍晚需求高峰 (17:00 - 21:00)
        """
        risk = 0.0
        is_no_solar = (current_hour >= 18) or (current_hour < 7)
        is_evening_peak = 17 <= current_hour <= 21

        if wind_change < -1000: risk += 0.2
        if wind_change < -2000: risk += 0.2
        if wind_change < -3000: risk += 0.2
        if is_no_solar: risk += 0.2
        if is_evening_peak and is_no_solar: risk += 0.2  # 最危险组合

        return min(risk, 1.0)
```

**输出特征：**
- `ws_change_1h`, `ws_change_3h`: 风速变化
- `ramp_down_1h`, `ramp_down_3h`: 风电下降量 (MW, 负值表示下降)
- `power_sensitivity`: 功率曲线敏感度
- `frontal_indicator`: 锋面概率
- **`is_no_solar_period`**: 是否无光伏时段 (18:00-07:00)
- **`is_evening_peak`**: 是否傍晚高峰 (17:00-21:00)
- **`ramp_down_no_solar_risk`**: 组合风险分数 (0-1)
- **`minutes_to_sunset`**: 距日落分钟数
- **`minutes_since_sunset`**: 日落后分钟数

**2.3 temporal_features.py - 时间特征**

```python
class TemporalFeatureEngineer:
    """时间特征"""

    @staticmethod
    def encode_cyclical(value, period):
        """周期编码 (sin/cos)"""
        return np.sin(2*np.pi*value/period), np.cos(2*np.pi*value/period)
```

**输出特征：**
- `hour_sin`, `hour_cos`: 小时周期
- `doy_sin`, `doy_cos`: 年内日期周期

---

### Phase 3: 模型层 (src/models/)

**3.1 base.py - 模型接口**

```python
class BaseWindForecastModel(ABC):
    """风电预测模型基类"""

    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """训练模型"""

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """点预测 (p50)"""

    @abstractmethod
    def predict_quantiles(self, X, quantiles=[0.1, 0.5, 0.9]):
        """分位数预测"""

    def predict_ramp(self, X, current_gen, threshold=2000):
        """爬坡预测"""
```

**3.2 gbm_model.py - LightGBM 基线**

参考 `RTM_LMP_Price_Forecast/src/rtm_short_term_forecast.py`：

```python
class GBMWindModel(BaseWindForecastModel):
    """LightGBM 分位数回归"""

    def __init__(self, quantiles=[0.1, 0.5, 0.9], use_gpu=True):
        self.models = {}  # 每个分位数一个模型

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

**3.3 lstm_model.py - LSTM 模型**

参考 `Load-_forecast/Load_Forecast/src/lstm_forecast.py`：

```python
class LSTMWindModel(nn.Module):
    """LSTM 序列模型"""

    def __init__(self, n_features, hidden_dim=256, num_layers=3):
        self.lstm = nn.LSTM(n_features, hidden_dim, num_layers)
        self.quantile_heads = nn.ModuleList([...])  # 多头输出分位数
```

**3.4 ensemble.py - 集成模型**

```python
class EnsembleWindModel(BaseWindForecastModel):
    """模型集成"""

    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights or [1/len(models)] * len(models)

    def predict_quantiles(self, X, quantiles=None):
        """加权平均各模型分位数"""
```

---

### Phase 4: 评估 (src/evaluation/)

**4.1 metrics.py - 标准指标**

```python
def mae(y_true, y_pred): ...
def rmse(y_true, y_pred): ...
def nmae(y_true, y_pred, capacity): ...  # 归一化 MAE
def skill_score(y_true, y_pred, y_baseline): ...
```

**4.2 ramp_metrics.py - 爬坡指标**

> **重点评估 Ramp-Down 检测能力**

```python
def detect_ramps(values, threshold, window, direction='both'):
    """
    检测爬坡事件

    direction: 'both', 'up', 'down'
    """

def compute_ramp_metrics(y_true, y_pred, threshold=2000, window=12):
    """计算爬坡检测指标"""
    # POD: 命中率 = hits / (hits + misses)
    # FAR: 虚警率 = false_alarms / (hits + false_alarms)
    # CSI: 关键成功指数 = hits / (hits + misses + false_alarms)

def evaluate_ramp_down_in_no_solar(
    y_true, y_pred, timestamps, threshold=-2000
):
    """
    专门评估无光伏时段的风电下降检测

    这是最关键的场景:
    - 仅评估 18:00-07:00 时段
    - 仅评估下降事件 (change < threshold)
    - 计算提前预警时间 (lead time)
    """
```

**重点评估指标：**
- **Ramp-Down POD**: 风电下降事件命中率 (目标 > 0.8)
- **No-Solar POD**: 无光伏时段下降事件命中率 (最关键)
- **Lead Time**: 平均提前预警时间 (目标 > 2小时)
- **Miss Rate**: 漏报率 (必须 < 0.2)

---

## 依赖项 (requirements.txt)

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

## 验证方案

1. **数据层验证：**
   ```bash
   python scripts/fetch_hrrr_data.py --date 2025-01-20 --hours 12
   # 验证输出 xarray 形状和变量
   ```

2. **特征验证：**
   ```bash
   python scripts/build_features.py --start 2024-01-01 --end 2024-12-31
   # 检查特征 DataFrame 无 NaN/Inf
   ```

3. **模型训练验证：**
   ```bash
   python scripts/train_models.py
   # 输出 MAE, RMSE, CSI 指标
   ```

4. **爬坡检测验证：**
   - 在历史大爬坡事件 (>3000 MW/h) 上测试 POD
   - 目标: POD > 0.7, FAR < 0.4

---

## 实现优先级

| 优先级 | 组件 | 说明 |
|-------|------|-----|
| P0 | hrrr_client.py | 数据获取基础 |
| P0 | wind_features.py | 核心特征 |
| P0 | gbm_model.py | 基线模型 |
| P1 | ramp_features.py | 爬坡特征 |
| P1 | lstm_model.py | 序列模型 |
| P1 | ramp_metrics.py | 爬坡评估 |
| P2 | ensemble.py | 模型集成 |
| P2 | 实时推理 | 生产部署 |

---

## 关键参考文件

1. `/home/lanxin/projects/weather-forecast/earth2studio/earth2studio/data/hrrr.py` - HRRR 数据获取模式
2. `/home/lanxin/projects/trueflux/spike-forecast/src/data/feature_engineering.py` - 特征工程模式
3. `/home/lanxin/projects/trueflux/Load-_forecast/Load_Forecast/src/lstm_forecast.py` - LSTM 实现
4. `/home/lanxin/projects/trueflux/RTM_LMP_Price_Forecast/src/rtm_short_term_forecast.py` - 多模型训练
