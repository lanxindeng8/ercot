# ERCOT RTM LMP Spike Forecasting

ERCOT 实时市场 LMP Spike 预测算法与储能策略优化系统

## 项目概述

本项目旨在预测 ERCOT 实时市场（RTM）中的区域性 LMP Spike 事件，并优化电池储能系统（BESS）的调度策略，避免"提前放电"错误，最大化在极值窗口的收益。

**核心能力**：
- Zone-level Spike 事件预测（提前 60-90 分钟）
- 系统状态识别（Normal → Tight → Scarcity）
- Hold-SOC 策略优化
- 预计收益提升：30-60%

## 项目结构

```
spike-forecast/
├── data/
│   ├── raw/              # 原始数据
│   ├── processed/        # 处理后数据
│   └── features/         # 特征数据
├── src/
│   ├── data/
│   │   ├── feature_engineering.py  # 特征计算模块
│   │   ├── ingestion.py            # 数据采集（待实现）
│   │   └── preprocessing.py        # 预处理（待实现）
│   ├── models/
│   │   ├── baseline.py             # XGBoost/LightGBM（待实现）
│   │   └── cfc_model.py            # CfC/LTC（待实现）
│   ├── strategy/
│   │   ├── rules.py                # 硬规则（待实现）
│   │   ├── mpc.py                  # MPC 优化（待实现）
│   │   └── backtest.py             # 回测框架（待实现）
│   └── utils/
│       ├── labels.py               # 标签生成模块
│       └── metrics.py              # 评估指标（待实现）
├── notebooks/
│   └── 01_feature_engineering_example.py  # 特征工程示例
├── configs/              # 配置文件
├── tests/                # 测试
├── docs/
│   ├── DESIGN.md         # 算法设计文档
│   └── ...               # 参考文档
├── requirements.txt
└── README.md
```

## 快速开始

### 1. 环境设置

```bash
# 克隆仓库
git clone https://github.com/powerA-ai/spike-forecast.git
cd spike-forecast

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行特征工程示例

```bash
cd notebooks
python 01_feature_engineering_example.py
```

这将：
- 创建示例数据（模拟 7 天的 ERCOT 市场数据）
- 计算所有特征（~40-50 维）
- 生成标签（SpikeEvent, LeadSpike, Regime）
- 保存处理后的数据

### 3. 查看设计文档

详细的算法设计文档位于 [DESIGN.md](DESIGN.md)，包含：
- 系统架构
- 特征工程详细说明
- 预测模型设计
- 策略优化框架
- 实施路线

## 特征工程

### 特征分组（~40-50 维）

#### 1. 价格结构特征（Price Structure）
捕捉区域性稀缺与拥塞信号
- 区域-系统价差（`spread_zone_hub`）
- 实时-日前溢价（`spread_rt_da`）
- 价格斜率和加速度

#### 2. 供需平衡特征（Supply-Demand Balance）
捕捉系统/区域紧张状态
- 净负荷及其爬坡、加速度
- 风电异常（`wind_anomaly`）
- 气电饱和度（`gas_saturation`）
- 煤电夜间上行（`coal_stress`）
- 储能净出力

#### 3. 天气驱动特征（Weather-Driven, Zone-level）
捕捉需求侧冲击信号
- 温度异常（`T_anomaly_zone`）
- 降温速度（`T_ramp_zone`）
- 风寒指数（`WindChill_zone`）
- 冷锋标志（`ColdFront_zone`）

#### 4. 时间特征（Temporal）
捕捉日内模式与光伏修复窗口
- 小时、星期、月份
- 晚高峰标志
- 距日出时间（`minutes_to_sunrise`）
- 光伏爬坡预期

### 使用示例

```python
from src.data.feature_engineering import FeatureEngineer

# 初始化
feature_engineer = FeatureEngineer(
    zones=['CPS', 'West', 'Houston'],
    lookback_days=30
)

# 计算所有特征
df_with_features = feature_engineer.calculate_all_features(raw_data)

# 获取特征名称
price_features = feature_engineer.get_feature_names('price')
```

## 标签生成

### 标签类型

1. **SpikeEvent**：Spike 事件标识（0/1）
2. **LeadSpike**：提前预警标签（0/1，提前 60 分钟）
3. **Regime**：系统状态（Normal / Tight / Scarcity）

### 使用示例

```python
from src.utils.labels import LabelGenerator

# 初始化
label_generator = LabelGenerator(
    zones=['CPS', 'West', 'Houston'],
    P_hi=400,      # Spike 价格阈值
    S_hi=50,       # Spike 价差阈值
    H=60,          # Lead Spike 预警窗口（分钟）
)

# 生成所有标签
labels = label_generator.generate_all_labels(df)

# 识别独立 Spike 事件
events = label_generator.identify_spike_events(labels['SpikeEvent_CPS'])
```

## 数据要求

### 必需数据

#### 市场数据（5分钟/15分钟）
- RT LMP（实时市场价格）：P_CPS, P_West, P_Houston, P_Hub
- DA LMP（日前市场价格）：P_CPS_DA, P_West_DA, P_Houston_DA

#### 系统数据（5分钟）
- Load（负荷）
- Wind（风电出力）
- Solar（光伏出力）
- Gas（天然气出力）
- Coal（煤电出力）
- ESR（储能净出力）

#### 天气数据（15分钟，Zone-level）
- T_{zone}（温度）
- WindSpeed_{zone}（风速）
- WindDir_{zone}（风向）

### 数据来源

- **ERCOT 数据**：http://www.ercot.com/
  - Real-Time Market (RTM) prices
  - Day-Ahead Market (DAM) prices
  - Fuel Mix data

- **天气数据**：NOAA / 气象 API
  - San Antonio (CPS)
  - West Texas
  - Houston

## 实施路线

详见 [DESIGN.md](DESIGN.md) 第 6 节。

**核心策略**：
1. **Phase 1**: 历史数据准备（数据收集、特征工程、标签生成）
2. **Phase 2**: 模型训练与验证（XGBoost baseline + CfC 高级模型）
3. **Phase 3**: 策略回测（硬规则 + MPC 优化）
4. **Phase 4**: 12/14-15 Case Study（展示 PowerA 技术能力）
5. **Phase 5-6**: 实时系统规划与上线（后续）

## 关键成果

### 12/14-15 Case Study 目标

针对 2025-12-14 晚高峰 Spike 事件的案例分析：

**问题**：
- 储能在 16:30-20:00 提前放电（$100-$325 区间）
- 错过 20:00-22:00 极值窗口（$400-$686）

**PowerA 解决方案**：
- 提前 60 分钟预警（16:00 时预测 20:00-22:00 spike）
- Hold-SOC 策略（在 Tight 状态限制放电）
- 在极值窗口释放 SOC

**预期收益提升**：30-60%

## 技术栈

- **数据**：Pandas, NumPy
- **ML Baseline**：XGBoost, LightGBM, Scikit-learn
- **深度学习**：PyTorch, ncps (CfC/LTC)
- **优化**：SciPy, CVXPY
- **可视化**：Matplotlib, Seaborn, Plotly

## 贡献

本项目由 PowerA AI Team 开发。

## 许可

Private Repository - PowerA AI

## 联系方式

- **Organization**: powerA-ai
- **Repository**: https://github.com/powerA-ai/spike-forecast

---

**Last Updated**: 2025-12-29
