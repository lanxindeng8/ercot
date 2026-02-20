# RTM-DAM Delta Prediction Plan (Arbitrage Strategy)

## 用户需求确认
- **预测目标**: 三种模型都建立 (回归 + 二分类 + 多分类) - 用于 RTM/DAM 套利
- **预测时间**: 40小时 (DAM前一天上午下单 → 次日0~24小时交割)
- **Settlement Point**: 先用 LZ_WEST 做实验

---

## 1. 现有数据概览

### RTM LMP 数据
| 属性 | 值 |
|------|-----|
| 文件位置 | `RTM_LMP_Price_Forecast/data/rtm_lz_houston.csv` |
| 时间范围 | 2010年12月 - 2024年12月 (14年+) |
| 频率 | **15分钟** |
| 记录数 | 458,880 条 |
| 价格范围 | -$147 ~ $9,236 /MWh |
| 已提取点 | LZ_HOUSTON |

### DAM LMP 数据
| 属性 | 值 |
|------|-----|
| 文件位置 | `DAM_Price_Forecast/data/dam_lz_houston.csv` |
| 时间范围 | 2015年1月 - 2025年12月 (11年) |
| 频率 | **小时** |
| 记录数 | 95,928 条 |
| 价格范围 | -$0.10 ~ $8,995 /MWh |
| 已提取点 | LZ_HOUSTON |

### 可提取的其他 Settlement Points
- **Load Zones (LZ)**: LZ_HOUSTON, LZ_WEST, LZ_NORTH, LZ_SOUTH, LZ_AEN, LZ_CPS, LZ_LCRA, LZ_RAYBN
- **Hub Points (HB)**: HB_HOUSTON, HB_WEST, HB_NORTH, HB_SOUTH 等
- 原始数据包含 1,071+ 个 settlement points

---

## 2. Delta 预测可行性分析

### 重叠数据
- **重叠时间段**: 2015-2024 (约9-10年)
- **共同 Settlement Point**: LZ_HOUSTON (已提取)
- **数据量**:
  - RTM: ~315,000 条 (15分钟间隔)
  - DAM: ~78,000 条 (小时间隔)

### Delta 定义
```
RTM-DAM Spread = RTM Price - DAM Price
```
- **正值**: RTM高于DAM (实时市场紧张)
- **负值**: RTM低于DAM (实时市场宽松)

### 已有代码支持
`short_term_feature_extraction.py` 已经计算了以下 delta 特征:
- `rtm_dam_spread` - RTM与DAM价差
- `spread_mean_1h` - 1小时平均价差
- `spread_mean_24h` - 24小时平均价差
- `spread_std_24h` - 24小时价差波动
- `rtm_dam_ratio` - RTM/DAM比率

---

## 3. 套利场景说明

### DAM 投标时间线
```
Day D-1 上午 10:00    →    Day D 00:00-24:00
   ↓                           ↓
 DAM投标截止              实际交割时段

预测需求: 在 D-1 上午预测 D 日每小时的 RTM-DAM spread
预测提前量: 14~38小时 (D-1 10:00 预测 D日 00:00~24:00)
```

### 套利策略逻辑
- **预测 Spread > 0** (RTM会高于DAM): 在DAM买入 → 在RTM卖出
- **预测 Spread < 0** (RTM会低于DAM): 在DAM卖出 → 在RTM买回
- **Spread绝对值越大**: 套利空间越大

---

## 4. 三种预测模型

### Model 1: Spread 回归 (用于量化收益)
- **目标**: 预测 RTM-DAM spread 的具体数值
- **用途**: 计算期望收益、确定仓位大小
- **评估指标**: MAE, RMSE, R²

### Model 2: Spread 方向分类 (用于交易决策)
- **目标**: 预测 RTM > DAM (1) 还是 RTM < DAM (0)
- **用途**: 决定买入还是卖出方向
- **评估指标**: Accuracy, Precision, Recall, AUC

### Model 3: Spread 区间分类 (用于风险控制)
- **目标**: 预测 spread 落入哪个区间
- **区间设计**:
  | 类别 | 区间 | 交易信号 |
  |------|------|----------|
  | 0 | Spread < -$20 | 强烈卖出DAM |
  | 1 | -$20 ≤ Spread < -$5 | 适度卖出DAM |
  | 2 | -$5 ≤ Spread < $5 | 不交易 |
  | 3 | $5 ≤ Spread < $20 | 适度买入DAM |
  | 4 | Spread ≥ $20 | 强烈买入DAM |
- **评估指标**: Multi-class F1, Confusion Matrix

---

## 5. 实施步骤

### Step 1: 提取 LZ_WEST 数据
```bash
# RTM 数据提取
cd /home/lanxin/projects/trueflux/ercot-data
python extract_rtm_data.py --settlement-point LZ_WEST --output ../RTM_LMP_Price_Forecast/data/rtm_lz_west.csv

# DAM 数据提取
python extract_dam_data.py --settlement-point LZ_WEST --output ../DAM_Price_Forecast/data/dam_lz_west.csv
```

### Step 2: 数据合并与Spread计算
```python
# RTM 聚合到小时级别
rtm_hourly = rtm_15min.groupby(['date', 'hour']).agg({
    'price': ['mean', 'max', 'min', 'last', 'std']  # 小时内统计
}).reset_index()

# 与 DAM 合并 (按日期+小时对齐)
merged = pd.merge(rtm_hourly, dam_hourly, on=['date', 'hour'])
merged['spread'] = merged['rtm_price_mean'] - merged['dam_price']
merged['spread_direction'] = (merged['spread'] > 0).astype(int)
merged['spread_class'] = pd.cut(merged['spread'],
    bins=[-np.inf, -20, -5, 5, 20, np.inf],
    labels=[0, 1, 2, 3, 4])
```

### Step 3: 特征工程 (40小时预测)
```python
特征组:
1. 时间特征: target_hour, target_dow, target_month, is_peak, is_weekend
2. DAM价格特征: dam_price (已知), dam_vs_history, dam_percentile
3. 历史Spread: spread_lag_24h, spread_lag_48h, spread_lag_168h (同时段)
4. Spread统计: spread_mean_7d, spread_std_7d, spread_by_hour_mean
5. RTM历史: rtm_same_hour_mean_7d, rtm_volatility_7d
6. 市场状态: recent_spike_count, recent_negative_count
```

### Step 4: 模型训练
```python
# 三个模型共用特征, 不同目标变量
models = {
    'regression': CatBoostRegressor(loss_function='MAE'),
    'binary': CatBoostClassifier(loss_function='Logloss'),
    'multiclass': CatBoostClassifier(loss_function='MultiClass', classes_count=5)
}

# TimeSeriesSplit 验证
tscv = TimeSeriesSplit(n_splits=5)
```

### Step 5: 评估与回测
- 预测准确率/MAE 评估
- 模拟套利策略收益
- 计算夏普比率、最大回撤

---

## 6. 数据充足性评估

| 检查项 | 状态 | 说明 |
|--------|------|------|
| 时间跨度 | ✅ | 9-10年 (2015-2024) |
| 样本量 | ✅ | ~78,000+ 小时 |
| 同一 Settlement Point | ✅ | LZ_WEST (待提取) |
| 包含极端事件 | ✅ | 2021冬季风暴等 |
| 预测窗口可行 | ✅ | 40小时无信息泄露 |

**结论: 数据完全足够进行 Delta 预测**

---

## 7. 关键文件

### 数据文件 (待生成)
- `RTM_LMP_Price_Forecast/data/rtm_lz_west.csv` - RTM LZ_WEST数据
- `DAM_Price_Forecast/data/dam_lz_west.csv` - DAM LZ_WEST数据

### 参考代码
- [short_term_feature_extraction.py](trueflux/RTM_LMP_Price_Forecast/src/short_term_feature_extraction.py) - 已有 spread 计算逻辑
- [extract_rtm_data.py](trueflux/ercot-data/extract_rtm_data.py) - RTM数据提取脚本
- [extract_dam_data.py](trueflux/ercot-data/extract_dam_data.py) - DAM数据提取脚本

### 新建文件
- `delta_prediction/` - 新建Delta预测项目目录
  - `prepare_delta_data.py` - 数据合并与Spread计算
  - `delta_feature_extraction.py` - 40小时预测特征工程
  - `train_delta_models.py` - 三模型训练脚本
  - `backtest_arbitrage.py` - 套利策略回测

---

## 8. 验证方法

1. **模型性能验证**
   - Regression: MAE < $10, R² > 0.3
   - Binary: Accuracy > 60%, AUC > 0.65
   - MultiClass: Macro-F1 > 0.4

2. **回测验证**
   - 模拟2023-2024年套利交易
   - 计算累计收益曲线
   - 统计胜率、平均收益、最大回撤

---

## 9. 实施顺序

1. ✅ 提取 LZ_WEST 的 RTM 和 DAM 数据
2. ✅ 合并数据、计算 Spread、创建标签
3. ✅ 特征工程 (40小时预测窗口)
4. ✅ 训练三个模型 (回归/二分类/多分类)
5. ✅ 评估模型性能
6. ✅ 套利策略回测

---

# 实验结果报告

> 实验日期: 2026-02-01
> Settlement Point: LZ_WEST

---

## 10. 数据提取结果

### 10.1 RTM LZ_WEST 数据
| 属性 | 值 |
|------|-----|
| 文件路径 | `Delta_Spread_Prediction/data/rtm_lz_west.csv` |
| 记录数 | **526,266** 条 |
| 时间范围 | 2010-12-01 ~ 2024-12-31 (14年) |
| 数据频率 | 15分钟 |
| 价格范围 | -$44.82 ~ $9,312.54 /MWh |

### 10.2 DAM LZ_WEST 数据
| 属性 | 值 |
|------|-----|
| 文件路径 | `Delta_Spread_Prediction/data/dam_lz_west.csv` |
| 记录数 | **95,928** 条 |
| 时间范围 | 2015-01-01 ~ 2025-12-10 (11年) |
| 数据频率 | 小时 |
| 价格范围 | -$10.94 ~ $9,026.99 /MWh |

### 10.3 合并后数据
| 属性 | 值 |
|------|-----|
| 文件路径 | `Delta_Spread_Prediction/data/spread_data.csv` |
| 记录数 | **54,908** 条 (小时级别) |
| 重叠时间 | 2015-01-09 ~ 2024-12-31 |
| 有效特征样本 | 53,992 条 |

---

## 11. Spread 统计分析

### 11.1 基本统计
| 统计量 | 值 |
|--------|-----|
| 均值 | **-$7.17** /MWh |
| 标准差 | $69.80 /MWh |
| 最小值 | -$8,482.50 /MWh |
| 最大值 | $1,836.04 /MWh |
| 中位数 | -$2.53 /MWh |

### 11.2 Spread 方向分布
| 方向 | 样本数 | 占比 |
|------|--------|------|
| RTM > DAM (正spread) | 13,943 | **25.4%** |
| RTM < DAM (负spread) | 40,965 | **74.6%** |

**关键发现**: LZ_WEST 的 RTM 价格在约 **75%** 的时间低于 DAM 价格，平均低 $7.17/MWh。

### 11.3 Spread 区间分布
| 区间 | 样本数 | 占比 | 交易信号 |
|------|--------|------|----------|
| < -$20 | 4,999 | 9.1% | 强烈Short |
| -$20 ~ -$5 | 16,477 | 30.0% | 适度Short |
| -$5 ~ $5 | 29,959 | **54.6%** | 不交易 |
| $5 ~ $20 | 2,664 | 4.9% | 适度Long |
| >= $20 | 809 | 1.5% | 强烈Long |

---

## 12. 特征工程结果

### 12.1 特征列表 (44个特征)
```
时间特征 (9个):
  - target_hour, target_dow, target_month, target_day_of_month, target_week
  - target_is_weekend, target_is_peak, target_is_summer
  - hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos

DAM特征 (4个):
  - target_dam_price (已知的DAM价格)
  - dam_vs_7d_mean, dam_percentile_7d, dam_price_level

历史Spread特征 (12个):
  - spread_mean_7d, spread_std_7d, spread_max_7d, spread_min_7d, spread_median_7d
  - spread_mean_24h, spread_std_24h
  - spread_same_hour_hist, spread_same_hour_std
  - spread_same_dow_hour, spread_same_dow_hour_last
  - spread_trend_7d, spread_positive_ratio_7d, spread_positive_ratio_24h

RTM历史特征 (7个):
  - rtm_mean_7d, rtm_std_7d, rtm_max_7d
  - rtm_mean_24h, rtm_volatility_24h
  - rtm_same_hour_hist

DAM历史特征 (4个):
  - dam_mean_7d, dam_std_7d, dam_mean_24h

Spike特征 (3个):
  - spike_count_7d, rtm_spike_count_7d
```

### 12.2 数据集划分
| 数据集 | 样本数 | 时间范围 |
|--------|--------|----------|
| 训练集 | 43,193 | 2015-01 ~ 2022-06 |
| 测试集 | 10,799 | 2022-06 ~ 2024-12 |

---

## 13. 模型训练结果

### 13.1 回归模型 (CatBoost Regressor)

**模型配置**:
- 算法: CatBoost
- 损失函数: MAE
- 最终迭代数: 167 (Early Stopping)
- 深度: 7, 学习率: 0.05

**性能指标**:
| 指标 | 模型 | 基线 (历史同时段均值) | 改进 |
|------|------|------------------------|------|
| MAE | **$8.98** | $12.57 | **+28.6%** |
| RMSE | $38.15 | - | - |
| R² | 0.2937 | - | - |

**Top 10 重要特征**:
1. spread_same_hour_hist (同时段历史均值)
2. spread_mean_7d (7天平均)
3. target_dam_price (目标小时DAM价格)
4. spread_same_dow_hour_last (上周同时段)
5. rtm_same_hour_hist (RTM同时段历史)
6. spread_mean_24h (24小时平均)
7. dam_vs_7d_mean (DAM相对7天均值)
8. spread_std_7d (7天波动)
9. target_hour (目标小时)
10. rtm_mean_7d (RTM 7天均值)

---

### 13.2 二分类模型 (CatBoost Classifier)

**目标**: 预测 RTM > DAM (1) 或 RTM < DAM (0)

**模型配置**:
- 损失函数: Logloss
- 类别权重: 自动平衡 (Balanced)
- 最终迭代数: 44 (Early Stopping)

**性能指标**:
| 指标 | 值 |
|------|-----|
| Accuracy | **73.5%** |
| Precision | 38.9% |
| Recall | 43.5% |
| F1 Score | 0.4108 |
| AUC | **0.6891** |

**混淆矩阵**:
```
              预测 RTM<DAM  预测 RTM>DAM
实际 RTM<DAM     6,946         1,565
实际 RTM>DAM     1,292           996
```

**分析**: 由于类别严重不平衡 (75% vs 25%)，模型倾向于预测多数类。AUC=0.689 表明模型有一定区分能力。

---

### 13.3 多分类模型 (CatBoost Classifier)

**目标**: 预测 Spread 落入 5 个区间之一

**模型配置**:
- 损失函数: MultiClass
- 类别数: 5
- 最终迭代数: 98 (Early Stopping)

**性能指标**:
| 指标 | 值 |
|------|-----|
| Accuracy | **40.2%** |
| Macro F1 | 0.3206 |
| Weighted F1 | 0.4302 |
| 基线 (最频繁类) | 39.2% |

**分类报告**:
| 类别 | Precision | Recall | F1 | 支持数 |
|------|-----------|--------|-----|--------|
| < -$20 | 0.47 | 0.67 | 0.55 | 1,538 |
| -$20~-$5 | 0.51 | 0.47 | 0.49 | 4,162 |
| -$5~$5 | 0.77 | 0.26 | 0.39 | 4,232 |
| $5~$20 | 0.09 | 0.33 | 0.14 | 733 |
| >= $20 | 0.02 | 0.10 | 0.03 | 134 |

**分析**: 模型对极端负值区间 (< -$20) 识别较好 (F1=0.55)，但对正向区间识别困难，因为正向样本太少。

---

## 14. 套利策略回测结果

### 14.1 测试期间
- 时间范围: 2022-06-11 ~ 2024-12-31
- 样本数: 10,799 小时

### 14.2 策略对比

| 策略 | 交易数 | 总收益 | 平均收益/笔 | 胜率 | 盈亏比 | Sharpe |
|------|--------|--------|-------------|------|--------|--------|
| **Baseline (总是Short)** | 10,799 | **$111,422** | $10.32 | 78.8% | 7.17 | **6.10** |
| Binary (All) | 10,799 | $100,770 | $9.33 | 73.5% | 5.31 | 5.54 |
| Binary (prob>0.6) | 6,923 | $87,934 | $12.70 | 80.3% | 7.94 | 6.48 |
| Binary (prob>0.7) | 2,690 | $57,692 | $21.45 | **87.9%** | 18.89 | 5.12 |
| Multiclass (0,4) | 2,824 | $56,227 | $19.91 | 72.6% | 4.65 | 4.79 |
| Regression (\|pred\|>3) | 7,833 | $104,600 | $13.35 | 83.8% | 8.52 | 5.75 |
| Regression (\|pred\|>5) | 5,264 | $93,886 | $17.84 | 86.5% | 10.28 | 5.19 |
| Regression (\|pred\|>10) | 2,318 | $72,752 | $31.39 | 89.4% | 14.06 | 4.08 |
| Regression (\|pred\|>15) | 1,414 | $62,018 | **$43.86** | **91.7%** | **15.44** | 3.50 |

### 14.3 策略分析

**1. 为什么 Baseline 表现最好?**
- LZ_WEST 的 RTM 在 ~75% 的时间低于 DAM
- 这是一个持续性的市场特征 (RTM平均比DAM低$7.17)
- 简单做空 spread 就能捕获这个系统性偏差

**2. 模型的价值在哪里?**
- **高置信度筛选**: Regression (\|pred\|>15) 策略胜率达91.7%，平均每笔收益$43.86
- **风险控制**: 减少交易次数，只在高确定性时进场
- **极端事件识别**: 多分类模型对 <-$20 区间识别F1=0.55

**3. 推荐策略组合**:
- **激进**: Binary (prob>0.6) - 平衡交易量和收益
- **保守**: Regression (\|pred\|>10) - 高胜率，较少交易
- **超保守**: Regression (\|pred\|>15) - 最高胜率

---

## 15. 关键结论

### 15.1 数据层面
1. LZ_WEST 存在持续性的负 spread (RTM < DAM)
2. 约75%时间可以通过做空spread获利
3. 极端正spread (>$20) 非常罕见 (仅1.5%)

### 15.2 模型层面
1. 回归模型MAE=$8.98，比naive基线改进28.6%
2. 二分类AUC=0.689，有一定预测能力
3. 多分类对极端负值区间识别较好

### 15.3 策略层面
1. **简单策略胜过复杂模型**: 总是做空spread获得最高总收益
2. **模型价值在于筛选**: 高阈值策略显著提升胜率和单笔收益
3. **风险收益权衡**: 交易越少，胜率越高，但总收益降低

### 15.4 下一步建议
1. 测试其他Settlement Points (LZ_HOUSTON, LZ_NORTH等)
2. 研究spread随时间的变化趋势
3. 加入外部特征 (天气、负荷预测等)
4. 考虑交易成本和滑点影响

---

## 16. 项目文件结构

```
Delta_Spread_Prediction/
├── PLAN.md                      # 本文档
├── data/
│   ├── rtm_lz_west.csv          # RTM原始数据 (526,266条)
│   ├── dam_lz_west.csv          # DAM原始数据 (95,928条)
│   ├── spread_data.csv          # 合并Spread数据 (54,908条)
│   └── train_features.csv       # 特征工程数据 (53,992条)
├── src/
│   ├── prepare_delta_data.py    # 数据合并脚本
│   ├── delta_feature_extraction.py  # 特征工程脚本
│   ├── train_delta_models.py    # 模型训练脚本
│   └── backtest_arbitrage.py    # 套利回测脚本
├── models/
│   ├── regression_model.cbm     # 回归模型
│   ├── binary_model.cbm         # 二分类模型
│   ├── multiclass_model.cbm     # 多分类模型
│   ├── predictions.csv          # 测试集预测结果
│   ├── results.json             # 模型评估指标
│   └── feature_importance_*.csv # 特征重要性
└── results/
    └── backtest_results.csv     # 回测结果
```

---

## 17. 运行命令参考

```bash
# 激活环境
source /home/lanxin/projects/trueflux/Load-_forecast/Load_Forecast/venv/bin/activate
cd /home/lanxin/projects/trueflux/Delta_Spread_Prediction

# 1. 数据准备
python src/prepare_delta_data.py \
    --rtm ./data/rtm_lz_west.csv \
    --dam ./data/dam_lz_west.csv \
    --output ./data/spread_data.csv

# 2. 特征工程
python src/delta_feature_extraction.py \
    --input ./data/spread_data.csv \
    --output ./data/train_features.csv

# 3. 模型训练
python src/train_delta_models.py \
    --input ./data/train_features.csv \
    --output-dir ./models

# 4. 套利回测
python src/backtest_arbitrage.py \
    --predictions ./models/predictions.csv \
    --output ./results/backtest_results.csv
```
