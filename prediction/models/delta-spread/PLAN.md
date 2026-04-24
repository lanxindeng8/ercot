# RTM-DAM Delta Prediction Plan (Arbitrage Strategy)

## User Requirements Confirmation
- **Prediction Target**: Build all three models (Regression + Binary Classification + Multi-class Classification) - for RTM/DAM arbitrage
- **Prediction Horizon**: 40 hours (DAM order placed morning of the day before -> delivery 0~24 hours the next day)
- **Settlement Point**: Start with LZ_WEST for experimentation

---

## 1. Existing Data Overview

### RTM LMP Data
| Attribute | Value |
|------|-----|
| File Location | `RTM_LMP_Price_Forecast/data/rtm_lz_houston.csv` |
| Time Range | December 2010 - December 2024 (14+ years) |
| Frequency | **15-minute** |
| Record Count | 458,880 records |
| Price Range | -$147 ~ $9,236 /MWh |
| Extracted Point | LZ_HOUSTON |

### DAM LMP Data
| Attribute | Value |
|------|-----|
| File Location | `DAM_Price_Forecast/data/dam_lz_houston.csv` |
| Time Range | January 2015 - December 2025 (11 years) |
| Frequency | **Hourly** |
| Record Count | 95,928 records |
| Price Range | -$0.10 ~ $8,995 /MWh |
| Extracted Point | LZ_HOUSTON |

### Other Extractable Settlement Points
- **Load Zones (LZ)**: LZ_HOUSTON, LZ_WEST, LZ_NORTH, LZ_SOUTH, LZ_AEN, LZ_CPS, LZ_LCRA, LZ_RAYBN
- **Hub Points (HB)**: HB_HOUSTON, HB_WEST, HB_NORTH, HB_SOUTH, etc.
- Raw data contains 1,071+ settlement points

---

## 2. Delta Prediction Feasibility Analysis

### Overlapping Data
- **Overlapping Time Period**: 2015-2024 (approximately 9-10 years)
- **Common Settlement Point**: LZ_HOUSTON (already extracted)
- **Data Volume**:
  - RTM: ~315,000 records (15-minute intervals)
  - DAM: ~78,000 records (hourly intervals)

### Delta Definition
```
RTM-DAM Spread = RTM Price - DAM Price
```
- **Positive value**: RTM higher than DAM (real-time market tight)
- **Negative value**: RTM lower than DAM (real-time market loose)

### Existing Code Support
`short_term_feature_extraction.py` already computes the following delta features:
- `rtm_dam_spread` - RTM-DAM price spread
- `spread_mean_1h` - 1-hour average spread
- `spread_mean_24h` - 24-hour average spread
- `spread_std_24h` - 24-hour spread volatility
- `rtm_dam_ratio` - RTM/DAM ratio

---

## 3. Arbitrage Scenario Description

### DAM Bidding Timeline
```
Day D-1 10:00 AM         ->    Day D 00:00-24:00
   |                              |
 DAM bidding deadline        Actual delivery period

Prediction requirement: Predict hourly RTM-DAM spread for Day D on the morning of D-1
Prediction lead time: 14~38 hours (D-1 10:00 predicting D 00:00~24:00)
```

### Arbitrage Strategy Logic
- **Predicted Spread > 0** (RTM will be higher than DAM): Buy in DAM -> Sell in RTM
- **Predicted Spread < 0** (RTM will be lower than DAM): Sell in DAM -> Buy back in RTM
- **Larger absolute Spread**: Greater arbitrage opportunity

---

## 4. Three Prediction Models

### Model 1: Spread Regression (for quantifying returns)
- **Target**: Predict the specific RTM-DAM spread value
- **Purpose**: Calculate expected returns, determine position size
- **Evaluation Metrics**: MAE, RMSE, R²

### Model 2: Spread Direction Classification (for trading decisions)
- **Target**: Predict RTM > DAM (1) or RTM < DAM (0)
- **Purpose**: Determine buy or sell direction
- **Evaluation Metrics**: Accuracy, Precision, Recall, AUC

### Model 3: Spread Interval Classification (for risk control)
- **Target**: Predict which interval the spread falls into
- **Interval Design**:
  | Class | Interval | Trading Signal |
  |------|------|----------|
  | 0 | Spread < -$20 | Strong Sell DAM |
  | 1 | -$20 ≤ Spread < -$5 | Moderate Sell DAM |
  | 2 | -$5 ≤ Spread < $5 | No Trade |
  | 3 | $5 ≤ Spread < $20 | Moderate Buy DAM |
  | 4 | Spread ≥ $20 | Strong Buy DAM |
- **Evaluation Metrics**: Multi-class F1, Confusion Matrix

---

## 5. Implementation Steps

### Step 1: Extract LZ_WEST Data
```bash
# RTM data extraction
cd /home/lanxin/projects/trueflux/ercot-data
python extract_rtm_data.py --settlement-point LZ_WEST --output ../RTM_LMP_Price_Forecast/data/rtm_lz_west.csv

# DAM data extraction
python extract_dam_data.py --settlement-point LZ_WEST --output ../DAM_Price_Forecast/data/dam_lz_west.csv
```

### Step 2: Data Merging and Spread Calculation
```python
# Aggregate RTM to hourly level
rtm_hourly = rtm_15min.groupby(['date', 'hour']).agg({
    'price': ['mean', 'max', 'min', 'last', 'std']  # Intra-hour statistics
}).reset_index()

# Merge with DAM (aligned by date + hour)
merged = pd.merge(rtm_hourly, dam_hourly, on=['date', 'hour'])
merged['spread'] = merged['rtm_price_mean'] - merged['dam_price']
merged['spread_direction'] = (merged['spread'] > 0).astype(int)
merged['spread_class'] = pd.cut(merged['spread'],
    bins=[-np.inf, -20, -5, 5, 20, np.inf],
    labels=[0, 1, 2, 3, 4])
```

### Step 3: Feature Engineering (40-hour prediction)
```python
Feature groups:
1. Time features: target_hour, target_dow, target_month, is_peak, is_weekend
2. DAM price features: dam_price (known), dam_vs_history, dam_percentile
3. Historical Spread: spread_lag_24h, spread_lag_48h, spread_lag_168h (same time period)
4. Spread statistics: spread_mean_7d, spread_std_7d, spread_by_hour_mean
5. RTM history: rtm_same_hour_mean_7d, rtm_volatility_7d
6. Market state: recent_spike_count, recent_negative_count
```

### Step 4: Model Training
```python
# Three models share features, different target variables
models = {
    'regression': CatBoostRegressor(loss_function='MAE'),
    'binary': CatBoostClassifier(loss_function='Logloss'),
    'multiclass': CatBoostClassifier(loss_function='MultiClass', classes_count=5)
}

# TimeSeriesSplit validation
tscv = TimeSeriesSplit(n_splits=5)
```

### Step 5: Evaluation and Backtesting
- Prediction accuracy/MAE evaluation
- Simulated arbitrage strategy returns
- Calculate Sharpe ratio, maximum drawdown

---

## 6. Data Sufficiency Assessment

| Check Item | Status | Description |
|--------|------|------|
| Time Span | ✅ | 9-10 years (2015-2024) |
| Sample Size | ✅ | ~78,000+ hours |
| Same Settlement Point | ✅ | LZ_WEST (to be extracted) |
| Includes Extreme Events | ✅ | 2021 Winter Storm, etc. |
| Prediction Window Feasible | ✅ | 40 hours with no information leakage |

**Conclusion: Data is fully sufficient for Delta prediction**

---

## 7. Key Files

### Data Files (to be generated)
- `RTM_LMP_Price_Forecast/data/rtm_lz_west.csv` - RTM LZ_WEST data
- `DAM_Price_Forecast/data/dam_lz_west.csv` - DAM LZ_WEST data

### Reference Code
- [short_term_feature_extraction.py](trueflux/RTM_LMP_Price_Forecast/src/short_term_feature_extraction.py) - Existing spread calculation logic
- [extract_rtm_data.py](trueflux/ercot-data/extract_rtm_data.py) - RTM data extraction script
- [extract_dam_data.py](trueflux/ercot-data/extract_dam_data.py) - DAM data extraction script

### New Files
- `delta_prediction/` - New Delta prediction project directory
  - `prepare_delta_data.py` - Data merging and Spread calculation
  - `delta_feature_extraction.py` - 40-hour prediction feature engineering
  - `train_delta_models.py` - Three-model training script
  - `backtest_arbitrage.py` - Arbitrage strategy backtesting

---

## 8. Validation Methods

1. **Model Performance Validation**
   - Regression: MAE < $10, R² > 0.3
   - Binary: Accuracy > 60%, AUC > 0.65
   - MultiClass: Macro-F1 > 0.4

2. **Backtesting Validation**
   - Simulate arbitrage trades for 2023-2024
   - Calculate cumulative return curve
   - Report win rate, average return, maximum drawdown

---

## 9. Implementation Order

1. ✅ Extract RTM and DAM data for LZ_WEST
2. ✅ Merge data, calculate Spread, create labels
3. ✅ Feature engineering (40-hour prediction window)
4. ✅ Train three models (Regression / Binary Classification / Multi-class Classification)
5. ✅ Evaluate model performance
6. ✅ Arbitrage strategy backtesting

---

# Experiment Results Report

> Experiment Date: 2026-02-01
> Settlement Point: LZ_WEST

---

## 10. Data Extraction Results

### 10.1 RTM LZ_WEST Data
| Attribute | Value |
|------|-----|
| File Path | `Delta_Spread_Prediction/data/rtm_lz_west.csv` |
| Record Count | **526,266** records |
| Time Range | 2010-12-01 ~ 2024-12-31 (14 years) |
| Data Frequency | 15-minute |
| Price Range | -$44.82 ~ $9,312.54 /MWh |

### 10.2 DAM LZ_WEST Data
| Attribute | Value |
|------|-----|
| File Path | `Delta_Spread_Prediction/data/dam_lz_west.csv` |
| Record Count | **95,928** records |
| Time Range | 2015-01-01 ~ 2025-12-10 (11 years) |
| Data Frequency | Hourly |
| Price Range | -$10.94 ~ $9,026.99 /MWh |

### 10.3 Merged Data
| Attribute | Value |
|------|-----|
| File Path | `Delta_Spread_Prediction/data/spread_data.csv` |
| Record Count | **54,908** records (hourly level) |
| Overlapping Period | 2015-01-09 ~ 2024-12-31 |
| Valid Feature Samples | 53,992 records |

---

## 11. Spread Statistical Analysis

### 11.1 Basic Statistics
| Statistic | Value |
|--------|-----|
| Mean | **-$7.17** /MWh |
| Standard Deviation | $69.80 /MWh |
| Minimum | -$8,482.50 /MWh |
| Maximum | $1,836.04 /MWh |
| Median | -$2.53 /MWh |

### 11.2 Spread Direction Distribution
| Direction | Sample Count | Proportion |
|------|--------|------|
| RTM > DAM (positive spread) | 13,943 | **25.4%** |
| RTM < DAM (negative spread) | 40,965 | **74.6%** |

**Key Finding**: RTM price at LZ_WEST is lower than DAM price approximately **75%** of the time, averaging $7.17/MWh lower.

### 11.3 Spread Interval Distribution
| Interval | Sample Count | Proportion | Trading Signal |
|------|--------|------|----------|
| < -$20 | 4,999 | 9.1% | Strong Short |
| -$20 ~ -$5 | 16,477 | 30.0% | Moderate Short |
| -$5 ~ $5 | 29,959 | **54.6%** | No Trade |
| $5 ~ $20 | 2,664 | 4.9% | Moderate Long |
| >= $20 | 809 | 1.5% | Strong Long |

---

## 12. Feature Engineering Results

### 12.1 Feature List (44 features)
```
Time Features (9):
  - target_hour, target_dow, target_month, target_day_of_month, target_week
  - target_is_weekend, target_is_peak, target_is_summer
  - hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos

DAM Features (4):
  - target_dam_price (known DAM price)
  - dam_vs_7d_mean, dam_percentile_7d, dam_price_level

Historical Spread Features (12):
  - spread_mean_7d, spread_std_7d, spread_max_7d, spread_min_7d, spread_median_7d
  - spread_mean_24h, spread_std_24h
  - spread_same_hour_hist, spread_same_hour_std
  - spread_same_dow_hour, spread_same_dow_hour_last
  - spread_trend_7d, spread_positive_ratio_7d, spread_positive_ratio_24h

RTM Historical Features (7):
  - rtm_mean_7d, rtm_std_7d, rtm_max_7d
  - rtm_mean_24h, rtm_volatility_24h
  - rtm_same_hour_hist

DAM Historical Features (4):
  - dam_mean_7d, dam_std_7d, dam_mean_24h

Spike Features (3):
  - spike_count_7d, rtm_spike_count_7d
```

### 12.2 Dataset Split
| Dataset | Sample Count | Time Range |
|--------|--------|----------|
| Training Set | 43,193 | 2015-01 ~ 2022-06 |
| Test Set | 10,799 | 2022-06 ~ 2024-12 |

---

## 13. Model Training Results

### 13.1 Regression Model (CatBoost Regressor)

**Model Configuration**:
- Algorithm: CatBoost
- Loss Function: MAE
- Final Iterations: 167 (Early Stopping)
- Depth: 7, Learning Rate: 0.05

**Performance Metrics**:
| Metric | Model | Baseline (Historical Same-Hour Mean) | Improvement |
|------|------|------------------------|------|
| MAE | **$8.98** | $12.57 | **+28.6%** |
| RMSE | $38.15 | - | - |
| R² | 0.2937 | - | - |

**Top 10 Important Features**:
1. spread_same_hour_hist (same-hour historical mean)
2. spread_mean_7d (7-day average)
3. target_dam_price (target hour DAM price)
4. spread_same_dow_hour_last (same hour last week)
5. rtm_same_hour_hist (RTM same-hour historical)
6. spread_mean_24h (24-hour average)
7. dam_vs_7d_mean (DAM relative to 7-day mean)
8. spread_std_7d (7-day volatility)
9. target_hour (target hour)
10. rtm_mean_7d (RTM 7-day mean)

---

### 13.2 Binary Classification Model (CatBoost Classifier)

**Target**: Predict RTM > DAM (1) or RTM < DAM (0)

**Model Configuration**:
- Loss Function: Logloss
- Class Weights: Auto-balanced (Balanced)
- Final Iterations: 44 (Early Stopping)

**Performance Metrics**:
| Metric | Value |
|------|-----|
| Accuracy | **73.5%** |
| Precision | 38.9% |
| Recall | 43.5% |
| F1 Score | 0.4108 |
| AUC | **0.6891** |

**Confusion Matrix**:
```
              Predicted RTM<DAM  Predicted RTM>DAM
Actual RTM<DAM     6,946              1,565
Actual RTM>DAM     1,292                996
```

**Analysis**: Due to severe class imbalance (75% vs 25%), the model tends to predict the majority class. AUC=0.689 indicates the model has some discriminative ability.

---

### 13.3 Multi-class Classification Model (CatBoost Classifier)

**Target**: Predict which of 5 intervals the Spread falls into

**Model Configuration**:
- Loss Function: MultiClass
- Number of Classes: 5
- Final Iterations: 98 (Early Stopping)

**Performance Metrics**:
| Metric | Value |
|------|-----|
| Accuracy | **40.2%** |
| Macro F1 | 0.3206 |
| Weighted F1 | 0.4302 |
| Baseline (Most Frequent Class) | 39.2% |

**Classification Report**:
| Class | Precision | Recall | F1 | Support |
|------|-----------|--------|-----|--------|
| < -$20 | 0.47 | 0.67 | 0.55 | 1,538 |
| -$20~-$5 | 0.51 | 0.47 | 0.49 | 4,162 |
| -$5~$5 | 0.77 | 0.26 | 0.39 | 4,232 |
| $5~$20 | 0.09 | 0.33 | 0.14 | 733 |
| >= $20 | 0.02 | 0.10 | 0.03 | 134 |

**Analysis**: The model identifies the extreme negative interval (< -$20) relatively well (F1=0.55), but struggles with positive intervals due to insufficient positive samples.

---

## 14. Arbitrage Strategy Backtest Results

### 14.1 Test Period
- Time Range: 2022-06-11 ~ 2024-12-31
- Sample Count: 10,799 hours

### 14.2 Strategy Comparison

| Strategy | Trades | Total Return | Avg Return/Trade | Win Rate | Profit Factor | Sharpe |
|------|--------|--------|-------------|------|--------|--------|
| **Baseline (Always Short)** | 10,799 | **$111,422** | $10.32 | 78.8% | 7.17 | **6.10** |
| Binary (All) | 10,799 | $100,770 | $9.33 | 73.5% | 5.31 | 5.54 |
| Binary (prob>0.6) | 6,923 | $87,934 | $12.70 | 80.3% | 7.94 | 6.48 |
| Binary (prob>0.7) | 2,690 | $57,692 | $21.45 | **87.9%** | 18.89 | 5.12 |
| Multiclass (0,4) | 2,824 | $56,227 | $19.91 | 72.6% | 4.65 | 4.79 |
| Regression (\|pred\|>3) | 7,833 | $104,600 | $13.35 | 83.8% | 8.52 | 5.75 |
| Regression (\|pred\|>5) | 5,264 | $93,886 | $17.84 | 86.5% | 10.28 | 5.19 |
| Regression (\|pred\|>10) | 2,318 | $72,752 | $31.39 | 89.4% | 14.06 | 4.08 |
| Regression (\|pred\|>15) | 1,414 | $62,018 | **$43.86** | **91.7%** | **15.44** | 3.50 |

### 14.3 Strategy Analysis

**1. Why does the Baseline perform best?**
- RTM at LZ_WEST is lower than DAM ~75% of the time
- This is a persistent market characteristic (RTM averages $7.17 lower than DAM)
- Simply shorting the spread captures this systematic bias

**2. Where is the model's value?**
- **High-confidence filtering**: Regression (|pred|>15) strategy achieves 91.7% win rate, averaging $43.86 per trade
- **Risk control**: Reduces trade count, only entering on high-certainty signals
- **Extreme event identification**: Multi-class model identifies < -$20 interval with F1=0.55

**3. Recommended strategy combinations**:
- **Aggressive**: Binary (prob>0.6) - Balances trade volume and returns
- **Conservative**: Regression (|pred|>10) - High win rate, fewer trades
- **Ultra-conservative**: Regression (|pred|>15) - Highest win rate

---

## 15. Key Conclusions

### 15.1 Data Level
1. LZ_WEST exhibits a persistent negative spread (RTM < DAM)
2. Approximately 75% of the time, profits can be made by shorting the spread
3. Extreme positive spreads (>$20) are very rare (only 1.5%)

### 15.2 Model Level
1. Regression model MAE=$8.98, a 28.6% improvement over the naive baseline
2. Binary classification AUC=0.689, demonstrating some predictive capability
3. Multi-class model identifies extreme negative intervals relatively well

### 15.3 Strategy Level
1. **Simple strategy outperforms complex models**: Always shorting the spread yields the highest total return
2. **Model value lies in filtering**: High-threshold strategies significantly improve win rate and per-trade returns
3. **Risk-return tradeoff**: Fewer trades lead to higher win rates but lower total returns

### 15.4 Next Steps
1. Test other Settlement Points (LZ_HOUSTON, LZ_NORTH, etc.)
2. Study spread trends over time
3. Incorporate external features (weather, load forecasts, etc.)
4. Consider transaction costs and slippage effects

---

## 16. Project File Structure

```
Delta_Spread_Prediction/
├── PLAN.md                      # This document
├── data/
│   ├── rtm_lz_west.csv          # RTM raw data (526,266 records)
│   ├── dam_lz_west.csv          # DAM raw data (95,928 records)
│   ├── spread_data.csv          # Merged Spread data (54,908 records)
│   └── train_features.csv       # Feature-engineered data (53,992 records)
├── src/
│   ├── prepare_delta_data.py    # Data merging script
│   ├── delta_feature_extraction.py  # Feature engineering script
│   ├── train_delta_models.py    # Model training script
│   └── backtest_arbitrage.py    # Arbitrage backtesting script
├── models/
│   ├── regression_model.cbm     # Regression model
│   ├── binary_model.cbm         # Binary classification model
│   ├── multiclass_model.cbm     # Multi-class classification model
│   ├── predictions.csv          # Test set prediction results
│   ├── results.json             # Model evaluation metrics
│   └── feature_importance_*.csv # Feature importance
└── results/
    └── backtest_results.csv     # Backtest results
```

---

## 17. Command Reference

```bash
# Activate environment
source /home/lanxin/projects/trueflux/Load-_forecast/Load_Forecast/venv/bin/activate
cd /home/lanxin/projects/trueflux/Delta_Spread_Prediction

# 1. Data preparation
python src/prepare_delta_data.py \
    --rtm ./data/rtm_lz_west.csv \
    --dam ./data/dam_lz_west.csv \
    --output ./data/spread_data.csv

# 2. Feature engineering
python src/delta_feature_extraction.py \
    --input ./data/spread_data.csv \
    --output ./data/train_features.csv

# 3. Model training
python src/train_delta_models.py \
    --input ./data/train_features.csv \
    --output-dir ./models

# 4. Arbitrage backtesting
python src/backtest_arbitrage.py \
    --predictions ./models/predictions.csv \
    --output ./results/backtest_results.csv
```
