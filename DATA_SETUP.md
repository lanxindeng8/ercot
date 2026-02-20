# 数据设置指南 (Data Setup Guide)

## 概览

本项目需要两类数据：
1. **HRRR 天气数据**（2GB，不在 git 仓库中）
2. **ERCOT 风电发电量数据**（需要用户提供）

---

## 1. HRRR 天气数据

### 数据描述
- **来源**: NOAA HRRR (High-Resolution Rapid Refresh)
- **分辨率**: 3km
- **区域**: 德州 (Texas)
- **变量**: u10m, v10m, u80m, v80m, t2m, sp
- **时间范围**: 2024-07-01 至 2025-01-22
- **大小**: ~2GB (130 个 zarr 文件)

### 下载步骤

激活虚拟环境并安装依赖：
```bash
cd /path/to/wind-generation-forecast
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 安装 earth2studio（如果需要）
# PIP_CONFIG_FILE=/dev/null pip install -e /path/to/earth2studio
```

下载 HRRR 数据（示例）：
```bash
# 下载单个日期的所有初始化时刻
python scripts/fetch_hrrr_data.py --date 2024-07-01 --hours 0 6 12 18

# 批量下载（建议使用后台任务）
for date in 2024-07-01 2024-07-08 2024-07-15; do
    python scripts/fetch_hrrr_data.py --date $date --hours 0 6 12 18 &
done
```

### 数据文件结构
```
data/hrrr/
├── hrrr_20240701_00.zarr/
├── hrrr_20240701_06.zarr/
├── hrrr_20240701_12.zarr/
├── hrrr_20240701_18.zarr/
├── ...
└── hrrr_20250122_18.zarr/
```

每个 zarr 文件包含：
- 形状: (1, 7, 6, 424, 445)
  - 1 个初始化时刻
  - 7 个预报时效 (0, 1, 2, 3, 6, 9, 12 小时)
  - 6 个气象变量
  - 424×445 网格点覆盖德州

### 已下载的日期列表
参考当前仓库的下载日期：
- 2024-07: 01, 08, 15, 22, 29 (每天 4 个初始化时刻)
- 2024-08: 01, 08, 15, 22, 29
- 2024-09: 01, 08, 15, 22, 29
- 2024-10: 01, 08, 15, 22, 29
- 2024-11: 01, 08, 15, 22, 29
- 2024-12: 01, 08, 15, 22, 28, 29
- 2025-01: 01, 05, 10, 15, 18, 19, 20, 21, 22

---

## 2. ERCOT 风电发电量数据

### 数据要求

**格式**: CSV 或 Parquet
**必需字段**:
- `timestamp` 或 `valid_time`: datetime 格式
- `wind_generation`: 风电发电量 (MW)

**时间要求**:
- 时间范围: 2024-07-01 至 2025-01-22
- 频率: 小时级数据
- 时区: UTC 或明确标注

### 示例数据格式

CSV:
```csv
timestamp,wind_generation
2024-07-01 00:00:00,15234.5
2024-07-01 01:00:00,16012.3
2024-07-01 02:00:00,14890.1
...
```

或 Parquet 格式存储在 `data/ercot/wind_generation.parquet`

### 数据获取来源
- ERCOT 官网: http://www.ercot.com/gridinfo/generation
- 具体页面: 查找 "Wind Power Production" 或 "Actual System Load by Fuel Type"

---

## 3. 数据对齐与特征构建

获取两类数据后，运行特征构建脚本：

```bash
python scripts/build_features.py
```

该脚本会：
1. 读取 HRRR 天气数据
2. 读取 ERCOT 风电数据
3. 对齐时间戳
4. 计算风速、风力特征
5. 计算 ramp 特征（变化率、加速度等）
6. 计算时间特征（小时、星期几、是否夜间等）
7. 保存到 `data/processed/features.parquet`

---

## 4. 训练模型

```bash
# 使用 LightGBM 训练
python scripts/train_models.py --model gbm

# 使用 LSTM 训练（需要更多数据）
python scripts/train_models.py --model lstm

# 使用 ensemble（组合模型）
python scripts/train_models.py --model ensemble
```

---

## 5. 常见问题

### Q: 下载 HRRR 数据很慢怎么办？
A: 使用并行下载，每次下载 4-5 个日期。NOAA 服务器可能限速，耐心等待。

### Q: 如何验证 HRRR 数据完整性？
A: 运行以下命令检查：
```bash
ls data/hrrr/*.zarr | wc -l  # 应该有 ~130 个文件
```

### Q: ERCOT 数据时区不一致怎么办？
A: 统一转换为 UTC：
```python
import pandas as pd
df = pd.read_csv('ercot_data.csv')
df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize('America/Chicago').dt.tz_convert('UTC')
```

### Q: 训练数据不足怎么办？
A: 可以下载更多历史 HRRR 数据（2023-2024 年份）。参考 `scripts/fetch_hrrr_data.py` 修改日期范围。

---

## 联系方式

如有问题，请提交 GitHub Issue 或联系项目维护者。
