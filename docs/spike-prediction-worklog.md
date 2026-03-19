# Zone-Level Spike Prediction — 工作日志

> 按步骤记录实施过程，每步包含：目标、设计、结果、教训

---

## Phase 0: 数据获取基础设施

### Step 0.1: 共享数据层目录结构

**目标**: 创建 `prediction/src/data/weather/` 和 `prediction/src/data/ercot/` 目录结构

**当前状态**: 天气数据代码在 `prediction/models/wind/src/data/hrrr_client.py`（耦合在 wind 模型内）

**设计**:
```
prediction/src/data/
├── __init__.py
├── weather/
│   ├── __init__.py
│   ├── openmeteo_client.py    # Open-Meteo Archive API 客户端
│   ├── stations.py            # Zone → 城市坐标映射
│   └── zone_weather.py        # T_anom, ΔT, Wind Chill, Cold Front 计算
└── ercot/
    ├── __init__.py
    ├── wind_forecast.py       # NP4-732-CD: GEN + STWPF (forecast vs actual)
    └── reserves.py            # NP6-792-ER: PRC, ORDC price adders
```

**注意**: 
- `hrrr_client.py` 暂不移动（wind 模型仍在用），后续 refactor
- `ercot/` 下的模块复用 `scraper/src/ercot_client.py` 的认证逻辑
- credentials 从环境变量读取（LaunchAgent 已设置）

---

### Step 0.2: Open-Meteo 天气数据获取

**目标**: 获取 6 个城市、2015-2026 的小时级天气数据，存入 SQLite

**API**: `https://archive-api.open-meteo.com/v1/archive`

**验证结果 (2026-03-19)**:
- ✅ 免费、无 key、REST JSON
- ✅ San Antonio 2025-12-14 冷锋确认 (18.6→6.3°C)
- ✅ 1 城市 1 年 ≈ 481 KB JSON
- ⚠️ `wind_speed_80m` 全 null，需用 ERA5 endpoint 或只用 10m
- 6 城市 × 11 年 → SQLite ≈ 16 MB

**城市坐标映射**:
| Zone | 城市 | 纬度 | 经度 |
|------|------|------|------|
| LZ_CPS | San Antonio | 29.42 | -98.49 |
| LZ_WEST | Midland/Odessa | 31.95 | -102.18 |
| LZ_HOUSTON | Houston | 29.76 | -95.37 |
| HB_NORTH / LZ_NORTH | Dallas/Fort Worth | 32.78 | -96.80 |
| HB_SOUTH / LZ_SOUTH | Corpus Christi | 27.80 | -97.40 |
| System (HB_BUSAVG/HUBAVG) | Austin | 30.27 | -97.74 |

**变量**:
- `temperature_2m` (°C) — 核心
- `wind_speed_10m` (km/h)
- `wind_direction_10m` (°)
- `relative_humidity_2m` (%)
- `surface_pressure` (hPa)
- `dew_point_2m` (°C) — 算 Wind Chill 需要

**SQLite 表设计**:
```sql
CREATE TABLE weather_hourly (
    station TEXT NOT NULL,           -- 'san_antonio', 'houston', etc.
    time TEXT NOT NULL,              -- ISO-8601, hourly
    temperature_2m REAL,
    wind_speed_10m REAL,
    wind_direction_10m REAL,
    relative_humidity_2m REAL,
    surface_pressure REAL,
    dew_point_2m REAL,
    PRIMARY KEY (station, time)
);
```

**获取策略**:
- Open-Meteo 限制单次请求最多 ~1 年
- 分 11 次请求 per 城市 (2015~2025 各一年 + 2026 ytd)
- 6 城市 × 12 请求 = 72 次 HTTP 请求
- 加 sleep(1) 避免 rate limit → ~2 分钟完成

**测试计划**:
1. 先拉 1 城市 1 年验证数据完整性
2. 检查 null 值比例
3. 确认时区对齐 (America/Chicago)
4. 全量拉取
5. 写 SQLite 导入

---

### Step 0.3: ERCOT Wind Forecast 获取 (NP4-732-CD)

**目标**: 获取 wind forecast (STWPF) vs actual (GEN)，存入 SQLite

**API**: `https://api.ercot.com/api/public-reports/np4-732-cd/wpp_hrly_avrg_actl_fcast`

**验证结果 (2026-03-19)**:
- ✅ Data API（不是 archive），同现有 LMP scraper
- ✅ 21 个字段: postedDatetime, deliveryDate, hourEnding, + 4 metrics × 4 regions (SystemWide, SouthHouston, West, North) + HSL + DST
- ✅ 认证: 复用现有 ercot_client.py + LaunchAgent credentials
- ⚠️ 2025-12-14 查询返回 5184 行 (多个 posted versions)

**字段映射** (per region):
- `genXxx` — 实际发电 MW
- `STWPFXxx` — 短期风电功率预报 MW
- `WGRPPXxx` — Wind Generation Resource Production Potential MW
- `COPHSLXxx` — Current Operating Plan HSL MW

**核心特征计算**:
- `wind_forecast_error = GEN - STWPF` (surprise: 负值 = 实际比预测少)
- `wind_capacity_factor = GEN / COPHSL`
- `wind_surprise_pct = (GEN - STWPF) / STWPF * 100`

**SQLite 表设计**:
```sql
CREATE TABLE wind_forecast (
    delivery_date TEXT NOT NULL,
    hour_ending INTEGER NOT NULL,
    posted_datetime TEXT NOT NULL,
    region TEXT NOT NULL,             -- 'system', 'south_houston', 'west', 'north'
    gen_mw REAL,
    stwpf_mw REAL,
    wgrpp_mw REAL,
    cop_hsl_mw REAL,
    PRIMARY KEY (delivery_date, hour_ending, region, posted_datetime)
);
```

**获取策略**:
- 用现有 `ercot_client.fetch_paginated_data()` 方法
- deliveryDateFrom/To 分月请求
- ~8760 hours × 4 regions × 多个 posted versions = 大量数据
- 先拉 2025-12 一个月测试量

---

### Step 0.4: RT Reserves / ORDC 获取 (NP6-792-ER)

**目标**: 获取 RT reserve margin (PRC) + ORDC price adders 历史数据

**API**: Archive 下载 (不是 data API)
- 列表: `https://api.ercot.com/api/public-reports/archive/np6-792-er`
- 下载: `https://api.ercot.com/api/public-reports/archive/np6-792-er?download={docId}`

**验证结果 (2026-03-19)**:
- ✅ 年度 XLSX 文件，每月一个 sheet
- ✅ 33 列: Batch ID, SCED Timestamp, PRC, System Lambda, RTOLCAP, RTOFFCAP, RTORPA, RTORDPA, RTOLHSL, RTBP, etc.
- ✅ ~9K rows/月，~108K rows/年
- ✅ 数据从 2017 年开始 (8 年)
- ⚠️ Header 在 row 8 (前面有空行)
- ⚠️ 2025 年的 HIST_RT_SCED_PRC_ADDR 文件 sheet 全空；RTM_ORDC 版本有数据

**可用文件清单**:
- RTM_ORDC_REL_DPLY_PRC_ADDR_RSRV_2024 (docId: 1065495488) — 18 MB xlsx
- RTM_ORDC_REL_DPLY_PRC_ADDR_RSRV_2023 (docId: 969827183)
- RTM_ORDC_REL_DPLY_PRC_ADDR_RSRV_2022 (docId: 899479048)
- RTM_ORDC_REL_DPLY_PRC_ADDR_RSRV_2021 (docId: 814938847)
- RTM_ORDC_REL_DPLY_PRC_ADDR_RSRV_2020 (docId: 751366904)
- RTM_ORDC_REL_DPLY_PRC_ADDR_RSRV_2019 (docId: 694452063)
- RTM_ORDC_REL_DPLY_PRC_ADDR_RSRV_2017 (docId: 644817015)
- HIST_RT_SCED_PRC_ADDR_2026 (docId: 1204109813) — 新格式
- HIST_RT_SCED_PRC_ADDR_2025 (docId: 1180319793) — 新格式但 sheets 空?

**SQLite 表设计**:
```sql
CREATE TABLE rt_reserves (
    sced_timestamp TEXT NOT NULL,
    repeated_hour TEXT,
    batch_id INTEGER,
    system_lambda REAL,
    prc REAL,                         -- Physical Responsive Capability (MW)
    rtolcap REAL,                     -- RT Online Capacity
    rtoffcap REAL,                    -- RT Offline Capacity
    rtorpa REAL,                      -- RT Online Reserve Price Adder
    rtoffpa REAL,                     -- RT Offline Reserve Price Adder
    rtolhsl REAL,                     -- RT Online HSL
    rtbp REAL,                        -- RT Base Point
    rtordpa REAL,                     -- RT ORDC Price Adder
    PRIMARY KEY (sced_timestamp, batch_id)
);
```

**获取策略**:
1. 通过 archive API 列出所有可用文件
2. 逐年下载 zip → 解压 xlsx
3. 解析每个月的 sheet (header=row 8)
4. 写入 SQLite
5. 预计 8 年 × 18MB = ~144 MB 下载，解析后 ~230 MB SQLite

---

## 执行记录

### Step 0.1: 目录结构 ✅
- Kira 自己做，commit `57aec0a`

### Step 0.2: Open-Meteo 天气数据 ✅ → 🔧 Codex Review 修复中
- CC 写完 (session `glow-daisy`): 5 文件，29 tests pass
- Codex review (session `mellow-falcon`) 发现 6 个问题:
  1. 🔴 Wind Chill 输出华氏度（应为摄氏度）
  2. 🔴 DST 时间冲突 — local time PK 在 fall-back 丢数据
  3. 🔴 shift(1) 是 row-based，DST 日变成 2h delta
  4. 🟡 API 错误无容错
  5. 🟡 INSERT OR REPLACE 语义不对
  6. 🟡 测试太弱
- CC 修复中 (session `crisp-cedar`)

### Step 0.3: Wind Forecast 数据 ✅ → 待 Codex Review
- CC 写完 (session `faint-orbit`): 3 文件，13 tests pass
- 待 Codex review

### Step 0.4: RT Reserves 数据 — 任务已规划，待启动

### Step 0.5: 验证 — 用 2025-12-14 案例日检查所有数据对齐
