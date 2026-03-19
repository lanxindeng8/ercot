# TrueFlux — 架构设计文档

> 最后更新: 2026-03-18

---

## 系统架构

```
┌──────────────────────────────────────────────────────────────────┐
│                        Mac mini (Nancy)                         │
│                                                                  │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────────┐  │
│  │  Scraper     │───▶│  SQLite DB   │◀───│  Prediction API    │  │
│  │  (LaunchAgent│    │  (primary)   │    │  FastAPI :8011     │  │
│  │   cron 5min) │    │              │    │  (LaunchAgent)     │  │
│  └─────────────┘    │  + InfluxDB   │    └────────┬───────────┘  │
│                      │  (legacy)    │             │              │
│                      └──────────────┘             │              │
│                                                    │              │
│  ┌─────────────┐    ┌──────────────┐    ┌────────▼───────────┐  │
│  │  Prediction  │───▶│  ML Models   │    │  Viewer (Next.js)  │  │
│  │  Runner      │    │  7 trained   │    │  localhost:3000     │  │
│  │  (5min loop) │    │  .pkl/.cbm   │    │  4 pages           │  │
│  └─────────────┘    └──────────────┘    └────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Telegram Alert Service (矿场开停机 + Spike 告警)         │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 目录结构

```
ercot/
├── scraper/                   # 数据采集
│   ├── src/
│   │   ├── ercot_client.py        # ERCOT API 客户端
│   │   ├── scraper_rtm_lmp.py     # RTM 价格爬虫 (历史)
│   │   ├── scraper_rtm_lmp_realtime.py  # RTM 实时
│   │   ├── scraper_dam_lmp.py     # DAM 价格爬虫
│   │   ├── scraper_dam_lmp_cdr.py # DAM CDR
│   │   ├── cdr_scraper.py         # CDR 通用爬虫
│   │   ├── sqlite_archive.py      # SQLite 写入
│   │   └── influxdb_writer.py     # InfluxDB 写入 (legacy)
│   └── scripts/                   # 一次性脚本 (回填、审计)
│
├── prediction/                # 预测服务
│   ├── src/
│   │   ├── main.py               # FastAPI 应用 (23+ 端点)
│   │   ├── config.py             # 配置常量
│   │   ├── models/               # 模型加载 + 推理
│   │   │   ├── dam_v2_predictor.py
│   │   │   ├── rtm_predictor.py
│   │   │   ├── delta_spread.py
│   │   │   ├── spike_predictor.py
│   │   │   ├── wind_predictor.py
│   │   │   ├── load_predictor.py
│   │   │   └── bess_predictor.py
│   │   ├── features/             # 特征工程
│   │   │   ├── unified_features.py
│   │   │   └── dam_features_v2.py
│   │   ├── data/                 # 共享数据层 (计划中)
│   │   │   ├── weather/              # 天气数据 (所有模型共享)
│   │   │   │   ├── openmeteo_client.py   # Open-Meteo 历史天气 API
│   │   │   │   ├── hrrr_client.py        # HRRR 高精度 (从 wind 模型提升)
│   │   │   │   ├── zone_weather.py       # zone-level 特征: T_anom, ΔT, WC, CF
│   │   │   │   └── stations.py           # 6 城市坐标 → zone 映射
│   │   │   └── ercot/                # ERCOT 市场数据 (新数据源)
│   │   │       ├── reserves.py           # RT reserves / ORDC (NP6-792-ER)
│   │   │       └── wind_forecast.py      # Wind forecast vs actual (NP4-732-CD)
│   │   ├── dispatch/             # 调度信号
│   │   │   ├── mining_dispatch.py    # 矿场开停机
│   │   │   ├── bess_signals.py       # BESS 套利
│   │   │   └── alert_service.py      # Telegram 告警
│   │   └── auth/
│   │       └── api_keys.py          # API key 管理
│   ├── scripts/                  # 管理脚本
│   │   ├── run_predictions.py        # 预测循环 runner
│   │   ├── score_predictions.py      # 准确率评分
│   │   ├── manage_keys.py           # API key CLI
│   │   └── train_*.py               # 训练脚本
│   └── tests/                    # 188 tests
│
├── viewer/                    # 前端 Dashboard
│   └── src/
│       ├── app/
│       │   ├── dashboard/            # 总览页
│       │   ├── predictions/          # 模型预测页
│       │   ├── dispatch/             # 调度信号页
│       │   └── api/                  # Next.js API routes
│       └── components/
│           ├── dashboard/            # StatusBanner, MiningScheduleCard, etc.
│           ├── AccuracyPanel.tsx
│           ├── BessSignalPanel.tsx
│           ├── WindForecastChart.tsx
│           └── ...
│
├── PLAN.md                    # 路线图 (本文件)
├── DESIGN.md                  # 架构设计
├── WORKPLAN.md                # Sprint 执行记录
└── .dev-state.json            # Kira 状态机
```

---

## 数据层

### 存储

| 存储 | 用途 | 数据量 |
|------|------|--------|
| **SQLite** (primary) | RTM/DAM 价格, 燃料混合, 预测结果, API keys | RTM 1070万条, DAM 260万条 |
| **InfluxDB Cloud** (legacy) | 旧数据源, viewer 部分 API 仍在用 | 同上 (镜像) |

### 数据表

| 表名 | 内容 | 更新频率 |
|------|------|---------|
| `rtm_lmp_hist` | RTM 历史价格 (2015~昨天) | 每日 |
| `rtm_lmp_cdr` | RTM 实时价格 | 每 5 分钟 |
| `dam_lmp_hist` | DAM 历史价格 | 每日 |
| `dam_lmp_cdr` | DAM 实时/次日 | 每日 |
| `fuel_mix_hist` | 发电燃料组合 (含 Wind/Load) | 每 5 分钟 |
| `predictions` | 模型预测结果 | 每 5 分钟 |
| `api_keys` | API key + tier + 用量 | 按需 |

**⚠️ 关键设计决策**: SQLite fetcher 必须合并 `_hist` + `_cdr` 表才能拿到最新数据。`_hist` 只到昨天，`_cdr` 有当天实时数据。

### 数据时间范围
- 价格: 2015-01-01 ~ 实时 (11 年+)
- 燃料/Wind/Load: 537 万条

---

## ML 模型

### 模型清单

| 模型 | 算法 | 特征数 | 训练数据 | 核心指标 |
|------|------|--------|---------|---------|
| DAM v2 | LightGBM | 42 | 11年, 5 settlement points | MAE=5.88 (HB_WEST) |
| RTM | LightGBM | 42 | 11年, 3 horizons (1h/4h/24h) | MAE=12.1 (1h) |
| Delta-Spread | CatBoost 回归+分类 | 42 | RTM-DAM spread | Sharpe=34.86, WR=94.6% |
| Spike | CatBoost 二分类 | 42 | 极端价格事件 | F1=0.939, Recall=1.0 |
| Wind | LightGBM 分位回归 | 58 | fuel_mix_hist | MAE=14.3MW, Q10/Q50/Q90 |
| Load | CatBoost+LightGBM Ensemble | 34 | 87K 记录 | MAPE=0.81% |
| BESS | LP 优化器 | N/A | DAM 预测输入 | 充放电时间表 |

### 特征工程

核心特征管道: `unified_features.py`

- **价格特征**: lag (1h~168h), rolling mean/std/min/max, DAM-RTM 差异
- **时间特征**: hour, day_of_week, month, is_weekend, is_holiday
- **风电/负荷**: 从 fuel_mix 提取, lag + rolling
- **DAM v2 专有**: `dam_features_v2.py` (settlement point 独立)

### 模型文件位置
```
prediction/
├── models/dam_v2/          # 5 个 .pkl (per settlement point)
├── models/rtm/             # 3 个 .pkl (per horizon)
├── catboost_info/          # CatBoost 训练日志
└── *.pkl / *.cbm           # 其他模型文件
```

---

## API 设计

**Base URL**: `http://localhost:8011`

### 端点分类

| 分类 | 端点 | 方法 |
|------|------|------|
| **System** | `/health` | GET |
| | `/settlement-points` | GET |
| **Model Info** (×7) | `/models/{dam,rtm,delta-spread,spike,wind,load,bess}/info` | GET |
| **Predictions** (×7) | `/predictions/dam/next-day` | GET |
| | `/predictions/rtm` | GET |
| | `/predictions/delta-spread` | GET |
| | `/predictions/spike` | GET |
| | `/predictions/wind` | GET |
| | `/predictions/load` | GET |
| | `/predictions/bess` | GET |
| **Accuracy** | `/accuracy` | GET |
| **Dispatch** | `/dispatch/mining/schedule` | GET |
| | `/dispatch/mining/savings` | GET |
| | `/dispatch/alerts/config` | POST |
| | `/dispatch/bess/daily-signals` | GET |
| | `/dispatch/bess/pnl` | GET |
| | `/dispatch/bess/risk` | GET |
| **Admin** | `/admin/keys/*` | GET/POST/DELETE |

### 认证
- `X-API-Key` header (所有预测/调度端点)
- `X-Admin-Token` header (admin 端点)
- 免认证: `/health`, `/docs`, `/openapi.json`
- Rate limit: 按 tier (free: 100/day, pro: 10K/day, enterprise: unlimited)

---

## 部署

### LaunchAgents (macOS)

| 服务 | Label | 行为 |
|------|-------|------|
| Prediction API | `com.trueflux.prediction-api` | uvicorn :8011, 自动重启 |
| Prediction Runner | `com.trueflux.prediction-runner` | 每 5 分钟运行预测循环 |
| Scraper (各 CDR) | `com.ercot.scraper.*` | cron 采集 |

### Python 环境
- venv: `/Users/bot/.venvs/ercot-prediction/`
- 依赖: catboost, lightgbm, fastapi, uvicorn, loguru, pyyaml, pandas, numpy, scipy

### Viewer
- Next.js, localhost:3000
- 4 页面: Dashboard / Market / Predictions / Dispatch
- 暗色主题, 5min 自动刷新

---

## 共享数据层设计 (计划中)

### 设计原则

天气数据、ERCOT 市场数据是**基础设施**，不属于任何一个模型。
Wind 模型、Spike 模型、Load 模型、BESS 策略都从 `src/data/` 读取。

当前 HRRR 天气代码位于 `models/wind/src/data/hrrr_client.py`，
需要提升到 `src/data/weather/` 成为共享模块。

### 天气数据方案

| 来源 | 变量 | 粒度 | 历史深度 | 数据量 | 用途 |
|------|------|------|---------|--------|------|
| **Open-Meteo Archive** | T, WindSpeed, WindDir, Humidity, Pressure | 1h, 6 城市 | 2015~now (11yr) | ~16 MB (SQLite) | Zone-level 特征 (T_anom, ΔT, WC, CF) |
| **HRRR via Herbie** | u/v 风速 (10m, 80m), t2m, 气压 | 1h, 3km 格点 | 已有 6 月 | 数 GB/年 (GRIB2) | 高精度补充 (后续) |

天气站 → Zone 映射:
- LZ_CPS → San Antonio (29.42°N, 98.49°W)
- LZ_WEST → Midland/Odessa (31.95°N, 102.18°W)  
- LZ_HOUSTON → Houston (29.76°N, 95.37°W)
- HB_NORTH → Dallas/Fort Worth (32.78°N, 96.80°W)
- HB_SOUTH → Corpus Christi (27.80°N, 97.40°W)
- System → Austin (30.27°N, 97.74°W)

### ERCOT 新数据源

| 数据 | ERCOT Product | 格式 | 历史 | 数据量 (11yr) | 可用性 |
|------|--------------|------|------|--------------|--------|
| **RT Reserves + ORDC** | NP6-792-ER | XLSX (年度), 33 列, ~9K rows/月 | 2017~2026 | ~1.2M rows, ~230 MB | ✅ 已验证: MIS 匿名下载, 含 PRC/RTOLCAP/RTOFFCAP/RTORPA/System Lambda 等 |
| **Wind Forecast** | NP4-732-CD | CSV (小时), 含 GEN/STWPF/WGRPP per region | 滚动 48h+168h | ~100K rows/yr | ✅ 已验证: System-wide + 3 regions (South_Houston, West, North) |
| **Binding Constraints** | NP6-86-CD | CSV (per SCED ~5min) | 滚动 28h | 每个文件 ~2KB, ~105K 文件/年 | ⚠️ 已验证可下但历史回填量太大 (~200MB/yr zips, 100K+ HTTP requests) |

### 存储预算

| 数据 | 新增 SQLite 大小 |
|------|-----------------|
| Open-Meteo 天气 (6城×11yr) | ~16 MB |
| RT Reserves ORDC (8yr) | ~230 MB |
| Wind Forecast (hourly, 近期) | ~20 MB |
| Binding Constraints | ❌ 暂不回填历史 (量太大) |
| **总计** | **~270 MB** |

当前 SQLite DB 4.9 GB，磁盘剩余 59 GB。存储没有问题。

---

## 设计决策记录

| # | 决策 | 理由 |
|---|------|------|
| 1 | SQLite 替代 InfluxDB 作为主数据源 | InfluxDB Cloud 查询延迟高, SQLite 本地读取快 |
| 2 | LightGBM/CatBoost 而非深度学习 | 表格数据, 11年历史足够, 训练快, 可解释 |
| 3 | 单 main.py 而非微服务 | 开发速度优先, 后续可拆 router |
| 4 | LaunchAgent 而非 Docker | Mac mini 本地部署, 简单可靠 |
| 5 | 15 settlement points (全部 HB_* + LZ_*) | 覆盖 ERCOT 所有主要交易节点 |
| 6 | API key + tier 而非 OAuth | MVP 阶段简单认证足够 |
| 7 | LP 优化器做 BESS 调度 | 确定性最优, 不需要 RL 复杂度 |
| 8 | 天气数据共享层 (不放 wind 模型内) | Wind/Spike/Load/BESS 都需要天气，避免重复 |
| 9 | Open-Meteo 先行, HRRR 后补 | Open-Meteo 免费 REST + 完整历史，HRRR 需下载大文件 |
| 10 | Binding Constraints 暂不回填历史 | 每 5min 一个文件, 11 年 ~100 万文件, 回填成本太高 |
