# TrueFlux — 架构设计文档

> 最后更新: 2026-03-25

---

## 系统架构

```
┌──────────────────────────────────────────────────────────────────┐
│                        Mac mini (Nancy)                         │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  Scraper Layer (LaunchAgents)                               │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │ │
│  │  │ rtm-lmp-cdr  │  │ rtm-lmp-api  │  │ dam-pipeline     │  │ │
│  │  │ 5min, HTML   │  │ 1h, ERCOT API│  │ 14:00 daily      │  │ │
│  │  │ (no API)     │  │ (bandwidth   │  │ pred→fetch→CDR   │  │ │
│  │  │              │  │  limited)    │  │ →Telegram        │  │ │
│  │  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘  │ │
│  └─────────┼─────────────────┼───────────────────┼─────────────┘ │
│            │                 │                   │               │
│            ▼                 ▼                   ▼               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  SQLite DB (6.3 GB) — primary storage                    │   │
│  │  RTM 17M | DAM 1.4M | Fuel 5.4M | Reserves 1.1M        │   │
│  │  Weather 590K | Wind 114K | Spike Labels 5.3M            │   │
│  │  + InfluxDB Cloud (legacy mirror)                        │   │
│  └────────────────────────────┬─────────────────────────────┘   │
│                               │                                  │
│  ┌────────────────────────────▼─────────────────────────────┐   │
│  │  Prediction API (FastAPI :8011, always-on)               │   │
│  │  7 models | 23+ endpoints | API key auth (localhost免)    │   │
│  └────────────────────────────┬─────────────────────────────┘   │
│                               │                                  │
│  ┌──────────────┐   ┌────────▼──────────┐   ┌──────────────┐   │
│  │ Prediction   │   │ Viewer (Next.js)  │   │ Telegram     │   │
│  │ Runner 5min  │──▶│ localhost:3000    │   │ Alerts       │   │
│  │ (local API)  │   │ 4 pages           │   │ (daily)      │   │
│  └──────────────┘   └───────────────────┘   └──────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

---

## LaunchAgent 架构 (2026-03-23 重构)

| Job | Label | 频率 | ERCOT API? | 说明 |
|-----|-------|------|-----------|------|
| RTM CDR | `com.trueflux.rtm-lmp-cdr` | 5min | ❌ HTML scrape | 实时 RTM 数据 |
| RTM API | `com.trueflux.rtm-lmp-api` | 1h | ✅ | 历史回填（6h延迟） |
| DAM Pipeline | `com.trueflux.dam-pipeline` | 每天14:00 | ✅ (1次) | predictions → fetch → CDR → Telegram |
| Prediction API | `com.trueflux.prediction-api` | 常驻 | ❌ | FastAPI :8011 |
| Prediction Runner | `com.trueflux.prediction-runner` | 5min | ❌ | 调 localhost:8011 |
| Telegram Summary | `com.trueflux.telegram-lmp-summary` | 每天06:30 | ❌ | 读 InfluxDB |
| BTC Monitor | `com.trueflux.btc-price-monitor` | 5min | ❌ | PolyManager 项目 |

**ERCOT API 调用者只有 2 个**，共享同一个 subscription key 的 hourly bandwidth quota。
详见 [docs/ercot-api-rate-limits.md](docs/ercot-api-rate-limits.md)。

**已废弃（不再加载）**:
- `rtm-lmp-scraper` → 拆为 `rtm-lmp-cdr` + `rtm-lmp-api`
- `dam-lmp-scraper`, `dam-lmp-cdr-scraper`, `dam-predictions`, `telegram-dam-schedule` → 合并为 `dam-pipeline`

---

## 目录结构

```
ercot/
├── scraper/                   # 数据采集
│   ├── src/
│   │   ├── ercot_client.py        # ERCOT API 客户端 (429退避: 10s初始, 5min上限, jitter)
│   │   ├── scraper_rtm_lmp.py     # RTM API 爬虫 (历史, 6h延迟)
│   │   ├── scraper_rtm_lmp_realtime.py  # RTM CDR 爬虫 (实时)
│   │   ├── scraper_dam_lmp.py     # DAM API 爬虫
│   │   ├── scraper_dam_lmp_cdr.py # DAM CDR HTML 爬虫
│   │   ├── cdr_scraper.py         # CDR 通用爬虫
│   │   ├── sqlite_archive.py      # SQLite 写入
│   │   └── influxdb_writer.py     # InfluxDB 写入 (legacy)
│   └── scripts/
│       ├── run_rtm_cdr_scraper.sh     # CDR 5min (no API)
│       ├── run_rtm_api_scraper.sh     # API 1h (rate limited)
│       └── run_dam_pipeline.sh        # DAM 全流水线
│
├── prediction/                # 预测服务
│   ├── src/
│   │   ├── main.py               # FastAPI (1882行, 23+端点)
│   │   ├── config.py
│   │   ├── models/               # 模型推理
│   │   │   ├── dam_v2_predictor.py
│   │   │   ├── rtm_predictor.py
│   │   │   ├── delta_spread.py
│   │   │   ├── spike_predictor.py
│   │   │   ├── spike_v2_predictor.py   # Phase 1: 14 SP, Optuna tuned
│   │   │   ├── wind_predictor.py
│   │   │   ├── load_predictor.py
│   │   │   └── bess_predictor.py
│   │   ├── features/
│   │   │   ├── unified_features.py     # 80 features 统一管道
│   │   │   ├── spike_features.py       # 30 spike-specific features
│   │   │   └── dam_features_v2.py
│   │   ├── data/                 # 共享数据层
│   │   │   ├── weather/              # Open-Meteo 天气 (6站, 2015-2026)
│   │   │   └── ercot/                # ERCOT 市场数据
│   │   │       ├── reserves.py           # RT reserves/ORDC (NP6-792-ER)
│   │   │       └── wind_forecast.py      # Wind forecast (NP4-732-CD)
│   │   ├── dispatch/
│   │   │   ├── mining_dispatch.py
│   │   │   ├── bess_signals.py
│   │   │   └── alert_service.py
│   │   └── auth/
│   │       └── api_keys.py          # API key + tier + rate limit
│   ├── scripts/
│   │   ├── run_predictions.py        # 预测循环 runner
│   │   ├── retrain_all_v2.py         # 全模型重训 (80 features)
│   │   ├── train_spike.py            # Spike baseline 训练
│   │   ├── tune_spike.py             # Spike Optuna 调参
│   │   ├── fetch_reserves.py         # RT reserves 回填
│   │   ├── fetch_wind_forecast.py    # Wind forecast 回填
│   │   ├── fetch_wind_one_month.py   # Cron 单月增量获取
│   │   └── manage_keys.py           # API key CLI
│   ├── models/                   # 训练产出
│   │   ├── dam_v2/                   # 15 SP LightGBM
│   │   ├── rtm/                      # 15 SP × 3 horizons
│   │   ├── spike/                    # 14 SP baseline
│   │   └── spike_tuned/              # 14 SP Optuna tuned
│   └── tests/                    # 323 tests
│
├── viewer/                    # 前端 Dashboard (Next.js)
├── docs/                      # 运维文档
│   ├── ercot-api-rate-limits.md      # ERCOT API 限流经验
│   ├── rtm-retrain-80features-report.md  # RTM 重训效果对比
│   ├── zone-spike-prediction-plan.md     # Spike Phase 0-2 规划
│   └── spike-prediction-worklog.md       # Spike 实施日志
└── .dev-state.json            # Kira 状态机
```

---

## 数据层

### 存储

| 存储 | 用途 | 大小 |
|------|------|------|
| **SQLite** (primary) | 所有数据 | 6.3 GB |
| **InfluxDB Cloud** (legacy) | Viewer 部分 API 仍在用 | 镜像 |

### 数据表

| 表名 | 行数 | 内容 | 更新频率 |
|------|------|------|---------|
| `rtm_lmp_cdr` | 11.4M | RTM 实时价格 | 每 5 分钟 |
| `rtm_lmp_hist` | 5.8M | RTM 历史价格 (2015~昨天) | 每小时 |
| `fuel_mix_hist` | 5.4M | 发电燃料组合 (含 Wind/Load) | 每 5 分钟 |
| `spike_labels` | 5.3M | 3 层 spike 标签 (SpikeEvent/LeadSpike/Regime) | 训练时生成 |
| `dam_lmp_hist` | 1.4M | DAM 历史价格 | 每日 |
| `dam_lmp_cdr` | 14.7K | DAM 实时/次日 | 每日 |
| `rt_reserves` | 1.1M | RT reserves/ORDC (2016–2025, 完整) | 训练时回填 |
| `weather_hourly` | 590K | 6 站天气 (2015–2026, 完整) | 训练时回填 |
| `wind_forecast` | 114K | 风电预测 (2022-12–2026-03, 完整) | 训练时回填 |
| `predictions` | — | 模型预测结果 | 每 5 分钟 |
| `api_keys` | — | API key + tier + 用量 | 按需 |

**⚠️ 关键**: SQLite fetcher 必须合并 `_hist` + `_cdr` 表。`_hist` 只到昨天，`_cdr` 有当天实时。

---

## ML 模型

### 模型清单

| 模型 | 算法 | 特征数 | 范围 | 核心指标 |
|------|------|--------|------|---------|
| DAM v2 | LightGBM | 80 | 15 SP | MAE 5.37–7.50 $/MWh |
| RTM | LightGBM/CatBoost | 80 | 15 SP × 3 horizons | 1h MAE 10.9–20.2 |
| Delta-Spread | CatBoost 回归+分类 | 80 | System | Sharpe=34.86, WR=94.6% |
| Spike V1 | CatBoost | 80 | 15 SP | F1≈0 (无 reserve/weather) |
| Spike V2 | LightGBM (Optuna) | 30 | 14 SP | PR-AUC +0.22 vs baseline |
| Wind | LightGBM 分位回归 | 58 | System | MAE=14.3MW |
| Load | CatBoost+LightGBM | 34 | System | MAPE=0.81% |
| BESS | LP 优化器 | N/A | DAM 输入 | 充放电时间表 |

### 特征工程

**统一管道** (`unified_features.py`, 80 features):
- 价格: lag (1h~168h), rolling mean/std/min/max, DAM-RTM spread
- 时间: hour, day_of_week, month, is_weekend, is_holiday
- 风电/负荷: fuel_mix 提取, lag + rolling
- 天气: T, WindSpeed, Humidity (6站, zone-mapped)
- Reserves: PRC, RTOLCAP, RTOFFCAP, system lambda
- Wind forecast: GEN vs STWPF deviation

**Spike 专用** (`spike_features.py`, 30 features):
- 价格波动, reserve 紧张度, 天气异常, 风电 ramp, regime

### RTM 80-Feature Retrain 状态 (2026-03-22)
- 17/45 模型改善, 18/45 回退, 10/45 持平
- 1h 预测大多改善, 24h 部分严重回退 (lcra/hubavg/houston)
- 详见 [docs/rtm-retrain-80features-report.md](docs/rtm-retrain-80features-report.md)
- TODO: 对回退模型增加 Optuna trials 或做 feature selection

---

## API 设计

**Base URL**: `http://localhost:8011`

### 认证
- `X-API-Key` header (外部请求)
- Localhost (`127.0.0.1` / `::1`) 免认证 (内部 runner 用)
- `X-Admin-Token` header (admin 端点)
- Rate limit: 按 tier (free: 100/day, pro: 10K/day, enterprise: unlimited)

### 端点分类

| 分类 | 端点 | 数量 |
|------|------|------|
| System | `/health`, `/settlement-points` | 2 |
| Model Info | `/models/{name}/info` | 7 |
| Predictions | `/predictions/{dam,rtm,spike,wind,load,bess}`, `/predict/spike/v2/all` | 8 |
| Accuracy | `/accuracy` | 1 |
| Dispatch | `/dispatch/mining/*`, `/dispatch/bess/*` | 6 |
| Admin | `/admin/keys/*` | 3 |

---

## Python 环境

| 用途 | venv | 主要包 |
|------|------|--------|
| 预测服务 | `/Users/bot/.venvs/ercot-prediction/` | fastapi, catboost, lightgbm, optuna |
| 数据采集 | `/Users/bot/.venvs/ercot-scraper/` | requests, influxdb-client |

---

## 设计决策记录

| # | 决策 | 理由 |
|---|------|------|
| 1 | SQLite 替代 InfluxDB 作为主数据源 | 本地读取快, InfluxDB Cloud 查询延迟高 |
| 2 | LightGBM/CatBoost 而非深度学习 | 表格数据, 训练快, 可解释 |
| 3 | 拆 RTM scraper 为 CDR(5min) + API(1h) | CDR 不走 API 无 429 风险, API 数据有 6h 延迟 |
| 4 | 合并 DAM 4 jobs 为 1 条 pipeline | 顺序依赖, 减少竞争 ERCOT API 配额 |
| 5 | Localhost 免 API key 认证 | prediction-runner 是内部调用 |
| 6 | 429 退避: 不用 urllib3, 自己处理 | urllib3 快速重试反而耗尽 hourly bandwidth |
| 7 | Wind/Reserve/Weather 共享数据层 | 多模型共用, 避免重复获取 |
| 8 | Spike V2 用 Optuna 调参 | PR-AUC 比 F1 更适合不平衡分类 |
| 9 | Cron 滴灌 backfill (每小时1批) | 匹配 ERCOT hourly bandwidth reset |
| 10 | LaunchAgent 而非 Docker | Mac mini 本地部署, 简单可靠 |
