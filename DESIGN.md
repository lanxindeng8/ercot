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
│   └── tests/                    # 160 tests
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

## 设计决策记录

| # | 决策 | 理由 |
|---|------|------|
| 1 | SQLite 替代 InfluxDB 作为主数据源 | InfluxDB Cloud 查询延迟高, SQLite 本地读取快 |
| 2 | LightGBM/CatBoost 而非深度学习 | 表格数据, 11年历史足够, 训练快, 可解释 |
| 3 | 单 main.py 而非微服务 | 开发速度优先, 后续可拆 router |
| 4 | LaunchAgent 而非 Docker | Mac mini 本地部署, 简单可靠 |
| 5 | 6 settlement points (HB_WEST, HB_HOUSTON, HB_NORTH, HB_SOUTH, LZ_LCRA, LZ_WEST) | 覆盖主要交易节点 |
| 6 | API key + tier 而非 OAuth | MVP 阶段简单认证足够 |
| 7 | LP 优化器做 BESS 调度 | 确定性最优, 不需要 RL 复杂度 |
