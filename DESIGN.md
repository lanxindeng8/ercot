# TrueFlux — Architecture Design Document

> Last updated: 2026-03-25

---

## System Architecture

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
│  │  7 models | 23+ endpoints | API key auth (localhost exempt) │   │
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

## LaunchAgent Architecture (2026-03-23 Refactor)

| Job | Label | Frequency | ERCOT API? | Description |
|-----|-------|-----------|-----------|-------------|
| RTM CDR | `com.trueflux.rtm-lmp-cdr` | 5min | ❌ HTML scrape | Real-time RTM data |
| RTM API | `com.trueflux.rtm-lmp-api` | 1h | ✅ | Historical backfill (6h delay) |
| DAM Pipeline | `com.trueflux.dam-pipeline` | Daily 14:00 | ✅ (1x) | predictions → fetch → CDR → Telegram |
| Prediction API | `com.trueflux.prediction-api` | Always-on | ❌ | FastAPI :8011 |
| Prediction Runner | `com.trueflux.prediction-runner` | 5min | ❌ | Calls localhost:8011 |
| Telegram Summary | `com.trueflux.telegram-lmp-summary` | Daily 06:30 | ❌ | Reads InfluxDB |
| BTC Monitor | `com.trueflux.btc-price-monitor` | 5min | ❌ | PolyManager project |

**Only 2 ERCOT API callers**, sharing the same subscription key's hourly bandwidth quota.
See [docs/ercot-api-rate-limits.md](docs/ercot-api-rate-limits.md) for details.

**Deprecated (no longer loaded)**:
- `rtm-lmp-scraper` → Split into `rtm-lmp-cdr` + `rtm-lmp-api`
- `dam-lmp-scraper`, `dam-lmp-cdr-scraper`, `dam-predictions`, `telegram-dam-schedule` → Merged into `dam-pipeline`

---

## Directory Structure

```
ercot/
├── scraper/                   # Data collection
│   ├── src/
│   │   ├── ercot_client.py        # ERCOT API client (429 backoff: 10s initial, 5min cap, jitter)
│   │   ├── scraper_rtm_lmp.py     # RTM API scraper (historical, 6h delay)
│   │   ├── scraper_rtm_lmp_realtime.py  # RTM CDR scraper (real-time)
│   │   ├── scraper_dam_lmp.py     # DAM API scraper
│   │   ├── scraper_dam_lmp_cdr.py # DAM CDR HTML scraper
│   │   ├── cdr_scraper.py         # CDR generic scraper
│   │   ├── sqlite_archive.py      # SQLite writer
│   │   └── influxdb_writer.py     # InfluxDB writer (legacy)
│   └── scripts/
│       ├── run_rtm_cdr_scraper.sh     # CDR 5min (no API)
│       ├── run_rtm_api_scraper.sh     # API 1h (rate limited)
│       └── run_dam_pipeline.sh        # DAM full pipeline
│
├── prediction/                # Prediction service
│   ├── src/
│   │   ├── main.py               # FastAPI (1882 lines, 23+ endpoints)
│   │   ├── config.py
│   │   ├── models/               # Model inference
│   │   │   ├── dam_v2_predictor.py
│   │   │   ├── rtm_predictor.py
│   │   │   ├── delta_spread.py
│   │   │   ├── spike_predictor.py
│   │   │   ├── spike_v2_predictor.py   # Phase 1: 14 SP, Optuna tuned
│   │   │   ├── wind_predictor.py
│   │   │   ├── load_predictor.py
│   │   │   └── bess_predictor.py
│   │   ├── features/
│   │   │   ├── unified_features.py     # 80 features unified pipeline
│   │   │   ├── spike_features.py       # 30 spike-specific features
│   │   │   └── dam_features_v2.py
│   │   ├── data/                 # Shared data layer
│   │   │   ├── weather/              # Open-Meteo weather (6 stations, 2015-2026)
│   │   │   └── ercot/                # ERCOT market data
│   │   │       ├── reserves.py           # RT reserves/ORDC (NP6-792-ER)
│   │   │       └── wind_forecast.py      # Wind forecast (NP4-732-CD)
│   │   ├── dispatch/
│   │   │   ├── mining_dispatch.py
│   │   │   ├── bess_signals.py
│   │   │   └── alert_service.py
│   │   └── auth/
│   │       └── api_keys.py          # API key + tier + rate limit
│   ├── scripts/
│   │   ├── run_predictions.py        # Prediction loop runner
│   │   ├── retrain_all_v2.py         # Full model retrain (80 features)
│   │   ├── train_spike.py            # Spike baseline training
│   │   ├── tune_spike.py             # Spike Optuna hyperparameter tuning
│   │   ├── fetch_reserves.py         # RT reserves backfill
│   │   ├── fetch_wind_forecast.py    # Wind forecast backfill
│   │   ├── fetch_wind_one_month.py   # Cron single-month incremental fetch
│   │   └── manage_keys.py           # API key CLI
│   ├── models/                   # Training artifacts
│   │   ├── dam_v2/                   # 15 SP LightGBM
│   │   ├── rtm/                      # 15 SP × 3 horizons
│   │   ├── spike/                    # 14 SP baseline
│   │   └── spike_tuned/              # 14 SP Optuna tuned
│   └── tests/                    # 323 tests
│
├── viewer/                    # Frontend Dashboard (Next.js)
├── docs/                      # Operations documentation
│   ├── ercot-api-rate-limits.md      # ERCOT API rate limiting lessons
│   ├── rtm-retrain-80features-report.md  # RTM retrain performance comparison
│   ├── zone-spike-prediction-plan.md     # Spike Phase 0-2 plan
│   └── spike-prediction-worklog.md       # Spike implementation log
└── .dev-state.json            # Kira state machine
```

---

## Data Layer

### Storage

| Storage | Purpose | Size |
|---------|---------|------|
| **SQLite** (primary) | All data | 6.3 GB |
| **InfluxDB Cloud** (legacy) | Some Viewer APIs still use it | Mirror |

### Data Tables

| Table | Rows | Content | Update Frequency |
|-------|------|---------|-----------------|
| `rtm_lmp_cdr` | 11.4M | RTM real-time prices | Every 5 minutes |
| `rtm_lmp_hist` | 5.8M | RTM historical prices (2015~yesterday) | Hourly |
| `fuel_mix_hist` | 5.4M | Generation fuel mix (incl. Wind/Load) | Every 5 minutes |
| `spike_labels` | 5.3M | 3-tier spike labels (SpikeEvent/LeadSpike/Regime) | Generated during training |
| `dam_lmp_hist` | 1.4M | DAM historical prices | Daily |
| `dam_lmp_cdr` | 14.7K | DAM real-time/next-day | Daily |
| `rt_reserves` | 1.1M | RT reserves/ORDC (2016–2025, complete) | Backfilled during training |
| `weather_hourly` | 590K | 6-station weather (2015–2026, complete) | Backfilled during training |
| `wind_forecast` | 114K | Wind power forecast (2022-12–2026-03, complete) | Backfilled during training |
| `predictions` | — | Model prediction results | Every 5 minutes |
| `api_keys` | — | API key + tier + usage | On demand |

**⚠️ Important**: SQLite fetcher must merge `_hist` + `_cdr` tables. `_hist` only goes up to yesterday; `_cdr` has today's real-time data.

---

## ML Models

### Model Inventory

| Model | Algorithm | Features | Scope | Key Metric |
|-------|-----------|----------|-------|------------|
| DAM v2 | LightGBM | 80 | 15 SP | MAE 5.37–7.50 $/MWh |
| RTM | LightGBM/CatBoost | 80 | 15 SP × 3 horizons | 1h MAE 10.9–20.2 |
| Delta-Spread | CatBoost regression+classification | 80 | System | Sharpe=34.86, WR=94.6% |
| Spike V1 | CatBoost | 80 | 15 SP | F1≈0 (no reserve/weather) |
| Spike V2 | LightGBM (Optuna) | 30 | 14 SP | PR-AUC +0.22 vs baseline |
| Wind | LightGBM quantile regression | 58 | System | MAE=14.3MW |
| Load | CatBoost+LightGBM | 34 | System | MAPE=0.81% |
| BESS | LP optimizer | N/A | DAM input | Charge/discharge schedule |

### Feature Engineering

**Unified pipeline** (`unified_features.py`, 80 features):
- Price: lag (1h~168h), rolling mean/std/min/max, DAM-RTM spread
- Time: hour, day_of_week, month, is_weekend, is_holiday
- Wind/Load: extracted from fuel_mix, lag + rolling
- Weather: T, WindSpeed, Humidity (6 stations, zone-mapped)
- Reserves: PRC, RTOLCAP, RTOFFCAP, system lambda
- Wind forecast: GEN vs STWPF deviation

**Spike-specific** (`spike_features.py`, 30 features):
- Price volatility, reserve tightness, weather anomalies, wind ramp, regime

### RTM 80-Feature Retrain Status (2026-03-22)
- 17/45 models improved, 18/45 regressed, 10/45 unchanged
- 1h predictions mostly improved, 24h partially severely regressed (lcra/hubavg/houston)
- See [docs/rtm-retrain-80features-report.md](docs/rtm-retrain-80features-report.md) for details
- TODO: Add Optuna trials or perform feature selection for regressed models

---

## API Design

**Base URL**: `http://localhost:8011`

### Authentication
- `X-API-Key` header (external requests)
- Localhost (`127.0.0.1` / `::1`) authentication exempt (used by internal runner)
- `X-Admin-Token` header (admin endpoints)
- Rate limit: by tier (free: 100/day, pro: 10K/day, enterprise: unlimited)

### Endpoint Categories

| Category | Endpoints | Count |
|----------|-----------|-------|
| System | `/health`, `/settlement-points` | 2 |
| Model Info | `/models/{name}/info` | 7 |
| Predictions | `/predictions/{dam,rtm,spike,wind,load,bess}`, `/predict/spike/v2/all` | 8 |
| Accuracy | `/accuracy` | 1 |
| Dispatch | `/dispatch/mining/*`, `/dispatch/bess/*` | 6 |
| Admin | `/admin/keys/*` | 3 |

---

## Python Environments

| Purpose | venv | Key Packages |
|---------|------|--------------|
| Prediction service | `/Users/bot/.venvs/ercot-prediction/` | fastapi, catboost, lightgbm, optuna |
| Data collection | `/Users/bot/.venvs/ercot-scraper/` | requests, influxdb-client |

---

## Design Decision Log

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | SQLite replaces InfluxDB as primary data source | Fast local reads; InfluxDB Cloud has high query latency |
| 2 | LightGBM/CatBoost instead of deep learning | Tabular data, fast training, interpretable |
| 3 | Split RTM scraper into CDR(5min) + API(1h) | CDR does not use API so no 429 risk; API data has 6h delay |
| 4 | Merge 4 DAM jobs into 1 pipeline | Sequential dependencies; reduces contention for ERCOT API quota |
| 5 | Localhost exempt from API key authentication | prediction-runner is an internal call |
| 6 | 429 backoff: custom handling instead of urllib3 | urllib3 fast retries actually exhaust hourly bandwidth |
| 7 | Wind/Reserve/Weather shared data layer | Used by multiple models; avoids redundant fetching |
| 8 | Spike V2 uses Optuna for hyperparameter tuning | PR-AUC is more suitable than F1 for imbalanced classification |
| 9 | Cron drip-feed backfill (1 batch per hour) | Matches ERCOT hourly bandwidth reset |
| 10 | LaunchAgent instead of Docker | Mac mini local deployment, simple and reliable |
