# TrueFlux — Project Roadmap

> ERCOT Electricity Market Forecasting + Intelligent Dispatch Platform
> Last updated: 2026-03-25

---

## Overview

TrueFlux starts from ERCOT real-time/DAM data, trains ML models to forecast prices, wind, load, and price spikes, and generates mine start/stop signals and battery arbitrage strategies accordingly. Ultimate goal: SaaS signal service.

## Progress

| Sprint | Content | Status | Date |
|--------|---------|--------|------|
| 1 | Infrastructure: data audit, gap backfill, training pipeline | ✅ Done | 03-17 |
| 2 | Model training: DAM v2, RTM, Delta-Spread, Spike + quality fixes | ✅ Done | 03-17~18 |
| 3 | Productionization: Prediction API v3→v4, Dashboard | ✅ Done | 03-18 |
| 4 | Wind/Load/BESS models + full-stack integration | ✅ Done | 03-18 |
| 5 | Deployment: LaunchAgent, SQLite migration, accuracy tracking | ✅ Done | 03-18 |
| 6 | Commercial MVP: mine dispatch, BESS arbitrage, customer Dashboard | ✅ Done | 03-18 |
| 7.1 | API Key management + Rate Limiting | ✅ Done | 03-18 |
| Phase 0 | Spike data acquisition infrastructure (weather/reserves/wind) | ✅ Done | 03-19 |
| Phase 1 | Zone-level Spike V2 (labels + features + 14 SP models + Optuna + API) | ✅ Done | 03-19~22 |
| Ops | RTM retrain 80 features (15 SP × 3 horizons) | ✅ Done | 03-22 |
| Ops | Data backfill (reserves 2016-2025, wind 2022-2026) | ✅ Done | 03-22~23 |
| Ops | LaunchAgent refactor (9→7 jobs, API/CDR separation) | ✅ Done | 03-23 |
| Ops | ERCOT 429 fix (backoff strategy, documentation) | ✅ Done | 03-23 |

---

## Current To-Do

### High Priority

| # | Task | Description |
|---|------|-------------|
| 1 | **RTM model regression fix** | After 80-feature retrain, lcra/hubavg/houston 24h MAE doubled. Plan: increase Optuna trials (50+) or horizon-aware feature selection |
| 2 | **Validate new LaunchAgent architecture** | dam-pipeline first run pending validation (daily at 14:00), rtm-lmp-api 429 backoff effectiveness to be observed |

### Medium Priority

| # | Task | Description |
|---|------|-------------|
| 3 | **Phase 2: multi-horizon spike prediction** | 15min/30min/4h spike prediction, leveraging completed data infrastructure |
| 4 | **Real-time pipeline optimization** | CDR → zone aggregation, improve real-time feature quality |

### Low Priority / On Hold

| # | Task | Description |
|---|------|-------------|
| 5 | Sprint 7.2 multi-tenant isolation | Isolate data views by API key |
| 6 | Sprint 7.3 production deployment | HTTPS, domain, public network |
| 7 | Sprint 8 model enhancement | Auto-retrain, drift detection, A/B testing |
| 8 | Sprint 9 customer acquisition | Landing page, Stripe |

---

## Key Metrics

| Model | Core Metric | Value | Updated |
|-------|-------------|-------|---------|
| DAM v2 | MAE (best/worst SP) | 5.37 / 7.50 | 03-19 |
| RTM 1h | MAE (best/worst SP) | 10.9 / 20.2 | 03-22 |
| Delta-Spread | Sharpe | 34.86 | 03-18 |
| Spike V2 | PR-AUC improvement | +0.22 avg | 03-22 |
| Wind | MAE | 14.3 MW | 03-18 |
| Load | MAPE | 0.81% | 03-18 |

## Data Infrastructure

| Data | Rows | Coverage | Status |
|------|------|----------|--------|
| RTM prices | 17.2M | 2015–real-time | ✅ Complete |
| DAM prices | 1.4M | 2015–real-time | ✅ Complete |
| RT Reserves/ORDC | 1.1M | 2016–2025 | ✅ Complete |
| Weather (6 stations) | 590K | 2015–2026 | ✅ Complete |
| Wind Forecast | 114K | 2022-12–2026-03 | ✅ Complete |
| Spike Labels | 5.3M | 2015–2026, 15 SP | ✅ Complete |
| Tests | 323 | — | ✅ All passing |

## Technical Debt

- [x] ~~main.py 1882 lines~~ → Split into 119-line main.py + 6 router files + schemas.py + helpers.py (`807a80d`)
- [x] ~~RTM 24h model regression~~ → 50 trials fixed 5/6 regressions (`5fadea8`). hb_houston/24h still 19.07 vs original 12.97 (needs feature analysis)
- [x] ~~SQLite missing UNIQUE constraints~~ → Added UNIQUE index to all 4 tables, deduplicated 26,743 rows
- [x] ~~`data/__init__.py`'s `__getattr__`~~ → Removed (`47357a3`)
- [x] ~~Deprecated LaunchAgent plist~~ → 5 old plist files moved to .archived/
- [x] ~~InfluxDB migration~~ → Not migrating. InfluxDB Cloud free tier, time-series queries better suited for frontend than SQLite, low dual-write cost
- [ ] hb_houston/24h MAE regression (12.97→19.07), may need horizon-aware feature selection
