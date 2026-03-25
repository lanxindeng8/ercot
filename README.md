# TrueFlux — ERCOT Electricity Market Intelligence

Real-time price prediction, dispatch optimization, and arbitrage signals for the ERCOT wholesale electricity market.

## What It Does

- **7 ML models** trained on 11 years of ERCOT data (2015–2026), 80 unified features
- **23+ API endpoints** serving predictions, dispatch signals, and risk metrics
- **15 settlement points**: all HB_* hubs + LZ_* load zones
- **Customer Dashboard** with real-time price forecasts and dispatch recommendations
- **Automated alerts** via Telegram for price spikes and mining on/off signals

## Models

| Model | Algorithm | Scope | Key Metric |
|-------|-----------|-------|------------|
| DAM v2 | LightGBM | 15 SP | MAE 5.37–7.50 $/MWh |
| RTM (1h/4h/24h) | LightGBM/CatBoost | 15 SP × 3 horizons | 1h MAE 10.9–20.2 $/MWh |
| Delta-Spread | CatBoost | Regression + Binary + Multiclass | Sharpe 34.86, WR 94.6% |
| Spike V2 | LightGBM (Optuna tuned) | 14 SP, 30 features | PR-AUC +0.22 over baseline |
| Wind Generation | LightGBM (quantile) | System-wide | MAE 14.3 MW |
| System Load | CatBoost + LightGBM Ensemble | System-wide | MAPE 0.81% |
| BESS Schedule | LP Optimizer | Charge/discharge | Risk-adjusted arbitrage |

## Architecture

```
Mac mini (Nancy)
├── Scraper Layer
│   ├── rtm-lmp-cdr    (5min)   → CDR HTML scrape, no ERCOT API
│   ├── rtm-lmp-api    (1h)     → ERCOT API historical backfill
│   └── dam-pipeline   (14:00)  → predictions → API fetch → CDR fetch → Telegram
├── Prediction API     (FastAPI :8011, always-on)
├── Prediction Runner  (5min)   → calls local API, stores to SQLite
├── Viewer             (Next.js :3000, 4 pages)
└── Telegram Alerts    (daily LMP summary at 06:30)
```

See [DESIGN.md](DESIGN.md) for full architecture details.
See [PLAN.md](PLAN.md) for roadmap and sprint history.
See [docs/](docs/) for operational documentation.

## Structure

```
ercot/
├── scraper/          # ERCOT data collection (RTM, DAM, CDR)
├── prediction/       # ML models + FastAPI service + dispatch signals
│   ├── src/
│   │   ├── models/       # 7 model predictors
│   │   ├── features/     # Feature engineering (unified 80-feature pipeline)
│   │   ├── data/         # Shared data layer (weather, ERCOT reserves/wind)
│   │   ├── dispatch/     # Mining dispatch + BESS arbitrage + alerts
│   │   └── auth/         # API key management
│   └── tests/            # 323 tests
├── viewer/           # Next.js dashboard (Dashboard/Market/Predictions/Dispatch)
└── docs/             # Operational docs (rate limits, retrain reports, etc.)
```

## Quick Start

```bash
# API
cd prediction
source /Users/bot/.venvs/ercot-prediction/bin/activate
uvicorn src.main:app --host 0.0.0.0 --port 8011

# Viewer
cd viewer
npm run dev
```

API docs: http://localhost:8011/docs

## Data

- **11 years** of RTM/DAM prices (2015–2026)
- **15 settlement points**: HB_BUSAVG, HB_HOUSTON, HB_HUBAVG, HB_NORTH, HB_PAN, HB_SOUTH, HB_WEST, LZ_AEN, LZ_CPS, LZ_HOUSTON, LZ_LCRA, LZ_NORTH, LZ_RAYBN, LZ_SOUTH, LZ_WEST
- **SQLite DB** (6.3 GB): 25M+ rows across RTM, DAM, weather, wind forecast, reserves, spike labels
- **Real-time pipeline**: CDR scraper every 5 minutes (no ERCOT API dependency)
- **Historical pipeline**: ERCOT API backfill every hour (subject to hourly bandwidth limit)

| Table | Rows | Description |
|-------|------|-------------|
| rtm_lmp_cdr | 11.4M | RTM real-time prices |
| rtm_lmp_hist | 5.8M | RTM historical prices |
| fuel_mix_hist | 5.4M | Generation fuel mix |
| spike_labels | 5.3M | 3-layer spike event labels |
| dam_lmp_hist | 1.4M | DAM historical prices |
| rt_reserves | 1.1M | RT reserves/ORDC (2016–2025) |
| weather_hourly | 590K | 6-station weather (2015–2026) |
| wind_forecast | 114K | Wind gen forecast (2022–2026) |

## Status

- Sprint 1–6 complete. Sprint 7.1 (API keys) complete. Sprint 7.2+ paused.
- Phase 1 Spike V2 complete (zone-level, 14 SP, Optuna tuned).
- 323 tests passing. All services active via LaunchAgent.
- See [docs/ercot-api-rate-limits.md](docs/ercot-api-rate-limits.md) for ERCOT API operational notes.
