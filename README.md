# TrueFlux — ERCOT Electricity Market Intelligence

Real-time price prediction, dispatch optimization, and arbitrage signals for the ERCOT wholesale electricity market.

## What It Does

- **7 ML models** trained on 11 years of ERCOT data (2015–2026)
- **23+ API endpoints** serving predictions, dispatch signals, and risk metrics
- **Customer Dashboard** with real-time price forecasts and dispatch recommendations
- **Automated alerts** via Telegram for price spikes and mining on/off signals

## Models

| Model | Algorithm | Key Metric |
|-------|-----------|------------|
| DAM v2 | LightGBM | MAE = 5.88 $/MWh |
| RTM (1h/4h/24h) | LightGBM | MAE = 12.1 $/MWh (1h) |
| Delta-Spread | CatBoost | Sharpe = 34.86 |
| Spike Detection | CatBoost | F1 = 0.939, Recall = 1.0 |
| Wind Generation | LightGBM (quantile) | MAE = 14.3 MW |
| System Load | CatBoost + LightGBM Ensemble | MAPE = 0.81% |
| BESS Schedule | LP Optimizer | Optimal charge/discharge |

## Architecture

```
Mac mini (Nancy)
├── Scraper (LaunchAgent cron)  →  SQLite DB (primary) + InfluxDB (legacy)
├── Prediction API (FastAPI :8011, LaunchAgent)
├── Prediction Runner (5min cycle)
├── Viewer (Next.js :3000, 4 pages)
└── Telegram Alert Service
```

See [DESIGN.md](DESIGN.md) for full architecture details.
See [PLAN.md](PLAN.md) for roadmap and sprint history.

## Structure

```
ercot/
├── scraper/          # ERCOT data collection (RTM, DAM, CDR, fuel mix)
├── prediction/       # ML models + FastAPI service + dispatch signals
│   ├── src/models/       # 7 model predictors
│   ├── src/features/     # Feature engineering pipelines
│   ├── src/dispatch/     # Mining dispatch + BESS arbitrage + alerts
│   ├── src/auth/         # API key management
│   └── tests/            # 160 tests
└── viewer/           # Next.js dashboard (Dashboard/Market/Predictions/Dispatch)
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

- **11 years** of RTM/DAM prices (2015–2026), 10M+ rows
- **6 settlement points**: HB_WEST, HB_HOUSTON, HB_NORTH, HB_SOUTH, LZ_LCRA, LZ_WEST
- **Real-time pipeline**: CDR scraper every 5 minutes
- **Storage**: SQLite (primary), InfluxDB Cloud (legacy mirror)

## Status

Sprint 1–6 complete. Sprint 7 (API keys, multi-tenant, deployment) in progress.

160 tests passing. API + Runner services active via LaunchAgent.
