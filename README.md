# ercot — ERCOT Data Pipeline Monorepo

ERCOT electricity market data pipeline: scraper + visualization.

## Structure

```
ercot/
├── scraper/   # Python scraper (runs as cron on Mac mini)
└── viewer/    # Next.js frontend (deployed on Vercel)
```

## Deployment

### scraper — Mac mini cron

```bash
cd scraper
pip install -r requirements.txt
# Add to crontab:
# */5 * * * * /path/to/venv/bin/python /path/to/ercot/scraper/scripts/run_all.py
```

### viewer — Vercel

Deployed via Vercel with Root Directory set to `viewer/`.

Environment variables required:
- `INFLUXDB_URL`
- `INFLUXDB_TOKEN`
- `INFLUXDB_DATABASE`

```bash
cd viewer
vercel --prod
```

## Data Flow

```
Mac mini (cron)        InfluxDB Cloud         Vercel
┌─────────────┐        ┌─────────────┐        ┌─────────────┐
│  scraper/   │ write  │             │ query  │   viewer/   │
│  *.py       │───────▶│ ercot bucket│◀───────│  Next.js    │
└─────────────┘        └─────────────┘        └─────────────┘
```
