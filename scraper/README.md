# ERCOT Data Scraper

Automated ERCOT electricity market data scraper with dual storage (SQLite + InfluxDB).

## Architecture

```
ERCOT API → Scraper → SQLite (all data, primary storage)
                   ↘
                     InfluxDB Cloud (15 settlement points, for frontend)
```

### Data Flow

| Destination | Settlement Points | Purpose |
|-------------|-------------------|---------|
| **SQLite** | All (~1,000+) | Primary storage, historical analysis |
| **InfluxDB** | 15 (Hubs + Load Zones) | Frontend (ercot-viewer) |

### InfluxDB Settlement Points

**Hubs:** HB_BUSAVG, HB_HOUSTON, HB_HUBAVG, HB_NORTH, HB_PAN, HB_SOUTH, HB_WEST

**Load Zones:** LZ_AEN, LZ_CPS, LZ_HOUSTON, LZ_LCRA, LZ_NORTH, LZ_RAYBN, LZ_SOUTH, LZ_WEST

## Data Sources

| Scraper | API Endpoint | Frequency | Data |
|---------|--------------|-----------|------|
| RTM LMP | `/np6-788-cd/lmp_node_zone_hub` | Every 5 min | Real-time prices |
| DAM LMP | `/np4-190-cd/dam_stlmnt_pnt_prices` | Daily 2:15 PM | Day-ahead prices |

## Repository Structure

```
ercot-scraper/
├── src/
│   ├── ercot_client.py        # ERCOT API client
│   ├── influxdb_writer.py     # InfluxDB writer
│   ├── sqlite_archive.py      # SQLite writer (primary)
│   ├── scraper_rtm_lmp.py          # RTM API scraper (~6h delay)
│   ├── scraper_rtm_lmp_realtime.py # RTM CDR scraper (~5min delay)
│   ├── scraper_dam_lmp.py          # DAM scraper
│   └── cdr_scraper.py              # CDR HTML parser
├── scripts/
│   ├── run_rtm_scraper.sh     # RTM launcher
│   ├── run_dam_scraper.sh     # DAM launcher
│   ├── download_historical.py # Download historical data
│   └── fetch_dam_to_csv.py    # Export to CSV
├── historical_data/           # Historical data files
│   ├── processed/             # Processed CSV files
│   └── raw/                   # Raw ERCOT zip files
├── data/
│   └── ercot_archive.db       # SQLite database
└── logs/                      # Runtime logs
```

## Local Setup (macOS)

```bash
# Clone
git clone git@github.com:lanxindeng8/ercot-scraper.git
cd ercot-scraper

# Setup Python
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your credentials
```

## launchd Services

| Service | Schedule | Description |
|---------|----------|-------------|
| `com.trueflux.rtm-lmp-scraper` | Every 5 min | Runs API scraper (backfill) + CDR scraper (real-time) |
| `com.trueflux.dam-lmp-scraper` | Daily 2:15 PM | Day-ahead market prices (published ~1-2 PM) |

### Commands

```bash
# Check status
launchctl list | grep trueflux

# View logs
tail -f logs/rtm_stdout.log
tail -f logs/dam_stdout.log

# Restart
launchctl unload ~/Library/LaunchAgents/com.trueflux.rtm-lmp-scraper.plist
launchctl load ~/Library/LaunchAgents/com.trueflux.rtm-lmp-scraper.plist
```

## SQLite Schema

```sql
-- RTM LMP from API (finalized, ~6h delay)
CREATE TABLE rtm_lmp_api (
    time DATETIME NOT NULL,
    settlement_point TEXT NOT NULL,
    lmp REAL NOT NULL,
    energy_component REAL,
    congestion_component REAL,
    loss_component REAL,
    PRIMARY KEY (time, settlement_point)
);

-- RTM LMP from CDR (real-time, ~5min delay)
CREATE TABLE rtm_lmp_cdr (
    time DATETIME NOT NULL,
    settlement_point TEXT NOT NULL,
    lmp REAL NOT NULL,
    PRIMARY KEY (time, settlement_point)
);

-- DAM LMP (hourly)
CREATE TABLE dam_lmp (
    time DATETIME NOT NULL,
    settlement_point TEXT NOT NULL,
    settlement_point_type TEXT,
    lmp REAL NOT NULL,
    PRIMARY KEY (time, settlement_point)
);
```

## Environment Variables

```bash
# ERCOT API
ERCOT_API_USERNAME=your_email@example.com
ERCOT_API_PASSWORD=your_password
ERCOT_PUBLIC_API_SUBSCRIPTION_KEY=your_key

# InfluxDB
INFLUXDB_URL=https://us-east-1-1.aws.cloud2.influxdata.com
INFLUXDB_TOKEN=your_token
INFLUXDB_ORG=your_org_id
INFLUXDB_BUCKET=ercot
```

## Cost

**$0/month** - Uses free tiers:
- InfluxDB Cloud: 30-day retention
- GitHub: Public repository
- SQLite: Local storage

## License

MIT
