# Zone-Level Spike Prediction — Work Log

> Step-by-step implementation record, each step includes: objective, design, results, lessons learned

---

## Phase 0: Data Acquisition Infrastructure

### Step 0.1: Shared Data Layer Directory Structure

**Objective**: Create `prediction/src/data/weather/` and `prediction/src/data/ercot/` directory structure

**Current state**: Weather data code is in `prediction/models/wind/src/data/hrrr_client.py` (coupled within the wind model)

**Design**:
```
prediction/src/data/
├── __init__.py
├── weather/
│   ├── __init__.py
│   ├── openmeteo_client.py    # Open-Meteo Archive API client
│   ├── stations.py            # Zone → city coordinate mapping
│   └── zone_weather.py        # T_anom, ΔT, Wind Chill, Cold Front calculations
└── ercot/
    ├── __init__.py
    ├── wind_forecast.py       # NP4-732-CD: GEN + STWPF (forecast vs actual)
    └── reserves.py            # NP6-792-ER: PRC, ORDC price adders
```

**Notes**:
- `hrrr_client.py` will not be moved for now (wind model still uses it), refactor later
- Modules under `ercot/` reuse authentication logic from `scraper/src/ercot_client.py`
- Credentials are read from environment variables (already set in LaunchAgent)

---

### Step 0.2: Open-Meteo Weather Data Acquisition

**Objective**: Acquire hourly weather data for 6 cities from 2015-2026, store in SQLite

**API**: `https://archive-api.open-meteo.com/v1/archive`

**Validation results (2026-03-19)**:
- ✅ Free, no key required, REST JSON
- ✅ San Antonio 2025-12-14 cold front confirmed (18.6→6.3°C)
- ✅ 1 city × 1 year ≈ 481 KB JSON
- ⚠️ `wind_speed_80m` is all null, need to use ERA5 endpoint or only use 10m
- 6 cities × 11 years → SQLite ≈ 16 MB

**City coordinate mapping**:
| Zone | City | Latitude | Longitude |
|------|------|------|------|
| LZ_CPS | San Antonio | 29.42 | -98.49 |
| LZ_WEST | Midland/Odessa | 31.95 | -102.18 |
| LZ_HOUSTON | Houston | 29.76 | -95.37 |
| HB_NORTH / LZ_NORTH | Dallas/Fort Worth | 32.78 | -96.80 |
| HB_SOUTH / LZ_SOUTH | Corpus Christi | 27.80 | -97.40 |
| System (HB_BUSAVG/HUBAVG) | Austin | 30.27 | -97.74 |

**Variables**:
- `temperature_2m` (°C) — Core
- `wind_speed_10m` (km/h)
- `wind_direction_10m` (°)
- `relative_humidity_2m` (%)
- `surface_pressure` (hPa)
- `dew_point_2m` (°C) — Needed for Wind Chill calculation

**SQLite table design**:
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

**Acquisition strategy**:
- Open-Meteo limits single requests to approximately 1 year
- 11 requests per city (one per year for 2015~2025 + 2026 YTD)
- 6 cities × 12 requests = 72 HTTP requests
- Add sleep(1) to avoid rate limits → ~2 minutes to complete

**Test plan**:
1. First pull 1 city × 1 year to verify data completeness
2. Check null value ratio
3. Confirm timezone alignment (America/Chicago)
4. Full pull
5. Write SQLite import

---

### Step 0.3: ERCOT Wind Forecast Acquisition (NP4-732-CD)

**Objective**: Acquire wind forecast (STWPF) vs actual (GEN), store in SQLite

**API**: `https://api.ercot.com/api/public-reports/np4-732-cd/wpp_hrly_avrg_actl_fcast`

**Validation results (2026-03-19)**:
- ✅ Data API (not archive), same as existing LMP scraper
- ✅ 21 fields: postedDatetime, deliveryDate, hourEnding, + 4 metrics × 4 regions (SystemWide, SouthHouston, West, North) + HSL + DST
- ✅ Authentication: Reuse existing ercot_client.py + LaunchAgent credentials
- ⚠️ 2025-12-14 query returned 5184 rows (multiple posted versions)

**Field mapping** (per region):
- `genXxx` — Actual generation MW
- `STWPFXxx` — Short-Term Wind Power Forecast MW
- `WGRPPXxx` — Wind Generation Resource Production Potential MW
- `COPHSLXxx` — Current Operating Plan HSL MW

**Core feature calculations**:
- `wind_forecast_error = GEN - STWPF` (surprise: negative value = actual less than forecast)
- `wind_capacity_factor = GEN / COPHSL`
- `wind_surprise_pct = (GEN - STWPF) / STWPF * 100`

**SQLite table design**:
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

**Acquisition strategy**:
- Use existing `ercot_client.fetch_paginated_data()` method
- Request by month using deliveryDateFrom/To
- ~8760 hours × 4 regions × multiple posted versions = large data volume
- First pull December 2025 (one month) to test volume

---

### Step 0.4: RT Reserves / ORDC Acquisition (NP6-792-ER)

**Objective**: Acquire RT reserve margin (PRC) + ORDC price adders historical data

**API**: Archive download (not a data API)
- Listing: `https://api.ercot.com/api/public-reports/archive/np6-792-er`
- Download: `https://api.ercot.com/api/public-reports/archive/np6-792-er?download={docId}`

**Validation results (2026-03-19)**:
- ✅ Annual XLSX files, one sheet per month
- ✅ 33 columns: Batch ID, SCED Timestamp, PRC, System Lambda, RTOLCAP, RTOFFCAP, RTORPA, RTORDPA, RTOLHSL, RTBP, etc.
- ✅ ~9K rows/month, ~108K rows/year
- ✅ Data available from 2017 onward (8 years)
- ⚠️ Header is at row 8 (empty rows before it)
- ⚠️ 2025 HIST_RT_SCED_PRC_ADDR file has empty sheets; RTM_ORDC version has data

**Available file list**:
- RTM_ORDC_REL_DPLY_PRC_ADDR_RSRV_2024 (docId: 1065495488) — 18 MB xlsx
- RTM_ORDC_REL_DPLY_PRC_ADDR_RSRV_2023 (docId: 969827183)
- RTM_ORDC_REL_DPLY_PRC_ADDR_RSRV_2022 (docId: 899479048)
- RTM_ORDC_REL_DPLY_PRC_ADDR_RSRV_2021 (docId: 814938847)
- RTM_ORDC_REL_DPLY_PRC_ADDR_RSRV_2020 (docId: 751366904)
- RTM_ORDC_REL_DPLY_PRC_ADDR_RSRV_2019 (docId: 694452063)
- RTM_ORDC_REL_DPLY_PRC_ADDR_RSRV_2017 (docId: 644817015)
- HIST_RT_SCED_PRC_ADDR_2026 (docId: 1204109813) — New format
- HIST_RT_SCED_PRC_ADDR_2025 (docId: 1180319793) — New format but sheets are empty?

**SQLite table design**:
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

**Acquisition strategy**:
1. List all available files via archive API
2. Download zip per year → Extract xlsx
3. Parse each month's sheet (header=row 8)
4. Write to SQLite
5. Estimated 8 years × 18MB = ~144 MB download, ~230 MB SQLite after parsing

---

## Execution Log

### Phase 0: Data Acquisition Infrastructure ✅ (2026-03-19)

| Step | Content | Status | Commit |
|------|------|------|--------|
| 0.1 | Directory structure | ✅ Complete | `57aec0a` |
| 0.2 | Open-Meteo weather data | ✅ Complete | 589,680 rows, 6 stations × 2015-2026 |
| 0.3 | Wind Forecast (NP4-732-CD) | ✅ Complete | 113,924 rows, 2022-12 → 2026-03 |
| 0.4 | RT Reserves (NP6-792-ER) | ✅ Complete | 1,056,444 rows, 2016-2025 continuous |

### Phase 1: Zone-Level Spike V2 ✅ (2026-03-19~22)

| Step | Content | Status | Notes |
|------|------|------|------|
| 1.1 | Three-layer labels (SpikeEvent/LeadSpike/Regime) | ✅ Complete | 5.3M labels, validated with 2025-12-14 LZ_CPS case |
| 1.2 | Feature engineering v2 | ✅ Complete | 30 spike-specific features |
| 1.3 | LightGBM baseline training | ✅ Complete | 14 SP models, avg ROC-AUC 0.93, 97% event recall |
| 1.4 | Optuna tuning | ✅ Complete | 50 trials/SP, avg PR-AUC +0.22 over baseline |
| 1.5 | API integration | ✅ Complete | 3 endpoints + `/predict/spike/v2/all` |
| 1.6 | Predictor → tuned models | ✅ Complete | Commit `7403672` |

### Ops: RTM Retrain + Infrastructure (2026-03-22~23)

| Task | Status | Notes |
|------|------|------|
| RTM retrain 80 features | ✅ Complete | 15 SP × 3 horizons. 17 improved, 18 regressed. See `docs/rtm-retrain-80features-report.md` |
| Localhost auth bypass | ✅ Complete | prediction-runner 401 fix |
| Data backfill (reserves) | ✅ Complete | 2020-2023 backfilled, 1.1M rows total |
| Data backfill (wind) | ✅ Complete | Cron drip-feed completed, 40 months / 0 gaps |
| ERCOT 429 fix | ✅ Complete | urllib3 removed 429, custom backoff 10s+jitter, 5min cap |
| LaunchAgent refactoring | ✅ Complete | 9→7 jobs, RTM CDR/API split, DAM pipeline merged |

### Lessons Learned

1. **ERCOT archive filename ≠ data year** — `_2023.xlsx` contains 2022 data. Must verify timestamps
2. **urllib3 Retry on 429 is a disaster** — Rapid retries exhaust hourly bandwidth. Handle 429 only with custom backoff
3. **ERCOT limits hourly bandwidth** — Not request frequency. Quota resets every hour
4. **80 features may introduce noise for 24h predictions** — 1h improved but 24h regressed. Need horizon-aware feature selection
5. **Concurrent fetches trigger 429 instantly** — Serial + cron drip-feed (one batch per hour) is the only reliable approach
