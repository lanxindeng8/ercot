# Zone-Level RTM Spike Prediction — Research and Implementation Plan

> Based on Lanxin's discussion document (2025-12-14 evening peak LZ_CPS spike case)
> Research date: 2026-03-19
> Last updated: 2026-03-25
>
> **Status**: Phase 0 ✅ + Phase 1 ✅ complete. GBM baseline + Optuna tuned models deployed.

---

## 1. Current State Audit: What We Have and What We're Missing

### ✅ Existing Data

| Data | Table/Source | Granularity | Time Range | Coverage |
|------|---------|------|---------|------|
| DAM LMP | `dam_lmp_hist` | Hourly | 2015-01 ~ 2026-03 | 15 SPs |
| RTM LMP | `rtm_lmp_hist` | 15min (4 intervals/hr) | 2015-01 ~ 2026-03 | 15 SPs |
| RTM LMP + Components | `rtm_lmp_api` | 5min | 2026-02-09 ~ now | 1092 SPs (including resource nodes) |
| DAM Ancillary Services | `dam_asmcpc_hist` | Hourly | 2015-01 ~ 2026-02 | RegDn/Up, RRS, NSPIN, ECRS |
| Fuel Mix | `fuel_mix_hist` | 15min | 2007-01 ~ 2024-12 | Wind, Solar, Gas, Gas-CC, Coal, Nuclear, Hydro... |

### Key Observations

1. **RTM congestion component has only 5 weeks of data** (`rtm_lmp_api` started from 2026-02-09). The zone-vs-hub spread and its derivatives required by the document can be computed from `rtm_lmp_hist` (LMP differences), but it is not possible to separate the energy/congestion/loss components.
2. **Fuel mix data ends at 2024-12-31**, missing the most recent 3 months. Need to resume or backfill scraping.
3. **2025-12-14 case day**: RTM hist has data, DAM hist has data. LZ_CPS spike confirmed (Hour 21: $686.2 max, $630.8 avg). Houston was only $97.7 during the same period. Regional characteristics are very prominent.

### Data Acquisition Status (2026-03-25 Update)

| Document Requirement | ERCOT Data Product | Status | Notes |
|---------|---------------|------|------|
| **RT Reserve Margin (PRC)** | NP6-792-ER | ✅ Acquired | 1,056,444 rows, 2016-2025 continuous |
| **ORDC Price Adders** | NP6-792-ER | ✅ Acquired | Same as above (same report) |
| **Zone-level weather data** | Open-Meteo Archive | ✅ Acquired | 589,680 rows, 6 stations, 2015-2026 |
| **Wind Forecast (STWPF)** | NP4-732-CD | ✅ Acquired | 113,924 rows, 2022-12 → 2026-03 |
| **Wind Forecast Error** | GEN vs STWPF | ✅ Computable | Data is complete |
| **Net Load** | fuel_mix calculation | ✅ Computable | Total Load - Wind - Solar |
| **Binding Constraints** | NP6-86-CD | ❌ Abandoned | Data volume too large (100K+ files/year), using zone-hub spread as approximation |
| **System Load Forecast** | NP3-565-CD | ⏳ Not yet acquired | Will scrape when needed in Phase 2 |

### Model Architecture Status (2026-03-25 Update)

| Document Requirement | Current Status |
|---------|---------|
| CfC / LTC / Neural ODE | ⏳ Phase 2+. Currently using LightGBM baseline |
| Zone-level Regime Switching | ✅ Three-layer labels implemented (SpikeEvent/LeadSpike/Regime) |
| Three-layer labels | ✅ Complete. 5.3M labels, validated with 2025-12-14 case |
| Zone-level Spike Models | ✅ 14 SP, Optuna tuned, PR-AUC +0.22 |
| 5min prediction frequency | ⏳ Training uses 15min, production can use 5min (pending Phase 2) |

---

## 2. Data Acquisition Plan

### Phase 0: Complete Existing Data (1 day)

1. **Fuel Mix 2025-01 ~ 2026-03**: Backfill 15 months of missing data. `fuel_mix_hist` stopped at 2024-12-31.
2. **RTM congestion components history**: `rtm_lmp_api` has only 5 weeks (2026-02-09~now). ERCOT does not provide earlier 5min congestion components via public API. **Alternative**: Use LMP differences (LZ - Hub) from `rtm_lmp_hist` to approximate congestion signal.

### Phase 1: New Data Source Scrapers (3-5 days)

Each data source requires: Understand API format → Write scraper → Historical backfill → Store in SQLite → Write quality checks.

**Small-scale validation results (2026-03-19):**

| Priority | Data Source | ERCOT Product | Validation Status | Data Volume (11yr) | Effort |
|--------|--------|--------------|---------|-------------|-------|
| P0 | **RT Reserves + ORDC** | NP6-792-ER | ✅ **Validated** — MIS anonymous download XLSX, 33 columns including PRC/System Lambda/RTOLCAP etc, ~9K rows/month, header at row 8 | ~230 MB | 2 days |
| P0 | **Zone-level weather** | Open-Meteo Archive API | ✅ **Validated** — Free REST, 1h granularity, T/Wind/Humidity/Pressure, San Antonio 2025-12-14 cold front confirmed (18.6→6.3°C) | ~16 MB | 1 day |
| P1 | **Wind Forecast** | NP4-732-CD | ✅ **Validated** — CSV, includes GEN+STWPF+WGRPP per region (South_Houston/West/North), rolling 48h history | ~20 MB (recent) | 1 day |
| ~~P1~~ | ~~Binding Constraints~~ | ~~NP6-86-CD~~ | ⚠️ **Abandoned after validation** — One CSV per SCED interval (~5min), 105K files/year, backfill requires 100K+ HTTP requests (~200MB/yr zips) | ~~TB-scale~~ | ~~Not feasible~~ |
| P2 | **System Load Forecast** | NP3-565-CD | Not validated | TBD | 1 day |

**Key Findings:**
- **ERCOT API credentials are configured** (in LaunchAgent plist EnvironmentVariables, not in `.env`). Scraper has been running normally
- ERCOT API client (`ercot_client.py`) can be directly reused, already has `fetch_paginated_data()` method
- **NP4-732-CD (Wind Forecast) is a data endpoint** — Direct API query, same as existing LMP scraper
- **NP6-792-ER (RT Reserves) is a report type** — Only archive download (yearly XLSX zip), not a data API
- **NP6-323-CD (RT ORDC real-time) is also a report type** — Same archive-only download
- Open-Meteo `wind_speed_80m` is all null in the archive API, need to use ERA5 endpoint or only use 10m wind speed
- Binding Constraints historical backfill is not feasible, switching to zone-hub spread (LMP differences) to approximate congestion

**ERCOT API credentials (LaunchAgent):**
- Username: truetest86@gmail.com
- Subscription Key: 049314186c4a42c5af5b321f238fb83b
- Auth: B2C ROPC flow → Bearer token

### Weather Data Approach Comparison

**Existing foundation**: Wind model already has HRRR data pipeline (`prediction/models/wind/scripts/fetch_hrrr_herbie.py`)
- Uses Herbie library to download HRRR GRIB2 from AWS
- Extracts u/v wind speed (10m, 80m), 2m temperature, surface pressure
- But only has Texas bounding box spatial average, not zone-level point data
- Only ~6 months of data (second half of 2024)

| Approach | Advantages | Disadvantages | Recommendation |
|------|------|------|------|
| **HRRR via Herbie (extend existing)** | 3km resolution, existing pipeline, can extract point data | Historical backfill is slow (~50MB per file) | ✅ Primary approach — reuse wind model pipeline |
| **Open-Meteo API** | Free, REST, complete historical archive | Resolution ~11km | ✅ Supplementary approach — for quickly obtaining long history |
| **NOAA Weather API** | Official | Difficult to get historical data | ❌ |

**Strategy**: Run Open-Meteo first (quickly obtain full 2015-2026 history), HRRR point extraction as high-precision supplement.

**Weather station selection** (corresponding to the Zone-level mentioned in the document):
- LZ_CPS → San Antonio (29.42°N, 98.49°W)
- LZ_WEST → Midland/Odessa (31.95°N, 102.18°W)
- LZ_HOUSTON → Houston (29.76°N, 95.37°W)
- HB_NORTH → Dallas/Fort Worth (32.78°N, 96.80°W)
- HB_SOUTH → Corpus Christi (27.80°N, 97.40°W)
- System-wide → Weighted average or select Austin (30.27°N, 97.74°W)

---

## 3. Label System Redesign

### Current Labels (Too Simplistic)
```python
spike = rtm_lmp > max(100, 3 * rolling_24h_mean)
```

### Three-Layer Labels Proposed in the Document

#### Layer 1: SpikeEvent_z(t)
```python
# Price condition (either one)
price_cond = (P_z >= P_hi) | (P_z >= rolling_Q99_30d)

# Constraint-dominated condition (at least one)
spread_z_hub = P_z - P_hub
spread_z_hou = P_z - P_houston
constraint_cond = (spread_z_hub >= S_hi) | (spread_z_hou >= S_cps_hou_hi)

# Duration condition
raw = price_cond & constraint_cond
spike_event = rolling_min_count(raw, m=3)  # 5min granularity: continuous 15min
```

Initial thresholds: P_hi=400, S_hi=50, S_cps_hou_hi=80 (to be replaced by rolling quantiles later)

#### Layer 2: LeadSpike_z(t, H)
```python
# Whether a SpikeEvent will occur within the next H minutes
lead_spike = spike_event.rolling(H//dt, min_periods=1).max().shift(-H//dt)
```
H=60 or 90 minutes. This is the training target.

#### Layer 3: Regime_z(t)
```python
regime = 'Normal'   # default
regime[(P_z >= P_mid) | (spread_z_hub >= S_mid)] = 'Tight'
regime[spike_event] = 'Scarcity'
```
P_mid=150, S_mid=20

### Key Note: Data Granularity

The document requires **5min** granularity. Our historical data:
- `rtm_lmp_hist`: 15min (4 intervals/hr) — 11 years
- `rtm_lmp_api`: 5min — only 5 weeks

**Decision**: Build training set with 15min first (covering 11 years), use `rtm_lmp_api` 5min data for recent validation. Use 5min inference in production.

---

## 4. Feature Engineering Plan

### Document-Required Feature List vs Data Availability

#### Power System Side

| Feature | Required Data | Do We Have It | Plan |
|------|-----------|---------|------|
| Net Load | Total Load - Wind - Solar | ✅ Computable from fuel_mix | Phase 0 |
| ΔNet Load / Δt | Same as above | ✅ | Phase 0 |
| Δ²Net Load / Δt² | Same as above | ✅ | Phase 0 |
| Zone-Hub Spread | LZ_z LMP - Hub LMP | ✅ Available in rtm_lmp_hist | Phase 0 |
| d/dt(Spread) | Same as above | ✅ | Phase 0 |
| Online Gas Capacity vs Ramp | Requires unit-level data | ❌ | Phase 2+ (can approximate using Gas rate of change from fuel_mix) |
| RT Reserve Margin (PRC) | NP6-792-ER | ❌ | Phase 1 scraper |
| Storage Net Output | Does fuel_mix have storage type? | 🟡 Need to confirm | Phase 0 check |
| ORDC Price Adder | NP6-792-ER | ❌ | Phase 1 scraper |
| Binding Constraints | NP6-86-CD | ❌ | Phase 1 scraper |

#### Weather Side (Zone-level)

| Feature | Data Source | Formula | Plan |
|------|--------|------|------|
| T_anom_z(t) | Open-Meteo / HRRR | T_z(t) - T_norm_z(t) (30d rolling mean) | Phase 1 |
| ΔT_z(t) | Same as above | T_z(t) - T_z(t-1h) | Phase 1 |
| Wind Chill WC_z(t) | Requires T + WindSpeed | Standard NWS formula | Phase 1 |
| Cold Front CF_z(t) | Requires T + WindDir | ΔT < -3°C/h AND wind shift to N | Phase 1 |

#### Wind Power Side

| Feature | Data Source | Formula | Plan |
|------|--------|------|------|
| W(t) | fuel_mix_hist | Wind generation MW | ✅ Available |
| W_anom(t) | Same as above | W(t) - rolling_mean_30d(W) | ✅ Computable |
| ΔW(t) | Same as above | W(t) - W(t-1) | ✅ |
| CF_w(t) | Requires capacity data | W(t) / W_cap | 🟡 Hardcode capacity |
| e_w(t) | NP4-732-CD forecast | W_actual - W_forecast | ❌ Phase 1 |

---

## 5. Model Architecture Plan

### Phased Approach

| Phase | Model | Input | Output | Goal |
|------|------|------|------|------|
| **V1: GBM Baseline** | LightGBM / CatBoost | New feature set (15min) | LeadSpike_z(t, 60min) binary | PR-AUC baseline |
| **V2: CfC / LTC** | CfC-RNN (PyTorch) | Same as above + irregular time steps | LeadSpike + Regime probability | Outperform GBM |
| **V3: Spatiotemporal Model** | CfC + GNN (zone graph structure) | Multi-zone joint | Joint zone regime | Final form |

### Why GBM First for V1

1. The document says "baseline XGBoost/LightGBM as a control" — this is correct
2. GBM is usually a strong baseline on tabular data, and we have experience with it
3. Use GBM to quickly validate feature effectiveness, then move to CfC
4. If GBM can achieve reasonable AUC (>0.85), it means the feature engineering is solid

### V2 CfC/LTC Implementation Approach

```
pip install ncps  # Neural Circuit Policies — PyTorch implementation of CfC/LTC
```

- From MIT (Hasani et al., Nature Machine Intelligence 2022)
- Native PyTorch support
- Naturally handles irregular time steps (core requirement from the document)
- 20-50 hidden units are sufficient (few parameters, less prone to overfitting)

---

## 6. Data Acquisition Priority Roadmap

```
Week 1: Data Foundation
├─ Day 1-2: Complete fuel_mix (2025-01 ~ now)
│           Write zone-spread calculation + Net Load calculation
│           Verify 2025-12-14 case is reproducible
├─ Day 3-4: Open-Meteo weather data integration
│           6 cities, 2015~now, hourly
│           Compute T_anom, ΔT, Wind Chill, Cold Front
├─ Day 5:   ERCOT RT Reserves (NP6-792-ER) scraper
│           Historical weekly reports → SQLite

Week 2: Labels + Features + GBM Baseline
├─ Day 1-2: Three-layer label implementation (SpikeEvent/LeadSpike/Regime)
│           Per zone, 15min granularity
├─ Day 3-4: Feature pipeline v2 — integrate all new features
│           Power system + weather + wind power + spread
├─ Day 5:   GBM baseline training + evaluation
│           Per zone, LeadSpike(60min) prediction
│           Evaluation: PR-AUC, event-level recall

Week 3: Advanced Model + Strategy
├─ Day 1-3: CfC/LTC model implementation
│           PyTorch, ncps library
│           Compare against GBM baseline
├─ Day 4-5: BESS Hold-SOC strategy v1
│           Three-state dispatch (Normal/Tight/Scarcity)
│           2025-12-14 counterfactual backtest

Week 4: Integration + Validation
├─ Day 1-2: Wind forecast (NP4-732-CD) scraper
│           Binding constraints (NP6-86-CD) scraper
├─ Day 3-4: Feature expansion + model retraining
├─ Day 5:   API integration — /predictions/spike/zone-regime
│           5min inference frequency
│           Output: {regime, p_hat, recommended_power_cap}
```

---

## 7. Counterfactual Validation Plan (2025-12-14 Case)

Before any model goes live, it must pass this litmus test:

### Data
- RTM 15min LMP: LZ_CPS, LZ_WEST, LZ_HOUSTON, HB_HUBAVG (✅ Available)
- Fuel Mix: Wind, Gas, Solar (✅ Available through 2024-12)
- Weather: San Antonio temperature on that day (acquired in Phase 1)

### Expected Model Behavior
1. **16:30**: Net Load inflection point detected → Regime from Normal → Tight
2. **18:00**: Spread(CPS-Houston) starts widening → Maintain Tight
3. **19:30**: Reserve tightening + spread acceleration → Tight → Scarcity
4. **20:00-22:00**: LeadSpike = 1 → Hold-SOC strategy → Full power discharge

### Evaluation
- Traditional strategy (discharge 16:30-20:00): Releases SOC in the $100-$325 range
- Improved strategy (Hold-SOC until 20:00): Releases SOC in the $400-$686 range
- Expected revenue difference: 30-60%+ improvement

---

## 8. Technology Stack

| Component | Tool | Notes |
|------|------|------|
| Data scraper | Python + ERCOT Public API | Reuse existing ercot_client.py |
| Weather data | Open-Meteo API (free, REST) | pip install openmeteo-requests |
| HRRR (future) | Herbie library | pip install herbie-data |
| Feature store | SQLite (existing) | New tables |
| GBM model | LightGBM + CatBoost + Optuna | Reuse existing pipeline |
| CfC/LTC model | PyTorch + ncps | pip install ncps |
| API integration | FastAPI (existing) | Add zone-regime endpoint |

---

## 9. Risks and Open Questions

1. **RTM congestion components historical data**: `rtm_lmp_api` has only 5 weeks. Does ERCOT provide historical 5min LMP with components? If not, we can only use LMP spread to approximate congestion (lower precision).

2. **Weather data alignment**: Does Open-Meteo's historical weather accurately cover the entire 2015-2026 time range? Need to verify.

3. **CfC model effectiveness in the ERCOT context**: CfC papers were primarily validated on autonomous driving and Physionet. Power market regime switching is a new scenario and requires experimentation.

4. **15min → 5min granularity transition**: Train with 15min (11 years) or 5min (5 weeks)? Recommendation: 15min training + 5min inference (fine-tune).

5. **Binding constraints data volume**: NP6-86-CD has one entry per SCED interval (~5min), 10 years of data could be very large.

---

*This plan cuts no corners. Every phase is based on investigation of actual data, assuming nothing we don't have.*
