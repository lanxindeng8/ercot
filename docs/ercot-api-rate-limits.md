# ERCOT API Rate Limit — Experience Document

**Last updated**: 2026-03-23
**Applies to**: ERCOT Public API (`api.ercot.com`)

## Account Information

- Only one set of ERCOT API credentials (truetest86@gmail.com)
- All endpoints and all jobs share the same `ERCOT_PUBLIC_API_SUBSCRIPTION_KEY` rate limit quota
- Credentials are stored in LaunchAgent plist `EnvironmentVariables`, not in `.env` files

## Rate Limit Behavior Observations

### Throttling Type: Hourly Bandwidth Limit
429 response body clearly states:
```json
{"statusCode": 429, "error": "BandwidthQuotaExceeded",
 "message": "You have exceeded the hourly bandwidth limit. Your quota will reset within the hour."}
```

- **The limit is on bandwidth (data transfer volume), not request count**
- Quota **resets every hour** (on the hour or rolling window)
- All endpoints share quota: rtm-lmp, dam-lmp, wind-forecast, archive downloads all count toward it
- ERCOT API **does not return a `Retry-After` header**

### Empirical Data
- Wind forecast ~160K rows per month (~10MB), triggers after querying 3-5 consecutive months
- Archive downloads ~16MB per file, 4 consecutive downloads succeed (with 90s intervals), 5th one very likely gets 429
- After triggering 429, subsequent requests within the same hour are basically all rejected

### Differences Between Two Request Types
| Type | Endpoint Example | Characteristics |
|---|---|---|
| **Archive download** | `np6-792-ER` (reserves) | Downloads entire year XLSX at once (~16MB), low total request count |
| **Live query** | `np4-732-cd` (wind forecast) | Paginated queries by month, ~160K rows per month split across 4 pages, request-intensive |

Archive downloads are more likely to succeed because a single file contains an entire year of data (4 requests = 4 years).
Live queries require 1-4 paginated requests per month, 12 months = 12-48 requests, extremely easy to trigger throttling.

### Pitfalls Encountered

1. **urllib3 `Retry(status_forcelist=[429])` is a disaster**
   urllib3 does rapid retries on 429 (second-level intervals), which actually accelerates quota exhaustion.
   **Fix**: Removed 429 from `status_forcelist`, handle with custom backoff logic only.
   Commit: `6ffd9f3`

2. **Multiple concurrent jobs = faster rate limit hits**
   Previously rtm-scraper + prediction-runner triggered simultaneously every 5 minutes, plus manual backfill — three concurrent streams hit 429 instantly.
   **Rule**: At most one process accessing the ERCOT API at any time.

3. **Archive file naming doesn't match content**
   `RTM_ORDC_..._2023.xlsx` actually contains 2022 data (ERCOT names by publication year, not data year).
   **Rule**: Must verify the actual year range of `sced_timestamp` after ingestion.

## Current Backoff Strategy (`scraper/src/ercot_client.py`)

```
Initial delay: 10 seconds
Backoff method: Exponential × 2 + jitter (±50%)
Max delay: 300 seconds (5 minutes)
Max retries: 5
Sequence example: 10s → 20s → 40s → 80s → 160s (with random jitter)
```

## Data Acquisition Strategy (Recommended)

### Principles
1. **Serial, not concurrent** — Only one ERCOT API caller at a time
2. **Archive first** — If data is available in yearly archive format (e.g., reserves), prefer downloading archives
3. **Interval ≥ 90 seconds** — Wait 90 seconds after each successful request before sending the next
4. **Don't rush on failure** — Wait at least 5 minutes before retrying after a 429

### Bulk Backfill
- **Archive types (reserves, etc.)**: Run script directly, 90-second intervals, ≤4 files per hour
- **Query types (wind forecast, etc.)**: Use cron to trigger hourly, pull sequentially until 429, continue next time
- **Large batches (>20 requests)**: Spread across multiple days, or find archive alternatives
- **Script mode**: Abort immediately when throttled, don't do long backoff retries (wastes time, quota doesn't reset until the next hour)

### LaunchAgent Architecture (2026-03-23 Refactored)

| Job | Frequency | ERCOT API? | Description |
|---|---|---|---|
| `rtm-lmp-cdr` | 5min | ❌ CDR HTML | Real-time RTM data, does not use API |
| `rtm-lmp-api` | 1h | ✅ | RTM historical backfill (6h delayed data) |
| `dam-pipeline` | Daily 14:00 | ✅ (1 time) | Combined pipeline: predictions → API fetch → CDR fetch → Telegram |
| `prediction-runner` | 5min | ❌ | Calls local localhost:8011 |
| `prediction-api` | Always-on | ❌ | FastAPI service |
| `telegram-lmp-summary` | Daily 06:30 | ❌ | Reads from InfluxDB |
| `btc-price-monitor` | 5min | ❌ | PolyManager project, unrelated |

**Only 2 ERCOT API callers**: `rtm-lmp-api` (hourly) + `dam-pipeline` (daily at 14:00 once)

**Deprecated (no longer loaded)**:
- `rtm-lmp-scraper` → Split into `rtm-lmp-cdr` + `rtm-lmp-api`
- `dam-lmp-scraper`, `dam-lmp-cdr-scraper`, `dam-predictions`, `telegram-dam-schedule` → Merged into `dam-pipeline`

## Data Backfill Status (2026-03-23)

### RT Reserves ✅ Complete
| Year | Rows | Status |
|---|---|---|
| 2016-2025 | 1,056,444 | ~106K rows per year, continuously complete |

### Wind Forecast ✅ Complete
| Range | Rows | Status |
|---|---|---|
| 2022-12 → 2026-03 (40 months) | 113,924 | Complete, no gaps |

Automatically completed via OpenClaw cron job `wind-forecast-backfill` (pulls a batch each hour until 429, continues next hour).
Cron job was automatically deleted after completion. Script preserved at `prediction/scripts/fetch_wind_one_month.py` for future use.

## Cron Jobs

### wind-forecast-backfill
- **Frequency**: Every 1 hour
- **Script**: `prediction/scripts/fetch_wind_one_month.py`
- **Logic**: Query DB to find all missing months → Pull sequentially month by month → Stop immediately on 429 and wait for next cron → Delete cron when all backfilled
- **Each run** pulls as many as possible (within quota), not just one month
- **Status**: Running (created 2026-03-23)

## Future Optimization Directions

- [ ] Obtain a second subscription key (if ERCOT allows multiple keys)
- [ ] Implement request quota pool: All jobs share a token bucket to prevent concurrent overuse
- [x] ~~Check if NP4-732-CD has an archive endpoint~~ — Could not verify (blocked by 429), but data API is sufficient
- [ ] Reduce RTM scraper frequency to 15 minutes + add backoff
- [ ] Adjust backoff strategy: Don't do long backoff after 429, just abort and wait for the next hour (matches hourly quota reset)
