# RTM Model Retrain Report: 43 → 80 Features

**Date**: 2026-03-22  
**Author**: Kira (automated)  
**Commit**: `fe650aa`

## Background

Phase 1 spike prediction added 37 new features (weather, reserves, wind forecast, regime) to the unified feature pipeline, bringing the total from 43 to 80. The existing RTM models were trained on 43 features and threw a feature mismatch error at inference time:

```
The number of features in data (83) is not the same as it was in training data (43)
```

## Retrain Configuration

- **Script**: `prediction/scripts/retrain_all_v2.py --skip-dam --skip-spike --rtm-trials 15`
- **Scope**: 15 Settlement Points × 3 horizons (1h, 4h, 24h) = 45 models
- **Optuna trials**: 15 per model (reduced from previous 30 for speed)
- **Data**: Same SQLite archive, no new data fetched
- **Training time**: ~70 minutes

## Results Summary

| Metric | Count |
|--------|-------|
| Improved (MAE drop > 0.1) | 17 |
| Regressed (MAE rise > 0.1) | 18 |
| Flat (±0.1) | 10 |
| **Total** | **45** |

## Significant Improvements (MAE drop > 1.0)

| SP | Horizon | Before MAE | After MAE | Δ | Δ% |
|---|---|---|---|---|---|
| hb_west | 24h | 40.41 | 22.09 | **-18.31** | -45.3% |
| lz_west | 4h | 31.91 | 26.75 | -5.17 | -16.2% |
| lz_raybn | 24h | 20.28 | 16.02 | -4.26 | -21.0% |
| lz_north | 4h | 19.90 | 15.81 | -4.10 | -20.6% |
| lz_houston | 4h | 16.53 | 13.65 | -2.89 | -17.5% |
| hb_west | 1h | 16.85 | 14.75 | -2.10 | -12.4% |
| lz_north | 24h | 17.19 | 15.72 | -1.48 | -8.6% |
| hb_busavg | 4h | 16.99 | 15.75 | -1.24 | -7.3% |
| lz_houston | 1h | 12.03 | 10.90 | -1.14 | -9.5% |

Note: `hb_west/24h` before MAE=40.41 was anomalous (likely overfitting); the new 22.09 is a correction, not just an improvement.

## Significant Regressions (MAE rise > 1.0)

| SP | Horizon | Before MAE | After MAE | Δ | Δ% |
|---|---|---|---|---|---|
| lz_lcra | 24h | 18.14 | 30.47 | **+12.34** | +68.0% |
| hb_hubavg | 4h | 16.89 | 27.97 | +11.09 | +65.7% |
| hb_houston | 24h | 12.97 | 19.07 | +6.09 | +47.0% |
| hb_hubavg | 24h | 15.09 | 20.81 | +5.72 | +37.9% |
| lz_west | 1h | 17.24 | 20.20 | +2.96 | +17.2% |
| lz_aen | 4h | 17.97 | 20.77 | +2.80 | +15.6% |
| lz_raybn | 4h | 15.70 | 17.39 | +1.69 | +10.7% |
| hb_south | 4h | 17.26 | 18.85 | +1.59 | +9.2% |
| lz_west | 24h | 22.88 | 24.28 | +1.40 | +6.1% |
| hb_south | 24h | 14.06 | 15.19 | +1.12 | +8.0% |

## Full Comparison Table

| SP/Horizon | Before MAE | After MAE | Δ | Before RMSE | After RMSE | Δ RMSE |
|---|---|---|---|---|---|---|
| hb_busavg/1h | 11.70 | 11.78 | +0.07 | 40.76 | 40.51 | -0.25 |
| hb_busavg/4h | 16.99 | 15.75 | -1.24 | 72.13 | 53.03 | -19.10 |
| hb_busavg/24h | 14.78 | 14.37 | -0.41 | 44.04 | 43.20 | -0.84 |
| hb_houston/1h | 11.32 | 10.89 | -0.43 | 33.74 | 32.83 | -0.91 |
| hb_houston/4h | 15.14 | 14.38 | -0.76 | 56.67 | 47.07 | -9.61 |
| hb_houston/24h | 12.97 | 19.07 | +6.09 | 34.56 | 114.39 | +79.83 |
| hb_hubavg/1h | 12.91 | 12.84 | -0.07 | 45.21 | 44.09 | -1.12 |
| hb_hubavg/4h | 16.89 | 27.97 | +11.09 | 63.01 | 150.43 | +87.42 |
| hb_hubavg/24h | 15.09 | 20.81 | +5.72 | 42.30 | 87.90 | +45.60 |
| hb_north/1h | 12.59 | 12.36 | -0.23 | 47.01 | 44.83 | -2.17 |
| hb_north/4h | 14.88 | 14.81 | -0.08 | 48.46 | 47.78 | -0.68 |
| hb_north/24h | 14.97 | 14.92 | -0.06 | 49.34 | 49.17 | -0.18 |
| hb_pan/1h | 13.62 | 13.56 | -0.06 | 45.12 | 45.52 | +0.40 |
| hb_pan/4h | 18.23 | 18.14 | -0.10 | 50.79 | 50.19 | -0.60 |
| hb_pan/24h | 22.20 | 22.83 | +0.63 | 62.74 | 62.31 | -0.42 |
| hb_south/1h | 12.84 | 13.35 | +0.50 | 53.16 | 37.46 | -15.70 |
| hb_south/4h | 17.26 | 18.85 | +1.59 | 86.49 | 105.96 | +19.47 |
| hb_south/24h | 14.06 | 15.19 | +1.12 | 38.02 | 53.60 | +15.59 |
| hb_west/1h | 16.85 | 14.75 | -2.10 | 62.13 | 46.05 | -16.09 |
| hb_west/4h | 19.33 | 20.04 | +0.71 | 51.21 | 56.60 | +5.39 |
| hb_west/24h | 40.41 | 22.09 | -18.31 | 163.66 | 65.10 | -98.56 |
| lz_aen/1h | 14.60 | 14.59 | -0.01 | 48.57 | 48.08 | -0.49 |
| lz_aen/4h | 17.97 | 20.77 | +2.80 | 51.97 | 86.99 | +35.02 |
| lz_aen/24h | 16.92 | 16.97 | +0.04 | 51.78 | 51.89 | +0.11 |
| lz_cps/1h | 13.67 | 14.03 | +0.36 | 45.54 | 47.41 | +1.88 |
| lz_cps/4h | 17.50 | 17.26 | -0.24 | 51.70 | 51.28 | -0.42 |
| lz_cps/24h | 16.28 | 16.62 | +0.34 | 50.82 | 51.14 | +0.32 |
| lz_houston/1h | 12.03 | 10.90 | -1.14 | 42.66 | 33.79 | -8.86 |
| lz_houston/4h | 16.53 | 13.65 | -2.89 | 81.59 | 38.62 | -42.97 |
| lz_houston/24h | 13.02 | 13.08 | +0.06 | 35.00 | 35.40 | +0.41 |
| lz_lcra/1h | 15.11 | 15.59 | +0.48 | 58.51 | 59.49 | +0.98 |
| lz_lcra/4h | 18.98 | 19.32 | +0.34 | 65.97 | 68.84 | +2.87 |
| lz_lcra/24h | 18.14 | 30.47 | +12.34 | 65.64 | 130.11 | +64.47 |
| lz_north/1h | 12.94 | 13.04 | +0.11 | 47.90 | 59.49 | +11.58 |
| lz_north/4h | 19.90 | 15.81 | -4.10 | 80.48 | 50.61 | -29.87 |
| lz_north/24h | 17.19 | 15.72 | -1.48 | 65.03 | 51.64 | -13.39 |
| lz_raybn/1h | 15.11 | 15.08 | -0.03 | 68.21 | 68.08 | -0.13 |
| lz_raybn/4h | 15.70 | 17.39 | +1.69 | 52.77 | 65.59 | +12.82 |
| lz_raybn/24h | 20.28 | 16.02 | -4.26 | 82.35 | 53.58 | -28.76 |
| lz_south/1h | 12.51 | 12.00 | -0.51 | 39.66 | 39.51 | -0.16 |
| lz_south/4h | 15.33 | 15.13 | -0.20 | 47.31 | 43.17 | -4.14 |
| lz_south/24h | 15.31 | 14.85 | -0.47 | 45.61 | 43.76 | -1.85 |
| lz_west/1h | 17.24 | 20.20 | +2.96 | 50.52 | 56.64 | +6.12 |
| lz_west/4h | 31.91 | 26.75 | -5.17 | 101.37 | 78.58 | -22.78 |
| lz_west/24h | 22.88 | 24.28 | +1.40 | 54.59 | 55.48 | +0.89 |

## Analysis

### What Worked
- **1h horizon**: Mostly flat or improved. New weather/reserve/wind features provide useful short-term signal.
- **Extreme corrections**: `hb_west/24h` dropped from 40→22 (previous model was clearly overfit). Several 4h models improved significantly.

### What Didn't Work
- **24h regressions**: `lz_lcra`, `hb_hubavg`, `hb_houston` — MAE roughly doubled. The 37 new features likely introduce noise for day-ahead prediction.
- **RMSE spikes**: Some regressed models show massive RMSE increases (e.g., `hb_hubavg/4h` RMSE 63→150), suggesting the new models occasionally produce wild outliers.

### Root Cause Hypotheses
1. **Insufficient Optuna trials**: Used 15 trials (vs. 30 previously). With 80 features, the search space is larger and needs more exploration.
2. **Feature noise at longer horizons**: Weather/reserve features are most predictive at 1-4h; at 24h they may add more noise than signal.
3. **No feature selection**: All 80 features fed directly. LightGBM handles irrelevant features reasonably well, but with limited trials, Optuna may converge to suboptimal hyperparameters.

## Proposed Fix Options

1. **More Optuna trials** — Retrain regressed models with 50+ trials
2. **Horizon-aware feature selection** — Use different feature subsets for 1h vs 4h vs 24h
3. **Hybrid deployment** — Use new models for 1h (where they're better), keep old models for severely regressed 24h
4. **Recursive feature elimination** — Train with importance-based feature pruning for 24h models

## Other Changes in This Commit

- **Localhost auth bypass**: `prediction/src/main.py` — requests from `127.0.0.1`/`::1` skip X-API-Key authentication. Fixes prediction-runner 401 errors since Sprint 7 API key feature was added.
- **Spike features .gitignore**: Added `prediction/data/spike_features/` to `.gitignore` (~400MB parquet files).
