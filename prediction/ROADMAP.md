# Prediction Service Roadmap

## Current Status

### Working
- Delta Spread prediction (RTM - DAM spread) - 3 CatBoost models
- FastAPI service running on port 8001
- InfluxDB data fetcher

### Pending
- DAM price prediction models (pipeline ready, needs training)
- RTM rolling prediction integration
- ercot-viewer integration

---

## Phase 1: RTM Rolling Predictions (Priority)

### Goal
Fill the "Pred" column in ercot-viewer's RTM table with rolling predictions.

### Approach: Delta Spread Method
Use existing Delta Spread models to predict RTM:
```
Predicted RTM = Actual DAM (current hour) + Predicted Delta Spread
```

### Required Work

#### 1.1 Create Rolling Prediction Endpoint
```python
GET /predictions/rtm/rolling
Parameters:
  - settlement_point: string (e.g., "HB_HOUSTON")
  - hours_ahead: int (default: 1, how many hours to predict)

Response:
{
  "settlement_point": "HB_HOUSTON",
  "predictions": [
    {"time": "2026-02-08T10:10:00Z", "predicted_lmp": 25.50, "confidence": 0.85},
    {"time": "2026-02-08T10:15:00Z", "predicted_lmp": 26.10, "confidence": 0.82},
    ...
  ],
  "model_info": {
    "method": "delta_spread",
    "last_actual_time": "2026-02-08T10:05:00Z"
  }
}
```

#### 1.2 RTM Rolling Predictor Class
- Fetch latest RTM and DAM prices from InfluxDB
- Calculate features for delta spread model
- Generate predictions for upcoming intervals
- Handle the 6-hour API delay gracefully

#### 1.3 Feature Engineering for Real-time
Current delta spread features need:
- `rtm_lag_5m`, `rtm_lag_10m`, etc. - Recent RTM prices
- `dam_current_hour` - Current hour's DAM price
- `hour`, `day_of_week`, etc. - Time features

---

## Phase 2: DAM Price Predictions

### Goal
Predict next-day DAM prices before the market clears (~6 AM publication).

### Approach
Train ensemble models (XGBoost, LightGBM, CatBoost) on historical DAM data.

### Required Work

#### 2.1 Train DAM Models
```bash
python scripts/train_dam_models.py --source influxdb --settlement-point LZ_HOUSTON
```

#### 2.2 Create DAM Prediction Endpoint
```python
GET /predictions/dam/next_day
Parameters:
  - settlement_point: string
  - date: string (YYYY-MM-DD, default: tomorrow)

Response:
{
  "settlement_point": "HB_HOUSTON",
  "delivery_date": "2026-02-09",
  "predictions": [
    {"hour_ending": "01:00", "predicted_price": 22.50},
    {"hour_ending": "02:00", "predicted_price": 21.80},
    ...
  ]
}
```

---

## Phase 3: ercot-viewer Integration

### Goal
Display predictions in the ercot-viewer frontend.

### Required Changes

#### 3.1 API Route Updates
Modify `/api/rtm-spp` and `/api/dam-spp` to include predictions:

```typescript
interface PivotedRow {
  time: string;
  prices: Record<string, number | null>;
  predictions: Record<string, number | null>;  // NEW
}
```

#### 3.2 Fetch Predictions from prediction-service
```typescript
// In ercot-viewer API route
const predictions = await fetch(
  `http://localhost:8001/predictions/rtm/rolling?settlement_point=${point}`
);
```

#### 3.3 Update PriceTable Component
```tsx
<td className="price-cell pred-cell">
  {formatPrice(row.predictions[point])}  // Changed from empty
</td>
```

---

## Phase 4: Advanced Features (Future)

### 4.1 Multi-horizon RTM Predictions
- Predict 5, 10, 15, 30, 60 minutes ahead
- Show prediction accuracy degradation over time

### 4.2 Confidence Intervals
- Display prediction confidence bands
- Color-code based on uncertainty

### 4.3 Spike Predictions
- Flag potential price spike events
- Binary classification for high-price periods

### 4.4 Model Retraining Pipeline
- Automated daily/weekly model updates
- Performance monitoring and alerts

---

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────┐
│  ercot-viewer   │────▶│ prediction-svc   │────▶│  InfluxDB   │
│   (Next.js)     │     │   (FastAPI)      │     │  (prices)   │
└─────────────────┘     └──────────────────┘     └─────────────┘
        │                       │
        │                       ▼
        │               ┌──────────────────┐
        │               │  ML Models       │
        │               │  - Delta Spread  │
        │               │  - DAM Ensemble  │
        │               │  - RTM Forecast  │
        │               └──────────────────┘
        │
        ▼
┌─────────────────┐
│  ERCOT CDR/API  │
│  (real-time)    │
└─────────────────┘
```

---

## Timeline Estimate

| Phase | Task | Status |
|-------|------|--------|
| 1.1 | RTM Rolling Endpoint | 🔴 Not Started |
| 1.2 | RTM Rolling Predictor | 🔴 Not Started |
| 1.3 | Real-time Features | 🔴 Not Started |
| 2.1 | Train DAM Models | 🔴 Not Started |
| 2.2 | DAM Prediction Endpoint | 🟡 Partial |
| 3.1 | ercot-viewer API Update | 🔴 Not Started |
| 3.2 | Fetch Predictions | 🔴 Not Started |
| 3.3 | Update PriceTable | 🔴 Not Started |

---

## Quick Start for Phase 1

1. **Implement RTM Rolling Predictor**
   - Create `src/models/rtm_rolling.py`
   - Use delta spread models + current DAM price

2. **Add API Endpoint**
   - Add `/predictions/rtm/rolling` to `src/main.py`

3. **Test with curl**
   ```bash
   curl "http://localhost:8001/predictions/rtm/rolling?settlement_point=HB_HOUSTON"
   ```

4. **Update ercot-viewer**
   - Call prediction API from `/api/rtm-spp`
   - Merge predictions with actual prices
