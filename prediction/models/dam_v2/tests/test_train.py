"""Smoke tests for DAM v2 training pipeline."""

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path

# Allow imports from parent
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train import (
    FEATURE_COLS,
    TARGET,
    CAT_FEATURES,
    compute_metrics,
    naive_baseline_metrics,
)


def _make_synthetic_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic training data with realistic structure."""
    rng = np.random.RandomState(seed)
    data = {}
    data["delivery_date"] = pd.date_range("2023-01-01", periods=n, freq="h").strftime("%Y-%m-%d")
    data["hour_ending"] = np.tile(np.arange(1, 25), n // 24 + 1)[:n]

    # Temporal
    data["hour_of_day"] = data["hour_ending"]
    data["day_of_week"] = np.tile(np.arange(7), n // 7 + 1)[:n]
    data["month"] = np.tile(np.arange(1, 13), n // 12 + 1)[:n]
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)
    data["is_peak_hour"] = ((data["hour_of_day"] >= 7) & (data["hour_of_day"] <= 22)).astype(int)
    data["is_holiday"] = rng.choice([0, 1], n, p=[0.97, 0.03])
    data["is_summer"] = ((data["month"] >= 6) & (data["month"] <= 9)).astype(int)

    # Base price with hourly pattern
    base = 25 + 10 * np.sin(2 * np.pi * data["hour_of_day"] / 24) + rng.normal(0, 5, n)
    data[TARGET] = base

    # Price lags (shifted versions of base)
    for lag_name, shift in [("dam_lag_1h", 1), ("dam_lag_4h", 4), ("dam_lag_24h", 24), ("dam_lag_168h", 168)]:
        data[lag_name] = np.roll(base, shift) + rng.normal(0, 1, n)
    for lag_name, shift in [("rtm_lag_1h", 1), ("rtm_lag_4h", 4), ("rtm_lag_24h", 24), ("rtm_lag_168h", 168)]:
        data[lag_name] = np.roll(base, shift) + rng.normal(0, 3, n)

    # Rolling stats
    for prefix in ["dam", "rtm"]:
        for window in ["24h", "168h"]:
            data[f"{prefix}_roll_{window}_mean"] = base + rng.normal(0, 2, n)
            data[f"{prefix}_roll_{window}_std"] = np.abs(rng.normal(5, 2, n))
            data[f"{prefix}_roll_{window}_min"] = base - np.abs(rng.normal(10, 3, n))
            data[f"{prefix}_roll_{window}_max"] = base + np.abs(rng.normal(10, 3, n))

    # Cross-market
    data["dam_rtm_spread"] = rng.normal(0, 5, n)
    data["spread_roll_24h_mean"] = rng.normal(0, 3, n)
    data["spread_roll_168h_mean"] = rng.normal(0, 2, n)

    # Fuel mix pct
    data["wind_pct"] = rng.uniform(0.05, 0.35, n)
    data["solar_pct"] = rng.uniform(0.0, 0.15, n)
    data["gas_pct"] = rng.uniform(0.3, 0.6, n)
    data["nuclear_pct"] = rng.uniform(0.05, 0.12, n)
    data["coal_pct"] = rng.uniform(0.05, 0.20, n)
    data["hydro_pct"] = rng.uniform(0.0, 0.03, n)

    # Ancillary service (13)
    data["regdn"] = rng.uniform(5, 30, n)
    data["regup"] = rng.uniform(5, 50, n)
    data["rrs"] = rng.uniform(3, 25, n)
    data["nspin"] = rng.uniform(1, 10, n)
    data["ecrs"] = rng.uniform(1, 15, n)
    data["reg_spread"] = data["regup"] - data["regdn"]
    data["total_as_cost"] = data["regdn"] + data["regup"] + data["rrs"] + data["nspin"] + data["ecrs"]
    data["regup_lag_24h"] = np.roll(data["regup"], 24)
    data["rrs_lag_24h"] = np.roll(data["rrs"], 24)
    data["nspin_lag_24h"] = np.roll(data["nspin"], 24)
    data["total_as_lag_24h"] = np.roll(data["total_as_cost"], 24)
    data["total_as_roll_24h_mean"] = data["total_as_cost"] + rng.normal(0, 2, n)
    data["total_as_roll_24h_std"] = np.abs(rng.normal(5, 2, n))

    # RTM components (6)
    data["congestion_pct"] = rng.uniform(-10, 10, n)
    data["loss_pct"] = rng.uniform(-5, 5, n)
    data["energy_pct"] = rng.uniform(80, 100, n)
    data["congestion_ma_4h"] = rng.uniform(-5, 5, n)
    data["congestion_volatility_24h"] = np.abs(rng.normal(3, 1, n))
    data["high_congestion_flag"] = (np.abs(data["congestion_pct"]) > 20).astype(int)

    # Fuel gen MW (15)
    data["gas_gen_mw"] = rng.uniform(10000, 30000, n)
    data["gas_cc_gen_mw"] = rng.uniform(8000, 20000, n)
    data["coal_gen_mw"] = rng.uniform(5000, 15000, n)
    data["nuclear_gen_mw"] = rng.uniform(4000, 6000, n)
    data["solar_gen_mw"] = rng.uniform(0, 10000, n)
    data["wind_gen_mw"] = rng.uniform(2000, 25000, n)
    data["hydro_gen_mw"] = rng.uniform(100, 1000, n)
    data["biomass_gen_mw"] = rng.uniform(50, 500, n)
    data["total_gen_mw"] = sum(data[f"{f}_gen_mw"] for f in ["gas", "gas_cc", "coal", "nuclear", "solar", "wind", "hydro", "biomass"])
    data["renewable_ratio"] = (data["solar_gen_mw"] + data["wind_gen_mw"]) / data["total_gen_mw"]
    data["thermal_ratio"] = (data["gas_gen_mw"] + data["gas_cc_gen_mw"] + data["coal_gen_mw"]) / data["total_gen_mw"]
    data["net_load_mw"] = data["total_gen_mw"] - data["solar_gen_mw"] - data["wind_gen_mw"]
    data["solar_ramp_1h"] = rng.normal(0, 500, n)
    data["wind_ramp_1h"] = rng.normal(0, 500, n)
    data["gas_ramp_1h"] = rng.normal(0, 300, n)

    # Cross-domain (6)
    data["dam_as_ratio"] = base / (data["total_as_cost"] + 1)
    data["reg_spread_roll_24h_mean"] = data["reg_spread"] + rng.normal(0, 2, n)
    data["ecrs_lag_24h"] = np.roll(data["ecrs"], 24)
    data["gas_cc_share"] = data["gas_cc_gen_mw"] / data["total_gen_mw"]
    data["wind_ramp_4h"] = rng.normal(0, 1000, n)
    data["solar_ramp_4h"] = rng.normal(0, 1000, n)

    # RTM target (not used for DAM training but present in parquet)
    data["rtm_lmp"] = base + rng.normal(0, 8, n)

    return pd.DataFrame(data)


class TestComputeMetrics:
    def test_perfect_predictions(self):
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        m = compute_metrics(y, y)
        assert m["mae"] == 0.0
        assert m["rmse"] == 0.0

    def test_known_values(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([12.0, 18.0, 33.0])
        m = compute_metrics(y_true, y_pred)
        assert m["mae"] == pytest.approx(2.333, abs=0.01)
        assert m["rmse"] > 0
        assert 0 < m["mape_pct"] < 100

    def test_directional_accuracy(self):
        y_true = np.array([10.0, 20.0, 15.0, 25.0])
        y_pred = np.array([10.0, 22.0, 14.0, 26.0])  # all directions correct
        m = compute_metrics(y_true, y_pred)
        assert m["directional_accuracy_pct"] == 100.0

    def test_mape_skips_near_zero(self):
        y_true = np.array([0.5, 0.0, 100.0])
        y_pred = np.array([1.0, 0.5, 110.0])
        m = compute_metrics(y_true, y_pred)
        # MAPE only computed on |y| > 1.0, so only the 100->110 pair
        assert m["mape_pct"] == pytest.approx(10.0, abs=0.1)


class TestNaiveBaseline:
    def test_naive_uses_lag24(self):
        df = _make_synthetic_data(200)
        m = naive_baseline_metrics(df)
        assert "mae" in m
        assert "rmse" in m
        assert m["mae"] > 0


class TestSyntheticData:
    def test_has_all_features(self):
        df = _make_synthetic_data()
        for col in FEATURE_COLS:
            assert col in df.columns, f"Missing feature: {col}"
        assert TARGET in df.columns

    def test_no_nans_in_features(self):
        df = _make_synthetic_data()
        assert df[FEATURE_COLS + [TARGET]].isna().sum().sum() == 0


class TestModelTrainSmoke:
    """Smoke test: train a small CatBoost model on synthetic data."""

    def test_catboost_smoke(self):
        from catboost import CatBoostRegressor, Pool

        df = _make_synthetic_data(500)
        X = df[FEATURE_COLS]
        y = df[TARGET].values

        cat_idx = [list(X.columns).index(c) for c in CAT_FEATURES]
        model = CatBoostRegressor(
            iterations=50, depth=4, learning_rate=0.1,
            loss_function="MAE", verbose=0, random_seed=42,
            allow_writing_files=False,
        )
        model.fit(Pool(X, y, cat_features=cat_idx))
        preds = model.predict(X)

        assert preds.shape == (500,)
        assert preds.dtype == np.float64
        assert not np.any(np.isnan(preds))

    def test_lightgbm_smoke(self):
        from lightgbm import LGBMRegressor

        df = _make_synthetic_data(500)
        X = df[FEATURE_COLS]
        y = df[TARGET].values

        cat_cols = [c for c in CAT_FEATURES if c in X.columns]
        model = LGBMRegressor(
            n_estimators=50, max_depth=4, learning_rate=0.1,
            objective="mae", random_state=42, verbosity=-1,
        )
        model.fit(X, y, categorical_feature=cat_cols)
        preds = model.predict(X)

        assert preds.shape == (500,)
        assert not np.any(np.isnan(preds))

    def test_predictions_reasonable_range(self):
        """Model predictions should be in a reasonable price range."""
        from catboost import CatBoostRegressor, Pool

        df = _make_synthetic_data(500)
        X = df[FEATURE_COLS]
        y = df[TARGET].values

        cat_idx = [list(X.columns).index(c) for c in CAT_FEATURES]
        model = CatBoostRegressor(
            iterations=100, depth=4, learning_rate=0.1,
            loss_function="MAE", verbose=0, random_seed=42,
            allow_writing_files=False,
        )
        model.fit(Pool(X, y, cat_features=cat_idx))
        preds = model.predict(X)

        # Predictions should be within a reasonable range of the training data
        assert preds.min() > y.min() - 50
        assert preds.max() < y.max() + 50
