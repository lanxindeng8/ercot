"""Tests for RTM multi-horizon training pipeline."""

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train import (
    ALL_FEATURE_COLS,
    BASE_TARGET,
    CAT_FEATURES,
    HORIZONS,
    compute_metrics,
    create_targets,
    add_rtm_features,
    naive_baseline_metrics,
)


def _make_synthetic_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic RTM training data with realistic structure."""
    rng = np.random.RandomState(seed)
    data = {}

    # Temporal
    data["hour_of_day"] = np.tile(np.arange(24), n // 24 + 1)[:n]
    data["day_of_week"] = np.tile(np.arange(7), n // 7 + 1)[:n]
    data["month"] = np.tile(np.arange(1, 13), n // 12 + 1)[:n]
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)
    data["is_peak_hour"] = ((data["hour_of_day"] >= 7) & (data["hour_of_day"] <= 22)).astype(int)
    data["is_holiday"] = rng.choice([0, 1], n, p=[0.97, 0.03])
    data["is_summer"] = ((data["month"] >= 6) & (data["month"] <= 9)).astype(int)

    # Base RTM price with hourly pattern + noise
    base = 30 + 15 * np.sin(2 * np.pi * data["hour_of_day"] / 24) + rng.normal(0, 8, n)
    data[BASE_TARGET] = base

    # DAM price (less volatile)
    dam_base = 25 + 10 * np.sin(2 * np.pi * data["hour_of_day"] / 24) + rng.normal(0, 5, n)
    data["dam_lmp"] = dam_base

    # Price lags
    for lag_name, shift in [("dam_lag_1h", 1), ("dam_lag_4h", 4), ("dam_lag_24h", 24), ("dam_lag_168h", 168)]:
        data[lag_name] = np.roll(dam_base, shift) + rng.normal(0, 1, n)
    for lag_name, shift in [("rtm_lag_1h", 1), ("rtm_lag_4h", 4), ("rtm_lag_24h", 24), ("rtm_lag_168h", 168)]:
        data[lag_name] = np.roll(base, shift) + rng.normal(0, 3, n)

    # Rolling stats
    for prefix, price in [("dam", dam_base), ("rtm", base)]:
        for window in ["24h", "168h"]:
            data[f"{prefix}_roll_{window}_mean"] = price + rng.normal(0, 2, n)
            data[f"{prefix}_roll_{window}_std"] = np.abs(rng.normal(5, 2, n))
            data[f"{prefix}_roll_{window}_min"] = price - np.abs(rng.normal(10, 3, n))
            data[f"{prefix}_roll_{window}_max"] = price + np.abs(rng.normal(10, 3, n))

    # Cross-market
    data["dam_rtm_spread"] = rng.normal(0, 5, n)
    data["spread_roll_24h_mean"] = rng.normal(0, 3, n)
    data["spread_roll_168h_mean"] = rng.normal(0, 2, n)

    # Fuel
    data["wind_pct"] = rng.uniform(0.05, 0.35, n)
    data["solar_pct"] = rng.uniform(0.0, 0.15, n)
    data["gas_pct"] = rng.uniform(0.3, 0.6, n)
    data["nuclear_pct"] = rng.uniform(0.05, 0.12, n)
    data["coal_pct"] = rng.uniform(0.05, 0.20, n)
    data["hydro_pct"] = rng.uniform(0.0, 0.03, n)

    return pd.DataFrame(data)


class TestCreateTargets:
    def test_target_columns_created(self):
        df = _make_synthetic_data(100)
        result = create_targets(df)
        for target_name in HORIZONS:
            assert target_name in result.columns, f"Missing target: {target_name}"

    def test_1h_target_is_shifted(self):
        df = _make_synthetic_data(100)
        result = create_targets(df)
        # Row 0's 1h target should be row 1's rtm_lmp
        assert result["rtm_lmp_1h"].iloc[0] == df["rtm_lmp"].iloc[1]

    def test_4h_target_is_shifted(self):
        df = _make_synthetic_data(100)
        result = create_targets(df)
        assert result["rtm_lmp_4h"].iloc[0] == df["rtm_lmp"].iloc[4]

    def test_24h_target_is_shifted(self):
        df = _make_synthetic_data(100)
        result = create_targets(df)
        assert result["rtm_lmp_24h"].iloc[0] == df["rtm_lmp"].iloc[24]

    def test_tail_nans(self):
        df = _make_synthetic_data(100)
        result = create_targets(df)
        # Last row should have NaN for 1h target
        assert pd.isna(result["rtm_lmp_1h"].iloc[-1])
        # Last 4 rows should have NaN for 4h target
        assert pd.isna(result["rtm_lmp_4h"].iloc[-1])
        assert pd.isna(result["rtm_lmp_4h"].iloc[-4])
        # Last 24 rows should have NaN for 24h target
        assert pd.isna(result["rtm_lmp_24h"].iloc[-1])
        assert pd.isna(result["rtm_lmp_24h"].iloc[-24])


class TestAddRtmFeatures:
    def test_features_added(self):
        df = _make_synthetic_data(100)
        result = add_rtm_features(df)
        assert "rtm_volatility_24h" in result.columns
        assert "rtm_momentum_4h" in result.columns
        assert "rtm_mean_revert_signal" in result.columns

    def test_volatility_non_negative(self):
        df = _make_synthetic_data(100)
        result = add_rtm_features(df)
        # volatility = std / (|mean| + 1), std is abs so always >= 0
        assert (result["rtm_volatility_24h"] >= 0).all()

    def test_momentum_value(self):
        df = _make_synthetic_data(100)
        result = add_rtm_features(df)
        expected = df["rtm_lag_1h"] - df["rtm_lag_4h"]
        np.testing.assert_array_almost_equal(result["rtm_momentum_4h"].values, expected.values)


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
        y_pred = np.array([10.0, 22.0, 14.0, 26.0])
        m = compute_metrics(y_true, y_pred)
        assert m["directional_accuracy_pct"] == 100.0

    def test_mape_skips_near_zero(self):
        y_true = np.array([0.5, 0.0, 100.0])
        y_pred = np.array([1.0, 0.5, 110.0])
        m = compute_metrics(y_true, y_pred)
        assert m["mape_pct"] == pytest.approx(10.0, abs=0.1)


class TestNaiveBaseline:
    def test_1h_baseline(self):
        df = _make_synthetic_data(200)
        df = create_targets(df)
        df = df.dropna(subset=["rtm_lmp_1h"])
        m = naive_baseline_metrics(df, "rtm_lmp_1h", 1)
        assert "mae" in m
        assert m["mae"] > 0

    def test_24h_baseline(self):
        df = _make_synthetic_data(200)
        df = create_targets(df)
        df = df.dropna(subset=["rtm_lmp_24h"])
        m = naive_baseline_metrics(df, "rtm_lmp_24h", 24)
        assert "mae" in m
        assert m["mae"] > 0


class TestSyntheticData:
    def test_has_all_features(self):
        df = _make_synthetic_data()
        df = add_rtm_features(df)
        for col in ALL_FEATURE_COLS:
            assert col in df.columns, f"Missing feature: {col}"
        assert BASE_TARGET in df.columns

    def test_no_nans_in_features(self):
        df = _make_synthetic_data()
        df = add_rtm_features(df)
        assert df[ALL_FEATURE_COLS + [BASE_TARGET]].isna().sum().sum() == 0


class TestModelTrainSmoke:
    """Smoke test: train small models on synthetic data."""

    def _prepare_data(self, n=500):
        df = _make_synthetic_data(n)
        df = create_targets(df)
        df = add_rtm_features(df)
        df = df.dropna(subset=["rtm_lmp_1h"])
        X = df[ALL_FEATURE_COLS]
        y = df["rtm_lmp_1h"].values
        return X, y

    def test_catboost_smoke(self):
        from catboost import CatBoostRegressor, Pool

        X, y = self._prepare_data()
        cat_idx = [list(X.columns).index(c) for c in CAT_FEATURES]
        model = CatBoostRegressor(
            iterations=50, depth=4, learning_rate=0.1,
            loss_function="MAE", verbose=0, random_seed=42,
            allow_writing_files=False,
        )
        model.fit(Pool(X, y, cat_features=cat_idx))
        preds = model.predict(X)

        assert preds.shape == (len(X),)
        assert preds.dtype == np.float64
        assert not np.any(np.isnan(preds))

    def test_lightgbm_smoke(self):
        from lightgbm import LGBMRegressor

        X, y = self._prepare_data()
        cat_cols = [c for c in CAT_FEATURES if c in X.columns]
        model = LGBMRegressor(
            n_estimators=50, max_depth=4, learning_rate=0.1,
            objective="mae", random_state=42, verbosity=-1,
        )
        model.fit(X, y, categorical_feature=cat_cols)
        preds = model.predict(X)

        assert preds.shape == (len(X),)
        assert not np.any(np.isnan(preds))

    def test_predictions_reasonable_range(self):
        from catboost import CatBoostRegressor, Pool

        X, y = self._prepare_data()
        cat_idx = [list(X.columns).index(c) for c in CAT_FEATURES]
        model = CatBoostRegressor(
            iterations=100, depth=4, learning_rate=0.1,
            loss_function="MAE", verbose=0, random_seed=42,
            allow_writing_files=False,
        )
        model.fit(Pool(X, y, cat_features=cat_idx))
        preds = model.predict(X)

        assert preds.min() > y.min() - 50
        assert preds.max() < y.max() + 50

    def test_multi_horizon_shapes(self):
        """Verify each horizon produces correct prediction shapes."""
        from catboost import CatBoostRegressor, Pool

        df = _make_synthetic_data(500)
        df = create_targets(df)
        df = add_rtm_features(df)

        for target_name, shift in HORIZONS.items():
            subset = df.dropna(subset=[target_name])
            X = subset[ALL_FEATURE_COLS]
            y = subset[target_name].values

            cat_idx = [list(X.columns).index(c) for c in CAT_FEATURES]
            model = CatBoostRegressor(
                iterations=20, depth=4, learning_rate=0.1,
                loss_function="MAE", verbose=0, random_seed=42,
                allow_writing_files=False,
            )
            model.fit(Pool(X, y, cat_features=cat_idx))
            preds = model.predict(X)
            assert preds.shape == (len(X),), f"Wrong shape for {target_name}"
            assert not np.any(np.isnan(preds)), f"NaN preds for {target_name}"
