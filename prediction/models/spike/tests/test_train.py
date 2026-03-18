"""Smoke tests for spike model training with synthetic data."""

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import (
    FEATURE_COLS,
    FUEL_COLS,
    add_spike_features,
    binary_confusion_counts,
    compute_classification_metrics,
    generate_spike_labels,
)


def _make_synthetic_data(n: int = 200, spike_frac: float = 0.05, seed: int = 42):
    """Create a synthetic DataFrame mimicking training parquet schema."""
    rng = np.random.RandomState(seed)

    # Base price around 25-40 with occasional spikes
    base_price = 30 + rng.randn(n) * 5
    n_spikes = int(n * spike_frac)
    spike_idx = rng.choice(n, n_spikes, replace=False)
    base_price[spike_idx] = rng.uniform(150, 500, n_spikes)

    df = pd.DataFrame({
        "hour_of_day": np.tile(np.arange(1, 25), n // 24 + 1)[:n],
        "day_of_week": rng.randint(0, 7, n),
        "month": rng.randint(1, 13, n),
        "is_weekend": rng.randint(0, 2, n),
        "is_peak_hour": rng.randint(0, 2, n),
        "is_holiday": np.zeros(n, dtype=int),
        "is_summer": rng.randint(0, 2, n),
        # DAM lags
        "dam_lag_1h": base_price * 0.95 + rng.randn(n) * 2,
        "dam_lag_4h": base_price * 0.9 + rng.randn(n) * 3,
        "dam_lag_24h": base_price * 0.85 + rng.randn(n) * 5,
        "dam_lag_168h": base_price * 0.8 + rng.randn(n) * 8,
        # RTM lags
        "rtm_lag_1h": base_price * 1.05 + rng.randn(n) * 3,
        "rtm_lag_4h": base_price * 1.0 + rng.randn(n) * 5,
        "rtm_lag_24h": base_price * 0.95 + rng.randn(n) * 7,
        "rtm_lag_168h": base_price * 0.9 + rng.randn(n) * 10,
        # DAM rolling
        "dam_roll_24h_mean": base_price * 0.92,
        "dam_roll_24h_std": np.abs(rng.randn(n) * 5) + 1,
        "dam_roll_24h_min": base_price * 0.7,
        "dam_roll_24h_max": base_price * 1.3,
        "dam_roll_168h_mean": base_price * 0.9,
        "dam_roll_168h_std": np.abs(rng.randn(n) * 8) + 2,
        "dam_roll_168h_min": base_price * 0.5,
        "dam_roll_168h_max": base_price * 1.5,
        # RTM rolling
        "rtm_roll_24h_mean": base_price * 1.02,
        "rtm_roll_24h_std": np.abs(rng.randn(n) * 6) + 1,
        "rtm_roll_24h_min": base_price * 0.6,
        "rtm_roll_24h_max": base_price * 1.4,
        "rtm_roll_168h_mean": base_price * 1.0,
        "rtm_roll_168h_std": np.abs(rng.randn(n) * 9) + 2,
        "rtm_roll_168h_min": base_price * 0.4,
        "rtm_roll_168h_max": base_price * 1.6,
        # Cross-market
        "dam_rtm_spread": rng.randn(n) * 10,
        "spread_roll_24h_mean": rng.randn(n) * 5,
        "spread_roll_168h_mean": rng.randn(n) * 3,
        # Fuel mix
        "wind_pct": rng.uniform(0, 0.4, n),
        "solar_pct": rng.uniform(0, 0.2, n),
        "gas_pct": rng.uniform(0.3, 0.6, n),
        "nuclear_pct": np.full(n, 0.05),
        "coal_pct": rng.uniform(0.05, 0.15, n),
        "hydro_pct": rng.uniform(0, 0.03, n),
        # Targets
        "rtm_lmp": base_price,
        "dam_lmp": base_price * 0.95 + rng.randn(n) * 2,
    })

    return df


class TestAddSpikeFeatures:
    """Tests for spike-specific feature engineering."""

    def test_adds_all_spike_features(self):
        """All spike-specific feature columns should be added."""
        df = _make_synthetic_data()
        result = add_spike_features(df)
        expected = ["price_accel", "volatility_regime", "hour_spike_prob",
                     "price_momentum", "price_ratio_to_mean", "rtm_range_24h"]
        for col in expected:
            assert col in result.columns, f"Missing spike feature: {col}"

    def test_no_inf_values(self):
        """Spike features should not contain inf values."""
        df = _make_synthetic_data()
        result = add_spike_features(df)
        spike_cols = ["price_accel", "volatility_regime", "price_momentum",
                      "price_ratio_to_mean", "rtm_range_24h"]
        for col in spike_cols:
            assert not np.isinf(result[col]).any(), f"Inf values in {col}"

    def test_volatility_regime_nonnegative(self):
        """Volatility regime should be non-negative."""
        df = _make_synthetic_data()
        result = add_spike_features(df)
        assert (result["volatility_regime"] >= 0).all()

    def test_hour_spike_prob_bounded(self):
        """Hour spike probability should be in [0, 1]."""
        df = _make_synthetic_data()
        result = add_spike_features(df)
        assert (result["hour_spike_prob"] >= 0).all()
        assert (result["hour_spike_prob"] <= 1).all()

    def test_with_train_hour_probs(self):
        """Should use provided hour probs instead of computing from data."""
        df = _make_synthetic_data()
        fixed_probs = {h: 0.1 for h in range(1, 25)}
        result = add_spike_features(df, train_hour_probs=fixed_probs)
        unique_probs = result["hour_spike_prob"].unique()
        # All mapped values should be 0.1 (or 0 for unmapped hours)
        assert all(p in (0.0, 0.1) for p in unique_probs)

    def test_preserves_original_columns(self):
        """Original columns should not be modified."""
        df = _make_synthetic_data()
        orig_rtm = df["rtm_lmp"].copy()
        result = add_spike_features(df)
        pd.testing.assert_series_equal(result["rtm_lmp"], orig_rtm)

    def test_current_rtm_price_does_not_change_same_row_features(self):
        """Derived features should not depend on the current-hour RTM price."""
        df = _make_synthetic_data(n=48)
        fixed_probs = {h: 0.1 for h in range(1, 25)}

        base = add_spike_features(df, train_hour_probs=fixed_probs)

        altered = df.copy()
        altered.loc[10, "rtm_lmp"] += 1000
        altered.loc[10, "dam_rtm_spread"] += 1000
        altered.loc[10, "rtm_roll_24h_mean"] += 1000
        altered.loc[10, "rtm_roll_24h_std"] += 100
        altered.loc[10, "rtm_roll_24h_min"] -= 500
        altered.loc[10, "rtm_roll_24h_max"] += 500
        altered = add_spike_features(altered, train_hour_probs=fixed_probs)

        same_row_cols = [
            "dam_rtm_spread",
            "rtm_roll_24h_mean",
            "rtm_roll_24h_std",
            "rtm_roll_24h_min",
            "rtm_roll_24h_max",
            "volatility_regime",
            "price_momentum",
            "price_ratio_to_mean",
            "rtm_range_24h",
        ]
        for col in same_row_cols:
            assert altered.loc[10, col] == pytest.approx(base.loc[10, col], nan_ok=True)


class TestComputeClassificationMetrics:
    """Tests for the metrics computation function."""

    def test_perfect_predictions(self):
        """Perfect predictions should yield precision=recall=F1=1."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.9, 0.8, 0.1, 0.95])
        metrics = compute_classification_metrics(y_true, y_pred, y_prob)
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

    def test_all_false_positives(self):
        """All false positives should yield precision=0."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1])
        y_prob = np.array([0.9, 0.8, 0.7, 0.6])
        metrics = compute_classification_metrics(y_true, y_pred, y_prob)
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0  # no positives to recall

    def test_confusion_matrix_correct(self):
        """Confusion matrix values should be correct."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 0, 1])
        y_prob = np.array([0.1, 0.6, 0.9, 0.3, 0.2, 0.8])
        metrics = compute_classification_metrics(y_true, y_pred, y_prob)
        cm = metrics["confusion_matrix"]
        assert cm["tn"] == 2  # correctly predicted 0
        assert cm["fp"] == 1  # predicted 1, actual 0
        assert cm["fn"] == 1  # predicted 0, actual 1
        assert cm["tp"] == 2  # correctly predicted 1

    def test_all_negatives(self):
        """All-negative ground truth should not crash."""
        y_true = np.zeros(10)
        y_pred = np.zeros(10)
        y_prob = np.random.rand(10) * 0.3
        metrics = compute_classification_metrics(y_true, y_pred, y_prob)
        assert metrics["n_positive"] == 0
        assert metrics["n_negative"] == 10
        assert metrics["confusion_matrix"] == {"tn": 10, "fp": 0, "fn": 0, "tp": 0}

    def test_all_positives_confusion_matrix(self):
        """All-positive labels should still produce a full 2x2 confusion matrix."""
        y_true = np.ones(6)
        y_pred = np.array([1, 1, 0, 1, 0, 1])
        y_prob = np.array([0.9, 0.8, 0.1, 0.7, 0.2, 0.6])
        metrics = compute_classification_metrics(y_true, y_pred, y_prob)
        assert metrics["confusion_matrix"] == {"tn": 0, "fp": 0, "fn": 2, "tp": 4}

    def test_binary_confusion_counts_single_class(self):
        """The low-level helper should handle single-class inputs directly."""
        assert binary_confusion_counts(np.zeros(4), np.ones(4)) == (0, 4, 0, 0)
        assert binary_confusion_counts(np.ones(4), np.zeros(4)) == (0, 0, 4, 0)


class TestSpikeLabelsNoLeakage:
    """Regression tests for spike label leakage."""

    def test_current_hour_spike_does_not_label_same_row(self):
        """A spike in the current hour should only affect the prior row's label."""
        df = pd.DataFrame({
            "rtm_lmp": [20.0, 20.0, 20.0, 20.0, 1000.0, 20.0],
        })

        labels = generate_spike_labels(df)

        assert labels.iloc[3] == 1
        assert labels.iloc[4] == 0
        assert pd.isna(labels.iloc[-1])


class TestSyntheticTrainSmoke:
    """Smoke test: train a tiny model on synthetic data to verify the pipeline runs."""

    def test_catboost_smoke(self):
        """Verify CatBoost can train on synthetic spike data without errors."""
        from catboost import CatBoostClassifier, Pool

        df = _make_synthetic_data(n=500, spike_frac=0.1)
        df = add_spike_features(df)
        labels = generate_spike_labels(df)
        valid = labels.notna()
        df = df.loc[valid].copy()
        labels = labels.loc[valid].astype(int)

        available = [c for c in FEATURE_COLS if c in df.columns]
        X = df[available]
        y = labels.values

        cat_idx = [list(X.columns).index(c) for c in
                   ["hour_of_day", "day_of_week", "month"] if c in X.columns]

        model = CatBoostClassifier(
            iterations=10,
            depth=4,
            learning_rate=0.1,
            loss_function="Logloss",
            verbose=0,
            random_seed=42,
            allow_writing_files=False,
        )
        model.fit(Pool(X, y, cat_features=cat_idx))
        probs = model.predict_proba(X)[:, 1]

        assert probs.shape == (len(y),)
        assert (probs >= 0).all() and (probs <= 1).all()
        assert model.get_feature_importance() is not None

    def test_end_to_end_metrics(self):
        """Verify full pipeline: features -> labels -> train -> metrics."""
        from catboost import CatBoostClassifier, Pool

        df = _make_synthetic_data(n=300, spike_frac=0.1, seed=99)
        df = add_spike_features(df)
        labels = generate_spike_labels(df)
        valid = labels.notna()
        df = df.loc[valid].copy()
        labels = labels.loc[valid].astype(int)

        available = [c for c in FEATURE_COLS if c in df.columns]
        X = df[available]
        y = labels.values

        # Split
        split = int(0.8 * len(df))
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y[:split], y[split:]

        cat_idx = [list(X.columns).index(c) for c in
                   ["hour_of_day", "day_of_week", "month"] if c in X.columns]

        model = CatBoostClassifier(
            iterations=20, depth=3, learning_rate=0.1,
            verbose=0, random_seed=42, allow_writing_files=False,
        )
        model.fit(Pool(X_train, y_train, cat_features=cat_idx))

        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)

        metrics = compute_classification_metrics(y_test, preds, probs)
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "auc_roc" in metrics
        assert "confusion_matrix" in metrics
        # Sanity: metrics are in valid ranges
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
