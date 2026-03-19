"""Tests for spike_model: splits, training, metrics, event recall."""

import numpy as np
import pandas as pd
import pytest

from prediction.src.models.spike_model import (
    LABEL_COLS,
    TARGET,
    TRAIN_END,
    VAL_END,
    compute_event_recall,
    compute_metrics,
    get_feature_cols,
    split_data,
    train_model,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_df():
    """Create a synthetic DataFrame mimicking spike feature parquets."""
    np.random.seed(42)
    n = 2000
    dates = pd.date_range("2024-06-01", periods=n, freq="15min", tz="UTC")
    df = pd.DataFrame(
        {
            "lmp_lag1": np.random.randn(n) * 30 + 50,
            "lmp_lag4": np.random.randn(n) * 30 + 50,
            "lmp_mean_4": np.random.randn(n) * 20 + 50,
            "lmp_std_96": np.abs(np.random.randn(n) * 10),
            "spread_lag1": np.random.randn(n) * 5,
            "hour_of_day": np.tile(np.arange(24), n // 24 + 1)[:n],
            "day_of_week": np.tile(np.arange(7), n // 7 + 1)[:n],
            "month": np.ones(n, dtype=int) * 6,
            "is_weekend": np.zeros(n, dtype=int),
            "temp_2m": np.random.randn(n) * 5 + 30,
            "spike_event": np.zeros(n),
            "lead_spike_60": np.zeros(n),
            "regime": ["Normal"] * n,
        },
        index=pd.DatetimeIndex(dates, name="time"),
    )
    # Inject some positive labels
    spike_idx = np.random.choice(n, size=max(1, n // 50), replace=False)
    df.iloc[spike_idx, df.columns.get_loc("lead_spike_60")] = 1.0
    df.iloc[spike_idx, df.columns.get_loc("spike_event")] = 1.0
    return df


@pytest.fixture
def wide_date_df():
    """DataFrame spanning train/val/test periods."""
    np.random.seed(123)
    dates = pd.date_range("2024-01-01", "2026-06-01", freq="1h", tz="UTC")
    n = len(dates)
    df = pd.DataFrame(
        {
            "feat_a": np.random.randn(n),
            "feat_b": np.random.randn(n),
            "feat_c": np.random.randn(n),
            "spike_event": np.zeros(n),
            "lead_spike_60": np.zeros(n),
            "regime": ["Normal"] * n,
        },
        index=pd.DatetimeIndex(dates, name="time"),
    )
    # Sprinkle positives across all periods
    pos_idx = np.random.choice(n, size=n // 100, replace=False)
    df.iloc[pos_idx, df.columns.get_loc("lead_spike_60")] = 1.0
    return df


# ---------------------------------------------------------------------------
# Split tests
# ---------------------------------------------------------------------------

class TestSplitData:

    def test_no_overlap(self, wide_date_df):
        train, val, test = split_data(wide_date_df)
        assert train.index.max() < val.index.min()
        assert val.index.max() < test.index.min()

    def test_chronological(self, wide_date_df):
        train, val, test = split_data(wide_date_df)
        assert train.index.max() < TRAIN_END
        assert val.index.min() >= TRAIN_END
        assert val.index.max() < VAL_END
        assert test.index.min() >= VAL_END

    def test_all_data_accounted(self, wide_date_df):
        train, val, test = split_data(wide_date_df)
        assert len(train) + len(val) + len(test) == len(wide_date_df)

    def test_no_future_leak(self, wide_date_df):
        train, val, test = split_data(wide_date_df)
        # No training data from val/test periods
        assert (train.index < TRAIN_END).all()
        # No val data from test period
        assert (val.index < VAL_END).all()


# ---------------------------------------------------------------------------
# Feature column tests
# ---------------------------------------------------------------------------

class TestFeatureCols:

    def test_excludes_labels(self, synthetic_df):
        feat_cols = get_feature_cols(synthetic_df)
        for lbl in LABEL_COLS:
            assert lbl not in feat_cols

    def test_includes_features(self, synthetic_df):
        feat_cols = get_feature_cols(synthetic_df)
        assert "lmp_lag1" in feat_cols
        assert "hour_of_day" in feat_cols


# ---------------------------------------------------------------------------
# Model training test
# ---------------------------------------------------------------------------

class TestTrainModel:

    def test_fit_on_synthetic(self, synthetic_df):
        """Model should fit on synthetic data and produce probabilities."""
        feature_cols = get_feature_cols(synthetic_df)
        # Use first 80% as train, rest as val
        split_point = int(len(synthetic_df) * 0.8)
        train_df = synthetic_df.iloc[:split_point]
        val_df = synthetic_df.iloc[split_point:]

        model = train_model(
            train_df, val_df, feature_cols,
            params={"num_leaves": 8, "learning_rate": 0.1},
            num_boost_round=20,
            early_stopping_rounds=5,
        )

        preds = model.predict(val_df[feature_cols])
        assert len(preds) == len(val_df)
        assert (preds >= 0).all() and (preds <= 1).all()


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

class TestMetrics:

    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.9, 0.8, 0.1, 0.95])
        m = compute_metrics(y_true, y_prob)
        assert m["roc_auc"] > 0.9
        assert m["pr_auc"] > 0.8
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0

    def test_random_predictions(self):
        np.random.seed(0)
        y_true = np.array([0] * 100 + [1] * 5)
        y_prob = np.random.rand(105)
        m = compute_metrics(y_true, y_prob)
        assert "roc_auc" in m
        assert "pr_auc" in m
        assert m["n_positive"] == 5
        assert m["n_total"] == 105

    def test_single_class(self):
        y_true = np.zeros(10)
        y_prob = np.random.rand(10)
        m = compute_metrics(y_true, y_prob)
        assert np.isnan(m["roc_auc"])


# ---------------------------------------------------------------------------
# Event-level recall tests
# ---------------------------------------------------------------------------

class TestEventRecall:

    def test_single_event_detected(self):
        """One event with one high-prob prediction → detected."""
        ts = pd.date_range("2025-06-01", periods=10, freq="15min", tz="UTC")
        y_true = np.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 0])
        y_prob = np.array([0.1, 0.1, 0.6, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1])
        r = compute_event_recall(ts, y_true, y_prob)
        assert r["n_events"] == 1
        assert r["events_detected"] == 1
        assert r["event_recall"] == 1.0

    def test_single_event_missed(self):
        """One event with all predictions below threshold → missed."""
        ts = pd.date_range("2025-06-01", periods=10, freq="15min", tz="UTC")
        y_true = np.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 0])
        y_prob = np.array([0.1, 0.1, 0.2, 0.3, 0.4, 0.1, 0.1, 0.1, 0.1, 0.1])
        r = compute_event_recall(ts, y_true, y_prob)
        assert r["n_events"] == 1
        assert r["events_detected"] == 0
        assert r["event_recall"] == 0.0

    def test_two_events(self):
        """Two events separated by > 60 min gap."""
        ts = pd.DatetimeIndex([
            "2025-06-01 00:00", "2025-06-01 00:15", "2025-06-01 00:30",
            # gap > 60 min
            "2025-06-01 03:00", "2025-06-01 03:15",
        ], tz="UTC")
        y_true = np.array([1, 1, 1, 1, 1])
        y_prob = np.array([0.8, 0.6, 0.4, 0.3, 0.2])
        r = compute_event_recall(ts, y_true, y_prob)
        assert r["n_events"] == 2
        assert r["events_detected"] == 1  # only first event detected

    def test_no_positives(self):
        ts = pd.date_range("2025-06-01", periods=5, freq="15min", tz="UTC")
        y_true = np.zeros(5)
        y_prob = np.zeros(5)
        r = compute_event_recall(ts, y_true, y_prob)
        assert r["n_events"] == 0
        assert np.isnan(r["event_recall"])
