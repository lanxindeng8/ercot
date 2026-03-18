"""Tests for spike label generation correctness."""

import numpy as np
import pandas as pd
import pytest

# Add parent to path so we can import train module
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import (
    ROLLING_WINDOW,
    SPIKE_MULTIPLIER,
    SPIKE_THRESHOLD,
    generate_spike_labels,
)


def _make_price_series(prices: list[float], name: str = "rtm_lmp") -> pd.DataFrame:
    """Helper to create a DataFrame with rtm_lmp from a list of prices."""
    return pd.DataFrame({name: prices})


class TestSpikeLabels:
    """Tests for the generate_spike_labels function."""

    def test_below_threshold_no_spike(self):
        """Prices always below 100 should never be spikes."""
        df = _make_price_series([20, 30, 25, 40, 50, 35, 20, 30] * 10)
        labels = generate_spike_labels(df)
        assert labels.sum() == 0, "No prices above threshold should yield no spikes"

    def test_above_absolute_threshold(self):
        """Price > 100 with low rolling mean should be a spike."""
        # 30 hours of low prices, then a spike
        prices = [20.0] * 30 + [150.0]
        df = _make_price_series(prices)
        labels = generate_spike_labels(df)
        # Rolling mean of ~20, so threshold = max(100, 3*20) = 100
        # 150 > 100, so last element should be spike
        assert labels.iloc[-1] == 1, "150 > max(100, 3*20=60) should be spike"

    def test_above_rolling_mean_threshold(self):
        """Price > 3 * rolling_mean when rolling_mean > 33.33 uses multiplier."""
        # High base prices: rolling mean includes current value
        # 30 values of 80 + [500]: rolling_24h includes 23*80 + 500 = 2340/24 ≈ 97.5
        # threshold = max(100, 3*97.5) = max(100, 292.5) = 292.5
        # 500 > 292.5 => spike
        prices = [80.0] * 30 + [500.0]
        df = _make_price_series(prices)
        labels = generate_spike_labels(df)
        assert labels.iloc[-1] == 1, "500 > max(100, ~292) should be spike"

    def test_between_thresholds_no_spike(self):
        """Price above 100 but below 3x rolling mean should not be spike."""
        # Rolling mean ~80, threshold = max(100, 240) = 240
        prices = [80.0] * 30 + [200.0]
        df = _make_price_series(prices)
        labels = generate_spike_labels(df)
        assert labels.iloc[-1] == 0, "200 < max(100, 3*80=240) should NOT be spike"

    def test_spike_labels_are_binary(self):
        """Labels should only be 0 or 1."""
        prices = [20.0] * 10 + [500.0] * 3 + [20.0] * 10
        df = _make_price_series(prices)
        labels = generate_spike_labels(df)
        assert set(labels.unique()).issubset({0, 1}), "Labels must be binary"

    def test_rolling_window_effect(self):
        """After sustained high prices, the rolling mean rises so threshold increases."""
        # 24 hours at 200 => rolling mean = 200, threshold = max(100, 600) = 600
        prices = [200.0] * 24 + [500.0]
        df = _make_price_series(prices)
        labels = generate_spike_labels(df)
        # 500 < 600, so NOT a spike (high baseline shifts threshold)
        assert labels.iloc[-1] == 0, "500 < 3*200=600 when base is high"

    def test_empty_dataframe(self):
        """Empty DataFrame should return empty labels."""
        df = pd.DataFrame({"rtm_lmp": pd.Series([], dtype=float)})
        labels = generate_spike_labels(df)
        assert len(labels) == 0

    def test_single_row(self):
        """Single row should use that value as rolling mean."""
        df = _make_price_series([500.0])
        labels = generate_spike_labels(df)
        # rolling mean = 500 (window=1), threshold = max(100, 1500) = 1500
        # 500 < 1500, so not a spike
        assert labels.iloc[0] == 0

    def test_negative_prices_no_spike(self):
        """Negative prices (happens in ERCOT) should never be spikes."""
        prices = [-50.0, -20.0, 10.0, -30.0, 5.0]
        df = _make_price_series(prices)
        labels = generate_spike_labels(df)
        assert labels.sum() == 0

    def test_known_spike_pattern(self):
        """Verify a realistic spike pattern produces expected labels."""
        # Normal prices with a clear spike event
        normal = [25.0 + np.sin(i / 4) * 5 for i in range(48)]  # 2 days normal
        spike = [300.0, 450.0, 250.0]  # 3-hour spike
        recovery = [40.0] * 10
        prices = normal + spike + recovery
        df = _make_price_series(prices)
        labels = generate_spike_labels(df)

        # Spike hours should be labeled
        spike_start = len(normal)
        assert labels.iloc[spike_start] == 1, "300 should be a spike vs ~25 mean"
        assert labels.iloc[spike_start + 1] == 1, "450 should be a spike"
        # Recovery should not be spike
        assert labels.iloc[-1] == 0, "Recovery should not be spike"
