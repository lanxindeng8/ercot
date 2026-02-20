#!/usr/bin/env python
"""
Basic tests for wind forecasting modules.

Run with: python -m pytest tests/ -v
Or simply: python tests/test_basic.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd


def test_wind_features():
    """Test wind feature engineering."""
    from features.wind_features import WindFeatureEngineer

    engineer = WindFeatureEngineer()

    # Test wind speed calculation
    u = np.array([3.0, 4.0, 0.0])
    v = np.array([4.0, 3.0, 5.0])
    ws = engineer.compute_wind_speed(u, v)

    assert np.allclose(ws, [5.0, 5.0, 5.0]), "Wind speed calculation failed"
    print("✓ Wind speed calculation OK")

    # Test power curve
    wind_speeds = np.array([0, 3, 7.5, 12, 15, 25, 30])
    power = engineer.apply_power_curve(wind_speeds)

    assert power[0] == 0, "Below cut-in should be 0"
    assert power[3] == 1.0, "At rated should be 1.0"
    assert power[6] == 0, "Above cut-out should be 0"
    print("✓ Power curve OK")


def test_temporal_features():
    """Test temporal feature engineering."""
    from features.temporal_features import TemporalFeatureEngineer

    engineer = TemporalFeatureEngineer()

    timestamps = pd.date_range('2025-01-01', periods=48, freq='h')
    features = engineer.compute_features(timestamps)

    assert 'hour_sin' in features.columns
    assert 'hour_cos' in features.columns
    assert 'is_peak_hour' in features.columns
    assert len(features) == 48
    print("✓ Temporal features OK")


def test_ramp_detection():
    """Test ramp detection."""
    from evaluation.ramp_metrics import detect_ramps, compute_ramp_metrics

    # Create synthetic data with a ramp
    values = np.array([10000, 10000, 10000, 7000, 5000, 5000, 5000, 8000, 10000])

    events = detect_ramps(values, threshold=2000, window=2, direction='down')

    assert len(events) >= 1, "Should detect ramp-down event"
    assert events[0].is_ramp_down, "Should be ramp-down"
    print("✓ Ramp detection OK")


def test_metrics():
    """Test evaluation metrics."""
    from evaluation.metrics import mae, rmse, skill_score

    y_true = np.array([100, 200, 300, 400, 500])
    y_pred = np.array([110, 190, 310, 390, 510])

    assert mae(y_true, y_pred) == 10.0, "MAE calculation failed"
    print("✓ MAE OK")

    y_baseline = np.array([150, 150, 150, 150, 150])
    ss = skill_score(y_true, y_pred, y_baseline)
    assert ss > 0, "Skill score should be positive (better than baseline)"
    print("✓ Skill score OK")


def test_gbm_model():
    """Test GBM model (if lightgbm available)."""
    try:
        from models.gbm_model import GBMWindModel
    except ImportError:
        print("⚠ LightGBM not installed, skipping")
        return

    # Create synthetic data
    np.random.seed(42)
    n_samples = 500
    X = pd.DataFrame({
        'ws_80m': np.random.uniform(0, 20, n_samples),
        'hour_sin': np.sin(2 * np.pi * np.random.randint(0, 24, n_samples) / 24),
        'hour_cos': np.cos(2 * np.pi * np.random.randint(0, 24, n_samples) / 24),
    })
    y = pd.Series(X['ws_80m'] ** 2 * 100 + np.random.normal(0, 500, n_samples))

    # Train
    model = GBMWindModel(
        quantiles=[0.1, 0.5, 0.9],
        n_estimators=50,
        verbose=-1,
    )

    X_train, X_val = X[:400], X[400:]
    y_train, y_val = y[:400], y[400:]

    model.fit(X_train, y_train, X_val, y_val)

    # Predict
    preds = model.predict(X_val)
    assert len(preds) == len(X_val), "Prediction length mismatch"

    q_preds = model.predict_quantiles(X_val)
    assert 0.5 in q_preds, "Median quantile missing"
    print("✓ GBM model OK")


def test_config():
    """Test configuration loading."""
    from utils.config import Config, load_config

    config = Config()

    assert config.data.wind_capacity == 40000.0
    assert 0.5 in config.model.quantiles
    assert config.ramp.ramp_threshold_medium == 2000.0
    print("✓ Config OK")


def test_ramp_no_solar():
    """Test ramp-down in no-solar period detection."""
    from evaluation.ramp_metrics import is_no_solar_period
    from datetime import datetime

    # 8 PM should be no-solar
    evening = datetime(2025, 1, 15, 20, 0)
    assert is_no_solar_period(evening), "8 PM should be no-solar"

    # 12 PM (noon) should have solar
    noon = datetime(2025, 1, 15, 12, 0)
    assert not is_no_solar_period(noon), "Noon should have solar"

    print("✓ No-solar detection OK")


def run_all_tests():
    """Run all tests."""
    print("=" * 50)
    print("Running Wind Forecast Tests")
    print("=" * 50)

    tests = [
        test_wind_features,
        test_temporal_features,
        test_ramp_detection,
        test_metrics,
        test_config,
        test_ramp_no_solar,
        test_gbm_model,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            print(f"\n[TEST] {test.__name__}")
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1

    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
