#!/usr/bin/env python
"""
Tests for trained wind prediction models.

Validates that checkpoints load correctly, produce reasonable predictions,
and meet performance targets on test data.

Run with: python -m pytest tests/test_trained_model.py -v
Or: python tests/test_trained_model.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent.parent
CHECKPOINT_DIR = BASE_DIR / 'checkpoints'
DATA_DIR = BASE_DIR / 'data'


def _load_test_data():
    """Load test split from augmented features."""
    df = pd.read_parquet(DATA_DIR / 'features_augmented.parquet')
    df['valid_time'] = pd.to_datetime(df['valid_time'])
    test = df[(df['valid_time'] >= '2024-12-01') & (df['valid_time'] <= '2024-12-31')].copy()
    exclude = ['valid_time', 'init_time', 'wind_generation', 'lead_time', 'timestamp']
    feature_cols = [c for c in test.columns if c not in exclude]
    return test, feature_cols


# ========== Checkpoint Loading Tests ==========

def test_gbm_checkpoint_exists():
    """GBM checkpoint files exist."""
    model_dir = CHECKPOINT_DIR / 'gbm_model'
    assert model_dir.exists(), f"GBM checkpoint dir missing: {model_dir}"
    assert (model_dir / 'metadata.json').exists(), "metadata.json missing"
    assert (model_dir / 'model_q0.10.txt').exists(), "q0.10 model missing"
    assert (model_dir / 'model_q0.50.txt').exists(), "q0.50 model missing"
    assert (model_dir / 'model_q0.90.txt').exists(), "q0.90 model missing"
    print("  GBM checkpoint files OK")


def test_ensemble_checkpoint_exists():
    """Ensemble checkpoint files exist."""
    model_dir = CHECKPOINT_DIR / 'ensemble_best_model'
    assert model_dir.exists(), f"Ensemble checkpoint dir missing: {model_dir}"
    assert (model_dir / 'metadata.json').exists()
    assert (CHECKPOINT_DIR / 'ensemble_config.json').exists(), "ensemble_config.json missing"
    print("  Ensemble checkpoint files OK")


def test_gbm_load_from_checkpoint():
    """GBM model loads from checkpoint and has correct metadata."""
    from models.gbm_model import GBMWindModel

    model = GBMWindModel.load(str(CHECKPOINT_DIR / 'gbm_model'))
    assert model.quantiles == [0.1, 0.5, 0.9], f"Wrong quantiles: {model.quantiles}"
    assert model.feature_names is not None, "Feature names not loaded"
    assert len(model.feature_names) > 0, "Empty feature names"
    assert len(model.models) == 3, f"Expected 3 quantile models, got {len(model.models)}"
    print(f"  GBM loaded: {len(model.feature_names)} features, 3 quantile models")


def test_metrics_json_valid():
    """Metrics JSON files are valid and contain expected fields."""
    metrics_path = CHECKPOINT_DIR / 'gbm_model_metrics.json'
    assert metrics_path.exists()

    with open(metrics_path) as f:
        data = json.load(f)

    assert 'model_name' in data
    assert 'metrics' in data
    assert 'standard' in data['metrics']
    assert 'mae' in data['metrics']['standard']
    assert 'rmse' in data['metrics']['standard']
    print(f"  Metrics JSON valid: MAE={data['metrics']['standard']['mae']:.1f}")


# ========== Prediction Tests ==========

def test_gbm_predictions_shape():
    """GBM predictions have correct shape."""
    from models.gbm_model import GBMWindModel

    model = GBMWindModel.load(str(CHECKPOINT_DIR / 'gbm_model'))
    test_df, feature_cols = _load_test_data()

    preds = model.predict(test_df[feature_cols])
    assert len(preds) == len(test_df), f"Shape mismatch: {len(preds)} vs {len(test_df)}"
    print(f"  Predictions shape OK: {len(preds)} samples")


def test_gbm_predictions_reasonable():
    """GBM predictions are in reasonable range for wind generation."""
    from models.gbm_model import GBMWindModel

    model = GBMWindModel.load(str(CHECKPOINT_DIR / 'gbm_model'))
    test_df, feature_cols = _load_test_data()

    preds = model.predict(test_df[feature_cols])

    # Wind generation should be non-negative and under capacity (40 GW)
    assert np.all(preds >= -500), f"Predictions too negative: min={preds.min():.0f}"
    assert np.all(preds <= 50000), f"Predictions too high: max={preds.max():.0f}"
    assert np.mean(preds) > 0, "Mean prediction should be positive"
    print(f"  Predictions range: {preds.min():.0f} to {preds.max():.0f} MW (mean={preds.mean():.0f})")


def test_quantile_ordering():
    """Quantile predictions maintain correct ordering (P10 <= P50 <= P90)."""
    from models.gbm_model import GBMWindModel

    model = GBMWindModel.load(str(CHECKPOINT_DIR / 'gbm_model'))
    test_df, feature_cols = _load_test_data()

    q_preds = model.predict_quantiles(test_df[feature_cols])
    p10 = q_preds[0.1]
    p50 = q_preds[0.5]
    p90 = q_preds[0.9]

    # Allow small violations due to independent models
    violations_10_50 = np.mean(p10 > p50 + 10)
    violations_50_90 = np.mean(p50 > p90 + 10)

    assert violations_10_50 < 0.05, f"Too many P10>P50 violations: {violations_10_50:.1%}"
    assert violations_50_90 < 0.05, f"Too many P50>P90 violations: {violations_50_90:.1%}"
    print(f"  Quantile ordering OK (violations: P10>P50={violations_10_50:.1%}, P50>P90={violations_50_90:.1%})")


# ========== Performance Target Tests ==========

def test_gbm_mae_target():
    """GBM MAE meets target (NMAE < 15%)."""
    from models.gbm_model import GBMWindModel

    model = GBMWindModel.load(str(CHECKPOINT_DIR / 'gbm_model'))
    test_df, feature_cols = _load_test_data()

    preds = model.predict(test_df[feature_cols])
    y_test = test_df['wind_generation'].values
    capacity = 40000

    model_mae = np.mean(np.abs(y_test - preds))
    nmae_pct = 100 * model_mae / capacity

    assert nmae_pct < 15, f"NMAE {nmae_pct:.1f}% exceeds 15% target"
    print(f"  NMAE: {nmae_pct:.2f}% (target < 15%)")


def test_ramp_detection_pod():
    """Ramp detection POD meets target (> 0.80)."""
    from models.gbm_model import GBMWindModel
    from evaluation.ramp_metrics import compute_ramp_metrics

    model = GBMWindModel.load(str(CHECKPOINT_DIR / 'gbm_model'))
    test_df, feature_cols = _load_test_data()

    preds = model.predict(test_df[feature_cols])
    y_test = test_df['wind_generation'].values
    timestamps = pd.DatetimeIndex(test_df['valid_time'])

    ramp_metrics = compute_ramp_metrics(
        y_true=y_test, y_pred=preds,
        timestamps=timestamps,
        threshold=2000, window=3,
    )

    pod = ramp_metrics.get('pod', 0)
    assert pod >= 0.80, f"Ramp POD {pod:.2f} below 0.80 target"
    print(f"  Ramp POD: {pod:.3f} (target >= 0.80)")


def test_no_solar_ramp_down_pod():
    """No-solar ramp-down POD meets critical target (> 0.80)."""
    from models.gbm_model import GBMWindModel
    from evaluation.ramp_metrics import evaluate_ramp_down_in_no_solar

    model = GBMWindModel.load(str(CHECKPOINT_DIR / 'gbm_model'))
    test_df, feature_cols = _load_test_data()

    preds = model.predict(test_df[feature_cols])
    y_test = test_df['wind_generation'].values
    timestamps = pd.DatetimeIndex(test_df['valid_time'])

    no_solar = evaluate_ramp_down_in_no_solar(
        y_true=y_test, y_pred=preds,
        timestamps=timestamps,
        threshold=-2000,
    )

    pod = no_solar.get('pod_no_solar', 0)
    miss_rate = no_solar.get('miss_rate_no_solar', 1)
    n_events = no_solar.get('n_actual_ramp_down_no_solar', 0)

    assert pod >= 0.80, f"No-solar ramp-down POD {pod:.2f} below 0.80 target"
    assert miss_rate <= 0.20, f"Miss rate {miss_rate:.2f} above 0.20 target"
    print(f"  No-solar POD: {pod:.3f}, miss rate: {miss_rate:.3f} ({n_events} events)")


# ========== Runner ==========

def run_all_tests():
    """Run all trained model tests."""
    print("=" * 55)
    print("Wind Model Tests — Trained Model Validation")
    print("=" * 55)

    tests = [
        ("Checkpoint: GBM exists", test_gbm_checkpoint_exists),
        ("Checkpoint: Ensemble exists", test_ensemble_checkpoint_exists),
        ("Checkpoint: GBM loads", test_gbm_load_from_checkpoint),
        ("Checkpoint: Metrics JSON", test_metrics_json_valid),
        ("Prediction: Shape", test_gbm_predictions_shape),
        ("Prediction: Reasonable range", test_gbm_predictions_reasonable),
        ("Prediction: Quantile ordering", test_quantile_ordering),
        ("Target: NMAE < 15%", test_gbm_mae_target),
        ("Target: Ramp POD >= 0.80", test_ramp_detection_pod),
        ("Target: No-solar POD >= 0.80", test_no_solar_ramp_down_pod),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            print(f"\n[TEST] {name}")
            test_fn()
            passed += 1
            print(f"  PASSED")
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1

    print("\n" + "=" * 55)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print("=" * 55)

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
