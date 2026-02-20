"""
Evaluation Module for Wind Forecasting

Metrics for assessing forecast accuracy and ramp detection.
"""

from .metrics import (
    mae,
    rmse,
    mape,
    nmae,
    nrmse,
    bias,
    skill_score,
    persistence_forecast,
    quantile_loss,
    coverage,
    interval_sharpness,
    winkler_score,
    compute_all_metrics,
    compute_metrics_by_horizon,
)

from .ramp_metrics import (
    RampEvent,
    detect_ramps,
    match_events,
    compute_ramp_metrics,
    is_no_solar_period,
    evaluate_ramp_down_in_no_solar,
    compute_ramp_metrics_by_magnitude,
    generate_ramp_report,
)

__all__ = [
    # Standard metrics
    'mae',
    'rmse',
    'mape',
    'nmae',
    'nrmse',
    'bias',
    'skill_score',
    'persistence_forecast',
    'quantile_loss',
    'coverage',
    'interval_sharpness',
    'winkler_score',
    'compute_all_metrics',
    'compute_metrics_by_horizon',
    # Ramp metrics
    'RampEvent',
    'detect_ramps',
    'match_events',
    'compute_ramp_metrics',
    'is_no_solar_period',
    'evaluate_ramp_down_in_no_solar',
    'compute_ramp_metrics_by_magnitude',
    'generate_ramp_report',
]
