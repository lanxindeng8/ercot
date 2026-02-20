"""
Standard Evaluation Metrics for Wind Forecasting

Metrics for assessing forecast accuracy.
"""

from typing import Dict, Optional, Union
import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        MAE in same units as input
    """
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        RMSE in same units as input
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    Mean Absolute Percentage Error.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        epsilon: Small value to avoid division by zero

    Returns:
        MAPE as percentage (0-100)
    """
    return 100 * np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon)))


def nmae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    capacity: float,
) -> float:
    """
    Normalized Mean Absolute Error.

    Normalizes MAE by installed capacity for comparability.

    Args:
        y_true: Actual values (MW)
        y_pred: Predicted values (MW)
        capacity: Installed wind capacity (MW)

    Returns:
        NMAE as percentage (0-100)
    """
    return 100 * mae(y_true, y_pred) / capacity


def nrmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    capacity: float,
) -> float:
    """
    Normalized Root Mean Squared Error.

    Args:
        y_true: Actual values (MW)
        y_pred: Predicted values (MW)
        capacity: Installed wind capacity (MW)

    Returns:
        NRMSE as percentage (0-100)
    """
    return 100 * rmse(y_true, y_pred) / capacity


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Bias Error.

    Positive means over-forecasting, negative means under-forecasting.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        Bias in same units as input
    """
    return np.mean(y_pred - y_true)


def skill_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_baseline: np.ndarray,
) -> float:
    """
    Skill Score relative to a baseline forecast.

    SS = 1 - MAE(forecast) / MAE(baseline)

    Interpretation:
    - SS > 0: Better than baseline
    - SS = 0: Same as baseline
    - SS < 0: Worse than baseline
    - SS = 1: Perfect forecast

    Args:
        y_true: Actual values
        y_pred: Predicted values
        y_baseline: Baseline forecast (e.g., persistence)

    Returns:
        Skill score (-inf to 1)
    """
    mae_pred = mae(y_true, y_pred)
    mae_baseline = mae(y_true, y_baseline)

    if mae_baseline == 0:
        return 0.0

    return 1.0 - mae_pred / mae_baseline


def persistence_forecast(series: pd.Series, horizon: int = 1) -> pd.Series:
    """
    Generate persistence forecast (naive baseline).

    Uses value from `horizon` steps ago as forecast.

    Args:
        series: Time series of actual values
        horizon: Forecast horizon (steps ahead)

    Returns:
        Persistence forecast series
    """
    return series.shift(horizon)


def quantile_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantile: float,
) -> float:
    """
    Quantile Loss (Pinball Loss).

    Asymmetric loss function for quantile regression.

    Args:
        y_true: Actual values
        y_pred: Predicted quantile values
        quantile: Quantile level (0-1)

    Returns:
        Quantile loss
    """
    errors = y_true - y_pred
    return np.mean(np.maximum(quantile * errors, (quantile - 1) * errors))


def coverage(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
) -> float:
    """
    Prediction Interval Coverage.

    Fraction of actual values within prediction interval.

    Args:
        y_true: Actual values
        y_lower: Lower bound of prediction interval
        y_upper: Upper bound of prediction interval

    Returns:
        Coverage ratio (0-1)
    """
    within = (y_true >= y_lower) & (y_true <= y_upper)
    return np.mean(within)


def interval_sharpness(y_lower: np.ndarray, y_upper: np.ndarray) -> float:
    """
    Prediction Interval Sharpness.

    Average width of prediction intervals (narrower = sharper).

    Args:
        y_lower: Lower bound of prediction interval
        y_upper: Upper bound of prediction interval

    Returns:
        Average interval width
    """
    return np.mean(y_upper - y_lower)


def winkler_score(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    alpha: float = 0.1,
) -> float:
    """
    Winkler Score for prediction intervals.

    Combines coverage and sharpness into single metric.
    Lower is better.

    Args:
        y_true: Actual values
        y_lower: Lower bound (e.g., p10)
        y_upper: Upper bound (e.g., p90)
        alpha: Significance level (0.1 for 80% interval)

    Returns:
        Winkler score
    """
    width = y_upper - y_lower

    # Penalties for misses
    below = y_true < y_lower
    above = y_true > y_upper

    penalty = np.zeros_like(y_true)
    penalty[below] = 2 * (y_lower[below] - y_true[below]) / alpha
    penalty[above] = 2 * (y_true[above] - y_upper[above]) / alpha

    return np.mean(width + penalty)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_baseline: Optional[np.ndarray] = None,
    capacity: Optional[float] = None,
    quantile_preds: Optional[Dict[float, np.ndarray]] = None,
) -> Dict[str, float]:
    """
    Compute comprehensive set of evaluation metrics.

    Args:
        y_true: Actual values (MW)
        y_pred: Point predictions (MW)
        y_baseline: Baseline forecast for skill score
        capacity: Installed capacity for normalized metrics
        quantile_preds: Dict mapping quantile to predictions

    Returns:
        Dictionary of metric names to values
    """
    metrics = {
        'mae': mae(y_true, y_pred),
        'rmse': rmse(y_true, y_pred),
        'bias': bias(y_true, y_pred),
    }

    # Normalized metrics
    if capacity is not None:
        metrics['nmae'] = nmae(y_true, y_pred, capacity)
        metrics['nrmse'] = nrmse(y_true, y_pred, capacity)

    # Skill score
    if y_baseline is not None:
        metrics['skill_score'] = skill_score(y_true, y_pred, y_baseline)

    # Quantile metrics
    if quantile_preds is not None:
        for q, q_pred in quantile_preds.items():
            metrics[f'quantile_loss_q{int(q*100)}'] = quantile_loss(y_true, q_pred, q)

        # Coverage and sharpness for p10-p90 interval
        if 0.1 in quantile_preds and 0.9 in quantile_preds:
            metrics['coverage_80'] = coverage(
                y_true, quantile_preds[0.1], quantile_preds[0.9]
            )
            metrics['sharpness_80'] = interval_sharpness(
                quantile_preds[0.1], quantile_preds[0.9]
            )
            metrics['winkler_80'] = winkler_score(
                y_true, quantile_preds[0.1], quantile_preds[0.9], alpha=0.2
            )

    return metrics


def compute_metrics_by_horizon(
    y_true: pd.Series,
    y_pred: pd.Series,
    horizons: pd.Series,
    capacity: Optional[float] = None,
) -> pd.DataFrame:
    """
    Compute metrics grouped by forecast horizon.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        horizons: Forecast horizon for each prediction
        capacity: Installed capacity

    Returns:
        DataFrame with metrics by horizon
    """
    results = []

    for horizon in sorted(horizons.unique()):
        mask = horizons == horizon
        if mask.sum() == 0:
            continue

        y_t = y_true[mask].values
        y_p = y_pred[mask].values

        row = {
            'horizon': horizon,
            'n_samples': mask.sum(),
            'mae': mae(y_t, y_p),
            'rmse': rmse(y_t, y_p),
            'bias': bias(y_t, y_p),
        }

        if capacity is not None:
            row['nmae'] = nmae(y_t, y_p, capacity)

        results.append(row)

    return pd.DataFrame(results)
