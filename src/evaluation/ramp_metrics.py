"""
Ramp Detection Metrics for Wind Forecasting

Specialized metrics for evaluating ramp event detection.
Focus on ramp-down events during no-solar periods.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from loguru import logger

try:
    from astral import LocationInfo
    from astral.sun import sun
    ASTRAL_AVAILABLE = True
except ImportError:
    ASTRAL_AVAILABLE = False


@dataclass
class RampEvent:
    """Represents a detected ramp event."""
    start_idx: int
    end_idx: int
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    magnitude: float  # MW change (negative for ramp-down)
    duration_hours: float
    rate: float  # MW/hour
    is_ramp_down: bool
    is_no_solar: bool = False


def detect_ramps(
    values: np.ndarray,
    timestamps: Optional[pd.DatetimeIndex] = None,
    threshold: float = 2000,
    window: int = 3,
    direction: str = 'both',
) -> List[RampEvent]:
    """
    Detect ramp events in a time series.

    A ramp is defined as a change exceeding threshold within window hours.

    Args:
        values: Wind generation values (MW)
        timestamps: Optional timestamps for the values
        threshold: Minimum absolute change to qualify as ramp (MW)
        window: Window size in hours/steps for ramp detection
        direction: 'both', 'up', or 'down'

    Returns:
        List of detected RampEvent objects
    """
    events = []
    n = len(values)

    if n < window + 1:
        return events

    # Compute changes over window
    changes = np.zeros(n)
    for i in range(window, n):
        changes[i] = values[i] - values[i - window]

    # Find ramp events
    i = window
    while i < n:
        change = changes[i]
        is_ramp_up = change >= threshold
        is_ramp_down = change <= -threshold

        if direction == 'up' and not is_ramp_up:
            i += 1
            continue
        if direction == 'down' and not is_ramp_down:
            i += 1
            continue
        if direction == 'both' and not (is_ramp_up or is_ramp_down):
            i += 1
            continue

        # Found a ramp - determine extent
        start_idx = i - window
        end_idx = i

        # Extend end if ramp continues
        while end_idx + 1 < n:
            next_change = values[end_idx + 1] - values[end_idx]
            if is_ramp_down and next_change < 0:
                end_idx += 1
            elif is_ramp_up and next_change > 0:
                end_idx += 1
            else:
                break

        magnitude = values[end_idx] - values[start_idx]
        duration = (end_idx - start_idx)

        event = RampEvent(
            start_idx=start_idx,
            end_idx=end_idx,
            start_time=timestamps[start_idx] if timestamps is not None else None,
            end_time=timestamps[end_idx] if timestamps is not None else None,
            magnitude=magnitude,
            duration_hours=duration,
            rate=magnitude / duration if duration > 0 else 0,
            is_ramp_down=magnitude < 0,
        )

        events.append(event)

        # Skip past this event
        i = end_idx + 1

    return events


def match_events(
    true_events: List[RampEvent],
    pred_events: List[RampEvent],
    tolerance_hours: int = 2,
) -> Tuple[int, int, int]:
    """
    Match predicted events to actual events.

    Args:
        true_events: Actual ramp events
        pred_events: Predicted ramp events
        tolerance_hours: Maximum time difference for matching

    Returns:
        Tuple of (hits, misses, false_alarms)
    """
    matched_true = set()
    matched_pred = set()

    # Try to match each predicted event to an actual event
    for i, pred in enumerate(pred_events):
        for j, true in enumerate(true_events):
            if j in matched_true:
                continue

            # Check if events overlap or are within tolerance
            time_diff = abs(pred.start_idx - true.start_idx)

            if time_diff <= tolerance_hours:
                # Also check direction match
                if pred.is_ramp_down == true.is_ramp_down:
                    matched_true.add(j)
                    matched_pred.add(i)
                    break

    hits = len(matched_true)
    misses = len(true_events) - hits
    false_alarms = len(pred_events) - len(matched_pred)

    return hits, misses, false_alarms


def compute_ramp_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: Optional[pd.DatetimeIndex] = None,
    threshold: float = 2000,
    window: int = 3,
    direction: str = 'both',
    tolerance_hours: int = 2,
) -> Dict[str, float]:
    """
    Compute ramp detection metrics.

    Args:
        y_true: Actual wind generation (MW)
        y_pred: Predicted wind generation (MW)
        timestamps: Optional timestamps
        threshold: Ramp threshold (MW)
        window: Detection window (hours)
        direction: 'both', 'up', or 'down'
        tolerance_hours: Matching tolerance

    Returns:
        Dictionary with POD, FAR, CSI, etc.
    """
    # Detect events
    true_events = detect_ramps(y_true, timestamps, threshold, window, direction)
    pred_events = detect_ramps(y_pred, timestamps, threshold, window, direction)

    # Match events
    hits, misses, false_alarms = match_events(true_events, pred_events, tolerance_hours)

    # Compute metrics
    total_actual = hits + misses
    total_predicted = hits + false_alarms

    # POD: Probability of Detection (hit rate)
    pod = hits / total_actual if total_actual > 0 else 0.0

    # FAR: False Alarm Ratio
    far = false_alarms / total_predicted if total_predicted > 0 else 0.0

    # CSI: Critical Success Index (Threat Score)
    # CSI = hits / (hits + misses + false_alarms)
    denominator = hits + misses + false_alarms
    csi = hits / denominator if denominator > 0 else 0.0

    # Bias: ratio of predicted to actual events
    bias = total_predicted / total_actual if total_actual > 0 else 0.0

    # Miss rate
    miss_rate = misses / total_actual if total_actual > 0 else 0.0

    return {
        'n_actual_events': total_actual,
        'n_predicted_events': total_predicted,
        'hits': hits,
        'misses': misses,
        'false_alarms': false_alarms,
        'pod': pod,  # Probability of Detection
        'far': far,  # False Alarm Ratio
        'csi': csi,  # Critical Success Index
        'bias': bias,  # Forecast bias
        'miss_rate': miss_rate,
    }


def is_no_solar_period(
    timestamp: datetime,
    lat: float = 31.0,  # Texas center latitude
    lon: float = -100.0,  # Texas center longitude
) -> bool:
    """
    Check if timestamp is during no-solar period (before sunrise or after sunset).

    Args:
        timestamp: Datetime to check
        lat: Latitude
        lon: Longitude

    Returns:
        True if no solar generation expected
    """
    if not ASTRAL_AVAILABLE:
        # Fallback: use fixed hours (6am-7pm local time approximation)
        hour = timestamp.hour
        return hour < 7 or hour >= 19

    location = LocationInfo(
        name="Texas",
        region="USA",
        timezone="America/Chicago",
        latitude=lat,
        longitude=lon,
    )

    try:
        s = sun(location.observer, date=timestamp.date())
        sunrise = s['sunrise'].replace(tzinfo=None)
        sunset = s['sunset'].replace(tzinfo=None)

        # Add buffer for solar ramp-up/down
        sunrise_buffer = sunrise.replace(hour=sunrise.hour + 1)
        sunset_buffer = sunset.replace(hour=sunset.hour - 1)

        ts_naive = timestamp.replace(tzinfo=None) if timestamp.tzinfo else timestamp
        return ts_naive < sunrise_buffer or ts_naive > sunset_buffer
    except Exception:
        # Fallback on error
        hour = timestamp.hour
        return hour < 7 or hour >= 19


def evaluate_ramp_down_in_no_solar(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: pd.DatetimeIndex,
    threshold: float = -2000,
    window: int = 3,
    tolerance_hours: int = 2,
) -> Dict[str, float]:
    """
    Evaluate ramp-down detection specifically during no-solar periods.

    This is the most critical scenario:
    - Wind generation dropping rapidly
    - During hours when solar cannot compensate (evening/night/early morning)
    - System must rely on gas, prices can spike

    Args:
        y_true: Actual wind generation (MW)
        y_pred: Predicted wind generation (MW)
        timestamps: Timestamps for the data
        threshold: Ramp-down threshold (negative MW, e.g., -2000)
        window: Detection window (hours)
        tolerance_hours: Matching tolerance

    Returns:
        Dictionary with specialized metrics for no-solar ramp-down events
    """
    # Find indices during no-solar periods
    no_solar_mask = np.array([is_no_solar_period(ts) for ts in timestamps])

    # Detect all ramp-down events
    true_events = detect_ramps(
        y_true, timestamps, abs(threshold), window, direction='down'
    )
    pred_events = detect_ramps(
        y_pred, timestamps, abs(threshold), window, direction='down'
    )

    # Filter to no-solar periods
    def is_event_in_no_solar(event: RampEvent) -> bool:
        if event.start_time is not None:
            return is_no_solar_period(event.start_time)
        # Use index if no timestamp
        return no_solar_mask[event.start_idx] if event.start_idx < len(no_solar_mask) else False

    true_no_solar = [e for e in true_events if is_event_in_no_solar(e)]
    pred_no_solar = [e for e in pred_events if is_event_in_no_solar(e)]

    # Mark events
    for e in true_no_solar:
        e.is_no_solar = True
    for e in pred_no_solar:
        e.is_no_solar = True

    # Match events
    hits, misses, false_alarms = match_events(true_no_solar, pred_no_solar, tolerance_hours)

    total_actual = hits + misses
    total_predicted = hits + false_alarms

    # Core metrics
    pod = hits / total_actual if total_actual > 0 else 0.0
    far = false_alarms / total_predicted if total_predicted > 0 else 0.0
    csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else 0.0
    miss_rate = misses / total_actual if total_actual > 0 else 0.0

    # Compute lead time for hits (how early we detected)
    lead_times = []
    for i, pred in enumerate(pred_no_solar):
        for j, true in enumerate(true_no_solar):
            time_diff = true.start_idx - pred.start_idx
            if 0 <= time_diff <= tolerance_hours and pred.is_ramp_down == true.is_ramp_down:
                lead_times.append(time_diff)
                break

    avg_lead_time = np.mean(lead_times) if lead_times else 0.0

    # Evening peak analysis (17:00-21:00)
    def is_evening_peak(ts):
        return 17 <= ts.hour <= 21

    evening_mask = np.array([is_evening_peak(ts) for ts in timestamps])
    true_evening = [e for e in true_no_solar
                    if e.start_time is not None and is_evening_peak(e.start_time)]
    n_evening_events = len(true_evening)

    # Compute severity (average magnitude of missed events)
    missed_magnitudes = []
    matched_true_indices = set()
    for i, pred in enumerate(pred_no_solar):
        for j, true in enumerate(true_no_solar):
            if abs(pred.start_idx - true.start_idx) <= tolerance_hours:
                matched_true_indices.add(j)
                break

    for j, true in enumerate(true_no_solar):
        if j not in matched_true_indices:
            missed_magnitudes.append(abs(true.magnitude))

    avg_missed_magnitude = np.mean(missed_magnitudes) if missed_magnitudes else 0.0

    return {
        # Event counts
        'n_actual_ramp_down_no_solar': total_actual,
        'n_predicted_ramp_down_no_solar': total_predicted,
        'n_evening_peak_events': n_evening_events,

        # Core detection metrics
        'hits': hits,
        'misses': misses,
        'false_alarms': false_alarms,

        # Rates
        'pod_no_solar': pod,  # Critical: must be high (> 0.8)
        'far_no_solar': far,
        'csi_no_solar': csi,
        'miss_rate_no_solar': miss_rate,  # Must be low (< 0.2)

        # Lead time
        'avg_lead_time_hours': avg_lead_time,

        # Severity
        'avg_missed_magnitude_mw': avg_missed_magnitude,
    }


def compute_ramp_metrics_by_magnitude(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: Optional[pd.DatetimeIndex] = None,
    thresholds: List[float] = [1000, 2000, 3000, 4000],
    window: int = 3,
) -> pd.DataFrame:
    """
    Compute ramp metrics at multiple thresholds.

    Helps understand model performance across ramp severities.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        timestamps: Optional timestamps
        thresholds: List of ramp thresholds to evaluate
        window: Detection window

    Returns:
        DataFrame with metrics at each threshold
    """
    results = []

    for thresh in thresholds:
        # Ramp-up
        up_metrics = compute_ramp_metrics(
            y_true, y_pred, timestamps,
            threshold=thresh, window=window, direction='up'
        )
        up_metrics['threshold'] = thresh
        up_metrics['direction'] = 'up'
        results.append(up_metrics)

        # Ramp-down
        down_metrics = compute_ramp_metrics(
            y_true, y_pred, timestamps,
            threshold=thresh, window=window, direction='down'
        )
        down_metrics['threshold'] = thresh
        down_metrics['direction'] = 'down'
        results.append(down_metrics)

    return pd.DataFrame(results)


def generate_ramp_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: pd.DatetimeIndex,
    capacity: float = 40000,  # ERCOT wind capacity ~40 GW
) -> str:
    """
    Generate a comprehensive ramp detection report.

    Args:
        y_true: Actual wind generation
        y_pred: Predicted wind generation
        timestamps: Timestamps
        capacity: Installed capacity

    Returns:
        Formatted report string
    """
    lines = ["=" * 60]
    lines.append("WIND RAMP DETECTION EVALUATION REPORT")
    lines.append("=" * 60)

    # Overall ramp metrics
    lines.append("\n1. OVERALL RAMP DETECTION (threshold=2000 MW, window=3h)")
    lines.append("-" * 40)

    overall = compute_ramp_metrics(y_true, y_pred, timestamps, threshold=2000)
    lines.append(f"   Actual ramp events:    {overall['n_actual_events']}")
    lines.append(f"   Predicted ramp events: {overall['n_predicted_events']}")
    lines.append(f"   Hits:                  {overall['hits']}")
    lines.append(f"   Misses:                {overall['misses']}")
    lines.append(f"   False alarms:          {overall['false_alarms']}")
    lines.append(f"   POD (hit rate):        {overall['pod']:.3f}")
    lines.append(f"   FAR (false alarm):     {overall['far']:.3f}")
    lines.append(f"   CSI (threat score):    {overall['csi']:.3f}")

    # Ramp-down during no-solar (CRITICAL)
    lines.append("\n2. RAMP-DOWN DURING NO-SOLAR PERIODS (CRITICAL)")
    lines.append("-" * 40)

    no_solar = evaluate_ramp_down_in_no_solar(y_true, y_pred, timestamps)
    lines.append(f"   Actual no-solar ramp-down events:    {no_solar['n_actual_ramp_down_no_solar']}")
    lines.append(f"   Predicted no-solar ramp-down events: {no_solar['n_predicted_ramp_down_no_solar']}")
    lines.append(f"   Evening peak events (17-21h):        {no_solar['n_evening_peak_events']}")
    lines.append(f"   ")
    lines.append(f"   POD (no-solar):        {no_solar['pod_no_solar']:.3f}  {'[GOOD]' if no_solar['pod_no_solar'] > 0.8 else '[NEEDS IMPROVEMENT]'}")
    lines.append(f"   Miss rate (no-solar):  {no_solar['miss_rate_no_solar']:.3f}  {'[GOOD]' if no_solar['miss_rate_no_solar'] < 0.2 else '[HIGH RISK]'}")
    lines.append(f"   Avg lead time:         {no_solar['avg_lead_time_hours']:.1f} hours")
    lines.append(f"   Avg missed magnitude:  {no_solar['avg_missed_magnitude_mw']:.0f} MW")

    # Ramp-up metrics
    lines.append("\n3. RAMP-UP DETECTION")
    lines.append("-" * 40)

    ramp_up = compute_ramp_metrics(y_true, y_pred, timestamps, threshold=2000, direction='up')
    lines.append(f"   Actual ramp-up events: {ramp_up['n_actual_events']}")
    lines.append(f"   POD:                   {ramp_up['pod']:.3f}")
    lines.append(f"   CSI:                   {ramp_up['csi']:.3f}")

    # Ramp-down metrics
    lines.append("\n4. RAMP-DOWN DETECTION (ALL PERIODS)")
    lines.append("-" * 40)

    ramp_down = compute_ramp_metrics(y_true, y_pred, timestamps, threshold=2000, direction='down')
    lines.append(f"   Actual ramp-down events: {ramp_down['n_actual_events']}")
    lines.append(f"   POD:                     {ramp_down['pod']:.3f}")
    lines.append(f"   CSI:                     {ramp_down['csi']:.3f}")

    lines.append("\n" + "=" * 60)
    lines.append("KEY TARGETS:")
    lines.append("  - POD (no-solar ramp-down) > 0.80")
    lines.append("  - Miss rate (no-solar) < 0.20")
    lines.append("  - Lead time > 2 hours")
    lines.append("=" * 60)

    return "\n".join(lines)
