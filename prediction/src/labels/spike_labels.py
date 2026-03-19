"""
Three-layer spike labeling system for ERCOT zone-level RTM prediction.

Layer 1: SpikeEvent — binary flag for current spike conditions
Layer 2: LeadSpike  — forward-looking training target (spike within horizon)
Layer 3: Regime     — categorical market regime (Normal/Tight/Scarcity)
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


# ---------------------------------------------------------------------------
# Time conversion helpers
# ---------------------------------------------------------------------------

def _ercot_rows_to_utc(df: pd.DataFrame) -> pd.DataFrame:
    """Convert ERCOT delivery_date/delivery_hour/delivery_interval to UTC datetime.

    ERCOT times are Central (America/Chicago).  Hour-ending convention:
    delivery_hour=1, interval=1 → 00:00 CT (start of 15-min interval).
    """
    minutes = (df["delivery_hour"] - 1) * 60 + (df["delivery_interval"] - 1) * 15
    ct = pd.to_datetime(df["delivery_date"]) + pd.to_timedelta(minutes, unit="m")
    ct = ct.dt.tz_localize("America/Chicago", ambiguous="NaT", nonexistent="shift_forward")
    df = df.copy()
    df["time"] = ct.dt.tz_convert("UTC")
    return df


# ---------------------------------------------------------------------------
# Layer 1: SpikeEvent
# ---------------------------------------------------------------------------

def compute_spike_events(
    df: pd.DataFrame,
    settlement_point: str,
    hub: str = "HB_HUBAVG",
    p_hi: float = 400.0,
    s_hi: float = 50.0,
    min_consecutive: int = 2,
) -> pd.Series:
    """Compute binary spike-event flags for a single zone.

    SpikeEvent = price_condition AND spread_condition, sustained for
    ``min_consecutive`` consecutive 15-min intervals.

    price_condition:  (P_z >= p_hi) OR (P_z >= rolling_Q99_30d)
    spread_condition: abs(P_z - P_hub) >= s_hi

    Parameters
    ----------
    df : DataFrame
        Must contain columns ``settlement_point``, ``hub``, and a
        DatetimeIndex (UTC, 15-min frequency).
    settlement_point : str
        Column name for zone LMP.
    hub : str
        Column name for hub LMP.
    p_hi : float
        Absolute price threshold ($/MWh).
    s_hi : float
        Absolute spread threshold ($/MWh).
    min_consecutive : int
        Minimum consecutive intervals the conditions must hold.

    Returns
    -------
    pd.Series[bool]
        True where a spike event is detected, aligned to *df.index*.
    """
    p_z = df[settlement_point].astype(float)
    p_hub = df[hub].astype(float)

    # Rolling 99th percentile over 30 days (2880 15-min intervals)
    rolling_q99 = p_z.rolling(window=2880, min_periods=96).quantile(0.99)

    price_cond = (p_z >= p_hi) | (p_z >= rolling_q99)
    spread_cond = (p_z - p_hub).abs() >= s_hi
    raw = price_cond & spread_cond

    # Require min_consecutive True in a row
    if min_consecutive <= 1:
        return raw

    # Rolling sum — if the last N are all True the sum == N
    consec = raw.astype(int).rolling(window=min_consecutive, min_periods=min_consecutive).sum()
    sustained = consec >= min_consecutive

    # Back-fill so the *first* interval of the run is also marked True
    # We need to mark all intervals that are part of a sustained run.
    # Forward-propagate the sustained flag backward within each run.
    result = pd.Series(False, index=df.index)
    # Mark every True in raw that belongs to a run of >= min_consecutive
    groups = (raw != raw.shift()).cumsum()
    for _, grp in raw.groupby(groups):
        if grp.all() and len(grp) >= min_consecutive:
            result.loc[grp.index] = True

    return result


# ---------------------------------------------------------------------------
# Layer 2: LeadSpike (training target — forward-looking)
# ---------------------------------------------------------------------------

def compute_lead_spike(
    spike_events: pd.Series,
    horizon_minutes: int = 60,
    interval_minutes: int = 15,
) -> pd.Series:
    """Forward-looking label: will a spike occur within the next *horizon*?

    LeadSpike(t) = 1 if any SpikeEvent in (t, t + H].

    **This is the TRAINING TARGET and must NOT be used as a feature.**

    Parameters
    ----------
    spike_events : Series[bool]
        Output of :func:`compute_spike_events`.
    horizon_minutes : int
        Look-ahead window in minutes.
    interval_minutes : int
        Data frequency in minutes.

    Returns
    -------
    pd.Series[bool]
    """
    steps = horizon_minutes // interval_minutes
    # Shift spike_events backward (i.e., rolling-max on future values).
    # A reverse rolling max is equivalent to: for each t, check if any
    # spike_events[t+1 .. t+steps] is True.
    shifted = spike_events.astype(int)
    # Reverse, rolling max, reverse back
    lead = (
        shifted
        .iloc[::-1]
        .rolling(window=steps, min_periods=1)
        .max()
        .iloc[::-1]
        .shift(-1)  # exclude current interval — only future
    )
    return lead.fillna(0).astype(bool)


# ---------------------------------------------------------------------------
# Layer 3: Regime
# ---------------------------------------------------------------------------

def compute_regime(
    df: pd.DataFrame,
    settlement_point: str,
    hub: str = "HB_HUBAVG",
    spike_events: pd.Series = None,
    p_mid: float = 150.0,
    s_mid: float = 20.0,
) -> pd.Series:
    """Classify each interval into a market regime.

    - **Scarcity**: spike_events is True
    - **Tight**: (P_z >= p_mid) OR (abs(P_z - P_hub) >= s_mid)
    - **Normal**: everything else

    Returns
    -------
    pd.Series
        Categorical with values ``['Normal', 'Tight', 'Scarcity']``.
    """
    p_z = df[settlement_point].astype(float)
    p_hub = df[hub].astype(float)

    tight_cond = (p_z >= p_mid) | ((p_z - p_hub).abs() >= s_mid)

    regime = pd.Series("Normal", index=df.index, dtype="object")
    regime[tight_cond] = "Tight"
    if spike_events is not None:
        regime[spike_events] = "Scarcity"

    return pd.Categorical(regime, categories=["Normal", "Tight", "Scarcity"], ordered=True)


# ---------------------------------------------------------------------------
# Convenience: label all zones from the archive DB
# ---------------------------------------------------------------------------

def _load_rtm_wide(
    db_path: Path,
    settlement_points: list[str],
    start_date: str = "2015-01-01",
    end_date: str = "2026-03-19",
) -> pd.DataFrame:
    """Load RTM LMP history and pivot to wide format with UTC time index."""
    placeholders = ",".join(["?"] * len(settlement_points))
    query = f"""
        SELECT delivery_date, delivery_hour, delivery_interval,
               settlement_point, lmp
        FROM rtm_lmp_hist
        WHERE settlement_point IN ({placeholders})
          AND delivery_date BETWEEN ? AND ?
          AND repeated_hour = 0
        ORDER BY delivery_date, delivery_hour, delivery_interval
    """
    params = settlement_points + [start_date, end_date]

    logger.info("Loading RTM LMP data from {}", db_path)
    with sqlite3.connect(str(db_path)) as conn:
        raw = pd.read_sql_query(query, conn, params=params)

    logger.info("Loaded {:,} rows", len(raw))

    raw = _ercot_rows_to_utc(raw)
    # Drop rows where tz conversion produced NaT (DST fall-back ambiguous)
    raw = raw.dropna(subset=["time"])

    wide = raw.pivot_table(
        index="time", columns="settlement_point", values="lmp", aggfunc="first"
    )
    wide = wide.sort_index()
    return wide


def label_all_zones(
    db_path: Path,
    settlement_points: list[str],
    start_date: str = "2015-01-01",
    end_date: str = "2026-03-19",
    horizon_minutes: int = 60,
    hub: str = "HB_HUBAVG",
) -> pd.DataFrame:
    """Load RTM LMP data and compute all three label layers for every zone.

    Returns a long-format DataFrame with columns:
        time, settlement_point, lmp, hub_lmp, spread,
        spike_event, lead_spike_60, regime
    """
    sps = list(settlement_points)
    if hub not in sps:
        sps.append(hub)

    wide = _load_rtm_wide(db_path, sps, start_date, end_date)

    if hub not in wide.columns:
        raise ValueError(f"Hub '{hub}' not found in loaded data")

    frames = []
    for sp in settlement_points:
        if sp == hub:
            continue
        if sp not in wide.columns:
            logger.warning("Settlement point {} not in data — skipping", sp)
            continue

        sub = wide[[sp, hub]].dropna()
        if sub.empty:
            continue

        logger.info("Labeling {} ({:,} intervals)", sp, len(sub))

        spike_ev = compute_spike_events(sub, sp, hub)
        lead_sp = compute_lead_spike(spike_ev, horizon_minutes=horizon_minutes)
        regime = compute_regime(sub, sp, hub, spike_events=spike_ev)

        out = pd.DataFrame({
            "time": sub.index,
            "settlement_point": sp,
            "lmp": sub[sp].values,
            "hub_lmp": sub[hub].values,
            "spread": (sub[sp] - sub[hub]).values,
            "spike_event": spike_ev.values,
            "lead_spike_60": lead_sp.values,
            "regime": regime.values,
        })
        frames.append(out)

    result = pd.concat(frames, ignore_index=True)
    result["spike_event"] = result["spike_event"].astype(bool)
    result["lead_spike_60"] = result["lead_spike_60"].astype(bool)
    logger.info(
        "Labeled {:,} intervals across {} zones — {:,} spike events, {:,} lead targets",
        len(result),
        result["settlement_point"].nunique(),
        result["spike_event"].sum(),
        result["lead_spike_60"].sum(),
    )
    return result
