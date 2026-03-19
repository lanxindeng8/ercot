"""
Derived weather features for ERCOT zone-level spike prediction.

All derived features use .shift(1) to avoid lookahead bias — they only
reference past data relative to the current timestamp.
"""

import numpy as np
import pandas as pd


def compute_weather_features(station_data: pd.DataFrame) -> pd.DataFrame:
    """Compute derived weather features from raw station data.

    Expects columns: time, temperature_2m, wind_speed_10m, wind_direction_10m.
    The 'time' column is parsed to datetime and used as the index.

    All features are lagged by 1 hour (shift(1)) to prevent data leakage.

    Returns a copy of the input with additional columns:
        t_anom, delta_t_1h, delta_t_3h, wind_chill, cold_front
    """
    df = station_data.copy()

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time").sort_index()

    # Ensure uniform hourly frequency (guards against DST gaps or missing rows)
    df = df.asfreq("h")

    t = df["temperature_2m"]
    ws = df["wind_speed_10m"]
    wd = df["wind_direction_10m"]

    # --- Temperature anomaly: T - rolling 30-day same-hour mean ---
    # Group by hour-of-day, compute 30-day (720h / 24 = 30 points per hour) rolling mean
    hour = df.index.hour
    rolling_mean = t.copy() * np.nan
    for h in range(24):
        mask = hour == h
        series = t[mask]
        # 30-day rolling mean within same hour
        rm = series.rolling(window=30, min_periods=1).mean()
        rolling_mean[mask] = rm
    df["t_anom"] = (t - rolling_mean).shift(1)

    # --- Hourly temperature change ---
    df["delta_t_1h"] = (t - t.shift(1)).shift(1)

    # --- 3-hour temperature change ---
    df["delta_t_3h"] = (t - t.shift(3)).shift(1)

    # --- Wind chill (NWS formula) ---
    # Convert: T_f = T_c * 9/5 + 32, V_mph = V_kmh * 0.621371
    t_f = t * 9.0 / 5.0 + 32.0
    v_mph = ws * 0.621371

    wc_f = (
        35.74
        + 0.6215 * t_f
        - 35.75 * v_mph**0.16
        + 0.4275 * t_f * v_mph**0.16
    )
    # Convert back to Celsius (NWS formula outputs Fahrenheit)
    wc_c = (wc_f - 32.0) * 5.0 / 9.0
    # Only valid when T < 10°C and wind > 4.8 km/h
    valid = (t < 10.0) & (ws > 4.8)
    wind_chill_raw = np.where(valid, wc_c, t)
    df["wind_chill"] = pd.Series(wind_chill_raw, index=df.index).shift(1)

    # --- Cold front flag ---
    # Rapid cooling (delta_t_3h < -5) AND wind from north (315-360 or 0-45)
    delta_3h_raw = t - t.shift(3)
    north_wind = ((wd >= 315) | (wd <= 45)) & (wd.notna())
    cold_front_raw = (delta_3h_raw < -5.0) & north_wind
    df["cold_front"] = cold_front_raw.astype(int).shift(1)

    return df
