"""Chronological features for one-hour-ahead load forecasts."""

import numpy as np
import pandas as pd

CORE_LAGS = (1, 2, 3, 24)
EXTENDED_LAGS = (1, 2, 3, 6, 12, 24, 48, 168)


def build_features(data, profile="core"):
    """Every lag and rolling statistic is computed from observations before t.

    Input must be hourly and ordered. External weather forecasts are not silently
    substituted with actual future temperatures.
    """
    if profile not in ("calendar", "autoregressive", "core", "extended"):
        raise ValueError(f"Unknown feature profile: {profile}")
    frame = data[["time", "power_load"]].reset_index(drop=True).copy()
    times = frame["time"]
    load = frame["power_load"].astype(float)
    columns = {}
    if profile != "autoregressive":
        columns.update({f"hour_{h:02d}": (times.dt.hour == h).astype("int8") for h in range(24)})
        columns.update({f"month_{m:02d}": (times.dt.month == m).astype("int8") for m in range(1, 13)})
    if profile != "calendar":
        lags = EXTENDED_LAGS if profile == "extended" else CORE_LAGS
        columns.update({f"lag_{h}h": load.shift(h) for h in lags})
    if profile == "extended":
        columns.update({
            "day_of_week": times.dt.dayofweek.astype("int8"),
            "is_weekend": (times.dt.dayofweek >= 5).astype("int8"),
            "hour_sin": np.sin(2 * np.pi * times.dt.hour / 24),
            "hour_cos": np.cos(2 * np.pi * times.dt.hour / 24),
            "dow_sin": np.sin(2 * np.pi * times.dt.dayofweek / 7),
            "dow_cos": np.cos(2 * np.pi * times.dt.dayofweek / 7),
            "rolling_mean_3h": load.shift(1).rolling(3).mean(),
            "rolling_mean_6h": load.shift(1).rolling(6).mean(),
            "rolling_mean_24h": load.shift(1).rolling(24).mean(),
            "rolling_std_24h": load.shift(1).rolling(24).std(),
            "delta_1h_vs_24h": load.shift(1) - load.shift(24),
        })
    features = pd.DataFrame(columns, index=frame.index)
    names = list(features.columns)
    out = pd.concat([frame, features], axis=1).dropna().reset_index(drop=True)
    return out, names


def build_core_feature_row(timestamp, history):
    """Build a single prediction row; useful for testing streaming inference."""
    t = pd.Timestamp(timestamp)
    columns = {f"hour_{h:02d}": int(t.hour == h) for h in range(24)}
    columns.update({f"month_{m:02d}": int(t.month == m) for m in range(1, 13)})
    for lag in CORE_LAGS:
        previous = t - pd.Timedelta(hours=lag)
        if previous not in history:
            raise KeyError(f"Missing historical load at {previous}")
        columns[f"lag_{lag}h"] = float(history[previous])
    return pd.DataFrame([columns])
