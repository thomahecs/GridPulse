import numpy as np
import pandas as pd


CORE_LAGS = [1, 2, 3, 24]
EXTENDED_LAGS = [1, 2, 3, 6, 12, 24, 48, 168]


def _calendar_features(times):
    features = pd.DataFrame(index=times.index)

    for hour in range(24):
        features["hour_{:02d}".format(hour)] = (times.dt.hour == hour).astype(np.int8)

    for month in range(1, 13):
        features["month_{:02d}".format(month)] = (times.dt.month == month).astype(np.int8)

    return features


def build_features(data, profile="core"):
    """Turn the hourly load series into supervised regression rows.

    Every load-derived feature is shifted before it reaches the model. This
    keeps the target at time t out of the feature vector for time t.
    """
    frame = data[["time", "power_load"]].copy().reset_index(drop=True)
    times = frame["time"]
    load = frame["power_load"].astype(float)
    features = _calendar_features(times)

    if profile == "calendar":
        pass
    elif profile == "autoregressive":
        features = pd.DataFrame(index=frame.index)
        for lag in CORE_LAGS:
            features["lag_{}h".format(lag)] = load.shift(lag)
    elif profile in ("core", "extended"):
        for lag in CORE_LAGS:
            features["lag_{}h".format(lag)] = load.shift(lag)

        if profile == "extended":
            features["day_of_week"] = times.dt.dayofweek.astype(np.int8)
            features["is_weekend"] = (times.dt.dayofweek >= 5).astype(np.int8)
            features["hour_sin"] = np.sin(2.0 * np.pi * times.dt.hour / 24.0)
            features["hour_cos"] = np.cos(2.0 * np.pi * times.dt.hour / 24.0)
            features["dow_sin"] = np.sin(2.0 * np.pi * times.dt.dayofweek / 7.0)
            features["dow_cos"] = np.cos(2.0 * np.pi * times.dt.dayofweek / 7.0)

            for lag in EXTENDED_LAGS:
                name = "lag_{}h".format(lag)
                if name not in features:
                    features[name] = load.shift(lag)

            shifted = load.shift(1)
            for window in (3, 6, 24):
                features["rolling_mean_{}h".format(window)] = shifted.rolling(window).mean()
            features["rolling_std_24h"] = shifted.rolling(24).std()
            features["delta_1h_vs_24h"] = features["lag_1h"] - features["lag_24h"]
    else:
        raise ValueError("Unknown feature profile: {}".format(profile))

    result = pd.concat([frame, features], axis=1).dropna().reset_index(drop=True)
    feature_names = list(features.columns)
    return result, feature_names


def build_core_feature_row(timestamp, history):
    """Create one next-hour feature row using only already observed loads."""
    timestamp = pd.Timestamp(timestamp)
    values = {}

    for hour in range(24):
        values["hour_{:02d}".format(hour)] = int(timestamp.hour == hour)
    for month in range(1, 13):
        values["month_{:02d}".format(month)] = int(timestamp.month == month)

    for lag in CORE_LAGS:
        lag_time = timestamp - pd.Timedelta(hours=lag)
        if lag_time not in history:
            raise KeyError("Missing historical load for {}".format(lag_time))
        values["lag_{}h".format(lag)] = float(history[lag_time])

    columns = ["hour_{:02d}".format(h) for h in range(24)]
    columns += ["month_{:02d}".format(m) for m in range(1, 13)]
    columns += ["lag_{}h".format(lag) for lag in CORE_LAGS]

    return pd.DataFrame([[values[col] for col in columns]], columns=columns)
