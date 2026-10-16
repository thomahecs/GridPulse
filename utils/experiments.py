"""One split policy shared by training, backtesting and ablations."""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

DEFAULT_PARAMS = {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.05}


def make_xgb(params=None):
    return XGBRegressor(
        objective="reg:squarederror", tree_method="hist", random_state=42,
        n_jobs=1, verbosity=0, **(params or DEFAULT_PARAMS)
    )


def make_ridge():
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0))


def rolling_windows(size, n_splits=6, window_hours=28 * 24, min_history_hours=365 * 24):
    """Six adjacent windows at the end of the development data; no future training rows."""
    first = size - n_splits * window_hours
    if first < min_history_hours:
        raise ValueError("Not enough historical hours for the requested rolling folds")
    for fold in range(n_splits):
        start = first + fold * window_hours
        end = start + window_hours
        yield fold + 1, np.arange(start), np.arange(start, end)
