import numpy as np
import pandas as pd
import pytest

from utils.experiments import rolling_windows
from utils.features import build_core_feature_row, build_features
from src.predict import prepare_evaluation


def sample_data(hours=1000):
    return pd.DataFrame({"time": pd.date_range("2020-01-01", periods=hours, freq="h"),
                         "power_load": np.arange(hours, dtype=float) + 50})


@pytest.mark.parametrize("profile", ["calendar", "autoregressive", "core", "extended"])
def test_future_target_cannot_influence_current_features(profile):
    data = sample_data()
    before, names = build_features(data, profile)
    changed = data.copy()
    changed.loc[800:, "power_load"] += 100000
    after, again = build_features(changed, profile)
    assert names == again
    at_t_before = before.loc[before.time == data.time.iloc[800], names]
    at_t_after = after.loc[after.time == data.time.iloc[800], names]
    np.testing.assert_array_equal(at_t_before.to_numpy(), at_t_after.to_numpy())


def test_single_row_matches_vectorized_builder():
    data = sample_data()
    processed, names = build_features(data)
    t = data.time.iloc[500]
    history = dict(zip(data.time.iloc[:500], data.power_load.iloc[:500]))
    one = build_core_feature_row(t, history)
    many = processed.loc[processed.time == t, names]
    np.testing.assert_array_equal(one[names].to_numpy(), many.to_numpy())


def test_six_backtest_windows_have_no_overlap_or_lookahead():
    windows = list(rolling_windows(16728))
    assert len(windows) == 6
    for fold, train_idx, test_idx in windows:
        assert train_idx[-1] + 1 == test_idx[0]
        assert len(test_idx) == 672
        assert len(train_idx) >= 365 * 24
    for left, right in zip(windows, windows[1:]):
        assert left[2][-1] + 1 == right[2][0]


def test_august_forecast_uses_past_readings_only():
    data = sample_data(200)
    train = data.iloc[:150].copy()
    test = data.iloc[126:].copy()
    future, cols = prepare_evaluation(train, test)
    assert len(future) == 50
    assert future.iloc[0]["lag_1h"] == train.power_load.iloc[-1]
    assert future.iloc[0]["lag_24h"] == train.power_load.iloc[-24]
    assert future.iloc[-1]["lag_1h"] == test.power_load.iloc[-2]
