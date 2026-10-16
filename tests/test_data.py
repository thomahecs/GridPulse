import pandas as pd
import pytest

from utils.common import full_series, load_load_data, regression_metrics


def test_input_rejects_gaps_and_duplicates(tmp_path):
    data = pd.DataFrame({"time": ["2020-01-01 00:00", "2020-01-01 02:00"], "power_load": [1.0, 2.0]})
    file = tmp_path / "sample.csv"
    data.to_csv(file, index=False)
    with pytest.raises(ValueError, match="gaps"):
        load_load_data(file)
    data.loc[1, "time"] = data.loc[0, "time"]
    data.to_csv(file, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        load_load_data(file)


def test_conflicting_train_test_overlap_is_rejected():
    t = pd.date_range("2020-01-01", periods=30, freq="h")
    training = pd.DataFrame({"time": t, "power_load": range(30)})
    testing = pd.DataFrame({"time": t[29:], "power_load": [999]})
    with pytest.raises(ValueError, match="conflicting"):
        full_series(training, testing)


def test_metrics():
    metrics = regression_metrics([10, 20], [13, 21])
    assert metrics["mae"] == pytest.approx(2)
    assert metrics["rmse"] == pytest.approx((5) ** .5)
