from pathlib import Path
import json
import pandas as pd
import pytest

from utils.common import regression_metrics

ROOT = Path(__file__).resolve().parents[1]


def test_archived_forecast_report_is_recomputable():
    data = pd.read_csv(ROOT / "reports/future_predictions.csv")
    report = json.loads((ROOT / "reports/future_metrics.json").read_text())
    assert len(data) == 744 == report["rows"]
    assert regression_metrics(data.actual_load, data.predicted_load)["mae"] == pytest.approx(report["model"]["mae"])
    assert regression_metrics(data.actual_load, data.persistence_1h)["mae"] == pytest.approx(report["persistence_1h"]["mae"])


def test_backtest_scores_cover_six_non_overlapping_future_windows():
    rows = pd.read_csv(ROOT / "reports/temporal_backtest.csv")
    assert rows.fold.tolist() == list(range(1, 7))
    assert (rows.rows == 672).all()
    assert (pd.to_datetime(rows.train_end) < pd.to_datetime(rows.valid_start)).all()
    assert (pd.to_datetime(rows.valid_end.iloc[:-1]).to_numpy() + pd.Timedelta(hours=1) == pd.to_datetime(rows.valid_start.iloc[1:]).to_numpy()).all()
