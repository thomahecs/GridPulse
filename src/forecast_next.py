"""Make one forward forecast beyond the last known timestamp (no true label required)."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from xgboost import XGBRegressor
from utils.common import file_sha256, full_series, load_json, load_load_data, save_json
from utils.features import build_features


def forecast_next():
    data_dir = ROOT / "data"
    model_dir = ROOT / "model"
    metadata = load_json(model_dir / "metadata.json")
    for name, key in (("train.csv", "training_data_sha256"), ("test.csv", "test_data_sha256")):
        if file_sha256(data_dir / name) != metadata[key]:
            raise ValueError(f"{name} changed since fitting; retrain before forecasting")
    history = full_series(load_load_data(data_dir / "train.csv"), load_load_data(data_dir / "test.csv"))
    next_time = history.time.iloc[-1] + pd.Timedelta(hours=1)

    # The placeholder is never a feature: every load feature is shifted by at
    # least one hour. It only lets us reuse the same feature-building function.
    future = pd.concat([history, pd.DataFrame([{"time": next_time, "power_load": 0.0}])], ignore_index=True)
    processed, names = build_features(future, metadata["feature_profile"])
    if names != metadata["feature_names"] or processed.time.iloc[-1] != next_time:
        raise ValueError("The future feature row does not match the saved model")
    model = XGBRegressor()
    model.load_model(str(model_dir / "gridpulse_xgb.json"))
    features = processed.iloc[-1:][names]
    prediction = float(model.predict(features)[0])
    if metadata.get("prediction_target") == "delta_from_previous_hour":
        prediction += float(processed.iloc[-1]["lag_1h"])
    result = {
        "last_observed_timestamp": str(history.time.iloc[-1]),
        "forecast_timestamp": str(next_time),
        "forecast_horizon_hours": 1,
        "predicted_load_dataset_units": prediction,
        "evaluation_note": "Forward prediction only; the actual next-hour load is not present in the archive",
    }
    save_json(result, ROOT / "reports/next_hour_forecast.json")
    print(f"Next-hour forecast for {next_time}: {prediction:.3f} (dataset load units)")
    return result


if __name__ == "__main__":
    forecast_next()
