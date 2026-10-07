from pathlib import Path
import json
import numpy as np
import pandas as pd


def load_load_data(path):
    """Read a load file, normalize timestamps, and enforce one row per hour."""
    path = Path(path)
    data = pd.read_csv(path)

    required = {"time", "power_load"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError("Missing required columns: {}".format(sorted(missing)))

    data = data[["time", "power_load"]].copy()
    data["time"] = pd.to_datetime(data["time"], errors="raise")
    data["power_load"] = pd.to_numeric(data["power_load"], errors="raise")
    data = data.sort_values("time").drop_duplicates(subset="time", keep="last")
    data = data.reset_index(drop=True)
    return data


def regression_metrics(y_true, y_pred):
    """Return a small set of regression metrics used across the project."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    nonzero = np.abs(y_true) > 1e-8
    mape = np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
    mse = mean_squared_error(y_true, y_pred)

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mape_pct": float(mape),
        "r2": float(r2_score(y_true, y_pred)),
    }


def save_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
