"""File validation and small reporting helpers shared by the experiments."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_load_data(path):
    path = Path(path)
    data = pd.read_csv(path)
    required = {"time", "power_load"}
    if not required.issubset(data.columns):
        raise ValueError(f"{path.name} is missing {sorted(required - set(data.columns))}")
    data = data[["time", "power_load"]].copy()
    data["time"] = pd.to_datetime(data["time"], errors="raise")
    data["power_load"] = pd.to_numeric(data["power_load"], errors="raise")
    if data["time"].isna().any() or not np.isfinite(data["power_load"]).all():
        raise ValueError(f"{path.name} has missing or non-finite observations")
    if data["time"].duplicated().any():
        raise ValueError(f"{path.name} contains duplicate timestamps")
    data = data.sort_values("time").reset_index(drop=True)
    if not (data["time"].diff().iloc[1:] == pd.Timedelta(hours=1)).all():
        raise ValueError(f"{path.name} has gaps or non-hourly timestamps")
    return data


def full_series(train, test):
    """Join the 24-hour context with the training series, rejecting conflicts."""
    overlap = train.merge(test, on="time", suffixes=("_train", "_test"))
    if not np.allclose(overlap["power_load_train"], overlap["power_load_test"], atol=1e-8):
        raise ValueError("Train/test overlap has conflicting load readings")
    if test["time"].max() <= train["time"].max():
        raise ValueError("Test file has no later observations")
    if test["time"].min() > train["time"].max() + pd.Timedelta(hours=1):
        raise ValueError("Gap between training and test observations")
    data = pd.concat([train, test], ignore_index=True)
    data = data.drop_duplicates("time", keep="first").sort_values("time").reset_index(drop=True)
    if not (data["time"].diff().iloc[1:] == pd.Timedelta(hours=1)).all():
        raise ValueError("Merged observations are not a continuous hourly series")
    return data


def regression_metrics(y_true, y_pred):
    true, pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    if not len(true) or len(true) != len(pred) or not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("Metric inputs must be nonempty, equal-length finite arrays")
    mse = mean_squared_error(true, pred)
    nonzero = np.abs(true) > 1e-8
    return {
        "mae": float(mean_absolute_error(true, pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mape_pct": float(np.mean(np.abs((true[nonzero] - pred[nonzero]) / true[nonzero])) * 100) if nonzero.any() else None,
        "r2": float(r2_score(true, pred)) if len(true) >= 2 else None,
    }


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        json.dump(payload, out, indent=2, ensure_ascii=False, allow_nan=False)
        out.write("\n")


def load_json(path):
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)
