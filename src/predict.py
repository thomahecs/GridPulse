"""Evaluate true next-hour rolling forecasts on the unseen August 2015 window."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from utils.common import file_sha256, full_series, load_json, load_load_data, regression_metrics, save_json
from utils.features import build_features

DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"
FIG_DIR = DATA_DIR / "fig"
MODEL_DIR = ROOT / "model"


def plot_forecast(results, width):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(results.time, results.actual_load, label="Observed", linewidth=0.9)
    ax.plot(results.time, results.predicted_load, label="Predicted", linewidth=0.9)
    ax.fill_between(results.time, results.predicted_load - width, results.predicted_load + width,
                    alpha=0.15, label="90th percentile backtest absolute error (illustrative)")
    ax.set(xlabel="Time", ylabel="Load (dataset units)", title="August 2015 next-hour forecasts")
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "future_forecast.png", dpi=150)
    plt.close(fig)


def prepare_evaluation(train_data, test_data, profile="core"):
    """Lag values at t come from observed load strictly before t.

    At the close of each forecast hour the true reading is assumed to arrive,
    so this is one-hour-ahead rolling evaluation, not a 744-hour open-loop forecast.
    """
    combined = full_series(train_data, test_data)
    processed, names = build_features(combined, profile)
    cutoff = train_data["time"].max()
    evaluation = processed.loc[processed.time > cutoff].reset_index(drop=True)
    if len(evaluation) != len(test_data.loc[test_data.time > cutoff]):
        raise ValueError("Some future timestamps have insufficient history")
    return evaluation, names


def run_forecast():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    train_data = load_load_data(DATA_DIR / "train.csv")
    test_data = load_load_data(DATA_DIR / "test.csv")
    metadata = load_json(MODEL_DIR / "metadata.json")
    if file_sha256(DATA_DIR / "train.csv") != metadata["training_data_sha256"]:
        raise ValueError("Training file has changed since model training; rerun train.py")
    if file_sha256(DATA_DIR / "test.csv") != metadata["test_data_sha256"]:
        raise ValueError("Test file has changed since model training; rerun train.py")
    model = XGBRegressor()
    model.load_model(str(MODEL_DIR / "gridpulse_xgb.json"))
    ridge = joblib.load(MODEL_DIR / "gridpulse_ridge.joblib")  # Project-generated, trusted artifact only.
    future, names = prepare_evaluation(train_data, test_data, profile=metadata["feature_profile"])
    if names != metadata["feature_names"]:
        raise ValueError("Saved model's features do not match the current feature builder")
    pred = model.predict(future[names]) + future["lag_1h"].to_numpy() if metadata.get("prediction_target") == "delta_from_previous_hour" else model.predict(future[names])
    ridge_pred = ridge.predict(future[names])
    results = pd.DataFrame({
        "time": future.time,
        "actual_load": future.power_load,
        "predicted_load": pred,
        "persistence_1h": future["lag_1h"],
        "seasonal_naive_24h": future["lag_24h"],
        "ridge_predicted_load": ridge_pred,
    })
    results["abs_error"] = (results.actual_load - results.predicted_load).abs()
    results.to_csv(REPORT_DIR / "future_predictions.csv", index=False)
    # Hour-of-day diagnostics make "peak hour" assertions checkable.
    hourly = (results.assign(hour=results.time.dt.hour)
              .groupby("hour", as_index=False)
              .agg(mae=("abs_error", "mean"), mean_load=("actual_load", "mean"), samples=("abs_error", "size")))
    hourly.to_csv(REPORT_DIR / "hourly_error_analysis.csv", index=False)
    scores = {
        "model": regression_metrics(results.actual_load, pred),
        "persistence_1h": regression_metrics(results.actual_load, results.persistence_1h),
        "seasonal_naive_24h": regression_metrics(results.actual_load, results.seasonal_naive_24h),
        "ridge": regression_metrics(results.actual_load, ridge_pred),
    }
    model_mae = scores["model"]["mae"]
    payload = {
        "evaluation": "out-of-time rolling next-hour; actual loads are observed before the next prediction",
        "forecast_start": str(results.time.iloc[0]),
        "forecast_end": str(results.time.iloc[-1]),
        "rows": len(results),
        "mean_observed_load": float(results.actual_load.mean()),
        **scores,
        "mae_pct_of_mean_load": model_mae / float(results.actual_load.mean()) * 100,
        "mae_improvement_vs_persistence_pct": 100 * (scores["persistence_1h"]["mae"] - model_mae) / scores["persistence_1h"]["mae"],
        "mae_improvement_vs_seasonal_naive_pct": 100 * (scores["seasonal_naive_24h"]["mae"] - model_mae) / scores["seasonal_naive_24h"]["mae"],
        "mae_improvement_vs_ridge_pct": 100 * (scores["ridge"]["mae"] - model_mae) / scores["ridge"]["mae"],
    }
    save_json(payload, REPORT_DIR / "future_metrics.json")
    # Replace the obsolete random-split benchmark file from the old project.
    save_json({"evaluation": "August 2015 chronological holdout", "model": scores["model"],
               "persistence_1h": scores["persistence_1h"], "seasonal_naive_24h": scores["seasonal_naive_24h"],
               "ridge": scores["ridge"]}, REPORT_DIR / "benchmark_metrics.json")
    plot_forecast(results, float(metadata["prediction_interval_abs_error_q90"]))
    print("\nOut-of-time rolling evaluation:")
    for name, value in scores.items():
        print(f"  {name:20s} MAE = {value['mae']:.3f}")
    print(f"  Model MAE / mean load: {payload['mae_pct_of_mean_load']:.2f}%")
    print("Reports saved under reports/; forecast plot under data/fig/")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run_forecast()
