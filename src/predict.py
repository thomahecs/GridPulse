import datetime
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.common import load_json, load_load_data, regression_metrics, save_json
from utils.features import build_core_feature_row
from utils.log import Logger

DATA_DIR = ROOT / "data"
FIG_DIR = DATA_DIR / "fig"
MODEL_DIR = ROOT / "model"
REPORT_DIR = ROOT / "reports"
FORECAST_START = pd.Timestamp("2015-08-01 00:00:00")


def load_model():
    model_path = MODEL_DIR / "gridpulse_xgb.json"
    if not model_path.exists():
        raise FileNotFoundError("Model file not found. Run src/train.py first.")
    model = XGBRegressor()
    model.load_model(str(model_path))
    return model


def plot_forecast(results, interval_width):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(results["time"], results["actual_load"], label="Actual", linewidth=1.4)
    ax.plot(results["time"], results["predicted_load"], label="Forecast", linewidth=1.2)
    ax.fill_between(
        results["time"],
        results["predicted_load"] - interval_width,
        results["predicted_load"] + interval_width,
        alpha=0.16,
        label="Empirical 90% error band",
    )
    ax.set_title("GridPulse next-hour load forecast")
    ax.set_xlabel("Time")
    ax.set_ylabel("Power load")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "future_forecast.png", dpi=170)
    plt.close(fig)


def run_forecast():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    logger = Logger(ROOT, "predict_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")).get_logger()

    train_data = load_load_data(DATA_DIR / "train.csv")
    test_data = load_load_data(DATA_DIR / "test.csv")
    metadata = load_json(MODEL_DIR / "metadata.json")
    model = load_model()

    # Test data includes a one-day warm-up window. We expose only observations
    # that are strictly earlier than the timestamp being forecast.
    combined = pd.concat([train_data, test_data], ignore_index=True)
    combined = combined.sort_values("time").drop_duplicates(subset="time", keep="last")
    actual_by_time = dict(zip(combined["time"], combined["power_load"]))

    pred_times = test_data.loc[test_data["time"] >= FORECAST_START, "time"].tolist()
    if not pred_times:
        raise ValueError("No forecast timestamps found after {}".format(FORECAST_START))

    first_time = pred_times[0]
    history = {
        timestamp: load
        for timestamp, load in actual_by_time.items()
        if timestamp < first_time
    }

    rows = []
    for index, pred_time in enumerate(pred_times, start=1):
        features = build_core_feature_row(pred_time, history)
        pred_value = float(model.predict(features)[0])
        true_value = float(actual_by_time[pred_time])
        seasonal_naive = float(history[pred_time - pd.Timedelta(hours=24)])

        rows.append([pred_time, true_value, pred_value, seasonal_naive])

        # The observed value becomes available only after the forecast is scored.
        # This keeps the loop faithful to a rolling one-hour-ahead deployment.
        history[pred_time] = true_value

        if index == 1 or index % 24 == 0 or index == len(pred_times):
            print("Forecasted {}/{} timestamps; latest: {}".format(index, len(pred_times), pred_time))

    results = pd.DataFrame(
        rows,
        columns=["time", "actual_load", "predicted_load", "seasonal_naive_24h"],
    )
    results["abs_error"] = (results["actual_load"] - results["predicted_load"]).abs()
    results.to_csv(REPORT_DIR / "future_predictions.csv", index=False)

    model_metrics = regression_metrics(results["actual_load"], results["predicted_load"])
    baseline_metrics = regression_metrics(results["actual_load"], results["seasonal_naive_24h"])
    improvement = 100.0 * (baseline_metrics["mae"] - model_metrics["mae"]) / baseline_metrics["mae"]

    payload = {
        "forecast_start": str(results["time"].min()),
        "forecast_end": str(results["time"].max()),
        "rows": int(len(results)),
        "model": model_metrics,
        "seasonal_naive_24h": baseline_metrics,
        "mae_improvement_vs_24h_naive_pct": float(improvement),
    }
    save_json(payload, REPORT_DIR / "future_metrics.json")

    interval_width = float(metadata.get("prediction_interval_abs_error_q90", 0.0))
    plot_forecast(results, interval_width)

    print("\nGridPulse out-of-time evaluation")
    print("MAE:  {:.3f}".format(model_metrics["mae"]))
    print("MSE:  {:.3f}".format(model_metrics["mse"]))
    print("RMSE: {:.3f}".format(model_metrics["rmse"]))
    print("MAPE: {:.2f}%".format(model_metrics["mape_pct"]))
    print("R2:   {:.4f}".format(model_metrics["r2"]))
    print("MAE improvement vs. 24h naive baseline: {:.2f}%".format(improvement))
    print("Saved predictions to reports/future_predictions.csv")
    print("Saved plot to data/fig/future_forecast.png")

    logger.info("Out-of-time metrics: %s", payload)
    return payload


if __name__ == "__main__":
    run_forecast()
