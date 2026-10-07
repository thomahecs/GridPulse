import argparse
import datetime
import json
from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, TimeSeriesSplit, train_test_split
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.common import load_load_data, regression_metrics, save_json
from utils.features import build_features
from utils.log import Logger

DATA_DIR = ROOT / "data"
FIG_DIR = DATA_DIR / "fig"
MODEL_DIR = ROOT / "model"
REPORT_DIR = ROOT / "reports"


def plot_load_analysis(data):
    """Save a compact EDA figure for the historical load series."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    frame = data.copy()
    frame["hour"] = frame["time"].dt.hour
    frame["month"] = frame["time"].dt.month
    frame["is_weekend"] = frame["time"].dt.dayofweek >= 5

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes[0, 0].hist(frame["power_load"], bins=60)
    axes[0, 0].set_title("Load distribution")
    axes[0, 0].set_xlabel("Power load")

    hourly = frame.groupby("hour")["power_load"].mean()
    axes[0, 1].plot(hourly.index, hourly.values)
    axes[0, 1].set_title("Average load by hour")
    axes[0, 1].set_xlabel("Hour")

    monthly = frame.groupby("month")["power_load"].mean()
    axes[1, 0].plot(monthly.index, monthly.values, marker="o")
    axes[1, 0].set_title("Average load by month")
    axes[1, 0].set_xlabel("Month")

    weekday_mean = frame.loc[~frame["is_weekend"], "power_load"].mean()
    weekend_mean = frame.loc[frame["is_weekend"], "power_load"].mean()
    axes[1, 1].bar(["Weekday", "Weekend"], [weekday_mean, weekend_mean])
    axes[1, 1].set_title("Weekday vs. weekend")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "load_analysis.png", dpi=160)
    plt.close(fig)


def make_estimator(**params):
    defaults = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": 1,
        "verbosity": 0,
    }
    defaults.update(params)
    return XGBRegressor(**defaults)


def tune_model(x_train, y_train, logger, quick=False):
    if quick:
        params = {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.05}
        logger.info("Quick mode: using the stored reproducible parameter set %s", params)
        model = make_estimator(**params)
        model.fit(x_train, y_train)
        return model, params, None

    param_grid = {
        "n_estimators": [100, 150, 200],
        "max_depth": [4, 6],
        "learning_rate": [0.05, 0.10],
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(
        estimator=make_estimator(),
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        return_train_score=False,
    )
    search.fit(x_train, y_train)
    logger.info("Grid search best parameters: %s", search.best_params_)
    logger.info("Grid search best CV MSE: %.4f", -search.best_score_)
    return search.best_estimator_, search.best_params_, float(-search.best_score_)


def run_temporal_backtest(data, features, best_params, logger):
    """Run a small rolling-origin diagnostic with the selected hyperparameters."""
    x_data = data[features]
    y_data = data["power_load"]
    splitter = TimeSeriesSplit(n_splits=4)
    rows = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(x_data), start=1):
        model = make_estimator(**best_params)
        model.fit(x_data.iloc[train_idx], y_data.iloc[train_idx])
        pred = model.predict(x_data.iloc[valid_idx])
        metrics = regression_metrics(y_data.iloc[valid_idx], pred)
        metrics["fold"] = fold
        metrics["train_end"] = str(data.iloc[train_idx[-1]]["time"])
        metrics["valid_start"] = str(data.iloc[valid_idx[0]]["time"])
        metrics["valid_end"] = str(data.iloc[valid_idx[-1]]["time"])
        rows.append(metrics)

    backtest = pd.DataFrame(rows)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    backtest.to_csv(REPORT_DIR / "temporal_backtest.csv", index=False)
    logger.info("Temporal backtest MAE mean/std: %.4f / %.4f", backtest["mae"].mean(), backtest["mae"].std())
    return backtest


def plot_feature_importance(model, feature_names):
    importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(10, 7))
    importance.sort_values().plot(kind="barh", ax=ax)
    ax.set_title("Top feature importances")
    ax.set_xlabel("XGBoost importance")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_importance.png", dpi=160)
    plt.close(fig)


def train(quick=False):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    logger = Logger(ROOT, "train_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")).get_logger()
    train_data = load_load_data(DATA_DIR / "train.csv")
    plot_load_analysis(train_data)

    processed, features = build_features(train_data, profile="core")
    x_data = processed[features]
    y_data = processed["power_load"]

    # This holdout reproduces the original project benchmark. The separate
    # August forecast in predict.py is the stricter out-of-time evaluation.
    x_train, x_holdout, y_train, y_holdout = train_test_split(
        x_data, y_data, test_size=0.30, random_state=22
    )

    model, best_params, cv_mse = tune_model(x_train, y_train, logger, quick=quick)
    holdout_pred = model.predict(x_holdout)
    holdout_metrics = regression_metrics(y_holdout, holdout_pred)

    naive_1h = regression_metrics(y_holdout, x_holdout["lag_1h"].values)
    naive_24h = regression_metrics(y_holdout, x_holdout["lag_24h"].values)
    residual_q90 = float(np.quantile(np.abs(y_holdout.values - holdout_pred), 0.90))

    print("\nGridPulse benchmark holdout")
    print("Best parameters: {}".format(best_params))
    print("MAE:  {:.3f}".format(holdout_metrics["mae"]))
    print("MSE:  {:.3f}".format(holdout_metrics["mse"]))
    print("RMSE: {:.3f}".format(holdout_metrics["rmse"]))
    print("R2:   {:.4f}".format(holdout_metrics["r2"]))

    backtest = run_temporal_backtest(processed, features, best_params, logger)

    # Refit on every training row after evaluation. This is the model used by
    # the forward forecasting script.
    final_model = make_estimator(**best_params)
    final_model.fit(x_data, y_data)
    final_model.save_model(str(MODEL_DIR / "gridpulse_xgb.json"))
    joblib.dump(final_model, MODEL_DIR / "gridpulse_xgb.joblib")
    plot_feature_importance(final_model, features)

    metadata = {
        "project": "GridPulse - Short-Horizon Energy Forecasting ML Engine",
        "trained_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "feature_profile": "core",
        "feature_names": features,
        "best_params": best_params,
        "grid_search_cv_mse": cv_mse,
        "benchmark_holdout": holdout_metrics,
        "baseline_lag_1h": naive_1h,
        "baseline_lag_24h": naive_24h,
        "temporal_backtest_mae_mean": float(backtest["mae"].mean()),
        "temporal_backtest_mae_std": float(backtest["mae"].std()),
        "prediction_interval_abs_error_q90": residual_q90,
        "training_rows": int(len(processed)),
        "training_start": str(processed["time"].min()),
        "training_end": str(processed["time"].max()),
    }
    save_json(metadata, MODEL_DIR / "metadata.json")
    save_json({"holdout": holdout_metrics, "lag_1h": naive_1h, "lag_24h": naive_24h}, REPORT_DIR / "benchmark_metrics.json")

    logger.info("Training complete. Holdout metrics: %s", json.dumps(holdout_metrics))
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Train the GridPulse XGBoost forecaster.")
    parser.add_argument("--quick", action="store_true", help="Skip grid search and use the stored parameter set.")
    args = parser.parse_args()
    train(quick=args.quick)


if __name__ == "__main__":
    main()
