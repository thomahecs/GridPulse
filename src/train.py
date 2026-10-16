"""Fit GridPulse and evaluate six strictly forward-moving validation windows."""

import argparse
from datetime import datetime, timezone
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
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from utils.common import file_sha256, load_load_data, regression_metrics, save_json
from utils.experiments import DEFAULT_PARAMS, make_ridge, make_xgb, rolling_windows
from utils.features import build_features

DATA_DIR = ROOT / "data"
FIG_DIR = DATA_DIR / "fig"
MODEL_DIR = ROOT / "model"
REPORT_DIR = ROOT / "reports"


def tune_model(features, feature_names, first_eval, quick):
    """Select parameters only on data preceding all six backtest windows."""
    if quick:
        return dict(DEFAULT_PARAMS), None
    cv = TimeSeriesSplit(n_splits=3)
    search = GridSearchCV(
        make_xgb(),
        {"n_estimators": [100, 200, 300], "max_depth": [4, 6], "learning_rate": [0.05, 0.10]},
        cv=cv, scoring="neg_mean_absolute_error", n_jobs=1,
    )
    print("Tuning 12 XGBoost settings with 3 chronological CV folds", flush=True)
    search.fit(features.iloc[:first_eval][feature_names],
               features.iloc[:first_eval]["power_load"] - features.iloc[:first_eval]["lag_1h"])
    return search.best_params_, {"cv_mae": float(-search.best_score_), "cv_folds": 3, "candidates": 12}


def run_temporal_backtest(data, feature_names, params):
    rows = []
    residuals = []
    for fold, history, valid in rolling_windows(len(data)):
        train_frame, val_frame = data.iloc[history], data.iloc[valid]
        x_train, y_train = train_frame[feature_names], train_frame["power_load"]
        x_valid, y_valid = val_frame[feature_names], val_frame["power_load"]
        model = make_xgb(params).fit(x_train, y_train - train_frame["lag_1h"])
        ridge = make_ridge().fit(x_train, y_train)
        pred = model.predict(x_valid) + val_frame["lag_1h"].to_numpy()
        residuals.extend(np.abs(y_valid.to_numpy() - pred).tolist())
        model_metrics = regression_metrics(y_valid, pred)
        naive1 = regression_metrics(y_valid, val_frame["lag_1h"])
        naive24 = regression_metrics(y_valid, val_frame["lag_24h"])
        ridge_metrics = regression_metrics(y_valid, ridge.predict(x_valid))
        rows.append({
            "fold": fold,
            "train_start": str(train_frame["time"].iloc[0]),
            "train_end": str(train_frame["time"].iloc[-1]),
            "valid_start": str(val_frame["time"].iloc[0]),
            "valid_end": str(val_frame["time"].iloc[-1]),
            "rows": len(valid),
            "mae": model_metrics["mae"],
            "persistence_mae": naive1["mae"],
            "seasonal_naive_mae": naive24["mae"],
            "ridge_mae": ridge_metrics["mae"],
            "beats_persistence": model_metrics["mae"] < naive1["mae"],
            "beats_seasonal_naive": model_metrics["mae"] < naive24["mae"],
        })
        print(f"Fold {fold}/6: MAE {model_metrics['mae']:.2f}, persistence {naive1['mae']:.2f}", flush=True)
    backtest = pd.DataFrame(rows)
    backtest.to_csv(REPORT_DIR / "temporal_backtest.csv", index=False)
    return backtest, float(np.quantile(residuals, 0.9))


def plot_load_analysis(data):
    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    axs[0].plot(data["time"], data["power_load"], linewidth=0.35)
    axs[0].set(title="Historical hourly load", xlabel="Date", ylabel="Load (dataset units)")
    hours = data.groupby(data["time"].dt.hour)["power_load"].mean()
    axs[1].plot(hours.index, hours.values)
    axs[1].set(title="Average load by hour", xlabel="Hour", ylabel="Load (dataset units)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "load_analysis.png", dpi=150)
    plt.close(fig)


def plot_feature_importance(model, columns):
    top = pd.Series(model.feature_importances_, index=columns).sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(9, 6))
    top.plot.barh(ax=ax)
    ax.set(xlabel="XGBoost gain importance (normalized)", title="Model feature importance")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)


def train(quick=False):
    for directory in (FIG_DIR, MODEL_DIR, REPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    train_data = load_load_data(DATA_DIR / "train.csv")
    test_data = load_load_data(DATA_DIR / "test.csv")
    if not train_data["time"].max() < test_data["time"].max():
        raise ValueError("Test data must extend beyond the training period")
    plot_load_analysis(train_data)
    processed, names = build_features(train_data, profile="extended")
    first_eval = next(rolling_windows(len(processed)))[1].size
    params, search_summary = tune_model(processed, names, first_eval, quick)
    backtest, interval = run_temporal_backtest(processed, names, params)

    model = make_xgb(params).fit(processed[names], processed["power_load"] - processed["lag_1h"])
    model.save_model(str(MODEL_DIR / "gridpulse_xgb.json"))
    joblib.dump(model, MODEL_DIR / "gridpulse_xgb.joblib")
    ridge = make_ridge().fit(processed[names], processed["power_load"])
    joblib.dump(ridge, MODEL_DIR / "gridpulse_ridge.joblib")
    plot_feature_importance(model, names)
    summary = {
        "evaluation": "six expanding-window chronological backtests; 28-day nonoverlapping test windows",
        "folds": 6,
        "mae_min": float(backtest["mae"].min()),
        "mae_max": float(backtest["mae"].max()),
        "mae_mean": float(backtest["mae"].mean()),
        "beats_persistence_folds": int(backtest["beats_persistence"].sum()),
        "beats_seasonal_naive_folds": int(backtest["beats_seasonal_naive"].sum()),
    }
    metadata = {
        "project": "GridPulse",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_mode": "quick" if quick else "tuned",
        "forecast_horizon_hours": 1,
        "feature_profile": "extended",
        "prediction_target": "delta_from_previous_hour",
        "feature_names": names,
        "best_params": params,
        "tuning": search_summary,
        "backtest": summary,
        "prediction_interval_abs_error_q90": interval,
        "prediction_interval_note": "Descriptive quantile of backtest residuals; not a calibrated coverage guarantee",
        "training_rows": int(len(processed)),
        "raw_training_rows": int(len(train_data)),
        "training_start": str(train_data["time"].iloc[0]),
        "training_end": str(train_data["time"].iloc[-1]),
        "training_mean_load": float(train_data["power_load"].mean()),
        "training_data_sha256": file_sha256(DATA_DIR / "train.csv"),
        "test_data_sha256": file_sha256(DATA_DIR / "test.csv"),
        "temperature_available": False,
        "holiday_available": False,
        "note": "The supplied files contain load and timestamps only; no temperature/holiday covariates or additional training years.",
    }
    save_json(metadata, MODEL_DIR / "metadata.json")
    save_json(summary, REPORT_DIR / "backtest_summary.json")
    print(f"Final model trained on {len(processed):,} rows. Six-fold MAE range: {summary['mae_min']:.2f}–{summary['mae_max']:.2f}", flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Train GridPulse with honest time-series splits")
    parser.add_argument("--quick", action="store_true", help="Use documented parameters instead of chronological grid search")
    args = parser.parse_args()
    train(quick=args.quick)


if __name__ == "__main__":
    main()
