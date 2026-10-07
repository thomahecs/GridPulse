# GridPulse — Short-Horizon Energy Forecasting ML Engine

GridPulse is a short-horizon electricity load forecasting project built with historical China Southern Power Grid data. The main goal is to predict the next hour of electricity demand from recent load history and time-based patterns.

The project started as a basic XGBoost forecasting pipeline and was later extended with model tuning, time-series evaluation, leakage checks, a seasonal baseline, feature ablation, and saved experiment results.

## Features

The main model uses:

- hour of day
- month of year
- load from 1, 2, and 3 hours earlier
- load from the same hour on the previous day

For each new timestamp, the model only uses information that would already be available at prediction time. Future load values are not used when building lag features.

There is also an extended feature set used in the ablation experiment. It adds features such as day of week, weekend indicators, cyclical time encoding, longer lag windows, and rolling statistics.

## Project structure

```text
GridPulse/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── fig/
├── log/
├── model/
│   ├── gridpulse_xgb.json
│   ├── gridpulse_xgb.joblib
│   └── metadata.json
├── reports/
├── src/
│   ├── run_all.py
│   ├── train.py
│   ├── predict.py
│   └── ablation.py
├── utils/
│   ├── common.py
│   ├── features.py
│   └── log.py
├── requirements.txt
└── README.md
```

## Running the project

### Full pipeline

Run:

```text
src/run_all.py
```

This trains the model first and then runs the future-window evaluation.

For a faster test run, use:

```text
--quick
```

The quick option skips the full grid search and uses the saved default hyperparameters.

### Train the model

Run:

```text
src/train.py
```

The regular training run performs hyperparameter search with cross-validation and then fits the final XGBoost model.

Main outputs:

- `model/gridpulse_xgb.json`
- `model/gridpulse_xgb.joblib`
- `model/metadata.json`
- `reports/benchmark_metrics.json`
- `reports/temporal_backtest.csv`
- `data/fig/load_analysis.png`
- `data/fig/feature_importance.png`

### Run the future forecast

Run:

```text
src/predict.py
```

This evaluates the model on the August 2015 future window using rolling next-hour predictions.

Outputs:

- `reports/future_predictions.csv`
- `reports/future_metrics.json`
- `data/fig/future_forecast.png`

The forecast plot includes the real load, predicted load, and an empirical 90% error band estimated from validation residuals.

### Feature ablation

Run:

```text
src/ablation.py
```

This compares four feature groups:

- calendar features only
- autoregressive features only
- the main GridPulse feature set
- the extended feature set

Results are saved to:

```text
reports/ablation_results.csv
```

I added this experiment to check whether the extra time-series features actually help instead of assuming that a larger feature set is always better.

## Evaluation

The project uses two main evaluation settings.

**Benchmark holdout:** a fixed 70/30 split used to compare model settings during development.

**Out-of-time evaluation:** August 2015 is kept as a later time window and evaluated with rolling next-hour forecasts. At each step, the model only has access to load values that occurred earlier in time.

A 24-hour seasonal naive baseline is also included. It predicts the current hour using the load from the same hour one day earlier, which gives the XGBoost model a simple time-series baseline to beat.

The project reports MAE, MSE, RMSE, MAPE, and R². Actual run results are written to the files under `reports/` instead of being hard-coded in this README.

## Notes

Model results can vary slightly across XGBoost and scikit-learn versions. The model metadata and report files store the parameters, feature list, data ranges, and metrics from each run so the experiments are easier to reproduce and compare.
