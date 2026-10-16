# GridPulse — Next-Hour Electricity Load Forecasting

GridPulse predicts electricity load one hour ahead using recent observations and calendar patterns. The model is built with XGBoost and evaluated on a separate later month of hourly readings. A small set of baselines helps answer the question that matters most here: does the model improve on simply using the last observed load?

This repository is a revised, runnable version of my original forecasting project. I kept the original CSV data and added a more careful time-series evaluation, an autoregressive residual target, reproducible reports, and tests for common sources of leakage.

## Data and prediction setup

The supplied dataset has two files:

| File | Period | Hourly records |
| --- | --- | ---: |
| `data/train.csv` | September 2, 2013 – July 31, 2015 | 16,752 |
| `data/test.csv` | July 31 – August 31, 2015 | 768, including 24 hours of overlap |

There are **744 scored hours** in August 2015. The overlapping July 31 readings supply the initial lag history; they are not scored twice. The project originally described the source as China Southern Power Grid, but the archive does not include a source citation or a data dictionary confirming the physical units. Those details should be checked before using the dataset in a publication.

Each forecast is made with information available before the hour being predicted. Once the actual load for that hour is observed, it can be used to prepare the next prediction. This is a **rolling, one-hour-ahead forecast**, not a forecast of the entire month made on July 31.

The raw CSVs have timestamps and load values only. There are no temperature readings, weather forecasts, or verified holiday labels in the archive, and I have not filled them in with estimates. The data also covers about two years, not 36 months. The resume comparison in `reports/resume_audit.md` records these limitations explicitly.

## Model

The production model uses XGBoost to predict the **change from the previous hour's load**, rather than the absolute load directly:

$$
\widehat{L}_t = L_{t-1} + f(X_t)
$$

The features include load lags at 1, 2, 3, 6, 12, 24, 48, and 168 hours; hour and month indicators; day-of-week and weekend indicators; cyclical time encodings; and lagged rolling means, standard deviation, and load differences. All rolling statistics are shifted so the current target cannot enter its own features.

For comparison, the pipeline also trains Ridge regression and evaluates two naive forecasts: persistence (the previous hour) and seasonal naive (the same hour one day earlier).

## Updating an existing GitHub project

To retain your existing Git history, **do not replace or delete the old `.git` folder**. Extract the update ZIP somewhere else (for example, Downloads), then run the update script from the extracted `GridPulse` folder:

```bash
bash scripts/apply_update.sh /absolute/path/to/your/existing/GridPulse
```

This copies the new files into the original repository without touching `.git`, `.venv`, or PyCharm settings. It overwrites project files with the revised versions, so commit or back up any local edits you want to keep first. Then `cd` to the original project directory and follow the run command below. You can review the changed files with `git status` before committing.

## Run it on a Mac

Use **Python 3.10–3.13**. Python 3.12 is a good choice for Apple Silicon. From the project directory, run:

```bash
bash scripts/run_mac.sh
```

The script creates `.venv`, installs dependencies, runs the tests, then performs chronological hyperparameter search, six backtests, the August evaluation, a feature ablation, the resume claim audit, and one forward forecast past the last known timestamp. It does not use AWS or any paid API.

For a quicker functional check with a fixed parameter set:

```bash
bash scripts/run_mac.sh --quick
```

If Python is missing or too old, install a recent release (for example, `brew install python@3.12`). If XGBoost reports that `libomp.dylib` cannot be found on macOS, install OpenMP with `brew install libomp`, then rerun the script.

For PyCharm, open the `GridPulse` folder, select `.venv/bin/python` as the project interpreter, and run `src.run_all` as a Python **module**. You can also run each stage from a terminal:

```bash
python -m src.train             # chronological grid search, six backtests, save models
python -m src.predict           # August 2015 forecast + baselines
python -m src.ablation          # feature-group comparison on the same future month
python -m src.audit             # check each relevant resume claim against actual results
python -m src.forecast_next     # predict September 1, 2015 00:00 from the latest known readings
python -m pytest -q            # automated tests
```

Use `.venv/bin/python` in place of `python` if the virtual environment is not activated. Training must run before prediction if you change the data or feature code.

## Evaluation

The workflow deliberately separates model selection from the final test period.

1. Hyperparameter search uses three chronological folds drawn **only from the early training history**. The search tests 12 settings for tree depth, learning rate, and number of estimators. No shuffled K-fold split is used.
2. Six expanding-window backtests each predict the next 28 days. Every validation window comes after its training window. These windows are useful for examining changes in error across seasons and shifts in demand.
3. After that, the selected model is fit on all development data and evaluated against the separate August 2015 observations.
4. Feature ablation trains calendar-only, autoregressive-only, core, and extended models. It uses the same August period for descriptive comparison. **The ablation results are not a second independent holdout**, so selecting a model from these results would require a fresh final test period.

The archived full run gives the following results on August 2015. These values come from `reports/future_metrics.json`, not from a hard-coded scoring function.

| Forecast | MAE (dataset units) |
| --- | ---: |
| GridPulse XGBoost | **47.572** |
| Previous-hour persistence | 58.95 |
| Previous-day seasonal naive | 76.48 |
| Ridge regression | 51.73 |

The XGBoost MAE was about **4.86% of the observed mean load**. It beat persistence by **19.30%** on this period. Across the six rolling backtests, the MAE varied from **17.32 to 116.19** and XGBoost beat persistence in all six windows. That spread is a reminder that one favorable month does not tell the whole story.

These numbers describe the included data and the recorded full run. A quick run uses different parameters and will overwrite the report files with its own measurements. Library versions can also affect results slightly.

## Project layout

```text
GridPulse/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── fig/                    # plots from the latest run
├── model/
│   ├── gridpulse_xgb.json      # portable XGBoost model
│   ├── gridpulse_xgb.joblib
│   ├── gridpulse_ridge.joblib
│   └── metadata.json          # parameters, feature order, data hashes
├── reports/
│   ├── future_metrics.json
│   ├── future_predictions.csv
│   ├── temporal_backtest.csv
│   ├── hourly_error_analysis.csv
│   ├── next_hour_forecast.json
│   ├── ablation_results.csv
│   ├── backtest_summary.json
│   ├── resume_audit.md
│   └── resume_audit.json
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── ablation.py
│   ├── audit.py
│   ├── forecast_next.py
│   └── run_all.py
├── utils/
├── tests/
├── scripts/run_mac.sh
└── requirements.txt
```

The saved predictions let you recompute each reported error. Model metadata includes hashes of both input CSV files; forecasting fails rather than silently scoring a saved model against changed data. The CI workflow also runs the unit tests and a quick end-to-end pipeline on GitHub Actions.

## What is not established by this dataset

The experiment does **not** verify claims about three years of training data, a three-month unseen holdout, temperature or holiday gains, or the exact metrics from a different run or dataset. Those would require the underlying records and another properly isolated evaluation. I have left the missing data and unsupported results out rather than constructing values to match a target.

For the current, item-by-item evidence, start with **[`reports/resume_audit.md`](reports/resume_audit.md)**.
