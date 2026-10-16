"""Out-of-time feature ablation on the same August 2015 evaluation period."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from utils.common import load_json, load_load_data, regression_metrics
from utils.experiments import make_xgb
from utils.features import build_features
from src.predict import prepare_evaluation


def run_ablation():
    train = load_load_data(ROOT / "data/train.csv")
    test = load_load_data(ROOT / "data/test.csv")
    params = load_json(ROOT / "model/metadata.json")["best_params"]
    rows = []
    for profile in ("calendar", "autoregressive", "core", "extended"):
        dev, names = build_features(train, profile)
        future, eval_names = prepare_evaluation(train, test, profile)
        assert names == eval_names
        if "lag_1h" in names:
            model = make_xgb(params).fit(dev[names], dev.power_load - dev["lag_1h"])
            pred = model.predict(future[names]) + future["lag_1h"].to_numpy()
            target = "delta_from_previous_hour"
        else:
            model = make_xgb(params).fit(dev[names], dev.power_load)
            pred = model.predict(future[names])
            target = "raw_load"
        metrics = regression_metrics(future.power_load, pred)
        rows.append({"feature_profile": profile, "features": len(names), "test_rows": len(future), "prediction_target": target, **metrics})
        print(f"{profile:17s} MAE {metrics['mae']:.3f}", flush=True)
    result = pd.DataFrame(rows)
    output = ROOT / "reports/ablation_results.csv"
    output.parent.mkdir(exist_ok=True)
    result.to_csv(output, index=False)
    return result


if __name__ == "__main__":
    run_ablation()
