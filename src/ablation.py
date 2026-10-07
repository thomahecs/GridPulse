from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.common import load_json, load_load_data, regression_metrics
from utils.features import build_features

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"
REPORT_DIR = ROOT / "reports"


def run_ablation():
    """Compare feature groups while keeping the model family fixed."""
    data = load_load_data(DATA_DIR / "train.csv")
    metadata = load_json(MODEL_DIR / "metadata.json")
    params = metadata["best_params"]
    rows = []

    for profile in ("calendar", "autoregressive", "core", "extended"):
        processed, features = build_features(data, profile=profile)
        x_train, x_test, y_train, y_test = train_test_split(
            processed[features], processed["power_load"], test_size=0.30, random_state=22
        )
        model = XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            **params
        )
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        metrics = regression_metrics(y_test, pred)
        rows.append({"feature_profile": profile, "feature_count": len(features), **metrics})

    result = pd.DataFrame(rows).sort_values("mae")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(REPORT_DIR / "ablation_results.csv", index=False)
    print(result.to_string(index=False))
    print("\nSaved to reports/ablation_results.csv")
    return result


if __name__ == "__main__":
    run_ablation()
