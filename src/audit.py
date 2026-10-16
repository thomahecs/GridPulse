"""Compare recorded experiments with each assertion in the current resume.

This is intentionally an audit, not an attempt to force metrics to match a target.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from utils.common import load_json, save_json


def run_audit():
    metadata = load_json(ROOT / "model/metadata.json")
    future = load_json(ROOT / "reports/future_metrics.json")
    folds = pd.read_csv(ROOT / "reports/temporal_backtest.csv")
    ablation = pd.read_csv(ROOT / "reports/ablation_results.csv")
    hourly = pd.read_csv(ROOT / "reports/hourly_error_analysis.csv")
    highest_error_hour = int(hourly.loc[hourly.mae.idxmax(), "hour"])
    evening = hourly.loc[hourly.hour.between(18, 22), "mae"].mean()
    other = hourly.loc[~hourly.hour.between(18, 22), "mae"].mean()
    evening_highest = bool(18 <= highest_error_hour <= 22 and evening > other)
    top_importance = None
    try:
        import joblib
        model = joblib.load(ROOT / "model/gridpulse_xgb.joblib")  # Trusted generated model only.
        idx = int(model.feature_importances_.argmax())
        top_importance = metadata["feature_names"][idx]
    except (FileNotFoundError, ValueError, IndexError):
        pass
    tolerance = 0.051  # One decimal place in the printed resume numbers.
    check = lambda actual, target: abs(actual - target) < tolerance
    model = future["model"]["mae"]
    checks = [
        ("36 months of hourly training", False, f"Only {metadata['raw_training_rows']} hourly training rows are supplied, about 23 months"),
        ("1–3 hour lags and previous-day load", True, "lag_1h, lag_2h, lag_3h, lag_24h"),
        ("Calendar features", True, "Hour/month encoded from timestamps"),
        ("Temperature features", False, "No measured or forecast temperature input in the archive"),
        ("Holiday features", False, "No documented holiday features or holiday file in original training data"),
        ("Grid search with chronological CV", metadata["run_mode"] == "tuned", "Full run uses 12 settings / 3 ordered CV folds; quick mode does not tune"),
        ("Unseen 3-month holdout", False, f"Supplied test is {future['rows']} hours ({future['forecast_start']} to {future['forecast_end']})"),
        ("54.9 MW MAE", check(model, 54.9), f"Measured {model:.3f} in the dataset's load units; MW unit is not verified"),
        ("MAE equals 0.9% of mean load", check(future["mae_pct_of_mean_load"], 0.9), f"Measured {future['mae_pct_of_mean_load']:.3f}%"),
        ("Persistence MAE 91.7", check(future["persistence_1h"]["mae"], 91.7), f"Measured {future['persistence_1h']['mae']:.3f}"),
        ("Seasonal naive MAE 109.2", check(future["seasonal_naive_24h"]["mae"], 109.2), f"Measured {future['seasonal_naive_24h']['mae']:.3f}"),
        ("Ridge MAE 78.9", check(future["ridge"]["mae"], 78.9), f"Measured {future['ridge']['mae']:.3f}"),
        ("40.1% below persistence", check(future["mae_improvement_vs_persistence_pct"], 40.1), f"Measured {future['mae_improvement_vs_persistence_pct']:.2f}%"),
        ("49.7% below seasonal naive", check(future["mae_improvement_vs_seasonal_naive_pct"], 49.7), f"Measured {future['mae_improvement_vs_seasonal_naive_pct']:.2f}%"),
        ("30.4% below Ridge", check(future["mae_improvement_vs_ridge_pct"], 30.4), f"Measured {future['mae_improvement_vs_ridge_pct']:.2f}%"),
        ("Lag-1 top importance", top_importance == "lag_1h", f"Current top feature: {top_importance or 'unknown'}"),
        ("Evening peaks largest error source", evening_highest, f"Highest-error hour: {highest_error_hour}:00; evening (18–22) MAE {evening:.2f}, other hours {other:.2f}"),
        ("Temperature/holiday ablation 59.8 → 54.9 (8.2%)", False, "No temperature or holiday covariate data; current ablation tests calendar/lag profiles only"),
        ("Six chronological backtests", len(folds) == 6, f"Measured {len(folds)} windows"),
        ("Backtest MAE range 52.1–61.4", check(folds.mae.min(), 52.1) and check(folds.mae.max(), 61.4), f"Observed {folds.mae.min():.2f}–{folds.mae.max():.2f}"),
        ("Beat persistence in all six folds", len(folds) == 6 and bool(folds.beats_persistence.all()), f"Won {int(folds.beats_persistence.sum())} of {len(folds)} folds"),
    ]
    data = {"all_claims_verified": bool(all(x[1] for x in checks)),
            "verified_count": int(sum(bool(x[1]) for x in checks)), "total_count": len(checks),
            "checks": [{"resume_claim": name, "verified": bool(ok), "evidence": detail} for name, ok, detail in checks],
            "context": "Audit of the uploaded resume against the supplied archive and this run. Unverified does not mean invented; it means these files and tests cannot establish the claim."}
    save_json(data, ROOT / "reports/resume_audit.json")
    lines = ["# Resume claim audit", "", "Based on the supplied data and this experimental run, not on desired resume metrics.", "", "| Claim | Status | Evidence |", "| --- | --- | --- |"]
    for name, ok, evidence in checks:
        lines.append(f"| {name} | {'Verified' if ok else 'Not verified'} | {evidence} |")
    lines.extend(["", f"**Verified: {data['verified_count']} / {len(checks)}.**", "", "An unverified claim should not be presented as a reproduced result.", ""])
    (ROOT / "reports/resume_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Resume claims supported: {data['verified_count']}/{data['total_count']}. See reports/resume_audit.md")
    return data


if __name__ == "__main__":
    run_audit()
