"""Single entry point for training, forecasting, ablation and claim audit."""

import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ablation import run_ablation
from src.audit import run_audit
from src.forecast_next import forecast_next
from src.predict import run_forecast
from src.train import train


def main():
    parser = argparse.ArgumentParser(description="Run GridPulse experiments end to end")
    parser.add_argument("--quick", action="store_true", help="Skip hyperparameter search, not the six backtests")
    parser.add_argument("--strict-resume", action="store_true", help="Exit nonzero if any resume claim is unverified")
    args = parser.parse_args()
    print("\n1/5 Training and six backtests", flush=True)
    train(quick=args.quick)
    print("\n2/5 August 2015 unseen evaluation", flush=True)
    run_forecast()
    print("\n3/5 Feature ablation", flush=True)
    run_ablation()
    print("\n4/5 Resume claim audit", flush=True)
    audit = run_audit()
    print("\n5/5 Next-hour forecast beyond the sample", flush=True)
    forecast_next()
    if args.strict_resume and not audit["all_claims_verified"]:
        raise SystemExit("Some resume claims are not supported by the supplied data and experimental results")
    print("\nPipeline completed. Please read reports/resume_audit.json.")


if __name__ == "__main__":
    main()
