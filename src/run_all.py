import argparse

from train import train
from predict import run_forecast


def main():
    parser = argparse.ArgumentParser(description="Run GridPulse training and future evaluation end to end.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip GridSearchCV and use the stored parameter set for a faster smoke test.",
    )
    args = parser.parse_args()

    print("=== GridPulse: training ===")
    train(quick=args.quick)
    print("\n=== GridPulse: out-of-time evaluation ===")
    run_forecast()


if __name__ == "__main__":
    main()
