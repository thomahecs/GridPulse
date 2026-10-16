#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=""
for name in python3.12 python3.13 python3.11 python3.10 python3; do
  if command -v "$name" >/dev/null 2>&1 && "$name" -c 'import sys; assert (3, 10) <= sys.version_info[:2] < (3, 14)' >/dev/null 2>&1; then
    PYTHON="$name"
    break
  fi
done
if [[ -z "$PYTHON" ]]; then
  echo 'Python 3.10–3.13 is required. On a Mac: brew install python@3.12' >&2
  exit 1
fi
if [[ -d .venv ]] && ! .venv/bin/python -c 'import sys; assert (3, 10) <= sys.version_info[:2] < (3, 14)' >/dev/null 2>&1; then
  if [[ -e .venv.gridpulse-backup ]]; then
    echo 'Existing virtual environment is incompatible, and .venv.gridpulse-backup already exists.' >&2
    exit 1
  fi
  mv .venv .venv.gridpulse-backup
  echo 'Saved incompatible .venv as .venv.gridpulse-backup'
fi
if [[ ! -f .venv/bin/python ]]; then
  "$PYTHON" -m venv .venv
fi
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q
if [[ "${1:-}" == "--quick" ]]; then
  .venv/bin/python -m src.run_all --quick
else
  .venv/bin/python -m src.run_all
fi
printf '\nDONE. Open reports/resume_audit.md and reports/future_metrics.json.\n'
