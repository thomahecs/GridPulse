#!/usr/bin/env bash
set -euo pipefail
if [[ "$#" != 1 ]]; then
  echo 'Usage: bash scripts/apply_update.sh /absolute/path/to/your/existing/GridPulse' >&2
  exit 2
fi
SRC="$(cd "$(dirname "$0")/.." && pwd)"
if [[ ! -d "$1" ]]; then
  echo "Project folder not found: $1" >&2
  exit 1
fi
DEST="$(cd "$1" && pwd)"
if [[ "$SRC" == "$DEST" ]]; then
  echo 'This is already the extracted update folder; no files need copying.'
  exit 0
fi
if [[ ! -d "$DEST/.git" || ! -f "$DEST/data/train.csv" ]]; then
  echo 'Target must be your existing GridPulse Git repository containing data/train.csv.' >&2
  exit 1
fi
if ! command -v rsync >/dev/null 2>&1; then
  echo 'rsync was not found. It is normally included with macOS.' >&2
  exit 1
fi
rsync -av --exclude '.git/' --exclude '.venv/' --exclude '.idea/' --exclude '__pycache__/' --exclude '.pytest_cache/' "$SRC/" "$DEST/"
echo "Updated: $DEST"
echo 'Your existing .git history and virtual environment were preserved.'
echo 'Next: cd to your existing GridPulse folder and run bash scripts/run_mac.sh'
