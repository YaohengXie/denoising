#!/usr/bin/env bash
set -euo pipefail
export PYTHONUTF8=1
export PYTHONDONTWRITEBYTECODE=1

DATA_ROOT="data"
RUN_ROOT=""
PYTHON_BIN="python"
DEVICE=""
INSTALL=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: scripts/reproduce_m143_v2.sh [options]

Options:
  --data-root PATH   Processed authorised data directory (default: data)
  --run-root PATH    Output directory (default: timestamped reproduction_runs entry)
  --python PATH      Python interpreter (default: python)
  --device DEVICE    Optional PyTorch device override, e.g. cpu or cuda
  --install          Upgrade pip and install .[dev]
  --dry-run          Verify checkpoints and write the command plan only
  -h, --help         Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root) DATA_ROOT="${2:?missing value for --data-root}"; shift 2 ;;
    --run-root) RUN_ROOT="${2:?missing value for --run-root}"; shift 2 ;;
    --python) PYTHON_BIN="${2:?missing value for --python}"; shift 2 ;;
    --device) DEVICE="${2:?missing value for --device}"; shift 2 ;;
    --install) INSTALL=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$REPOSITORY_ROOT"
export PYTHONPATH="$REPOSITORY_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$DATA_ROOT" != /* ]]; then
  DATA_ROOT="$REPOSITORY_ROOT/$DATA_ROOT"
fi
if [[ "$(basename "$DATA_ROOT")" != "data" ]]; then
  echo "--data-root must name the supplied data directory: $DATA_ROOT" >&2
  exit 1
fi
if [[ "$DRY_RUN" -eq 0 && ! -d "$DATA_ROOT" ]]; then
  echo "Controlled processed-data directory is missing: $DATA_ROOT" >&2
  exit 1
fi
if [[ -z "$RUN_ROOT" ]]; then
  RUN_ROOT="$REPOSITORY_ROOT/reproduction_runs/$(date -u +%Y%m%d_%H%M%S)"
elif [[ "$RUN_ROOT" != /* ]]; then
  RUN_ROOT="$REPOSITORY_ROOT/$RUN_ROOT"
fi
case "$RUN_ROOT" in
  "$DATA_ROOT"|"$DATA_ROOT"/*)
    echo "Run root must be outside the read-only data directory." >&2
    exit 1
    ;;
esac
mkdir -p "$RUN_ROOT"

run_python() {
  printf '> %q' "$PYTHON_BIN"
  printf ' %q' "$@"
  printf '\n'
  "$PYTHON_BIN" "$@"
}

if [[ "$INSTALL" -eq 1 ]]; then
  run_python -m pip install --upgrade pip
  run_python -m pip install -e '.[dev]'
fi

echo "M14.3-v2 fixed-checkpoint reproduction"
echo "Repository: $REPOSITORY_ROOT"
echo "Data root: $DATA_ROOT"
echo "Run root: $RUN_ROOT"

run_python scripts/capture_environment.py --output "$RUN_ROOT/environment.json"
run_python -m pytest -q -p no:cacheprovider

ARGS=(-m ecg_pcg_denoise.repro run --data-root "$DATA_ROOT" --run-root "$RUN_ROOT")
if [[ -n "$DEVICE" ]]; then ARGS+=(--device "$DEVICE"); fi
if [[ "$DRY_RUN" -eq 1 ]]; then ARGS+=(--dry-run); fi
run_python "${ARGS[@]}"

echo "M14.3-v2 reproduction completed."
