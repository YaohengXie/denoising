#!/usr/bin/env bash
set -euo pipefail

scope="adapter"
data_root="data"
run_root=""
python_cmd="python"
device=""
install=0
dry_run=0

usage() {
  cat <<'EOF'
Usage: bash scripts/retrain_m143_v2.sh [options]
  --scope adapter|full
  --data-root PATH
  --run-root PATH
  --python COMMAND
  --device DEVICE
  --install
  --dry-run
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scope) scope="$2"; shift 2 ;;
    --data-root) data_root="$2"; shift 2 ;;
    --run-root) run_root="$2"; shift 2 ;;
    --python) python_cmd="$2"; shift 2 ;;
    --device) device="$2"; shift 2 ;;
    --install) install=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$scope" != "adapter" && "$scope" != "full" ]]; then
  echo "--scope must be adapter or full" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export PYTHONUTF8=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ $install -eq 1 ]]; then
  "$python_cmd" -m pip install --upgrade pip
  "$python_cmd" -m pip install -e '.[dev]'
fi

if [[ "$data_root" != /* ]]; then
  data_root="$repo_root/$data_root"
fi
data_root="$("$python_cmd" -c \
  'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' \
  "$data_root")"
if [[ "$(basename "$data_root")" != "data" ]]; then
  echo "--data-root must name the supplied data directory" >&2
  exit 2
fi
if [[ $dry_run -eq 0 && ! -d "$data_root" ]]; then
  echo "Controlled processed-data directory is missing: $data_root" >&2
  exit 3
fi
if [[ -z "$run_root" ]]; then
  run_root="$repo_root/retraining_runs/${scope}_$(date -u +%Y%m%d_%H%M%S)"
elif [[ "$run_root" != /* ]]; then
  run_root="$repo_root/$run_root"
fi
run_root="$("$python_cmd" -c \
  'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' \
  "$run_root")"
case "$run_root" in
  "$data_root"|"$data_root"/*)
    echo "Run root must be outside the read-only data directory." >&2
    exit 2
    ;;
esac
mkdir -p "$run_root"

"$python_cmd" scripts/capture_environment.py --output "$run_root/environment.json"
"$python_cmd" -m pytest -q -p no:cacheprovider

args=(
  -m ecg_pcg_denoise.retrain
  --scope "$scope"
  --data-root "$data_root"
  --run-root "$run_root"
  --python "$python_cmd"
)
if [[ -n "$device" ]]; then
  args+=(--device "$device")
fi
if [[ $dry_run -eq 1 ]]; then
  args+=(--dry-run)
fi
"$python_cmd" "${args[@]}"
