#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Prepare GPSBench evaluation data by downloading the constructed 10% split.

Defaults:
  DATA_REPO=zhangdw/GPSBench-10pct
  DATA_INCLUDE=data/**
  LOCAL_DIR=.

Optional env:
  DATA_REPO             Hugging Face dataset repo id
  DATA_INCLUDE          Single include glob to download (default: data/**)
  LOCAL_DIR             Destination directory (default: project root/current dir)
  HF_MAX_WORKERS        Download workers for hf download (default: 8)
  FORCE_DOWNLOAD        Set to 1 to force redownload
  DRY_RUN               Set to 1 to print the download plan without writing files
  HF_CLI                Command used to run hf. Default: "uvx hf"

Examples:
  bash scripts/prepare_data.sh
  DRY_RUN=1 bash scripts/prepare_data.sh
  LOCAL_DIR=/tmp/gpsbench-data bash scripts/prepare_data.sh
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

DATA_REPO=${DATA_REPO:-zhangdw/GPSBench-10pct}
DATA_INCLUDE=${DATA_INCLUDE:-data/**}
LOCAL_DIR=${LOCAL_DIR:-.}
HF_MAX_WORKERS=${HF_MAX_WORKERS:-8}

HF_CLI_STRING="${HF_CLI:-uvx hf}"
read -r -a HF_CMD <<< "$HF_CLI_STRING"

extra_args=()
if [[ "${FORCE_DOWNLOAD:-0}" == "1" || "${FORCE_DOWNLOAD:-}" == "true" ]]; then
  extra_args+=(--force-download)
fi
if [[ "${DRY_RUN:-0}" == "1" || "${DRY_RUN:-}" == "true" ]]; then
  extra_args+=(--dry-run)
fi

echo "download data: hf://datasets/$DATA_REPO ($DATA_INCLUDE) -> $LOCAL_DIR"
"${HF_CMD[@]}" download "$DATA_REPO" \
  --repo-type dataset \
  --include "$DATA_INCLUDE" \
  --local-dir "$LOCAL_DIR" \
  --max-workers "$HF_MAX_WORKERS" \
  "${extra_args[@]}"

if [[ "${DRY_RUN:-0}" == "1" || "${DRY_RUN:-}" == "true" ]]; then
  exit 0
fi

pure_dir="${LOCAL_DIR%/}/data/track_pure_gps/splits"
applied_dir="${LOCAL_DIR%/}/data/track_applied/splits"

if [[ ! -d "$pure_dir" || ! -d "$applied_dir" ]]; then
  echo "ERROR: expected split directories were not found under ${LOCAL_DIR%/}/data/." >&2
  exit 1
fi

shopt -s nullglob
pure_tests=("$pure_dir"/*_test.json)
applied_tests=("$applied_dir"/*_test.json)
shopt -u nullglob

if (( ${#pure_tests[@]} == 0 || ${#applied_tests[@]} == 0 )); then
  echo "ERROR: expected *_test.json files were not found in both tracks." >&2
  exit 1
fi

echo "ready: ${#pure_tests[@]} pure_gps test files and ${#applied_tests[@]} applied test files found."
