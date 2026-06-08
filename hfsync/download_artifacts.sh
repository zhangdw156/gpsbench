#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Download GPSBench experiment artifacts from the Leo benchmark Hugging Face bucket.

Default remote root:
  hf://buckets/zhangdw/leo-benchmark/GPSBench

Optional env:
  BUCKET_ROOT            Bucket prefix to sync from (default above)
  ARTIFACT_PATHS         Space-separated artifact dirs to download (default: "results batch_files")
  REMOTE_PREFIX          Optional sub-prefix under BUCKET_ROOT, e.g. qwen3-4b-thinking-2507
  LOCAL_DIR              Destination project/local root (default: .)
  DRY_RUN                Set to 1 to print the sync plan without downloading
  DELETE                 Set to 1 to delete local files missing remotely
  VERBOSE                Set to 1 for verbose sync logging
  HF_USE_ENV_TOKEN       Set to 1 to force using HF_TOKEN from the environment

Examples:
  bash hfsync/download_artifacts.sh
  REMOTE_PREFIX=qwen3-4b-thinking-2507 bash hfsync/download_artifacts.sh
  ARTIFACT_PATHS="results" DRY_RUN=1 bash hfsync/download_artifacts.sh
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

BUCKET_ROOT=${BUCKET_ROOT:-hf://buckets/zhangdw/leo-benchmark/GPSBench}
ARTIFACT_PATHS=${ARTIFACT_PATHS:-"results batch_files"}
REMOTE_PREFIX=${REMOTE_PREFIX:-}
LOCAL_DIR=${LOCAL_DIR:-.}

hf_cmd=(uvx hf)
if [[ -n "${HF_TOKEN:-}" && "${HF_USE_ENV_TOKEN:-0}" != "1" ]]; then
  echo "note: HF_TOKEN is set; ignoring it so the stored 'hf auth login' token is used. Set HF_USE_ENV_TOKEN=1 to force HF_TOKEN." >&2
  hf_cmd=(env -u HF_TOKEN uvx hf)
fi

sync_args=()
if [[ "${DRY_RUN:-0}" == "1" || "${DRY_RUN:-}" == "true" ]]; then
  sync_args+=(--dry-run)
fi
if [[ "${DELETE:-0}" == "1" || "${DELETE:-}" == "true" ]]; then
  sync_args+=(--delete)
fi
if [[ "${VERBOSE:-0}" == "1" || "${VERBOSE:-}" == "true" ]]; then
  sync_args+=(--verbose)
fi

for artifact_path in $ARTIFACT_PATHS; do
  remote_path="${BUCKET_ROOT%/}"
  if [[ -n "$REMOTE_PREFIX" ]]; then
    remote_path="$remote_path/${REMOTE_PREFIX%/}"
  fi
  remote_path="$remote_path/$artifact_path"
  local_path="${LOCAL_DIR%/}/$artifact_path"

  mkdir -p "$local_path"
  echo "sync download: $remote_path -> $local_path"
  "${hf_cmd[@]}" buckets sync "$remote_path" "$local_path" "${sync_args[@]}"
done
