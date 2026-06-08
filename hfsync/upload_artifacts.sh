#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Upload GPSBench experiment artifacts to the Leo benchmark Hugging Face bucket.

Default remote root:
  hf://buckets/zhangdw/leo-benchmark/GPSBench

Optional env:
  BUCKET_ROOT            Bucket prefix to sync into (default above)
  ARTIFACT_PATHS         Space-separated local artifact dirs (default: "results batch_files")
  REMOTE_PREFIX          Optional sub-prefix under BUCKET_ROOT, e.g. qwen3-4b-thinking-2507
  DRY_RUN                Set to 1 to print the sync plan without uploading
  DELETE                 Set to 1 to delete remote files missing locally
  VERBOSE                Set to 1 for verbose sync logging
  HF_USE_ENV_TOKEN       Set to 1 to force using HF_TOKEN from the environment

Examples:
  bash hfsync/upload_artifacts.sh
  REMOTE_PREFIX=qwen3-4b-thinking-2507 bash hfsync/upload_artifacts.sh
  ARTIFACT_PATHS="results" DRY_RUN=1 bash hfsync/upload_artifacts.sh
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

BUCKET_ROOT=${BUCKET_ROOT:-hf://buckets/zhangdw/leo-benchmark/GPSBench}
ARTIFACT_PATHS=${ARTIFACT_PATHS:-"results batch_files"}
REMOTE_PREFIX=${REMOTE_PREFIX:-}

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

uploaded_any=0
for local_path in $ARTIFACT_PATHS; do
  if [[ ! -e "$local_path" ]]; then
    echo "skip: $local_path does not exist" >&2
    continue
  fi

  remote_path="${BUCKET_ROOT%/}"
  if [[ -n "$REMOTE_PREFIX" ]]; then
    remote_path="$remote_path/${REMOTE_PREFIX%/}"
  fi
  remote_path="$remote_path/$local_path"

  echo "sync upload: $local_path -> $remote_path"
  "${hf_cmd[@]}" buckets sync "$local_path" "$remote_path" "${sync_args[@]}"
  uploaded_any=1
done

if [[ "$uploaded_any" == "0" ]]; then
  echo "ERROR: no artifact paths existed. Nothing uploaded." >&2
  exit 1
fi
