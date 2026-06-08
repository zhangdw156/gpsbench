#!/usr/bin/env bash
set -euo pipefail

# Full evaluation on the constructed 10% GPSBench test split.
# Expected data layout: data/track_*/splits/*_test.json
# Override defaults with env vars, e.g. MODEL=qwen3-4b-thinking-2507 MAX_WORKERS=8 bash scripts/run_full_10pct_evaluation.sh

MODEL=${MODEL:-qwen3-4b-thinking-2507}
PROVIDER=${PROVIDER:-openai}
OPENAI_BASE_URL=${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}
OPENAI_API_KEY=${OPENAI_API_KEY:-EMPTY}
TRACK=${TRACK:-both}
MAX_WORKERS=${MAX_WORKERS:-16}
DELAY=${DELAY:-0}
DATA_DIR=${DATA_DIR:-data}
RESULTS_DIR=${RESULTS_DIR:-results}

model_safe=$(printf '%s' "$MODEL" | tr '/:' '__')
OUTPUT=${OUTPUT:-${model_safe}_10pct_test}

sample_args=()
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  sample_args+=(--max-samples "$MAX_SAMPLES")
fi

task_args=()
if [[ -n "${TASKS:-}" ]]; then
  # TASKS is intentionally word-split to match run_benchmark.py's nargs=+ interface.
  task_args+=(--tasks $TASKS)
fi

export OPENAI_BASE_URL OPENAI_API_KEY

uv run python run_benchmark.py \
  --provider "$PROVIDER" \
  --model "$MODEL" \
  --track "$TRACK" \
  "${task_args[@]}" \
  "${sample_args[@]}" \
  --concurrent \
  --max-workers "$MAX_WORKERS" \
  --delay "$DELAY" \
  --data-dir "$DATA_DIR" \
  --results-dir "$RESULTS_DIR" \
  --output "$OUTPUT" \
  "$@"
