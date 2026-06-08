# GPSBench Evaluation Harness

This fork is trimmed to the code needed to evaluate GPSBench-style test splits with an OpenAI-compatible model endpoint, including local vLLM servers.

## What is kept

- `run_benchmark.py` — main evaluation runner
- `evaluation/llm_client.py` — OpenAI/OpenRouter/Gemini-compatible client wrapper
- `prompts/` — task prompt templates
- `.env.example` / `pyproject.toml` / `uv.lock` — reproducible uv-managed runtime setup

Generated data and results are intentionally not versioned.

## Setup

Install the uv-managed environment after cloning:

```bash
uv sync
```

This creates `.venv/` from `pyproject.toml` and the committed `uv.lock`. Run project commands through `uv run ...` or activate the environment with `source .venv/bin/activate`.

## Prepare evaluation data

Download the constructed 10% GPSBench evaluation split from Hugging Face into the project root:

```bash
bash scripts/prepare_data.sh
```

The script downloads only the benchmark `data/` tree from `zhangdw/GPSBench-10pct` using `uvx hf`, then checks that both track test splits exist. For a no-write preview, run `DRY_RUN=1 bash scripts/prepare_data.sh`.

After download, the runner uses the normal default paths:

```text
data/track_pure_gps/splits/*_test.json
data/track_applied/splits/*_test.json
```

## Evaluate a vLLM-served model

Point the OpenAI client at the vLLM server:

```bash
export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
export OPENAI_API_KEY="EMPTY"
```

Use the model id exposed by:

```bash
curl -s "$OPENAI_BASE_URL/models"
```

By default, the runner also reads this `/v1/models` metadata to set the output cap to half of the served model's context length (for example, `max_model_len: 131072` -> `--max-tokens 65536`). If the endpoint does not expose a recognizable context field, the runner falls back to `8192`. Override explicitly with `--max-tokens N` or `MAX_TOKENS=N bash scripts/run_full_10pct_evaluation.sh`.

Run a smoke test:

```bash
uv run python run_benchmark.py \
  --provider openai \
  --model my-vllm-model \
  --track pure_gps \
  --tasks distance_calculation \
  --max-samples 5 \
  --concurrent \
  --max-workers 4 \
  --delay 0
```

Run the full 10% test set:

```bash
uv run python run_benchmark.py \
  --provider openai \
  --model my-vllm-model \
  --track both \
  --concurrent \
  --max-workers 16 \
  --delay 0
```

Results are written under `results/`, which is ignored by git.

## Resume interrupted evaluations

The evaluation runner checkpoints each sample into the current output folder as soon as that sample finishes. Re-running with the same `--output` value resumes from `task_results/`: samples with valid responses are reused, while missing samples and samples with an `error` field are evaluated again. Incorrect but valid model answers are not retried, so benchmark accuracy is not biased by repeated attempts.

The provided full-evaluation script uses a stable output folder by default (`<model>_10pct_test`), so restarting the same command will continue unfinished or API-failed samples instead of re-evaluating completed ones.
