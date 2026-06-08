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

Download the constructed 10% GPSBench evaluation split from Hugging Face into the project root. Use `uvx hf` so the download command works even before the project environment is activated:

```bash
uvx hf download zhangdw/GPSBench-10pct \
  --type dataset \
  --include 'data/**' \
  --local-dir .
```

This downloads only the benchmark `data/` tree. After download, the runner uses the normal default paths:

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
