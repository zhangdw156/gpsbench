# GPSBench Evaluation Harness

This fork is trimmed to the code needed to evaluate GPSBench-style test splits with an OpenAI-compatible model endpoint, including local vLLM servers.

## What is kept

- `run_benchmark.py` — main evaluation runner
- `evaluation/llm_client.py` — OpenAI/OpenRouter/Gemini-compatible client wrapper
- `prompts/` — task prompt templates
- `.env.example` / `requirements.txt` — minimal runtime setup

Generated data and results are intentionally not versioned.

## Setup

```bash
pip install -r requirements.txt
```

## Download GPSBench-10pct data

Download only the benchmark `data/` tree into the project root:

```bash
hf download zhangdw/GPSBench-10pct \
  --type dataset \
  --include 'data/**' \
  --local-dir .
```

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

Run a smoke test:

```bash
python run_benchmark.py \
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
python run_benchmark.py \
  --provider openai \
  --model my-vllm-model \
  --track both \
  --concurrent \
  --max-workers 16 \
  --delay 0
```

Results are written under `results/`, which is ignored by git.
