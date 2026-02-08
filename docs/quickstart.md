# mlx-lab Quickstart

## Prerequisites

- macOS on Apple Silicon
- Python 3.10+
- `uv` installed

## 1) Install project dependencies

```bash
uv sync --frozen
```

## 2) Discover candidate models and datasets

These commands query Hugging Face APIs.

```bash
uv run mlx-lab model search llama --size-class medium --tag text-generation --json
uv run mlx-lab dataset search instruction --language en --task text-generation --json
```

Inspect one model and one dataset:

```bash
uv run mlx-lab model inspect mlx-community/Llama-3.2-1B-Instruct-4bit --json
uv run mlx-lab dataset inspect tatsu-lab/alpaca --json
```

## 3) Prepare cleaned instruction data

Create a small raw file:

```bash
cat > /tmp/mlx-lab-raw.jsonl <<'EOF'
{"instruction":"What is LoRA?","answer":"Low-Rank Adaptation for efficient fine-tuning."}
{"instruction":"What is MLX?","answer":"A machine learning framework from Apple."}
EOF
```

Clean into canonical training JSONL:

```bash
uv run mlx-lab data clean \
  --input /tmp/mlx-lab-raw.jsonl \
  --output /tmp/mlx-lab-cleaned.jsonl \
  --map-prompt instruction \
  --map-completion answer \
  --source-dataset-id local/demo \
  --source-dataset-version v1 \
  --json
```

Output records are:

```json
{"prompt":"...","completion":"..."}
```

## 4) Run LoRA training

Example with deterministic defaults:

```bash
uv run mlx-lab train lora \
  --model mlx-community/Llama-3.2-1B-Instruct-4bit \
  --dataset /tmp/mlx-lab-cleaned.jsonl \
  --run-dir /tmp/mlx-lab-run \
  --json
```

If you do not have MLX runtime dependencies installed in your environment, use:

```bash
uv run mlx-lab train lora \
  --model mlx-community/Llama-3.2-1B-Instruct-4bit \
  --dataset /tmp/mlx-lab-cleaned.jsonl \
  --run-dir /tmp/mlx-lab-run \
  --backend simulated \
  --json
```

The run directory includes:

- `run_manifest.json`
- `metrics.jsonl`
- `checkpoints/`
- `preflight.json`
- `train_config.resolved.json`

## 5) Replay and compare runs

```bash
uv run mlx-lab run replay /tmp/mlx-lab-run/run_manifest.json --dry-run --json
uv run mlx-lab run replay /tmp/mlx-lab-run/run_manifest.json --run-name replay-1 --json
uv run mlx-lab run compare /tmp/mlx-lab-run/run_manifest.json /tmp/mlx-lab-run-replay-1/run_manifest.json --json
```

