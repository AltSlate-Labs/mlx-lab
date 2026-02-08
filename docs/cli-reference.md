# CLI Reference

## Root

```bash
mlx-lab --help
mlx-lab --version
```

Top-level command groups:

- `model`
- `dataset`
- `data`
- `train`
- `run`

## model

### Search

```bash
mlx-lab model search <query> [options]
```

Key options:

- `--page <int>` default `1`
- `--limit <int>` default `20`
- `--size-class any|small|medium|large|xlarge` default `any`
- `--tag <value>` repeatable
- `--license <value>`
- `--json`

Example:

```bash
mlx-lab model search llama --size-class medium --tag text-generation --json
```

### Inspect

```bash
mlx-lab model inspect <model_id> [--json]
```

Example:

```bash
mlx-lab model inspect mlx-community/Llama-3.2-1B-Instruct-4bit --json
```

## dataset

### Search

```bash
mlx-lab dataset search <query> [options]
```

Key options:

- `--page <int>` default `1`
- `--limit <int>` default `20`
- `--language <value>`
- `--task <value>`
- `--license <value>`
- `--json`

Example:

```bash
mlx-lab dataset search instruction --language en --task text-generation --json
```

### Inspect

```bash
mlx-lab dataset inspect <dataset_id> [--json]
```

Example:

```bash
mlx-lab dataset inspect tatsu-lab/alpaca --json
```

## data

### Clean

```bash
mlx-lab data clean --input <path> --output <path> [options]
```

Extraction configuration (exactly one per side):

- prompt extraction:
  - `--map-prompt <field.path>` or
  - `--prompt-template "<template>"`
- completion extraction:
  - `--map-completion <field.path>` or
  - `--completion-template "<template>"`

Key options:

- `--input-format auto|jsonl|json` default `auto`
- `--source-dataset-id <id>`
- `--source-dataset-version <version>`
- `--manifest <path>`
- `--dedupe`
- `--keep-empty`
- `--max-prompt-chars <int>`
- `--max-completion-chars <int>`
- `--json`

Example:

```bash
mlx-lab data clean \
  --input /tmp/raw.jsonl \
  --output /tmp/cleaned.jsonl \
  --map-prompt instruction \
  --map-completion answer \
  --dedupe \
  --json
```

## train

### LoRA

```bash
mlx-lab train lora [options]
```

Required inputs:

- `--model <id_or_path>`
- `--dataset <path_to_cleaned_jsonl>`

Run location options:

- `--output-dir <dir>` default `runs`
- `--run-dir <dir>` explicit path (required for `--resume` safety in practice)
- `--run-name <name>`

Behavior and hyperparameters:

- `--resume`
- `--backend auto|mlx|simulated`
- `--config <path_to_json>`
- `--max-steps <int>`
- `--checkpoint-interval <int>`
- `--learning-rate <float>`
- `--batch-size <int>`
- `--lora-rank <int>`
- `--seed <int>`
- `--json`

Deterministic defaults:

- `max_steps=50`
- `checkpoint_interval=10`
- `learning_rate=2e-4`
- `batch_size=4`
- `lora_rank=16`
- `seed=7`

Example:

```bash
mlx-lab train lora \
  --model mlx-community/Llama-3.2-1B-Instruct-4bit \
  --dataset /tmp/cleaned.jsonl \
  --run-dir /tmp/mlx-lab-run \
  --backend simulated \
  --json
```

## run

### Replay

```bash
mlx-lab run replay <run_ref> [options]
```

`run_ref` can be:

- a manifest path
- a run directory containing `run_manifest.json`
- a run id under `./runs/<run_id>/run_manifest.json`

Key options:

- `--dry-run`
- `--run-dir <dir>`
- `--run-name <name>`
- `--backend auto|mlx|simulated`
- `--max-steps <int>`
- `--checkpoint-interval <int>`
- `--json`

Examples:

```bash
mlx-lab run replay /tmp/mlx-lab-run/run_manifest.json --dry-run --json
mlx-lab run replay /tmp/mlx-lab-run/run_manifest.json --run-name replay-1 --json
```

### Compare

```bash
mlx-lab run compare <run_a> <run_b> [--json]
```

Outputs:

- metric deltas (`run_b - run_a`)
- config differences

Example:

```bash
mlx-lab run compare /tmp/run-a/run_manifest.json /tmp/run-b/run_manifest.json --json
```

## Error Behavior

- User/config/input errors return non-zero exit status with a clear message on
  `stderr`.
- Unsupported platform exits early (macOS + Apple Silicon required).
- API/network failures in discovery commands return actionable connectivity
  errors.
