# mlx-lab Architecture

## Scope

`mlx-lab` is a local-first toolkit for MLX LoRA workflows on Apple Silicon.
It focuses on deterministic data preparation and reproducible run artifacts.

## Component Overview

```text
                  +-----------------------+
                  |     CLI Entrypoint    |
                  |    src/mlx_lab/cli.py |
                  +-----------+-----------+
                              |
        +---------------------+---------------------+
        |                     |                     |
        v                     v                     v
+-------+--------+   +--------+--------+   +--------+--------+
| Discovery layer|   | Data cleaning   |   | Training + runs |
| hf_models.py   |   | data_cleaning.py|   | training_lora.py|
| hf_datasets.py |   | commands/data.py|   | run_ops.py      |
+-------+--------+   +--------+--------+   +--------+--------+
        |                     |                     |
        v                     v                     v
 Hugging Face APIs       Canonical JSONL       Run artifacts directory
 (models, datasets)      prompt/completion     manifests, metrics, ckpts
```

## Runtime Constraints

- OS: macOS only
- CPU architecture: Apple Silicon (`arm64`, `arm64e`)
- Guard location: `src/mlx_lab/runtime.py`

The platform guard runs before command execution and exits with status `2` on
unsupported systems.

## Command Layer

Command modules live in `src/mlx_lab/commands/`:

- `model.py`: `model search`, `model inspect`
- `dataset.py`: `dataset search`, `dataset inspect`
- `data.py`: `data clean`
- `train.py`: `train lora`
- `run.py`: `run replay`, `run compare`

Each command supports structured output with `--json` for automation-friendly
integration.

## Data Flow

```text
raw JSON/JSONL
    |
    | mlx-lab data clean
    v
cleaned JSONL (prompt/completion)
    |
    | mlx-lab train lora
    v
run_dir/
  - run_manifest.json
  - metrics.jsonl
  - checkpoints/
  - train_config.resolved.json
  - preflight.json
  - run_state.json
    |
    +--> mlx-lab run replay
    +--> mlx-lab run compare
```

## Determinism Model

Determinism is produced through:

- stable JSON serialization for config hashing
- byte-level dataset hashing
- explicit seeded training defaults
- immutable run manifests containing effective config and environment details

Current defaults:

- `max_steps=50`
- `checkpoint_interval=10`
- `learning_rate=2e-4`
- `batch_size=4`
- `lora_rank=16`
- `seed=7`

Limits:

- backend/library version changes can alter numeric trajectories
- replay fidelity assumes source dataset and checkpoint artifacts are unchanged

## Artifact Contracts

### `data clean` manifest

Default path: `<output>.manifest.json`

Includes:

- source metadata (`dataset_id`, `dataset_version`, `input_sha256`)
- transform config and `config_sha256`
- output stats and `sha256`

### `train lora` manifest

Path: `<run_dir>/run_manifest.json`

Includes:

- run identity and status
- dataset hash and record count
- effective config and hash
- runtime dependency/environment snapshot
- artifact paths for metrics/checkpoints/config/preflight/state

## Dependency Boundaries

- Discovery commands depend on Hugging Face APIs and network availability.
- Cleaning, training (simulated backend), replay, and compare work with local
  files once data is available.
- MLX backend selection is resolved during preflight and can fall back to
  `simulated` when requested.
