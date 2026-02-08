# mlx-lab

Local-first MLX-native toolkit for discovering, preparing, and fine-tuning
small language models on Apple Silicon.

## What It Is

`mlx-lab` provides:

- a Python library (`mlx_lab`)
- a CLI (`mlx-lab`)
- deterministic data-cleaning and LoRA run workflows
- reproducibility artifacts for replay and run comparison

## System Diagram

```text
                             +----------------------+
                             | Hugging Face Hub API |
                             +----------+-----------+
                                        |
                                        | model/dataset metadata
                                        v
+------------------+        +-----------+------------+        +-------------------+
| Local raw data   +------->+ mlx-lab CLI + library +------->+ Cleaned JSONL     |
| JSON / JSONL     |        | (model/dataset/data/  |        | prompt/completion |
+------------------+        |  train/run commands)  |        +---------+---------+
                            +-----------+------------+                  |
                                        |                               | train lora
                                        v                               v
                            +-----------+------------+        +---------+---------+
                            | Runtime + preflight    |------->+ Run directory      |
                            | checks (macOS arm64,   |        | metrics, checkpoints|
                            | dataset validity, deps)|        | manifests, state    |
                            +------------------------+        +---------------------+
```

## Workflow Diagram

```text
1) Discover candidates
   mlx-lab model search / dataset search
            |
            v
2) Inspect one model + dataset
   mlx-lab model inspect / dataset inspect
            |
            v
3) Clean raw data to canonical JSONL
   mlx-lab data clean
            |
            v
4) Run LoRA training (mlx or simulated)
   mlx-lab train lora
            |
            v
5) Replay or compare runs
   mlx-lab run replay / run compare
```

## Documentation

- Docs index: `docs/README.md`
- Quickstart: `docs/quickstart.md`
- Architecture and internals: `docs/architecture.md`
- CLI command reference: `docs/cli-reference.md`
- Release process: `docs/release-checklist.md`

## Platform

- macOS only
- Apple Silicon only
- Python 3.10+

## Reproducible Setup (uv)

```bash
uv sync --frozen
uv run mlx-lab --help
```

## Command Map

```text
mlx-lab
|-- model
|   |-- search
|   `-- inspect
|-- dataset
|   |-- search
|   `-- inspect
|-- data
|   `-- clean
|-- train
|   `-- lora
`-- run
    |-- replay
    `-- compare
```

## Reproducibility Artifacts

`mlx-lab train lora` writes reproducibility artifacts into each run directory:

- `run_manifest.json`: model/dataset fingerprint, effective config hash,
  backend, and environment snapshot.
- `metrics.jsonl`: per-step structured logs (`step`, `loss`,
  `throughput_tokens_per_s`, `learning_rate`, timestamp).
- `checkpoints/`: periodic adapter checkpoints for resume/replay.
- `train_config.resolved.json`: resolved config plus config hash.
- `preflight.json`: dataset/runtime validation report.
- `run_state.json`: latest step/checkpoint summary.

Deterministic defaults:

- `max_steps=50`
- `checkpoint_interval=10`
- `learning_rate=2e-4`
- `batch_size=4`
- `lora_rank=16`
- `seed=7`

Determinism limits:

- Requires identical cleaned dataset bytes and effective config.
- Numeric behavior can differ across backend/library versions.
- Resume and replay depend on unchanged checkpoint artifacts.

## Development Notes

- Source package: `src/mlx_lab/`
- CLI entrypoint: `mlx-lab`
- Tests: `tests/`

Run tests:

```bash
uv run python -m unittest discover -s tests -p "test_*.py"
uv run python -m unittest tests/test_release_packaging.py -v
```
