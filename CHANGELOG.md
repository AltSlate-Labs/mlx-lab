# Changelog

All notable changes to `mlx-lab` are documented in this file.

## [0.1.0] - 2026-02-07

### Added

- Python package and CLI scaffold with macOS/Apple Silicon runtime guard
- Dataset discovery and inspect workflows via Hugging Face API
- Deterministic data cleaning pipeline to instruction-style JSONL
- Model discovery and MLX compatibility classification
- LoRA-first MLX training flow with resume and checkpoint support
- Run manifest, replay, and compare commands for reproducibility
- Quickstart and release-checklist documentation
- Packaging and install smoke tests for release readiness

### Known Issues

- `ISS-001`: offline editable install may fail when build backend resolution
  requires unavailable package indexes
