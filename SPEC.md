# mlx-lab Specification

## Overview

`mlx-lab` is a local-first, MLX-native toolkit for discovering, preparing, and fine-tuning small language models using high-quality, domain-specific datasets.

It provides:
- A Python library
- A CLI tool
- Opinionated workflows for MLX fine-tuning
- Deterministic, reproducible experiments via `uv`

The project prioritizes simplicity, reproducibility, and data quality over maximal configurability.

## Goals

### Primary Goals
- Make MLX-based fine-tuning accessible and repeatable on Apple Silicon
- Provide a clean abstraction over:
  - Model discovery (Hugging Face)
  - Dataset discovery and curation
  - Data cleaning and reshaping
  - MLX fine-tuning (LoRA-first)
- Enable CLI-driven workflows suitable for agents and automation
- Be publishable and consumable as a PyPI package

### Secondary Goals
- Serve as a reference implementation for MLX workflows
- Encourage small-model, data-first training practices
- Enable future extensions such as synthetic data generation and evaluation

## Non-Goals

- Training foundation models from scratch
- Supporting non-MLX backends (PyTorch, JAX, etc.)
- Managing distributed training
- Providing hosted inference or APIs
- Acting as a full experiment tracking platform

## Target Audience

- Developers working on Apple Silicon (M1/M2/M3)
- Engineers experimenting with small language models
- Teams building domain-specific assistants or agents
- Researchers wanting a clean MLX workflow without notebook sprawl

## Platform Support

- macOS only
- Apple Silicon only
- Python >= 3.10

Linux and Windows are explicitly out of scope.

## Core Concepts

### Model
A Hugging Face-hosted language model that is:
- MLX-compatible
- Already converted or convertible to MLX format
- Typically <= 7B parameters

### Dataset
A Hugging Face dataset that can be transformed into instruction-style JSONL records.

### Cleaned Dataset
A deterministic, auditable transformation of raw dataset files into:

```json
{
  "prompt": "...",
  "completion": "..."
}
```
