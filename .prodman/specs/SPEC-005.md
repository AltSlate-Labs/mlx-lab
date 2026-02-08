---
id: "SPEC-005"
title: "LoRA-first MLX fine-tuning workflow"
epic: "EP-005"
status: draft
author: null
reviewers: []
created_at: "2026-02-06"
updated_at: "2026-02-06"
---

# LoRA-first MLX fine-tuning workflow

## Overview

Deliver the core `mlx-lab` workflow to run LoRA-based fine-tuning on MLX from
cleaned instruction data and selected models.

## Problem Statement

Users need one reliable local command path to start, monitor, and resume
fine-tuning without piecing together many scripts.

## Goals

- Expose a single training entrypoint for LoRA fine-tuning
- Validate inputs before launch to avoid wasted runs
- Persist checkpoints and support resuming interrupted jobs
- Emit structured training logs for analysis and automation

## Non-Goals

- Full hyperparameter search platform
- Multi-node or distributed training
- Non-LoRA adapter strategies in the first release

## Detailed Design

- Add `mlx-lab train lora` command with model, dataset, and config options.
- Resolve config with sensible defaults and explicit override precedence.
- Add preflight checks for dataset schema, local files, and runtime dependencies.
- Run MLX fine-tuning loop with periodic checkpointing in run-specific directories.
- Add resume mode from checkpoint metadata.
- Emit structured metrics logs including step, loss, learning rate, and throughput.

## Acceptance Criteria

- [ ] Training command starts successfully with valid model and cleaned dataset
- [ ] Preflight validation blocks unsupported inputs with actionable errors
- [ ] Checkpoints are written at configured intervals
- [ ] Resume flow restores state and continues training
- [ ] Logs include machine-readable per-step metrics

## Open Questions

- [ ] What should be the default checkpoint frequency for local workflows?

