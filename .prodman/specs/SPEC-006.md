---
id: "SPEC-006"
title: "Deterministic experiments and run replay"
epic: "EP-006"
status: draft
author: null
reviewers: []
created_at: "2026-02-06"
updated_at: "2026-02-06"
---

# Deterministic experiments and run replay

## Overview

Add end-to-end experiment manifests and replay tooling so training runs are
auditable and reproducible.

## Problem Statement

Without run provenance, teams cannot confidently compare outcomes or reproduce
previous experiments for debugging and iteration.

## Goals

- Persist complete run manifests for every training execution
- Capture data/config/environment fingerprints
- Support replaying prior runs from manifest references
- Provide lightweight run comparison tooling

## Non-Goals

- Building hosted experiment tracking services
- Real-time collaborative dashboards
- Long-term artifact storage orchestration

## Detailed Design

- Define manifest schema containing run metadata, model id/revision, dataset hashes, and resolved config.
- Capture environment info: macOS version, Python version, MLX versions, and hardware identifier.
- Add `mlx-lab run replay <manifest_or_run_id>` command to rehydrate and relaunch runs.
- Add `mlx-lab run compare <run_a> <run_b>` for metric and config deltas.
- Include deterministic seed handling and record seed state in manifests.

## Acceptance Criteria

- [ ] Every run writes a complete manifest in a predictable location
- [ ] Manifest includes checksums for cleaned data and effective config
- [ ] Replay command can launch using prior manifest inputs
- [ ] Compare command summarizes key metrics and config differences
- [ ] Determinism controls are documented and applied consistently

## Open Questions

- [ ] Should replay default to dry-run mode unless `--execute` is provided?

