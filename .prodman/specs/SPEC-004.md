---
id: "SPEC-004"
title: "Model discovery and MLX compatibility assessment"
epic: "EP-004"
status: draft
author: null
reviewers: []
created_at: "2026-02-06"
updated_at: "2026-02-06"
---

# Model discovery and MLX compatibility assessment

## Overview

Implement model search and inspection workflows that surface MLX-ready or
MLX-convertible small language models from Hugging Face.

## Problem Statement

Users need quick guidance to select models that fit local Apple Silicon
constraints and MLX training compatibility.

## Goals

- Provide search and inspect commands for candidate models
- Surface compatibility and size constraints relevant to MLX workflows
- Keep outputs script-friendly for agentic workflows

## Non-Goals

- Automated model downloads during search
- Benchmarking model quality on downstream tasks
- Supporting non-MLX backends

## Detailed Design

- Add `mlx-lab model search <query>` with ranking and limit controls.
- Support optional filters: parameter scale, license, architecture tags.
- Add `mlx-lab model inspect <model_id>` for key metadata and tokenizer details.
- Implement compatibility classification: `mlx_ready`, `convertible`, or `unsupported`.
- Include `--json` output for both commands.
- Add clear diagnostics for invalid model ids and unavailable metadata.

## Acceptance Criteria

- [ ] Search returns ranked model candidates for a query
- [ ] Filter options constrain result set when metadata is present
- [ ] Inspect command returns normalized metadata fields
- [ ] Compatibility label is present for inspected models
- [ ] All command outputs support text and JSON modes

## Open Questions

- [ ] Should the default ranking prioritize popularity, recency, or size fit?

