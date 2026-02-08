---
id: "SPEC-001"
title: "Architecture bootstrap for library and CLI"
epic: "EP-001"
status: draft
author: null
reviewers: []
created_at: "2026-02-06"
updated_at: "2026-02-06"
---

# Architecture bootstrap for library and CLI

## Overview

Establish a reproducible project baseline for `mlx-lab` as a Python package and
CLI with strict platform guards for macOS on Apple Silicon.

## Problem Statement

Without a stable and deterministic foundation, later data and training workflows
will diverge across machines and be difficult to automate.

## Goals

- Create a uv-managed Python project with locked dependencies
- Expose an installable `mlx-lab` CLI and importable `mlx_lab` package
- Enforce supported platform constraints with clear errors
- Define stable command groups to support future epics

## Non-Goals

- Implementing model, dataset, or training logic
- Supporting Linux, Windows, or non-Apple Silicon hosts
- Building advanced plugin or extension systems

## Detailed Design

- Use `pyproject.toml` with console script entrypoint `mlx-lab`.
- Place package source under `src/mlx_lab/`.
- Add command group skeletons for `model`, `dataset`, `data`, `train`, and `run`.
- Add runtime environment checks for Darwin and ARM64 before heavy operations.
- Commit lockfile and rely on `uv sync` plus `uv run` for deterministic execution.
- Add smoke tests for package import and CLI help rendering.

## Acceptance Criteria

- [ ] Dependency install and execution are reproducible through `uv`
- [ ] `mlx-lab --help` and group help commands execute successfully
- [ ] Unsupported hosts fail with actionable messages
- [ ] Package can be imported in a clean environment
- [ ] Smoke tests verify baseline behavior

## Open Questions

- [ ] Should the default CLI output mode be text or JSON?

