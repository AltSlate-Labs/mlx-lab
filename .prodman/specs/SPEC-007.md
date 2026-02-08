---
id: "SPEC-007"
title: "Packaging, documentation, and release readiness"
epic: "EP-007"
status: draft
author: null
reviewers: []
created_at: "2026-02-06"
updated_at: "2026-02-06"
---

# Packaging, documentation, and release readiness

## Overview

Prepare `mlx-lab` for public consumption as a PyPI package with clear docs,
smoke-tested install paths, and release procedures.

## Problem Statement

Even with working functionality, adoption fails if installation and onboarding
are unclear or release processes are inconsistent.

## Goals

- Ensure package artifacts are buildable and installable
- Document end-to-end CLI and library quickstart paths
- Add smoke tests that validate install and minimal workflow
- Define repeatable release process for versioning and publishing

## Non-Goals

- Full documentation portal with exhaustive tutorials
- Multi-platform packaging support beyond macOS Apple Silicon
- Hosted deployment guides

## Detailed Design

- Finalize project metadata for wheel and source distribution.
- Verify console script registration and CLI execution post-install.
- Write quickstart docs for discovery, cleaning, and LoRA training workflow.
- Add smoke tests for clean-environment install and representative command runs.
- Document release checklist: version bump, changelog update, build, and publish.

## Acceptance Criteria

- [ ] `uv build` produces valid sdist and wheel artifacts
- [ ] Fresh install exposes working `mlx-lab` CLI
- [ ] Quickstart docs cover full path from dataset/model discovery to train command
- [ ] Smoke tests pass for install and minimal command execution
- [ ] Release checklist is documented and actionable

## Open Questions

- [ ] Should first release include a constrained API stability policy?

