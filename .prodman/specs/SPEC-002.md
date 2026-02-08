---
id: "SPEC-002"
title: "Dataset discovery and metadata inspection"
epic: "EP-002"
status: draft
author: null
reviewers: []
created_at: "2026-02-06"
updated_at: "2026-02-06"
---

# Dataset discovery and metadata inspection

## Overview

Implement dataset search and inspection commands for Hugging Face datasets to
support fast, informed curation before data cleaning.

## Problem Statement

Users need a reliable way to discover and evaluate datasets without manually
browsing web pages or writing custom scripts.

## Goals

- Provide CLI-first dataset search by query
- Support practical metadata filters for curation
- Expose inspect command for schema and split details
- Provide machine-readable output for automation

## Non-Goals

- Downloading full datasets by default
- Performing cleaning/transformation operations
- Scoring dataset quality with model-based evaluation

## Detailed Design

- Add `mlx-lab dataset search <query>` command.
- Integrate Hugging Face dataset metadata endpoints.
- Support filters such as language, tag/task, and license when available.
- Add `mlx-lab dataset inspect <dataset_id>` with split sizes and feature schema.
- Support `--json` for all dataset commands.
- Standardize error handling for network failures and missing datasets.

## Acceptance Criteria

- [ ] Search returns relevant dataset candidates with configurable limits
- [ ] Filter flags constrain results where metadata exists
- [ ] Inspect returns split and feature metadata for a valid dataset id
- [ ] `--json` output is valid JSON and stable for scripts
- [ ] Error paths return non-zero exit code and actionable diagnostics

## Open Questions

- [ ] Should cached metadata be persisted between runs by default?

