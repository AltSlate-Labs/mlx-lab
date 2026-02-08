---
id: "SPEC-003"
title: "Deterministic dataset cleaning to instruction JSONL"
epic: "EP-003"
status: draft
author: null
reviewers: []
created_at: "2026-02-06"
updated_at: "2026-02-06"
---

# Deterministic dataset cleaning to instruction JSONL

## Overview

Create a deterministic pipeline that transforms raw dataset records into
instruction-style JSONL records with `prompt` and `completion` fields.

## Problem Statement

High-quality fine-tuning requires clean, auditable training data. Ad hoc
transforms produce irreproducible results and hidden data quality issues.

## Goals

- Convert source records to canonical prompt/completion schema
- Enforce deterministic ordering and transformation behavior
- Validate and report data quality issues
- Produce manifest artifacts for auditability

## Non-Goals

- Synthetic data generation
- Automated labeling or rewriting with external LLMs
- Distributed data processing infrastructure

## Detailed Design

- Add `mlx-lab data clean` command that accepts input dataset plus mapping config.
- Support field mapping and templating for `prompt` and `completion`.
- Apply deterministic transform pipeline with explicit stage order.
- Add validators for empty fields, duplicate rows, length constraints, and bad encoding.
- Emit cleaned JSONL and summary statistics.
- Generate data manifest containing source reference, transform hash, and output checksum.

## Acceptance Criteria

- [ ] Cleaning command outputs valid JSONL with one record per line
- [ ] Each record contains only `prompt` and `completion` string fields
- [ ] Identical inputs and config produce identical output bytes
- [ ] Validation report includes counts of kept, dropped, and invalid records
- [ ] Manifest captures provenance and checksums for audit/replay

## Open Questions

- [ ] Should deduplication be exact-match only or include normalized matching?

