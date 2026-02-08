"""Run manifest utilities for replay and comparison workflows."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .training_lora import TrainingError, run_lora_training

JsonObject = dict[str, Any]


class RunOpsError(RuntimeError):
    """Raised when run manifest operations fail."""


def replay_run(
    run_id_or_manifest: str,
    *,
    execute: bool = True,
    run_dir: str | None = None,
    run_name: str | None = None,
    backend: str | None = None,
    max_steps: int | None = None,
    checkpoint_interval: int | None = None,
) -> JsonObject:
    """Replay a prior run from its manifest."""
    manifest_path = resolve_manifest_path(run_id_or_manifest)
    manifest = _load_manifest(manifest_path)
    _validate_manifest(manifest, manifest_path)

    effective_config = manifest["config"]["effective"]
    source_run_dir = Path(manifest["run"]["run_dir"])
    source_output_dir = str(source_run_dir.parent)

    replay_run_name = run_name
    replay_run_dir = run_dir
    if not replay_run_name and not replay_run_dir:
        replay_run_name = (
            f"{manifest['run']['id']}-replay-"
            f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        )

    replay_parameters = {
        "model": manifest["model"]["id_or_path"],
        "dataset_path": manifest["dataset"]["path"],
        "output_dir": source_output_dir,
        "run_dir": replay_run_dir,
        "run_name": replay_run_name,
        "resume": False,
        "backend": backend if backend is not None else effective_config.get("backend"),
        "max_steps": max_steps if max_steps is not None else effective_config.get("max_steps"),
        "checkpoint_interval": (
            checkpoint_interval
            if checkpoint_interval is not None
            else effective_config.get("checkpoint_interval")
        ),
        "learning_rate": effective_config.get("learning_rate"),
        "batch_size": effective_config.get("batch_size"),
        "lora_rank": effective_config.get("lora_rank"),
        "seed": effective_config.get("seed"),
    }

    if not execute:
        return {
            "mode": "dry_run",
            "source_manifest": str(manifest_path),
            "source_run_id": manifest["run"]["id"],
            "replay_parameters": replay_parameters,
        }

    try:
        replay_summary = run_lora_training(**replay_parameters)
    except TrainingError as exc:
        raise RunOpsError(str(exc)) from exc

    return {
        "mode": "executed",
        "source_manifest": str(manifest_path),
        "source_run_id": manifest["run"]["id"],
        "replay_parameters": replay_parameters,
        "replay_result": replay_summary,
    }


def compare_runs(run_a: str, run_b: str) -> JsonObject:
    """Compare two runs by manifest/config/metrics."""
    manifest_a_path = resolve_manifest_path(run_a)
    manifest_b_path = resolve_manifest_path(run_b)
    manifest_a = _load_manifest(manifest_a_path)
    manifest_b = _load_manifest(manifest_b_path)
    _validate_manifest(manifest_a, manifest_a_path)
    _validate_manifest(manifest_b, manifest_b_path)

    metrics_a = _load_metrics(manifest_a["artifacts"]["metrics_path"])
    metrics_b = _load_metrics(manifest_b["artifacts"]["metrics_path"])
    summary_a = _summarize_metrics(metrics_a)
    summary_b = _summarize_metrics(metrics_b)

    config_a = _flat_compare_config(manifest_a)
    config_b = _flat_compare_config(manifest_b)
    config_differences = _config_differences(config_a, config_b)

    deltas = {
        "final_loss_delta": _subtract(summary_b["final_loss"], summary_a["final_loss"]),
        "best_loss_delta": _subtract(summary_b["best_loss"], summary_a["best_loss"]),
        "avg_throughput_tokens_per_s_delta": _subtract(
            summary_b["avg_throughput_tokens_per_s"],
            summary_a["avg_throughput_tokens_per_s"],
        ),
        "total_steps_delta": _subtract(summary_b["total_steps"], summary_a["total_steps"]),
    }

    return {
        "run_a": {
            "manifest_path": str(manifest_a_path),
            "run_id": manifest_a["run"]["id"],
            "summary": summary_a,
            "config": config_a,
        },
        "run_b": {
            "manifest_path": str(manifest_b_path),
            "run_id": manifest_b["run"]["id"],
            "summary": summary_b,
            "config": config_b,
        },
        "deltas": deltas,
        "config_differences": config_differences,
    }


def resolve_manifest_path(run_id_or_manifest: str) -> Path:
    """Resolve run reference into a concrete manifest path."""
    candidate = Path(run_id_or_manifest)
    if candidate.exists():
        if candidate.is_file():
            return candidate
        if candidate.is_dir():
            manifest = candidate / "run_manifest.json"
            if manifest.exists():
                return manifest
            raise RunOpsError(f"Run directory does not contain run_manifest.json: {candidate}")

    default_runs_manifest = Path("runs") / run_id_or_manifest / "run_manifest.json"
    if default_runs_manifest.exists():
        return default_runs_manifest

    raise RunOpsError(
        f"Unable to resolve run reference '{run_id_or_manifest}'. "
        "Provide a run directory, manifest path, or run id under ./runs."
    )


def _load_manifest(path: Path) -> JsonObject:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        raise RunOpsError(f"Manifest file is not valid UTF-8: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RunOpsError(f"Manifest file is not valid JSON: {path} ({exc})") from exc

    if not isinstance(payload, dict):
        raise RunOpsError(f"Manifest root must be an object: {path}")
    return payload


def _validate_manifest(manifest: JsonObject, path: Path) -> None:
    required_paths = [
        ("manifest_type",),
        ("run", "id"),
        ("run", "run_dir"),
        ("model", "id_or_path"),
        ("dataset", "path"),
        ("dataset", "sha256"),
        ("config", "effective"),
        ("config", "effective_config_sha256"),
        ("artifacts", "metrics_path"),
    ]
    if manifest.get("manifest_type") != "lora_run":
        raise RunOpsError(f"Unsupported manifest type in {path}: {manifest.get('manifest_type')}")
    for field_path in required_paths:
        value = _lookup(manifest, field_path)
        if value in (None, ""):
            dotted = ".".join(field_path)
            raise RunOpsError(f"Manifest missing required field `{dotted}` in {path}")


def _load_metrics(metrics_path: str) -> list[JsonObject]:
    path = Path(metrics_path)
    if not path.exists() or not path.is_file():
        raise RunOpsError(f"Metrics file does not exist: {path}")

    records: list[JsonObject] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RunOpsError(
                f"Metrics file contains invalid JSON at line {index}: {path}"
            ) from exc
        if not isinstance(payload, dict):
            raise RunOpsError(
                f"Metrics file line {index} must be a JSON object: {path}"
            )
        records.append(payload)
    return records


def _summarize_metrics(records: list[JsonObject]) -> JsonObject:
    if not records:
        return {
            "total_steps": 0,
            "final_step": 0,
            "final_loss": None,
            "best_loss": None,
            "avg_throughput_tokens_per_s": None,
        }

    total_steps = len(records)
    final_record = records[-1]
    losses = [value for value in (record.get("loss") for record in records) if isinstance(value, (int, float))]
    throughputs = [
        value
        for value in (record.get("throughput_tokens_per_s") for record in records)
        if isinstance(value, (int, float))
    ]

    return {
        "total_steps": total_steps,
        "final_step": final_record.get("step"),
        "final_loss": final_record.get("loss"),
        "best_loss": min(losses) if losses else None,
        "avg_throughput_tokens_per_s": (
            round(sum(throughputs) / len(throughputs), 6) if throughputs else None
        ),
    }


def _flat_compare_config(manifest: JsonObject) -> JsonObject:
    effective = manifest["config"]["effective"]
    return {
        "model": manifest["model"]["id_or_path"],
        "dataset_sha256": manifest["dataset"]["sha256"],
        "selected_backend": effective.get("selected_backend"),
        "max_steps": effective.get("max_steps"),
        "checkpoint_interval": effective.get("checkpoint_interval"),
        "learning_rate": effective.get("learning_rate"),
        "batch_size": effective.get("batch_size"),
        "lora_rank": effective.get("lora_rank"),
        "seed": effective.get("seed"),
        "effective_config_sha256": manifest["config"]["effective_config_sha256"],
    }


def _config_differences(config_a: JsonObject, config_b: JsonObject) -> list[JsonObject]:
    differences: list[JsonObject] = []
    keys = sorted(set(config_a.keys()) | set(config_b.keys()))
    for key in keys:
        if config_a.get(key) == config_b.get(key):
            continue
        differences.append(
            {
                "key": key,
                "run_a": config_a.get(key),
                "run_b": config_b.get(key),
            }
        )
    return differences


def _subtract(left: Any, right: Any) -> float | None:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return round(float(left) - float(right), 6)
    return None


def _lookup(payload: JsonObject, path: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current

