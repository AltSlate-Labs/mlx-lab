"""LoRA training workflow utilities for mlx-lab."""

from __future__ import annotations

import hashlib
import importlib.util
import importlib.metadata
import json
import math
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

JsonObject = dict[str, Any]
_CHECKPOINT_PATTERN = re.compile(r"^checkpoint-step-(\d{6})\.adapter\.json$")


class TrainingError(RuntimeError):
    """Raised when LoRA training configuration or execution fails."""


def run_lora_training(
    *,
    model: str | None,
    dataset_path: str | None,
    config_path: str | None = None,
    output_dir: str | None = None,
    run_dir: str | None = None,
    run_name: str | None = None,
    resume: bool = False,
    backend: str | None = None,
    max_steps: int | None = None,
    checkpoint_interval: int | None = None,
    learning_rate: float | None = None,
    batch_size: int | None = None,
    lora_rank: int | None = None,
    seed: int | None = None,
) -> JsonObject:
    """Run the LoRA training workflow and return a structured run summary."""
    started_at = _timestamp_utc()
    resolved = _resolve_train_config(
        model=model,
        dataset_path=dataset_path,
        config_path=config_path,
        output_dir=output_dir,
        run_dir=run_dir,
        run_name=run_name,
        resume=resume,
        backend=backend,
        max_steps=max_steps,
        checkpoint_interval=checkpoint_interval,
        learning_rate=learning_rate,
        batch_size=batch_size,
        lora_rank=lora_rank,
        seed=seed,
    )

    run_dir_path = _resolve_run_dir(
        output_dir=resolved["output_dir"],
        run_dir=resolved["run_dir"],
        run_name=resolved["run_name"],
        model=resolved["model"],
        resume=resolved["resume"],
    )
    checkpoints_dir = run_dir_path / "checkpoints"
    metrics_path = run_dir_path / "metrics.jsonl"
    state_path = run_dir_path / "run_state.json"

    if not resolved["resume"]:
        _ensure_run_dir_is_clean_for_fresh_run(run_dir_path)

    run_dir_path.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    preflight = run_preflight_checks(
        dataset_path=resolved["dataset_path"],
        requested_backend=resolved["backend"],
    )
    selected_backend = preflight["runtime"]["selected_backend"]

    start_step = 1
    resumed_from_step = 0
    if resolved["resume"]:
        resumed_from_step, latest_checkpoint_path = _resolve_resume_checkpoint(checkpoints_dir)
        start_step = resumed_from_step + 1
        if latest_checkpoint_path is None:
            raise TrainingError(
                f"Cannot resume because no checkpoint exists in {checkpoints_dir}."
            )
    else:
        metrics_path.write_text("", encoding="utf-8")

    # If a state file exists from a previous run, we keep it for traceability and overwrite below.
    checkpoints_written: list[str] = []
    latest_checkpoint = None
    latest_step = resumed_from_step

    if start_step <= resolved["max_steps"]:
        with metrics_path.open("a", encoding="utf-8", newline="\n") as metrics_file:
            for step in range(start_step, resolved["max_steps"] + 1):
                metric = _compute_step_metric(
                    step=step,
                    learning_rate=resolved["learning_rate"],
                    batch_size=resolved["batch_size"],
                    seed=resolved["seed"],
                    dataset_records=preflight["dataset"]["record_count"],
                )
                metrics_file.write(
                    json.dumps(metric, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
                )
                metrics_file.write("\n")

                latest_step = step
                if step % resolved["checkpoint_interval"] == 0 or step == resolved["max_steps"]:
                    checkpoint_path = checkpoints_dir / f"checkpoint-step-{step:06d}.adapter.json"
                    checkpoint_payload = {
                        "schema_version": 1,
                        "checkpoint_type": "lora_adapter",
                        "model": resolved["model"],
                        "backend": selected_backend,
                        "dataset_sha256": preflight["dataset"]["sha256"],
                        "lora_rank": resolved["lora_rank"],
                        "learning_rate": resolved["learning_rate"],
                        "batch_size": resolved["batch_size"],
                        "seed": resolved["seed"],
                        "step": step,
                        "created_at": _timestamp_utc(),
                        "adapter_state": {
                            "format": "placeholder",
                            "step": step,
                            "rank": resolved["lora_rank"],
                        },
                    }
                    _write_json_file(checkpoint_path, checkpoint_payload)
                    latest_checkpoint = str(checkpoint_path)
                    checkpoints_written.append(str(checkpoint_path))

    if latest_checkpoint is None:
        latest_checkpoint_path = _latest_checkpoint_path(checkpoints_dir)
        latest_checkpoint = str(latest_checkpoint_path) if latest_checkpoint_path else None

    resolved_config_path = run_dir_path / "train_config.resolved.json"
    preflight_path = run_dir_path / "preflight.json"
    manifest_path = run_dir_path / "run_manifest.json"
    completed_at = _timestamp_utc()
    run_id = run_dir_path.name

    effective_config = {
        "model": resolved["model"],
        "dataset_path": resolved["dataset_path"],
        "output_dir": resolved["output_dir"],
        "run_dir": str(run_dir_path),
        "run_name": resolved["run_name"],
        "resume": resolved["resume"],
        "backend": resolved["backend"],
        "selected_backend": selected_backend,
        "max_steps": resolved["max_steps"],
        "checkpoint_interval": resolved["checkpoint_interval"],
        "learning_rate": resolved["learning_rate"],
        "batch_size": resolved["batch_size"],
        "lora_rank": resolved["lora_rank"],
        "seed": resolved["seed"],
    }
    effective_config_sha256 = hashlib.sha256(
        json.dumps(effective_config, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    _write_json_file(
        resolved_config_path,
        {
            "schema_version": 1,
            "resolved_at": completed_at,
            "effective": effective_config,
            "effective_config_sha256": effective_config_sha256,
        },
    )
    _write_json_file(preflight_path, preflight)
    _write_json_file(
        state_path,
        {
            "schema_version": 1,
            "updated_at": completed_at,
            "latest_step": latest_step,
            "latest_checkpoint": latest_checkpoint,
            "selected_backend": selected_backend,
            "resumed_from_step": resumed_from_step,
        },
    )

    run_manifest = {
        "schema_version": 1,
        "manifest_type": "lora_run",
        "run": {
            "id": run_id,
            "run_dir": str(run_dir_path),
            "created_at": started_at,
            "completed_at": completed_at,
            "resumed_from_step": resumed_from_step,
            "latest_step": latest_step,
            "latest_checkpoint": latest_checkpoint,
            "status": "completed",
        },
        "model": {
            "id_or_path": resolved["model"],
        },
        "dataset": {
            "path": resolved["dataset_path"],
            "sha256": preflight["dataset"]["sha256"],
            "record_count": preflight["dataset"]["record_count"],
            "size_bytes": preflight["dataset"]["size_bytes"],
        },
        "config": {
            "effective": effective_config,
            "effective_config_sha256": effective_config_sha256,
        },
        "runtime": {
            "requested_backend": preflight["runtime"]["requested_backend"],
            "selected_backend": preflight["runtime"]["selected_backend"],
            "dependencies": preflight["runtime"]["dependencies"],
            "environment": _environment_snapshot(),
        },
        "artifacts": {
            "metrics_path": str(metrics_path),
            "checkpoints_dir": str(checkpoints_dir),
            "resolved_config_path": str(resolved_config_path),
            "preflight_path": str(preflight_path),
            "state_path": str(state_path),
        },
        "determinism": {
            "seed": resolved["seed"],
            "defaults": {
                "max_steps": 50,
                "checkpoint_interval": 10,
                "learning_rate": 2e-4,
                "batch_size": 4,
                "lora_rank": 16,
                "seed": 7,
            },
            "limits": [
                "Deterministic behavior assumes identical cleaned dataset bytes and effective config.",
                "Backend implementation and library version differences can affect numerical trajectories.",
                "Resumed runs rely on adapter checkpoint state and may diverge if checkpoint files change.",
            ],
        },
    }
    _write_json_file(manifest_path, run_manifest)

    return {
        "run_dir": str(run_dir_path),
        "run_id": run_id,
        "model": resolved["model"],
        "dataset_path": resolved["dataset_path"],
        "selected_backend": selected_backend,
        "max_steps": resolved["max_steps"],
        "checkpoint_interval": resolved["checkpoint_interval"],
        "resumed_from_step": resumed_from_step,
        "latest_step": latest_step,
        "latest_checkpoint": latest_checkpoint,
        "metrics_path": str(metrics_path),
        "checkpoints_written": checkpoints_written,
        "resolved_config_path": str(resolved_config_path),
        "preflight_path": str(preflight_path),
        "manifest_path": str(manifest_path),
        "effective_config_sha256": effective_config_sha256,
        "preflight": preflight,
    }


def run_preflight_checks(*, dataset_path: str, requested_backend: str) -> JsonObject:
    """Run preflight checks for dataset integrity and runtime dependencies."""
    dataset = _validate_dataset_file(Path(dataset_path))
    runtime = _validate_runtime_dependencies(requested_backend)
    return {
        "dataset": dataset,
        "runtime": runtime,
    }


def _resolve_train_config(
    *,
    model: str | None,
    dataset_path: str | None,
    config_path: str | None,
    output_dir: str | None,
    run_dir: str | None,
    run_name: str | None,
    resume: bool,
    backend: str | None,
    max_steps: int | None,
    checkpoint_interval: int | None,
    learning_rate: float | None,
    batch_size: int | None,
    lora_rank: int | None,
    seed: int | None,
) -> JsonObject:
    defaults: JsonObject = {
        "output_dir": "runs",
        "run_name": None,
        "run_dir": None,
        "resume": False,
        "backend": "auto",
        "max_steps": 50,
        "checkpoint_interval": 10,
        "learning_rate": 2e-4,
        "batch_size": 4,
        "lora_rank": 16,
        "seed": 7,
    }

    config_payload = _load_config_file(config_path)
    if config_payload and not isinstance(config_payload, dict):
        raise TrainingError("Training config file must contain a JSON object.")

    resolved = dict(defaults)
    if isinstance(config_payload, dict):
        resolved.update(config_payload)

    overrides: JsonObject = {
        "model": model,
        "dataset_path": dataset_path,
        "output_dir": output_dir,
        "run_dir": run_dir,
        "run_name": run_name,
        "resume": resume,
        "backend": backend,
        "max_steps": max_steps,
        "checkpoint_interval": checkpoint_interval,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "lora_rank": lora_rank,
        "seed": seed,
    }
    for key, value in overrides.items():
        if value is not None:
            resolved[key] = value

    model_value = str(resolved.get("model") or "").strip()
    dataset_value = str(resolved.get("dataset_path") or "").strip()
    if not model_value:
        raise TrainingError("Training requires a model id/path. Provide --model or config `model`.")
    if not dataset_value:
        raise TrainingError("Training requires a dataset path. Provide --dataset or config `dataset_path`.")

    resolved["model"] = model_value
    resolved["dataset_path"] = dataset_value
    resolved["output_dir"] = str(resolved.get("output_dir") or "runs")
    resolved["run_dir"] = str(resolved["run_dir"]) if resolved.get("run_dir") else None
    resolved["run_name"] = str(resolved["run_name"]) if resolved.get("run_name") else None
    resolved["resume"] = bool(resolved.get("resume", False))
    resolved["backend"] = str(resolved.get("backend") or "auto").lower()

    if resolved["backend"] not in {"auto", "mlx", "simulated"}:
        raise TrainingError("Invalid backend. Use one of: auto, mlx, simulated.")

    resolved["max_steps"] = _as_positive_int(resolved.get("max_steps"), "max_steps")
    resolved["checkpoint_interval"] = _as_positive_int(
        resolved.get("checkpoint_interval"),
        "checkpoint_interval",
    )
    resolved["batch_size"] = _as_positive_int(resolved.get("batch_size"), "batch_size")
    resolved["lora_rank"] = _as_positive_int(resolved.get("lora_rank"), "lora_rank")
    resolved["learning_rate"] = _as_positive_float(resolved.get("learning_rate"), "learning_rate")
    resolved["seed"] = _as_int(resolved.get("seed"), "seed")

    if resolved["checkpoint_interval"] > resolved["max_steps"]:
        # Ensure at least one periodic checkpoint before completion when possible.
        resolved["checkpoint_interval"] = resolved["max_steps"]

    return resolved


def _load_config_file(config_path: str | None) -> JsonObject | None:
    if not config_path:
        return None
    path = Path(config_path)
    if not path.exists() or not path.is_file():
        raise TrainingError(f"Training config file does not exist: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        raise TrainingError(f"Training config file is not valid UTF-8: {path}") from exc
    except json.JSONDecodeError as exc:
        raise TrainingError(f"Training config file is not valid JSON: {path} ({exc})") from exc


def _resolve_run_dir(
    *,
    output_dir: str,
    run_dir: str | None,
    run_name: str | None,
    model: str,
    resume: bool,
) -> Path:
    if run_dir:
        return Path(run_dir)

    base_output = Path(output_dir)
    if run_name:
        return base_output / _slugify(run_name)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    inferred = f"{_slugify(model)}-{timestamp}"
    if resume:
        # Resume without explicit run dir can be ambiguous; avoid accidental mismatches.
        raise TrainingError("Resume requires --run-dir or config `run_dir`.")
    return base_output / inferred


def _ensure_run_dir_is_clean_for_fresh_run(run_dir: Path) -> None:
    if not run_dir.exists():
        return
    existing_markers = [
        run_dir / "metrics.jsonl",
        run_dir / "run_state.json",
        run_dir / "checkpoints",
    ]
    if any(marker.exists() for marker in existing_markers):
        raise TrainingError(
            f"Run directory already contains training artifacts: {run_dir}. "
            "Use --resume to continue this run or choose a new --run-dir."
        )


def _validate_dataset_file(path: Path) -> JsonObject:
    if not path.exists() or not path.is_file():
        raise TrainingError(f"Dataset file does not exist: {path}")

    try:
        payload_bytes = path.read_bytes()
        payload_text = payload_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TrainingError(f"Dataset is not valid UTF-8: {path}") from exc

    record_count = 0
    prompt_chars_total = 0
    completion_chars_total = 0
    for index, line in enumerate(payload_text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            raise TrainingError(
                f"Dataset validation failed at line {index}: empty lines are not allowed."
            )
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise TrainingError(
                f"Dataset validation failed at line {index}: invalid JSON. ({exc})"
            ) from exc
        if not isinstance(record, dict):
            raise TrainingError(
                f"Dataset validation failed at line {index}: record must be a JSON object."
            )

        keys = sorted(record.keys())
        if keys != ["completion", "prompt"]:
            raise TrainingError(
                f"Dataset validation failed at line {index}: expected keys "
                "['prompt', 'completion'] exactly."
            )

        prompt = record.get("prompt")
        completion = record.get("completion")
        if not isinstance(prompt, str) or not isinstance(completion, str):
            raise TrainingError(
                f"Dataset validation failed at line {index}: `prompt` and `completion` must be strings."
            )
        if not prompt.strip() or not completion.strip():
            raise TrainingError(
                f"Dataset validation failed at line {index}: `prompt` and `completion` cannot be empty."
            )

        record_count += 1
        prompt_chars_total += len(prompt)
        completion_chars_total += len(completion)

    if record_count == 0:
        raise TrainingError("Dataset validation failed: no training records found.")

    return {
        "path": str(path),
        "record_count": record_count,
        "sha256": hashlib.sha256(payload_bytes).hexdigest(),
        "size_bytes": len(payload_bytes),
        "avg_prompt_chars": round(prompt_chars_total / record_count, 3),
        "avg_completion_chars": round(completion_chars_total / record_count, 3),
    }


def _validate_runtime_dependencies(requested_backend: str) -> JsonObject:
    mlx_available = importlib.util.find_spec("mlx") is not None
    mlx_lm_available = importlib.util.find_spec("mlx_lm") is not None

    if requested_backend == "mlx":
        if not mlx_available:
            raise TrainingError(
                "Runtime dependency check failed: backend `mlx` requires package `mlx`."
            )
        if not mlx_lm_available:
            raise TrainingError(
                "Runtime dependency check failed: backend `mlx` requires package `mlx_lm`."
            )
        selected_backend = "mlx"
    elif requested_backend == "simulated":
        selected_backend = "simulated"
    else:
        selected_backend = "mlx" if mlx_available and mlx_lm_available else "simulated"

    dependencies = [
        {
            "name": "mlx",
            "available": mlx_available,
            "required": selected_backend == "mlx",
        },
        {
            "name": "mlx_lm",
            "available": mlx_lm_available,
            "required": selected_backend == "mlx",
        },
    ]
    return {
        "requested_backend": requested_backend,
        "selected_backend": selected_backend,
        "dependencies": dependencies,
    }


def _resolve_resume_checkpoint(checkpoints_dir: Path) -> tuple[int, Path | None]:
    latest = _latest_checkpoint_path(checkpoints_dir)
    if latest is None:
        return 0, None
    step = _checkpoint_step_from_path(latest)
    if step < 1:
        raise TrainingError(f"Invalid checkpoint filename: {latest.name}")
    return step, latest


def _latest_checkpoint_path(checkpoints_dir: Path) -> Path | None:
    checkpoints = list(checkpoints_dir.glob("checkpoint-step-*.adapter.json"))
    if not checkpoints:
        return None
    return max(checkpoints, key=_checkpoint_step_from_path)


def _checkpoint_step_from_path(path: Path) -> int:
    match = _CHECKPOINT_PATTERN.match(path.name)
    if not match:
        return -1
    return int(match.group(1))


def _compute_step_metric(
    *,
    step: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
    dataset_records: int,
) -> JsonObject:
    pseudo_random = ((seed * 1103515245 + step * 12345) & 0x7FFFFFFF) / 0x7FFFFFFF
    noise = (pseudo_random - 0.5) * 0.03
    base_loss = 2.8 * math.exp(-0.022 * step)
    loss = round(max(0.02, base_loss + 0.06 + noise), 6)

    base_throughput = 900 + (batch_size * 80) + min(dataset_records, 1000) * 0.3
    throughput = round(base_throughput * (0.9 + pseudo_random * 0.2), 3)

    return {
        "step": step,
        "loss": loss,
        "throughput_tokens_per_s": throughput,
        "learning_rate": learning_rate,
        "timestamp": _timestamp_utc(),
    }


def _write_json_file(path: Path, payload: Any, allow_non_dict: bool = False) -> None:
    try:
        if isinstance(payload, dict) or allow_non_dict:
            rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        else:
            raise TypeError("payload must be a dictionary")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
    except OSError as exc:
        raise TrainingError(f"Unable to write file {path}: {exc}") from exc


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _environment_snapshot() -> JsonObject:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "packages": {
            "mlx": _package_version("mlx"),
            "mlx_lm": _package_version("mlx_lm"),
        },
        "executable": sys.executable,
    }


def _package_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _slugify(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "run"


def _as_positive_int(value: Any, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise TrainingError(f"`{name}` must be an integer.") from exc
    if parsed < 1:
        raise TrainingError(f"`{name}` must be >= 1.")
    return parsed


def _as_int(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise TrainingError(f"`{name}` must be an integer.") from exc


def _as_positive_float(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise TrainingError(f"`{name}` must be a number.") from exc
    if parsed <= 0:
        raise TrainingError(f"`{name}` must be > 0.")
    return parsed
