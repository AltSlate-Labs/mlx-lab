"""Deterministic cleaning utilities for instruction-style JSONL datasets."""

from __future__ import annotations

import hashlib
import json
import string
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

JsonObject = dict[str, Any]
_FORMATTER = string.Formatter()


class DataCleaningError(RuntimeError):
    """Raised when dataset cleaning configuration or execution fails."""


def clean_dataset(
    *,
    input_path: str,
    output_path: str,
    prompt_field: str | None = None,
    completion_field: str | None = None,
    prompt_template: str | None = None,
    completion_template: str | None = None,
    source_dataset_id: str | None = None,
    source_dataset_version: str | None = None,
    manifest_path: str | None = None,
    input_format: str = "auto",
    dedupe: bool = False,
    drop_empty: bool = True,
    max_prompt_chars: int | None = None,
    max_completion_chars: int | None = None,
) -> JsonObject:
    """Clean raw input records into deterministic prompt/completion JSONL output."""
    _validate_config(
        prompt_field=prompt_field,
        completion_field=completion_field,
        prompt_template=prompt_template,
        completion_template=completion_template,
        input_format=input_format,
        max_prompt_chars=max_prompt_chars,
        max_completion_chars=max_completion_chars,
    )

    source_path = Path(input_path)
    destination_path = Path(output_path)
    manifest_destination = Path(manifest_path) if manifest_path else Path(f"{output_path}.manifest.json")

    if not source_path.exists() or not source_path.is_file():
        raise DataCleaningError(f"Input file does not exist: {source_path}")

    resolved_input_format = _resolve_input_format(source_path, input_format)
    records = _iter_input_records(source_path, resolved_input_format)

    transform_config: JsonObject = {
        "completion_field": completion_field,
        "completion_template": completion_template,
        "dedupe": dedupe,
        "drop_empty": drop_empty,
        "max_completion_chars": max_completion_chars,
        "max_prompt_chars": max_prompt_chars,
        "prompt_field": prompt_field,
        "prompt_template": prompt_template,
    }
    transform_hash = _sha256_text(_stable_json(transform_config))

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_destination.parent.mkdir(parents=True, exist_ok=True)

    drop_reasons: Counter[str] = Counter()
    total_read = 0
    written = 0
    seen_records: set[tuple[str, str]] = set()
    output_hasher = hashlib.sha256()

    try:
        with destination_path.open("w", encoding="utf-8", newline="\n") as output_file:
            for record in records:
                total_read += 1
                cleaned, reason = _clean_record(
                    record,
                    prompt_field=prompt_field,
                    completion_field=completion_field,
                    prompt_template=prompt_template,
                    completion_template=completion_template,
                    drop_empty=drop_empty,
                    max_prompt_chars=max_prompt_chars,
                    max_completion_chars=max_completion_chars,
                )
                if cleaned is None:
                    drop_reasons[reason or "unknown_error"] += 1
                    continue

                signature = (cleaned["prompt"], cleaned["completion"])
                if dedupe and signature in seen_records:
                    drop_reasons["duplicate_record"] += 1
                    continue
                if dedupe:
                    seen_records.add(signature)

                line = json.dumps(
                    cleaned,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=False,
                )
                encoded = f"{line}\n".encode("utf-8")
                output_file.write(line)
                output_file.write("\n")
                output_hasher.update(encoded)
                written += 1
    except UnicodeDecodeError as exc:
        raise DataCleaningError(
            f"Input file contains invalid UTF-8 content: {source_path}"
        ) from exc
    except OSError as exc:
        raise DataCleaningError(
            f"Unable to write cleaned dataset output to {destination_path}: {exc}"
        ) from exc

    dropped = total_read - written
    normalized_drop_reasons = dict(sorted(drop_reasons.items()))
    output_sha256 = output_hasher.hexdigest()
    input_sha256 = _sha256_bytes(source_path.read_bytes())

    manifest_payload: JsonObject = {
        "schema_version": 1,
        "source": {
            "dataset_id": source_dataset_id,
            "dataset_version": source_dataset_version,
            "input_format": resolved_input_format,
            "input_path": str(source_path),
            "input_sha256": input_sha256,
        },
        "transform": {
            **transform_config,
            "config_sha256": transform_hash,
            "ordering": "input_order",
        },
        "stats": {
            "total_read": total_read,
            "written": written,
            "dropped": dropped,
            "drop_reasons": normalized_drop_reasons,
        },
        "output": {
            "path": str(destination_path),
            "encoding": "utf-8",
            "record_count": written,
            "sha256": output_sha256,
        },
    }
    _write_json_file(manifest_destination, manifest_payload)

    return {
        "input_path": str(source_path),
        "output_path": str(destination_path),
        "manifest_path": str(manifest_destination),
        "input_format": resolved_input_format,
        "total_read": total_read,
        "written": written,
        "dropped": dropped,
        "drop_reasons": normalized_drop_reasons,
        "source_dataset_id": source_dataset_id,
        "source_dataset_version": source_dataset_version,
        "transform_config_sha256": transform_hash,
        "output_sha256": output_sha256,
    }


def _validate_config(
    *,
    prompt_field: str | None,
    completion_field: str | None,
    prompt_template: str | None,
    completion_template: str | None,
    input_format: str,
    max_prompt_chars: int | None,
    max_completion_chars: int | None,
) -> None:
    if bool(prompt_field) == bool(prompt_template):
        raise DataCleaningError(
            "Configure prompt extraction with exactly one of --map-prompt or --prompt-template."
        )
    if bool(completion_field) == bool(completion_template):
        raise DataCleaningError(
            "Configure completion extraction with exactly one of "
            "--map-completion or --completion-template."
        )

    valid_input_formats = {"auto", "jsonl", "json"}
    if input_format not in valid_input_formats:
        raise DataCleaningError(
            f"Unsupported input format '{input_format}'. Valid options: auto, jsonl, json."
        )

    if max_prompt_chars is not None and max_prompt_chars < 1:
        raise DataCleaningError("--max-prompt-chars must be >= 1 when provided.")
    if max_completion_chars is not None and max_completion_chars < 1:
        raise DataCleaningError("--max-completion-chars must be >= 1 when provided.")


def _resolve_input_format(source_path: Path, input_format: str) -> str:
    if input_format != "auto":
        return input_format
    if source_path.suffix.lower() == ".json":
        return "json"
    return "jsonl"


def _iter_input_records(source_path: Path, input_format: str) -> Iterator[JsonObject]:
    if input_format == "json":
        yield from _iter_json_records(source_path)
        return
    if input_format == "jsonl":
        yield from _iter_jsonl_records(source_path)
        return
    raise DataCleaningError(f"Unsupported input format: {input_format}")


def _iter_json_records(source_path: Path) -> Iterator[JsonObject]:
    try:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        raise DataCleaningError(f"Input file contains invalid UTF-8 content: {source_path}") from exc
    except json.JSONDecodeError as exc:
        raise DataCleaningError(f"Input JSON file is invalid: {source_path} ({exc})") from exc

    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        records = payload.get("records")
        data = payload.get("data")
        if isinstance(records, list):
            items = records
        elif isinstance(data, list):
            items = data
        else:
            items = [payload]
    else:
        raise DataCleaningError("Input JSON payload must be an object or list of objects.")

    for item in items:
        if isinstance(item, dict):
            yield item
        else:
            yield {"__invalid_record__": item, "__reason__": "non_object_record"}


def _iter_jsonl_records(source_path: Path) -> Iterator[JsonObject]:
    try:
        with source_path.open("r", encoding="utf-8") as input_file:
            for line in input_file:
                stripped = line.strip()
                if not stripped:
                    yield {"__reason__": "empty_line"}
                    continue
                try:
                    parsed = json.loads(stripped)
                except json.JSONDecodeError:
                    yield {"__reason__": "invalid_json"}
                    continue
                if not isinstance(parsed, dict):
                    yield {"__invalid_record__": parsed, "__reason__": "non_object_record"}
                    continue
                yield parsed
    except UnicodeDecodeError as exc:
        raise DataCleaningError(f"Input file contains invalid UTF-8 content: {source_path}") from exc


def _clean_record(
    record: JsonObject,
    *,
    prompt_field: str | None,
    completion_field: str | None,
    prompt_template: str | None,
    completion_template: str | None,
    drop_empty: bool,
    max_prompt_chars: int | None,
    max_completion_chars: int | None,
) -> tuple[JsonObject | None, str | None]:
    precomputed_reason = record.get("__reason__")
    if isinstance(precomputed_reason, str):
        return None, precomputed_reason

    prompt_value, prompt_reason = _extract_value(
        record,
        field=prompt_field,
        template=prompt_template,
        target_key="prompt",
    )
    if prompt_value is None:
        return None, prompt_reason

    completion_value, completion_reason = _extract_value(
        record,
        field=completion_field,
        template=completion_template,
        target_key="completion",
    )
    if completion_value is None:
        return None, completion_reason

    if drop_empty and prompt_value == "":
        return None, "empty_prompt"
    if drop_empty and completion_value == "":
        return None, "empty_completion"

    if max_prompt_chars is not None and len(prompt_value) > max_prompt_chars:
        return None, "prompt_too_long"
    if max_completion_chars is not None and len(completion_value) > max_completion_chars:
        return None, "completion_too_long"

    return {
        "prompt": prompt_value,
        "completion": completion_value,
    }, None


def _extract_value(
    record: JsonObject,
    *,
    field: str | None,
    template: str | None,
    target_key: str,
) -> tuple[str | None, str | None]:
    if field:
        try:
            value = _lookup_field(record, field)
        except KeyError:
            return None, f"missing_{target_key}_field"
    else:
        assert template is not None
        try:
            value = _render_template(template, record)
        except KeyError:
            return None, f"missing_{target_key}_template_key"

    if value is None:
        return None, f"null_{target_key}"
    if isinstance(value, str):
        return value.strip(), None
    return str(value).strip(), None


def _lookup_field(record: JsonObject, field_path: str) -> Any:
    current: Any = record
    for segment in field_path.split("."):
        if isinstance(current, dict) and segment in current:
            current = current[segment]
            continue
        raise KeyError(field_path)
    return current


def _render_template(template: str, record: JsonObject) -> str:
    chunks: list[str] = []
    for literal_text, field_name, format_spec, conversion in _FORMATTER.parse(template):
        chunks.append(literal_text)
        if field_name is None:
            continue

        value = _lookup_field(record, field_name)
        if conversion == "r":
            value = repr(value)
        elif conversion == "a":
            value = ascii(value)
        else:
            value = str(value)

        if format_spec:
            value = format(value, format_spec)
        chunks.append(str(value))
    return "".join(chunks).strip()


def _stable_json(payload: JsonObject) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json_file(path: Path, payload: JsonObject) -> None:
    try:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise DataCleaningError(f"Unable to write manifest file to {path}: {exc}") from exc

