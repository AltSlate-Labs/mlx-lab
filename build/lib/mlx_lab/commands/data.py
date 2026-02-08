"""Data preparation command handlers."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from typing import Any

from ..data_cleaning import DataCleaningError, clean_dataset

Runner = Callable[[argparse.Namespace], int]


def register_data_commands(subparsers: Any) -> None:
    """Register `mlx-lab data ...` subcommands."""
    data_parser = subparsers.add_parser(
        "data",
        help="Dataset cleaning and reshaping workflows.",
        description="Dataset cleaning and reshaping workflows.",
    )
    data_subparsers = data_parser.add_subparsers(dest="data_command", metavar="DATA_COMMAND")
    data_parser.set_defaults(func=_help_runner(data_parser))

    clean_parser = data_subparsers.add_parser(
        "clean",
        help="Clean raw records into prompt/completion JSONL.",
        description="Clean raw records into prompt/completion JSONL.",
    )
    clean_parser.add_argument("--input", required=True, help="Path to raw input JSON or JSONL file.")
    clean_parser.add_argument("--output", required=True, help="Path to output JSONL file.")
    clean_parser.add_argument(
        "--input-format",
        choices=("auto", "jsonl", "json"),
        default="auto",
        help="Input parsing format.",
    )
    clean_parser.add_argument(
        "--map-prompt",
        "--prompt-field",
        dest="prompt_field",
        help="Field path for prompt extraction (for example: instruction or item.prompt).",
    )
    clean_parser.add_argument(
        "--map-completion",
        "--completion-field",
        dest="completion_field",
        help="Field path for completion extraction (for example: answer or item.response).",
    )
    clean_parser.add_argument(
        "--prompt-template",
        help="Template for prompt extraction (for example: 'Q: {question}').",
    )
    clean_parser.add_argument(
        "--completion-template",
        help="Template for completion extraction (for example: '{answer}').",
    )
    clean_parser.add_argument(
        "--source-dataset-id",
        help="Source dataset identifier for manifest provenance.",
    )
    clean_parser.add_argument(
        "--source-dataset-version",
        help="Source dataset version or revision for manifest provenance.",
    )
    clean_parser.add_argument(
        "--manifest",
        dest="manifest_path",
        help="Path for output manifest JSON file.",
    )
    clean_parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Drop duplicate prompt/completion pairs while preserving first occurrence order.",
    )
    clean_parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Keep records where prompt or completion is empty after trimming.",
    )
    clean_parser.add_argument(
        "--max-prompt-chars",
        type=_positive_int,
        help="Drop records where prompt exceeds this character limit.",
    )
    clean_parser.add_argument(
        "--max-completion-chars",
        type=_positive_int,
        help="Drop records where completion exceeds this character limit.",
    )
    clean_parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Output machine-readable summary JSON.",
    )
    clean_parser.set_defaults(func=run_data_clean)


def run_data_clean(args: argparse.Namespace) -> int:
    try:
        payload = clean_dataset(
            input_path=args.input,
            output_path=args.output,
            prompt_field=args.prompt_field,
            completion_field=args.completion_field,
            prompt_template=args.prompt_template,
            completion_template=args.completion_template,
            source_dataset_id=args.source_dataset_id,
            source_dataset_version=args.source_dataset_version,
            manifest_path=args.manifest_path,
            input_format=args.input_format,
            dedupe=args.dedupe,
            drop_empty=not args.keep_empty,
            max_prompt_chars=args.max_prompt_chars,
            max_completion_chars=args.max_completion_chars,
        )
    except DataCleaningError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    _render_clean_text(payload)
    return 0


def _help_runner(parser: argparse.ArgumentParser) -> Runner:
    def _run(_args: argparse.Namespace) -> int:
        parser.print_help()
        return 0

    return _run


def _positive_int(raw_value: str) -> int:
    try:
        parsed = int(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def _render_clean_text(payload: dict[str, Any]) -> None:
    print("Data cleaning complete.")
    print(f"Input: {payload['input_path']} ({payload['input_format']})")
    print(f"Output: {payload['output_path']}")
    print(f"Manifest: {payload['manifest_path']}")
    print(f"Total read: {payload['total_read']}")
    print(f"Written: {payload['written']}")
    print(f"Dropped: {payload['dropped']}")
    print(f"Transform config SHA256: {payload['transform_config_sha256']}")
    print(f"Output SHA256: {payload['output_sha256']}")

    print("Drop reasons:")
    drop_reasons = payload.get("drop_reasons", {})
    if not drop_reasons:
        print("- none")
        return

    for reason, count in drop_reasons.items():
        print(f"- {reason}: {count}")

