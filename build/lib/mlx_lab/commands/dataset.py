"""Dataset command handlers."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from typing import Any

from ..hf_datasets import APIRequestError, HFDatasetClient

Runner = Callable[[argparse.Namespace], int]


def register_dataset_commands(subparsers: Any) -> None:
    """Register `mlx-lab dataset ...` subcommands."""
    dataset_parser = subparsers.add_parser(
        "dataset",
        help="Dataset discovery and inspection workflows.",
        description="Dataset discovery and inspection workflows.",
    )
    dataset_subparsers = dataset_parser.add_subparsers(
        dest="dataset_command",
        metavar="DATASET_COMMAND",
    )
    dataset_parser.set_defaults(func=_help_runner(dataset_parser))

    search_parser = dataset_subparsers.add_parser(
        "search",
        help="Search Hugging Face datasets.",
        description="Search Hugging Face datasets.",
    )
    search_parser.add_argument("query", help="Search query.")
    search_parser.add_argument(
        "--page",
        type=_positive_int,
        default=1,
        help="Page number (1-indexed).",
    )
    search_parser.add_argument(
        "--limit",
        type=_positive_int,
        default=20,
        help="Number of items per page.",
    )
    search_parser.add_argument(
        "--language",
        help="Filter by language metadata.",
    )
    search_parser.add_argument(
        "--task",
        help="Filter by task metadata.",
    )
    search_parser.add_argument(
        "--license",
        dest="license_name",
        help="Filter by license metadata.",
    )
    search_parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Output machine-readable JSON.",
    )
    search_parser.set_defaults(func=run_dataset_search)

    inspect_parser = dataset_subparsers.add_parser(
        "inspect",
        help="Inspect one dataset by id.",
        description="Inspect one dataset by id.",
    )
    inspect_parser.add_argument("dataset_id", help="Dataset id (for example, owner/name).")
    inspect_parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Output machine-readable JSON.",
    )
    inspect_parser.set_defaults(func=run_dataset_inspect)


def run_dataset_search(args: argparse.Namespace) -> int:
    client = HFDatasetClient()
    try:
        payload = client.search_datasets(
            args.query,
            page=args.page,
            limit=args.limit,
            language=args.language,
            task=args.task,
            license_name=args.license_name,
        )
    except APIRequestError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    _render_search_text(payload)
    return 0


def run_dataset_inspect(args: argparse.Namespace) -> int:
    client = HFDatasetClient()
    try:
        payload = client.inspect_dataset(args.dataset_id)
    except APIRequestError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    _render_inspect_text(payload)
    warnings = payload.get("warnings", [])
    if isinstance(warnings, list):
        for warning in warnings:
            print(f"Warning: {warning}", file=sys.stderr)
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


def _render_search_text(payload: dict[str, Any]) -> None:
    print(
        "Dataset search results "
        f"(page={payload['page']}, limit={payload['limit']}, returned={payload['total_returned']}):"
    )
    results = payload.get("results", [])
    if not results:
        print("No datasets matched the requested query and filters.")
        return

    for item in results:
        dataset_id = item.get("id", "")
        languages = ", ".join(item.get("languages") or []) or "unknown"
        tasks = ", ".join(item.get("task_tags") or []) or "unknown"
        license_name = item.get("license") or "unknown"
        downloads = item.get("downloads")
        likes = item.get("likes")
        summary = item.get("summary") or ""
        print(f"- {dataset_id}")
        print(f"  language: {languages}")
        print(f"  task tags: {tasks}")
        print(f"  license: {license_name}")
        print(f"  downloads: {downloads if downloads is not None else 'unknown'}")
        print(f"  likes: {likes if likes is not None else 'unknown'}")
        if summary:
            print(f"  summary: {summary}")


def _render_inspect_text(payload: dict[str, Any]) -> None:
    print(f"Dataset: {payload.get('dataset_id', '')}")
    print(f"Summary: {payload.get('summary') or 'n/a'}")
    print(f"License: {payload.get('license') or 'n/a'}")
    print(f"Languages: {', '.join(payload.get('languages') or []) or 'n/a'}")
    print(f"Task tags: {', '.join(payload.get('task_tags') or []) or 'n/a'}")
    print(f"Downloads: {payload.get('downloads') if payload.get('downloads') is not None else 'n/a'}")
    print(f"Likes: {payload.get('likes') if payload.get('likes') is not None else 'n/a'}")
    print(f"Last modified: {payload.get('last_modified') or 'n/a'}")
    print("Splits:")
    splits = payload.get("splits") or []
    if not splits:
        print("- unavailable")
    else:
        for split in splits:
            name = split.get("name") or "unknown"
            num_rows = split.get("num_rows")
            formatted_rows = num_rows if num_rows is not None else "unknown"
            print(f"- {name}: {formatted_rows} rows")
    print("Feature schema:")
    feature_schema = payload.get("feature_schema")
    if feature_schema is None:
        print("unavailable")
    else:
        print(json.dumps(feature_schema, indent=2, sort_keys=True))

