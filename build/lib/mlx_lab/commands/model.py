"""Model discovery command handlers."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from typing import Any

from ..hf_models import APIRequestError, HFModelClient

Runner = Callable[[argparse.Namespace], int]
SIZE_CLASS_CHOICES = ("any", "small", "medium", "large", "xlarge")


def register_model_commands(subparsers: Any) -> None:
    """Register `mlx-lab model ...` subcommands."""
    model_parser = subparsers.add_parser(
        "model",
        help="Model discovery workflows.",
        description="Model discovery workflows.",
    )
    model_subparsers = model_parser.add_subparsers(dest="model_command", metavar="MODEL_COMMAND")
    model_parser.set_defaults(func=_help_runner(model_parser))

    search_parser = model_subparsers.add_parser(
        "search",
        help="Search Hugging Face models.",
        description="Search Hugging Face models.",
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
        help="Number of results per page.",
    )
    search_parser.add_argument(
        "--size-class",
        choices=SIZE_CLASS_CHOICES,
        default="any",
        help="Filter by parameter size class.",
    )
    search_parser.add_argument(
        "--tag",
        dest="tags",
        action="append",
        default=[],
        help="Filter by required model tag (can be repeated).",
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
    search_parser.set_defaults(func=run_model_search)

    inspect_parser = model_subparsers.add_parser(
        "inspect",
        help="Inspect one model by id.",
        description="Inspect one model by id.",
    )
    inspect_parser.add_argument("model_id", help="Model id (for example, owner/name).")
    inspect_parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Output machine-readable JSON.",
    )
    inspect_parser.set_defaults(func=run_model_inspect)


def run_model_search(args: argparse.Namespace) -> int:
    client = HFModelClient()
    size_class = None if args.size_class == "any" else args.size_class
    try:
        payload = client.search_models(
            args.query,
            page=args.page,
            limit=args.limit,
            size_class=size_class,
            tags=args.tags,
            license_name=args.license_name,
        )
    except APIRequestError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    _render_search_text(payload)
    return 0


def run_model_inspect(args: argparse.Namespace) -> int:
    client = HFModelClient()
    try:
        payload = client.inspect_model(args.model_id)
    except APIRequestError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    _render_inspect_text(payload)
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


def _format_param_count(value: int | None) -> str:
    if value is None:
        return "unknown"
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    return str(value)


def _render_search_text(payload: dict[str, Any]) -> None:
    print(
        "Model search results "
        f"(page={payload['page']}, limit={payload['limit']}, returned={payload['total_returned']}):"
    )
    results = payload.get("results", [])
    if not results:
        print("No models matched the requested query and filters.")
        return

    for item in results:
        compatibility = item.get("compatibility", {})
        status = compatibility.get("status", "unknown")
        print(f"- {item.get('id', '')}")
        print(f"  compatibility: {status}")
        print(f"  params: {_format_param_count(item.get('parameter_count'))}")
        print(f"  size class: {item.get('parameter_size_class') or 'unknown'}")
        print(f"  architecture: {item.get('architecture') or 'unknown'}")
        print(f"  license: {item.get('license') or 'unknown'}")
        print(f"  task: {item.get('task_tag') or 'unknown'}")
        print(f"  downloads: {item.get('downloads') if item.get('downloads') is not None else 'unknown'}")
        print(f"  likes: {item.get('likes') if item.get('likes') is not None else 'unknown'}")
        summary = item.get("summary")
        if summary:
            print(f"  summary: {summary}")


def _render_inspect_text(payload: dict[str, Any]) -> None:
    compatibility = payload.get("compatibility", {})
    tokenizer = payload.get("tokenizer", {})
    print(f"Model: {payload.get('model_id', '')}")
    print(f"Summary: {payload.get('summary') or 'n/a'}")
    print(f"Compatibility: {compatibility.get('status', 'unknown')}")
    print(f"Compatibility reason: {compatibility.get('reason', 'n/a')}")
    print(f"Parameter count: {_format_param_count(payload.get('parameter_count'))}")
    print(f"Parameter size class: {payload.get('parameter_size_class') or 'unknown'}")
    print(f"Architecture: {payload.get('architecture') or 'n/a'}")
    print(f"License: {payload.get('license') or 'n/a'}")
    print(f"Task: {payload.get('task_tag') or 'n/a'}")
    print(f"Downloads: {payload.get('downloads') if payload.get('downloads') is not None else 'n/a'}")
    print(f"Likes: {payload.get('likes') if payload.get('likes') is not None else 'n/a'}")
    print(f"Last modified: {payload.get('last_modified') or 'n/a'}")
    print("Tokenizer:")
    print(f"- class: {tokenizer.get('class') or 'n/a'}")
    print(
        f"- vocab_size: {tokenizer.get('vocab_size') if tokenizer.get('vocab_size') is not None else 'n/a'}"
    )
    print(
        "- model_max_length: "
        f"{tokenizer.get('model_max_length') if tokenizer.get('model_max_length') is not None else 'n/a'}"
    )

