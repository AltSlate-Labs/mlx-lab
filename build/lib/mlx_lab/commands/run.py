"""Run management command handlers."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from typing import Any

from ..run_ops import RunOpsError, compare_runs, replay_run

Runner = Callable[[argparse.Namespace], int]


def register_run_commands(subparsers: Any) -> None:
    """Register `mlx-lab run ...` subcommands."""
    run_parser = subparsers.add_parser(
        "run",
        help="Run and experiment management workflows.",
        description="Run and experiment management workflows.",
    )
    run_subparsers = run_parser.add_subparsers(dest="run_command", metavar="RUN_COMMAND")
    run_parser.set_defaults(func=_help_runner(run_parser))

    replay_parser = run_subparsers.add_parser(
        "replay",
        help="Replay a prior run from its manifest.",
        description="Replay a prior run from its manifest.",
    )
    replay_parser.add_argument("run_ref", help="Run directory, run id, or manifest path.")
    replay_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show replay parameters without launching training.",
    )
    replay_parser.add_argument("--run-dir", help="Explicit run directory for replay output.")
    replay_parser.add_argument("--run-name", help="Run name used when creating replay output.")
    replay_parser.add_argument(
        "--backend",
        choices=("auto", "mlx", "simulated"),
        help="Override replay backend.",
    )
    replay_parser.add_argument("--max-steps", type=_positive_int, help="Override max steps.")
    replay_parser.add_argument(
        "--checkpoint-interval",
        type=_positive_int,
        help="Override checkpoint interval.",
    )
    replay_parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Output machine-readable JSON.",
    )
    replay_parser.set_defaults(func=run_run_replay)

    compare_parser = run_subparsers.add_parser(
        "compare",
        help="Compare two run manifests.",
        description="Compare two run manifests.",
    )
    compare_parser.add_argument("run_a", help="First run reference.")
    compare_parser.add_argument("run_b", help="Second run reference.")
    compare_parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Output machine-readable JSON.",
    )
    compare_parser.set_defaults(func=run_run_compare)


def run_run_replay(args: argparse.Namespace) -> int:
    try:
        payload = replay_run(
            args.run_ref,
            execute=not args.dry_run,
            run_dir=args.run_dir,
            run_name=args.run_name,
            backend=args.backend,
            max_steps=args.max_steps,
            checkpoint_interval=args.checkpoint_interval,
        )
    except RunOpsError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    _render_replay_text(payload)
    return 0


def run_run_compare(args: argparse.Namespace) -> int:
    try:
        payload = compare_runs(args.run_a, args.run_b)
    except RunOpsError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    _render_compare_text(payload)
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


def _render_replay_text(payload: dict[str, Any]) -> None:
    if payload.get("mode") == "dry_run":
        print("Replay dry-run summary:")
        print(f"Source manifest: {payload['source_manifest']}")
        print("Replay parameters:")
        for key, value in payload["replay_parameters"].items():
            print(f"- {key}: {value}")
        return

    result = payload["replay_result"]
    print("Replay executed successfully.")
    print(f"Source manifest: {payload['source_manifest']}")
    print(f"Source run id: {payload['source_run_id']}")
    print(f"Replay run dir: {result['run_dir']}")
    print(f"Replay latest step: {result['latest_step']}")
    print(f"Replay latest checkpoint: {result['latest_checkpoint'] or 'none'}")
    print(f"Replay manifest: {result['manifest_path']}")


def _render_compare_text(payload: dict[str, Any]) -> None:
    run_a = payload["run_a"]
    run_b = payload["run_b"]
    deltas = payload["deltas"]

    print("Run comparison summary:")
    print(f"- run_a: {run_a['run_id']} ({run_a['manifest_path']})")
    print(f"- run_b: {run_b['run_id']} ({run_b['manifest_path']})")
    print("Metric deltas (run_b - run_a):")
    for key, value in deltas.items():
        print(f"- {key}: {value}")

    print("Configuration differences:")
    differences = payload["config_differences"]
    if not differences:
        print("- none")
        return
    for diff in differences:
        print(f"- {diff['key']}: run_a={diff['run_a']} run_b={diff['run_b']}")

