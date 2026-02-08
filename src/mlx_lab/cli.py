"""Command-line interface for mlx-lab."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence

from ._version import __version__
from .commands.data import register_data_commands
from .commands.dataset import register_dataset_commands
from .commands.model import register_model_commands
from .commands.run import register_run_commands
from .commands.train import register_train_commands
from .runtime import UnsupportedPlatformError, ensure_supported_platform

Runner = Callable[[argparse.Namespace], int]


def _placeholder_runner(group_name: str) -> Runner:
    def _run(_args: argparse.Namespace) -> int:
        print(f"`{group_name}` workflows are scaffolded but not implemented yet.")
        return 0

    return _run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlx-lab",
        description="Local-first MLX-native toolkit for small-model fine-tuning.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    register_model_commands(subparsers)
    register_dataset_commands(subparsers)
    register_data_commands(subparsers)
    register_train_commands(subparsers)
    register_run_commands(subparsers)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    try:
        ensure_supported_platform()
    except UnsupportedPlatformError as exc:
        parser.exit(status=2, message=f"Error: {exc}\n")

    runner = getattr(args, "func", None)
    if runner is None:
        parser.print_help()
        return 0
    return runner(args)


if __name__ == "__main__":
    raise SystemExit(main())
