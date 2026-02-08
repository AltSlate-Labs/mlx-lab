"""Training command handlers."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from typing import Any

from ..training_lora import TrainingError, run_lora_training

Runner = Callable[[argparse.Namespace], int]


def register_train_commands(subparsers: Any) -> None:
    """Register `mlx-lab train ...` subcommands."""
    train_parser = subparsers.add_parser(
        "train",
        help="Fine-tuning workflows.",
        description="Fine-tuning workflows.",
    )
    train_subparsers = train_parser.add_subparsers(dest="train_command", metavar="TRAIN_COMMAND")
    train_parser.set_defaults(func=_help_runner(train_parser))

    lora_parser = train_subparsers.add_parser(
        "lora",
        help="Run LoRA fine-tuning workflow.",
        description="Run LoRA fine-tuning workflow.",
    )
    lora_parser.add_argument("--model", help="Model id or local model path.")
    lora_parser.add_argument("--dataset", dest="dataset_path", help="Path to cleaned JSONL dataset.")
    lora_parser.add_argument("--config", dest="config_path", help="Path to training config JSON file.")
    lora_parser.add_argument("--output-dir", help="Base directory for new run outputs.")
    lora_parser.add_argument("--run-dir", help="Explicit run directory for training artifacts.")
    lora_parser.add_argument("--run-name", help="Run name used when creating run directories.")
    lora_parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in run dir.")
    lora_parser.add_argument(
        "--backend",
        choices=("auto", "mlx", "simulated"),
        help="Training backend selection.",
    )
    lora_parser.add_argument("--max-steps", type=_positive_int, help="Maximum training steps.")
    lora_parser.add_argument(
        "--checkpoint-interval",
        type=_positive_int,
        help="Checkpoint write interval in steps.",
    )
    lora_parser.add_argument(
        "--learning-rate",
        type=_positive_float,
        help="Learning rate.",
    )
    lora_parser.add_argument("--batch-size", type=_positive_int, help="Batch size.")
    lora_parser.add_argument("--lora-rank", type=_positive_int, help="LoRA rank.")
    lora_parser.add_argument("--seed", type=int, help="Training seed.")
    lora_parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Output machine-readable JSON summary.",
    )
    lora_parser.set_defaults(func=run_train_lora)


def run_train_lora(args: argparse.Namespace) -> int:
    try:
        payload = run_lora_training(
            model=args.model,
            dataset_path=args.dataset_path,
            config_path=args.config_path,
            output_dir=args.output_dir,
            run_dir=args.run_dir,
            run_name=args.run_name,
            resume=args.resume,
            backend=args.backend,
            max_steps=args.max_steps,
            checkpoint_interval=args.checkpoint_interval,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            lora_rank=args.lora_rank,
            seed=args.seed,
        )
    except TrainingError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    _render_train_text(payload)
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


def _positive_float(raw_value: str) -> float:
    try:
        parsed = float(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def _render_train_text(payload: dict[str, Any]) -> None:
    runtime = payload["preflight"]["runtime"]
    print("LoRA training run complete.")
    print(f"Run directory: {payload['run_dir']}")
    print(f"Model: {payload['model']}")
    print(f"Dataset: {payload['dataset_path']}")
    print(f"Backend: {runtime['selected_backend']} (requested={runtime['requested_backend']})")
    print(f"Resumed from step: {payload['resumed_from_step']}")
    print(f"Latest step: {payload['latest_step']}")
    print(f"Latest checkpoint: {payload['latest_checkpoint'] or 'none'}")
    print(f"Metrics log: {payload['metrics_path']}")
    print(f"Run manifest: {payload['manifest_path']}")
    print(f"Resolved config: {payload['resolved_config_path']}")
    print(f"Preflight report: {payload['preflight_path']}")
