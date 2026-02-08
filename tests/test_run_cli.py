"""Tests for run CLI commands."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mlx_lab.cli import main  # noqa: E402
from mlx_lab.training_lora import run_lora_training  # noqa: E402


def _write_dataset(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                json.dumps({"prompt": "Prompt 1", "completion": "Completion 1"}),
                json.dumps({"prompt": "Prompt 2", "completion": "Completion 2"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )


class RunCLITests(unittest.TestCase):
    def _run_cli(self, argv: list[str]) -> tuple[int, str, str]:
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(argv)
        return exit_code, stdout.getvalue(), stderr.getvalue()

    @patch("mlx_lab.cli.ensure_supported_platform", return_value=None)
    @patch("mlx_lab.training_lora.importlib.util.find_spec", return_value=None)
    def test_run_replay_dry_run_json(self, _find_spec_mock: object, _platform_mock: object) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dataset_path = temp_path / "cleaned.jsonl"
            _write_dataset(dataset_path)
            source = run_lora_training(
                model="org/base-3b",
                dataset_path=str(dataset_path),
                run_dir=str(temp_path / "run-a"),
                backend="simulated",
                max_steps=3,
                checkpoint_interval=2,
            )

            exit_code, stdout, stderr = self._run_cli(
                ["run", "replay", source["manifest_path"], "--dry-run", "--json"]
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(stderr, "")
            payload = json.loads(stdout)
            self.assertEqual(payload["mode"], "dry_run")
            self.assertEqual(payload["source_manifest"], source["manifest_path"])

    @patch("mlx_lab.cli.ensure_supported_platform", return_value=None)
    @patch("mlx_lab.training_lora.importlib.util.find_spec", return_value=None)
    def test_run_compare_json(self, _find_spec_mock: object, _platform_mock: object) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dataset_path = temp_path / "cleaned.jsonl"
            _write_dataset(dataset_path)
            run_a = run_lora_training(
                model="org/base-3b",
                dataset_path=str(dataset_path),
                run_dir=str(temp_path / "run-a"),
                backend="simulated",
                max_steps=3,
                checkpoint_interval=2,
                seed=5,
            )
            run_b = run_lora_training(
                model="org/base-3b",
                dataset_path=str(dataset_path),
                run_dir=str(temp_path / "run-b"),
                backend="simulated",
                max_steps=5,
                checkpoint_interval=2,
                seed=9,
            )

            exit_code, stdout, stderr = self._run_cli(
                ["run", "compare", run_a["manifest_path"], run_b["manifest_path"], "--json"]
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(stderr, "")
            payload = json.loads(stdout)
            self.assertEqual(payload["run_a"]["summary"]["total_steps"], 3)
            self.assertEqual(payload["run_b"]["summary"]["total_steps"], 5)
            self.assertEqual(payload["deltas"]["total_steps_delta"], 2.0)

    @patch("mlx_lab.cli.ensure_supported_platform", return_value=None)
    def test_run_compare_returns_non_zero_for_invalid_reference(self, _platform_mock: object) -> None:
        exit_code, stdout, stderr = self._run_cli(["run", "compare", "missing-a", "missing-b"])

        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("Unable to resolve run reference", stderr)


if __name__ == "__main__":
    unittest.main()

