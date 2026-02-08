"""Tests for train CLI command behavior."""

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


class TrainCLITests(unittest.TestCase):
    def _run_cli(self, argv: list[str]) -> tuple[int, str, str]:
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(argv)
        return exit_code, stdout.getvalue(), stderr.getvalue()

    @patch("mlx_lab.cli.ensure_supported_platform", return_value=None)
    @patch("mlx_lab.training_lora.importlib.util.find_spec", return_value=None)
    def test_train_lora_json_output(self, _find_spec_mock: object, _platform_mock: object) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dataset_path = temp_path / "cleaned.jsonl"
            run_dir = temp_path / "run-a"
            dataset_path.write_text(
                "\n".join(
                    [
                        json.dumps({"prompt": "Prompt 1", "completion": "Completion 1"}),
                        json.dumps({"prompt": "Prompt 2", "completion": "Completion 2"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            exit_code, stdout, stderr = self._run_cli(
                [
                    "train",
                    "lora",
                    "--model",
                    "org/base-3b",
                    "--dataset",
                    str(dataset_path),
                    "--run-dir",
                    str(run_dir),
                    "--backend",
                    "simulated",
                    "--max-steps",
                    "3",
                    "--checkpoint-interval",
                    "2",
                    "--json",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(stderr, "")
            payload = json.loads(stdout)
            self.assertEqual(payload["latest_step"], 3)
            self.assertEqual(payload["resumed_from_step"], 0)
            self.assertEqual(payload["selected_backend"], "simulated")
            self.assertTrue(Path(payload["metrics_path"]).exists())

    @patch("mlx_lab.cli.ensure_supported_platform", return_value=None)
    @patch("mlx_lab.training_lora.importlib.util.find_spec", return_value=None)
    def test_train_lora_returns_non_zero_for_validation_error(
        self,
        _find_spec_mock: object,
        _platform_mock: object,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dataset_path = temp_path / "bad.jsonl"
            dataset_path.write_text(json.dumps({"prompt": "Missing completion"}) + "\n", encoding="utf-8")

            exit_code, stdout, stderr = self._run_cli(
                [
                    "train",
                    "lora",
                    "--model",
                    "org/base-3b",
                    "--dataset",
                    str(dataset_path),
                    "--run-dir",
                    str(temp_path / "run-b"),
                    "--backend",
                    "simulated",
                ]
            )

            self.assertEqual(exit_code, 1)
            self.assertEqual(stdout, "")
            self.assertIn("Dataset validation failed", stderr)


if __name__ == "__main__":
    unittest.main()

