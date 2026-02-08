"""Tests for data cleaning CLI command wiring and behavior."""

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


class DataCLITests(unittest.TestCase):
    def _run_cli(self, argv: list[str]) -> tuple[int, str, str]:
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(argv)
        return exit_code, stdout.getvalue(), stderr.getvalue()

    @patch("mlx_lab.cli.ensure_supported_platform", return_value=None)
    def test_data_clean_json_summary(self, _platform_mock: object) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "raw.jsonl"
            output_path = temp_path / "cleaned.jsonl"
            input_path.write_text(
                "\n".join(
                    [
                        json.dumps({"instruction": "Q1", "answer": "A1"}),
                        json.dumps({"instruction": "Q2", "answer": "A2"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            exit_code, stdout, stderr = self._run_cli(
                [
                    "data",
                    "clean",
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--map-prompt",
                    "instruction",
                    "--map-completion",
                    "answer",
                    "--source-dataset-id",
                    "org/sample",
                    "--source-dataset-version",
                    "v1",
                    "--json",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(stderr, "")
            payload = json.loads(stdout)
            self.assertEqual(payload["written"], 2)
            self.assertEqual(payload["dropped"], 0)
            self.assertEqual(payload["source_dataset_id"], "org/sample")
            self.assertTrue(Path(payload["manifest_path"]).exists())
            self.assertTrue(output_path.exists())

    @patch("mlx_lab.cli.ensure_supported_platform", return_value=None)
    def test_data_clean_returns_non_zero_for_invalid_config(self, _platform_mock: object) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "raw.jsonl"
            output_path = temp_path / "cleaned.jsonl"
            input_path.write_text(json.dumps({"prompt": "p", "completion": "c"}) + "\n", encoding="utf-8")

            exit_code, stdout, stderr = self._run_cli(
                [
                    "data",
                    "clean",
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--map-prompt",
                    "prompt",
                    "--prompt-template",
                    "{prompt}",
                    "--map-completion",
                    "completion",
                ]
            )

            self.assertEqual(exit_code, 1)
            self.assertEqual(stdout, "")
            self.assertIn("Error: Configure prompt extraction", stderr)


if __name__ == "__main__":
    unittest.main()

