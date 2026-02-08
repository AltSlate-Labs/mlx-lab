"""Tests for model CLI command wiring and output modes."""

from __future__ import annotations

import json
import sys
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
from mlx_lab.hf_models import APIRequestError  # noqa: E402


class ModelCLITests(unittest.TestCase):
    def _run_cli(self, argv: list[str]) -> tuple[int, str, str]:
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(argv)
        return exit_code, stdout.getvalue(), stderr.getvalue()

    @patch("mlx_lab.commands.model.HFModelClient")
    @patch("mlx_lab.cli.ensure_supported_platform", return_value=None)
    def test_model_search_json_output(self, _platform_mock: object, client_cls: object) -> None:
        fake_payload = {
            "query": "llama",
            "page": 1,
            "limit": 10,
            "offset": 0,
            "source_count": 2,
            "total_returned": 1,
            "filters": {
                "size_class": "medium",
                "license": "apache",
                "tags": ["text-generation"],
            },
            "results": [
                {
                    "id": "org/base-7b",
                    "summary": "General model",
                    "license": "apache-2.0",
                    "tags": ["text-generation", "llama", "7b"],
                    "task_tag": "text-generation",
                    "architecture": "llama",
                    "parameter_count": 7_000_000_000,
                    "parameter_size_class": "medium",
                    "compatibility": {"status": "convertible", "reason": "test"},
                    "downloads": 10,
                    "likes": 2,
                    "last_modified": None,
                    "score": 1.2,
                }
            ],
        }
        client_cls.return_value.search_models.return_value = fake_payload

        exit_code, stdout, stderr = self._run_cli(
            [
                "model",
                "search",
                "llama",
                "--size-class",
                "medium",
                "--tag",
                "text-generation",
                "--license",
                "apache",
                "--json",
            ]
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr, "")
        payload = json.loads(stdout)
        self.assertEqual(payload["results"][0]["id"], "org/base-7b")
        client_cls.return_value.search_models.assert_called_once_with(
            "llama",
            page=1,
            limit=20,
            size_class="medium",
            tags=["text-generation"],
            license_name="apache",
        )

    @patch("mlx_lab.commands.model.HFModelClient")
    @patch("mlx_lab.cli.ensure_supported_platform", return_value=None)
    def test_model_inspect_returns_non_zero_on_api_error(self, _platform_mock: object, client_cls: object) -> None:
        client_cls.return_value.inspect_model.side_effect = APIRequestError("network failure")

        exit_code, stdout, stderr = self._run_cli(["model", "inspect", "org/base-7b"])

        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("Error: network failure", stderr)


if __name__ == "__main__":
    unittest.main()

