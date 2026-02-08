"""Tests for dataset CLI command wiring and output modes."""

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
from mlx_lab.hf_datasets import APIRequestError  # noqa: E402


class DatasetCLITests(unittest.TestCase):
    def _run_cli(self, argv: list[str]) -> tuple[int, str, str]:
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(argv)
        return exit_code, stdout.getvalue(), stderr.getvalue()

    @patch("mlx_lab.commands.dataset.HFDatasetClient")
    @patch("mlx_lab.cli.ensure_supported_platform", return_value=None)
    def test_dataset_search_json_output(self, _platform_mock: object, client_cls: object) -> None:
        fake_payload = {
            "query": "medical",
            "page": 2,
            "limit": 5,
            "offset": 5,
            "source_count": 2,
            "total_returned": 1,
            "filters": {
                "language": "en",
                "task": "classification",
                "license": "apache",
            },
            "results": [
                {
                    "id": "org/en-dataset",
                    "summary": "English classification dataset",
                    "languages": ["en"],
                    "license": "apache-2.0",
                    "task_tags": ["text-classification"],
                    "downloads": 10,
                    "likes": 1,
                    "last_modified": None,
                }
            ],
        }
        client_cls.return_value.search_datasets.return_value = fake_payload

        exit_code, stdout, stderr = self._run_cli(
            [
                "dataset",
                "search",
                "medical",
                "--page",
                "2",
                "--limit",
                "5",
                "--language",
                "en",
                "--task",
                "classification",
                "--license",
                "apache",
                "--json",
            ]
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr, "")
        rendered_payload = json.loads(stdout)
        self.assertEqual(rendered_payload["results"][0]["id"], "org/en-dataset")
        client_cls.return_value.search_datasets.assert_called_once_with(
            "medical",
            page=2,
            limit=5,
            language="en",
            task="classification",
            license_name="apache",
        )

    @patch("mlx_lab.commands.dataset.HFDatasetClient")
    @patch("mlx_lab.cli.ensure_supported_platform", return_value=None)
    def test_dataset_inspect_returns_non_zero_on_api_error(
        self,
        _platform_mock: object,
        client_cls: object,
    ) -> None:
        client_cls.return_value.inspect_dataset.side_effect = APIRequestError("network failure")

        exit_code, stdout, stderr = self._run_cli(["dataset", "inspect", "org/data"])

        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("Error: network failure", stderr)


if __name__ == "__main__":
    unittest.main()

