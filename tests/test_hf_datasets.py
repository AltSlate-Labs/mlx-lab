"""Tests for Hugging Face dataset client behavior."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from urllib.error import URLError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mlx_lab.hf_datasets import APIRequestError, HFDatasetClient  # noqa: E402


class _FakeResponse:
    def __init__(self, payload: object) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False


class HFDatasetClientTests(unittest.TestCase):
    def test_search_supports_pagination_and_filters(self) -> None:
        requested_urls: list[str] = []

        def opener(request: object, timeout: int = 15) -> _FakeResponse:
            url = request.full_url  # type: ignore[attr-defined]
            requested_urls.append(url)
            if "huggingface.co/api/datasets?" in url:
                return _FakeResponse(
                    [
                        {
                            "id": "org/en-dataset",
                            "downloads": 120,
                            "likes": 4,
                            "tags": [
                                "language:en",
                                "license:apache-2.0",
                                "task_categories:text-classification",
                            ],
                            "cardData": {"summary": "English classification dataset"},
                        },
                        {
                            "id": "org/fr-dataset",
                            "downloads": 80,
                            "likes": 2,
                            "tags": [
                                "language:fr",
                                "license:mit",
                                "task_categories:translation",
                            ],
                            "cardData": {"summary": "French translation dataset"},
                        },
                    ]
                )
            raise AssertionError(f"Unexpected URL: {url}")

        client = HFDatasetClient(opener=opener)
        payload = client.search_datasets(
            "medical",
            page=2,
            limit=5,
            language="en",
            task="classification",
            license_name="apache",
        )

        self.assertEqual(payload["page"], 2)
        self.assertEqual(payload["limit"], 5)
        self.assertEqual(payload["offset"], 5)
        self.assertEqual(payload["total_returned"], 1)
        self.assertEqual(payload["results"][0]["id"], "org/en-dataset")
        self.assertIn("search=medical", requested_urls[0])
        self.assertIn("offset=5", requested_urls[0])
        self.assertIn("limit=5", requested_urls[0])

    def test_inspect_returns_splits_and_feature_schema(self) -> None:
        def opener(request: object, timeout: int = 15) -> _FakeResponse:
            url = request.full_url  # type: ignore[attr-defined]
            if "huggingface.co/api/datasets/org/data?" in url:
                return _FakeResponse(
                    {
                        "id": "org/data",
                        "downloads": 300,
                        "likes": 20,
                        "lastModified": "2026-02-01T00:00:00.000Z",
                        "cardData": {
                            "summary": "A cleaned instruction dataset",
                            "language": ["en"],
                            "license": "apache-2.0",
                            "task_categories": ["question-answering"],
                        },
                    }
                )
            if "datasets-server.huggingface.co/splits?" in url:
                return _FakeResponse(
                    {
                        "splits": [
                            {"split": "train", "num_examples": 1000},
                            {"split": "validation", "num_examples": 200},
                        ]
                    }
                )
            if "datasets-server.huggingface.co/info?" in url:
                return _FakeResponse(
                    {
                        "dataset_info": {
                            "default": {
                                "features": {
                                    "prompt": {"dtype": "string"},
                                    "completion": {"dtype": "string"},
                                }
                            }
                        }
                    }
                )
            raise AssertionError(f"Unexpected URL: {url}")

        client = HFDatasetClient(opener=opener)
        payload = client.inspect_dataset("org/data")

        self.assertEqual(payload["dataset_id"], "org/data")
        self.assertEqual(payload["license"], "apache-2.0")
        self.assertEqual(payload["languages"], ["en"])
        self.assertEqual(payload["task_tags"], ["question-answering"])
        self.assertEqual([split["name"] for split in payload["splits"]], ["train", "validation"])
        self.assertIsInstance(payload["feature_schema"], dict)
        self.assertNotIn("warnings", payload)

    def test_inspect_reports_warnings_when_optional_endpoints_fail(self) -> None:
        def opener(request: object, timeout: int = 15) -> _FakeResponse:
            url = request.full_url  # type: ignore[attr-defined]
            if "huggingface.co/api/datasets/org/data?" in url:
                return _FakeResponse(
                    {
                        "id": "org/data",
                        "cardData": {
                            "summary": "Dataset with partial metadata",
                        },
                    }
                )
            if "datasets-server.huggingface.co/splits?" in url:
                raise URLError("network down")
            if "datasets-server.huggingface.co/info?" in url:
                raise URLError("network down")
            raise AssertionError(f"Unexpected URL: {url}")

        client = HFDatasetClient(opener=opener)
        payload = client.inspect_dataset("org/data")

        self.assertEqual(payload["splits"], [])
        self.assertIsNone(payload["feature_schema"])
        self.assertIn("warnings", payload)
        self.assertEqual(len(payload["warnings"]), 2)

    def test_search_raises_actionable_error_on_network_failure(self) -> None:
        def opener(_request: object, timeout: int = 15) -> _FakeResponse:
            raise URLError("offline")

        client = HFDatasetClient(opener=opener)
        with self.assertRaises(APIRequestError) as ctx:
            client.search_datasets("test")
        message = str(ctx.exception)
        self.assertIn("Unable to reach Hugging Face API endpoint", message)
        self.assertIn("offline", message)


if __name__ == "__main__":
    unittest.main()

