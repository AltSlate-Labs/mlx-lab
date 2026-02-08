"""Tests for Hugging Face model client behavior."""

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

from mlx_lab.hf_models import APIRequestError, HFModelClient  # noqa: E402


class _FakeResponse:
    def __init__(self, payload: object) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False


class HFModelClientTests(unittest.TestCase):
    def test_search_supports_filters_and_ranking(self) -> None:
        requested_urls: list[str] = []

        def opener(request: object, timeout: int = 15) -> _FakeResponse:
            url = request.full_url  # type: ignore[attr-defined]
            requested_urls.append(url)
            if "huggingface.co/api/models?" in url:
                return _FakeResponse(
                    [
                        {
                            "id": "org/base-3b",
                            "downloads": 400,
                            "likes": 12,
                            "pipeline_tag": "text-generation",
                            "tags": ["license:apache-2.0", "text-generation", "3b", "qwen"],
                            "config": {"model_type": "qwen2"},
                        },
                        {
                            "id": "mlx-community/base-7b",
                            "downloads": 300,
                            "likes": 40,
                            "pipeline_tag": "text-generation",
                            "tags": ["mlx", "license:apache-2.0", "text-generation", "7b", "llama"],
                            "config": {"model_type": "llama"},
                        },
                        {
                            "id": "org/base-13b",
                            "downloads": 350,
                            "likes": 18,
                            "pipeline_tag": "text-generation",
                            "tags": ["license:mit", "text-generation", "13b", "llama"],
                            "config": {"model_type": "llama"},
                        },
                    ]
                )
            raise AssertionError(f"Unexpected URL: {url}")

        client = HFModelClient(opener=opener)
        payload = client.search_models(
            "chat",
            page=2,
            limit=5,
            size_class="small",
            tags=["text-generation"],
            license_name="apache",
        )

        self.assertEqual(payload["page"], 2)
        self.assertEqual(payload["limit"], 5)
        self.assertEqual(payload["offset"], 5)
        self.assertEqual(payload["total_returned"], 1)
        self.assertEqual(payload["results"][0]["id"], "org/base-3b")
        self.assertEqual(payload["results"][0]["parameter_size_class"], "small")
        self.assertEqual(payload["results"][0]["compatibility"]["status"], "convertible")
        self.assertIn("search=chat", requested_urls[0])
        self.assertIn("offset=5", requested_urls[0])
        self.assertIn("limit=5", requested_urls[0])

    def test_inspect_returns_normalized_model_details(self) -> None:
        def opener(request: object, timeout: int = 15) -> _FakeResponse:
            url = request.full_url  # type: ignore[attr-defined]
            if "huggingface.co/api/models/org/base-7b?" in url:
                return _FakeResponse(
                    {
                        "id": "org/base-7b",
                        "downloads": 111,
                        "likes": 9,
                        "lastModified": "2026-02-01T00:00:00.000Z",
                        "pipeline_tag": "text-generation",
                        "tags": ["license:apache-2.0", "text-generation", "7b", "llama"],
                        "cardData": {"summary": "General purpose base model"},
                        "config": {"model_type": "llama", "tokenizer_class": "LlamaTokenizer", "vocab_size": 32000},
                        "tokenizer_config": {"model_max_length": 4096},
                    }
                )
            raise AssertionError(f"Unexpected URL: {url}")

        client = HFModelClient(opener=opener)
        payload = client.inspect_model("org/base-7b")

        self.assertEqual(payload["model_id"], "org/base-7b")
        self.assertEqual(payload["license"], "apache-2.0")
        self.assertEqual(payload["task_tag"], "text-generation")
        self.assertEqual(payload["architecture"], "llama")
        self.assertEqual(payload["parameter_size_class"], "medium")
        self.assertEqual(payload["compatibility"]["status"], "convertible")
        self.assertEqual(payload["tokenizer"]["class"], "LlamaTokenizer")
        self.assertEqual(payload["tokenizer"]["vocab_size"], 32000)
        self.assertEqual(payload["tokenizer"]["model_max_length"], 4096)

    def test_search_raises_actionable_error_on_network_failure(self) -> None:
        def opener(_request: object, timeout: int = 15) -> _FakeResponse:
            raise URLError("offline")

        client = HFModelClient(opener=opener)
        with self.assertRaises(APIRequestError) as ctx:
            client.search_models("llama")
        message = str(ctx.exception)
        self.assertIn("Unable to reach Hugging Face API endpoint", message)
        self.assertIn("offline", message)


if __name__ == "__main__":
    unittest.main()

