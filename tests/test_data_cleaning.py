"""Tests for deterministic data cleaning pipeline."""

from __future__ import annotations

import json
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mlx_lab.data_cleaning import DataCleaningError, clean_dataset  # noqa: E402


class DataCleaningTests(unittest.TestCase):
    def test_clean_dataset_mapping_is_deterministic_and_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "raw.jsonl"
            output_path = temp_path / "cleaned.jsonl"
            manifest_path = temp_path / "manifest.json"

            input_path.write_text(
                "\n".join(
                    [
                        json.dumps({"instruction": "Question 1", "answer": "Answer 1"}),
                        json.dumps({"instruction": "Question 2", "answer": "Answer 2"}),
                        json.dumps({"instruction": "  ", "answer": "Answer 3"}),
                        json.dumps({"instruction": "Question 1", "answer": "Answer 1"}),
                        json.dumps({"instruction": "Missing completion"}),
                        "not-json",
                        "",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            first_summary = clean_dataset(
                input_path=str(input_path),
                output_path=str(output_path),
                prompt_field="instruction",
                completion_field="answer",
                source_dataset_id="org/sample",
                source_dataset_version="v1",
                manifest_path=str(manifest_path),
                dedupe=True,
                input_format="jsonl",
            )

            first_bytes = output_path.read_bytes()
            first_sha = hashlib.sha256(first_bytes).hexdigest()

            second_summary = clean_dataset(
                input_path=str(input_path),
                output_path=str(output_path),
                prompt_field="instruction",
                completion_field="answer",
                source_dataset_id="org/sample",
                source_dataset_version="v1",
                manifest_path=str(manifest_path),
                dedupe=True,
                input_format="jsonl",
            )
            second_bytes = output_path.read_bytes()
            second_sha = hashlib.sha256(second_bytes).hexdigest()

            self.assertEqual(first_bytes, second_bytes)
            self.assertEqual(first_sha, second_sha)
            self.assertEqual(first_summary["output_sha256"], second_summary["output_sha256"])
            self.assertEqual(first_summary["written"], 2)
            self.assertEqual(first_summary["dropped"], 5)
            self.assertEqual(
                first_summary["drop_reasons"],
                {
                    "duplicate_record": 1,
                    "empty_line": 1,
                    "empty_prompt": 1,
                    "invalid_json": 1,
                    "missing_completion_field": 1,
                },
            )

            output_lines = output_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(output_lines), 2)
            for line in output_lines:
                record = json.loads(line)
                self.assertEqual(sorted(record.keys()), ["completion", "prompt"])
                self.assertIsInstance(record["prompt"], str)
                self.assertIsInstance(record["completion"], str)

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["source"]["dataset_id"], "org/sample")
            self.assertEqual(manifest["source"]["dataset_version"], "v1")
            self.assertEqual(manifest["transform"]["config_sha256"], first_summary["transform_config_sha256"])
            self.assertEqual(manifest["output"]["sha256"], first_summary["output_sha256"])

    def test_clean_dataset_supports_templates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "raw.json"
            output_path = temp_path / "cleaned.jsonl"

            input_path.write_text(
                json.dumps(
                    [
                        {"question": "What is MLX?", "context": "Apple ML framework", "answer": "A framework"},
                        {"question": "What is LoRA?", "context": "Adapter method", "answer": "A tuning method"},
                    ]
                ),
                encoding="utf-8",
            )

            summary = clean_dataset(
                input_path=str(input_path),
                output_path=str(output_path),
                prompt_template="Q: {question}\nContext: {context}",
                completion_template="{answer}",
                input_format="json",
            )

            lines = output_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(summary["written"], 2)
            self.assertEqual(len(lines), 2)
            first_record = json.loads(lines[0])
            self.assertEqual(first_record["prompt"], "Q: What is MLX?\nContext: Apple ML framework")
            self.assertEqual(first_record["completion"], "A framework")

    def test_clean_dataset_validates_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "raw.jsonl"
            output_path = temp_path / "cleaned.jsonl"
            input_path.write_text(json.dumps({"prompt": "a", "completion": "b"}) + "\n", encoding="utf-8")

            with self.assertRaises(DataCleaningError) as ctx:
                clean_dataset(
                    input_path=str(input_path),
                    output_path=str(output_path),
                    prompt_field="prompt",
                    prompt_template="{prompt}",
                    completion_field="completion",
                    input_format="jsonl",
                )
            self.assertIn("exactly one of --map-prompt or --prompt-template", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

