"""Tests for LoRA training workflow implementation."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mlx_lab.training_lora import TrainingError, run_lora_training  # noqa: E402


def _write_clean_dataset(path: Path, records: list[dict[str, str]]) -> None:
    lines = [json.dumps(record) for record in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class LoRATrainingTests(unittest.TestCase):
    @patch("mlx_lab.training_lora.importlib.util.find_spec", return_value=None)
    def test_training_writes_checkpoints_and_metrics(self, _find_spec_mock: object) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dataset_path = temp_path / "cleaned.jsonl"
            run_dir = temp_path / "run-a"
            _write_clean_dataset(
                dataset_path,
                [
                    {"prompt": "Prompt 1", "completion": "Completion 1"},
                    {"prompt": "Prompt 2", "completion": "Completion 2"},
                ],
            )

            summary = run_lora_training(
                model="org/base-3b",
                dataset_path=str(dataset_path),
                run_dir=str(run_dir),
                backend="simulated",
                max_steps=5,
                checkpoint_interval=2,
                batch_size=2,
                learning_rate=1e-4,
                lora_rank=8,
                seed=11,
            )

            self.assertEqual(summary["selected_backend"], "simulated")
            self.assertEqual(summary["latest_step"], 5)
            self.assertEqual(summary["resumed_from_step"], 0)
            self.assertIsNotNone(summary["manifest_path"])
            self.assertIsNotNone(summary["effective_config_sha256"])

            metrics_path = Path(summary["metrics_path"])
            metric_lines = metrics_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(metric_lines), 5)

            parsed_metrics = [json.loads(line) for line in metric_lines]
            self.assertEqual([entry["step"] for entry in parsed_metrics], [1, 2, 3, 4, 5])
            for entry in parsed_metrics:
                self.assertIn("loss", entry)
                self.assertIn("throughput_tokens_per_s", entry)
                self.assertIn("learning_rate", entry)

            checkpoints = sorted((run_dir / "checkpoints").glob("checkpoint-step-*.adapter.json"))
            self.assertEqual([path.name for path in checkpoints], [
                "checkpoint-step-000002.adapter.json",
                "checkpoint-step-000004.adapter.json",
                "checkpoint-step-000005.adapter.json",
            ])

            manifest_payload = json.loads(Path(summary["manifest_path"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest_payload["manifest_type"], "lora_run")
            self.assertEqual(manifest_payload["model"]["id_or_path"], "org/base-3b")
            self.assertEqual(
                manifest_payload["config"]["effective_config_sha256"],
                summary["effective_config_sha256"],
            )
            self.assertEqual(
                manifest_payload["dataset"]["sha256"],
                summary["preflight"]["dataset"]["sha256"],
            )
            self.assertEqual(manifest_payload["runtime"]["selected_backend"], "simulated")

    @patch("mlx_lab.training_lora.importlib.util.find_spec", return_value=None)
    def test_resume_continues_from_latest_checkpoint(self, _find_spec_mock: object) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dataset_path = temp_path / "cleaned.jsonl"
            run_dir = temp_path / "run-b"
            _write_clean_dataset(
                dataset_path,
                [
                    {"prompt": "Prompt 1", "completion": "Completion 1"},
                    {"prompt": "Prompt 2", "completion": "Completion 2"},
                ],
            )

            first = run_lora_training(
                model="org/base-3b",
                dataset_path=str(dataset_path),
                run_dir=str(run_dir),
                backend="simulated",
                max_steps=3,
                checkpoint_interval=2,
                seed=5,
            )
            self.assertEqual(first["latest_step"], 3)

            second = run_lora_training(
                model="org/base-3b",
                dataset_path=str(dataset_path),
                run_dir=str(run_dir),
                resume=True,
                backend="simulated",
                max_steps=6,
                checkpoint_interval=2,
                seed=5,
            )
            self.assertEqual(second["resumed_from_step"], 3)
            self.assertEqual(second["latest_step"], 6)
            manifest_payload = json.loads(Path(second["manifest_path"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest_payload["run"]["resumed_from_step"], 3)
            self.assertEqual(manifest_payload["run"]["latest_step"], 6)

            metrics_lines = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            steps = [json.loads(line)["step"] for line in metrics_lines]
            self.assertEqual(steps, [1, 2, 3, 4, 5, 6])

    @patch("mlx_lab.training_lora.importlib.util.find_spec", return_value=None)
    def test_preflight_validation_rejects_bad_dataset_schema(self, _find_spec_mock: object) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dataset_path = temp_path / "bad.jsonl"
            dataset_path.write_text(
                json.dumps({"prompt": "Prompt only"}) + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(TrainingError) as ctx:
                run_lora_training(
                    model="org/base-3b",
                    dataset_path=str(dataset_path),
                    run_dir=str(temp_path / "run-c"),
                    backend="simulated",
                    max_steps=2,
                )
            self.assertIn("expected keys ['prompt', 'completion'] exactly", str(ctx.exception))

    @patch("mlx_lab.training_lora.importlib.util.find_spec", return_value=None)
    def test_requires_mlx_dependencies_when_backend_is_mlx(self, _find_spec_mock: object) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dataset_path = temp_path / "cleaned.jsonl"
            _write_clean_dataset(
                dataset_path,
                [{"prompt": "Prompt 1", "completion": "Completion 1"}],
            )

            with self.assertRaises(TrainingError) as ctx:
                run_lora_training(
                    model="org/base-3b",
                    dataset_path=str(dataset_path),
                    run_dir=str(temp_path / "run-d"),
                    backend="mlx",
                )
            self.assertIn("backend `mlx` requires package `mlx`", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
