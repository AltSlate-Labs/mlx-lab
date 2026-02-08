"""Tests for replay and compare run operations."""

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

from mlx_lab.run_ops import compare_runs, replay_run  # noqa: E402
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


class RunOpsTests(unittest.TestCase):
    @patch("mlx_lab.training_lora.importlib.util.find_spec", return_value=None)
    def test_replay_dry_run_returns_replay_parameters(self, _find_spec_mock: object) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dataset_path = temp_path / "cleaned.jsonl"
            _write_dataset(dataset_path)

            source_summary = run_lora_training(
                model="org/base-3b",
                dataset_path=str(dataset_path),
                run_dir=str(temp_path / "source-run"),
                backend="simulated",
                max_steps=3,
                checkpoint_interval=2,
                seed=13,
            )

            payload = replay_run(source_summary["manifest_path"], execute=False)
            self.assertEqual(payload["mode"], "dry_run")
            self.assertEqual(payload["source_manifest"], source_summary["manifest_path"])
            self.assertEqual(payload["replay_parameters"]["model"], "org/base-3b")
            self.assertEqual(payload["replay_parameters"]["dataset_path"], str(dataset_path))

    @patch("mlx_lab.training_lora.importlib.util.find_spec", return_value=None)
    def test_replay_execute_runs_new_training(self, _find_spec_mock: object) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dataset_path = temp_path / "cleaned.jsonl"
            _write_dataset(dataset_path)

            source_summary = run_lora_training(
                model="org/base-3b",
                dataset_path=str(dataset_path),
                run_dir=str(temp_path / "source-run"),
                backend="simulated",
                max_steps=2,
                checkpoint_interval=2,
                seed=9,
            )

            payload = replay_run(
                source_summary["manifest_path"],
                execute=True,
                run_name="replayed-run",
            )
            self.assertEqual(payload["mode"], "executed")
            replay_result = payload["replay_result"]
            self.assertEqual(replay_result["latest_step"], 2)
            self.assertTrue(Path(replay_result["manifest_path"]).exists())
            self.assertNotEqual(replay_result["run_dir"], source_summary["run_dir"])

    @patch("mlx_lab.training_lora.importlib.util.find_spec", return_value=None)
    def test_compare_runs_reports_deltas_and_config_differences(self, _find_spec_mock: object) -> None:
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
                seed=6,
            )

            comparison = compare_runs(run_a["manifest_path"], run_b["manifest_path"])
            self.assertEqual(comparison["run_a"]["run_id"], Path(run_a["run_dir"]).name)
            self.assertEqual(comparison["run_b"]["run_id"], Path(run_b["run_dir"]).name)
            self.assertEqual(comparison["run_a"]["summary"]["total_steps"], 3)
            self.assertEqual(comparison["run_b"]["summary"]["total_steps"], 5)
            self.assertEqual(comparison["deltas"]["total_steps_delta"], 2.0)

            difference_keys = {item["key"] for item in comparison["config_differences"]}
            self.assertIn("max_steps", difference_keys)
            self.assertIn("seed", difference_keys)


if __name__ == "__main__":
    unittest.main()

