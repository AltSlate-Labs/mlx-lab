"""Release-readiness smoke tests for packaging and installation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ReleasePackagingTests(unittest.TestCase):
    def test_build_generates_sdist_and_wheel(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir) / "dist"
            out_dir.mkdir(parents=True, exist_ok=True)

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "build",
                    "--sdist",
                    "--wheel",
                    "--no-isolation",
                    "--outdir",
                    str(out_dir),
                    str(PROJECT_ROOT),
                ],
                capture_output=True,
                text=True,
                check=False,
                cwd=PROJECT_ROOT,
                env={**os.environ, "PIP_NO_INDEX": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1"},
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)
            artifacts = sorted(path.name for path in out_dir.iterdir() if path.is_file())
            self.assertTrue(any(name.endswith(".tar.gz") for name in artifacts), artifacts)
            self.assertTrue(any(name.endswith(".whl") for name in artifacts), artifacts)

    def test_install_and_minimal_cli_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            user_base = temp_path / "userbase"
            work_dir = temp_path / "workflow"
            work_dir.mkdir(parents=True, exist_ok=True)

            env = dict(os.environ)
            env["PYTHONUSERBASE"] = str(user_base)
            env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
            env["PIP_NO_INDEX"] = "1"
            env.pop("PYTHONPATH", None)

            install = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    "--no-build-isolation",
                    "--user",
                    str(PROJECT_ROOT),
                ],
                capture_output=True,
                text=True,
                check=False,
                cwd=PROJECT_ROOT,
                env=env,
            )
            self.assertEqual(install.returncode, 0, msg=install.stderr or install.stdout)

            cli_path = user_base / "bin" / "mlx-lab"
            self.assertTrue(cli_path.exists(), f"CLI not installed at {cli_path}")

            help_result = subprocess.run(
                [str(cli_path), "--help"],
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
            self.assertEqual(help_result.returncode, 0, msg=help_result.stderr or help_result.stdout)
            self.assertIn("usage: mlx-lab", help_result.stdout)

            raw_path = work_dir / "raw.jsonl"
            cleaned_path = work_dir / "cleaned.jsonl"
            run_dir = work_dir / "run"
            raw_path.write_text(
                "\n".join(
                    [
                        json.dumps({"instruction": "What is LoRA?", "answer": "Low-Rank Adaptation"}),
                        json.dumps({"instruction": "What is MLX?", "answer": "Apple ML framework"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            clean_result = subprocess.run(
                [
                    str(cli_path),
                    "data",
                    "clean",
                    "--input",
                    str(raw_path),
                    "--output",
                    str(cleaned_path),
                    "--map-prompt",
                    "instruction",
                    "--map-completion",
                    "answer",
                    "--json",
                ],
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
            self.assertEqual(clean_result.returncode, 0, msg=clean_result.stderr or clean_result.stdout)
            clean_payload = json.loads(clean_result.stdout)
            self.assertEqual(clean_payload["written"], 2)
            self.assertTrue(cleaned_path.exists())

            train_result = subprocess.run(
                [
                    str(cli_path),
                    "train",
                    "lora",
                    "--model",
                    "org/demo-1b",
                    "--dataset",
                    str(cleaned_path),
                    "--run-dir",
                    str(run_dir),
                    "--backend",
                    "simulated",
                    "--max-steps",
                    "2",
                    "--checkpoint-interval",
                    "1",
                    "--json",
                ],
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
            self.assertEqual(train_result.returncode, 0, msg=train_result.stderr or train_result.stdout)
            train_payload = json.loads(train_result.stdout)
            self.assertEqual(train_payload["latest_step"], 2)
            self.assertTrue(Path(train_payload["manifest_path"]).exists())
            self.assertTrue(Path(train_payload["metrics_path"]).exists())


if __name__ == "__main__":
    unittest.main()
