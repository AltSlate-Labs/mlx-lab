"""Smoke tests for package import, CLI help, and platform guard behavior."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mlx_lab import __version__  # noqa: E402
from mlx_lab.runtime import UnsupportedPlatformError, ensure_supported_platform  # noqa: E402


class SmokeTests(unittest.TestCase):
    def test_package_import_exposes_version(self) -> None:
        self.assertIsInstance(__version__, str)
        self.assertTrue(__version__)

    def test_cli_help_lists_command_groups(self) -> None:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(SRC_DIR)
        result = subprocess.run(
            [sys.executable, "-m", "mlx_lab.cli", "--help"],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        output = result.stdout
        for group in ("model", "dataset", "data", "train", "run"):
            self.assertIn(group, output)

    def test_runtime_guard_rejects_unsupported_platform(self) -> None:
        with self.assertRaises(UnsupportedPlatformError) as ctx:
            ensure_supported_platform(system="Linux", machine="x86_64")
        message = str(ctx.exception)
        self.assertIn("macOS on Apple Silicon", message)
        self.assertIn("Linux", message)
        self.assertIn("x86_64", message)


if __name__ == "__main__":
    unittest.main()

