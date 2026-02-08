"""Runtime platform checks for supported environments."""

from __future__ import annotations

import platform

SUPPORTED_SYSTEM = "Darwin"
SUPPORTED_ARCHES = {"arm64", "arm64e"}


class UnsupportedPlatformError(RuntimeError):
    """Raised when mlx-lab is run on an unsupported platform."""


def ensure_supported_platform(system: str | None = None, machine: str | None = None) -> None:
    """Validate that runtime is macOS on Apple Silicon."""
    detected_system = system or platform.system()
    detected_machine = machine or platform.machine().lower()

    if detected_system != SUPPORTED_SYSTEM or detected_machine not in SUPPORTED_ARCHES:
        raise UnsupportedPlatformError(
            "mlx-lab supports only macOS on Apple Silicon (arm64). "
            f"Detected system='{detected_system}', machine='{detected_machine}'."
        )

