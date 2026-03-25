"""Detection module for the optional Rust native engine.

The pipecat-ai-native package provides a Rust-based pipeline engine that
validates processor compatibility and will eventually drive the frame routing
hot path for improved performance. This module detects whether the native
engine is installed and compatible.

Escape hatch: set ``PIPECAT_NATIVE=0`` to disable the Rust engine even when
the package is installed.
"""

from __future__ import annotations

import os

_HAS_NATIVE = False
_NATIVE_DISABLED = os.environ.get("PIPECAT_NATIVE", "1") == "0"

if not _NATIVE_DISABLED:
    try:
        from pipecat._native import NATIVE_API_VERSION  # type: ignore[import-not-found]

        _HAS_NATIVE = NATIVE_API_VERSION == 1
    except ImportError:
        pass


def is_native_available() -> bool:
    """Return True if the Rust native engine is installed and enabled."""
    return _HAS_NATIVE and not _NATIVE_DISABLED


def is_native_engine_enabled() -> bool:
    """Return True if the native engine should drive pipeline execution.

    Activates automatically when ``pipecat-ai-native`` is installed.
    Escape hatch: set ``PIPECAT_NATIVE=0`` to force pure Python mode.
    """
    return is_native_available()
