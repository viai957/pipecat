"""setup.py — auto-build Rust native extension on `pip install -e .`."""

import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.develop import develop

NATIVE_CARGO_TOML = Path(__file__).parent / "pipecat-rs/crates/pipecat-python/Cargo.toml"


def _build_native(quiet: bool = False):
    if not NATIVE_CARGO_TOML.exists():
        return  # Not a source checkout with Rust extension
    if not shutil.which("cargo"):
        print("[pipecat] cargo not found — running in pure Python mode")
        return
    print("[pipecat] Building Rust native extension via maturin...")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "maturin",
            "develop",
            "--manifest-path",
            str(NATIVE_CARGO_TOML),
        ],
        capture_output=quiet,
    )
    if result.returncode != 0:
        print("[pipecat] Warning: Rust build failed — pure Python fallback active")


class DevelopWithNative(develop):
    """Custom develop command that builds the Rust native extension."""

    def run(self):
        """Run the standard develop command, then build the native extension."""
        super().run()
        _build_native()


setup(cmdclass={"develop": DevelopWithNative})
