#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Verify the Rust native engine is installed and check its capabilities.

Run:
    python examples/native/01-verify-native-engine.py

Expected output when native engine is installed:
    Native engine available: True
    API version: 1
    Type ID mismatches: 0
    Methods: cancel, is_running, queue_frame, queue_upstream_frame, run_async, sink_callback

Expected output without native engine:
    Native engine available: False
    Install with: make dev-native
"""

from pipecat._native_status import is_native_available, is_native_engine_enabled


def main():
    available = is_native_available()
    enabled = is_native_engine_enabled()

    print(f"Native engine available: {available}")
    print(f"Native engine enabled:   {enabled}")

    if not available:
        print("\nNative engine is not installed.")
        print("Install with:")
        print("  make dev-native")
        print("\nOr manually:")
        print("  cd pipecat-rs/crates/pipecat-python && pip install -e .")
        return

    # Import native module and inspect
    from pipecat._native import NATIVE_API_VERSION, NativePipelineTask, verify_type_ids

    print(f"API version: {NATIVE_API_VERSION}")

    # Check type ID alignment
    mismatches = verify_type_ids()
    print(f"Type ID mismatches: {len(mismatches)}")
    if mismatches:
        for name, py_val, rust_val in mismatches:
            print(f"  MISMATCH: {name} Python=0x{py_val:04X} Rust=0x{rust_val:04X}")

    # Show available methods
    methods = [m for m in dir(NativePipelineTask) if not m.startswith("_")]
    print(f"Methods: {', '.join(methods)}")

    print("\nNative engine is ready!")


if __name__ == "__main__":
    main()
