#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Show how existing Pipecat pipelines auto-detect the native engine.

This example demonstrates that standard Pipecat pipelines automatically
detect pipecat-ai-native when installed. No code changes are needed —
the native bridge is created transparently.

Run with native:
    python examples/native/04-existing-pipeline-with-native.py

Run without native:
    PIPECAT_NATIVE=0 python examples/native/04-existing-pipeline-with-native.py
"""

from pipecat._native_status import is_native_available, is_native_engine_enabled
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import FrameProcessor


class MockProcessor(FrameProcessor):
    async def process_frame(self, frame, direction):
        await self.push_frame(frame, direction)


def main():
    print(f"Native engine available: {is_native_available()}")
    print(f"Native engine enabled:   {is_native_engine_enabled()}")
    print()

    # Create a standard pipeline — same code regardless of native engine
    pipeline = Pipeline([
        MockProcessor(name="STT"),
        MockProcessor(name="LLM"),
        MockProcessor(name="TTS"),
    ])

    # Check if the pipeline auto-detected the native engine
    has_native = hasattr(pipeline, "_native_pipeline") and pipeline._native_pipeline is not None
    print(f"Pipeline native bridge: {'ACTIVE' if has_native else 'inactive'}")

    if has_native:
        print()
        print("The pipeline has a Rust native bridge ready.")
        print("When used with PipelineTask, frame routing can be")
        print("accelerated through Rust's mpsc channels.")
        print()
        print("Your existing code works unchanged — just install")
        print("pipecat-ai-native and pipelines get faster.")
    else:
        print()
        print("No native engine detected. Pipeline will use pure Python.")
        print("Install the native engine for acceleration:")
        print("  make dev-native")

    print()
    print("Escape hatch: PIPECAT_NATIVE=0 disables native engine")
    print()

    # For a full running pipeline example with Rust routing,
    # see 02-native-pipeline-passthrough.py


if __name__ == "__main__":
    main()
