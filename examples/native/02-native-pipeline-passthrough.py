#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Run a pipeline directly on the Rust engine with Python processors.

This example creates a NativePipelineTask with Python processors and routes
frames through Rust's mpsc channels. The processors are called via
GIL-protected callbacks while Rust owns the frame routing loop.

Run:
    python examples/native/02-native-pipeline-passthrough.py

Requires: pipecat-ai-native (install with `make dev-native`)
"""

import asyncio

from pipecat._native_status import is_native_available
from pipecat.frames.frames import EndFrame, TextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class TextLogger(FrameProcessor):
    """Logs text frames as they pass through the pipeline."""

    async def process_frame(self, frame, direction):
        if isinstance(frame, TextFrame):
            print(f"  [{self.name}] Text: {frame.text}")
        await self.push_frame(frame, direction)


class UpperCaseProcessor(FrameProcessor):
    """Transforms text frames to uppercase."""

    async def process_frame(self, frame, direction):
        if isinstance(frame, TextFrame):
            frame = TextFrame(text=frame.text.upper())
        await self.push_frame(frame, direction)


async def main():
    if not is_native_available():
        print("Native engine not available. Install with: make dev-native")
        return

    from pipecat._native import NativePipelineTask

    print("Creating Rust-driven pipeline with 3 processors...")

    # Create processors — standard Python FrameProcessors
    logger_in = TextLogger(name="Input")
    transformer = UpperCaseProcessor(name="Transform")
    logger_out = TextLogger(name="Output")

    # Create a NativePipelineTask — Rust owns the routing loop
    task = NativePipelineTask(
        processors=[logger_in, transformer, logger_out],
        heartbeat_secs=None,
        sink_callback=lambda reason: print(f"\nPipeline ended: {reason}"),
    )

    # Start the Rust pipeline in the background
    pipeline_future = asyncio.ensure_future(task.run_async())

    # Give the pipeline a moment to initialize
    await asyncio.sleep(0.3)

    # Send frames through Rust channels
    print("\nSending frames through Rust pipeline:")
    for text in ["hello world", "pipecat is fast", "rust engine active"]:
        print(f"\n  Queuing: {text}")
        task.queue_frame(TextFrame(text=text))
        await asyncio.sleep(0.1)

    # Terminate the pipeline
    task.queue_frame(EndFrame())

    # Wait for completion
    await asyncio.wait_for(pipeline_future, timeout=5.0)
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
