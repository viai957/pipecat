#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Benchmark: Rust native pipeline vs Python pipeline frame throughput.

Sends N frames through a 3-processor passthrough chain and measures
throughput. Shows the performance difference when the native engine
handles frame routing.

Run:
    python examples/native/03-native-vs-python-benchmark.py

Requires: pipecat-ai-native (install with `make dev-native`)
"""

import asyncio
import time

from pipecat._native_status import is_native_available
from pipecat.frames.frames import EndFrame, TextFrame
from pipecat.processors.frame_processor import FrameProcessor


class PassthroughProcessor(FrameProcessor):
    """Minimal processor that forwards all frames."""

    async def process_frame(self, frame, direction):
        await self.push_frame(frame, direction)


async def benchmark_rust(n_frames: int, n_processors: int) -> float:
    """Benchmark Rust-routed pipeline. Returns frames/sec."""
    from pipecat._native import NativePipelineTask

    processors = [PassthroughProcessor(name=f"p{i}") for i in range(n_processors)]

    task = NativePipelineTask(
        processors=processors,
        heartbeat_secs=None,
    )

    pipeline_future = asyncio.ensure_future(task.run_async())
    await asyncio.sleep(0.3)

    # Warm up
    for _ in range(20):
        task.queue_frame(TextFrame(text="warmup"))
    await asyncio.sleep(0.2)

    # Benchmark
    start = time.perf_counter()
    for i in range(n_frames):
        task.queue_frame(TextFrame(text=f"msg{i}"))

    # Let frames propagate before ending
    await asyncio.sleep(0.3)
    task.queue_frame(EndFrame())

    await asyncio.wait_for(pipeline_future, timeout=30.0)
    elapsed = time.perf_counter() - start

    return n_frames / elapsed


async def benchmark_python(n_frames: int, n_processors: int) -> float:
    """Benchmark Python-routed pipeline. Returns frames/sec."""
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineParams, PipelineTask

    processors = [PassthroughProcessor(name=f"p{i}") for i in range(n_processors)]

    pipeline = Pipeline(processors)
    task = PipelineTask(pipeline, params=PipelineParams())
    runner = PipelineRunner()

    # Run pipeline in background
    run_future = asyncio.ensure_future(runner.run(task))
    await asyncio.sleep(0.3)

    # Warm up
    for _ in range(20):
        await task.queue_frame(TextFrame(text="warmup"))
    await asyncio.sleep(0.2)

    # Benchmark
    start = time.perf_counter()
    for i in range(n_frames):
        await task.queue_frame(TextFrame(text=f"msg{i}"))

    await task.queue_frame(EndFrame())
    await asyncio.wait_for(run_future, timeout=30.0)
    elapsed = time.perf_counter() - start

    return n_frames / elapsed


async def main():
    n_frames = 200
    n_processors = 3

    print(f"Benchmark: {n_frames} frames x {n_processors} processors")
    print("=" * 50)

    # Python benchmark
    print("\nRunning Python pipeline benchmark...")
    py_fps = await benchmark_python(n_frames, n_processors)
    print(f"  Python: {py_fps:.0f} frames/sec")

    # Rust benchmark (if available)
    if is_native_available():
        print("\nRunning Rust pipeline benchmark...")
        rust_fps = await benchmark_rust(n_frames, n_processors)
        print(f"  Rust:   {rust_fps:.0f} frames/sec")

        speedup = rust_fps / py_fps if py_fps > 0 else 0
        print(f"\n  Speedup: {speedup:.1f}x")
    else:
        print("\nNative engine not available. Install with: make dev-native")
        print("Skipping Rust benchmark.")

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
