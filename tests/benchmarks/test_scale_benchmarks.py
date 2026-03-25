"""Scale benchmarks for concurrent pipeline capacity.

Measures:
- Memory and throughput for 10/100/1000 concurrent pipelines
- SharedThreadPool contention under load
- Backpressure behavior at scale
"""

import asyncio
import gc
import os
import sys
import time
import tracemalloc
from typing import List

import pytest

from pipecat.frames.frames import (
    EndFrame,
    Frame,
    TextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from .conftest import LatencyRecorder, Timer


# ---------------------------------------------------------------------------
# Helper processors
# ---------------------------------------------------------------------------


class MinimalProcessor(FrameProcessor):
    """Minimal processor for scale testing - lowest possible overhead."""

    def __init__(self, **kwargs):
        super().__init__(enable_direct_mode=True, **kwargs)
        self.frame_count = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        self.frame_count += 1
        await self.push_frame(frame, direction)


class CompletionTracker(FrameProcessor):
    """Tracks when expected number of frames have been processed."""

    def __init__(self, expected: int, done_event: asyncio.Event, **kwargs):
        super().__init__(**kwargs)
        self.count = 0
        self.expected = expected
        self.done_event = done_event

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TextFrame):
            self.count += 1
            if self.count >= self.expected:
                self.done_event.set()
        await self.push_frame(frame, direction)


# ---------------------------------------------------------------------------
# Concurrent pipeline benchmarks
# ---------------------------------------------------------------------------


class TestConcurrentPipelines:
    """Measure performance with many concurrent pipelines."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("num_pipelines", [10, 100])
    async def test_concurrent_pipeline_throughput(self, num_pipelines: int):
        """Throughput with N concurrent pipelines each processing frames."""
        frames_per_pipeline = 50
        total_frames = num_pipelines * frames_per_pipeline

        tasks: List[PipelineTask] = []
        trackers: List[CompletionTracker] = []
        done_events: List[asyncio.Event] = []

        # Measure memory before pipeline creation
        gc.collect()
        tracemalloc.start()
        mem_before = tracemalloc.get_traced_memory()[0]

        for i in range(num_pipelines):
            done = asyncio.Event()
            done_events.append(done)
            proc = MinimalProcessor(name=f"proc-{i}")
            tracker = CompletionTracker(
                expected=frames_per_pipeline,
                done_event=done,
                name=f"tracker-{i}",
            )
            trackers.append(tracker)
            pipeline = Pipeline([proc, tracker])
            task = PipelineTask(pipeline, params=PipelineParams())
            tasks.append(task)

        mem_after = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        mem_per_pipeline_kb = (mem_after - mem_before) / 1024 / num_pipelines

        # Run all pipelines concurrently
        start = time.perf_counter()

        async def run_pipeline(task: PipelineTask, done: asyncio.Event, idx: int):
            runner = PipelineRunner()

            async def push():
                await asyncio.sleep(0.05)
                for j in range(frames_per_pipeline):
                    await task.queue_frame(TextFrame(text=f"p{idx}-f{j}"))
                try:
                    await asyncio.wait_for(done.wait(), timeout=30.0)
                except asyncio.TimeoutError:
                    pass
                await task.queue_frame(EndFrame())

            await asyncio.gather(runner.run(task), push())

        await asyncio.gather(
            *[run_pipeline(t, d, i) for i, (t, d) in enumerate(zip(tasks, done_events))]
        )
        elapsed = time.perf_counter() - start

        total_processed = sum(t.count for t in trackers)
        fps = total_processed / elapsed if elapsed > 0 else 0

        print(
            f"\n{num_pipelines} concurrent pipelines:"
            f"\n  Throughput: {fps:.0f} total fps"
            f"\n  Per-pipeline: {fps/num_pipelines:.0f} fps"
            f"\n  Memory: {mem_per_pipeline_kb:.1f} KB/pipeline"
            f"\n  Total frames: {total_processed}/{total_frames}"
            f"\n  Elapsed: {elapsed:.2f}s"
        )

    def test_pipeline_creation_cost(self):
        """Measure cost of creating pipeline objects (no execution)."""
        num_pipelines = 1000

        gc.collect()
        tracemalloc.start()
        mem_before = tracemalloc.get_traced_memory()[0]

        with Timer() as t:
            pipelines = []
            for i in range(num_pipelines):
                proc = MinimalProcessor(name=f"p-{i}")
                pipeline = Pipeline([proc])
                pipelines.append(pipeline)

        mem_after = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()

        per_pipeline_us = t.elapsed_us / num_pipelines
        mem_per_pipeline_kb = (mem_after - mem_before) / 1024 / num_pipelines

        print(
            f"\nPipeline creation ({num_pipelines}):"
            f"\n  Time: {per_pipeline_us:.1f} us/pipeline"
            f"\n  Memory: {mem_per_pipeline_kb:.1f} KB/pipeline"
        )


# ---------------------------------------------------------------------------
# Backpressure under load
# ---------------------------------------------------------------------------


class TestBackpressureScale:
    """Test backpressure behavior under overload conditions."""

    @pytest.mark.asyncio
    async def test_queue_overflow_drop_oldest(self):
        """Measure drop rate with DROP_OLDEST policy under overload."""
        from pipecat.pipeline.backpressure import BoundedFrameQueue, DropPolicy, QueueConfig

        config = QueueConfig(max_size=100, drop_policy=DropPolicy.DROP_OLDEST)
        queue = BoundedFrameQueue(config=config, name="overflow-test")

        # Flood the queue
        num_frames = 1000
        with Timer() as t:
            for i in range(num_frames):
                frame = TextFrame(text=f"flood-{i}")
                await queue.put((frame, FrameDirection.DOWNSTREAM, None))

        stats = queue.stats
        drop_rate = stats["total_dropped"] / num_frames * 100 if num_frames > 0 else 0
        print(
            f"\nBackpressure (DROP_OLDEST, max=100):"
            f"\n  Enqueued: {stats['total_enqueued']}"
            f"\n  Dropped: {stats['total_dropped']}"
            f"\n  Drop rate: {drop_rate:.1f}%"
            f"\n  High water mark: {stats['high_water_mark']}"
            f"\n  Put time: {t.elapsed_us/num_frames:.1f} us/frame"
        )

    @pytest.mark.asyncio
    async def test_queue_overflow_drop_newest(self):
        """Measure behavior with DROP_NEWEST policy."""
        from pipecat.pipeline.backpressure import BoundedFrameQueue, DropPolicy, QueueConfig

        config = QueueConfig(max_size=100, drop_policy=DropPolicy.DROP_NEWEST)
        queue = BoundedFrameQueue(config=config, name="drop-newest-test")

        num_frames = 1000
        with Timer() as t:
            for i in range(num_frames):
                frame = TextFrame(text=f"flood-{i}")
                await queue.put((frame, FrameDirection.DOWNSTREAM, None))

        stats = queue.stats
        drop_rate = stats["total_dropped"] / num_frames * 100 if num_frames > 0 else 0
        print(
            f"\nBackpressure (DROP_NEWEST, max=100):"
            f"\n  Dropped: {stats['total_dropped']}"
            f"\n  Drop rate: {drop_rate:.1f}%"
            f"\n  Put time: {t.elapsed_us/num_frames:.1f} us/frame"
        )


# ---------------------------------------------------------------------------
# Frame allocation at scale
# ---------------------------------------------------------------------------


class TestFrameAllocationScale:
    """Test frame allocation patterns at scale."""

    def test_bulk_frame_allocation(self):
        """Measure cost of creating many frames (simulating scale)."""
        num_frames = 100_000
        gc.collect()

        tracemalloc.start()
        mem_before = tracemalloc.get_traced_memory()[0]

        with Timer() as t:
            frames = [TextFrame(text=f"scale-{i}") for i in range(num_frames)]

        mem_after = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()

        per_frame_ns = t.elapsed_ns / num_frames
        mem_per_frame = (mem_after - mem_before) / num_frames

        print(
            f"\nBulk frame allocation ({num_frames}):"
            f"\n  Time: {per_frame_ns:.0f} ns/frame"
            f"\n  Memory: {mem_per_frame:.0f} bytes/frame"
            f"\n  Total memory: {(mem_after - mem_before) / 1024 / 1024:.1f} MB"
        )
        assert len(frames) == num_frames
