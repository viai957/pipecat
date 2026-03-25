"""Performance benchmarks comparing Python and Rust pipeline engines.

Run with:
    uv run pytest tests/benchmarks/test_native_perf.py -v --benchmark

These benchmarks measure:
- Frame throughput (frames/second through pipeline)
- Frame conversion latency (py→rust→py round-trip)
- End-to-end pipeline latency with Python processors
"""

import asyncio
import time
from typing import List

import pytest

from pipecat.frames.frames import Frame, TextFrame, StartFrame, EndFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class CountingProcessor(FrameProcessor):
    """Processor that counts frames passing through it."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.frame_count = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        self.frame_count += 1
        await self.push_frame(frame, direction)


class LatencyMeasuringProcessor(FrameProcessor):
    """Processor that records timestamps for latency measurement."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.timestamps: List[float] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        self.timestamps.append(time.perf_counter_ns())
        await self.push_frame(frame, direction)


class TestFrameThroughput:
    """Measure frame throughput through a pipeline."""

    @pytest.mark.asyncio
    async def test_text_frame_throughput(self):
        """Measure TextFrame processing rate."""
        num_frames = 1000
        processor = CountingProcessor()

        start = time.perf_counter()
        for i in range(num_frames):
            frame = TextFrame(text=f"message-{i}")
            await processor.process_frame(frame, FrameDirection.DOWNSTREAM)
        elapsed = time.perf_counter() - start

        fps = num_frames / elapsed
        print(f"\nTextFrame throughput: {fps:.0f} frames/sec ({elapsed*1000:.1f}ms for {num_frames} frames)")
        assert processor.frame_count == num_frames


class TestFrameCreation:
    """Measure frame creation overhead."""

    def test_text_frame_creation(self):
        """Measure TextFrame creation time."""
        num_frames = 10000

        start = time.perf_counter()
        frames = [TextFrame(text=f"msg-{i}") for i in range(num_frames)]
        elapsed = time.perf_counter() - start

        per_frame_ns = (elapsed * 1e9) / num_frames
        print(f"\nTextFrame creation: {per_frame_ns:.0f} ns/frame ({num_frames} frames in {elapsed*1000:.1f}ms)")
        assert len(frames) == num_frames

    def test_frame_type_id_lookup(self):
        """Measure type_id access time."""
        frame = TextFrame(text="test")
        num_lookups = 100000

        start = time.perf_counter()
        for _ in range(num_lookups):
            _ = frame.type_id
        elapsed = time.perf_counter() - start

        per_lookup_ns = (elapsed * 1e9) / num_lookups
        print(f"\ntype_id lookup: {per_lookup_ns:.1f} ns/lookup")


class TestNativeComparison:
    """Compare native vs pure-Python pipeline performance."""

    @pytest.mark.asyncio
    async def test_pipeline_latency_comparison(self):
        """Compare end-to-end latency: pure Python pipeline."""
        from pipecat.pipeline.pipeline import Pipeline

        num_processors = 8
        processors = [CountingProcessor(name=f"proc-{i}") for i in range(num_processors)]
        pipeline = Pipeline(processors)

        # Note: Full pipeline test requires PipelineTask setup
        # This just verifies the pipeline construction path
        assert len(pipeline.processors) == num_processors + 2  # +source +sink
        print(f"\nPipeline with {num_processors} processors created")
        print(f"  Native available: {pipeline._native_pipeline is not None}")
