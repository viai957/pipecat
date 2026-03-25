"""Pipeline-level benchmarks for throughput and latency.

Measures:
- N-processor passthrough throughput (N=1,2,4,8,16)
- Latency percentiles through processor chains
- System frame priority over queued data frames
- Direct mode vs queued mode comparison
"""

import asyncio
import time
from typing import List

import pytest

from pipecat.frames.frames import (
    EndFrame,
    Frame,
    SystemFrame,
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


class CountingProcessor(FrameProcessor):
    """Counts frames passing through."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.frame_count = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        self.frame_count += 1
        await self.push_frame(frame, direction)


class TimestampingProcessor(FrameProcessor):
    """Records perf_counter_ns timestamp for each frame."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.timestamps: List[float] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        self.timestamps.append(time.perf_counter_ns())
        await self.push_frame(frame, direction)


class PassthroughProcessor(FrameProcessor):
    """Minimal processor that just forwards frames."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


class SinkProcessor(FrameProcessor):
    """Collects frames and signals completion."""

    def __init__(self, expected_count: int, **kwargs):
        super().__init__(**kwargs)
        self.frames: List[Frame] = []
        self.expected_count = expected_count
        self.done = asyncio.Event()
        self.first_frame_time: float = 0
        self.last_frame_time: float = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TextFrame):
            if not self.frames:
                self.first_frame_time = time.perf_counter_ns()
            self.frames.append(frame)
            self.last_frame_time = time.perf_counter_ns()
            if len(self.frames) >= self.expected_count:
                self.done.set()
        await self.push_frame(frame, direction)


# ---------------------------------------------------------------------------
# Throughput benchmarks
# ---------------------------------------------------------------------------


class TestPipelineThroughput:
    """Measure frame throughput through N-processor pipelines."""

    NUM_FRAMES = 1000

    @pytest.mark.asyncio
    @pytest.mark.parametrize("num_processors", [1, 2, 4, 8, 16])
    async def test_passthrough_throughput(self, num_processors: int):
        """Throughput through N passthrough processors."""
        processors = [PassthroughProcessor(name=f"pass-{i}") for i in range(num_processors)]
        sink = SinkProcessor(expected_count=self.NUM_FRAMES, name="sink")
        processors.append(sink)

        pipeline = Pipeline(processors)
        task = PipelineTask(pipeline, params=PipelineParams())

        async def push_frames():
            await asyncio.sleep(0.05)  # Let pipeline start
            start_time = time.perf_counter()
            for i in range(self.NUM_FRAMES):
                await task.queue_frame(TextFrame(text=f"msg-{i}"))

            # Wait for all frames to arrive at sink
            try:
                await asyncio.wait_for(sink.done.wait(), timeout=30.0)
            except asyncio.TimeoutError:
                pass

            elapsed = time.perf_counter() - start_time
            fps = len(sink.frames) / elapsed if elapsed > 0 else 0
            print(
                f"\n{num_processors}-proc throughput: {fps:.0f} fps "
                f"({len(sink.frames)}/{self.NUM_FRAMES} frames in {elapsed*1000:.1f}ms)"
            )
            await task.queue_frame(EndFrame())

        runner = PipelineRunner()
        await asyncio.gather(runner.run(task), push_frames())


class TestPipelineLatency:
    """Measure per-frame latency through processor chains."""

    @pytest.mark.asyncio
    async def test_8_processor_latency(self):
        """Latency percentiles through 8 processors."""
        NUM_FRAMES = 500
        entry = TimestampingProcessor(name="entry")
        middle = [PassthroughProcessor(name=f"mid-{i}") for i in range(6)]
        exit_proc = TimestampingProcessor(name="exit")
        sink = SinkProcessor(expected_count=NUM_FRAMES, name="sink")

        pipeline = Pipeline([entry] + middle + [exit_proc, sink])
        task = PipelineTask(pipeline, params=PipelineParams())

        async def push_frames():
            await asyncio.sleep(0.05)
            for i in range(NUM_FRAMES):
                await task.queue_frame(TextFrame(text=f"lat-{i}"))
            try:
                await asyncio.wait_for(sink.done.wait(), timeout=30.0)
            except asyncio.TimeoutError:
                pass
            await task.queue_frame(EndFrame())

        runner = PipelineRunner()
        await asyncio.gather(runner.run(task), push_frames())

        # Compute per-frame latency (entry timestamp -> exit timestamp)
        recorder = LatencyRecorder()
        pairs = min(len(entry.timestamps), len(exit_proc.timestamps))
        for i in range(pairs):
            latency = exit_proc.timestamps[i] - entry.timestamps[i]
            if latency > 0:
                recorder.record(latency)

        if recorder.count > 0:
            print(
                f"\n8-proc latency: p50={recorder.p50_us:.1f}us "
                f"p95={recorder.p95_us:.1f}us p99={recorder.p99_us:.1f}us "
                f"({recorder.count} samples)"
            )


# ---------------------------------------------------------------------------
# System frame priority
# ---------------------------------------------------------------------------


class TestSystemFramePriority:
    """Verify system frames are processed ahead of queued data frames."""

    @pytest.mark.asyncio
    async def test_system_frame_priority(self):
        """System frame should be processed before N queued data frames."""
        from pipecat.pipeline.backpressure import BoundedFrameQueue

        queue = BoundedFrameQueue(name="priority-test")

        # Fill queue with data frames
        num_data = 100
        for i in range(num_data):
            frame = TextFrame(text=f"data-{i}")
            await queue.put((frame, FrameDirection.DOWNSTREAM, None))

        # Add a system frame
        sys_frame = SystemFrame()
        await queue.put((sys_frame, FrameDirection.DOWNSTREAM, None))

        # First dequeue should be the system frame
        first_item = await queue.get()
        first_frame = first_item[0]

        is_system_first = isinstance(first_frame, SystemFrame)
        print(f"\nSystem frame priority: {'PASS' if is_system_first else 'FAIL'}")
        assert is_system_first, "System frame should be dequeued before data frames"


# ---------------------------------------------------------------------------
# Direct mode vs queued mode
# ---------------------------------------------------------------------------


class TestProcessFrameOverhead:
    """Measure raw process_frame call overhead using batched timing.

    Uses batched timing to amortize perf_counter_ns() overhead.
    Calls process_frame directly (no pipeline setup) to measure just
    the base class dispatch cost without queue or task overhead.
    """

    NUM_FRAMES = 10_000
    BATCH = 100

    @pytest.mark.asyncio
    async def test_process_frame_data_frame(self):
        """Cost of process_frame for a data frame (no dispatch match).

        Properly initializes __started to avoid logger.error() on every call,
        which would inflate measurements by ~16x.
        """
        processor = PassthroughProcessor(name="pf-data")
        # Initialize __started without full StartFrame (which requires task_manager)
        processor._FrameProcessor__started = True
        frame = TextFrame(text="test")
        num_samples = self.NUM_FRAMES // self.BATCH
        recorder = LatencyRecorder()

        for _ in range(num_samples):
            start = time.perf_counter_ns()
            for _ in range(self.BATCH):
                await processor.process_frame(frame, FrameDirection.DOWNSTREAM)
            elapsed = time.perf_counter_ns() - start
            recorder.record(elapsed / self.BATCH)

        print(
            f"\nprocess_frame (data): "
            f"p50={recorder.p50_ns:.0f}ns p95={recorder.p95_ns:.0f}ns "
            f"mean={recorder.mean_ns:.0f}ns (batched {self.BATCH}/sample)"
        )

    @pytest.mark.asyncio
    async def test_process_frame_system_frame(self):
        """Cost of process_frame for a system frame (dispatch match).

        Properly initializes __started to avoid logger.error() on every call.
        """
        processor = PassthroughProcessor(name="pf-sys")
        processor._FrameProcessor__started = True
        frame = SystemFrame()
        num_samples = self.NUM_FRAMES // self.BATCH
        recorder = LatencyRecorder()

        for _ in range(num_samples):
            start = time.perf_counter_ns()
            for _ in range(self.BATCH):
                await processor.process_frame(frame, FrameDirection.DOWNSTREAM)
            elapsed = time.perf_counter_ns() - start
            recorder.record(elapsed / self.BATCH)

        print(
            f"\nprocess_frame (system): "
            f"p50={recorder.p50_ns:.0f}ns p95={recorder.p95_ns:.0f}ns "
            f"mean={recorder.mean_ns:.0f}ns (batched {self.BATCH}/sample)"
        )
