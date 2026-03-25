"""Interruption handling benchmarks.

Measures how quickly the pipeline responds to interruptions:
- Idle pipeline interruption
- Under-load interruption (many queued frames)
- Mid-processing interruption
- Cascading interruption through processor chains
"""

import asyncio
import time
from typing import List

import pytest

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    StartFrame,
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


class SlowProcessor(FrameProcessor):
    """Processor that simulates work with configurable delay."""

    def __init__(self, delay_ms: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self._delay = delay_ms / 1000
        self.processed: List[Frame] = []
        self.interrupted = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InterruptionFrame):
            self.interrupted = True
            self.interrupt_time = time.perf_counter_ns()

        if isinstance(frame, TextFrame):
            await asyncio.sleep(self._delay)
            self.processed.append(frame)

        await self.push_frame(frame, direction)


class InterruptionTimer(FrameProcessor):
    """Records when InterruptionFrame arrives."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.interruption_received_time: float = 0
        self.frames_after_interruption: int = 0
        self._interrupted = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InterruptionFrame):
            self.interruption_received_time = time.perf_counter_ns()
            self._interrupted = True
        elif self._interrupted and isinstance(frame, TextFrame):
            self.frames_after_interruption += 1

        await self.push_frame(frame, direction)


class FrameCounter(FrameProcessor):
    """Counts text frames processed."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.count = 0
        self.done = asyncio.Event()
        self.target = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TextFrame):
            self.count += 1
            if self.target > 0 and self.count >= self.target:
                self.done.set()
        await self.push_frame(frame, direction)


# ---------------------------------------------------------------------------
# Interruption Response Time Benchmarks
# ---------------------------------------------------------------------------


class TestInterruptionResponse:
    """Measure interruption response latency in various scenarios."""

    @pytest.mark.asyncio
    async def test_idle_pipeline_interruption(self):
        """Interruption response time on an idle pipeline."""
        NUM_RUNS = 10
        recorder = LatencyRecorder()

        for _ in range(NUM_RUNS):
            timer_proc = InterruptionTimer(name="int-timer")
            pipeline = Pipeline([timer_proc])
            task = PipelineTask(pipeline, params=PipelineParams())

            async def push_interruption():
                await asyncio.sleep(0.1)  # Let pipeline fully start
                send_time = time.perf_counter_ns()
                await task.queue_frame(InterruptionFrame())
                await asyncio.sleep(0.1)  # Let it propagate
                await task.queue_frame(EndFrame())
                return send_time

            runner = PipelineRunner()
            results = await asyncio.gather(runner.run(task), push_interruption())
            send_time = results[1]

            if timer_proc.interruption_received_time > 0:
                latency = timer_proc.interruption_received_time - send_time
                recorder.record(latency)

        if recorder.count > 0:
            print(
                f"\nIdle interruption: p50={recorder.p50_us:.1f}us "
                f"p95={recorder.p95_us:.1f}us p99={recorder.p99_us:.1f}us"
            )

    @pytest.mark.asyncio
    async def test_loaded_pipeline_interruption(self):
        """Interruption response with 100 queued data frames."""
        NUM_RUNS = 5
        recorder = LatencyRecorder()

        for _ in range(NUM_RUNS):
            slow = SlowProcessor(delay_ms=5, name="slow")
            timer_proc = InterruptionTimer(name="int-timer")
            pipeline = Pipeline([slow, timer_proc])
            task = PipelineTask(pipeline, params=PipelineParams())

            async def push_load_then_interrupt():
                await asyncio.sleep(0.05)
                # Queue many data frames
                for i in range(100):
                    await task.queue_frame(TextFrame(text=f"load-{i}"))

                # Send interruption
                await asyncio.sleep(0.01)  # Let some frames start processing
                send_time = time.perf_counter_ns()
                await task.queue_frame(InterruptionFrame())
                await asyncio.sleep(0.5)  # Wait for processing
                await task.queue_frame(EndFrame())
                return send_time

            runner = PipelineRunner()
            results = await asyncio.gather(runner.run(task), push_load_then_interrupt())
            send_time = results[1]

            if timer_proc.interruption_received_time > 0:
                latency = timer_proc.interruption_received_time - send_time
                recorder.record(latency)

        if recorder.count > 0:
            print(
                f"\nLoaded interruption (100 queued): "
                f"p50={recorder.p50_us/1000:.1f}ms "
                f"p95={recorder.p95_us/1000:.1f}ms"
            )

    @pytest.mark.asyncio
    async def test_cascading_interruption_chain(self):
        """Interruption propagation through 8-processor chain."""
        NUM_PROCS = 8
        NUM_RUNS = 5
        recorder = LatencyRecorder()

        for _ in range(NUM_RUNS):
            timers = [InterruptionTimer(name=f"timer-{i}") for i in range(NUM_PROCS)]
            pipeline = Pipeline(timers)
            task = PipelineTask(pipeline, params=PipelineParams())

            async def push_interruption():
                await asyncio.sleep(0.1)
                send_time = time.perf_counter_ns()
                await task.queue_frame(InterruptionFrame())
                await asyncio.sleep(0.2)
                await task.queue_frame(EndFrame())
                return send_time

            runner = PipelineRunner()
            results = await asyncio.gather(runner.run(task), push_interruption())
            send_time = results[1]

            # Measure total cascade time (first proc to last proc)
            received_times = [
                t.interruption_received_time for t in timers
                if t.interruption_received_time > 0
            ]
            if len(received_times) >= 2:
                cascade_time = max(received_times) - min(received_times)
                recorder.record(cascade_time)

        if recorder.count > 0:
            print(
                f"\nCascading interruption ({NUM_PROCS} procs): "
                f"p50={recorder.p50_us:.1f}us "
                f"p95={recorder.p95_us:.1f}us"
            )


# ---------------------------------------------------------------------------
# Queue Drain on Interruption
# ---------------------------------------------------------------------------


class TestInterruptionQueueDrain:
    """Measure how quickly queued frames are cleared on interruption."""

    @pytest.mark.asyncio
    async def test_queue_drain_speed(self):
        """Time to drain N frames from queue after interruption."""
        from pipecat.pipeline.backpressure import BoundedFrameQueue

        queue = BoundedFrameQueue(name="drain-test")

        # Fill queue with data frames
        num_frames = 500
        for i in range(num_frames):
            frame = TextFrame(text=f"drain-{i}")
            await queue.put((frame, FrameDirection.DOWNSTREAM, None))

        # Measure drain time
        with Timer() as t:
            drained = 0
            while not queue.empty():
                _ = queue.get_nowait()
                drained += 1

        per_frame_ns = t.elapsed_ns / drained if drained > 0 else 0
        print(
            f"\nQueue drain ({drained} frames): "
            f"{per_frame_ns:.0f} ns/frame, "
            f"total: {t.elapsed_us:.1f} us"
        )

    @pytest.mark.asyncio
    async def test_interruption_frame_creation_cost(self):
        """Cost of creating InterruptionFrame (used in interruption fast path)."""
        num_creates = 100_000
        recorder = LatencyRecorder()

        for _ in range(num_creates):
            start = time.perf_counter_ns()
            _ = InterruptionFrame()
            elapsed = time.perf_counter_ns() - start
            recorder.record(elapsed)

        print(
            f"\nInterruptionFrame creation: "
            f"p50={recorder.p50_ns:.0f}ns "
            f"p95={recorder.p95_ns:.0f}ns "
            f"mean={recorder.mean_ns:.0f}ns"
        )
