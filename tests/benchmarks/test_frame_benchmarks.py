"""Microbenchmarks for frame creation, dispatch, and low-level operations.

Measures the cost of individual operations in the frame hot path:
- Frame instantiation (various types)
- isinstance chain vs type_id dispatch
- obj_id() throughput
- BoundedFrameQueue put/get
- Event handler check cost
- Observer notification cost
"""

import asyncio
import time
from typing import List

import numpy as np
import pytest

from pipecat.frames.frames import (
    AudioRawFrame,
    DataFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    StartFrame,
    SystemFrame,
    TextFrame,
    TranscriptionFrame,
)
from pipecat.frames.frame_types import FrameType
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.utils import obj_id

from .conftest import LatencyRecorder, Timer


# ---------------------------------------------------------------------------
# Frame Creation Benchmarks
# ---------------------------------------------------------------------------


class TestFrameCreation:
    """Measure frame instantiation cost for hot-path frame types."""

    NUM_FRAMES = 10_000

    def test_text_frame_creation(self):
        """TextFrame creation cost (ns/frame)."""
        start = time.perf_counter_ns()
        frames = [TextFrame(text=f"msg-{i}") for i in range(self.NUM_FRAMES)]
        elapsed = time.perf_counter_ns() - start

        per_frame = elapsed / self.NUM_FRAMES
        print(f"\nTextFrame creation: {per_frame:.0f} ns/frame ({self.NUM_FRAMES} frames)")
        assert len(frames) == self.NUM_FRAMES

    def test_audio_raw_frame_creation(self):
        """AudioRawFrame creation cost (ns/frame)."""
        audio_data = b"\x00" * 1024
        start = time.perf_counter_ns()
        frames = [
            AudioRawFrame(audio=audio_data, sample_rate=16000, num_channels=1)
            for _ in range(self.NUM_FRAMES)
        ]
        elapsed = time.perf_counter_ns() - start

        per_frame = elapsed / self.NUM_FRAMES
        print(f"\nAudioRawFrame creation: {per_frame:.0f} ns/frame ({self.NUM_FRAMES} frames)")
        assert len(frames) == self.NUM_FRAMES

    def test_system_frame_creation(self):
        """SystemFrame creation cost (ns/frame)."""
        start = time.perf_counter_ns()
        frames = [SystemFrame() for _ in range(self.NUM_FRAMES)]
        elapsed = time.perf_counter_ns() - start

        per_frame = elapsed / self.NUM_FRAMES
        print(f"\nSystemFrame creation: {per_frame:.0f} ns/frame ({self.NUM_FRAMES} frames)")
        assert len(frames) == self.NUM_FRAMES

    def test_data_frame_creation(self):
        """DataFrame creation cost (ns/frame)."""
        start = time.perf_counter_ns()
        frames = [DataFrame() for _ in range(self.NUM_FRAMES)]
        elapsed = time.perf_counter_ns() - start

        per_frame = elapsed / self.NUM_FRAMES
        print(f"\nDataFrame creation: {per_frame:.0f} ns/frame ({self.NUM_FRAMES} frames)")
        assert len(frames) == self.NUM_FRAMES

    def test_transcription_frame_creation(self):
        """TranscriptionFrame creation cost (ns/frame)."""
        start = time.perf_counter_ns()
        frames = [
            TranscriptionFrame(text=f"hello world {i}", user_id="user1", timestamp="now")
            for i in range(self.NUM_FRAMES)
        ]
        elapsed = time.perf_counter_ns() - start

        per_frame = elapsed / self.NUM_FRAMES
        print(f"\nTranscriptionFrame creation: {per_frame:.0f} ns/frame ({self.NUM_FRAMES} frames)")
        assert len(frames) == self.NUM_FRAMES


# ---------------------------------------------------------------------------
# Dispatch Cost Benchmarks
# ---------------------------------------------------------------------------


class TestDispatchCost:
    """Compare isinstance chain vs type_id dispatch overhead."""

    NUM_ITERS = 100_000

    def test_isinstance_chain_cost(self):
        """Cost of the 5-isinstance chain in process_frame.

        Uses batched timing (1000 iterations per sample) to avoid
        perf_counter_ns() overhead dominating sub-100ns measurements.
        """
        frame = TextFrame(text="test")
        BATCH = 1000
        num_samples = self.NUM_ITERS // BATCH
        recorder = LatencyRecorder()

        for _ in range(num_samples):
            start = time.perf_counter_ns()
            for _ in range(BATCH):
                if isinstance(frame, StartFrame):
                    pass
                elif isinstance(frame, InterruptionFrame):
                    pass
                elif isinstance(frame, EndFrame):
                    pass
                elif isinstance(frame, SystemFrame):
                    pass
                elif isinstance(frame, DataFrame):
                    pass  # This one matches for TextFrame
            elapsed = time.perf_counter_ns() - start
            recorder.record(elapsed / BATCH)

        print(f"\nisinstance chain: p50={recorder.p50_ns:.0f}ns p95={recorder.p95_ns:.0f}ns p99={recorder.p99_ns:.0f}ns (batched {BATCH}/sample)")

    def test_type_id_dispatch_cost(self):
        """Cost of type_id integer comparison dispatch.

        Uses batched timing to avoid timer overhead dominating.
        """
        frame = TextFrame(text="test")
        tid = frame.type_id
        BATCH = 1000
        num_samples = self.NUM_ITERS // BATCH
        recorder = LatencyRecorder()

        # Build a dispatch table
        dispatch = {
            FrameType.CTRL_START: lambda: None,
            FrameType.CTRL_INTERRUPT: lambda: None,
            FrameType.CTRL_END: lambda: None,
        }

        for _ in range(num_samples):
            start = time.perf_counter_ns()
            for _ in range(BATCH):
                handler = dispatch.get(tid)
                if handler:
                    handler()
            elapsed = time.perf_counter_ns() - start
            recorder.record(elapsed / BATCH)

        print(f"\ntype_id dispatch: p50={recorder.p50_ns:.0f}ns p95={recorder.p95_ns:.0f}ns p99={recorder.p99_ns:.0f}ns (batched {BATCH}/sample)")

    def test_type_id_lookup_throughput(self):
        """Pure type_id attribute access throughput."""
        frame = TextFrame(text="test")
        start = time.perf_counter_ns()
        for _ in range(self.NUM_ITERS):
            _ = frame.type_id
        elapsed = time.perf_counter_ns() - start

        per_lookup = elapsed / self.NUM_ITERS
        print(f"\ntype_id access: {per_lookup:.1f} ns/lookup ({self.NUM_ITERS} lookups)")

    def test_isinstance_single_check_cost(self):
        """Cost of a single isinstance check."""
        frame = TextFrame(text="test")
        recorder = LatencyRecorder()

        for _ in range(self.NUM_ITERS):
            start = time.perf_counter_ns()
            _ = isinstance(frame, TextFrame)
            elapsed = time.perf_counter_ns() - start
            recorder.record(elapsed)

        print(f"\nisinstance single: p50={recorder.p50_ns:.0f}ns p95={recorder.p95_ns:.0f}ns")


# ---------------------------------------------------------------------------
# obj_id Throughput
# ---------------------------------------------------------------------------


class TestObjIdThroughput:
    """Measure obj_id() call throughput."""

    def test_obj_id_throughput(self):
        """obj_id() calls per second."""
        num_calls = 1_000_000
        start = time.perf_counter_ns()
        for _ in range(num_calls):
            obj_id()
        elapsed = time.perf_counter_ns() - start

        per_call = elapsed / num_calls
        calls_per_sec = num_calls / (elapsed / 1e9)
        print(f"\nobj_id: {per_call:.1f} ns/call ({calls_per_sec:.0f} calls/sec)")


# ---------------------------------------------------------------------------
# BoundedFrameQueue Benchmarks
# ---------------------------------------------------------------------------


class TestQueueBenchmarks:
    """Measure BoundedFrameQueue put/get performance."""

    @pytest.mark.asyncio
    async def test_queue_put_data_frame(self):
        """Data frame put() cost."""
        from pipecat.pipeline.backpressure import BoundedFrameQueue

        queue = BoundedFrameQueue(name="bench-data")
        frame = TextFrame(text="test")
        num_ops = 10_000
        recorder = LatencyRecorder()

        for _ in range(num_ops):
            start = time.perf_counter_ns()
            await queue.put((frame, FrameDirection.DOWNSTREAM, None))
            elapsed = time.perf_counter_ns() - start
            recorder.record(elapsed)

        print(f"\nQueue put (data): p50={recorder.p50_ns:.0f}ns p95={recorder.p95_ns:.0f}ns mean={recorder.mean_ns:.0f}ns")

    @pytest.mark.asyncio
    async def test_queue_put_system_frame(self):
        """System frame put() cost."""
        from pipecat.pipeline.backpressure import BoundedFrameQueue

        queue = BoundedFrameQueue(name="bench-sys")
        frame = SystemFrame()
        num_ops = 10_000
        recorder = LatencyRecorder()

        for _ in range(num_ops):
            start = time.perf_counter_ns()
            await queue.put((frame, FrameDirection.DOWNSTREAM, None))
            elapsed = time.perf_counter_ns() - start
            recorder.record(elapsed)

        print(f"\nQueue put (system): p50={recorder.p50_ns:.0f}ns p95={recorder.p95_ns:.0f}ns mean={recorder.mean_ns:.0f}ns")

    @pytest.mark.asyncio
    async def test_queue_put_get_roundtrip(self):
        """Put+get round-trip latency."""
        from pipecat.pipeline.backpressure import BoundedFrameQueue

        queue = BoundedFrameQueue(name="bench-rt")
        frame = TextFrame(text="roundtrip")
        num_ops = 5_000
        recorder = LatencyRecorder()

        for _ in range(num_ops):
            start = time.perf_counter_ns()
            await queue.put((frame, FrameDirection.DOWNSTREAM, None))
            _ = await queue.get()
            elapsed = time.perf_counter_ns() - start
            recorder.record(elapsed)

        print(f"\nQueue put+get roundtrip: p50={recorder.p50_ns:.0f}ns p95={recorder.p95_ns:.0f}ns mean={recorder.mean_ns:.0f}ns")


# ---------------------------------------------------------------------------
# Event Handler Check Cost
# ---------------------------------------------------------------------------


class TestEventHandlerCost:
    """Measure the cost of checking for event handlers."""

    @pytest.mark.asyncio
    async def test_has_event_handler_check(self):
        """Cost of _has_handlers check (no handler registered)."""
        processor = FrameProcessor(name="bench-events")
        num_checks = 100_000
        recorder = LatencyRecorder()

        for _ in range(num_checks):
            start = time.perf_counter_ns()
            _ = processor._has_handlers("on_before_push_frame")
            elapsed = time.perf_counter_ns() - start
            recorder.record(elapsed)

        print(f"\n_has_handlers (empty): p50={recorder.p50_ns:.0f}ns p95={recorder.p95_ns:.0f}ns")

    @pytest.mark.asyncio
    async def test_event_handler_with_registration(self):
        """Cost of _has_handlers when one is registered."""
        processor = FrameProcessor(name="bench-events-reg")

        @processor.event_handler("on_before_push_frame")
        async def handler(proc, frame):
            pass

        num_checks = 100_000
        recorder = LatencyRecorder()

        for _ in range(num_checks):
            start = time.perf_counter_ns()
            _ = processor._has_handlers("on_before_push_frame")
            elapsed = time.perf_counter_ns() - start
            recorder.record(elapsed)

        print(f"\n_has_handlers (registered): p50={recorder.p50_ns:.0f}ns p95={recorder.p95_ns:.0f}ns")


# ---------------------------------------------------------------------------
# Observer Notification Cost
# ---------------------------------------------------------------------------


class TestObserverNotificationCost:
    """Measure the overhead of observer notifications on push_frame."""

    @pytest.mark.asyncio
    async def test_push_frame_without_observer(self):
        """push_frame cost without observer."""
        processor = FrameProcessor(name="bench-no-obs")
        frame = TextFrame(text="test")
        num_pushes = 10_000
        recorder = LatencyRecorder()

        for _ in range(num_pushes):
            start = time.perf_counter_ns()
            await processor.push_frame(frame, FrameDirection.DOWNSTREAM)
            elapsed = time.perf_counter_ns() - start
            recorder.record(elapsed)

        print(f"\npush_frame (no observer): p50={recorder.p50_ns:.0f}ns p95={recorder.p95_ns:.0f}ns")

    @pytest.mark.asyncio
    async def test_frame_pushed_creation_cost(self):
        """Cost of creating FramePushed dataclass (observer notification payload)."""
        from pipecat.observers.base_observer import FramePushed

        processor = FrameProcessor(name="bench-fp")
        frame = TextFrame(text="test")
        num_creates = 100_000
        recorder = LatencyRecorder()

        dest = FrameProcessor(name="bench-fp-dest")
        for _ in range(num_creates):
            start = time.perf_counter_ns()
            _ = FramePushed(
                source=processor,
                destination=dest,
                frame=frame,
                direction=FrameDirection.DOWNSTREAM,
                timestamp=0,
            )
            elapsed = time.perf_counter_ns() - start
            recorder.record(elapsed)

        print(f"\nFramePushed creation: p50={recorder.p50_ns:.0f}ns p95={recorder.p95_ns:.0f}ns")
