"""Benchmark: Rust pipeline routing with Python processors.

Measures frame processing throughput through a multi-processor passthrough chain.
Run with:

    uv run --no-sync pytest tests/benchmarks/test_rust_vs_python_routing.py -v -s
"""

import asyncio
import time

import pytest


def _native_available():
    try:
        from pipecat._native_status import is_native_available

        return is_native_available()
    except ImportError:
        return False


@pytest.mark.skipif(not _native_available(), reason="pipecat-ai-native not installed")
class TestRustRoutingBenchmark:
    """Benchmark Rust frame routing throughput."""

    @pytest.mark.asyncio
    async def test_rust_routing_throughput(self):
        """Measure Rust routing throughput: N frames through 3-processor chain."""
        from pipecat._native import NativePipelineTask

        from pipecat.frames.frames import EndFrame, TextFrame
        from pipecat.processors.frame_processor import FrameProcessor

        N_FRAMES = 200
        N_PROCESSORS = 3
        sink_received = []

        class CountingProcessor(FrameProcessor):
            async def process_frame(self, frame, direction):
                await self.push_frame(frame, direction)

        def on_sink(reason):
            sink_received.append(reason)

        processors = [CountingProcessor() for _ in range(N_PROCESSORS)]

        task = NativePipelineTask(
            processors=processors,
            heartbeat_secs=None,
            sink_callback=on_sink,
        )

        run_coro = task.run_async()
        pipeline_task = asyncio.ensure_future(run_coro)
        await asyncio.sleep(0.3)

        # Queue all frames first, then EndFrame
        start = time.perf_counter()

        for i in range(N_FRAMES):
            task.queue_frame(TextFrame(text=f"msg{i}"))

        # Small delay to let frames propagate before sending EndFrame
        await asyncio.sleep(0.5)
        task.queue_frame(EndFrame())

        try:
            await asyncio.wait_for(pipeline_task, timeout=30.0)
        except asyncio.TimeoutError:
            task.cancel()
            await asyncio.sleep(0.1)
            pytest.fail("Rust pipeline timed out")

        elapsed = time.perf_counter() - start

        fps = N_FRAMES / elapsed
        us_per_frame = (elapsed / N_FRAMES) * 1_000_000

        print(f"\n{'=' * 60}")
        print(f"  Rust Routing Benchmark")
        print(f"  {N_FRAMES} frames × {N_PROCESSORS} processors")
        print(f"  Elapsed: {elapsed:.3f}s")
        print(f"  Throughput: {fps:.0f} frames/sec")
        print(f"  Per-frame: {us_per_frame:.1f} µs (queue→sink)")
        print(f"  Sink callback: {sink_received}")
        print(f"{'=' * 60}")

        assert sink_received == ["End"], f"Expected ['End'], got {sink_received}"

    @pytest.mark.asyncio
    async def test_rust_routing_frame_ordering(self):
        """Verify frame processing order is correct in Rust pipeline."""
        from pipecat._native import NativePipelineTask

        from pipecat.frames.frames import EndFrame, TextFrame
        from pipecat.processors.frame_processor import FrameProcessor

        order = []

        class OrderTracker(FrameProcessor):
            def __init__(self, tag):
                super().__init__()
                self._tag = tag

            async def process_frame(self, frame, direction):
                if isinstance(frame, TextFrame):
                    order.append(f"{self._tag}:{frame.text}")
                await self.push_frame(frame, direction)

        task = NativePipelineTask(
            processors=[OrderTracker("A"), OrderTracker("B")],
            heartbeat_secs=None,
        )

        run_coro = task.run_async()
        pipeline_task = asyncio.ensure_future(run_coro)
        await asyncio.sleep(0.2)

        task.queue_frame(TextFrame(text="1"))
        await asyncio.sleep(0.05)
        task.queue_frame(TextFrame(text="2"))
        await asyncio.sleep(0.05)
        task.queue_frame(EndFrame())

        try:
            await asyncio.wait_for(pipeline_task, timeout=10.0)
        except asyncio.TimeoutError:
            task.cancel()
            await asyncio.sleep(0.1)
            pytest.fail(f"Timed out. Order so far: {order}")

        # A processes msg1 → B processes msg1 → A processes msg2 → B processes msg2
        assert "A:1" in order, f"A:1 not in {order}"
        assert "B:1" in order, f"B:1 not in {order}"
        assert "A:2" in order, f"A:2 not in {order}"
        assert "B:2" in order, f"B:2 not in {order}"

        # A should process before B for each message
        assert order.index("A:1") < order.index("B:1"), f"A:1 should precede B:1 in {order}"
        assert order.index("A:2") < order.index("B:2"), f"A:2 should precede B:2 in {order}"

        print(f"\n  Frame ordering verified: {order}")
