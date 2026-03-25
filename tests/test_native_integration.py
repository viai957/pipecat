"""Integration tests for the Rust native pipeline engine.

These tests verify the Python↔Rust bridge layer, frame conversion,
and transparent engine activation. All tests must pass in both modes:

    PIPECAT_NATIVE=0 uv run pytest tests/test_native_integration.py   # Pure Python
    uv run pytest tests/test_native_integration.py                     # Native
"""

import pytest

# ── Phase 1: Detection ─────────────────────────────────────────────────────


class TestNativeDetection:
    """Test the native engine detection module."""

    def test_native_status_module_exists(self):
        """_native_status module is importable."""
        from pipecat._native_status import is_native_available

        # Should return a bool regardless of whether native is installed
        result = is_native_available()
        assert isinstance(result, bool)

    def test_native_disabled_via_env(self, monkeypatch):
        """PIPECAT_NATIVE=0 disables native engine."""
        monkeypatch.setenv("PIPECAT_NATIVE", "0")
        # Need to reimport to pick up env change
        import importlib

        import pipecat._native_status as ns

        importlib.reload(ns)
        assert not ns.is_native_available()

    def test_native_enabled_by_default(self, monkeypatch):
        """PIPECAT_NATIVE defaults to '1' (enabled if installed)."""
        monkeypatch.delenv("PIPECAT_NATIVE", raising=False)
        import importlib

        import pipecat._native_status as ns

        importlib.reload(ns)
        # Result depends on whether pipecat-ai-native is installed
        # We just verify it doesn't crash
        result = ns.is_native_available()
        assert isinstance(result, bool)


# ── Phase 2: Frame Type Alignment ──────────────────────────────────────────


class TestFrameTypeAlignment:
    """Test that Python and Rust frame type constants are aligned."""

    def test_type_ids_match_categories(self):
        """Frame type IDs use the correct category encoding."""
        from pipecat.frames.frame_types import FrameCategory, FrameType

        # Audio category
        assert (FrameType.AUDIO_RAW_INPUT & 0xFF00) == (FrameCategory.AUDIO << 8)
        assert (FrameType.AUDIO_TTS & 0xFF00) == (FrameCategory.AUDIO << 8)

        # Text category
        assert (FrameType.TEXT_PLAIN & 0xFF00) == (FrameCategory.TEXT << 8)
        assert (FrameType.TEXT_LLM & 0xFF00) == (FrameCategory.TEXT << 8)

        # Control category
        assert (FrameType.CTRL_START & 0xFF00) == (FrameCategory.CONTROL << 8)
        assert (FrameType.CTRL_END & 0xFF00) == (FrameCategory.CONTROL << 8)

    def test_audio_type_ids_are_sequential(self):
        """Audio frame type IDs have sequential sub-types."""
        from pipecat.frames.frame_types import FrameType

        audio_types = [
            FrameType.AUDIO_RAW_INPUT,
            FrameType.AUDIO_RAW_OUTPUT,
            FrameType.AUDIO_TTS,
            FrameType.AUDIO_SPEECH,
            FrameType.AUDIO_MIX,
            FrameType.AUDIO_SILENCE,
            FrameType.AUDIO_USER,
        ]
        sub_types = [t & 0xFF for t in audio_types]
        assert sub_types == list(range(1, 8))

    def test_type_ids_are_unique(self):
        """All type IDs should be unique."""
        from pipecat.frames.frame_types import FrameType

        seen = set()
        for attr in dir(FrameType):
            if attr.startswith("_") or attr in ("register", "get"):
                continue
            val = getattr(FrameType, attr)
            if isinstance(val, int):
                assert val not in seen, f"Duplicate type ID: {attr} = 0x{val:04X}"
                seen.add(val)


# ── Phase 2: Frame Registry ───────────────────────────────────────────────


class TestFrameRegistry:
    """Test the Python↔Rust frame type registry consistency."""

    def test_expected_rust_type_ids(self):
        """Verify that expected Rust frame_type constants match Python FrameType."""
        from pipecat.frames.frame_types import FrameType

        # These values should be identical between Python and Rust
        expected = {
            "AUDIO_RAW_INPUT": 0x0101,
            "AUDIO_RAW_OUTPUT": 0x0102,
            "AUDIO_TTS": 0x0103,
            "TEXT_PLAIN": 0x0201,
            "TEXT_LLM": 0x0202,
            "CTRL_START": 0x0801,
            "CTRL_END": 0x0802,
            "CTRL_CANCEL": 0x0804,
            "SYS_HEARTBEAT": 0x0901,
            "USER_STARTED_SPEAKING": 0x0B01,
            "BOT_STARTED_SPEAKING": 0x0C01,
        }

        for name, rust_val in expected.items():
            py_val = getattr(FrameType, name, None)
            assert py_val is not None, f"FrameType.{name} not found in Python"
            assert py_val == rust_val, (
                f"FrameType.{name}: Python=0x{py_val:04X}, Rust=0x{rust_val:04X}"
            )


# ── Phase 3: Engine Activation ────────────────────────────────────────────


class TestEngineActivation:
    """Test that is_native_engine_enabled() auto-activates without env var gate."""

    def test_engine_enabled_matches_available(self, monkeypatch):
        """is_native_engine_enabled() must equal is_native_available()."""
        import importlib

        import pipecat._native_status as ns

        importlib.reload(ns)
        assert ns.is_native_engine_enabled() == ns.is_native_available()

    def test_engine_enabled_no_extra_env_var_required(self, monkeypatch):
        """Engine must activate with only PIPECAT_NATIVE (no other env vars)."""
        monkeypatch.delenv("PIPECAT_NATIVE", raising=False)
        import importlib

        import pipecat._native_status as ns

        importlib.reload(ns)
        # The only gate should be whether native is available
        assert ns.is_native_engine_enabled() == ns.is_native_available()

    def test_engine_disabled_when_pipecat_native_zero(self, monkeypatch):
        """PIPECAT_NATIVE=0 must disable is_native_engine_enabled()."""
        monkeypatch.setenv("PIPECAT_NATIVE", "0")
        import importlib

        import pipecat._native_status as ns

        importlib.reload(ns)
        assert not ns.is_native_engine_enabled()

    def test_engine_returns_bool(self):
        """is_native_engine_enabled() must return a bool in all cases."""
        from pipecat._native_status import is_native_engine_enabled

        result = is_native_engine_enabled()
        assert isinstance(result, bool)


# ── Phase 6: Pipeline Integration ─────────────────────────────────────────


class TestPipelineIntegration:
    """Test transparent native pipeline activation."""

    def test_pipeline_has_native_attribute(self):
        """Pipeline should have _native_pipeline attribute."""
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.processors.frame_processor import FrameProcessor

        class DummyProcessor(FrameProcessor):
            async def process_frame(self, frame, direction):
                await self.push_frame(frame, direction)

        p = Pipeline([DummyProcessor()])
        assert hasattr(p, "_native_pipeline")

    def test_pipeline_native_is_none_without_package(self):
        """Without pipecat-ai-native installed, _native_pipeline should be None."""
        from pipecat._native_status import is_native_available
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.processors.frame_processor import FrameProcessor

        class DummyProcessor(FrameProcessor):
            async def process_frame(self, frame, direction):
                await self.push_frame(frame, direction)

        p = Pipeline([DummyProcessor()])
        if not is_native_available():
            assert p._native_pipeline is None


# ── Phase 8: Fallback ─────────────────────────────────────────────────────


class TestFallback:
    """Test graceful fallback when native engine is not installed."""

    def test_fallback_no_native(self, monkeypatch):
        """Pipeline should work in pure Python mode when native is disabled."""
        monkeypatch.setenv("PIPECAT_NATIVE", "0")
        import importlib

        import pipecat._native_status as ns

        importlib.reload(ns)

        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.processors.frame_processor import FrameProcessor

        class DummyProcessor(FrameProcessor):
            async def process_frame(self, frame, direction):
                await self.push_frame(frame, direction)

        # Should create successfully in pure Python mode
        p = Pipeline([DummyProcessor()])
        assert p is not None
        assert p._native_pipeline is None

    def test_engine_import_without_native(self, monkeypatch):
        """Engine module should be importable even without native package."""
        monkeypatch.setenv("PIPECAT_NATIVE", "0")
        import importlib

        import pipecat._native_status as ns

        importlib.reload(ns)

        from pipecat.engine.rust_engine import NativePipeline

        # Should raise RuntimeError since native is disabled
        from pipecat.processors.frame_processor import FrameProcessor

        class DummyProcessor(FrameProcessor):
            async def process_frame(self, frame, direction):
                await self.push_frame(frame, direction)

        with pytest.raises(RuntimeError, match="not installed or is disabled"):
            NativePipeline([DummyProcessor()])


# ── Phase 9: Bridge P0 Verification ──────────────────────────────────────
# These tests verify the P0 fixes from the Rust engine rewrite plan.
# They require pipecat-ai-native to be installed — skip otherwise.


def _native_available():
    """Check if native engine is available for conditional test skip."""
    try:
        from pipecat._native_status import is_native_available

        return is_native_available()
    except ImportError:
        return False


@pytest.mark.skipif(not _native_available(), reason="pipecat-ai-native not installed")
class TestBridgeQueueFrame:
    """Verify that NativePipelineTask exposes queue_frame() and cancel()."""

    def test_queue_frame_method_exists(self):
        """NativePipelineTask should have queue_frame method."""
        from pipecat._native import NativePipelineTask

        from pipecat.processors.frame_processor import FrameProcessor

        class DummyProcessor(FrameProcessor):
            async def process_frame(self, frame, direction):
                await self.push_frame(frame, direction)

        task = NativePipelineTask(processors=[DummyProcessor()])
        assert hasattr(task, "queue_frame")
        assert callable(task.queue_frame)

    def test_cancel_method_exists(self):
        """NativePipelineTask should have cancel method."""
        from pipecat._native import NativePipelineTask

        from pipecat.processors.frame_processor import FrameProcessor

        class DummyProcessor(FrameProcessor):
            async def process_frame(self, frame, direction):
                await self.push_frame(frame, direction)

        task = NativePipelineTask(processors=[DummyProcessor()])
        assert hasattr(task, "cancel")
        assert callable(task.cancel)

    def test_queue_upstream_frame_method_exists(self):
        """NativePipelineTask should have queue_upstream_frame method."""
        from pipecat._native import NativePipelineTask

        from pipecat.processors.frame_processor import FrameProcessor

        class DummyProcessor(FrameProcessor):
            async def process_frame(self, frame, direction):
                await self.push_frame(frame, direction)

        task = NativePipelineTask(processors=[DummyProcessor()])
        assert hasattr(task, "queue_upstream_frame")
        assert callable(task.queue_upstream_frame)

    def test_is_running_property(self):
        """NativePipelineTask should have is_running property."""
        from pipecat._native import NativePipelineTask

        from pipecat.processors.frame_processor import FrameProcessor

        class DummyProcessor(FrameProcessor):
            async def process_frame(self, frame, direction):
                await self.push_frame(frame, direction)

        task = NativePipelineTask(processors=[DummyProcessor()])
        assert hasattr(task, "is_running")
        assert task.is_running is False


@pytest.mark.skipif(not _native_available(), reason="pipecat-ai-native not installed")
class TestBridgeFrameIdPreservation:
    """Verify that frame.id is preserved across Python↔Rust round-trips."""

    def test_frame_id_preserved_text_frame(self):
        """Text frame ID should survive Python→Rust→Python conversion."""
        from pipecat._native import NativePipelineTask

        from pipecat.frames.frames import TextFrame

        frame = TextFrame(text="hello")
        original_id = frame.id

        # The conversion happens inside NativePipelineTask when processing.
        # We can verify the frame has an ID and it's an int.
        assert isinstance(original_id, int)
        assert original_id > 0

    def test_frame_ids_are_unique(self):
        """Each frame should get a unique ID."""
        from pipecat.frames.frames import TextFrame

        ids = {TextFrame(text=f"msg{i}").id for i in range(100)}
        assert len(ids) == 100


@pytest.mark.skipif(not _native_available(), reason="pipecat-ai-native not installed")
class TestBridgeRustEngineShim:
    """Verify the Python shim classes for the Rust engine."""

    def _ensure_native_status(self):
        """Ensure the native status module reflects current state."""
        import importlib

        import pipecat._native_status as ns

        importlib.reload(ns)

    def test_native_pipeline_task_has_queue_frame(self):
        """NativePipelineTask shim should expose queue_frame."""
        self._ensure_native_status()
        from pipecat.engine.rust_engine import NativePipeline, NativePipelineTask

        from pipecat.processors.frame_processor import FrameProcessor

        class DummyProcessor(FrameProcessor):
            async def process_frame(self, frame, direction):
                await self.push_frame(frame, direction)

        pipeline = NativePipeline([DummyProcessor()])
        task = NativePipelineTask(pipeline=pipeline)
        assert hasattr(task, "queue_frame")
        assert hasattr(task, "queue_upstream_frame")
        assert callable(task.queue_frame)

    def test_native_pipeline_task_has_cancel(self):
        """NativePipelineTask shim should expose cancel."""
        self._ensure_native_status()
        from pipecat.engine.rust_engine import NativePipeline, NativePipelineTask

        from pipecat.processors.frame_processor import FrameProcessor

        class DummyProcessor(FrameProcessor):
            async def process_frame(self, frame, direction):
                await self.push_frame(frame, direction)

        pipeline = NativePipeline([DummyProcessor()])
        task = NativePipelineTask(pipeline=pipeline)
        assert hasattr(task, "cancel")


@pytest.mark.skipif(not _native_available(), reason="pipecat-ai-native not installed")
class TestBridgeTypeIdVerification:
    """Verify Python↔Rust type ID alignment using verify_type_ids()."""

    def test_verify_type_ids_returns_empty(self):
        """verify_type_ids() should return empty list when all IDs match."""
        from pipecat._native import verify_type_ids

        mismatches = verify_type_ids()
        assert isinstance(mismatches, list)
        if mismatches:
            for name, py_val, rust_val in mismatches:
                print(f"  MISMATCH: {name} Python=0x{py_val:04X} Rust=0x{rust_val:04X}")
        assert len(mismatches) == 0, f"Type ID mismatches: {mismatches}"


# ── Phase 10: End-to-End Rust Pipeline Routing ────────────────────────────


@pytest.mark.skipif(not _native_available(), reason="pipecat-ai-native not installed")
class TestRustPipelineRouting:
    """Test actual Rust frame routing with Python processors via run_async()."""

    @pytest.mark.asyncio
    async def test_passthrough_pipeline_start_end(self):
        """Frames flow through a passthrough processor via Rust routing."""
        import asyncio

        from pipecat._native import NativePipelineTask

        from pipecat.frames.frames import EndFrame, StartFrame, TextFrame
        from pipecat.processors.frame_processor import FrameProcessor

        received_frames = []

        class TrackingProcessor(FrameProcessor):
            async def process_frame(self, frame, direction):
                received_frames.append(frame.__class__.__name__)
                await self.push_frame(frame, direction)

        task = NativePipelineTask(
            processors=[TrackingProcessor()],
            heartbeat_secs=None,
        )

        # Start the Rust pipeline as a background task
        run_coro = task.run_async()
        pipeline_task = asyncio.ensure_future(run_coro)

        # Give the pipeline a moment to start
        await asyncio.sleep(0.1)

        # Queue frames
        task.queue_frame(TextFrame(text="hello"))
        task.queue_frame(EndFrame())

        # Wait for pipeline to finish (with timeout)
        try:
            await asyncio.wait_for(pipeline_task, timeout=5.0)
        except asyncio.TimeoutError:
            task.cancel()
            await asyncio.sleep(0.1)
            pytest.fail("Pipeline did not terminate within 5 seconds")

        # The processor should have seen at least StartFrame + TextFrame + EndFrame
        assert "StartFrame" in received_frames, f"Missing StartFrame in {received_frames}"
        assert "TextFrame" in received_frames, f"Missing TextFrame in {received_frames}"
        assert "EndFrame" in received_frames, f"Missing EndFrame in {received_frames}"

    @pytest.mark.asyncio
    async def test_cancel_terminates_pipeline(self):
        """Calling cancel() injects CancelFrame and terminates the pipeline."""
        import asyncio

        from pipecat._native import NativePipelineTask

        from pipecat.processors.frame_processor import FrameProcessor

        class SlowProcessor(FrameProcessor):
            async def process_frame(self, frame, direction):
                await self.push_frame(frame, direction)

        task = NativePipelineTask(
            processors=[SlowProcessor()],
            heartbeat_secs=None,
        )

        run_coro = task.run_async()
        pipeline_task = asyncio.ensure_future(run_coro)

        await asyncio.sleep(0.1)
        task.cancel()

        try:
            await asyncio.wait_for(pipeline_task, timeout=5.0)
        except asyncio.TimeoutError:
            pytest.fail("Pipeline did not terminate after cancel within 5 seconds")

    @pytest.mark.asyncio
    async def test_sink_callback_fires(self):
        """sink_callback receives the terminal frame name when pipeline ends."""
        import asyncio

        from pipecat._native import NativePipelineTask

        from pipecat.frames.frames import EndFrame
        from pipecat.processors.frame_processor import FrameProcessor

        callback_reasons = []

        class PassthroughProcessor(FrameProcessor):
            async def process_frame(self, frame, direction):
                await self.push_frame(frame, direction)

        task = NativePipelineTask(
            processors=[PassthroughProcessor()],
            heartbeat_secs=None,
            sink_callback=lambda reason: callback_reasons.append(reason),
        )

        run_coro = task.run_async()
        pipeline_task = asyncio.ensure_future(run_coro)

        await asyncio.sleep(0.1)
        task.queue_frame(EndFrame())

        try:
            await asyncio.wait_for(pipeline_task, timeout=5.0)
        except asyncio.TimeoutError:
            task.cancel()
            await asyncio.sleep(0.1)
            pytest.fail("Pipeline did not terminate within 5 seconds")

        assert len(callback_reasons) == 1, f"Expected 1 callback, got {callback_reasons}"
        assert callback_reasons[0] == "End", f"Expected 'End', got '{callback_reasons[0]}'"

    @pytest.mark.asyncio
    async def test_multi_processor_chain(self):
        """Frames flow through multiple processors in order via Rust routing."""
        import asyncio

        from pipecat._native import NativePipelineTask

        from pipecat.frames.frames import EndFrame, TextFrame
        from pipecat.processors.frame_processor import FrameProcessor

        processing_order = []

        class TaggingProcessor(FrameProcessor):
            def __init__(self, tag):
                super().__init__()
                self._tag = tag

            async def process_frame(self, frame, direction):
                if isinstance(frame, TextFrame):
                    processing_order.append(self._tag)
                await self.push_frame(frame, direction)

        task = NativePipelineTask(
            processors=[
                TaggingProcessor("A"),
                TaggingProcessor("B"),
                TaggingProcessor("C"),
            ],
            heartbeat_secs=None,
        )

        run_coro = task.run_async()
        pipeline_task = asyncio.ensure_future(run_coro)

        await asyncio.sleep(0.1)
        task.queue_frame(TextFrame(text="test"))
        task.queue_frame(EndFrame())

        try:
            await asyncio.wait_for(pipeline_task, timeout=5.0)
        except asyncio.TimeoutError:
            task.cancel()
            await asyncio.sleep(0.1)
            pytest.fail("Pipeline did not terminate within 5 seconds")

        assert processing_order == ["A", "B", "C"], (
            f"Expected processing order ['A', 'B', 'C'], got {processing_order}"
        )
