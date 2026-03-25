"""VAD (Voice Activity Detection) inference benchmarks.

Measures:
- Silero VAD ONNX inference latency
- numpy buffer conversion overhead
- Model reset cost
- Per-call memory allocation
"""

import time

import numpy as np
import pytest

from tests.benchmarks.conftest import LatencyRecorder, Timer


class TestSileroVADInference:
    """Measure Silero VAD inference performance."""

    @pytest.fixture(autouse=True)
    def setup_vad(self):
        """Create a SileroVADAnalyzer for benchmarking."""
        try:
            from pipecat.audio.vad.silero import SileroVADAnalyzer

            self.analyzer = SileroVADAnalyzer(sample_rate=16000)
            # _sample_rate=0 until set_sample_rate() is called explicitly
            self.analyzer.set_sample_rate(16000)
            self.available = True
        except Exception:
            self.available = False

    def _make_audio_buffer(self, num_samples: int = 512) -> bytes:
        """Generate a realistic 16-bit PCM audio buffer."""
        # Generate sine wave at 440Hz to simulate speech-like audio
        t = np.linspace(0, num_samples / 16000, num_samples, dtype=np.float32)
        signal = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
        return signal.tobytes()

    def test_voice_confidence_latency(self):
        """Measure per-call voice_confidence latency."""
        if not self.available:
            pytest.skip("Silero VAD not available")

        buffer = self._make_audio_buffer(512)
        num_calls = 500
        recorder = LatencyRecorder()

        # Warmup: 100 calls to stabilize ONNX RT thread pool and CPU caches.
        # 10 calls is insufficient — ORT's per-session thread spinup and kernel
        # JIT compilation need ~50-100 iterations to reach steady state.
        for _ in range(100):
            self.analyzer.voice_confidence(buffer)

        for _ in range(num_calls):
            start = time.perf_counter_ns()
            _ = self.analyzer.voice_confidence(buffer)
            elapsed = time.perf_counter_ns() - start
            recorder.record(elapsed)

        print(
            f"\nVAD voice_confidence: "
            f"p50={recorder.p50_us:.1f}us "
            f"p95={recorder.p95_us:.1f}us "
            f"p99={recorder.p99_us:.1f}us "
            f"mean={recorder.mean_us:.1f}us "
            f"({num_calls} calls)"
        )

    def test_onnx_inference_only(self):
        """Measure raw ONNX model inference (no numpy conversion)."""
        if not self.available:
            pytest.skip("Silero VAD not available")

        buffer = self._make_audio_buffer(512)
        # Pre-convert to float32 outside the measurement loop
        audio_float32 = np.frombuffer(buffer, np.int16).astype(np.float32) / 32768.0

        num_calls = 500
        recorder = LatencyRecorder()

        # Warmup: must match voice_confidence warmup to ensure comparable
        # ORT thread pool state. Each new InferenceSession has its own
        # thread pool that needs OS scheduling before reaching steady state.
        for _ in range(100):
            _ = self.analyzer._model(audio_float32, 16000)

        for _ in range(num_calls):
            start = time.perf_counter_ns()
            _ = self.analyzer._model(audio_float32, 16000)
            elapsed = time.perf_counter_ns() - start
            recorder.record(elapsed)

        print(
            f"\nONNX inference only: "
            f"p50={recorder.p50_us:.1f}us "
            f"p95={recorder.p95_us:.1f}us "
            f"p99={recorder.p99_us:.1f}us "
            f"mean={recorder.mean_us:.1f}us "
            f"({num_calls} calls)"
        )

    def test_numpy_conversion_two_step(self):
        """Measure two-step numpy conversion (frombuffer + astype + divide)."""
        buffer = self._make_audio_buffer(512)
        num_calls = 10_000
        recorder = LatencyRecorder()

        for _ in range(num_calls):
            start = time.perf_counter_ns()
            # Two-step: frombuffer then astype+divide (2 allocations)
            audio_int16 = np.frombuffer(buffer, np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            elapsed = time.perf_counter_ns() - start
            recorder.record(elapsed)

        print(
            f"\nnumpy conversion (two-step): "
            f"p50={recorder.p50_us:.1f}us "
            f"p95={recorder.p95_us:.1f}us "
            f"mean={recorder.mean_us:.1f}us"
        )

    def test_numpy_conversion_optimized(self):
        """Measure optimized numpy conversion (single frombuffer)."""
        buffer = self._make_audio_buffer(512)
        num_calls = 10_000
        recorder = LatencyRecorder()

        for _ in range(num_calls):
            start = time.perf_counter_ns()
            audio_float32 = np.frombuffer(buffer, np.int16).astype(np.float32) / 32768.0
            elapsed = time.perf_counter_ns() - start
            recorder.record(elapsed)

        print(
            f"\nnumpy conversion (optimized): "
            f"p50={recorder.p50_us:.1f}us "
            f"p95={recorder.p95_us:.1f}us "
            f"mean={recorder.mean_us:.1f}us"
        )

    def test_model_reset_cost(self):
        """Measure cost of model state reset."""
        if not self.available:
            pytest.skip("Silero VAD not available")

        num_resets = 1000
        recorder = LatencyRecorder()

        for _ in range(num_resets):
            start = time.perf_counter_ns()
            self.analyzer._model.reset_states()
            elapsed = time.perf_counter_ns() - start
            recorder.record(elapsed)

        print(
            f"\nModel reset: "
            f"p50={recorder.p50_us:.1f}us "
            f"p95={recorder.p95_us:.1f}us "
            f"mean={recorder.mean_us:.1f}us"
        )

    def test_sr_array_allocation_cost(self):
        """Measure cost of np.array(sr) per-call allocation vs cached."""
        num_calls = 100_000

        # Current: allocate each time
        recorder_alloc = LatencyRecorder()
        for _ in range(num_calls):
            start = time.perf_counter_ns()
            _ = np.array(16000, dtype="int64")
            elapsed = time.perf_counter_ns() - start
            recorder_alloc.record(elapsed)

        # Optimized: pre-allocated
        cached_sr = np.array(16000, dtype="int64")
        recorder_cached = LatencyRecorder()
        for _ in range(num_calls):
            start = time.perf_counter_ns()
            _ = cached_sr  # Just reference
            elapsed = time.perf_counter_ns() - start
            recorder_cached.record(elapsed)

        print(
            f"\nnp.array(sr) per-call: p50={recorder_alloc.p50_ns:.0f}ns mean={recorder_alloc.mean_ns:.0f}ns"
        )
        print(
            f"cached sr reference:   p50={recorder_cached.p50_ns:.0f}ns mean={recorder_cached.mean_ns:.0f}ns"
        )
