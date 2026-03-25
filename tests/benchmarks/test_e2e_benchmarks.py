"""End-to-end pipeline benchmarks with realistic service mocks.

Simulates a real voice pipeline with mock services that have realistic
latency characteristics:
- MockSTT: 50ms TTFB, 200ms total
- MockLLM: 200ms first token, 10ms/token streaming
- MockTTS: 100ms TTFB, real-time audio generation

Measures framework-added latency on top of service latencies.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import List, Optional

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
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from .conftest import LatencyRecorder, Timer


# ---------------------------------------------------------------------------
# Mock Services
# ---------------------------------------------------------------------------


class MockSTTService(FrameProcessor):
    """Simulates STT with realistic latency.

    TTFB: 50ms, total processing: 200ms.
    Converts AudioRawFrames to TranscriptionFrames.
    """

    def __init__(self, ttfb_ms: float = 50, total_ms: float = 200, **kwargs):
        super().__init__(**kwargs)
        self._ttfb_ms = ttfb_ms
        self._total_ms = total_ms
        self._audio_chunks: List[bytes] = []
        self._collecting = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            self._collecting = True
            self._audio_chunks = []
            await self.push_frame(frame, direction)
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._collecting = False
            # Simulate STT processing
            await asyncio.sleep(self._ttfb_ms / 1000)
            await self.push_frame(
                TranscriptionFrame(
                    text="Hello, how are you?",
                    user_id="benchmark-user",
                    timestamp=str(time.time()),
                ),
                direction,
            )
            remaining = (self._total_ms - self._ttfb_ms) / 1000
            if remaining > 0:
                await asyncio.sleep(remaining)
            await self.push_frame(frame, direction)
        elif isinstance(frame, AudioRawFrame) and self._collecting:
            self._audio_chunks.append(frame.audio)
        else:
            await self.push_frame(frame, direction)


class MockLLMService(FrameProcessor):
    """Simulates LLM with streaming token output.

    First token: 200ms, subsequent: 10ms/token.
    Converts TranscriptionFrames to streamed TextFrames.
    """

    def __init__(
        self,
        first_token_ms: float = 200,
        per_token_ms: float = 10,
        num_tokens: int = 20,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._first_token_ms = first_token_ms
        self._per_token_ms = per_token_ms
        self._num_tokens = num_tokens

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            # Simulate first token latency
            await asyncio.sleep(self._first_token_ms / 1000)

            words = ["I'm", "doing", "great,", "thanks", "for", "asking!", "How", "can",
                     "I", "help", "you", "today?", "I'd", "be", "happy", "to", "assist",
                     "with", "anything", "you", "need."]

            for i in range(min(self._num_tokens, len(words))):
                if i > 0:
                    await asyncio.sleep(self._per_token_ms / 1000)
                await self.push_frame(TextFrame(text=words[i]), direction)
        else:
            await self.push_frame(frame, direction)


class MockTTSService(FrameProcessor):
    """Simulates TTS with audio generation.

    TTFB: 100ms, then generates audio chunks at real-time rate.
    Converts TextFrames to TTSAudioRawFrames.
    """

    def __init__(
        self,
        ttfb_ms: float = 100,
        sample_rate: int = 16000,
        chunk_duration_ms: float = 20,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._ttfb_ms = ttfb_ms
        self._sample_rate = sample_rate
        self._chunk_duration_ms = chunk_duration_ms
        self._text_buffer: List[str] = []
        self._generating = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            if not self._generating:
                self._generating = True
                await self.push_frame(TTSStartedFrame(), direction)
                # Simulate TTS TTFB
                await asyncio.sleep(self._ttfb_ms / 1000)

            # Generate audio chunks for this text
            samples_per_chunk = int(self._sample_rate * self._chunk_duration_ms / 1000)
            audio_data = b"\x00" * (samples_per_chunk * 2)  # 16-bit PCM

            # Generate 2 chunks per word
            for _ in range(2):
                await self.push_frame(
                    TTSAudioRawFrame(
                        audio=audio_data,
                        sample_rate=self._sample_rate,
                        num_channels=1,
                    ),
                    direction,
                )
        elif isinstance(frame, EndFrame):
            if self._generating:
                await self.push_frame(TTSStoppedFrame(), direction)
                self._generating = False
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)


class LatencyCapture(FrameProcessor):
    """Captures timing of specific frame types for latency measurement."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.first_text_time: float = 0
        self.first_audio_time: float = 0
        self.last_audio_time: float = 0
        self.text_count: int = 0
        self.audio_count: int = 0
        self.input_time: float = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        now = time.perf_counter_ns()

        if isinstance(frame, TranscriptionFrame) and self.input_time == 0:
            self.input_time = now
        elif isinstance(frame, TextFrame):
            self.text_count += 1
            if self.first_text_time == 0:
                self.first_text_time = now
        elif isinstance(frame, (TTSAudioRawFrame, AudioRawFrame)):
            self.audio_count += 1
            if self.first_audio_time == 0:
                self.first_audio_time = now
            self.last_audio_time = now

        await self.push_frame(frame, direction)


# ---------------------------------------------------------------------------
# E2E Pipeline Benchmarks
# ---------------------------------------------------------------------------


class TestE2EPipeline:
    """End-to-end voice pipeline latency benchmarks."""

    @pytest.mark.asyncio
    async def test_full_pipeline_ttfb(self):
        """Measure time from user speech end to first TTS audio output."""
        NUM_RUNS = 5
        recorder = LatencyRecorder()

        for run in range(NUM_RUNS):
            stt = MockSTTService(ttfb_ms=50, total_ms=100, name="stt")
            llm = MockLLMService(first_token_ms=200, per_token_ms=10, num_tokens=10, name="llm")
            tts = MockTTSService(ttfb_ms=100, name="tts")
            capture = LatencyCapture(name="capture")

            pipeline = Pipeline([stt, llm, tts, capture])
            task = PipelineTask(pipeline, params=PipelineParams())

            async def push_speech():
                await asyncio.sleep(0.05)
                input_start = time.perf_counter_ns()
                capture.input_time = input_start

                # Simulate user speaking then stopping
                await task.queue_frame(UserStartedSpeakingFrame())
                audio = b"\x00" * 1024
                for _ in range(5):
                    await task.queue_frame(
                        AudioRawFrame(audio=audio, sample_rate=16000, num_channels=1)
                    )
                await task.queue_frame(UserStoppedSpeakingFrame())

                # Wait for TTS output
                await asyncio.sleep(1.0)
                await task.queue_frame(EndFrame())

            runner = PipelineRunner()
            await asyncio.gather(runner.run(task), push_speech())

            if capture.first_audio_time > 0 and capture.input_time > 0:
                ttfb_ns = capture.first_audio_time - capture.input_time
                recorder.record(ttfb_ns)

        if recorder.count > 0:
            print(
                f"\nE2E TTFB: p50={recorder.p50_us/1000:.1f}ms "
                f"p95={recorder.p95_us/1000:.1f}ms "
                f"mean={recorder.mean_us/1000:.1f}ms "
                f"({recorder.count} runs)"
            )
            # Framework overhead = measured TTFB - (STT + LLM_first_token + TTS) service time
            service_time_ms = 50 + 200 + 100  # STT TTFB + LLM first token + TTS TTFB
            framework_overhead_ms = recorder.mean_us / 1000 - service_time_ms
            print(f"  Service time: {service_time_ms}ms, Framework overhead: {framework_overhead_ms:.1f}ms")

    @pytest.mark.asyncio
    async def test_streaming_token_throughput(self):
        """Measure LLM token streaming throughput through the pipeline."""
        llm = MockLLMService(first_token_ms=50, per_token_ms=5, num_tokens=50, name="llm")
        tts = MockTTSService(ttfb_ms=20, name="tts")
        capture = LatencyCapture(name="capture")

        pipeline = Pipeline([llm, tts, capture])
        task = PipelineTask(pipeline, params=PipelineParams())

        async def push_input():
            await asyncio.sleep(0.05)
            await task.queue_frame(
                TranscriptionFrame(
                    text="Hello",
                    user_id="bench",
                    timestamp=str(time.time()),
                )
            )
            await asyncio.sleep(2.0)
            await task.queue_frame(EndFrame())

        runner = PipelineRunner()
        await asyncio.gather(runner.run(task), push_input())

        print(
            f"\nStreaming: {capture.text_count} text frames, "
            f"{capture.audio_count} audio frames generated"
        )


# ---------------------------------------------------------------------------
# Context Aggregation Speed
# ---------------------------------------------------------------------------


class TestContextAggregation:
    """Measure context aggregation performance under streaming load."""

    def test_message_accumulation_speed(self):
        """Speed of accumulating messages in a list (simulating LLM context)."""
        from pipecat.frames.frames import TextFrame

        num_tokens = 1000
        context: List[dict] = []

        with Timer() as t:
            for i in range(num_tokens):
                context.append({"role": "assistant", "content": f"token_{i}"})

        per_token_ns = t.elapsed_ns / num_tokens
        print(f"\nContext accumulation: {per_token_ns:.0f} ns/token ({num_tokens} tokens)")

    def test_context_serialization_speed(self):
        """Speed of serializing a large context to JSON (common LLM prep step)."""
        import json

        context = [{"role": "assistant", "content": f"token_{i}"} for i in range(500)]

        num_iters = 100
        with Timer() as t:
            for _ in range(num_iters):
                _ = json.dumps(context)

        per_iter_us = t.elapsed_us / num_iters
        print(f"\nContext JSON serialization: {per_iter_us:.1f} us/call ({len(context)} messages)")
