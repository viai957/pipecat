#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Silero Voice Activity Detection (VAD) implementation for Pipecat.

This module provides a VAD analyzer based on the Silero VAD ONNX model,
which can detect voice activity in audio streams with high accuracy.
Supports 8kHz and 16kHz sample rates.
"""

import time
from typing import Optional

import numpy as np
from loguru import logger

from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams

# How often should we reset internal model state
_MODEL_RESET_STATES_TIME = 5.0

try:
    import onnxruntime

except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Silero VAD, you need to `pip install pipecat-ai[silero]`.")
    raise Exception(f"Missing module(s): {e}")


class SileroOnnxModel:
    """ONNX runtime wrapper for the Silero VAD model.

    Provides voice activity detection using the pre-trained Silero VAD model
    with ONNX runtime for efficient inference. Handles model state management
    and input validation for audio processing.

    Hot-path buffers are pre-allocated at initialization to avoid per-call
    malloc/free overhead (~6KB per inference at 30-50Hz).
    """

    def __init__(self, path, force_onnx_cpu=True):
        """Initialize the Silero ONNX model.

        Args:
            path: Path to the ONNX model file.
            force_onnx_cpu: Whether to force CPU execution provider.
        """
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL

        if force_onnx_cpu and "CPUExecutionProvider" in onnxruntime.get_available_providers():
            self.session = onnxruntime.InferenceSession(
                path, providers=["CPUExecutionProvider"], sess_options=opts
            )
        else:
            self.session = onnxruntime.InferenceSession(path, sess_options=opts)

        self.reset_states()
        self.sample_rates = [8000, 16000]
        # Cache sample rate arrays to avoid per-call np.array() allocation
        self._sr_arrays = {sr: np.array(sr, dtype="int64") for sr in self.sample_rates}
        # Pre-allocate the ORT input dict — values are updated in-place each call
        self._ort_inputs = {"input": None, "state": None, "sr": None}
        # Flag to skip validation after first successful call
        self._initialized = False

    def _validate_input(self, x, sr: int):
        """Validate and preprocess input audio data."""
        if x.ndim == 1:
            x = x[np.newaxis, :]
        if x.ndim > 2:
            raise ValueError(f"Too many dimensions for input audio chunk {x.ndim}")

        if sr not in self.sample_rates:
            raise ValueError(
                f"Supported sampling rates: {self.sample_rates} (or multiple of 16000)"
            )
        if sr / x.shape[1] > 31.25:
            raise ValueError("Input audio chunk is too short")

        return x, sr

    def reset_states(self, batch_size=1):
        """Reset the internal model states.

        Args:
            batch_size: Batch size for state initialization. Defaults to 1.
        """
        self._state = np.zeros((2, batch_size, 128), dtype="float32")
        self._context = np.zeros((batch_size, 0), dtype="float32")
        self._last_sr = 0
        self._last_batch_size = 0
        self._input_buf = None
        self._initialized = False

    def __call__(self, x, sr: int):
        """Process audio input through the VAD model."""
        num_samples = 512 if sr == 16000 else 256
        context_size = 64 if sr == 16000 else 32

        if not self._initialized:
            # First call: full validation and buffer setup
            x, sr = self._validate_input(x, sr)
            batch_size = x.shape[0]

            if not self._last_batch_size:
                self.reset_states(batch_size)
            if self._last_sr and self._last_sr != sr:
                self.reset_states(batch_size)
            if self._last_batch_size and self._last_batch_size != batch_size:
                self.reset_states(batch_size)

            # Initialize context to correct shape (avoids per-call check)
            if self._context.shape[1] == 0:
                self._context = np.zeros((batch_size, context_size), dtype="float32")

            # Pre-allocate input buffer: (batch, context_size + num_samples)
            self._input_buf = np.empty(
                (batch_size, context_size + num_samples), dtype="float32"
            )
            self._num_samples = num_samples
            self._context_size = context_size
            self._initialized = True
        else:
            # Steady-state fast path: skip validation, use cached sizes
            if x.ndim == 1:
                x = x[np.newaxis, :]

            # Check for sr/batch changes (rare but must handle)
            batch_size = x.shape[0]
            if self._last_sr != sr or self._last_batch_size != batch_size:
                # Fall back to full re-initialization
                self._initialized = False
                return self.__call__(x, sr)

        # Copy context and audio into pre-allocated buffer (no malloc)
        self._input_buf[:, :context_size] = self._context
        self._input_buf[:, context_size:] = x

        if sr in (8000, 16000):
            # Update pre-allocated dict values (no new dict allocation)
            self._ort_inputs["input"] = self._input_buf
            self._ort_inputs["state"] = self._state
            self._ort_inputs["sr"] = self._sr_arrays[sr]
            ort_outs = self.session.run(None, self._ort_inputs)
            out, state = ort_outs
            self._state = state
        else:
            raise ValueError()

        # Update context from the input buffer (must copy — buffer will be overwritten)
        self._context = self._input_buf[:, -context_size:].copy()
        self._last_sr = sr
        self._last_batch_size = batch_size

        return out


class SileroVADAnalyzer(VADAnalyzer):
    """Voice Activity Detection analyzer using the Silero VAD model.

    Implements VAD analysis using the pre-trained Silero ONNX model for
    accurate voice activity detection. Supports 8kHz and 16kHz sample rates
    with automatic model state management and periodic resets.
    """

    def __init__(self, *, sample_rate: Optional[int] = None, params: Optional[VADParams] = None):
        """Initialize the Silero VAD analyzer.

        Args:
            sample_rate: Audio sample rate (8000 or 16000 Hz). If None, will be set later.
            params: VAD parameters for detection thresholds and timing.
        """
        super().__init__(sample_rate=sample_rate, params=params)

        logger.debug("Loading Silero VAD model...")

        model_name = "silero_vad.onnx"
        package_path = "pipecat.audio.vad.data"

        try:
            import importlib_resources as impresources

            model_file_path = str(impresources.files(package_path).joinpath(model_name))
        except BaseException:
            from importlib import resources as impresources

            try:
                with impresources.path(package_path, model_name) as f:
                    model_file_path = f
            except BaseException:
                model_file_path = str(impresources.files(package_path).joinpath(model_name))

        self._model = SileroOnnxModel(model_file_path, force_onnx_cpu=True)

        self._last_reset_time = 0

        logger.debug("Loaded Silero VAD")

    #
    # VADAnalyzer
    #

    def set_sample_rate(self, sample_rate: int):
        """Set the sample rate for audio processing.

        Args:
            sample_rate: Audio sample rate (must be 8000 or 16000 Hz).

        Raises:
            ValueError: If sample rate is not 8000 or 16000 Hz.
        """
        if sample_rate != 16000 and sample_rate != 8000:
            raise ValueError(
                f"Silero VAD sample rate needs to be 16000 or 8000 (sample rate: {sample_rate})"
            )

        super().set_sample_rate(sample_rate)

    def num_frames_required(self) -> int:
        """Get the number of audio frames required for VAD analysis.

        Returns:
            Number of frames required (512 for 16kHz, 256 for 8kHz).
        """
        return 512 if self.sample_rate == 16000 else 256

    def voice_confidence(self, buffer) -> float:
        """Calculate voice activity confidence for the given audio buffer.

        Args:
            buffer: Audio buffer to analyze.

        Returns:
            Voice confidence score between 0.0 and 1.0.
        """
        try:
            # Convert 16-bit PCM to normalized float32 in [-1.0, 1.0]
            audio_float32 = np.frombuffer(buffer, np.int16).astype(np.float32) / 32768.0
            new_confidence = self._model(audio_float32, self.sample_rate)[0]

            # We need to reset the model from time to time because it doesn't
            # really need all the data and memory will keep growing otherwise.
            curr_time = time.monotonic()
            diff_time = curr_time - self._last_reset_time
            if diff_time >= _MODEL_RESET_STATES_TIME:
                self._model.reset_states()
                self._last_reset_time = curr_time

            return new_confidence
        except Exception as e:
            # This comes from an empty audio array
            logger.error(f"Error analyzing audio with Silero VAD: {e}")
            return 0
