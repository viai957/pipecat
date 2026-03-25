#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Audio utility functions for Pipecat.

This module provides common audio processing utilities including mixing,
format conversion, volume calculation, and codec transformations for
various audio formats used in Pipecat pipelines.
"""

import audioop

import numpy as np

from pipecat.audio.resamplers.base_audio_resampler import BaseAudioResampler
from pipecat.audio.resamplers.soxr_resampler import SOXRAudioResampler
from pipecat.audio.resamplers.soxr_stream_resampler import SOXRStreamAudioResampler

# Normal speech usually results in many samples between ±500 to ±5000, depending on loudness and mic gain.
# So we are using a threshold that is well below what real speech produces.
SPEAKING_THRESHOLD = 20

# Try to import Rust-accelerated audio utilities. Falls back to Python implementations.
try:
    from pipecat._native import audio as _native_audio

    _USE_NATIVE_AUDIO = True
except ImportError:
    _USE_NATIVE_AUDIO = False


def create_default_resampler(**kwargs) -> BaseAudioResampler:
    """Create a default audio resampler instance.

    .. deprecated:: 0.0.74
        This function is deprecated and will be removed in a future version.
        Use `create_stream_resampler` for real-time processing scenarios or
        `create_file_resampler` for batch processing of complete audio files.

    Args:
        **kwargs: Additional keyword arguments passed to the resampler constructor.

    Returns:
        A configured SOXRAudioResampler instance.
    """
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.warn(
            "`create_default_resampler` is deprecated. "
            "Use `create_stream_resampler` for real-time processing scenarios or "
            "`create_file_resampler` for batch processing of complete audio files.",
            DeprecationWarning,
            stacklevel=2,
        )
    return SOXRAudioResampler(**kwargs)


def create_file_resampler(**kwargs) -> BaseAudioResampler:
    """Create an audio resampler instance for batch processing of complete audio files.

    Args:
        **kwargs: Additional keyword arguments passed to the resampler constructor.

    Returns:
        A configured SOXRAudioResampler instance.
    """
    return SOXRAudioResampler(**kwargs)


def create_stream_resampler(**kwargs) -> BaseAudioResampler:
    """Create a stream audio resampler instance.

    Args:
        **kwargs: Additional keyword arguments passed to the resampler constructor.

    Returns:
        A configured SOXRStreamAudioResampler instance.
    """
    return SOXRStreamAudioResampler(**kwargs)


def mix_audio(audio1: bytes, audio2: bytes) -> bytes:
    """Mix two audio streams together by adding their samples.

    Both audio streams are assumed to be 16-bit signed integer PCM data.
    If the streams have different lengths, the shorter one is zero-padded
    to match the longer stream.

    Args:
        audio1: First audio stream as raw bytes (16-bit signed integers).
        audio2: Second audio stream as raw bytes (16-bit signed integers).

    Returns:
        Mixed audio data as raw bytes with samples clipped to 16-bit range.
    """
    data1 = np.frombuffer(audio1, dtype=np.int16)
    data2 = np.frombuffer(audio2, dtype=np.int16)

    # Max length
    max_length = max(len(data1), len(data2))

    # Zero-pad the arrays to the same length
    padded1 = np.pad(data1, (0, max_length - len(data1)), mode="constant")
    padded2 = np.pad(data2, (0, max_length - len(data2)), mode="constant")

    # Mix the arrays
    mixed_audio = padded1.astype(np.int32) + padded2.astype(np.int32)
    mixed_audio = np.clip(mixed_audio, -32768, 32767).astype(np.int16)

    return mixed_audio.astype(np.int16).tobytes()


def interleave_stereo_audio(left_audio: bytes, right_audio: bytes) -> bytes:
    """Interleave left and right mono audio channels into stereo audio.

    Takes two mono audio streams and combines them into a single stereo
    stream by interleaving the samples (L, R, L, R, ...). If the channels
    have different lengths, both are truncated to the shorter length.

    Args:
        left_audio: Left channel audio as raw bytes (16-bit signed integers).
        right_audio: Right channel audio as raw bytes (16-bit signed integers).

    Returns:
        Interleaved stereo audio data as raw bytes.
    """
    left = np.frombuffer(left_audio, dtype=np.int16)
    right = np.frombuffer(right_audio, dtype=np.int16)

    min_length = min(len(left), len(right))
    left = left[:min_length]
    right = right[:min_length]

    stereo = np.column_stack((left, right))

    return stereo.astype(np.int16).tobytes()


def normalize_value(value, min_value, max_value):
    """Normalize a value to the range [0, 1] and clamp it to bounds.

    Args:
        value: The value to normalize.
        min_value: The minimum value of the input range.
        max_value: The maximum value of the input range.

    Returns:
        Normalized value clamped to the range [0, 1].
    """
    normalized = (value - min_value) / (max_value - min_value)
    normalized_clamped = max(0, min(1, normalized))
    return normalized_clamped


def calculate_audio_volume(audio: bytes, sample_rate: int) -> float:
    """Calculate the loudness level of audio data using RMS amplitude.

    When the Rust native engine is available, delegates to the Rust
    implementation for ~10x faster processing. Falls back to numpy.

    Args:
        audio: Audio data as raw bytes (16-bit signed integers).
        sample_rate: Sample rate of the audio in Hz.

    Returns:
        Normalized loudness value between 0 (quiet) and 1 (loud).
    """
    if _USE_NATIVE_AUDIO:
        return _native_audio.calculate_audio_volume(audio, sample_rate)

    audio_np = np.frombuffer(audio, dtype=np.int16)

    # RMS amplitude (float64 to avoid int16 overflow in squaring)
    rms = np.sqrt(np.mean(audio_np.astype(np.float64) ** 2))

    if rms < 1.0:
        # Silence — avoid log10(0)
        return 0.0

    # dB relative to 16-bit full-scale (0 dBFS = 32768)
    db = 20.0 * np.log10(rms / 32768.0)

    # Typical speech range: roughly -60 dBFS (quiet) to 0 dBFS (clipping).
    # Normalize to [0, 1].
    loudness = normalize_value(db, -60.0, 0.0)

    return loudness


def exp_smoothing(value: float, prev_value: float, factor: float) -> float:
    """Apply exponential smoothing to a value.

    When Rust native engine is available, delegates to Rust.

    Exponential smoothing is used to reduce noise in time-series data by
    giving more weight to recent values while still considering historical data.

    Args:
        value: The new value to incorporate.
        prev_value: The previous smoothed value.
        factor: Smoothing factor between 0 and 1. Higher values give more
                weight to the new value.

    Returns:
        The exponentially smoothed value.
    """
    if _USE_NATIVE_AUDIO:
        return _native_audio.exp_smoothing(value, prev_value, factor)
    return prev_value + factor * (value - prev_value)


async def ulaw_to_pcm(
    ulaw_bytes: bytes, in_rate: int, out_rate: int, resampler: BaseAudioResampler
):
    """Convert μ-law encoded audio to PCM and optionally resample.

    Args:
        ulaw_bytes: μ-law encoded audio data as raw bytes.
        in_rate: Original sample rate of the μ-law audio in Hz.
        out_rate: Desired output sample rate in Hz.
        resampler: Audio resampler instance for rate conversion.

    Returns:
        PCM audio data as raw bytes at the specified output rate.
    """
    # Convert μ-law to PCM (Rust path is ~5x faster via lookup table)
    if _USE_NATIVE_AUDIO:
        in_pcm_bytes = bytes(_native_audio.ulaw_decode(ulaw_bytes))
    else:
        in_pcm_bytes = audioop.ulaw2lin(ulaw_bytes, 2)

    # Resample
    out_pcm_bytes = await resampler.resample(in_pcm_bytes, in_rate, out_rate)

    return out_pcm_bytes


async def pcm_to_ulaw(pcm_bytes: bytes, in_rate: int, out_rate: int, resampler: BaseAudioResampler):
    """Convert PCM audio to μ-law encoding and optionally resample.

    Args:
        pcm_bytes: PCM audio data as raw bytes (16-bit signed integers).
        in_rate: Original sample rate of the PCM audio in Hz.
        out_rate: Desired output sample rate in Hz.
        resampler: Audio resampler instance for rate conversion.

    Returns:
        μ-law encoded audio data as raw bytes at the specified output rate.
    """
    # Resample
    in_pcm_bytes = await resampler.resample(pcm_bytes, in_rate, out_rate)

    # Convert PCM to μ-law (Rust path is ~5x faster via lookup table)
    if _USE_NATIVE_AUDIO:
        out_ulaw_bytes = bytes(_native_audio.ulaw_encode(in_pcm_bytes))
    else:
        out_ulaw_bytes = audioop.lin2ulaw(in_pcm_bytes, 2)

    return out_ulaw_bytes


async def alaw_to_pcm(
    alaw_bytes: bytes, in_rate: int, out_rate: int, resampler: BaseAudioResampler
) -> bytes:
    """Convert A-law encoded audio to PCM and optionally resample.

    Args:
        alaw_bytes: A-law encoded audio data as raw bytes.
        in_rate: Original sample rate of the A-law audio in Hz.
        out_rate: Desired output sample rate in Hz.
        resampler: Audio resampler instance for rate conversion.

    Returns:
        PCM audio data as raw bytes at the specified output rate.
    """
    # Convert A-law to PCM (Rust path uses lookup table)
    if _USE_NATIVE_AUDIO:
        in_pcm_bytes = bytes(_native_audio.alaw_decode(alaw_bytes))
    else:
        in_pcm_bytes = audioop.alaw2lin(alaw_bytes, 2)

    # Resample
    out_pcm_bytes = await resampler.resample(in_pcm_bytes, in_rate, out_rate)

    return out_pcm_bytes


async def pcm_to_alaw(pcm_bytes: bytes, in_rate: int, out_rate: int, resampler: BaseAudioResampler):
    """Convert PCM audio to A-law encoding and optionally resample.

    Args:
        pcm_bytes: PCM audio data as raw bytes (16-bit signed integers).
        in_rate: Original sample rate of the PCM audio in Hz.
        out_rate: Desired output sample rate in Hz.
        resampler: Audio resampler instance for rate conversion.

    Returns:
        A-law encoded audio data as raw bytes at the specified output rate.
    """
    # Resample
    in_pcm_bytes = await resampler.resample(pcm_bytes, in_rate, out_rate)

    # Convert PCM to A-law (Rust path uses lookup table)
    if _USE_NATIVE_AUDIO:
        out_alaw_bytes = bytes(_native_audio.alaw_encode(in_pcm_bytes))
    else:
        out_alaw_bytes = audioop.lin2alaw(in_pcm_bytes, 2)

    return out_alaw_bytes


def is_silence(pcm_bytes: bytes) -> bool:
    """Determine if an audio sample contains silence by checking amplitude levels.

    When the Rust native engine is available, delegates to Rust for faster
    max-amplitude check. Falls back to numpy.

    Args:
        pcm_bytes: Raw PCM audio data as bytes (16-bit signed integers).

    Returns:
        bool: True if the audio sample is considered silence (below threshold),
              False otherwise.
    """
    if _USE_NATIVE_AUDIO:
        return _native_audio.is_silence(pcm_bytes)

    # Convert raw audio bytes to a NumPy array of int16 samples
    audio_data = np.frombuffer(pcm_bytes, dtype=np.int16)

    # Check the maximum absolute amplitude in the frame
    max_value = np.abs(audio_data).max()

    # If max value is lower than SPEAKING_THRESHOLD, consider it as silence
    return max_value <= SPEAKING_THRESHOLD
