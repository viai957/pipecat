//! PyO3 bindings for Rust audio processing utilities.
//!
//! Exposes high-performance audio functions to Python:
//! - Volume calculation (RMS-based, replaces expensive pyln.Meter)
//! - Silence detection (max-amplitude threshold)
//! - G.711 mu-law/A-law codec (lookup-table based)
//! - Exponential smoothing

use pyo3::prelude::*;

/// Calculate audio volume using RMS, normalized to [0, 1].
///
/// This replaces the Python `pyln.Meter` approach which allocates a new
/// Meter object on every call. The Rust version uses a simple RMS calculation
/// with dB-scale normalization — equivalent output for VAD thresholding.
#[pyfunction]
pub fn calculate_audio_volume(audio: &[u8], sample_rate: u32) -> f32 {
    // Interpret raw bytes as i16 PCM samples (little-endian)
    let samples: &[i16] = unsafe {
        std::slice::from_raw_parts(audio.as_ptr() as *const i16, audio.len() / 2)
    };
    pipecat_audio::calculate_audio_volume(samples, sample_rate)
}

/// Check whether PCM audio bytes represent silence.
///
/// Returns true if max absolute amplitude <= SPEAKING_THRESHOLD (20).
#[pyfunction]
pub fn is_silence(audio: &[u8]) -> bool {
    let samples: &[i16] = unsafe {
        std::slice::from_raw_parts(audio.as_ptr() as *const i16, audio.len() / 2)
    };
    pipecat_audio::is_silence(samples)
}

/// Apply exponential smoothing: prev + factor * (value - prev).
#[pyfunction]
pub fn exp_smoothing(value: f32, prev_value: f32, factor: f32) -> f32 {
    pipecat_audio::exp_smoothing(value, prev_value, factor)
}

/// Decode mu-law bytes to PCM i16 bytes (little-endian).
#[pyfunction]
pub fn ulaw_decode(input: &[u8]) -> Vec<u8> {
    let decoded = pipecat_audio::ulaw_decode(input);
    // Convert Vec<i16> to bytes (little-endian)
    let mut output = Vec::with_capacity(decoded.len() * 2);
    for sample in &decoded {
        output.extend_from_slice(&sample.to_le_bytes());
    }
    output
}

/// Encode PCM i16 bytes (little-endian) to mu-law bytes.
#[pyfunction]
pub fn ulaw_encode(input: &[u8]) -> Vec<u8> {
    let samples: &[i16] = unsafe {
        std::slice::from_raw_parts(input.as_ptr() as *const i16, input.len() / 2)
    };
    pipecat_audio::ulaw_encode(samples)
}

/// Decode A-law bytes to PCM i16 bytes (little-endian).
#[pyfunction]
pub fn alaw_decode(input: &[u8]) -> Vec<u8> {
    let decoded = pipecat_audio::alaw_decode(input);
    let mut output = Vec::with_capacity(decoded.len() * 2);
    for sample in &decoded {
        output.extend_from_slice(&sample.to_le_bytes());
    }
    output
}

/// Encode PCM i16 bytes (little-endian) to A-law bytes.
#[pyfunction]
pub fn alaw_encode(input: &[u8]) -> Vec<u8> {
    let samples: &[i16] = unsafe {
        std::slice::from_raw_parts(input.as_ptr() as *const i16, input.len() / 2)
    };
    pipecat_audio::alaw_encode(samples)
}

/// Register all audio utility functions into a Python submodule.
pub fn register_audio_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let audio_m = PyModule::new_bound(parent.py(), "audio")?;
    audio_m.add_function(wrap_pyfunction!(calculate_audio_volume, &audio_m)?)?;
    audio_m.add_function(wrap_pyfunction!(is_silence, &audio_m)?)?;
    audio_m.add_function(wrap_pyfunction!(exp_smoothing, &audio_m)?)?;
    audio_m.add_function(wrap_pyfunction!(ulaw_decode, &audio_m)?)?;
    audio_m.add_function(wrap_pyfunction!(ulaw_encode, &audio_m)?)?;
    audio_m.add_function(wrap_pyfunction!(alaw_decode, &audio_m)?)?;
    audio_m.add_function(wrap_pyfunction!(alaw_encode, &audio_m)?)?;
    parent.add_submodule(&audio_m)?;
    Ok(())
}
