//! Silero VAD model stub.
//!
//! The Silero VAD model uses ONNX Runtime (`ort`) for inference. This module
//! provides the type definition and documents the expected tensor layout, but
//! the actual inference is stubbed out until the `ort` crate dependency is
//! added.
//!
//! # Model details
//!
//! The Silero V5 ONNX model expects:
//!
//! | Sample rate | Samples per call | Context size |
//! |-------------|------------------|--------------|
//! | 16 000 Hz   | 512              | 64           |
//! | 8 000 Hz    | 256              | 32           |
//!
//! Internal state tensors: `h` and `c` — each shaped `[2, 1, 128]` (f32).
//!
//! The model should be reset every ~5 seconds of audio to prevent state drift.

use crate::analyzer::VadModel;

/// Silero VAD model using ONNX Runtime.
///
/// This is a placeholder implementation. To complete it:
///
/// 1. Add `ort = { workspace = true }` to `Cargo.toml` dependencies.
/// 2. Download the Silero ONNX model file (`silero_vad.onnx`).
/// 3. Replace the stub methods below with real ONNX session creation and
///    inference calls.
///
/// The model expects mono PCM i16 audio at 8 kHz or 16 kHz.
#[derive(Debug)]
pub struct SileroVadModel {
    sample_rate: u32,
    // When `ort` is available, add:
    // session: ort::Session,
    // h_tensor: ndarray::Array3<f32>,  // [2, 1, 128]
    // c_tensor: ndarray::Array3<f32>,  // [2, 1, 128]
    // context: Vec<f32>,
    // call_count: usize,
}

impl SileroVadModel {
    /// Create a new Silero VAD model stub.
    ///
    /// Args:
    ///     sample_rate: Audio sample rate. Must be 8000 or 16000.
    ///
    /// Returns an error if the sample rate is unsupported.
    pub fn new(sample_rate: u32) -> Result<Self, String> {
        if sample_rate != 16000 && sample_rate != 8000 {
            return Err(format!(
                "Silero VAD requires 16000 or 8000 sample rate, got {}",
                sample_rate
            ));
        }
        Ok(Self { sample_rate })
    }

    /// Return the sample rate this model was configured for.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

impl VadModel for SileroVadModel {
    fn num_frames_required(&self) -> usize {
        if self.sample_rate == 16000 {
            512
        } else {
            256
        }
    }

    fn voice_confidence(&mut self, _buffer: &[i16]) -> f32 {
        // Stub: always return 0.0 (quiet).
        // Real implementation would:
        // 1. Normalize i16 samples to f32 in [-1, 1]
        // 2. Prepend context samples
        // 3. Run ONNX session with input, h, c, sample_rate tensors
        // 4. Extract confidence from output, update h/c state
        // 5. Rotate context window
        // 6. Periodically reset state (every ~5 seconds)
        0.0
    }

    fn reset_states(&mut self) {
        // Stub: nothing to reset.
        // Real implementation would zero out h_tensor, c_tensor, and context.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_valid_sample_rates() {
        assert!(SileroVadModel::new(16000).is_ok());
        assert!(SileroVadModel::new(8000).is_ok());
    }

    #[test]
    fn new_invalid_sample_rate() {
        let result = SileroVadModel::new(44100);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("44100"));
    }

    #[test]
    fn num_frames_16khz() {
        let model = SileroVadModel::new(16000).unwrap();
        assert_eq!(model.num_frames_required(), 512);
    }

    #[test]
    fn num_frames_8khz() {
        let model = SileroVadModel::new(8000).unwrap();
        assert_eq!(model.num_frames_required(), 256);
    }

    #[test]
    fn stub_returns_zero_confidence() {
        let mut model = SileroVadModel::new(16000).unwrap();
        let buffer = vec![0i16; 512];
        assert!((model.voice_confidence(&buffer)).abs() < f32::EPSILON);
    }

    #[test]
    fn sample_rate_accessor() {
        let model = SileroVadModel::new(8000).unwrap();
        assert_eq!(model.sample_rate(), 8000);
    }
}
