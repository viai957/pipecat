//! Mock VAD model for testing.
//!
//! [`MockVadModel`] returns pre-configured confidence values in sequence,
//! making it easy to write deterministic tests for the state machine without
//! requiring a real inference engine.

use crate::analyzer::VadModel;

/// A mock VAD model that returns pre-configured confidence values.
///
/// The confidence list is cycled through on each call to
/// [`voice_confidence`](VadModel::voice_confidence). Use the convenience
/// constructors [`always_speaking`](MockVadModel::always_speaking) and
/// [`always_quiet`](MockVadModel::always_quiet) for common patterns.
pub struct MockVadModel {
    num_frames: usize,
    confidences: Vec<f32>,
    index: usize,
}

impl MockVadModel {
    /// Create a mock model with custom confidence values.
    ///
    /// Args:
    ///     num_frames: Number of samples required per inference call.
    ///     confidences: Sequence of confidence values to cycle through.
    pub fn new(num_frames: usize, confidences: Vec<f32>) -> Self {
        assert!(!confidences.is_empty(), "confidences must not be empty");
        Self {
            num_frames,
            confidences,
            index: 0,
        }
    }

    /// Create a mock that always reports high confidence (speaking).
    ///
    /// Args:
    ///     num_frames: Number of samples required per inference call.
    pub fn always_speaking(num_frames: usize) -> Self {
        Self::new(num_frames, vec![0.9])
    }

    /// Create a mock that always reports zero confidence (quiet).
    ///
    /// Args:
    ///     num_frames: Number of samples required per inference call.
    pub fn always_quiet(num_frames: usize) -> Self {
        Self::new(num_frames, vec![0.0])
    }
}

impl VadModel for MockVadModel {
    fn num_frames_required(&self) -> usize {
        self.num_frames
    }

    fn voice_confidence(&mut self, _buffer: &[i16]) -> f32 {
        let c = self.confidences[self.index % self.confidences.len()];
        self.index += 1;
        c
    }

    fn reset_states(&mut self) {
        self.index = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn always_speaking_returns_high_confidence() {
        let mut model = MockVadModel::always_speaking(512);
        let dummy = vec![0i16; 512];
        for _ in 0..10 {
            assert!(model.voice_confidence(&dummy) > 0.5);
        }
    }

    #[test]
    fn always_quiet_returns_zero() {
        let mut model = MockVadModel::always_quiet(512);
        let dummy = vec![0i16; 512];
        for _ in 0..10 {
            assert!((model.voice_confidence(&dummy)).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn cycles_through_values() {
        let mut model = MockVadModel::new(256, vec![0.1, 0.5, 0.9]);
        let dummy = vec![0i16; 256];
        assert!((model.voice_confidence(&dummy) - 0.1).abs() < f32::EPSILON);
        assert!((model.voice_confidence(&dummy) - 0.5).abs() < f32::EPSILON);
        assert!((model.voice_confidence(&dummy) - 0.9).abs() < f32::EPSILON);
        // Wraps around
        assert!((model.voice_confidence(&dummy) - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn reset_restarts_cycle() {
        let mut model = MockVadModel::new(256, vec![0.1, 0.9]);
        let dummy = vec![0i16; 256];
        model.voice_confidence(&dummy);
        model.voice_confidence(&dummy);
        model.reset_states();
        assert!((model.voice_confidence(&dummy) - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn num_frames_required_matches() {
        let model = MockVadModel::always_speaking(512);
        assert_eq!(model.num_frames_required(), 512);
    }

    #[test]
    #[should_panic(expected = "confidences must not be empty")]
    fn empty_confidences_panics() {
        MockVadModel::new(160, vec![]);
    }
}
