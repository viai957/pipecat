//! VAD parameters and state definitions.
//!
//! Ported from Python's `pipecat.audio.vad.vad_analyzer`.

/// Default confidence threshold for voice detection.
pub const VAD_CONFIDENCE: f32 = 0.7;
/// Default duration (seconds) to wait before confirming voice start.
pub const VAD_START_SECS: f32 = 0.2;
/// Default duration (seconds) to wait before confirming voice stop.
pub const VAD_STOP_SECS: f32 = 0.2;
/// Default minimum audio volume threshold for voice detection.
pub const VAD_MIN_VOLUME: f32 = 0.6;

/// Voice Activity Detection states.
///
/// The state machine transitions through these states based on consecutive
/// frames of speech or silence, providing hysteresis to reject flickers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadState {
    /// No voice activity detected.
    Quiet,
    /// Voice activity beginning — transitioning from quiet.
    Starting,
    /// Active voice detected and confirmed.
    Speaking,
    /// Voice activity ending — transitioning to quiet.
    Stopping,
}

/// Configuration parameters for Voice Activity Detection.
///
/// These mirror the Python `VADParams` model and control the sensitivity
/// and timing of voice activity detection.
#[derive(Debug, Clone)]
pub struct VadParams {
    /// Minimum confidence threshold for voice detection (0.0 – 1.0).
    pub confidence: f32,
    /// Duration in seconds of consecutive speech needed to confirm start.
    pub start_secs: f32,
    /// Duration in seconds of consecutive silence needed to confirm stop.
    pub stop_secs: f32,
    /// Minimum smoothed audio volume required to consider speech.
    pub min_volume: f32,
}

impl Default for VadParams {
    fn default() -> Self {
        Self {
            confidence: VAD_CONFIDENCE,
            start_secs: VAD_START_SECS,
            stop_secs: VAD_STOP_SECS,
            min_volume: VAD_MIN_VOLUME,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_params() {
        let params = VadParams::default();
        assert!((params.confidence - 0.7).abs() < f32::EPSILON);
        assert!((params.start_secs - 0.2).abs() < f32::EPSILON);
        assert!((params.stop_secs - 0.2).abs() < f32::EPSILON);
        assert!((params.min_volume - 0.6).abs() < f32::EPSILON);
    }

    #[test]
    fn vad_state_equality() {
        assert_eq!(VadState::Quiet, VadState::Quiet);
        assert_ne!(VadState::Quiet, VadState::Speaking);
    }

    #[test]
    fn vad_state_copy() {
        let s = VadState::Speaking;
        let s2 = s;
        assert_eq!(s, s2);
    }
}
