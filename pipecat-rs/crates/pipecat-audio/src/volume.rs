//! Audio volume calculation and smoothing utilities.
//!
//! Provides a simplified EBU R128 loudness calculation, exponential smoothing,
//! and value normalization — ported from Python's `audio/utils.py`.

/// Amplitude threshold below which audio is considered silence.
///
/// Normal speech typically produces sample amplitudes between +/-500 to +/-5000.
/// This threshold is set well below that range to reliably distinguish silence
/// from speech.
pub const SPEAKING_THRESHOLD: i16 = 20;

/// Calculate a simplified loudness level for an audio buffer.
///
/// Computes RMS of the i16 samples, converts to a dB-like scale, and normalizes
/// to the [0, 1] range using the -20 to 80 loudness range (matching the Python
/// implementation's normalization).
///
/// Args:
///     audio: Slice of PCM i16 samples.
///     _sample_rate: Sample rate in Hz (reserved for future EBU R128 gating).
///
/// Returns:
///     Normalized loudness value between 0.0 (quiet) and 1.0 (loud).
pub fn calculate_audio_volume(audio: &[i16], _sample_rate: u32) -> f32 {
    if audio.is_empty() {
        return 0.0;
    }

    // Compute RMS in f64 for precision.
    let sum_sq: f64 = audio.iter().map(|&s| (s as f64) * (s as f64)).sum();
    let rms = (sum_sq / audio.len() as f64).sqrt();

    // Convert RMS to dB-like loudness. Use a small epsilon to avoid log(0).
    let loudness = if rms < 1e-10 {
        -80.0
    } else {
        20.0 * (rms / 32768.0).log10()
    };

    // Normalize using the same -20..80 range as the Python code.
    normalize_value(loudness, -20.0, 80.0) as f32
}

/// Apply exponential smoothing to a value.
///
/// `smoothed = prev_value + factor * (value - prev_value)`
///
/// A higher `factor` gives more weight to the new value.
///
/// Args:
///     value: The new value to incorporate.
///     prev_value: The previous smoothed value.
///     factor: Smoothing factor in [0, 1].
pub fn exp_smoothing(value: f32, prev_value: f32, factor: f32) -> f32 {
    prev_value + factor * (value - prev_value)
}

/// Normalize a value from [min, max] to [0, 1], clamping to bounds.
pub fn normalize_value(value: f64, min_value: f64, max_value: f64) -> f64 {
    let normalized = (value - min_value) / (max_value - min_value);
    normalized.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_value_basics() {
        assert!((normalize_value(50.0, 0.0, 100.0) - 0.5).abs() < 1e-10);
        assert!((normalize_value(0.0, 0.0, 100.0) - 0.0).abs() < 1e-10);
        assert!((normalize_value(100.0, 0.0, 100.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn normalize_value_clamps() {
        assert!((normalize_value(-50.0, 0.0, 100.0) - 0.0).abs() < 1e-10);
        assert!((normalize_value(200.0, 0.0, 100.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn exp_smoothing_identity() {
        // When factor is 0, output equals previous value.
        assert!((exp_smoothing(100.0, 50.0, 0.0) - 50.0).abs() < 1e-6);
        // When factor is 1, output equals new value.
        assert!((exp_smoothing(100.0, 50.0, 1.0) - 100.0).abs() < 1e-6);
    }

    #[test]
    fn exp_smoothing_midpoint() {
        assert!((exp_smoothing(100.0, 0.0, 0.5) - 50.0).abs() < 1e-6);
    }

    #[test]
    fn calculate_audio_volume_silence() {
        let silence = vec![0i16; 1600];
        let vol = calculate_audio_volume(&silence, 16000);
        assert!(vol < 0.01, "silence should have near-zero volume, got {vol}");
    }

    #[test]
    fn calculate_audio_volume_loud() {
        let loud: Vec<i16> = vec![i16::MAX; 1600];
        let vol = calculate_audio_volume(&loud, 16000);
        assert!(vol > 0.0, "loud signal should have positive volume, got {vol}");
    }

    #[test]
    fn calculate_audio_volume_empty() {
        assert!((calculate_audio_volume(&[], 16000) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn speaking_threshold_value() {
        assert_eq!(SPEAKING_THRESHOLD, 20);
    }
}
