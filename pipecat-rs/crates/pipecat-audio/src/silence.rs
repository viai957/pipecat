//! Silence detection for PCM audio.

use crate::volume::SPEAKING_THRESHOLD;

/// Check whether a PCM i16 audio buffer is silence.
///
/// Returns `true` if the maximum absolute amplitude of all samples is at or
/// below [`SPEAKING_THRESHOLD`], or if the buffer is empty.
pub fn is_silence(pcm: &[i16]) -> bool {
    if pcm.is_empty() {
        return true;
    }
    let max_abs = pcm.iter().map(|&s| (s as i32).unsigned_abs()).max().unwrap_or(0);
    max_abs <= SPEAKING_THRESHOLD as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_is_silence() {
        assert!(is_silence(&[]));
    }

    #[test]
    fn all_zeros_is_silence() {
        assert!(is_silence(&[0; 160]));
    }

    #[test]
    fn below_threshold_is_silence() {
        let samples: Vec<i16> = vec![10, -15, 5, 20, -20, 0];
        assert!(is_silence(&samples));
    }

    #[test]
    fn above_threshold_is_not_silence() {
        let samples: Vec<i16> = vec![0, 0, 21, 0];
        assert!(!is_silence(&samples));
    }

    #[test]
    fn negative_above_threshold() {
        let samples: Vec<i16> = vec![0, -21, 0];
        assert!(!is_silence(&samples));
    }

    #[test]
    fn loud_speech() {
        let samples: Vec<i16> = vec![1000, -2000, 3000];
        assert!(!is_silence(&samples));
    }

    #[test]
    fn min_i16_is_not_silence() {
        assert!(!is_silence(&[i16::MIN]));
    }
}
