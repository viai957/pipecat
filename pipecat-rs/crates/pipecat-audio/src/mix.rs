//! Audio mixing and stereo interleaving utilities.

/// Mix two PCM i16 audio streams by summing their samples.
///
/// Both inputs are raw little-endian i16 bytes. If the streams differ in length
/// the shorter one is implicitly zero-padded. Samples are clipped to the i16
/// range to prevent overflow.
///
/// Returns an empty `Vec` if both inputs are empty.
pub fn mix_audio(audio1: &[u8], audio2: &[u8]) -> Vec<u8> {
    let samples1 = bytes_to_i16(audio1);
    let samples2 = bytes_to_i16(audio2);

    let max_len = samples1.len().max(samples2.len());
    let mut result = Vec::with_capacity(max_len * 2);

    for i in 0..max_len {
        let s1 = *samples1.get(i).unwrap_or(&0) as i32;
        let s2 = *samples2.get(i).unwrap_or(&0) as i32;
        let mixed = (s1 + s2).clamp(-32768, 32767) as i16;
        result.extend_from_slice(&mixed.to_le_bytes());
    }

    result
}

/// Interleave two mono PCM i16 streams into a single stereo stream.
///
/// Produces interleaved (L, R, L, R, ...) output. Both inputs are truncated
/// to the shorter length if they differ.
pub fn interleave_stereo(left: &[u8], right: &[u8]) -> Vec<u8> {
    let left_samples = bytes_to_i16(left);
    let right_samples = bytes_to_i16(right);

    let min_len = left_samples.len().min(right_samples.len());
    let mut result = Vec::with_capacity(min_len * 4); // 2 channels * 2 bytes each

    for i in 0..min_len {
        result.extend_from_slice(&left_samples[i].to_le_bytes());
        result.extend_from_slice(&right_samples[i].to_le_bytes());
    }

    result
}

/// Helper: convert little-endian byte slice to Vec<i16>.
fn bytes_to_i16(data: &[u8]) -> Vec<i16> {
    data.chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn i16_to_bytes(samples: &[i16]) -> Vec<u8> {
        samples.iter().flat_map(|s| s.to_le_bytes()).collect()
    }

    #[test]
    fn mix_equal_length() {
        let a = i16_to_bytes(&[100, -200, 300]);
        let b = i16_to_bytes(&[50, 200, -100]);
        let mixed = mix_audio(&a, &b);
        let result = bytes_to_i16(&mixed);
        assert_eq!(result, vec![150, 0, 200]);
    }

    #[test]
    fn mix_different_lengths() {
        let a = i16_to_bytes(&[100, 200]);
        let b = i16_to_bytes(&[50]);
        let mixed = mix_audio(&a, &b);
        let result = bytes_to_i16(&mixed);
        assert_eq!(result, vec![150, 200]);
    }

    #[test]
    fn mix_clipping() {
        let a = i16_to_bytes(&[i16::MAX]);
        let b = i16_to_bytes(&[1000]);
        let mixed = mix_audio(&a, &b);
        let result = bytes_to_i16(&mixed);
        assert_eq!(result, vec![i16::MAX]);
    }

    #[test]
    fn mix_negative_clipping() {
        let a = i16_to_bytes(&[i16::MIN]);
        let b = i16_to_bytes(&[-1000]);
        let mixed = mix_audio(&a, &b);
        let result = bytes_to_i16(&mixed);
        assert_eq!(result, vec![i16::MIN]);
    }

    #[test]
    fn mix_both_empty() {
        let mixed = mix_audio(&[], &[]);
        assert!(mixed.is_empty());
    }

    #[test]
    fn mix_one_empty() {
        let a = i16_to_bytes(&[42]);
        let mixed = mix_audio(&a, &[]);
        let result = bytes_to_i16(&mixed);
        assert_eq!(result, vec![42]);
    }

    #[test]
    fn interleave_basic() {
        let left = i16_to_bytes(&[100, 200]);
        let right = i16_to_bytes(&[300, 400]);
        let stereo = interleave_stereo(&left, &right);
        let result = bytes_to_i16(&stereo);
        assert_eq!(result, vec![100, 300, 200, 400]);
    }

    #[test]
    fn interleave_different_lengths_truncates() {
        let left = i16_to_bytes(&[100, 200, 300]);
        let right = i16_to_bytes(&[400, 500]);
        let stereo = interleave_stereo(&left, &right);
        let result = bytes_to_i16(&stereo);
        assert_eq!(result, vec![100, 400, 200, 500]);
    }

    #[test]
    fn interleave_empty() {
        let stereo = interleave_stereo(&[], &[]);
        assert!(stereo.is_empty());
    }
}
