//! Audio resampling using rubato (SIMD-accelerated sinc interpolation).
//!
//! Provides [`StreamResampler`] for real-time streaming use and [`FileResampler`]
//! for batch processing of complete audio buffers.

use pipecat_core::error::PipecatError;
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

/// A streaming audio resampler that maintains state across successive chunks.
///
/// Wraps `rubato::SincFixedIn` with SIMD acceleration. Designed for real-time
/// pipelines where audio arrives in small chunks.
pub struct StreamResampler {
    resampler: SincFixedIn<f32>,
    num_channels: usize,
    /// Leftover samples from a previous call that didn't fill a complete input chunk.
    remainder: Vec<Vec<f32>>,
}

impl StreamResampler {
    /// Create a new streaming resampler.
    ///
    /// Args:
    ///     from_rate: Source sample rate in Hz.
    ///     to_rate: Target sample rate in Hz.
    ///     num_channels: Number of audio channels (e.g. 1 for mono).
    pub fn new(from_rate: u32, to_rate: u32, num_channels: u16) -> Result<Self, PipecatError> {
        if from_rate == 0 || to_rate == 0 {
            return Err(PipecatError::Audio(
                "sample rates must be non-zero".into(),
            ));
        }
        if num_channels == 0 {
            return Err(PipecatError::Audio(
                "num_channels must be non-zero".into(),
            ));
        }

        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 256,
            window: WindowFunction::BlackmanHarris2,
        };

        let chunk_size = 1024;
        let ratio = to_rate as f64 / from_rate as f64;
        let channels = num_channels as usize;

        let resampler = SincFixedIn::new(ratio, 2.0, params, chunk_size, channels)
            .map_err(|e| PipecatError::Audio(format!("failed to create resampler: {e}")))?;

        let remainder = vec![Vec::new(); channels];

        Ok(Self {
            resampler,
            num_channels: channels,
            remainder,
        })
    }

    /// Resample a chunk of PCM i16 audio (little-endian bytes).
    ///
    /// Maintains internal state so successive calls produce a continuous output stream.
    /// Returns resampled PCM i16 bytes.
    pub fn resample(&mut self, input: &[u8]) -> Result<Vec<u8>, PipecatError> {
        if input.is_empty() {
            return Ok(Vec::new());
        }
        if !input.len().is_multiple_of(2) {
            return Err(PipecatError::Audio(
                "input length must be a multiple of 2 (i16 samples)".into(),
            ));
        }

        // Convert i16 LE bytes -> f32 samples, deinterleave into per-channel vecs.
        let samples: Vec<f32> = input
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0)
            .collect();

        let num_channels = self.num_channels;
        for (i, &s) in samples.iter().enumerate() {
            self.remainder[i % num_channels].push(s);
        }

        let chunk_size = self.resampler.input_frames_max();
        let mut output_all: Vec<Vec<f32>> = vec![Vec::new(); num_channels];

        // Process as many full chunks as possible.
        while self.remainder[0].len() >= chunk_size {
            let mut chunk: Vec<Vec<f32>> = Vec::with_capacity(num_channels);
            for ch in &mut self.remainder {
                let rest = ch.split_off(chunk_size);
                chunk.push(std::mem::replace(ch, rest));
            }

            let resampled = self
                .resampler
                .process(&chunk, None)
                .map_err(|e| PipecatError::Audio(format!("resample error: {e}")))?;

            for (ch_idx, ch_data) in resampled.into_iter().enumerate() {
                output_all[ch_idx].extend(ch_data.into_iter());
            }
        }

        // Interleave output channels back to i16 LE bytes.
        let out_len = output_all[0].len();
        let mut result = Vec::with_capacity(out_len * num_channels * 2);
        for i in 0..out_len {
            for ch in &output_all {
                let sample = (ch[i] * 32768.0).clamp(-32768.0, 32767.0) as i16;
                result.extend_from_slice(&sample.to_le_bytes());
            }
        }

        Ok(result)
    }
}

/// A batch (non-streaming) audio resampler for processing complete audio buffers.
///
/// Unlike [`StreamResampler`], this does not maintain state between calls — each
/// invocation processes an independent audio buffer.
pub struct FileResampler {
    from_rate: u32,
    to_rate: u32,
    num_channels: u16,
}

impl FileResampler {
    /// Create a new file resampler.
    ///
    /// Args:
    ///     from_rate: Source sample rate in Hz.
    ///     to_rate: Target sample rate in Hz.
    ///     num_channels: Number of audio channels.
    pub fn new(from_rate: u32, to_rate: u32, num_channels: u16) -> Result<Self, PipecatError> {
        if from_rate == 0 || to_rate == 0 {
            return Err(PipecatError::Audio(
                "sample rates must be non-zero".into(),
            ));
        }
        if num_channels == 0 {
            return Err(PipecatError::Audio(
                "num_channels must be non-zero".into(),
            ));
        }
        Ok(Self {
            from_rate,
            to_rate,
            num_channels,
        })
    }

    /// Resample a complete audio buffer (PCM i16 LE bytes).
    ///
    /// Each call is independent — no state is carried between calls.
    pub fn resample(&self, input: &[u8]) -> Result<Vec<u8>, PipecatError> {
        if input.is_empty() {
            return Ok(Vec::new());
        }
        if self.from_rate == self.to_rate {
            return Ok(input.to_vec());
        }
        if !input.len().is_multiple_of(2) {
            return Err(PipecatError::Audio(
                "input length must be a multiple of 2 (i16 samples)".into(),
            ));
        }

        let num_channels = self.num_channels as usize;
        let samples: Vec<f32> = input
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0)
            .collect();

        let num_frames = samples.len() / num_channels;

        // Deinterleave into per-channel vecs.
        let mut channels: Vec<Vec<f32>> = vec![Vec::with_capacity(num_frames); num_channels];
        for (i, &s) in samples.iter().enumerate() {
            channels[i % num_channels].push(s);
        }

        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 256,
            window: WindowFunction::BlackmanHarris2,
        };

        let ratio = self.to_rate as f64 / self.from_rate as f64;
        let chunk_size = num_frames;

        let mut resampler = SincFixedIn::new(ratio, 2.0, params, chunk_size, num_channels)
            .map_err(|e| PipecatError::Audio(format!("failed to create resampler: {e}")))?;

        let resampled = resampler
            .process(&channels, None)
            .map_err(|e| PipecatError::Audio(format!("resample error: {e}")))?;

        // Interleave back to i16 LE bytes.
        let out_len = resampled[0].len();
        let mut result = Vec::with_capacity(out_len * num_channels * 2);
        for i in 0..out_len {
            for ch in &resampled {
                let sample = (ch[i] * 32768.0).clamp(-32768.0, 32767.0) as i16;
                result.extend_from_slice(&sample.to_le_bytes());
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_resampler_empty_input() {
        let mut resampler = StreamResampler::new(16000, 8000, 1).unwrap();
        let out = resampler.resample(&[]).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn stream_resampler_odd_bytes_rejected() {
        let mut resampler = StreamResampler::new(16000, 8000, 1).unwrap();
        assert!(resampler.resample(&[0, 1, 2]).is_err());
    }

    #[test]
    fn stream_resampler_zero_rate_rejected() {
        assert!(StreamResampler::new(0, 8000, 1).is_err());
        assert!(StreamResampler::new(8000, 0, 1).is_err());
    }

    #[test]
    fn stream_resampler_zero_channels_rejected() {
        assert!(StreamResampler::new(16000, 8000, 0).is_err());
    }

    #[test]
    fn stream_resampler_produces_output() {
        let mut resampler = StreamResampler::new(16000, 8000, 1).unwrap();
        // Feed a large enough chunk to produce output (>= chunk_size frames).
        let input: Vec<u8> = (0..2048)
            .flat_map(|i| {
                let sample = (((i as f32) / 2048.0 * std::f32::consts::TAU).sin() * 16000.0)
                    as i16;
                sample.to_le_bytes()
            })
            .collect();
        let out = resampler.resample(&input).unwrap();
        // Downsampling 2:1 should produce roughly half the frames.
        assert!(!out.is_empty());
        // Output bytes must be even (i16 samples).
        assert_eq!(out.len() % 2, 0);
    }

    #[test]
    fn file_resampler_same_rate_passthrough() {
        let resampler = FileResampler::new(16000, 16000, 1).unwrap();
        let input: Vec<u8> = vec![0x00, 0x10, 0xFF, 0x7F];
        let out = resampler.resample(&input).unwrap();
        assert_eq!(out, input);
    }

    #[test]
    fn file_resampler_empty_input() {
        let resampler = FileResampler::new(16000, 8000, 1).unwrap();
        let out = resampler.resample(&[]).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn file_resampler_odd_bytes_rejected() {
        let resampler = FileResampler::new(16000, 8000, 1).unwrap();
        assert!(resampler.resample(&[0, 1, 2]).is_err());
    }
}
