//! Transport configuration parameters.
//!
//! [`TransportParams`] mirrors Python's `TransportParams` dataclass, providing
//! sensible defaults for audio/video input and output.

use serde::{Deserialize, Serialize};

/// Configuration parameters for input and output transports.
///
/// Controls sample rates, channel counts, chunk sizes, and which media
/// streams are enabled. All concrete transport implementations read from
/// this struct during initialization.
///
/// Parameters:
///     audio_in_enabled: Whether audio input capture is active (default `true`).
///     audio_in_sample_rate: Sample rate for input audio in Hz (default 16000).
///     audio_in_channels: Number of input audio channels (default 1).
///     audio_in_passthrough: Whether raw input audio is passed through to the
///         pipeline without processing (default `true`).
///     audio_out_enabled: Whether audio output playback is active (default `true`).
///     audio_out_sample_rate: Sample rate for output audio in Hz (default 24000).
///     audio_out_channels: Number of output audio channels (default 1).
///     audio_out_10ms_chunks: Number of 10ms chunks per output audio frame
///         (default 2, i.e. 20ms frames).
///     audio_out_destinations: Named destinations for audio output routing.
///     video_in_enabled: Whether video input is active (default `false`).
///     video_out_enabled: Whether video output is active (default `false`).
///     video_out_width: Output video width in pixels (default 1024).
///     video_out_height: Output video height in pixels (default 768).
///     video_out_framerate: Output video framerate in FPS (default 30).
///     video_out_destinations: Named destinations for video output routing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransportParams {
    /// Whether audio input capture is active.
    pub audio_in_enabled: bool,
    /// Sample rate for input audio in Hz.
    pub audio_in_sample_rate: u32,
    /// Number of input audio channels.
    pub audio_in_channels: u16,
    /// Whether raw input audio is passed through without processing.
    pub audio_in_passthrough: bool,
    /// Whether audio output playback is active.
    pub audio_out_enabled: bool,
    /// Sample rate for output audio in Hz.
    pub audio_out_sample_rate: u32,
    /// Number of output audio channels.
    pub audio_out_channels: u16,
    /// Number of 10ms chunks per output audio frame.
    pub audio_out_10ms_chunks: u32,
    /// Named destinations for audio output routing.
    pub audio_out_destinations: Vec<String>,
    /// Whether video input is active.
    pub video_in_enabled: bool,
    /// Whether video output is active.
    pub video_out_enabled: bool,
    /// Output video width in pixels.
    pub video_out_width: u32,
    /// Output video height in pixels.
    pub video_out_height: u32,
    /// Output video framerate in FPS.
    pub video_out_framerate: u32,
    /// Named destinations for video output routing.
    pub video_out_destinations: Vec<String>,
}

impl Default for TransportParams {
    fn default() -> Self {
        Self {
            audio_in_enabled: true,
            audio_in_sample_rate: 16000,
            audio_in_channels: 1,
            audio_in_passthrough: true,
            audio_out_enabled: true,
            audio_out_sample_rate: 24000,
            audio_out_channels: 1,
            audio_out_10ms_chunks: 2,
            audio_out_destinations: Vec::new(),
            video_in_enabled: false,
            video_out_enabled: false,
            video_out_width: 1024,
            video_out_height: 768,
            video_out_framerate: 30,
            video_out_destinations: Vec::new(),
        }
    }
}

impl TransportParams {
    /// Calculate the number of bytes in a single 10ms audio output chunk.
    ///
    /// Formula: `(sample_rate / 100) * channels * 2` bytes (16-bit PCM).
    pub fn audio_out_bytes_per_10ms(&self) -> usize {
        (self.audio_out_sample_rate as usize / 100) * self.audio_out_channels as usize * 2
    }

    /// Calculate the total audio output chunk size in bytes, accounting for
    /// the configured number of 10ms chunks.
    ///
    /// Formula: `audio_out_bytes_per_10ms * audio_out_10ms_chunks`.
    pub fn audio_out_chunk_size(&self) -> usize {
        self.audio_out_bytes_per_10ms() * self.audio_out_10ms_chunks as usize
    }

    /// Calculate the number of bytes in a single 10ms audio input chunk.
    ///
    /// Formula: `(sample_rate / 100) * channels * 2` bytes (16-bit PCM).
    pub fn audio_in_bytes_per_10ms(&self) -> usize {
        (self.audio_in_sample_rate as usize / 100) * self.audio_in_channels as usize * 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_audio_in() {
        let p = TransportParams::default();
        assert!(p.audio_in_enabled);
        assert_eq!(p.audio_in_sample_rate, 16000);
        assert_eq!(p.audio_in_channels, 1);
        assert!(p.audio_in_passthrough);
    }

    #[test]
    fn default_audio_out() {
        let p = TransportParams::default();
        assert!(p.audio_out_enabled);
        assert_eq!(p.audio_out_sample_rate, 24000);
        assert_eq!(p.audio_out_channels, 1);
        assert_eq!(p.audio_out_10ms_chunks, 2);
        assert!(p.audio_out_destinations.is_empty());
    }

    #[test]
    fn default_video_disabled() {
        let p = TransportParams::default();
        assert!(!p.video_in_enabled);
        assert!(!p.video_out_enabled);
        assert_eq!(p.video_out_width, 1024);
        assert_eq!(p.video_out_height, 768);
        assert_eq!(p.video_out_framerate, 30);
        assert!(p.video_out_destinations.is_empty());
    }

    #[test]
    fn audio_out_bytes_per_10ms_default() {
        let p = TransportParams::default();
        // 24000 / 100 * 1 * 2 = 480 bytes
        assert_eq!(p.audio_out_bytes_per_10ms(), 480);
    }

    #[test]
    fn audio_out_chunk_size_default() {
        let p = TransportParams::default();
        // 480 * 2 = 960 bytes (20ms at 24kHz mono 16-bit)
        assert_eq!(p.audio_out_chunk_size(), 960);
    }

    #[test]
    fn audio_out_chunk_size_custom() {
        let p = TransportParams {
            audio_out_sample_rate: 48000,
            audio_out_channels: 2,
            audio_out_10ms_chunks: 3,
            ..Default::default()
        };
        // 48000/100 * 2 * 2 = 1920 bytes per 10ms
        // 1920 * 3 = 5760 bytes
        assert_eq!(p.audio_out_bytes_per_10ms(), 1920);
        assert_eq!(p.audio_out_chunk_size(), 5760);
    }

    #[test]
    fn audio_in_bytes_per_10ms_default() {
        let p = TransportParams::default();
        // 16000 / 100 * 1 * 2 = 320 bytes
        assert_eq!(p.audio_in_bytes_per_10ms(), 320);
    }

    #[test]
    fn serde_roundtrip() {
        let p = TransportParams::default();
        let json = serde_json::to_string(&p).unwrap();
        let p2: TransportParams = serde_json::from_str(&json).unwrap();
        assert_eq!(p.audio_in_sample_rate, p2.audio_in_sample_rate);
        assert_eq!(p.audio_out_sample_rate, p2.audio_out_sample_rate);
        assert_eq!(p.video_out_width, p2.video_out_width);
    }
}
