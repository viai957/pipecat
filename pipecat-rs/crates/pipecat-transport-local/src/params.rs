//! Configuration parameters for the local audio transport.

use pipecat_transport_traits::TransportParams;

/// Parameters for the local audio transport backed by cpal.
///
/// Embeds the common [`TransportParams`] and adds local-specific settings
/// such as device name selection and ring buffer sizing.
#[derive(Debug, Clone)]
pub struct LocalAudioTransportParams {
    /// Common audio/video transport parameters (sample rates, channels, etc.).
    pub base: TransportParams,
    /// Name of the input device to use. `None` selects the system default.
    pub input_device_name: Option<String>,
    /// Name of the output device to use. `None` selects the system default.
    pub output_device_name: Option<String>,
    /// Capacity of the lock-free ring buffer in *bytes*.
    ///
    /// This buffer sits between the real-time cpal audio thread and the
    /// async tokio task. Larger values add latency but tolerate scheduling
    /// jitter. The default (8192 bytes) is approximately 256 ms of mono
    /// 16-bit 16 kHz audio.
    pub ring_buffer_size: usize,
}

impl Default for LocalAudioTransportParams {
    fn default() -> Self {
        Self {
            base: TransportParams::default(),
            input_device_name: None,
            output_device_name: None,
            ring_buffer_size: 8192,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_params() {
        let p = LocalAudioTransportParams::default();
        assert_eq!(p.base.audio_in_sample_rate, 16000);
        assert_eq!(p.base.audio_in_channels, 1);
        assert_eq!(p.base.audio_out_sample_rate, 24000);
        assert_eq!(p.base.audio_out_channels, 1);
        assert!(p.input_device_name.is_none());
        assert!(p.output_device_name.is_none());
        assert_eq!(p.ring_buffer_size, 8192);
    }

    #[test]
    fn custom_params() {
        let p = LocalAudioTransportParams {
            base: TransportParams {
                audio_in_sample_rate: 48000,
                audio_in_channels: 2,
                audio_out_sample_rate: 48000,
                audio_out_channels: 2,
                ..Default::default()
            },
            input_device_name: Some("Blue Yeti".into()),
            output_device_name: Some("Speakers".into()),
            ring_buffer_size: 16384,
        };
        assert_eq!(p.base.audio_in_sample_rate, 48000);
        assert_eq!(p.base.audio_in_channels, 2);
        assert_eq!(p.input_device_name.as_deref(), Some("Blue Yeti"));
        assert_eq!(p.output_device_name.as_deref(), Some("Speakers"));
        assert_eq!(p.ring_buffer_size, 16384);
    }

    #[test]
    fn clone_params() {
        let p = LocalAudioTransportParams::default();
        let p2 = p.clone();
        assert_eq!(p2.ring_buffer_size, p.ring_buffer_size);
        assert_eq!(p2.base.audio_in_sample_rate, p.base.audio_in_sample_rate);
    }
}
