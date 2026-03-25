//! Unified local audio transport.
//!
//! [`LocalAudioTransport`] is a convenience wrapper that creates matched
//! [`LocalAudioInput`] and [`LocalAudioOutput`] processors sharing the
//! same [`LocalAudioTransportParams`].

use crate::input::LocalAudioInput;
use crate::output::LocalAudioOutput;
use crate::params::LocalAudioTransportParams;

/// Unified local audio transport factory.
///
/// Creates matched input and output processors that share the same
/// configuration parameters. Use this when you need both microphone
/// capture and speaker playback in a single pipeline.
///
/// ```rust,no_run
/// use pipecat_transport_local::{LocalAudioTransport, LocalAudioTransportParams};
///
/// let transport = LocalAudioTransport::new(LocalAudioTransportParams::default());
/// let input = transport.input();   // microphone capture
/// let output = transport.output(); // speaker playback
/// // Add input and output as processors in your pipeline.
/// ```
pub struct LocalAudioTransport {
    params: LocalAudioTransportParams,
}

impl LocalAudioTransport {
    /// Create a new local audio transport with the given parameters.
    pub fn new(params: LocalAudioTransportParams) -> Self {
        Self { params }
    }

    /// Create a [`LocalAudioInput`] processor for microphone capture.
    ///
    /// Each call returns a new, independent processor instance.
    pub fn input(&self) -> LocalAudioInput {
        LocalAudioInput::new(self.params.clone())
    }

    /// Create a [`LocalAudioOutput`] processor for speaker playback.
    ///
    /// Each call returns a new, independent processor instance.
    pub fn output(&self) -> LocalAudioOutput {
        LocalAudioOutput::new(self.params.clone())
    }

    /// Create an input processor with a custom name.
    pub fn input_with_name(&self, name: impl Into<String>) -> LocalAudioInput {
        LocalAudioInput::with_name(name, self.params.clone())
    }

    /// Create an output processor with a custom name.
    pub fn output_with_name(&self, name: impl Into<String>) -> LocalAudioOutput {
        LocalAudioOutput::with_name(name, self.params.clone())
    }

    /// Get a reference to the transport parameters.
    pub fn params(&self) -> &LocalAudioTransportParams {
        &self.params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pipecat_pipeline::processor::FrameProcessor;

    #[test]
    fn transport_creates_input_and_output() {
        let transport = LocalAudioTransport::new(LocalAudioTransportParams::default());
        let input = transport.input();
        let output = transport.output();
        assert_eq!(input.name(), "LocalAudioInput");
        assert_eq!(output.name(), "LocalAudioOutput");
    }

    #[test]
    fn transport_custom_names() {
        let transport = LocalAudioTransport::new(LocalAudioTransportParams::default());
        let input = transport.input_with_name("mic_1");
        let output = transport.output_with_name("speakers_1");
        assert_eq!(input.name(), "mic_1");
        assert_eq!(output.name(), "speakers_1");
    }

    #[test]
    fn transport_params_accessible() {
        let params = LocalAudioTransportParams {
            ring_buffer_size: 4096,
            ..Default::default()
        };
        let transport = LocalAudioTransport::new(params);
        assert_eq!(transport.params().ring_buffer_size, 4096);
    }

    #[test]
    fn input_output_are_independent() {
        let transport = LocalAudioTransport::new(LocalAudioTransportParams::default());
        let input1 = transport.input();
        let input2 = transport.input();
        // Each call creates a separate processor instance.
        // They have the same name but are distinct objects.
        assert_eq!(input1.name(), input2.name());
    }

    #[test]
    fn transport_implements_send() {
        fn assert_send<T: Send>() {}
        assert_send::<LocalAudioTransport>();
    }
}
