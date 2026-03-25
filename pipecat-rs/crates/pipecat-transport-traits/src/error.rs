//! Transport-specific error types.

/// Errors specific to transport operations.
#[derive(Debug, thiserror::Error)]
pub enum TransportError {
    /// An audio device (microphone, speaker) reported an error.
    #[error("audio device error: {0}")]
    AudioDevice(String),

    /// An operation was attempted on a transport that has not been started.
    #[error("transport not started")]
    NotStarted,

    /// An operation was attempted on a transport that is already running.
    #[error("transport already started")]
    AlreadyStarted,

    /// A channel send failed, typically because the receiver was dropped.
    #[error("channel send error: {0}")]
    ChannelSend(String),

    /// An error propagated from `pipecat-core`.
    #[error(transparent)]
    Core(#[from] pipecat_core::PipecatError),
}

/// Convenience result alias for transport operations.
pub type Result<T> = std::result::Result<T, TransportError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_device_error_display() {
        let err = TransportError::AudioDevice("mic not found".into());
        assert_eq!(err.to_string(), "audio device error: mic not found");
    }

    #[test]
    fn not_started_display() {
        let err = TransportError::NotStarted;
        assert_eq!(err.to_string(), "transport not started");
    }

    #[test]
    fn already_started_display() {
        let err = TransportError::AlreadyStarted;
        assert_eq!(err.to_string(), "transport already started");
    }

    #[test]
    fn channel_send_display() {
        let err = TransportError::ChannelSend("receiver dropped".into());
        assert_eq!(err.to_string(), "channel send error: receiver dropped");
    }

    #[test]
    fn core_error_converts() {
        let core_err = pipecat_core::PipecatError::Transport("bad transport".into());
        let err: TransportError = core_err.into();
        assert!(matches!(err, TransportError::Core(_)));
        assert!(err.to_string().contains("bad transport"));
    }
}
