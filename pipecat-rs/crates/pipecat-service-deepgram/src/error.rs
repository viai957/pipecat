//! Error types for the Deepgram STT service.

use pipecat_core::PipecatError;

/// Errors specific to the Deepgram service.
#[derive(Debug, thiserror::Error)]
pub enum DeepgramError {
    /// WebSocket transport error.
    #[error("websocket error: {0}")]
    WebSocket(#[from] tokio_tungstenite::tungstenite::Error),

    /// Failed to establish a connection to the Deepgram API.
    #[error("connection failed: {0}")]
    ConnectionFailed(String),

    /// JSON serialization/deserialization error.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    /// Error propagated from the Pipecat core.
    #[error(transparent)]
    Core(#[from] PipecatError),
}

/// Convenience result alias for Deepgram operations.
pub type Result<T> = std::result::Result<T, DeepgramError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn websocket_error_display() {
        let err = DeepgramError::ConnectionFailed("timeout".into());
        assert_eq!(err.to_string(), "connection failed: timeout");
    }

    #[test]
    fn json_error_converts() {
        let json_err = serde_json::from_str::<serde_json::Value>("{{bad}}")
            .expect_err("should fail");
        let err: DeepgramError = json_err.into();
        assert!(matches!(err, DeepgramError::Json(_)));
    }

    #[test]
    fn core_error_converts() {
        let core_err = PipecatError::Service("test".into());
        let err: DeepgramError = core_err.into();
        assert!(matches!(err, DeepgramError::Core(_)));
    }
}
