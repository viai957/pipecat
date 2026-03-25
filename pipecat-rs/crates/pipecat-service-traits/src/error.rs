//! Service-specific error types.
//!
//! [`ServiceError`] covers failure modes common to AI service integrations:
//! connection failures, authentication issues, model errors, and timeouts.
//! It also converts transparently from [`pipecat_core::PipecatError`].

use thiserror::Error;

/// Error type for AI service operations.
#[derive(Debug, Error)]
pub enum ServiceError {
    /// A network or WebSocket connection could not be established or was lost.
    #[error("connection error: {0}")]
    Connection(String),

    /// API key, token, or credentials are invalid or expired.
    #[error("authentication error: {0}")]
    Authentication(String),

    /// The requested model does not exist or returned an error.
    #[error("model error: {0}")]
    Model(String),

    /// The operation exceeded its deadline.
    #[error("timeout: {0}")]
    Timeout(String),

    /// An error propagated from the core framework.
    #[error(transparent)]
    Core(#[from] pipecat_core::PipecatError),
}

/// Convenience alias for service results.
pub type Result<T> = std::result::Result<T, ServiceError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn connection_error_display() {
        let err = ServiceError::Connection("refused".into());
        assert_eq!(err.to_string(), "connection error: refused");
    }

    #[test]
    fn authentication_error_display() {
        let err = ServiceError::Authentication("invalid key".into());
        assert_eq!(err.to_string(), "authentication error: invalid key");
    }

    #[test]
    fn model_error_display() {
        let err = ServiceError::Model("model not found".into());
        assert_eq!(err.to_string(), "model error: model not found");
    }

    #[test]
    fn timeout_error_display() {
        let err = ServiceError::Timeout("30s exceeded".into());
        assert_eq!(err.to_string(), "timeout: 30s exceeded");
    }

    #[test]
    fn core_error_converts() {
        let core_err = pipecat_core::PipecatError::Service("upstream fail".into());
        let err: ServiceError = core_err.into();
        assert!(matches!(err, ServiceError::Core(_)));
        assert!(err.to_string().contains("upstream fail"));
    }

    #[test]
    fn result_alias() {
        fn ok_fn() -> Result<u32> {
            Ok(42)
        }
        fn err_fn() -> Result<u32> {
            Err(ServiceError::Model("missing".into()))
        }
        assert_eq!(ok_fn().unwrap(), 42);
        assert!(err_fn().is_err());
    }
}
