use thiserror::Error;

/// Top-level error type for the Pipecat framework.
#[derive(Debug, Error)]
pub enum PipecatError {
    #[error("pipeline error: {0}")]
    Pipeline(String),

    #[error("downstream send failed")]
    DownstreamSendFailed,

    #[error("upstream send failed")]
    UpstreamSendFailed,

    #[error("processor error: {message}")]
    Processor { message: String, fatal: bool },

    #[error("transport error: {0}")]
    Transport(String),

    #[error("service error: {0}")]
    Service(String),

    #[error("audio error: {0}")]
    Audio(String),

    #[error("frame error: {0}")]
    Frame(String),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    SerdeJson(#[from] serde_json::Error),
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, PipecatError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_error_display() {
        let err = PipecatError::Pipeline("broken pipe".into());
        assert_eq!(err.to_string(), "pipeline error: broken pipe");
    }

    #[test]
    fn processor_error_display() {
        let err = PipecatError::Processor {
            message: "bad input".into(),
            fatal: true,
        };
        assert_eq!(err.to_string(), "processor error: bad input");
    }

    #[test]
    fn io_error_converts() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "missing");
        let err: PipecatError = io_err.into();
        assert!(matches!(err, PipecatError::Io(_)));
    }

    #[test]
    fn serde_json_error_converts() {
        let json_err = serde_json::from_str::<serde_json::Value>("{{bad}}")
            .expect_err("should fail");
        let err: PipecatError = json_err.into();
        assert!(matches!(err, PipecatError::SerdeJson(_)));
    }

    #[test]
    fn result_alias_works() {
        fn ok_fn() -> Result<u32> {
            Ok(42)
        }
        fn err_fn() -> Result<u32> {
            Err(PipecatError::Frame("oops".into()))
        }
        assert_eq!(ok_fn().unwrap(), 42);
        assert!(err_fn().is_err());
    }
}
