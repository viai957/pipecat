//! # pipecat-transport-traits
//!
//! Abstract transport traits for Pipecat real-time AI pipelines.
//!
//! This crate defines the interfaces that concrete transport implementations
//! (Daily WebRTC, local audio, WebSocket, etc.) must satisfy. It also provides
//! shared utilities like [`TransportParams`] for configuration and
//! [`MediaSender`] for audio output buffering and pacing.
//!
//! ## Key types
//!
//! - [`InputTransport`] -- trait for transports that capture audio/video and
//!   inject it into the pipeline.
//! - [`OutputTransport`] -- trait for transports that play audio/video from
//!   the pipeline to an external destination.
//! - [`MediaSender`] -- per-destination audio buffering with chunk-based
//!   pacing and bot speaking state detection.
//! - [`TransportParams`] -- configuration struct for audio/video I/O settings.
//! - [`TransportError`] -- transport-specific error type.

pub mod error;
pub mod input;
pub mod output;
pub mod params;

// Re-export the most commonly used types at the crate root.
pub use error::{Result, TransportError};
pub use input::InputTransport;
pub use output::{BotSpeakingEvent, MediaSender, OutputTransport};
pub use params::TransportParams;
