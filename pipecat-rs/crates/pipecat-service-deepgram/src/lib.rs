//! # pipecat-service-deepgram
//!
//! Deepgram STT (Speech-to-Text) service integration for Pipecat pipelines.
//!
//! This crate provides [`DeepgramSTT`], a [`FrameProcessor`] that streams
//! audio to the [Deepgram](https://deepgram.com/) real-time transcription API
//! over WebSocket and emits transcription frames back into the pipeline.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use pipecat_service_deepgram::{DeepgramSTT, DeepgramConfig};
//!
//! let config = DeepgramConfig {
//!     api_key: std::env::var("DEEPGRAM_API_KEY").unwrap(),
//!     model: "nova-3-general".into(),
//!     ..Default::default()
//! };
//!
//! let stt = DeepgramSTT::new(config);
//! // Add `stt` to your pipeline — it will connect on Start and disconnect on End.
//! ```
//!
//! ## Architecture
//!
//! The processor connects lazily (on `StartFrame`) and manages two background
//! tokio tasks for WebSocket I/O:
//!
//! 1. **Writer task** — receives PCM audio bytes from `AudioRawInput` frames
//!    and sends them as binary WebSocket messages.
//! 2. **Reader task** — receives JSON transcription responses from Deepgram,
//!    parses them, and forwards transcription frames through an internal channel.
//!
//! Transcription frames are drained and pushed downstream each time an audio
//! frame is processed, ensuring low latency without requiring a separate
//! polling mechanism.
//!
//! [`DeepgramSTT`]: stt::DeepgramSTT
//! [`FrameProcessor`]: pipecat_pipeline::FrameProcessor

pub mod config;
pub mod error;
pub mod stt;
pub mod types;

// Re-export the most commonly used types at the crate root.
pub use config::DeepgramConfig;
pub use error::DeepgramError;
pub use stt::DeepgramSTT;
pub use types::DeepgramResponse;
