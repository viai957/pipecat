//! # pipecat-transport-local
//!
//! Local audio transport for Pipecat pipelines using [cpal](https://crates.io/crates/cpal)
//! for hardware audio I/O and [rtrb](https://crates.io/crates/rtrb) lock-free ring buffers
//! for bridging between real-time audio threads and async tokio tasks.
//!
//! ## Architecture
//!
//! ```text
//! cpal input callback (RT thread) --> rtrb SPSC --> tokio task --> AudioRawInput --> pipeline
//! pipeline --> AudioRawOutput --> write to rtrb SPSC --> cpal output callback (RT thread)
//! ```
//!
//! The real-time audio callbacks obey strict RT rules: **no allocations, no locks,
//! no syscalls**. Only `rtrb` wait-free ring buffer operations are used.
//!
//! ## Key types
//!
//! - [`LocalAudioTransport`] -- convenience factory that creates matched input
//!   and output processors.
//! - [`LocalAudioInput`] -- captures audio from a local microphone and pushes
//!   `AudioRawInput` frames downstream.
//! - [`LocalAudioOutput`] -- receives `AudioRawOutput` frames and plays them
//!   through local speakers.
//! - [`LocalAudioTransportParams`] -- configuration for device selection and
//!   ring buffer sizing.
//!
//! ## Example
//!
//! ```rust,no_run
//! use pipecat_transport_local::{LocalAudioTransport, LocalAudioTransportParams};
//!
//! let transport = LocalAudioTransport::new(LocalAudioTransportParams::default());
//! let input = transport.input();
//! let output = transport.output();
//! // Use input and output as FrameProcessors in your pipeline.
//! ```

pub mod input;
pub mod output;
pub mod params;
pub mod transport;

// Re-export the most commonly used types at the crate root.
pub use input::LocalAudioInput;
pub use output::LocalAudioOutput;
pub use params::LocalAudioTransportParams;
pub use transport::LocalAudioTransport;
