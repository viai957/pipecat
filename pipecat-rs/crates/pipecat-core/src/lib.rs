//! # pipecat-core
//!
//! Core frame types and traits for Pipecat real-time AI pipelines.
//!
//! This crate provides the foundational data types used throughout the Pipecat
//! Rust framework. It has **no async runtime dependency** — all types are
//! synchronous, `Send + Sync`, and suitable for use in any executor.
//!
//! ## Key modules
//!
//! - [`frames`] — The [`Frame`](frames::Frame) enum and associated data structs.
//! - [`frame_id`] — Atomic, globally-unique frame ID generation.
//! - [`frame_types`] — Numeric type IDs for O(1) dispatch tables.
//! - [`error`] — [`PipecatError`](error::PipecatError) and [`Result`](error::Result).
//! - [`clock`] — Monotonic pipeline clock for PTS generation.
//! - [`events`] — Synchronous typed event emitter (no tokio required).
//! - [`metrics`] — Metrics data types (TTFB, processing, LLM token usage).
//! - [`dtmf`] — DTMF keypad enum and parsing.
//! - [`language`] — BCP-47 language identifiers.

pub mod clock;
pub mod dtmf;
pub mod error;
pub mod events;
pub mod frame_id;
pub mod frame_types;
pub mod frames;
pub mod language;
pub mod metrics;

// Re-export the most commonly used types at the crate root.
pub use error::{PipecatError, Result};
pub use frame_id::FrameId;
pub use frames::{
    AudioData, ErrorData, Frame, FrameClass, FrameDirection, FrameHeader, ImageData,
    LlmContextData, TextData, TranscriptionData,
};
