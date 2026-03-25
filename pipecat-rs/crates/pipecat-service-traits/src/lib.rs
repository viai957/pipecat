//! # pipecat-service-traits
//!
//! Abstract traits for AI service integrations in Pipecat pipelines.
//!
//! This crate defines the contracts that concrete service implementations
//! (Deepgram, OpenAI, ElevenLabs, etc.) must fulfil. It has no provider-
//! specific dependencies -- only the pipeline and core crates.
//!
//! ## Key types
//!
//! - [`AIService`] -- base trait extended by all service categories.
//! - [`STTService`] -- Speech-to-Text: audio in, transcription frames out.
//! - [`TTSService`] -- Text-to-Speech: text in, audio frames out.
//! - [`LLMService`] -- Large Language Models: context in, text/tool-call frames out.
//! - [`Setting<T>`](Setting) -- delta-update primitive replacing Python's `NOT_GIVEN`.
//! - [`ServiceError`] -- error type for service operations.
//!
//! ## Settings system
//!
//! The [`Setting<T>`](Setting) enum allows callers to construct a "delta"
//! struct where only the fields that should change are set to
//! [`Setting::Value`], while unchanged fields remain [`Setting::NotGiven`].
//! The [`ServiceSettingsDelta::apply_to`] method patches a
//! [`ServiceSettings`] store and returns the names and old values of changed
//! fields.

pub mod ai_service;
pub mod decorator;
pub mod error;
pub mod llm;
pub mod settings;
pub mod stt;
pub mod tts;

// Re-export the most commonly used types at the crate root.
pub use ai_service::AIService;
pub use decorator::{
    RateLimitConfig, RateLimitedStt, RateLimitedTts, TracedStt, TracedTts,
};
pub use error::{Result, ServiceError};
pub use llm::{FunctionCallHandler, LLMService};
pub use settings::{
    LlmSettings, ServiceSettings, ServiceSettingsDelta, Setting, SttSettings, TtsSettings,
};
pub use stt::STTService;
pub use tts::TTSService;
