//! Voice Activity Detection (VAD) for Pipecat pipelines.
//!
//! This crate provides a VAD state machine with hysteresis-based transition logic,
//! a pluggable model backend trait ([`VadModel`]), and a Silero ONNX model stub
//! ([`SileroVadModel`]) that can be completed when the `ort` crate is available.
//!
//! The core state machine in [`VadAnalyzer`] is a direct port of the Python
//! `pipecat.audio.vad.vad_analyzer.VADAnalyzer`.
//!
//! # Quick start
//!
//! ```rust,no_run
//! use pipecat_vad::{VadAnalyzer, VadParams, VadState, MockVadModel};
//!
//! let model = MockVadModel::always_speaking(160);
//! let mut analyzer = VadAnalyzer::new(model, 16000, VadParams::default());
//!
//! // Feed audio chunks and observe state transitions.
//! let audio = vec![0u8; 320]; // 160 samples * 2 bytes per i16
//! let state = analyzer.analyze(&audio);
//! ```

pub mod analyzer;
pub mod mock;
pub mod params;
pub mod silero;

pub use analyzer::{VadAnalyzer, VadModel};
pub use mock::MockVadModel;
pub use params::{VadParams, VadState};
pub use silero::SileroVadModel;
