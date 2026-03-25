//! Audio processing utilities for Pipecat pipelines.
//!
//! This crate provides common audio processing functions including resampling,
//! volume calculation, silence detection, audio mixing, G.711 codec conversions,
//! and DTMF keypad types.

pub mod buffer_pool;
pub mod codec;
pub mod dtmf;
pub mod mix;
pub mod resampler;
pub mod ring_buffer;
pub mod silence;
pub mod volume;

pub use buffer_pool::AudioBufferPool;
pub use codec::{alaw_decode, alaw_encode, ulaw_decode, ulaw_encode};
pub use dtmf::KeypadEntry;
pub use mix::{interleave_stereo, mix_audio};
pub use resampler::{FileResampler, StreamResampler};
pub use ring_buffer::{audio_ring_buffer, AudioRingConsumer, AudioRingProducer};
pub use silence::is_silence;
pub use volume::{calculate_audio_volume, exp_smoothing, normalize_value, SPEAKING_THRESHOLD};
