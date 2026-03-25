//! The [`TTSService`] trait for Text-to-Speech integrations.
//!
//! TTS services receive text and produce audio frames
//! ([`Frame::AudioTts`]), bracketed by [`Frame::TtsStarted`] and
//! [`Frame::TtsStopped`] lifecycle events.

use async_trait::async_trait;

use pipecat_core::Frame;

use crate::ai_service::AIService;

/// Trait for Text-to-Speech services.
///
/// TTS services receive text strings and synthesise audio, returning a
/// sequence of frames that the pipeline can stream to the output transport.
///
/// # Frame sequence
///
/// A typical `run_tts` call returns frames in this order:
///
/// 1. [`Frame::TtsStarted`] -- signals the start of synthesis
/// 2. One or more [`Frame::AudioTts`] -- the synthesised audio chunks
/// 3. [`Frame::TtsStopped`] -- signals the end of synthesis
///
/// Streaming TTS providers may return audio chunks as they become available,
/// so the returned `Vec` may grow over multiple calls if the implementation
/// buffers internally.
///
/// # Context ID
///
/// The `context_id` parameter is threaded through to [`Frame::AudioTts`] so
/// that downstream processors (e.g. output transports) can correlate audio
/// with the conversational context that produced it.
#[async_trait]
pub trait TTSService: AIService {
    /// Synthesise `text` to audio, returning frames as they become available.
    ///
    /// The returned `Vec` typically contains:
    /// - [`Frame::TtsStarted`] -- synthesis start event
    /// - [`Frame::AudioTts`] -- audio data chunks
    /// - [`Frame::TtsStopped`] -- synthesis end event
    /// - [`Frame::Error`] -- if synthesis fails
    async fn run_tts(&mut self, text: &str, context_id: &str) -> Vec<Frame>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use bytes::Bytes;
    use pipecat_core::{AudioData, Frame, FrameDirection, FrameHeader, Result};
    use pipecat_pipeline::processor::{FrameProcessor, ProcessorContext};

    struct FakeTts;

    #[async_trait]
    impl FrameProcessor for FakeTts {
        async fn process_frame(
            &mut self,
            frame: Frame,
            direction: FrameDirection,
            ctx: &ProcessorContext,
        ) -> Result<()> {
            ctx.push_frame(frame, direction).await
        }
        fn name(&self) -> &str {
            "FakeTts"
        }
    }

    #[async_trait]
    impl AIService for FakeTts {
        fn model_name(&self) -> Option<&str> {
            Some("fake-tts-model")
        }
    }

    #[async_trait]
    impl TTSService for FakeTts {
        async fn run_tts(&mut self, text: &str, context_id: &str) -> Vec<Frame> {
            if text.is_empty() {
                return vec![];
            }
            vec![
                Frame::TtsStarted(FrameHeader::new()),
                Frame::AudioTts {
                    header: FrameHeader::new(),
                    audio: AudioData {
                        audio: Bytes::from(vec![0u8; 3200]),
                        sample_rate: 16000,
                        num_channels: 1,
                    },
                    context_id: Some(context_id.to_string()),
                },
                Frame::TtsStopped(FrameHeader::new()),
            ]
        }
    }

    #[tokio::test]
    async fn tts_produces_audio_sequence() {
        let mut tts = FakeTts;
        let frames = tts.run_tts("Hello, world!", "ctx-1").await;
        assert_eq!(frames.len(), 3);
        assert_eq!(frames[0].name(), "TtsStarted");
        assert_eq!(frames[1].name(), "AudioTts");
        assert_eq!(frames[2].name(), "TtsStopped");
    }

    #[tokio::test]
    async fn tts_context_id_threaded() {
        let mut tts = FakeTts;
        let frames = tts.run_tts("test", "my-context").await;
        if let Frame::AudioTts { context_id, .. } = &frames[1] {
            assert_eq!(context_id.as_deref(), Some("my-context"));
        } else {
            panic!("expected AudioTts frame");
        }
    }

    #[tokio::test]
    async fn tts_empty_text_produces_nothing() {
        let mut tts = FakeTts;
        let frames = tts.run_tts("", "ctx").await;
        assert!(frames.is_empty());
    }

    #[test]
    fn tts_model_name() {
        let tts = FakeTts;
        assert_eq!(tts.model_name(), Some("fake-tts-model"));
    }
}
