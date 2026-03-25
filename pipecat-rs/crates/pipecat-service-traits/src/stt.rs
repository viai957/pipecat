//! The [`STTService`] trait for Speech-to-Text integrations.
//!
//! STT services receive audio data and produce [`Frame::Transcription`] and
//! [`Frame::InterimTranscription`] frames. They also support muting (to
//! temporarily suppress recognition) and runtime settings updates.

use async_trait::async_trait;

use pipecat_core::Frame;

use crate::ai_service::AIService;

/// Trait for Speech-to-Text services.
///
/// STT services receive audio chunks (typically 20ms of 16-bit PCM at 16kHz)
/// and produce transcription frames when speech is recognised.
///
/// # Muting
///
/// When muted, the service should discard incoming audio without processing
/// it. This is useful during bot speech to avoid self-transcription.
///
/// # Implementor's contract
///
/// - Call [`run_stt`](STTService::run_stt) with raw PCM audio bytes.
/// - Return zero or more frames: [`Frame::Transcription`] for final results,
///   [`Frame::InterimTranscription`] for partial/streaming results.
/// - Honour the muted state -- when [`is_muted`](STTService::is_muted) returns
///   `true`, [`run_stt`](STTService::run_stt) should return an empty `Vec`.
#[async_trait]
pub trait STTService: AIService {
    /// Process an audio chunk and return resulting frames.
    ///
    /// The returned `Vec` may contain:
    /// - [`Frame::Transcription`] -- final transcription result
    /// - [`Frame::InterimTranscription`] -- partial/streaming result
    /// - [`Frame::Error`] -- if recognition fails for this chunk
    ///
    /// An empty `Vec` means no output was produced for this chunk (normal for
    /// streaming STT where results arrive after multiple chunks).
    async fn run_stt(&mut self, audio: &[u8]) -> Vec<Frame>;

    /// Whether this STT service is currently muted.
    ///
    /// When muted, audio should be discarded without processing.
    fn is_muted(&self) -> bool {
        false
    }

    /// Set the muted state.
    fn set_muted(&mut self, muted: bool);
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use pipecat_core::{Frame, FrameDirection, FrameHeader, Result, TranscriptionData};
    use pipecat_pipeline::processor::{FrameProcessor, ProcessorContext};

    struct FakeStt {
        muted: bool,
    }

    impl FakeStt {
        fn new() -> Self {
            Self { muted: false }
        }
    }

    #[async_trait]
    impl FrameProcessor for FakeStt {
        async fn process_frame(
            &mut self,
            frame: Frame,
            direction: FrameDirection,
            ctx: &ProcessorContext,
        ) -> Result<()> {
            ctx.push_frame(frame, direction).await
        }
        fn name(&self) -> &str {
            "FakeStt"
        }
    }

    #[async_trait]
    impl AIService for FakeStt {
        fn model_name(&self) -> Option<&str> {
            Some("fake-stt-model")
        }
    }

    #[async_trait]
    impl STTService for FakeStt {
        async fn run_stt(&mut self, audio: &[u8]) -> Vec<Frame> {
            if self.is_muted() || audio.is_empty() {
                return vec![];
            }
            vec![Frame::Transcription {
                header: FrameHeader::new(),
                data: Box::new(TranscriptionData {
                    text: "hello world".to_string(),
                    user_id: None,
                    timestamp: None,
                    language: Some("en".to_string()),
                }),
            }]
        }

        fn is_muted(&self) -> bool {
            self.muted
        }

        fn set_muted(&mut self, muted: bool) {
            self.muted = muted;
        }
    }

    #[tokio::test]
    async fn stt_produces_transcription() {
        let mut stt = FakeStt::new();
        let frames = stt.run_stt(&[0u8; 640]).await;
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].name(), "Transcription");
    }

    #[tokio::test]
    async fn stt_muted_produces_nothing() {
        let mut stt = FakeStt::new();
        stt.set_muted(true);
        assert!(stt.is_muted());
        let frames = stt.run_stt(&[0u8; 640]).await;
        assert!(frames.is_empty());
    }

    #[tokio::test]
    async fn stt_unmute() {
        let mut stt = FakeStt::new();
        stt.set_muted(true);
        assert!(stt.is_muted());
        stt.set_muted(false);
        assert!(!stt.is_muted());
        let frames = stt.run_stt(&[0u8; 640]).await;
        assert_eq!(frames.len(), 1);
    }

    #[tokio::test]
    async fn stt_empty_audio() {
        let mut stt = FakeStt::new();
        let frames = stt.run_stt(&[]).await;
        assert!(frames.is_empty());
    }

    #[test]
    fn stt_model_name() {
        let stt = FakeStt::new();
        assert_eq!(stt.model_name(), Some("fake-stt-model"));
    }
}
