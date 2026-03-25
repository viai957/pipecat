//! The [`AIService`] trait -- base abstraction for all AI service integrations.
//!
//! Every AI service (STT, TTS, LLM, image generation, etc.) is a
//! [`FrameProcessor`](pipecat_pipeline::processor::FrameProcessor) that
//! additionally exposes model metadata and metrics capability.

use async_trait::async_trait;

use pipecat_pipeline::processor::FrameProcessor;

/// Base trait for all AI services.
///
/// AI services are [`FrameProcessor`] instances that additionally manage
/// settings and model lifecycle. Concrete service traits ([`STTService`],
/// [`TTSService`], [`LLMService`]) extend this with domain-specific methods.
///
/// [`STTService`]: crate::stt::STTService
/// [`TTSService`]: crate::tts::TTSService
/// [`LLMService`]: crate::llm::LLMService
#[async_trait]
pub trait AIService: FrameProcessor {
    /// Get the current model name, if one is configured.
    fn model_name(&self) -> Option<&str>;

    /// Whether this service is capable of emitting metrics frames.
    ///
    /// Services that track TTFB, processing time, or token usage should
    /// override this to return `true`.
    fn can_generate_metrics(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use pipecat_core::{Frame, FrameDirection, Result};
    use pipecat_pipeline::processor::ProcessorContext;

    /// A minimal concrete AIService for testing.
    struct FakeService {
        model: Option<String>,
    }

    impl FakeService {
        fn new(model: Option<&str>) -> Self {
            Self {
                model: model.map(|s| s.to_string()),
            }
        }
    }

    #[async_trait]
    impl FrameProcessor for FakeService {
        async fn process_frame(
            &mut self,
            frame: Frame,
            direction: FrameDirection,
            ctx: &ProcessorContext,
        ) -> Result<()> {
            ctx.push_frame(frame, direction).await
        }

        fn name(&self) -> &str {
            "FakeService"
        }
    }

    #[async_trait]
    impl AIService for FakeService {
        fn model_name(&self) -> Option<&str> {
            self.model.as_deref()
        }
    }

    #[test]
    fn ai_service_model_name() {
        let svc = FakeService::new(Some("gpt-4o"));
        assert_eq!(svc.model_name(), Some("gpt-4o"));

        let svc = FakeService::new(None);
        assert_eq!(svc.model_name(), None);
    }

    #[test]
    fn ai_service_default_metrics() {
        let svc = FakeService::new(None);
        assert!(!svc.can_generate_metrics());
    }

    #[test]
    fn ai_service_is_frame_processor() {
        let svc = FakeService::new(Some("test"));
        // Verify FrameProcessor name method is accessible
        assert_eq!(svc.name(), "FakeService");
    }
}
