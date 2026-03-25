//! Composable service decorators for cross-cutting concerns.
//!
//! These wrappers add rate limiting and tracing to any service implementation
//! without modifying the underlying provider. Each decorator delegates
//! `FrameProcessor` and `AIService` to the inner service.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::Semaphore;

use pipecat_core::{Frame, FrameDirection, Result};
use pipecat_pipeline::processor::{FrameProcessor, ProcessorContext};

use crate::ai_service::AIService;
use crate::stt::STTService;
use crate::tts::TTSService;

/// Configuration for rate-limited service wrappers.
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Maximum number of concurrent requests to the underlying service.
    pub max_concurrent: usize,
    /// Optional timeout for acquiring a rate limit slot.
    pub acquire_timeout: Option<Duration>,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            max_concurrent: 10,
            acquire_timeout: Some(Duration::from_secs(30)),
        }
    }
}

// ── Rate-limited STT ──────────────────────────────────────────────────────

/// Rate-limited wrapper for STT services.
pub struct RateLimitedStt<S: STTService> {
    inner: S,
    semaphore: Arc<Semaphore>,
    config: RateLimitConfig,
}

impl<S: STTService> RateLimitedStt<S> {
    pub fn new(inner: S, config: RateLimitConfig) -> Self {
        let semaphore = Arc::new(Semaphore::new(config.max_concurrent));
        Self {
            inner,
            semaphore,
            config,
        }
    }
}

#[async_trait]
impl<S: STTService + Send + Sync> FrameProcessor for RateLimitedStt<S> {
    async fn process_frame(
        &mut self,
        frame: Frame,
        direction: FrameDirection,
        ctx: &ProcessorContext,
    ) -> Result<()> {
        self.inner.process_frame(frame, direction, ctx).await
    }

    fn name(&self) -> &str {
        self.inner.name()
    }
}

#[async_trait]
impl<S: STTService + Send + Sync> AIService for RateLimitedStt<S> {
    fn model_name(&self) -> Option<&str> {
        self.inner.model_name()
    }
}

#[async_trait]
impl<S: STTService + Send + Sync> STTService for RateLimitedStt<S> {
    async fn run_stt(&mut self, audio: &[u8]) -> Vec<Frame> {
        let permit = if let Some(timeout) = self.config.acquire_timeout {
            match tokio::time::timeout(timeout, self.semaphore.clone().acquire_owned()).await {
                Ok(Ok(permit)) => Some(permit),
                _ => {
                    tracing::warn!("Rate limit timeout acquiring STT slot");
                    return vec![];
                }
            }
        } else {
            self.semaphore.clone().acquire_owned().await.ok()
        };

        let result = self.inner.run_stt(audio).await;
        drop(permit);
        result
    }

    fn is_muted(&self) -> bool {
        self.inner.is_muted()
    }

    fn set_muted(&mut self, muted: bool) {
        self.inner.set_muted(muted);
    }
}

// ── Rate-limited TTS ──────────────────────────────────────────────────────

/// Rate-limited wrapper for TTS services.
pub struct RateLimitedTts<S: TTSService> {
    inner: S,
    semaphore: Arc<Semaphore>,
    config: RateLimitConfig,
}

impl<S: TTSService> RateLimitedTts<S> {
    pub fn new(inner: S, config: RateLimitConfig) -> Self {
        let semaphore = Arc::new(Semaphore::new(config.max_concurrent));
        Self {
            inner,
            semaphore,
            config,
        }
    }
}

#[async_trait]
impl<S: TTSService + Send + Sync> FrameProcessor for RateLimitedTts<S> {
    async fn process_frame(
        &mut self,
        frame: Frame,
        direction: FrameDirection,
        ctx: &ProcessorContext,
    ) -> Result<()> {
        self.inner.process_frame(frame, direction, ctx).await
    }

    fn name(&self) -> &str {
        self.inner.name()
    }
}

#[async_trait]
impl<S: TTSService + Send + Sync> AIService for RateLimitedTts<S> {
    fn model_name(&self) -> Option<&str> {
        self.inner.model_name()
    }
}

#[async_trait]
impl<S: TTSService + Send + Sync> TTSService for RateLimitedTts<S> {
    async fn run_tts(&mut self, text: &str, context_id: &str) -> Vec<Frame> {
        let permit = if let Some(timeout) = self.config.acquire_timeout {
            match tokio::time::timeout(timeout, self.semaphore.clone().acquire_owned()).await {
                Ok(Ok(permit)) => Some(permit),
                _ => {
                    tracing::warn!("Rate limit timeout acquiring TTS slot");
                    return vec![];
                }
            }
        } else {
            self.semaphore.clone().acquire_owned().await.ok()
        };

        let result = self.inner.run_tts(text, context_id).await;
        drop(permit);
        result
    }
}

// ── Traced STT ────────────────────────────────────────────────────────────

/// Traced wrapper that records call duration and result count for STT.
pub struct TracedStt<S: STTService> {
    inner: S,
    span_name: &'static str,
}

impl<S: STTService> TracedStt<S> {
    pub fn new(inner: S, span_name: &'static str) -> Self {
        Self { inner, span_name }
    }
}

#[async_trait]
impl<S: STTService + Send + Sync> FrameProcessor for TracedStt<S> {
    async fn process_frame(
        &mut self,
        frame: Frame,
        direction: FrameDirection,
        ctx: &ProcessorContext,
    ) -> Result<()> {
        self.inner.process_frame(frame, direction, ctx).await
    }

    fn name(&self) -> &str {
        self.inner.name()
    }
}

#[async_trait]
impl<S: STTService + Send + Sync> AIService for TracedStt<S> {
    fn model_name(&self) -> Option<&str> {
        self.inner.model_name()
    }
}

#[async_trait]
impl<S: STTService + Send + Sync> STTService for TracedStt<S> {
    async fn run_stt(&mut self, audio: &[u8]) -> Vec<Frame> {
        let start = std::time::Instant::now();
        let frames = self.inner.run_stt(audio).await;
        let elapsed = start.elapsed();

        tracing::debug!(
            span = self.span_name,
            audio_bytes = audio.len(),
            frames_produced = frames.len(),
            duration_us = elapsed.as_micros() as u64,
            "STT call completed"
        );

        frames
    }

    fn is_muted(&self) -> bool {
        self.inner.is_muted()
    }

    fn set_muted(&mut self, muted: bool) {
        self.inner.set_muted(muted);
    }
}

// ── Traced TTS ────────────────────────────────────────────────────────────

/// Traced wrapper that records call duration and result count for TTS.
pub struct TracedTts<S: TTSService> {
    inner: S,
    span_name: &'static str,
}

impl<S: TTSService> TracedTts<S> {
    pub fn new(inner: S, span_name: &'static str) -> Self {
        Self { inner, span_name }
    }
}

#[async_trait]
impl<S: TTSService + Send + Sync> FrameProcessor for TracedTts<S> {
    async fn process_frame(
        &mut self,
        frame: Frame,
        direction: FrameDirection,
        ctx: &ProcessorContext,
    ) -> Result<()> {
        self.inner.process_frame(frame, direction, ctx).await
    }

    fn name(&self) -> &str {
        self.inner.name()
    }
}

#[async_trait]
impl<S: TTSService + Send + Sync> AIService for TracedTts<S> {
    fn model_name(&self) -> Option<&str> {
        self.inner.model_name()
    }
}

#[async_trait]
impl<S: TTSService + Send + Sync> TTSService for TracedTts<S> {
    async fn run_tts(&mut self, text: &str, context_id: &str) -> Vec<Frame> {
        let start = std::time::Instant::now();
        let frames = self.inner.run_tts(text, context_id).await;
        let elapsed = start.elapsed();

        tracing::debug!(
            span = self.span_name,
            text_len = text.len(),
            context_id = context_id,
            frames_produced = frames.len(),
            duration_us = elapsed.as_micros() as u64,
            "TTS call completed"
        );

        frames
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pipecat_core::{Frame, FrameHeader, TranscriptionData};

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
            Some("fake")
        }
    }

    #[async_trait]
    impl STTService for FakeStt {
        async fn run_stt(&mut self, audio: &[u8]) -> Vec<Frame> {
            if self.muted || audio.is_empty() {
                return vec![];
            }
            vec![Frame::Transcription {
                header: FrameHeader::new(),
                data: Box::new(TranscriptionData {
                    text: "test".into(),
                    user_id: None,
                    timestamp: None,
                    language: None,
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
    async fn rate_limited_stt_passes_through() {
        let fake = FakeStt::new();
        let mut limited = RateLimitedStt::new(fake, RateLimitConfig::default());
        let frames = limited.run_stt(&[0u8; 640]).await;
        assert_eq!(frames.len(), 1);
    }

    #[tokio::test]
    async fn traced_stt_passes_through() {
        let fake = FakeStt::new();
        let mut traced = TracedStt::new(fake, "test_stt");
        let frames = traced.run_stt(&[0u8; 640]).await;
        assert_eq!(frames.len(), 1);
    }

    #[tokio::test]
    async fn rate_limited_preserves_mute() {
        let fake = FakeStt::new();
        let mut limited = RateLimitedStt::new(fake, RateLimitConfig::default());
        limited.set_muted(true);
        assert!(limited.is_muted());
        let frames = limited.run_stt(&[0u8; 640]).await;
        assert!(frames.is_empty());
    }

    #[tokio::test]
    async fn decorator_composition() {
        // TracedStt<RateLimitedStt<FakeStt>> — verifies decorator stacking
        let fake = FakeStt::new();
        let limited = RateLimitedStt::new(fake, RateLimitConfig::default());
        let mut traced = TracedStt::new(limited, "composed");
        let frames = traced.run_stt(&[0u8; 640]).await;
        assert_eq!(frames.len(), 1);
        assert_eq!(traced.model_name(), Some("fake"));
    }
}
