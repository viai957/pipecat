//! The [`FrameProcessor`] trait and [`ProcessorContext`] for building pipeline
//! stages.
//!
//! Every stage in a pipeline implements [`FrameProcessor`]. The runtime calls
//! [`process_frame`](FrameProcessor::process_frame) for each incoming frame.
//! The processor uses the provided [`ProcessorContext`] to push frames
//! downstream or upstream.

use std::sync::Arc;

use async_trait::async_trait;

use pipecat_core::clock::PipelineClock;
use pipecat_core::{Frame, FrameDirection, PipecatError, Result};

use crate::backpressure::{FrameEnvelope, FrameQueueSender};

/// Context provided to a [`FrameProcessor`] during frame processing.
///
/// Holds the senders connecting this processor to its neighbours and provides
/// convenience methods for pushing frames in either direction.
pub struct ProcessorContext {
    downstream: FrameQueueSender,
    upstream: FrameQueueSender,
    clock: PipelineClock,
}

impl ProcessorContext {
    /// Create a new context wired to the given downstream/upstream senders.
    pub fn new(
        downstream: FrameQueueSender,
        upstream: FrameQueueSender,
        clock: PipelineClock,
    ) -> Self {
        Self {
            downstream,
            upstream,
            clock,
        }
    }

    /// Push a frame in the specified direction.
    pub async fn push_frame(&self, frame: Frame, direction: FrameDirection) -> Result<()> {
        let envelope = FrameEnvelope { frame, direction };
        match direction {
            FrameDirection::Downstream => self
                .downstream
                .send(envelope)
                .await
                .map_err(|_| PipecatError::DownstreamSendFailed)?,
            FrameDirection::Upstream => self
                .upstream
                .send(envelope)
                .await
                .map_err(|_| PipecatError::UpstreamSendFailed)?,
        }
        Ok(())
    }

    /// Non-blocking push — use in [`FrameProcessor::process_frame_sync`].
    ///
    /// Applies the queue's drop policy synchronously. For `Block`/`Never`
    /// policies this returns `Err` if the bounded channel is full; callers
    /// needing guaranteed delivery should use the async [`push_frame`](Self::push_frame).
    pub fn try_push_frame(&self, frame: Frame, direction: FrameDirection) -> Result<()> {
        let envelope = FrameEnvelope { frame, direction };
        match direction {
            FrameDirection::Downstream => self
                .downstream
                .try_send(envelope)
                .map_err(|_| PipecatError::DownstreamSendFailed)?,
            FrameDirection::Upstream => self
                .upstream
                .try_send(envelope)
                .map_err(|_| PipecatError::UpstreamSendFailed)?,
        }
        Ok(())
    }

    /// Reference to the pipeline clock for generating PTS values.
    pub fn clock(&self) -> &PipelineClock {
        &self.clock
    }

    /// Get a clone of the downstream sender.
    ///
    /// This is intended for source processors (e.g. transport inputs) that
    /// need to push frames from a background task running outside the
    /// normal `process_frame` call path.
    pub fn downstream_sender(&self) -> Arc<FrameQueueSender> {
        Arc::new(self.downstream.clone())
    }

    /// Get a clone of the upstream sender.
    ///
    /// Used by the Python push bridge to route frames upstream through
    /// Rust channels when Python processors call `push_frame(frame, UPSTREAM)`.
    pub fn upstream_sender(&self) -> Arc<FrameQueueSender> {
        Arc::new(self.upstream.clone())
    }
}

/// Trait implemented by every pipeline stage.
///
/// The runtime guarantees:
/// - `process_frame` is called sequentially (no concurrent calls for the same
///   processor instance).
/// - `setup` is called once after the pipeline is assembled, before the first
///   data frame arrives.
/// - `cleanup` is called once when the pipeline is shutting down.
///
/// # Sync fast path
///
/// Simple processors that only forward frames can override [`is_sync`] to
/// return `true` and implement [`process_frame_sync`] using
/// [`ProcessorContext::try_push_frame`]. The runtime will call the sync
/// method for regular data frames, avoiding the per-call heap allocation
/// that `#[async_trait]` introduces for the async `process_frame`.
#[async_trait]
pub trait FrameProcessor: Send + Sync + 'static {
    /// Process a single frame. Implementations should use `ctx.push_frame()`
    /// to forward results.
    async fn process_frame(
        &mut self,
        frame: Frame,
        direction: FrameDirection,
        ctx: &ProcessorContext,
    ) -> Result<()>;

    /// Human-readable name of this processor (used in logging and metrics).
    fn name(&self) -> &str;

    /// Return `true` if the runtime should call [`process_frame_sync`] for
    /// regular data frames instead of the async [`process_frame`].
    ///
    /// Override this only for processors whose frame handling is fully
    /// synchronous (e.g. passthrough, simple filters).
    fn is_sync(&self) -> bool {
        false
    }

    /// Synchronous frame-processing fast path.
    ///
    /// Called for regular data frames when [`is_sync`] returns `true`. Use
    /// [`ProcessorContext::try_push_frame`] to forward frames without an
    /// async allocation. The default implementation panics — any processor
    /// that returns `true` from `is_sync` **must** override this.
    fn process_frame_sync(
        &mut self,
        _frame: Frame,
        _direction: FrameDirection,
        _ctx: &ProcessorContext,
    ) -> Result<()> {
        unreachable!("process_frame_sync called but is_sync() returned false")
    }

    /// Called once when the pipeline starts, after channel wiring is complete.
    async fn setup(&mut self, _ctx: &ProcessorContext) -> Result<()> {
        Ok(())
    }

    /// Called once when the pipeline is shutting down.
    async fn cleanup(&mut self) -> Result<()> {
        Ok(())
    }
}

/// A processor that passes every frame through unchanged. Useful for testing.
pub struct PassthroughProcessor {
    name: String,
}

impl PassthroughProcessor {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
        }
    }
}

#[async_trait]
impl FrameProcessor for PassthroughProcessor {
    async fn process_frame(
        &mut self,
        frame: Frame,
        direction: FrameDirection,
        ctx: &ProcessorContext,
    ) -> Result<()> {
        ctx.push_frame(frame, direction).await
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn is_sync(&self) -> bool {
        true
    }

    fn process_frame_sync(
        &mut self,
        frame: Frame,
        direction: FrameDirection,
        ctx: &ProcessorContext,
    ) -> Result<()> {
        ctx.try_push_frame(frame, direction)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backpressure::{BoundedFrameQueue, QueueConfig};
    use pipecat_core::FrameHeader;

    fn make_senders() -> (FrameQueueSender, BoundedFrameQueue, FrameQueueSender, BoundedFrameQueue)
    {
        let (ds_sender, ds_queue) = BoundedFrameQueue::new(
            QueueConfig {
                max_size: 100,
                ..Default::default()
            },
            "ds",
        );
        let (us_sender, us_queue) = BoundedFrameQueue::new(
            QueueConfig {
                max_size: 100,
                ..Default::default()
            },
            "us",
        );
        (ds_sender, ds_queue, us_sender, us_queue)
    }

    #[tokio::test]
    async fn passthrough_forwards_frame() {
        let (ds_sender, mut ds_queue, us_sender, _us_queue) = make_senders();
        let ctx = ProcessorContext::new(ds_sender, us_sender, PipelineClock::new());
        let mut proc = PassthroughProcessor::new("test");

        let frame = Frame::Start(FrameHeader::new());
        proc.process_frame(frame, FrameDirection::Downstream, &ctx)
            .await
            .unwrap();

        let env = ds_queue.get_nowait().unwrap();
        assert_eq!(env.frame.name(), "Start");
        assert_eq!(env.direction, FrameDirection::Downstream);
    }

    #[tokio::test]
    async fn passthrough_upstream() {
        let (ds_sender, _ds_queue, us_sender, mut us_queue) = make_senders();
        let ctx = ProcessorContext::new(ds_sender, us_sender, PipelineClock::new());
        let mut proc = PassthroughProcessor::new("test");

        let frame = Frame::Heartbeat(FrameHeader::new());
        proc.process_frame(frame, FrameDirection::Upstream, &ctx)
            .await
            .unwrap();

        let env = us_queue.get_nowait().unwrap();
        assert_eq!(env.frame.name(), "Heartbeat");
        assert_eq!(env.direction, FrameDirection::Upstream);
    }

    #[tokio::test]
    async fn processor_name() {
        let proc = PassthroughProcessor::new("my_proc");
        assert_eq!(proc.name(), "my_proc");
    }

    #[tokio::test]
    async fn setup_and_cleanup_default() {
        let (ds_sender, _ds_queue, us_sender, _us_queue) = make_senders();
        let ctx = ProcessorContext::new(ds_sender, us_sender, PipelineClock::new());
        let mut proc = PassthroughProcessor::new("test");
        proc.setup(&ctx).await.unwrap();
        proc.cleanup().await.unwrap();
    }

    #[tokio::test]
    async fn context_clock_works() {
        let (ds_sender, _ds_queue, us_sender, _us_queue) = make_senders();
        let ctx = ProcessorContext::new(ds_sender, us_sender, PipelineClock::new());
        let t = ctx.clock().now_us();
        // Should be very small since we just created it.
        assert!(t < 10_000);
    }
}
