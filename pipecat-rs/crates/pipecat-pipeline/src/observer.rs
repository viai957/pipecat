//! Pipeline observer trait for monitoring frame flow without modifying the pipeline.
//!
//! Observers are notified when a processor processes or pushes a frame. They
//! run outside the hot path and must not block the pipeline.

use async_trait::async_trait;
use pipecat_core::{Frame, FrameDirection};

/// Information about a frame that was processed by a processor.
#[derive(Debug, Clone)]
pub struct FrameProcessed {
    /// Name of the processor that processed the frame.
    pub processor_name: String,
    /// The frame that was processed.
    pub frame: Frame,
    /// Direction the frame was traveling.
    pub direction: FrameDirection,
    /// Pipeline clock timestamp in microseconds when the frame was processed.
    pub timestamp_us: u64,
}

/// Information about a frame that was pushed from one processor to another.
#[derive(Debug, Clone)]
pub struct FramePushed {
    /// Name of the processor that pushed the frame.
    pub source_name: String,
    /// Name of the destination processor.
    pub destination_name: String,
    /// The frame that was pushed.
    pub frame: Frame,
    /// Direction the frame was pushed.
    pub direction: FrameDirection,
    /// Pipeline clock timestamp in microseconds when the frame was pushed.
    pub timestamp_us: u64,
}

/// Trait for observing pipeline frame flow.
///
/// Observers are attached to a [`PipelineTask`](crate::task::PipelineTask) and
/// receive callbacks as frames move through the pipeline. Implementations must
/// be `Send + Sync` and should complete quickly to avoid stalling the observer
/// notification loop.
#[async_trait]
pub trait Observer: Send + Sync {
    /// Called after a processor has processed a frame.
    async fn on_process_frame(&self, _data: &FrameProcessed) {}

    /// Called after a processor has pushed a frame to the next stage.
    async fn on_push_frame(&self, _data: &FramePushed) {}
}

/// A no-op observer that does nothing. Useful as a default.
pub struct NullObserver;

#[async_trait]
impl Observer for NullObserver {
    async fn on_process_frame(&self, _data: &FrameProcessed) {}
    async fn on_push_frame(&self, _data: &FramePushed) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use pipecat_core::FrameHeader;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    struct CountingObserver {
        process_count: Arc<AtomicU32>,
        push_count: Arc<AtomicU32>,
    }

    #[async_trait]
    impl Observer for CountingObserver {
        async fn on_process_frame(&self, _data: &FrameProcessed) {
            self.process_count.fetch_add(1, Ordering::SeqCst);
        }

        async fn on_push_frame(&self, _data: &FramePushed) {
            self.push_count.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[tokio::test]
    async fn null_observer_does_nothing() {
        let obs = NullObserver;
        let data = FrameProcessed {
            processor_name: "test".into(),
            frame: Frame::Start(FrameHeader::new()),
            direction: FrameDirection::Downstream,
            timestamp_us: 0,
        };
        obs.on_process_frame(&data).await;
    }

    #[tokio::test]
    async fn counting_observer_tracks_calls() {
        let pc = Arc::new(AtomicU32::new(0));
        let puc = Arc::new(AtomicU32::new(0));
        let obs = CountingObserver {
            process_count: Arc::clone(&pc),
            push_count: Arc::clone(&puc),
        };

        let processed = FrameProcessed {
            processor_name: "p1".into(),
            frame: Frame::Start(FrameHeader::new()),
            direction: FrameDirection::Downstream,
            timestamp_us: 100,
        };
        obs.on_process_frame(&processed).await;
        obs.on_process_frame(&processed).await;

        let pushed = FramePushed {
            source_name: "p1".into(),
            destination_name: "p2".into(),
            frame: Frame::Start(FrameHeader::new()),
            direction: FrameDirection::Downstream,
            timestamp_us: 200,
        };
        obs.on_push_frame(&pushed).await;

        assert_eq!(pc.load(Ordering::SeqCst), 2);
        assert_eq!(puc.load(Ordering::SeqCst), 1);
    }
}
