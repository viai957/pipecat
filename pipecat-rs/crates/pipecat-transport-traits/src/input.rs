//! Input transport trait.
//!
//! An [`InputTransport`] is a [`FrameProcessor`] that additionally provides
//! methods for pushing audio from the transport layer into the pipeline and
//! controlling the audio input stream.

use async_trait::async_trait;

use pipecat_core::AudioData;
use pipecat_pipeline::processor::FrameProcessor;

use crate::Result;

/// Trait for transports that capture audio (and potentially video) from an
/// external source and inject it into the pipeline.
///
/// Implementors must also implement [`FrameProcessor`] so that the input
/// transport can participate in the pipeline graph.
#[async_trait]
pub trait InputTransport: FrameProcessor {
    /// Push incoming audio from the transport layer into the pipeline.
    ///
    /// The implementation should convert the raw audio into the appropriate
    /// frame variant (e.g. `Frame::AudioRawInput`) and push it downstream
    /// via the processor context.
    async fn push_audio_frame(&self, audio: AudioData) -> Result<()>;

    /// Start the audio input stream.
    ///
    /// This is transport-specific: it might open a microphone device, connect
    /// to a WebRTC track, or begin reading from a WebSocket.
    async fn start_audio_in(&mut self) -> Result<()>;

    /// Stop the audio input stream.
    ///
    /// After this call, no more audio frames should be produced until
    /// [`start_audio_in`](InputTransport::start_audio_in) is called again.
    async fn stop_audio_in(&mut self) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use bytes::Bytes;
    use pipecat_core::{Frame, FrameDirection, FrameHeader};
    use pipecat_pipeline::processor::ProcessorContext;

    /// A mock input transport for testing the trait.
    struct MockInputTransport {
        started: bool,
        frames_pushed: std::sync::Arc<std::sync::atomic::AtomicU32>,
    }

    impl MockInputTransport {
        fn new() -> Self {
            Self {
                started: false,
                frames_pushed: std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0)),
            }
        }
    }

    #[async_trait]
    impl FrameProcessor for MockInputTransport {
        async fn process_frame(
            &mut self,
            frame: Frame,
            direction: FrameDirection,
            ctx: &ProcessorContext,
        ) -> pipecat_core::Result<()> {
            ctx.push_frame(frame, direction).await
        }

        fn name(&self) -> &str {
            "MockInputTransport"
        }
    }

    #[async_trait]
    impl InputTransport for MockInputTransport {
        async fn push_audio_frame(&self, _audio: AudioData) -> Result<()> {
            self.frames_pushed
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(())
        }

        async fn start_audio_in(&mut self) -> Result<()> {
            if self.started {
                return Err(crate::TransportError::AlreadyStarted);
            }
            self.started = true;
            Ok(())
        }

        async fn stop_audio_in(&mut self) -> Result<()> {
            if !self.started {
                return Err(crate::TransportError::NotStarted);
            }
            self.started = false;
            Ok(())
        }
    }

    #[tokio::test]
    async fn mock_start_stop() {
        let mut transport = MockInputTransport::new();
        assert!(!transport.started);

        transport.start_audio_in().await.unwrap();
        assert!(transport.started);

        // Starting again should fail.
        let err = transport.start_audio_in().await.unwrap_err();
        assert!(matches!(err, crate::TransportError::AlreadyStarted));

        transport.stop_audio_in().await.unwrap();
        assert!(!transport.started);

        // Stopping again should fail.
        let err = transport.stop_audio_in().await.unwrap_err();
        assert!(matches!(err, crate::TransportError::NotStarted));
    }

    #[tokio::test]
    async fn mock_push_audio() {
        let transport = MockInputTransport::new();
        let audio = AudioData {
            audio: Bytes::from(vec![0u8; 640]),
            sample_rate: 16000,
            num_channels: 1,
        };
        transport.push_audio_frame(audio).await.unwrap();
        assert_eq!(
            transport
                .frames_pushed
                .load(std::sync::atomic::Ordering::SeqCst),
            1
        );
    }

    #[tokio::test]
    async fn mock_processor_name() {
        let transport = MockInputTransport::new();
        assert_eq!(transport.name(), "MockInputTransport");
    }

    #[tokio::test]
    async fn mock_process_frame_forwards() {
        use pipecat_core::clock::PipelineClock;
        use pipecat_pipeline::backpressure::{BoundedFrameQueue, QueueConfig};

        let (ds_sender, mut ds_queue) =
            BoundedFrameQueue::new(QueueConfig { max_size: 100, ..Default::default() }, "ds");
        let (us_sender, _us_queue) =
            BoundedFrameQueue::new(QueueConfig { max_size: 100, ..Default::default() }, "us");
        let ctx = ProcessorContext::new(ds_sender, us_sender, PipelineClock::new());

        let mut transport = MockInputTransport::new();
        let frame = Frame::Start(FrameHeader::new());
        transport
            .process_frame(frame, FrameDirection::Downstream, &ctx)
            .await
            .unwrap();

        let env = ds_queue.get_nowait().unwrap();
        assert_eq!(env.frame.name(), "Start");
    }
}
