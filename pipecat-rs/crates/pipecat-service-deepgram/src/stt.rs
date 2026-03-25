//! Deepgram STT (Speech-to-Text) frame processor.
//!
//! [`DeepgramSTT`] connects to the Deepgram streaming WebSocket API and
//! converts incoming audio frames into transcription frames.
//!
//! The WebSocket connection is established lazily on the first `Start` frame
//! and torn down on `End` or `Cancel`.

use async_trait::async_trait;
use futures_util::{SinkExt, StreamExt};
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite;

use pipecat_core::{Frame, FrameDirection, FrameHeader, PipecatError, TranscriptionData};
use pipecat_pipeline::{FrameProcessor, ProcessorContext};

use crate::config::DeepgramConfig;
use crate::error::DeepgramError;
use crate::types::DeepgramResponse;

/// Deepgram streaming STT frame processor.
///
/// Receives `AudioRawInput` frames, sends PCM audio to Deepgram via WebSocket,
/// and pushes `Transcription` / `InterimTranscription` frames downstream.
///
/// # Lifecycle
///
/// - **Start frame**: Opens the WebSocket connection to Deepgram.
/// - **AudioRawInput frame**: Forwards audio bytes to the WebSocket; drains
///   any pending transcription frames and pushes them downstream.
/// - **SttMute frame**: Toggles muting (audio is not sent while muted).
/// - **End / Cancel frame**: Closes the WebSocket connection.
///
/// Audio frames are always passed through downstream regardless of mute state,
/// so downstream processors still receive the raw audio.
pub struct DeepgramSTT {
    name: String,
    config: DeepgramConfig,
    /// Sender for audio data to the WebSocket writer task.
    ws_tx: Option<mpsc::UnboundedSender<Vec<u8>>>,
    /// Receiver for transcription frames produced by the WebSocket reader task.
    transcript_rx: Option<mpsc::UnboundedReceiver<Frame>>,
    /// Handle to the combined WebSocket I/O task.
    ws_task: Option<tokio::task::JoinHandle<()>>,
    /// Whether audio sending is muted.
    muted: bool,
}

impl DeepgramSTT {
    /// Create a new Deepgram STT processor with the given configuration.
    ///
    /// The WebSocket connection is NOT established until a `Start` frame is
    /// processed.
    ///
    /// # Arguments
    ///
    /// * `config` - Deepgram configuration including API key and model settings.
    pub fn new(config: DeepgramConfig) -> Self {
        Self {
            name: "DeepgramSTT".to_string(),
            config,
            ws_tx: None,
            transcript_rx: None,
            ws_task: None,
            muted: false,
        }
    }

    /// Create a new Deepgram STT processor with a custom name.
    pub fn with_name(name: impl Into<String>, config: DeepgramConfig) -> Self {
        Self {
            name: name.into(),
            config,
            ws_tx: None,
            transcript_rx: None,
            ws_task: None,
            muted: false,
        }
    }

    /// Whether the processor is currently connected to Deepgram.
    pub fn is_connected(&self) -> bool {
        self.ws_tx.is_some()
    }

    /// Whether audio sending is currently muted.
    pub fn is_muted(&self) -> bool {
        self.muted
    }

    /// Connect to the Deepgram WebSocket API.
    ///
    /// Spawns two background tasks:
    /// - A writer task that reads from `ws_tx` and sends audio to the WebSocket.
    /// - A reader task that receives JSON from the WebSocket and sends parsed
    ///   transcription frames to `transcript_rx`.
    async fn connect(&mut self) -> crate::error::Result<()> {
        let url = self.config.websocket_url();

        let request = tungstenite::http::Request::builder()
            .uri(&url)
            .header(
                "Authorization",
                format!("Token {}", self.config.api_key),
            )
            .header(
                "Sec-WebSocket-Key",
                tungstenite::handshake::client::generate_key(),
            )
            .header("Connection", "Upgrade")
            .header("Upgrade", "websocket")
            .header("Sec-WebSocket-Version", "13")
            .header("Host", "api.deepgram.com")
            .body(())
            .map_err(|e| DeepgramError::ConnectionFailed(format!("failed to build request: {e}")))?;

        let (ws_stream, _response) = tokio_tungstenite::connect_async(request)
            .await
            .map_err(DeepgramError::WebSocket)?;

        let (mut write, mut read) = ws_stream.split();

        // Channel: audio bytes from process_frame -> WebSocket writer task
        let (audio_tx, mut audio_rx) = mpsc::unbounded_channel::<Vec<u8>>();
        // Channel: transcription frames from WebSocket reader task -> process_frame
        let (transcript_tx, transcript_rx) = mpsc::unbounded_channel::<Frame>();

        let processor_name = self.name.clone();

        // Spawn writer task: forwards audio bytes to the WebSocket
        let writer_task = tokio::spawn(async move {
            while let Some(audio) = audio_rx.recv().await {
                if let Err(e) = write.send(tungstenite::Message::Binary(audio)).await {
                    tracing::error!(error = %e, "deepgram websocket write error");
                    break;
                }
            }
            // Send CloseStream message to gracefully shut down the Deepgram session
            let close_msg = r#"{"type": "CloseStream"}"#;
            if let Err(e) = write.send(tungstenite::Message::Text(close_msg.into())).await {
                tracing::debug!(error = %e, "failed to send CloseStream (connection may already be closed)");
            }
        });

        // Spawn reader task: receives JSON transcription responses from Deepgram
        let reader_task = tokio::spawn(async move {
            while let Some(msg) = read.next().await {
                match msg {
                    Ok(tungstenite::Message::Text(text)) => {
                        match serde_json::from_str::<DeepgramResponse>(&text.to_string()) {
                            Ok(response) => {
                                if let Some(transcript) = response.transcript() {
                                    let transcript_text = transcript.to_string();
                                    let frame = if response.is_final() {
                                        Frame::Transcription {
                                            header: FrameHeader::new(),
                                            data: Box::new(TranscriptionData {
                                                text: transcript_text,
                                                user_id: None,
                                                timestamp: None,
                                                language: None,
                                            }),
                                        }
                                    } else {
                                        Frame::InterimTranscription {
                                            header: FrameHeader::new(),
                                            data: Box::new(TranscriptionData {
                                                text: transcript_text,
                                                user_id: None,
                                                timestamp: None,
                                                language: None,
                                            }),
                                        }
                                    };
                                    if transcript_tx.send(frame).is_err() {
                                        tracing::debug!("transcript receiver dropped, stopping reader");
                                        break;
                                    }
                                }
                            }
                            Err(e) => {
                                tracing::warn!(
                                    error = %e,
                                    json = %text,
                                    "failed to parse deepgram response"
                                );
                            }
                        }
                    }
                    Ok(tungstenite::Message::Close(_)) => {
                        tracing::debug!("deepgram websocket closed by server");
                        break;
                    }
                    Ok(_) => {
                        // Ignore binary, ping, pong frames
                    }
                    Err(e) => {
                        tracing::error!(error = %e, "deepgram websocket read error");
                        break;
                    }
                }
            }
        });

        // Spawn a supervisor task that waits for both I/O tasks to complete
        self.ws_task = Some(tokio::spawn(async move {
            let _ = tokio::join!(writer_task, reader_task);
        }));

        self.ws_tx = Some(audio_tx);
        self.transcript_rx = Some(transcript_rx);

        tracing::info!(
            processor = %processor_name,
            model = %self.config.model,
            "connected to deepgram"
        );

        Ok(())
    }

    /// Disconnect from the Deepgram WebSocket.
    ///
    /// Drops the audio sender (which causes the writer task to send
    /// `CloseStream` and exit), then waits for the supervisor task to complete
    /// with a timeout.
    async fn disconnect(&mut self) {
        if let Some(tx) = self.ws_tx.take() {
            // Dropping the sender causes the writer task to exit its recv loop
            // and send the CloseStream message.
            drop(tx);
        }

        if let Some(task) = self.ws_task.take() {
            match tokio::time::timeout(std::time::Duration::from_secs(2), task).await {
                Ok(Ok(())) => {
                    tracing::debug!(processor = %self.name, "deepgram websocket tasks completed");
                }
                Ok(Err(e)) => {
                    tracing::warn!(
                        processor = %self.name,
                        error = %e,
                        "deepgram websocket task panicked"
                    );
                }
                Err(_) => {
                    tracing::warn!(
                        processor = %self.name,
                        "deepgram websocket shutdown timed out"
                    );
                }
            }
        }

        self.transcript_rx = None;
    }

    /// Drain all available transcription frames from the receiver and push
    /// them downstream through the processor context.
    async fn drain_transcriptions(&mut self, ctx: &ProcessorContext) -> pipecat_core::Result<()> {
        if let Some(ref mut rx) = self.transcript_rx {
            while let Ok(frame) = rx.try_recv() {
                ctx.push_frame(frame, FrameDirection::Downstream).await?;
            }
        }
        Ok(())
    }
}

#[async_trait]
impl FrameProcessor for DeepgramSTT {
    fn name(&self) -> &str {
        &self.name
    }

    async fn process_frame(
        &mut self,
        frame: Frame,
        direction: FrameDirection,
        ctx: &ProcessorContext,
    ) -> pipecat_core::Result<()> {
        match &frame {
            Frame::Start(_) => {
                self.connect()
                    .await
                    .map_err(|e| PipecatError::Service(e.to_string()))?;
                ctx.push_frame(frame, direction).await?;
            }
            Frame::AudioRawInput { audio, .. } => {
                // Send audio to Deepgram if not muted and connected
                if !self.muted {
                    if let Some(ref tx) = self.ws_tx {
                        if tx.send(audio.audio.to_vec()).is_err() {
                            tracing::warn!(
                                processor = %self.name,
                                "failed to send audio to deepgram (channel closed)"
                            );
                        }
                    }
                }

                // Drain any received transcriptions and push them downstream
                self.drain_transcriptions(ctx).await?;

                // Always pass audio through downstream
                ctx.push_frame(frame, direction).await?;
            }
            Frame::SttMute { mute, .. } => {
                self.muted = *mute;
                tracing::debug!(
                    processor = %self.name,
                    muted = self.muted,
                    "stt mute state changed"
                );
                ctx.push_frame(frame, direction).await?;
            }
            Frame::End(_) | Frame::Cancel(_) => {
                // Drain any remaining transcriptions before disconnecting
                self.drain_transcriptions(ctx).await?;
                self.disconnect().await;
                ctx.push_frame(frame, direction).await?;
            }
            _ => {
                // Pass all other frames through unchanged
                ctx.push_frame(frame, direction).await?;
            }
        }

        Ok(())
    }

    async fn cleanup(&mut self) -> pipecat_core::Result<()> {
        self.disconnect().await;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pipecat_core::{AudioData, FrameHeader};
    use pipecat_pipeline::ProcessorContext;

    fn make_config() -> DeepgramConfig {
        DeepgramConfig {
            api_key: "test-key".into(),
            ..Default::default()
        }
    }

    fn make_ctx() -> (
        ProcessorContext,
        pipecat_pipeline::BoundedFrameQueue,
        pipecat_pipeline::BoundedFrameQueue,
    ) {
        use pipecat_pipeline::backpressure::{BoundedFrameQueue, QueueConfig};
        let (ds_sender, ds_queue) =
            BoundedFrameQueue::new(QueueConfig { max_size: 100, ..Default::default() }, "ds");
        let (us_sender, us_queue) =
            BoundedFrameQueue::new(QueueConfig { max_size: 100, ..Default::default() }, "us");
        let ctx = ProcessorContext::new(
            ds_sender,
            us_sender,
            pipecat_core::clock::PipelineClock::new(),
        );
        (ctx, ds_queue, us_queue)
    }

    #[test]
    fn new_creates_disconnected_processor() {
        let stt = DeepgramSTT::new(make_config());
        assert_eq!(stt.name(), "DeepgramSTT");
        assert!(!stt.is_connected());
        assert!(!stt.is_muted());
    }

    #[test]
    fn with_name_sets_custom_name() {
        let stt = DeepgramSTT::with_name("my-stt", make_config());
        assert_eq!(stt.name(), "my-stt");
    }

    #[tokio::test]
    async fn passthrough_non_audio_frames() {
        let mut stt = DeepgramSTT::new(make_config());
        let (ctx, mut ds_rx, _us_rx) = make_ctx();

        // A text frame should pass through unchanged
        let frame = Frame::Text {
            header: FrameHeader::new(),
            data: pipecat_core::TextData {
                text: "hello".into(),
            },
        };
        stt.process_frame(frame, FrameDirection::Downstream, &ctx)
            .await
            .unwrap();

        let env = ds_rx.get_nowait().unwrap();
        assert_eq!(env.frame.name(), "Text");
    }

    #[tokio::test]
    async fn audio_passes_through_when_not_connected() {
        let mut stt = DeepgramSTT::new(make_config());
        let (ctx, mut ds_rx, _us_rx) = make_ctx();

        let frame = Frame::AudioRawInput {
            header: FrameHeader::new(),
            audio: AudioData {
                audio: bytes::Bytes::from(vec![0u8; 640]),
                sample_rate: 16000,
                num_channels: 1,
            },
        };
        stt.process_frame(frame, FrameDirection::Downstream, &ctx)
            .await
            .unwrap();

        let env = ds_rx.get_nowait().unwrap();
        assert_eq!(env.frame.name(), "AudioRawInput");
    }

    #[tokio::test]
    async fn mute_frame_toggles_mute_state() {
        let mut stt = DeepgramSTT::new(make_config());
        let (ctx, mut ds_rx, _us_rx) = make_ctx();

        assert!(!stt.is_muted());

        // Mute
        let frame = Frame::SttMute {
            header: FrameHeader::new(),
            mute: true,
        };
        stt.process_frame(frame, FrameDirection::Downstream, &ctx)
            .await
            .unwrap();
        assert!(stt.is_muted());

        let env = ds_rx.get_nowait().unwrap();
        assert_eq!(env.frame.name(), "SttMute");

        // Unmute
        let frame = Frame::SttMute {
            header: FrameHeader::new(),
            mute: false,
        };
        stt.process_frame(frame, FrameDirection::Downstream, &ctx)
            .await
            .unwrap();
        assert!(!stt.is_muted());
    }

    #[tokio::test]
    async fn end_frame_passes_through_when_not_connected() {
        let mut stt = DeepgramSTT::new(make_config());
        let (ctx, mut ds_rx, _us_rx) = make_ctx();

        let frame = Frame::End(FrameHeader::new());
        stt.process_frame(frame, FrameDirection::Downstream, &ctx)
            .await
            .unwrap();

        let env = ds_rx.get_nowait().unwrap();
        assert_eq!(env.frame.name(), "End");
    }

    #[tokio::test]
    async fn cancel_frame_passes_through_when_not_connected() {
        let mut stt = DeepgramSTT::new(make_config());
        let (ctx, mut ds_rx, _us_rx) = make_ctx();

        let frame = Frame::Cancel(FrameHeader::new());
        stt.process_frame(frame, FrameDirection::Downstream, &ctx)
            .await
            .unwrap();

        let env = ds_rx.get_nowait().unwrap();
        assert_eq!(env.frame.name(), "Cancel");
    }

    #[tokio::test]
    async fn cleanup_is_idempotent() {
        let mut stt = DeepgramSTT::new(make_config());
        // Calling cleanup when not connected should not error
        stt.cleanup().await.unwrap();
        stt.cleanup().await.unwrap();
    }

    #[tokio::test]
    async fn upstream_frames_pass_through() {
        let mut stt = DeepgramSTT::new(make_config());
        let (ctx, _ds_rx, mut us_rx) = make_ctx();

        let frame = Frame::Heartbeat(FrameHeader::new());
        stt.process_frame(frame, FrameDirection::Upstream, &ctx)
            .await
            .unwrap();

        let env = us_rx.get_nowait().unwrap();
        assert_eq!(env.frame.name(), "Heartbeat");
        assert_eq!(env.direction, FrameDirection::Upstream);
    }
}
