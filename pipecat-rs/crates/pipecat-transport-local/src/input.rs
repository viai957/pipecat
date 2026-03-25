//! Local audio input processor using cpal.
//!
//! [`LocalAudioInput`] captures audio from a local input device (microphone)
//! and pushes [`Frame::AudioRawInput`] frames downstream into the pipeline.
//!
//! The real-time cpal callback writes raw PCM bytes into a lock-free
//! [`rtrb`] SPSC ring buffer. A tokio task polls the consumer side on a
//! timer and converts the data into frames.
//!
//! # Real-time safety
//!
//! The cpal callback performs **zero** allocations, locks, or syscalls.
//! Only `rtrb::Producer::push_partial_slice` is called, which is wait-free.

use async_trait::async_trait;
use bytes::Bytes;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rtrb::RingBuffer;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing;

use pipecat_core::{AudioData, Frame, FrameDirection, FrameHeader, Result};
use pipecat_pipeline::processor::{FrameProcessor, ProcessorContext};
use pipecat_pipeline::{FrameEnvelope, FrameQueueSender};

use crate::params::LocalAudioTransportParams;

/// Interval at which the reader task polls the ring buffer.
const POLL_INTERVAL_MS: u64 = 10;

/// Local audio input processor.
///
/// On receiving a `Start` frame this processor:
/// 1. Opens the requested (or default) cpal input device.
/// 2. Creates an `rtrb` SPSC ring buffer.
/// 3. Starts the cpal input stream whose callback writes into the producer.
/// 4. Spawns a tokio task that reads from the consumer and pushes
///    `AudioRawInput` frames downstream.
///
/// On `End` or `Cancel` the stream and reader task are torn down.
pub struct LocalAudioInput {
    name: String,
    params: LocalAudioTransportParams,
    /// Handle for signaling the dedicated audio thread to stop.
    /// `cpal::Stream` is `!Send` so it must live on a dedicated thread;
    /// we communicate via this channel.
    stream_shutdown_tx: Option<mpsc::Sender<()>>,
    /// Handle to the tokio reader task.
    read_task: Option<tokio::task::JoinHandle<()>>,
}

impl LocalAudioInput {
    /// Create a new local audio input processor.
    pub fn new(params: LocalAudioTransportParams) -> Self {
        Self {
            name: "LocalAudioInput".to_string(),
            params,
            stream_shutdown_tx: None,
            read_task: None,
        }
    }

    /// Create a new local audio input processor with a custom name.
    pub fn with_name(name: impl Into<String>, params: LocalAudioTransportParams) -> Self {
        Self {
            name: name.into(),
            params,
            stream_shutdown_tx: None,
            read_task: None,
        }
    }

    /// Start capturing audio from the local input device.
    async fn start_capture(&mut self, ctx: &ProcessorContext) -> Result<()> {
        let sample_rate = self.params.base.audio_in_sample_rate;
        let num_channels = self.params.base.audio_in_channels;
        let ring_buffer_size = self.params.ring_buffer_size;
        let device_name = self.params.input_device_name.clone();

        // Create the ring buffer pair.
        let (producer, consumer) = RingBuffer::<u8>::new(ring_buffer_size);

        // Channel for signaling the stream thread to shut down.
        let (shutdown_tx, mut shutdown_rx) = mpsc::channel::<()>(1);
        self.stream_shutdown_tx = Some(shutdown_tx);

        // Channel for the stream thread to report setup success/failure.
        let (stream_ready_tx, stream_ready_rx) =
            tokio::sync::oneshot::channel::<std::result::Result<(), String>>();

        // Spawn the cpal stream on a dedicated OS thread because
        // `cpal::Stream` is `!Send`.
        std::thread::Builder::new()
            .name("pipecat-audio-input".into())
            .spawn(move || {
                let result = run_input_stream(
                    device_name,
                    sample_rate,
                    num_channels,
                    producer,
                    &mut shutdown_rx,
                );
                let _ = stream_ready_tx.send(result.map_err(|e| e.to_string()));
            })
            .map_err(|e| {
                pipecat_core::PipecatError::Transport(format!(
                    "failed to spawn audio input thread: {e}"
                ))
            })?;

        // Wait for the stream to be ready (or fail).
        match stream_ready_rx.await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                return Err(pipecat_core::PipecatError::Transport(format!(
                    "audio input stream setup failed: {e}"
                )));
            }
            Err(_) => {
                return Err(pipecat_core::PipecatError::Transport(
                    "audio input thread exited before signaling ready".into(),
                ));
            }
        }

        // Spawn the tokio reader task that polls the consumer and pushes
        // AudioRawInput frames downstream.
        let downstream_tx = ctx.downstream_sender();
        let read_task = tokio::spawn(reader_task(
            consumer,
            sample_rate,
            num_channels,
            downstream_tx,
        ));
        self.read_task = Some(read_task);

        tracing::info!(
            name = %self.name,
            sample_rate,
            num_channels,
            ring_buffer_size,
            "audio input capture started"
        );

        Ok(())
    }

    /// Stop capturing audio.
    async fn stop_capture(&mut self) {
        // Signal the stream thread to shut down.
        if let Some(tx) = self.stream_shutdown_tx.take() {
            let _ = tx.send(()).await;
        }

        // Cancel the reader task.
        if let Some(task) = self.read_task.take() {
            task.abort();
            let _ = task.await;
        }

        tracing::info!(name = %self.name, "audio input capture stopped");
    }
}

/// Run the cpal input stream on a dedicated OS thread.
///
/// This function blocks until the shutdown signal is received. The
/// `cpal::Stream` is created and held on this thread since it is `!Send`.
fn run_input_stream(
    device_name: Option<String>,
    sample_rate: u32,
    num_channels: u16,
    mut producer: rtrb::Producer<u8>,
    shutdown_rx: &mut mpsc::Receiver<()>,
) -> std::result::Result<(), String> {
    let host = cpal::default_host();

    let device = match &device_name {
        Some(name) => {
            let devices = host
                .input_devices()
                .map_err(|e| format!("failed to enumerate input devices: {e}"))?;
            let mut found = None;
            for d in devices {
                if let Ok(n) = d.name() {
                    if n == *name {
                        found = Some(d);
                        break;
                    }
                }
            }
            found.ok_or_else(|| format!("input device not found: {name}"))?
        }
        None => host
            .default_input_device()
            .ok_or_else(|| "no default input device available".to_string())?,
    };

    let config = cpal::StreamConfig {
        channels: num_channels,
        sample_rate: cpal::SampleRate(sample_rate),
        buffer_size: cpal::BufferSize::Default,
    };

    let err_fn = |err: cpal::StreamError| {
        tracing::error!(%err, "cpal input stream error");
    };

    // Build the input stream with an i16 callback.
    //
    // RT-safety: the callback only calls `producer.push_partial_slice()`
    // which is wait-free. No allocations, no locks, no syscalls.
    let stream = device
        .build_input_stream(
            &config,
            move |data: &[i16], _info: &cpal::InputCallbackInfo| {
                // Reinterpret &[i16] as &[u8] for byte-level ring buffer.
                let byte_len = std::mem::size_of_val(data);
                let bytes: &[u8] =
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, byte_len) };

                // Write to ring buffer. If the buffer is full, excess samples
                // are silently dropped -- correct behavior for real-time audio.
                let _result = producer.push_partial_slice(bytes);
            },
            err_fn,
            None,
        )
        .map_err(|e| format!("failed to build input stream: {e}"))?;

    stream
        .play()
        .map_err(|e| format!("failed to start input stream: {e}"))?;

    // Block until shutdown is signaled.
    let _ = shutdown_rx.blocking_recv();

    // Stream is dropped here, which stops and closes it.
    drop(stream);
    Ok(())
}

/// Async task that polls the ring buffer consumer and pushes
/// `AudioRawInput` frames downstream.
async fn reader_task(
    mut consumer: rtrb::Consumer<u8>,
    sample_rate: u32,
    num_channels: u16,
    downstream_tx: Arc<FrameQueueSender>,
) {
    let mut buf = vec![0u8; 4096];
    let mut interval =
        tokio::time::interval(tokio::time::Duration::from_millis(POLL_INTERVAL_MS));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        interval.tick().await;

        // Read all available data from the ring buffer.
        let available = consumer.slots();
        if available == 0 {
            continue;
        }

        // Resize temp buffer if needed.
        if buf.len() < available {
            buf.resize(available, 0);
        }

        let (filled, _unfilled) = consumer.pop_partial_slice(&mut buf[..available]);
        let read = filled.len();
        if read == 0 {
            continue;
        }

        let audio_bytes = Bytes::copy_from_slice(&buf[..read]);
        let frame = Frame::AudioRawInput {
            header: FrameHeader::new(),
            audio: AudioData {
                audio: audio_bytes,
                sample_rate,
                num_channels,
            },
        };

        let envelope = FrameEnvelope {
            frame,
            direction: FrameDirection::Downstream,
        };

        if downstream_tx.send(envelope).await.is_err() {
            tracing::debug!("audio input reader: downstream channel closed, stopping");
            break;
        }
    }
}

#[async_trait]
impl FrameProcessor for LocalAudioInput {
    fn name(&self) -> &str {
        &self.name
    }

    async fn process_frame(
        &mut self,
        frame: Frame,
        direction: FrameDirection,
        ctx: &ProcessorContext,
    ) -> Result<()> {
        match &frame {
            Frame::Start(_) => {
                self.start_capture(ctx).await?;
                ctx.push_frame(frame, direction).await?;
            }
            Frame::End(_) | Frame::Cancel(_) => {
                self.stop_capture().await;
                ctx.push_frame(frame, direction).await?;
            }
            _ => {
                ctx.push_frame(frame, direction).await?;
            }
        }
        Ok(())
    }

    async fn cleanup(&mut self) -> Result<()> {
        self.stop_capture().await;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_implements_frame_processor() {
        fn assert_processor<T: FrameProcessor>() {}
        assert_processor::<LocalAudioInput>();
    }

    #[test]
    fn input_default_name() {
        let input = LocalAudioInput::new(LocalAudioTransportParams::default());
        assert_eq!(input.name(), "LocalAudioInput");
    }

    #[test]
    fn input_custom_name() {
        let input = LocalAudioInput::with_name("my_mic", LocalAudioTransportParams::default());
        assert_eq!(input.name(), "my_mic");
    }

    #[tokio::test]
    async fn ring_buffer_bridging() {
        // Test the rtrb ring buffer data flow in isolation (no cpal).
        let (mut producer, mut consumer) = RingBuffer::<u8>::new(256);

        // Simulate writing audio samples as a cpal callback would.
        let samples: Vec<u8> = (0..64).collect();
        let (pushed, remainder) = producer.push_partial_slice(&samples);
        assert_eq!(pushed.len(), 64);
        assert!(remainder.is_empty());

        // Read back from the consumer side as the reader task would.
        let mut buf = vec![0u8; 128];
        let (filled, _unfilled) = consumer.pop_partial_slice(&mut buf[..64]);
        assert_eq!(filled.len(), 64);
        assert_eq!(&buf[..64], &samples[..]);
    }

    #[tokio::test]
    async fn ring_buffer_overflow_drops_excess() {
        let (mut producer, mut consumer) = RingBuffer::<u8>::new(16);

        let data = vec![42u8; 32];
        let (pushed, remainder) = producer.push_partial_slice(&data);
        // Should write at most the buffer capacity.
        assert!(pushed.len() <= 16);
        assert!(!remainder.is_empty());

        // Drain.
        let mut buf = vec![0u8; 32];
        let (filled, _) = consumer.pop_partial_slice(&mut buf);
        assert_eq!(filled.len(), pushed.len());
    }

    #[tokio::test]
    async fn ring_buffer_empty_read() {
        let (_producer, mut consumer) = RingBuffer::<u8>::new(64);
        let mut buf = vec![0u8; 32];
        let (filled, unfilled) = consumer.pop_partial_slice(&mut buf);
        assert!(filled.is_empty());
        assert_eq!(unfilled.len(), 32);
    }
}
