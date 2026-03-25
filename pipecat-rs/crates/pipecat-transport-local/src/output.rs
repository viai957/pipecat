//! Local audio output processor using cpal.
//!
//! [`LocalAudioOutput`] receives [`Frame::AudioRawOutput`] frames from the
//! pipeline and plays them through a local output device (speakers/headphones).
//!
//! Audio bytes are written into a lock-free [`rtrb`] SPSC ring buffer.
//! The cpal output callback reads from the consumer side and fills the
//! hardware buffer, zero-filling on underrun.
//!
//! # Real-time safety
//!
//! The cpal output callback performs **zero** allocations, locks, or syscalls.
//! Only `rtrb::Consumer::pop_partial_slice` is called, which is wait-free.

use async_trait::async_trait;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rtrb::RingBuffer;
use std::sync::Mutex;
use tokio::sync::mpsc;
use tracing;

use pipecat_core::{Frame, FrameDirection, Result};
use pipecat_pipeline::processor::{FrameProcessor, ProcessorContext};

use crate::params::LocalAudioTransportParams;

/// Local audio output processor.
///
/// On receiving a `Start` frame this processor:
/// 1. Opens the requested (or default) cpal output device.
/// 2. Creates an `rtrb` SPSC ring buffer.
/// 3. Starts the cpal output stream whose callback reads from the consumer.
/// 4. Keeps the producer side so that `AudioRawOutput` frames can be written.
///
/// On `End` or `Cancel` the stream is torn down.
///
/// The `rtrb::Producer` is wrapped in a `Mutex` to satisfy the `Sync` bound
/// required by [`FrameProcessor`]. Since `process_frame` is called
/// sequentially, the mutex never actually contends.
pub struct LocalAudioOutput {
    name: String,
    params: LocalAudioTransportParams,
    /// Handle for signaling the dedicated audio thread to stop.
    stream_shutdown_tx: Option<mpsc::Sender<()>>,
    /// Producer side of the ring buffer, used to feed audio to the output
    /// stream. Wrapped in `Mutex` because `rtrb::Producer` is `Send` but
    /// not `Sync`, and `FrameProcessor` requires `Sync`.
    producer: Mutex<Option<rtrb::Producer<u8>>>,
}

impl LocalAudioOutput {
    /// Create a new local audio output processor.
    pub fn new(params: LocalAudioTransportParams) -> Self {
        Self {
            name: "LocalAudioOutput".to_string(),
            params,
            stream_shutdown_tx: None,
            producer: Mutex::new(None),
        }
    }

    /// Create a new local audio output processor with a custom name.
    pub fn with_name(name: impl Into<String>, params: LocalAudioTransportParams) -> Self {
        Self {
            name: name.into(),
            params,
            stream_shutdown_tx: None,
            producer: Mutex::new(None),
        }
    }

    /// Start playback on the local output device.
    async fn start_playback(&mut self) -> Result<()> {
        let sample_rate = self.params.base.audio_out_sample_rate;
        let num_channels = self.params.base.audio_out_channels;
        let ring_buffer_size = self.params.ring_buffer_size;
        let device_name = self.params.output_device_name.clone();

        // Create the ring buffer pair.
        let (producer, consumer) = RingBuffer::<u8>::new(ring_buffer_size);
        *self.producer.lock().unwrap() = Some(producer);

        // Channel for signaling the stream thread to shut down.
        let (shutdown_tx, mut shutdown_rx) = mpsc::channel::<()>(1);
        self.stream_shutdown_tx = Some(shutdown_tx);

        // Channel for the stream thread to report setup success/failure.
        let (stream_ready_tx, stream_ready_rx) =
            tokio::sync::oneshot::channel::<std::result::Result<(), String>>();

        // Spawn the cpal stream on a dedicated OS thread because
        // `cpal::Stream` is `!Send`.
        std::thread::Builder::new()
            .name("pipecat-audio-output".into())
            .spawn(move || {
                let result = run_output_stream(
                    device_name,
                    sample_rate,
                    num_channels,
                    consumer,
                    &mut shutdown_rx,
                );
                let _ = stream_ready_tx.send(result.map_err(|e| e.to_string()));
            })
            .map_err(|e| {
                pipecat_core::PipecatError::Transport(format!(
                    "failed to spawn audio output thread: {e}"
                ))
            })?;

        // Wait for the stream to be ready (or fail).
        match stream_ready_rx.await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                *self.producer.lock().unwrap() = None;
                return Err(pipecat_core::PipecatError::Transport(format!(
                    "audio output stream setup failed: {e}"
                )));
            }
            Err(_) => {
                *self.producer.lock().unwrap() = None;
                return Err(pipecat_core::PipecatError::Transport(
                    "audio output thread exited before signaling ready".into(),
                ));
            }
        }

        tracing::info!(
            name = %self.name,
            sample_rate,
            num_channels,
            ring_buffer_size,
            "audio output playback started"
        );

        Ok(())
    }

    /// Write audio bytes into the ring buffer for the output stream.
    ///
    /// If the ring buffer is full, excess bytes are silently dropped.
    fn write_audio(&self, audio_bytes: &[u8]) {
        let mut guard = self.producer.lock().unwrap();
        if let Some(ref mut producer) = *guard {
            let (pushed, remainder) = producer.push_partial_slice(audio_bytes);
            if !remainder.is_empty() {
                tracing::trace!(
                    pushed = pushed.len(),
                    dropped = remainder.len(),
                    "audio output ring buffer full, dropped bytes"
                );
            }
        }
    }

    /// Stop playback.
    async fn stop_playback(&mut self) {
        // Drop the producer so the output callback sees an empty buffer.
        *self.producer.lock().unwrap() = None;

        // Signal the stream thread to shut down.
        if let Some(tx) = self.stream_shutdown_tx.take() {
            let _ = tx.send(()).await;
        }

        tracing::info!(name = %self.name, "audio output playback stopped");
    }
}

/// Run the cpal output stream on a dedicated OS thread.
///
/// This function blocks until the shutdown signal is received.
fn run_output_stream(
    device_name: Option<String>,
    sample_rate: u32,
    num_channels: u16,
    mut consumer: rtrb::Consumer<u8>,
    shutdown_rx: &mut mpsc::Receiver<()>,
) -> std::result::Result<(), String> {
    let host = cpal::default_host();

    let device = match &device_name {
        Some(name) => {
            let devices = host
                .output_devices()
                .map_err(|e| format!("failed to enumerate output devices: {e}"))?;
            let mut found = None;
            for d in devices {
                if let Ok(n) = d.name() {
                    if n == *name {
                        found = Some(d);
                        break;
                    }
                }
            }
            found.ok_or_else(|| format!("output device not found: {name}"))?
        }
        None => host
            .default_output_device()
            .ok_or_else(|| "no default output device available".to_string())?,
    };

    let config = cpal::StreamConfig {
        channels: num_channels,
        sample_rate: cpal::SampleRate(sample_rate),
        buffer_size: cpal::BufferSize::Default,
    };

    let err_fn = |err: cpal::StreamError| {
        tracing::error!(%err, "cpal output stream error");
    };

    // Build the output stream with an i16 callback.
    //
    // RT-safety: the callback only calls `consumer.pop_partial_slice()` which
    // is wait-free. Zero-fill on underrun uses `fill(0)` on the stack
    // buffer -- no allocation.
    let stream = device
        .build_output_stream(
            &config,
            move |data: &mut [i16], _info: &cpal::OutputCallbackInfo| {
                // Reinterpret &mut [i16] as &mut [u8] for byte-level ring buffer.
                let byte_len = std::mem::size_of_val(data);
                let bytes: &mut [u8] = unsafe {
                    std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, byte_len)
                };

                // Read from ring buffer (wait-free).
                let (_filled, unfilled) = consumer.pop_partial_slice(bytes);

                // Zero-fill any unfilled portion (silence on underrun).
                unfilled.fill(0);
            },
            err_fn,
            None,
        )
        .map_err(|e| format!("failed to build output stream: {e}"))?;

    stream
        .play()
        .map_err(|e| format!("failed to start output stream: {e}"))?;

    // Block until shutdown is signaled.
    let _ = shutdown_rx.blocking_recv();

    drop(stream);
    Ok(())
}

#[async_trait]
impl FrameProcessor for LocalAudioOutput {
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
                self.start_playback().await?;
                ctx.push_frame(frame, direction).await?;
            }
            Frame::AudioRawOutput { audio, .. } => {
                self.write_audio(&audio.audio);
                ctx.push_frame(frame, direction).await?;
            }
            Frame::End(_) | Frame::Cancel(_) => {
                self.stop_playback().await;
                ctx.push_frame(frame, direction).await?;
            }
            _ => {
                ctx.push_frame(frame, direction).await?;
            }
        }
        Ok(())
    }

    async fn cleanup(&mut self) -> Result<()> {
        self.stop_playback().await;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_implements_frame_processor() {
        fn assert_processor<T: FrameProcessor>() {}
        assert_processor::<LocalAudioOutput>();
    }

    #[test]
    fn output_default_name() {
        let output = LocalAudioOutput::new(LocalAudioTransportParams::default());
        assert_eq!(output.name(), "LocalAudioOutput");
    }

    #[test]
    fn output_custom_name() {
        let output =
            LocalAudioOutput::with_name("my_speakers", LocalAudioTransportParams::default());
        assert_eq!(output.name(), "my_speakers");
    }

    #[tokio::test]
    async fn ring_buffer_output_bridging() {
        // Test the ring buffer data flow for output in isolation (no cpal).
        let (mut producer, mut consumer) = RingBuffer::<u8>::new(256);

        // Simulate the processor writing audio data.
        let audio_data: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let (pushed, remainder) = producer.push_partial_slice(&audio_data);
        assert_eq!(pushed.len(), 8);
        assert!(remainder.is_empty());

        // Simulate the cpal output callback reading data.
        let mut output_buf = vec![0u8; 16];
        let filled_len = {
            let (filled, unfilled) = consumer.pop_partial_slice(&mut output_buf);
            // Zero-fill unfilled portion (as the callback would).
            unfilled.fill(0);
            filled.len()
        };
        assert_eq!(filled_len, 8);
        assert_eq!(&output_buf[..8], &audio_data[..]);
        assert!(output_buf[8..].iter().all(|&b| b == 0));
    }

    #[tokio::test]
    async fn ring_buffer_underrun_zeros() {
        // When the consumer reads from an empty ring buffer, it gets nothing.
        let (_producer, mut consumer) = RingBuffer::<u8>::new(64);

        let mut buf = vec![0xFFu8; 16];
        let (filled, unfilled) = consumer.pop_partial_slice(&mut buf);
        assert!(filled.is_empty());
        assert_eq!(unfilled.len(), 16);

        // Zero-fill as the output callback would.
        unfilled.fill(0);
        assert!(buf.iter().all(|&b| b == 0));
    }

    #[test]
    fn write_audio_without_producer_is_noop() {
        let output = LocalAudioOutput::new(LocalAudioTransportParams::default());
        // No producer set -- write_audio should silently do nothing.
        output.write_audio(&[1, 2, 3, 4]);
    }

    #[test]
    fn write_audio_with_producer() {
        let (producer, mut consumer) = RingBuffer::<u8>::new(256);
        let output = LocalAudioOutput {
            name: "test".into(),
            params: LocalAudioTransportParams::default(),
            stream_shutdown_tx: None,
            producer: Mutex::new(Some(producer)),
        };

        let data = vec![10u8, 20, 30, 40];
        output.write_audio(&data);

        let mut buf = vec![0u8; 4];
        let (filled, _) = consumer.pop_partial_slice(&mut buf);
        assert_eq!(filled.len(), 4);
        assert_eq!(&buf, &data.as_slice());
    }
}
