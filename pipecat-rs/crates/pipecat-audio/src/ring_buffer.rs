//! Lock-free SPSC ring buffer for bridging real-time audio threads to async tasks.
//!
//! This module wraps [`rtrb::RingBuffer`] with audio-specific helpers:
//!
//! - **Frame-oriented API**: Read/write in units of audio frames (not bytes).
//! - **Real-time safe producer**: `write_frames()` never allocates, never locks,
//!   never makes syscalls — safe for audio callback threads.
//! - **Async consumer**: `read_frames_async()` uses a `tokio::sync::Notify` to
//!   bridge the SPSC consumer to a tokio task without polling.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────┐   rtrb (lock-free)   ┌──────────────────┐
//! │  Audio Capture    │──────────────────────▶│  Pipeline Task   │
//! │  (OS thread)      │                       │  (tokio task)    │
//! │  write_frames()   │                       │  read_frames()   │
//! └──────────────────┘                       └──────────────────┘
//! ```

use std::sync::Arc;

use rtrb::{Consumer, Producer, RingBuffer};

/// Producer half of an audio ring buffer (for the real-time audio thread).
///
/// All methods are lock-free and allocation-free — safe for audio callbacks.
pub struct AudioRingProducer {
    inner: Producer<u8>,
    /// Bytes per audio frame (e.g., 2 for 16-bit mono, 4 for 16-bit stereo).
    bytes_per_frame: usize,
    /// Optional notify handle to wake the async consumer.
    notify: Option<Arc<tokio::sync::Notify>>,
}

impl AudioRingProducer {
    /// Write audio frames into the ring buffer.
    ///
    /// Returns the number of frames actually written (may be less than requested
    /// if the buffer is full). **Never blocks, never allocates.**
    pub fn write_frames(&mut self, data: &[u8]) -> usize {
        let frame_count = data.len() / self.bytes_per_frame;
        let byte_count = frame_count * self.bytes_per_frame;

        // Use the simple push-based API for correctness. For audio buffers
        // (typically 640-1920 bytes), this is fast enough.
        let src = &data[..byte_count];
        let mut written = 0;
        for &byte in src {
            if self.inner.push(byte).is_err() {
                break; // Buffer full
            }
            written += 1;
        }

        if written > 0 {
            if let Some(ref notify) = self.notify {
                notify.notify_one();
            }
        }

        written / self.bytes_per_frame
    }

    /// Number of frames that can be written without dropping data.
    pub fn available_write_frames(&self) -> usize {
        self.inner.slots() / self.bytes_per_frame
    }

    /// Whether the consumer has been dropped (ring buffer is disconnected).
    pub fn is_abandoned(&self) -> bool {
        self.inner.is_abandoned()
    }
}

/// Consumer half of an audio ring buffer (for the async pipeline task).
pub struct AudioRingConsumer {
    inner: Consumer<u8>,
    /// Bytes per audio frame.
    bytes_per_frame: usize,
    /// Notify handle for async wakeup.
    notify: Arc<tokio::sync::Notify>,
}

impl AudioRingConsumer {
    /// Read available audio frames into the provided buffer.
    ///
    /// Returns the number of frames read (may be less than `max_frames`).
    /// This is non-blocking — returns 0 if no data is available.
    pub fn read_frames(&mut self, output: &mut Vec<u8>, max_frames: usize) -> usize {
        let max_bytes = max_frames * self.bytes_per_frame;
        let available = self.inner.slots().min(max_bytes);
        let frame_aligned = (available / self.bytes_per_frame) * self.bytes_per_frame;

        if frame_aligned == 0 {
            return 0;
        }

        match self.inner.read_chunk(frame_aligned) {
            Ok(chunk) => {
                let slices = chunk.as_slices();
                output.extend_from_slice(slices.0);
                output.extend_from_slice(slices.1);
                chunk.commit_all();
                frame_aligned / self.bytes_per_frame
            }
            Err(_) => 0,
        }
    }

    /// Wait for data to become available, then read frames.
    ///
    /// This is the async-friendly version that suspends the tokio task
    /// until the producer writes data (via the Notify handle).
    pub async fn read_frames_async(&mut self, output: &mut Vec<u8>, max_frames: usize) -> usize {
        loop {
            let read = self.read_frames(output, max_frames);
            if read > 0 {
                return read;
            }
            if self.is_abandoned() {
                return 0;
            }
            self.notify.notified().await;
        }
    }

    /// Number of frames available to read.
    pub fn available_read_frames(&self) -> usize {
        self.inner.slots() / self.bytes_per_frame
    }

    /// Whether the producer has been dropped (no more data will arrive).
    pub fn is_abandoned(&self) -> bool {
        self.inner.is_abandoned()
    }
}

/// Create a paired audio ring buffer with the given capacity in frames.
///
/// # Arguments
///
/// - `capacity_frames`: Number of audio frames the buffer can hold.
/// - `bytes_per_frame`: Bytes per audio frame (e.g., 2 for 16-bit mono).
///
/// Returns `(producer, consumer)` — the producer goes to the audio thread,
/// the consumer goes to the async pipeline task.
pub fn audio_ring_buffer(
    capacity_frames: usize,
    bytes_per_frame: usize,
) -> (AudioRingProducer, AudioRingConsumer) {
    let capacity_bytes = capacity_frames * bytes_per_frame;
    let (producer, consumer) = RingBuffer::new(capacity_bytes);
    let notify = Arc::new(tokio::sync::Notify::new());

    let prod = AudioRingProducer {
        inner: producer,
        bytes_per_frame,
        notify: Some(Arc::clone(&notify)),
    };
    let cons = AudioRingConsumer {
        inner: consumer,
        bytes_per_frame,
        notify,
    };
    (prod, cons)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_write_read() {
        let (mut prod, mut cons) = audio_ring_buffer(100, 2); // 100 frames × 2 bytes

        let data = vec![0x01, 0x02, 0x03, 0x04]; // 2 frames
        let written = prod.write_frames(&data);
        assert_eq!(written, 2);

        let mut output = Vec::new();
        let read = cons.read_frames(&mut output, 10);
        assert_eq!(read, 2);
        assert_eq!(output, data);
    }

    #[test]
    fn write_respects_capacity() {
        let (mut prod, _cons) = audio_ring_buffer(4, 2); // 4 frames × 2 bytes = 8 bytes

        let data = vec![0u8; 20]; // 10 frames — exceeds capacity
        let written = prod.write_frames(&data);
        assert!(written <= 4); // At most 4 frames can fit
    }

    #[test]
    fn frame_alignment() {
        let (mut prod, mut cons) = audio_ring_buffer(100, 4); // 4 bytes per frame

        // Write 3 bytes (not a full frame) — should write 0 frames
        let data = vec![0u8; 3];
        let written = prod.write_frames(&data);
        assert_eq!(written, 0);

        // Write 8 bytes (2 full frames)
        let data = vec![0u8; 8];
        let written = prod.write_frames(&data);
        assert_eq!(written, 2);

        let mut output = Vec::new();
        let read = cons.read_frames(&mut output, 100);
        assert_eq!(read, 2);
        assert_eq!(output.len(), 8);
    }

    #[test]
    fn abandoned_detection() {
        let (prod, cons) = audio_ring_buffer(10, 2);
        assert!(!cons.is_abandoned());
        drop(prod);
        assert!(cons.is_abandoned());
    }

    #[tokio::test]
    async fn async_read() {
        let (mut prod, mut cons) = audio_ring_buffer(100, 2);

        // Spawn a task that writes after a small delay
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            let data = vec![0xAB, 0xCD, 0xEF, 0x01];
            prod.write_frames(&data);
        });

        let mut output = Vec::new();
        let read = cons.read_frames_async(&mut output, 10).await;
        assert_eq!(read, 2);
        assert_eq!(output, vec![0xAB, 0xCD, 0xEF, 0x01]);
    }
}
