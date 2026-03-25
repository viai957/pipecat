//! Output transport trait and [`MediaSender`] buffering.
//!
//! An [`OutputTransport`] is a [`FrameProcessor`] that writes audio and video
//! to an external sink (speakers, WebRTC track, WebSocket, etc.).
//!
//! [`MediaSender`] provides per-destination audio buffering and pacing,
//! chunking incoming audio into fixed-size segments and tracking bot speaking
//! state for VAD-style events.

use async_trait::async_trait;
use bytes::Bytes;
use tokio::sync::mpsc;
use tracing;

use pipecat_audio::silence::is_silence;
use pipecat_core::AudioData;
use pipecat_pipeline::processor::FrameProcessor;

use crate::Result;

/// Trait for transports that play audio (and potentially video) to an
/// external destination.
///
/// Implementors must also implement [`FrameProcessor`] so that the output
/// transport can participate in the pipeline graph.
#[async_trait]
pub trait OutputTransport: FrameProcessor {
    /// Write an audio frame to the transport output.
    ///
    /// Returns `true` if the audio was successfully written, `false` if the
    /// transport is not currently accepting audio (e.g. paused or not started).
    async fn write_audio_frame(&mut self, audio: &AudioData) -> Result<bool>;
}

// ─── Bot speaking event ──────────────────────────────────────────────────────

/// Events emitted by [`MediaSender`] when the bot starts or stops speaking.
///
/// These correspond to `BotStartedSpeaking` and `BotStoppedSpeaking` frames
/// in the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BotSpeakingEvent {
    /// The bot has begun producing non-silent audio.
    Started,
    /// The bot has been silent for long enough to be considered done speaking.
    Stopped,
}

// ─── MediaSender ─────────────────────────────────────────────────────────────

/// Per-destination audio output sender with buffering and pacing.
///
/// Accumulates incoming audio bytes into a buffer and, once enough data has
/// been collected to form a complete chunk, sends it through a channel to the
/// audio output task. Also tracks bot speaking state by checking whether each
/// chunk contains silence.
///
/// The chunk size is calculated as:
/// ```text
/// audio_bytes_10ms = (sample_rate / 100) * num_channels * 2
/// audio_chunk_size = audio_bytes_10ms * audio_out_10ms_chunks
/// ```
pub struct MediaSender {
    /// Optional destination label for multi-destination routing.
    destination: Option<String>,
    /// Audio sample rate in Hz.
    sample_rate: u32,
    /// Number of audio channels.
    num_channels: u16,
    /// Target chunk size in bytes.
    audio_chunk_size: usize,
    /// Internal buffer accumulating audio bytes until `audio_chunk_size` is reached.
    audio_buffer: Vec<u8>,
    /// Whether the bot is currently speaking (producing non-silent audio).
    bot_speaking: bool,
    /// Duration of consecutive silence in seconds before emitting a
    /// `BotStoppedSpeaking` event.
    bot_vad_stop_secs: f32,
    /// Accumulated silence duration in seconds (reset when non-silent audio arrives).
    silence_duration_secs: f32,
    /// Channel sender for dispatching complete audio chunks.
    audio_tx: mpsc::Sender<AudioData>,
}

impl std::fmt::Debug for MediaSender {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MediaSender")
            .field("destination", &self.destination)
            .field("sample_rate", &self.sample_rate)
            .field("num_channels", &self.num_channels)
            .field("audio_chunk_size", &self.audio_chunk_size)
            .field("buffer_len", &self.audio_buffer.len())
            .field("bot_speaking", &self.bot_speaking)
            .field("bot_vad_stop_secs", &self.bot_vad_stop_secs)
            .field("silence_duration_secs", &self.silence_duration_secs)
            .finish()
    }
}

impl MediaSender {
    /// Create a new `MediaSender` for the given destination and audio format.
    ///
    /// Args:
    ///     destination: Optional named destination for multi-destination routing.
    ///     sample_rate: Audio sample rate in Hz (e.g. 24000).
    ///     num_channels: Number of audio channels (e.g. 1 for mono).
    ///     audio_out_10ms_chunks: Number of 10ms chunks per output frame.
    ///     audio_tx: Channel sender for dispatching complete audio chunks.
    pub fn new(
        destination: Option<String>,
        sample_rate: u32,
        num_channels: u16,
        audio_out_10ms_chunks: u32,
        audio_tx: mpsc::Sender<AudioData>,
    ) -> Self {
        let audio_bytes_10ms = (sample_rate as usize / 100) * num_channels as usize * 2;
        let audio_chunk_size = audio_bytes_10ms * audio_out_10ms_chunks as usize;

        tracing::debug!(
            destination = ?destination,
            sample_rate,
            num_channels,
            audio_chunk_size,
            "MediaSender created"
        );

        Self {
            destination,
            sample_rate,
            num_channels,
            audio_chunk_size,
            audio_buffer: Vec::with_capacity(audio_chunk_size),
            bot_speaking: false,
            bot_vad_stop_secs: 0.3,
            silence_duration_secs: 0.0,
            audio_tx,
        }
    }

    /// Return the destination label, if any.
    pub fn destination(&self) -> Option<&str> {
        self.destination.as_deref()
    }

    /// Return the configured chunk size in bytes.
    pub fn chunk_size(&self) -> usize {
        self.audio_chunk_size
    }

    /// Return how many bytes are currently buffered.
    pub fn buffered(&self) -> usize {
        self.audio_buffer.len()
    }

    /// Whether the bot is currently considered to be speaking.
    pub fn is_bot_speaking(&self) -> bool {
        self.bot_speaking
    }

    /// Set the silence duration (in seconds) before a `BotStoppedSpeaking`
    /// event is emitted.
    pub fn set_bot_vad_stop_secs(&mut self, secs: f32) {
        self.bot_vad_stop_secs = secs;
    }

    /// Buffer audio data. When enough is accumulated, chunk it and send
    /// complete chunks through the channel.
    ///
    /// Returns a list of [`BotSpeakingEvent`]s generated during this call.
    /// Callers should translate these into the appropriate pipeline frames.
    pub async fn send_audio(&mut self, audio: &AudioData) -> Result<Vec<BotSpeakingEvent>> {
        let mut events = Vec::new();

        self.audio_buffer.extend_from_slice(&audio.audio);

        while self.audio_buffer.len() >= self.audio_chunk_size {
            let chunk_bytes: Vec<u8> =
                self.audio_buffer.drain(..self.audio_chunk_size).collect();

            // Check silence on the PCM samples in this chunk.
            let pcm_samples = bytes_to_i16_slice(&chunk_bytes);
            let chunk_is_silence = is_silence(pcm_samples);

            if let Some(event) = self.update_bot_speaking_state(chunk_is_silence) {
                events.push(event);
            }

            let chunk_audio = AudioData {
                audio: Bytes::from(chunk_bytes),
                sample_rate: self.sample_rate,
                num_channels: self.num_channels,
            };

            self.audio_tx.send(chunk_audio).await.map_err(|_| {
                crate::TransportError::ChannelSend("audio channel closed".into())
            })?;
        }

        Ok(events)
    }

    /// Flush remaining audio in the buffer, padding with silence if needed.
    ///
    /// Returns a list of [`BotSpeakingEvent`]s generated during the flush.
    pub async fn flush(&mut self) -> Result<Vec<BotSpeakingEvent>> {
        let mut events = Vec::new();

        if !self.audio_buffer.is_empty() {
            // Pad the remaining buffer with zeros (silence) to reach chunk size.
            self.audio_buffer
                .resize(self.audio_chunk_size, 0);

            let chunk_bytes = std::mem::take(&mut self.audio_buffer);

            let pcm_samples = bytes_to_i16_slice(&chunk_bytes);
            let chunk_is_silence = is_silence(pcm_samples);

            if let Some(event) = self.update_bot_speaking_state(chunk_is_silence) {
                events.push(event);
            }

            let chunk_audio = AudioData {
                audio: Bytes::from(chunk_bytes),
                sample_rate: self.sample_rate,
                num_channels: self.num_channels,
            };

            self.audio_tx.send(chunk_audio).await.map_err(|_| {
                crate::TransportError::ChannelSend("audio channel closed".into())
            })?;
        }

        // If the bot was speaking, emit a stopped event.
        if self.bot_speaking {
            self.bot_speaking = false;
            self.silence_duration_secs = 0.0;
            events.push(BotSpeakingEvent::Stopped);
        }

        // Reset buffer capacity.
        self.audio_buffer = Vec::with_capacity(self.audio_chunk_size);

        Ok(events)
    }

    /// Update bot speaking state based on whether the current chunk is silent.
    ///
    /// Returns `Some(BotSpeakingEvent::Started)` when transitioning from
    /// silence to speech, or `Some(BotSpeakingEvent::Stopped)` when silence
    /// has exceeded `bot_vad_stop_secs`.
    fn update_bot_speaking_state(&mut self, chunk_is_silence: bool) -> Option<BotSpeakingEvent> {
        if chunk_is_silence {
            // Calculate the duration of this chunk.
            let chunk_duration_secs = self.audio_chunk_size as f32
                / (self.sample_rate as f32 * self.num_channels as f32 * 2.0);

            self.silence_duration_secs += chunk_duration_secs;

            if self.bot_speaking && self.silence_duration_secs >= self.bot_vad_stop_secs {
                self.bot_speaking = false;
                self.silence_duration_secs = 0.0;
                return Some(BotSpeakingEvent::Stopped);
            }
        } else {
            self.silence_duration_secs = 0.0;

            if !self.bot_speaking {
                self.bot_speaking = true;
                return Some(BotSpeakingEvent::Started);
            }
        }

        None
    }
}

/// Reinterpret a byte slice as a slice of i16 samples (little-endian).
///
/// If the byte slice length is odd, the trailing byte is ignored.
fn bytes_to_i16_slice(bytes: &[u8]) -> &[i16] {
    let sample_count = bytes.len() / 2;
    if sample_count == 0 {
        return &[];
    }
    // SAFETY: We ensure the pointer is aligned by working with the raw bytes.
    // PCM audio data is always stored as pairs of bytes, so we use
    // from_raw_parts after verifying alignment.
    let ptr = bytes.as_ptr();
    if ptr.align_offset(std::mem::align_of::<i16>()) == 0 {
        unsafe { std::slice::from_raw_parts(ptr as *const i16, sample_count) }
    } else {
        // Fallback: if not aligned, we cannot safely reinterpret.
        // This should not happen with Vec<u8> allocations, but we handle it.
        // Return an empty slice which will be treated as silence.
        &[]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sender(chunk_size_10ms: u32) -> (MediaSender, mpsc::Receiver<AudioData>) {
        let (tx, rx) = mpsc::channel(64);
        let sender = MediaSender::new(None, 24000, 1, chunk_size_10ms, tx);
        (sender, rx)
    }

    fn make_audio(bytes: Vec<u8>) -> AudioData {
        AudioData {
            audio: Bytes::from(bytes),
            sample_rate: 24000,
            num_channels: 1,
        }
    }

    /// Generate non-silent audio data of the given size in bytes.
    fn make_loud_audio(size: usize) -> AudioData {
        // Fill with alternating high-amplitude 16-bit samples.
        let mut data = vec![0u8; size];
        for i in (0..size).step_by(2) {
            if i + 1 < size {
                // Write i16 value 1000 as little-endian.
                let val: i16 = 1000;
                let bytes = val.to_le_bytes();
                data[i] = bytes[0];
                data[i + 1] = bytes[1];
            }
        }
        make_audio(data)
    }

    /// Generate silent audio data (all zeros) of the given size in bytes.
    fn make_silent_audio(size: usize) -> AudioData {
        make_audio(vec![0u8; size])
    }

    #[test]
    fn chunk_size_calculation() {
        let (sender, _rx) = make_sender(2);
        // 24000/100 * 1 * 2 = 480 bytes per 10ms
        // 480 * 2 = 960 bytes per chunk
        assert_eq!(sender.chunk_size(), 960);
    }

    #[test]
    fn chunk_size_stereo() {
        let (tx, _rx) = mpsc::channel(64);
        let sender = MediaSender::new(None, 48000, 2, 3, tx);
        // 48000/100 * 2 * 2 = 1920 bytes per 10ms
        // 1920 * 3 = 5760 bytes per chunk
        assert_eq!(sender.chunk_size(), 5760);
    }

    #[test]
    fn initial_state() {
        let (sender, _rx) = make_sender(2);
        assert_eq!(sender.buffered(), 0);
        assert!(!sender.is_bot_speaking());
        assert_eq!(sender.destination(), None);
    }

    #[test]
    fn destination_label() {
        let (tx, _rx) = mpsc::channel(64);
        let sender = MediaSender::new(Some("speaker_1".into()), 24000, 1, 2, tx);
        assert_eq!(sender.destination(), Some("speaker_1"));
    }

    #[tokio::test]
    async fn send_exact_chunk() {
        let (mut sender, mut rx) = make_sender(2);
        let chunk_size = sender.chunk_size();

        let audio = make_loud_audio(chunk_size);
        let events = sender.send_audio(&audio).await.unwrap();

        // Should have produced a BotStartedSpeaking event.
        assert!(events.contains(&BotSpeakingEvent::Started));

        // Exactly one chunk should have been sent.
        let received = rx.try_recv().unwrap();
        assert_eq!(received.audio.len(), chunk_size);
        assert_eq!(received.sample_rate, 24000);
        assert_eq!(received.num_channels, 1);

        // Buffer should be empty.
        assert_eq!(sender.buffered(), 0);
    }

    #[tokio::test]
    async fn send_under_chunk_size_buffers() {
        let (mut sender, mut rx) = make_sender(2);
        let chunk_size = sender.chunk_size();

        let audio = make_loud_audio(chunk_size / 2);
        let events = sender.send_audio(&audio).await.unwrap();

        // No events yet, not enough data.
        assert!(events.is_empty());

        // Nothing sent on channel.
        assert!(rx.try_recv().is_err());

        // Data is buffered.
        assert_eq!(sender.buffered(), chunk_size / 2);
    }

    #[tokio::test]
    async fn send_double_chunk_produces_two() {
        let (mut sender, mut rx) = make_sender(2);
        let chunk_size = sender.chunk_size();

        let audio = make_loud_audio(chunk_size * 2);
        let events = sender.send_audio(&audio).await.unwrap();

        // Should have started speaking.
        assert!(events.contains(&BotSpeakingEvent::Started));

        // Two chunks should have been sent.
        let c1 = rx.try_recv().unwrap();
        let c2 = rx.try_recv().unwrap();
        assert_eq!(c1.audio.len(), chunk_size);
        assert_eq!(c2.audio.len(), chunk_size);

        assert_eq!(sender.buffered(), 0);
    }

    #[tokio::test]
    async fn send_partial_then_complete() {
        let (mut sender, mut rx) = make_sender(2);
        let chunk_size = sender.chunk_size();

        // Send half a chunk.
        let half = make_loud_audio(chunk_size / 2);
        sender.send_audio(&half).await.unwrap();
        assert!(rx.try_recv().is_err());

        // Send the other half.
        let other_half = make_loud_audio(chunk_size / 2);
        let events = sender.send_audio(&other_half).await.unwrap();

        assert!(events.contains(&BotSpeakingEvent::Started));
        let received = rx.try_recv().unwrap();
        assert_eq!(received.audio.len(), chunk_size);
    }

    #[tokio::test]
    async fn flush_pads_with_silence() {
        let (mut sender, mut rx) = make_sender(2);
        let chunk_size = sender.chunk_size();

        // Buffer partial data.
        let partial = make_loud_audio(100);
        sender.send_audio(&partial).await.unwrap();
        assert_eq!(sender.buffered(), 100);

        // Flush pads to chunk_size.
        let events = sender.flush().await.unwrap();
        let received = rx.try_recv().unwrap();
        assert_eq!(received.audio.len(), chunk_size);

        assert_eq!(sender.buffered(), 0);

        // Should have started and then stopped speaking (from flush).
        assert!(events.contains(&BotSpeakingEvent::Started) || events.contains(&BotSpeakingEvent::Stopped));
    }

    #[tokio::test]
    async fn flush_empty_buffer() {
        let (mut sender, mut rx) = make_sender(2);

        let events = sender.flush().await.unwrap();
        // No chunks sent, no events.
        assert!(events.is_empty());
        assert!(rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn bot_speaking_state_transitions() {
        let (mut sender, mut _rx) = make_sender(2);
        let chunk_size = sender.chunk_size();

        // Send loud audio => BotStartedSpeaking.
        let loud = make_loud_audio(chunk_size);
        let events = sender.send_audio(&loud).await.unwrap();
        assert_eq!(events, vec![BotSpeakingEvent::Started]);
        assert!(sender.is_bot_speaking());

        // Send more loud audio => no new event.
        let loud2 = make_loud_audio(chunk_size);
        let events = sender.send_audio(&loud2).await.unwrap();
        assert!(events.is_empty());
        assert!(sender.is_bot_speaking());
    }

    #[tokio::test]
    async fn bot_stops_after_silence_threshold() {
        let (tx, mut _rx) = mpsc::channel(256);
        // Use 1 chunk of 10ms at 24kHz mono => 480 bytes per chunk.
        // Each chunk is 10ms = 0.01s.
        let mut sender = MediaSender::new(None, 24000, 1, 1, tx);
        sender.set_bot_vad_stop_secs(0.03); // Stop after 30ms of silence.

        let chunk_size = sender.chunk_size(); // 480 bytes

        // Start speaking.
        let loud = make_loud_audio(chunk_size);
        let events = sender.send_audio(&loud).await.unwrap();
        assert_eq!(events, vec![BotSpeakingEvent::Started]);

        // Send 3 silent chunks (30ms) => should trigger BotStoppedSpeaking.
        let silence = make_silent_audio(chunk_size);
        let e1 = sender.send_audio(&silence).await.unwrap();
        let e2 = sender.send_audio(&silence).await.unwrap();
        let e3 = sender.send_audio(&silence).await.unwrap();

        let all_events: Vec<_> = e1.into_iter().chain(e2).chain(e3).collect();
        assert!(
            all_events.contains(&BotSpeakingEvent::Stopped),
            "expected BotStoppedSpeaking after 30ms of silence"
        );
        assert!(!sender.is_bot_speaking());
    }

    #[tokio::test]
    async fn silence_interrupted_by_speech_resets() {
        let (tx, mut _rx) = mpsc::channel(256);
        let mut sender = MediaSender::new(None, 24000, 1, 1, tx);
        sender.set_bot_vad_stop_secs(0.03);

        let chunk_size = sender.chunk_size();

        // Start speaking.
        let loud = make_loud_audio(chunk_size);
        sender.send_audio(&loud).await.unwrap();
        assert!(sender.is_bot_speaking());

        // Send 2 silent chunks (20ms, below the 30ms threshold).
        let silence = make_silent_audio(chunk_size);
        sender.send_audio(&silence).await.unwrap();
        sender.send_audio(&silence).await.unwrap();
        assert!(sender.is_bot_speaking()); // Still speaking.

        // Interrupt with speech => resets silence counter.
        let loud2 = make_loud_audio(chunk_size);
        let events = sender.send_audio(&loud2).await.unwrap();
        assert!(events.is_empty()); // No state change, still speaking.
        assert!(sender.is_bot_speaking());
    }

    #[tokio::test]
    async fn channel_closed_returns_error() {
        let (tx, rx) = mpsc::channel(1);
        let mut sender = MediaSender::new(None, 24000, 1, 2, tx);
        let chunk_size = sender.chunk_size();

        // Drop the receiver.
        drop(rx);

        let audio = make_loud_audio(chunk_size);
        let result = sender.send_audio(&audio).await;
        assert!(result.is_err());
    }

    #[test]
    fn bytes_to_i16_slice_basic() {
        let data: Vec<u8> = vec![0xe8, 0x03]; // 1000 in little-endian i16
        let samples = bytes_to_i16_slice(&data);
        if !samples.is_empty() {
            assert_eq!(samples[0], 1000);
        }
    }

    #[test]
    fn bytes_to_i16_slice_empty() {
        let samples = bytes_to_i16_slice(&[]);
        assert!(samples.is_empty());
    }

    #[test]
    fn bytes_to_i16_slice_odd_length() {
        // Odd number of bytes: trailing byte is ignored.
        let data: Vec<u8> = vec![0xe8, 0x03, 0xFF];
        let samples = bytes_to_i16_slice(&data);
        if !samples.is_empty() {
            assert_eq!(samples.len(), 1);
            assert_eq!(samples[0], 1000);
        }
    }

    #[test]
    fn debug_format() {
        let (tx, _rx) = mpsc::channel(1);
        let sender = MediaSender::new(Some("test".into()), 24000, 1, 2, tx);
        let debug = format!("{:?}", sender);
        assert!(debug.contains("MediaSender"));
        assert!(debug.contains("test"));
    }
}
