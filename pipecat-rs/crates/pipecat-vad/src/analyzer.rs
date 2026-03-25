//! VAD analyzer: pluggable model trait and hysteresis state machine.
//!
//! This module is a direct port of Python's
//! `pipecat.audio.vad.vad_analyzer.VADAnalyzer`. The state machine uses
//! configurable start/stop frame counts derived from [`VadParams`] to provide
//! hysteresis, rejecting brief flickers of speech or silence.

use pipecat_audio::volume::{calculate_audio_volume, exp_smoothing};

use crate::params::{VadParams, VadState};

/// Trait for VAD model backends.
///
/// Implementations wrap a specific inference engine (e.g. Silero ONNX) and
/// provide frame-level voice confidence scores. The trait is intentionally
/// synchronous — callers should run inference on a blocking thread pool.
pub trait VadModel: Send + 'static {
    /// Number of audio frames (samples) required per inference call.
    ///
    /// For example, Silero at 16 kHz requires 512 samples.
    fn num_frames_required(&self) -> usize;

    /// Compute voice confidence for the given audio buffer (PCM i16).
    ///
    /// Returns a confidence value between 0.0 (silence) and 1.0 (speech).
    fn voice_confidence(&mut self, buffer: &[i16]) -> f32;

    /// Reset model internal state (e.g. RNN hidden state).
    fn reset_states(&mut self);
}

/// VAD analyzer with a hysteresis state machine.
///
/// Wraps a [`VadModel`] and buffers incoming audio, running inference when
/// enough data has accumulated. The state machine requires `vad_start_frames`
/// consecutive speaking frames to transition from [`VadState::Quiet`] to
/// [`VadState::Speaking`], and `vad_stop_frames` consecutive quiet frames to
/// transition back — matching the Python implementation exactly.
pub struct VadAnalyzer<M: VadModel> {
    model: M,
    params: VadParams,
    sample_rate: u32,
    num_channels: u16,

    // Internal state
    state: VadState,
    vad_buffer: Vec<u8>,
    vad_frames_num_bytes: usize,
    vad_start_frames: usize,
    vad_stop_frames: usize,
    starting_count: usize,
    stopping_count: usize,

    // Volume smoothing
    smoothing_factor: f32,
    prev_volume: f32,
}

impl<M: VadModel> VadAnalyzer<M> {
    /// Create a new VAD analyzer.
    ///
    /// Derives internal frame counts from the model's frame size, the sample
    /// rate, and the timing parameters in `params`.
    ///
    /// Args:
    ///     model: The VAD model backend to use for inference.
    ///     sample_rate: Audio sample rate in Hz (e.g. 16000).
    ///     params: VAD timing and threshold parameters.
    pub fn new(model: M, sample_rate: u32, params: VadParams) -> Self {
        let num_channels: u16 = 1;
        let vad_frames = model.num_frames_required();
        // Each sample is i16 (2 bytes), times the number of channels.
        let vad_frames_num_bytes = vad_frames * (num_channels as usize) * 2;

        let vad_frames_per_sec = vad_frames as f32 / sample_rate as f32;
        let vad_start_frames = (params.start_secs / vad_frames_per_sec).round() as usize;
        let vad_stop_frames = (params.stop_secs / vad_frames_per_sec).round() as usize;

        Self {
            model,
            params,
            sample_rate,
            num_channels,
            state: VadState::Quiet,
            vad_buffer: Vec::new(),
            vad_frames_num_bytes,
            vad_start_frames,
            vad_stop_frames,
            starting_count: 0,
            stopping_count: 0,
            smoothing_factor: 0.2,
            prev_volume: 0.0,
        }
    }

    /// Return the current VAD state.
    pub fn state(&self) -> VadState {
        self.state
    }

    /// Return a reference to the current VAD parameters.
    pub fn params(&self) -> &VadParams {
        &self.params
    }

    /// Return the sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Return the number of audio channels (always 1).
    pub fn num_channels(&self) -> u16 {
        self.num_channels
    }

    /// Return how many consecutive start frames are needed to confirm speech.
    pub fn vad_start_frames(&self) -> usize {
        self.vad_start_frames
    }

    /// Return how many consecutive stop frames are needed to confirm silence.
    pub fn vad_stop_frames(&self) -> usize {
        self.vad_stop_frames
    }

    /// Update the VAD parameters and reset internal counters.
    ///
    /// This recalculates the start/stop frame thresholds from the new params.
    pub fn set_params(&mut self, params: VadParams) {
        let vad_frames = self.model.num_frames_required();
        let vad_frames_per_sec = vad_frames as f32 / self.sample_rate as f32;
        self.vad_start_frames = (params.start_secs / vad_frames_per_sec).round() as usize;
        self.vad_stop_frames = (params.stop_secs / vad_frames_per_sec).round() as usize;
        self.params = params;
        self.starting_count = 0;
        self.stopping_count = 0;
        self.state = VadState::Quiet;
    }

    /// Reset the model internal state (e.g. RNN hidden state).
    pub fn reset_model(&mut self) {
        self.model.reset_states();
    }

    /// Calculate exponentially smoothed volume for a raw audio chunk.
    fn get_smoothed_volume(&mut self, audio_bytes: &[u8]) -> f32 {
        // Interpret bytes as i16 samples (little-endian).
        let samples: Vec<i16> = audio_bytes
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
            .collect();
        let volume = calculate_audio_volume(&samples, self.sample_rate);
        exp_smoothing(volume, self.prev_volume, self.smoothing_factor)
    }

    /// Analyze an audio buffer and return the current VAD state.
    ///
    /// Appends `audio` to an internal buffer. While the buffer contains enough
    /// data for one inference window, extracts a chunk, runs the model, and
    /// advances the state machine. After draining the buffer the post-loop
    /// threshold checks may promote `Starting` to `Speaking` or `Stopping` to
    /// `Quiet`.
    ///
    /// This method is synchronous. For async usage, run it on a blocking
    /// thread pool via `tokio::task::spawn_blocking` or similar.
    pub fn analyze(&mut self, audio: &[u8]) -> VadState {
        self.vad_buffer.extend_from_slice(audio);

        let num_required_bytes = self.vad_frames_num_bytes;
        if self.vad_buffer.len() < num_required_bytes {
            return self.state;
        }

        while self.vad_buffer.len() >= num_required_bytes {
            // Extract the next chunk.
            let audio_frames: Vec<u8> = self.vad_buffer.drain(..num_required_bytes).collect();

            // Convert bytes to i16 samples for the model.
            let samples: Vec<i16> = audio_frames
                .chunks_exact(2)
                .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
                .collect();

            let confidence = self.model.voice_confidence(&samples);

            let volume = self.get_smoothed_volume(&audio_frames);
            self.prev_volume = volume;

            let speaking =
                confidence >= self.params.confidence && volume >= self.params.min_volume;

            if speaking {
                match self.state {
                    VadState::Quiet => {
                        self.state = VadState::Starting;
                        self.starting_count = 1;
                    }
                    VadState::Starting => {
                        self.starting_count += 1;
                    }
                    VadState::Stopping => {
                        self.state = VadState::Speaking;
                        self.stopping_count = 0;
                    }
                    VadState::Speaking => {
                        // Already speaking, nothing to do.
                    }
                }
            } else {
                match self.state {
                    VadState::Starting => {
                        self.state = VadState::Quiet;
                        self.starting_count = 0;
                    }
                    VadState::Speaking => {
                        self.state = VadState::Stopping;
                        self.stopping_count = 1;
                    }
                    VadState::Stopping => {
                        self.stopping_count += 1;
                    }
                    VadState::Quiet => {
                        // Already quiet, nothing to do.
                    }
                }
            }
        }

        // Post-loop threshold checks — promote transitional states when the
        // required number of consecutive frames has been reached.
        if self.state == VadState::Starting && self.starting_count >= self.vad_start_frames {
            self.state = VadState::Speaking;
            self.starting_count = 0;
        }

        if self.state == VadState::Stopping && self.stopping_count >= self.vad_stop_frames {
            self.state = VadState::Quiet;
            self.stopping_count = 0;
        }

        self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::MockVadModel;

    /// Helper: create a speaking audio chunk (loud signal) of `num_bytes` bytes.
    ///
    /// Uses a high-amplitude sine-ish pattern so that the volume calculation
    /// exceeds the default `min_volume` threshold.
    fn loud_audio(num_bytes: usize) -> Vec<u8> {
        assert!(num_bytes % 2 == 0);
        let num_samples = num_bytes / 2;
        let mut bytes = Vec::with_capacity(num_bytes);
        for i in 0..num_samples {
            // Alternate between positive and negative high-amplitude values
            // to produce a clearly non-silent signal.
            let sample: i16 = if i % 2 == 0 { 20000 } else { -20000 };
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        bytes
    }

    /// Helper: create a silent audio chunk of `num_bytes` bytes.
    fn silent_audio(num_bytes: usize) -> Vec<u8> {
        vec![0u8; num_bytes]
    }

    // -- Basic construction tests --

    #[test]
    fn new_analyzer_starts_quiet() {
        let model = MockVadModel::always_quiet(160);
        let analyzer = VadAnalyzer::new(model, 16000, VadParams::default());
        assert_eq!(analyzer.state(), VadState::Quiet);
    }

    #[test]
    fn frame_counts_calculated_correctly() {
        // num_frames=160, sample_rate=16000
        // vad_frames_per_sec = 160/16000 = 0.01
        // start_frames = round(0.2 / 0.01) = 20
        // stop_frames  = round(0.2 / 0.01) = 20
        let model = MockVadModel::always_quiet(160);
        let analyzer = VadAnalyzer::new(model, 16000, VadParams::default());
        assert_eq!(analyzer.vad_start_frames(), 20);
        assert_eq!(analyzer.vad_stop_frames(), 20);
    }

    // -- Quiet → Speaking transition --

    #[test]
    fn quiet_to_speaking_requires_start_frames() {
        // With min_volume=0.0 we bypass the volume gate, so only confidence
        // matters. Each call to analyze feeds exactly one model frame.
        let model = MockVadModel::always_speaking(160);
        let params = VadParams {
            min_volume: 0.0,
            ..VadParams::default()
        };
        let mut analyzer = VadAnalyzer::new(model, 16000, params);
        let chunk_bytes = 160 * 2; // 160 samples * 2 bytes

        // Feed 19 frames — should be STARTING, not yet SPEAKING.
        for _ in 0..19 {
            analyzer.analyze(&loud_audio(chunk_bytes));
        }
        assert_eq!(analyzer.state(), VadState::Starting);

        // The 20th frame should tip over to SPEAKING.
        let state = analyzer.analyze(&loud_audio(chunk_bytes));
        assert_eq!(state, VadState::Speaking);
    }

    // -- Speaking → Quiet transition --

    #[test]
    fn speaking_to_quiet_requires_stop_frames() {
        let confidences: Vec<f32> = std::iter::repeat(0.9)
            .take(20)
            .chain(std::iter::repeat(0.0).take(20))
            .collect();
        let model = MockVadModel::new(160, confidences);
        let params = VadParams {
            min_volume: 0.0,
            ..VadParams::default()
        };
        let mut analyzer = VadAnalyzer::new(model, 16000, params);
        let chunk_bytes = 160 * 2;

        // First 20 speaking frames → SPEAKING
        for _ in 0..20 {
            analyzer.analyze(&loud_audio(chunk_bytes));
        }
        assert_eq!(analyzer.state(), VadState::Speaking);

        // 19 quiet frames → STOPPING
        for _ in 0..19 {
            analyzer.analyze(&silent_audio(chunk_bytes));
        }
        assert_eq!(analyzer.state(), VadState::Stopping);

        // 20th quiet frame → QUIET
        let state = analyzer.analyze(&silent_audio(chunk_bytes));
        assert_eq!(state, VadState::Quiet);
    }

    // -- Flicker rejection --

    #[test]
    fn brief_silence_during_speech_does_not_stop() {
        // Speak for 20 frames (→ SPEAKING), then 5 silent frames, then speak
        // again. Should never reach QUIET.
        let confidences: Vec<f32> = std::iter::repeat(0.9)
            .take(20)
            .chain(std::iter::repeat(0.0).take(5))
            .chain(std::iter::repeat(0.9).take(5))
            .collect();
        let model = MockVadModel::new(160, confidences);
        let params = VadParams {
            min_volume: 0.0,
            ..VadParams::default()
        };
        let mut analyzer = VadAnalyzer::new(model, 16000, params);
        let chunk_bytes = 160 * 2;

        // Reach SPEAKING
        for _ in 0..20 {
            analyzer.analyze(&loud_audio(chunk_bytes));
        }
        assert_eq!(analyzer.state(), VadState::Speaking);

        // 5 silent frames — not enough to reach QUIET (need 20)
        for _ in 0..5 {
            analyzer.analyze(&silent_audio(chunk_bytes));
        }
        assert_eq!(analyzer.state(), VadState::Stopping);

        // Speech resumes — should go back to SPEAKING
        let state = analyzer.analyze(&loud_audio(chunk_bytes));
        assert_eq!(state, VadState::Speaking);
    }

    #[test]
    fn brief_noise_during_quiet_does_not_start() {
        // 5 speaking frames, then silence. Should never reach SPEAKING.
        let confidences: Vec<f32> = std::iter::repeat(0.9)
            .take(5)
            .chain(std::iter::repeat(0.0).take(5))
            .collect();
        let model = MockVadModel::new(160, confidences);
        let params = VadParams {
            min_volume: 0.0,
            ..VadParams::default()
        };
        let mut analyzer = VadAnalyzer::new(model, 16000, params);
        let chunk_bytes = 160 * 2;

        // 5 speaking frames — STARTING but not SPEAKING
        for _ in 0..5 {
            analyzer.analyze(&loud_audio(chunk_bytes));
        }
        assert_eq!(analyzer.state(), VadState::Starting);

        // 1 quiet frame resets to QUIET
        let state = analyzer.analyze(&silent_audio(chunk_bytes));
        assert_eq!(state, VadState::Quiet);
    }

    // -- Buffer accumulation --

    #[test]
    fn small_chunks_accumulate_in_buffer() {
        let model = MockVadModel::always_speaking(160);
        let params = VadParams {
            min_volume: 0.0,
            ..VadParams::default()
        };
        let mut analyzer = VadAnalyzer::new(model, 16000, params);

        // Feed half a frame — should stay QUIET (not enough data).
        let half_chunk = loud_audio(160); // 80 samples = half of 160
        let state = analyzer.analyze(&half_chunk);
        assert_eq!(state, VadState::Quiet);

        // Feed the other half — now we have a full frame.
        let state = analyzer.analyze(&half_chunk);
        assert_eq!(state, VadState::Starting);
    }

    // -- set_params resets state --

    #[test]
    fn set_params_resets_state() {
        let model = MockVadModel::always_speaking(160);
        let params = VadParams {
            min_volume: 0.0,
            ..VadParams::default()
        };
        let mut analyzer = VadAnalyzer::new(model, 16000, params);
        let chunk_bytes = 160 * 2;

        // Get to STARTING
        analyzer.analyze(&loud_audio(chunk_bytes));
        assert_eq!(analyzer.state(), VadState::Starting);

        // Reset params — should go back to QUIET
        analyzer.set_params(VadParams {
            min_volume: 0.0,
            ..VadParams::default()
        });
        assert_eq!(analyzer.state(), VadState::Quiet);
    }

    // -- Volume gating --

    #[test]
    fn low_volume_prevents_speaking_detection() {
        // High confidence but silent audio — volume gate should prevent
        // transition to STARTING.
        let model = MockVadModel::always_speaking(160);
        let params = VadParams::default(); // min_volume=0.6
        let mut analyzer = VadAnalyzer::new(model, 16000, params);
        let chunk_bytes = 160 * 2;

        for _ in 0..25 {
            analyzer.analyze(&silent_audio(chunk_bytes));
        }
        // Should still be QUIET because volume is too low.
        assert_eq!(analyzer.state(), VadState::Quiet);
    }

    // -- MockVadModel specific --

    #[test]
    fn mock_model_cycles_through_confidences() {
        let mut model = MockVadModel::new(160, vec![0.1, 0.9]);
        let dummy = vec![0i16; 160];
        assert!((model.voice_confidence(&dummy) - 0.1).abs() < f32::EPSILON);
        assert!((model.voice_confidence(&dummy) - 0.9).abs() < f32::EPSILON);
        // Wraps around
        assert!((model.voice_confidence(&dummy) - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn mock_model_reset() {
        let mut model = MockVadModel::new(160, vec![0.1, 0.9]);
        let dummy = vec![0i16; 160];
        model.voice_confidence(&dummy); // index 0
        model.voice_confidence(&dummy); // index 1
        model.reset_states();
        assert!((model.voice_confidence(&dummy) - 0.1).abs() < f32::EPSILON);
    }
}
