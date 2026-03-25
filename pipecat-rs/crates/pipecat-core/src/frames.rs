use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::frame_id::FrameId;
use crate::frame_types::frame_type;
use crate::metrics::MetricsData;

// ─── Shared data structs ─────────────────────────────────────────────────────

/// Common header present on every frame.
///
/// Hot fields (`id`, `pts`, `broadcast_sibling_id`) are stored inline.
/// Cold fields (metadata, transport routing) are behind a lazy `Box` so they
/// don't bloat the header for the common case where they're unused.
#[derive(Debug, Clone)]
pub struct FrameHeader {
    /// Unique identifier for this frame instance.
    pub id: FrameId,
    /// Presentation timestamp in microseconds (pipeline clock).
    pub pts: Option<u64>,
    /// When set, siblings in a parallel pipeline share this ID.
    pub broadcast_sibling_id: Option<FrameId>,
    /// Lazy-allocated cold fields (metadata, transport routing).
    cold: Option<Box<FrameHeaderCold>>,
}

/// Rarely-accessed header fields, heap-allocated only when needed.
#[derive(Debug, Clone, Default)]
struct FrameHeaderCold {
    metadata: Option<HashMap<String, serde_json::Value>>,
    transport_source: Option<String>,
    transport_destination: Option<String>,
}

impl FrameHeader {
    /// Create a new header with a fresh ID and all optional fields `None`.
    pub fn new() -> Self {
        Self {
            id: FrameId::next(),
            pts: None,
            broadcast_sibling_id: None,
            cold: None,
        }
    }

    /// Immutable access to metadata. Returns an empty map if never initialised.
    pub fn metadata(&self) -> &HashMap<String, serde_json::Value> {
        static EMPTY: std::sync::LazyLock<HashMap<String, serde_json::Value>> =
            std::sync::LazyLock::new(HashMap::new);
        self.cold
            .as_ref()
            .and_then(|c| c.metadata.as_ref())
            .unwrap_or(&EMPTY)
    }

    /// Mutable access to metadata, lazily allocating the cold block and map.
    pub fn metadata_mut(&mut self) -> &mut HashMap<String, serde_json::Value> {
        self.cold
            .get_or_insert_with(|| Box::new(FrameHeaderCold::default()))
            .metadata
            .get_or_insert_with(HashMap::new)
    }

    /// Returns true if metadata has been allocated and is non-empty.
    pub fn has_metadata(&self) -> bool {
        self.cold
            .as_ref()
            .and_then(|c| c.metadata.as_ref())
            .is_some_and(|m| !m.is_empty())
    }

    /// Transport that produced this frame (if set).
    pub fn transport_source(&self) -> Option<&str> {
        self.cold
            .as_ref()
            .and_then(|c| c.transport_source.as_deref())
    }

    /// Set the transport source.
    pub fn set_transport_source(&mut self, source: impl Into<String>) {
        self.cold
            .get_or_insert_with(|| Box::new(FrameHeaderCold::default()))
            .transport_source = Some(source.into());
    }

    /// Transport this frame should be routed to (if set).
    pub fn transport_destination(&self) -> Option<&str> {
        self.cold
            .as_ref()
            .and_then(|c| c.transport_destination.as_deref())
    }

    /// Set the transport destination.
    pub fn set_transport_destination(&mut self, destination: impl Into<String>) {
        self.cold
            .get_or_insert_with(|| Box::new(FrameHeaderCold::default()))
            .transport_destination = Some(destination.into());
    }
}

impl Default for FrameHeader {
    fn default() -> Self {
        Self::new()
    }
}

/// Audio data shared across audio frame variants.
#[derive(Debug, Clone)]
pub struct AudioData {
    /// Raw PCM audio bytes (16-bit signed LE).
    pub audio: Bytes,
    /// Sample rate in Hz (e.g. 16000, 48000).
    pub sample_rate: u32,
    /// Number of audio channels (1 = mono, 2 = stereo).
    pub num_channels: u16,
}

impl AudioData {
    /// Number of PCM frames (samples per channel) in the buffer.
    ///
    /// Assumes 16-bit (2 bytes per sample) signed PCM.
    pub fn num_frames(&self) -> usize {
        if self.num_channels == 0 {
            return 0;
        }
        self.audio.len() / (self.num_channels as usize * 2)
    }

    /// Duration of the audio in seconds.
    pub fn duration_secs(&self) -> f64 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        self.num_frames() as f64 / self.sample_rate as f64
    }
}

/// Plain text payload.
#[derive(Debug, Clone)]
pub struct TextData {
    pub text: String,
}

/// Transcription from an STT service.
#[derive(Debug, Clone)]
pub struct TranscriptionData {
    pub text: String,
    pub user_id: Option<String>,
    pub timestamp: Option<String>,
    pub language: Option<String>,
}

/// Image payload.
#[derive(Debug, Clone)]
pub struct ImageData {
    /// Raw image bytes (PNG, JPEG, etc.).
    pub image: Bytes,
    /// Width and height in pixels.
    pub size: (u32, u32),
    /// MIME type or format string (e.g. "image/png").
    pub format: Option<String>,
}

/// Error information carried in an error frame.
#[derive(Debug, Clone)]
pub struct ErrorData {
    pub message: String,
    pub fatal: bool,
    pub exception: Option<String>,
}

/// LLM conversation context (boxed in frames since it can be large).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmContextData {
    pub messages: Vec<serde_json::Value>,
}

/// LLM tool call payload (boxed to reduce Frame enum size).
#[derive(Debug, Clone)]
pub struct LlmToolCallData {
    pub tool_name: String,
    pub arguments: String,
    pub tool_call_id: Option<String>,
    pub run_llm: bool,
}

/// LLM tool result payload (boxed to reduce Frame enum size).
#[derive(Debug, Clone)]
pub struct LlmToolResultData {
    pub tool_name: String,
    pub tool_call_id: String,
    pub result: serde_json::Value,
    pub run_llm: bool,
}

/// Function call progress payload (boxed).
#[derive(Debug, Clone)]
pub struct FuncCallProgressData {
    pub function_name: String,
    pub tool_call_id: String,
    pub arguments: String,
}

/// Function call result payload (boxed).
#[derive(Debug, Clone)]
pub struct FuncCallResultData {
    pub function_name: String,
    pub tool_call_id: String,
    pub result: serde_json::Value,
    pub run_llm: bool,
}

/// Custom frame payload (boxed).
#[derive(Debug, Clone)]
pub struct CustomFrameData {
    pub type_name: String,
    pub data: serde_json::Value,
}

// ─── Frame classification ────────────────────────────────────────────────────

/// Classification of a frame for queue routing decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FrameClass {
    /// System frames use unbounded channels and are never dropped.
    System,
    /// Data frames carry payload (audio, text, images, etc.).
    Data,
    /// Control frames manage pipeline lifecycle.
    Control,
}

/// Direction a frame travels through the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FrameDirection {
    /// From input toward output (left to right).
    Downstream,
    /// From output toward input (right to left), e.g. errors, acknowledgments.
    Upstream,
}

// ─── The Frame enum ──────────────────────────────────────────────────────────

/// The main Frame enum representing all frame variants in the pipeline.
///
/// Every variant carries a `FrameHeader` for identification and metadata.
/// Variant-specific payloads are inlined or use shared data structs.
#[derive(Debug, Clone)]
pub enum Frame {
    // ── System frames (unbounded channel, never dropped) ─────────
    Start(FrameHeader),
    Cancel(FrameHeader),
    Heartbeat(FrameHeader),
    Interruption(FrameHeader),
    StartInterruption(FrameHeader),
    Metrics {
        header: FrameHeader,
        data: Vec<MetricsData>,
    },
    Error {
        header: FrameHeader,
        error: ErrorData,
    },

    // ── Audio frames ─────────────────────────────────────────────
    AudioRawInput {
        header: FrameHeader,
        audio: AudioData,
    },
    AudioRawOutput {
        header: FrameHeader,
        audio: AudioData,
    },
    AudioTts {
        header: FrameHeader,
        audio: AudioData,
        context_id: Option<String>,
    },
    AudioSpeech {
        header: FrameHeader,
        audio: AudioData,
    },
    AudioMix {
        header: FrameHeader,
        audio: AudioData,
        mix_handle_id: String,
    },
    AudioSilence {
        header: FrameHeader,
        audio: AudioData,
    },
    AudioUser {
        header: FrameHeader,
        audio: AudioData,
    },

    // ── Text frames ──────────────────────────────────────────────
    Text {
        header: FrameHeader,
        data: TextData,
    },
    TextLlm {
        header: FrameHeader,
        data: TextData,
    },
    TextAggregated {
        header: FrameHeader,
        data: TextData,
    },
    TextTts {
        header: FrameHeader,
        data: TextData,
    },
    TextInputRaw {
        header: FrameHeader,
        data: TextData,
    },

    // ── Transcription ────────────────────────────────────────────
    Transcription {
        header: FrameHeader,
        data: Box<TranscriptionData>,
    },
    InterimTranscription {
        header: FrameHeader,
        data: Box<TranscriptionData>,
    },

    // ── Image frames ─────────────────────────────────────────────
    ImageOutput {
        header: FrameHeader,
        image: Box<ImageData>,
    },
    ImageUrl {
        header: FrameHeader,
        image: Box<ImageData>,
        url: Option<String>,
    },
    ImageInput {
        header: FrameHeader,
        image: Box<ImageData>,
    },

    // ── LLM frames ───────────────────────────────────────────────
    LlmContext {
        header: FrameHeader,
        context: Box<LlmContextData>,
    },
    LlmMessages {
        header: FrameHeader,
        messages: Box<Vec<serde_json::Value>>,
    },
    LlmResponseStart(FrameHeader),
    LlmResponseEnd(FrameHeader),
    LlmRun(FrameHeader),
    LlmToolCall {
        header: FrameHeader,
        data: Box<LlmToolCallData>,
    },
    LlmToolResult {
        header: FrameHeader,
        data: Box<LlmToolResultData>,
    },
    LlmThoughtText {
        header: FrameHeader,
        data: TextData,
    },
    LlmThoughtStart(FrameHeader),
    LlmThoughtEnd(FrameHeader),
    LlmMessagesUpdate {
        header: FrameHeader,
        messages: Box<Vec<serde_json::Value>>,
    },
    LlmMessagesAppend {
        header: FrameHeader,
        messages: Box<Vec<serde_json::Value>>,
    },
    LlmSetTools {
        header: FrameHeader,
        tools: serde_json::Value,
    },
    LlmSetToolChoice {
        header: FrameHeader,
        tool_choice: serde_json::Value,
    },
    LlmEnableCaching(FrameHeader),
    LlmConfigureOutput {
        header: FrameHeader,
        config: serde_json::Value,
    },
    LlmCtxSummaryRequest(FrameHeader),
    LlmCtxSummaryResult {
        header: FrameHeader,
        summary: String,
    },
    LlmUpdateSettings {
        header: FrameHeader,
        settings: serde_json::Value,
    },

    // ── STT control ──────────────────────────────────────────────
    SttMute {
        header: FrameHeader,
        mute: bool,
    },
    SttUpdateSettings {
        header: FrameHeader,
        settings: serde_json::Value,
    },
    SttLanguageUpdate {
        header: FrameHeader,
        language: String,
    },

    // ── TTS events ───────────────────────────────────────────────
    TtsStarted(FrameHeader),
    TtsStopped(FrameHeader),
    TtsUpdateSettings {
        header: FrameHeader,
        settings: serde_json::Value,
    },
    TtsSpeak {
        header: FrameHeader,
        text: String,
    },

    // ── User events ──────────────────────────────────────────────
    UserStartedSpeaking(FrameHeader),
    UserStoppedSpeaking(FrameHeader),
    /// Periodic user speaking state (boolean payload).
    UserSpeaking {
        header: FrameHeader,
        speaking: bool,
    },
    UserMuteStarted(FrameHeader),
    UserMuteStopped(FrameHeader),
    /// Emulated speaking start (e.g., from text input).
    EmulateUserStartedSpeaking(FrameHeader),
    EmulateUserStoppedSpeaking(FrameHeader),
    /// VAD-triggered speaking events.
    VADUserStartedSpeaking(FrameHeader),
    VADUserStoppedSpeaking(FrameHeader),

    // ── Bot events ───────────────────────────────────────────────
    BotStartedSpeaking(FrameHeader),
    BotStoppedSpeaking(FrameHeader),
    /// Periodic bot speaking state (boolean payload).
    BotSpeaking {
        header: FrameHeader,
        speaking: bool,
    },

    // ── Control frames ───────────────────────────────────────────
    End(FrameHeader),
    Stop(FrameHeader),
    Pause {
        header: FrameHeader,
        processor_name: String,
    },
    Resume {
        header: FrameHeader,
        processor_name: String,
    },
    PauseUrgent {
        header: FrameHeader,
        processor_name: String,
    },
    ResumeUrgent {
        header: FrameHeader,
        processor_name: String,
    },
    /// Output transport is ready for frames.
    OutputTransportReady(FrameHeader),

    // ── Transport messages ────────────────────────────────────────
    TransportMessageOut {
        header: FrameHeader,
        message: serde_json::Value,
    },
    TransportMessageIn {
        header: FrameHeader,
        message: serde_json::Value,
    },
    TransportMessageBidir {
        header: FrameHeader,
        message: serde_json::Value,
    },

    // ── Task frames ──────────────────────────────────────────────
    InterruptionTask(FrameHeader),
    BotInterruption(FrameHeader),
    CancelTask(FrameHeader),
    StopTask(FrameHeader),
    /// Request graceful pipeline closure by converting to EndFrame downstream.
    EndTask(FrameHeader),

    // ── DTMF ─────────────────────────────────────────────────────
    DtmfOutput {
        header: FrameHeader,
        keys: String,
    },
    DtmfInput {
        header: FrameHeader,
        keys: String,
    },

    // ── Function calls ───────────────────────────────────────────
    FunctionCallProgress {
        header: FrameHeader,
        data: Box<FuncCallProgressData>,
    },
    FunctionCallResult {
        header: FrameHeader,
        data: Box<FuncCallResultData>,
    },
    /// Notification that function calls have started (LLM tool-use begin).
    FunctionCallsStarted(FrameHeader),
    /// Cancel an in-progress function call.
    FunctionCallCancel {
        header: FrameHeader,
        function_name: String,
    },

    // ── Vision ────────────────────────────────────────────────────
    VisionText {
        header: FrameHeader,
        data: TextData,
    },
    VisionResponseStart(FrameHeader),
    VisionResponseEnd(FrameHeader),

    // ── Text (extended) ───────────────────────────────────────────
    /// Translated text output.
    Translation {
        header: FrameHeader,
        data: TextData,
        /// Source language (BCP-47).
        language: Option<String>,
    },

    // ── Service ────────────────────────────────────────────────────
    ServiceMetadata {
        header: FrameHeader,
        service_name: String,
    },
    ServiceUpdateSettings {
        header: FrameHeader,
        settings: serde_json::Value,
    },
    ServiceSwitcher(FrameHeader),
    ServiceSwitchManual {
        header: FrameHeader,
        service_name: String,
    },

    // ── Filter ────────────────────────────────────────────────────
    FilterUpdateSettings {
        header: FrameHeader,
        settings: serde_json::Value,
    },
    FilterEnable {
        header: FrameHeader,
        enable: bool,
    },

    // ── Mixer ─────────────────────────────────────────────────────
    MixerUpdateSettings {
        header: FrameHeader,
        settings: serde_json::Value,
    },
    MixerEnable {
        header: FrameHeader,
        enable: bool,
    },

    // ── Extensible ───────────────────────────────────────────────
    Custom {
        header: FrameHeader,
        data: Box<CustomFrameData>,
    },
}

/// Helper macro that generates a match extracting `header` from every Frame variant.
///
/// Tuple variants: `tuple(VariantName)` — matches `Frame::VariantName(ref h)`.
/// Struct variants: `struc(VariantName)` — matches `Frame::VariantName { ref header, .. }`.
macro_rules! with_header {
    ($self:expr, $( tuple($t:ident) ),* ; $( struc($s:ident) ),* $(,)?) => {
        match $self {
            $( Frame::$t(ref h) => h, )*
            $( Frame::$s { ref header, .. } => header, )*
        }
    };
}

/// Same as `with_header` but yields `&mut FrameHeader`.
macro_rules! with_header_mut {
    ($self:expr, $( tuple($t:ident) ),* ; $( struc($s:ident) ),* $(,)?) => {
        match $self {
            $( Frame::$t(ref mut h) => h, )*
            $( Frame::$s { ref mut header, .. } => header, )*
        }
    };
}

/// Lists all tuple-style and struct-style variants for the header accessors.
macro_rules! all_variants {
    ($mac:ident, $self:expr) => {
        $mac!(
            $self,
            // Tuple variants (single FrameHeader field)
            tuple(Start),
            tuple(Cancel),
            tuple(Heartbeat),
            tuple(Interruption),
            tuple(StartInterruption),
            tuple(LlmResponseStart),
            tuple(LlmResponseEnd),
            tuple(LlmRun),
            tuple(LlmThoughtStart),
            tuple(LlmThoughtEnd),
            tuple(LlmEnableCaching),
            tuple(LlmCtxSummaryRequest),
            tuple(TtsStarted),
            tuple(TtsStopped),
            tuple(UserStartedSpeaking),
            tuple(UserStoppedSpeaking),
            tuple(BotStartedSpeaking),
            tuple(BotStoppedSpeaking),
            tuple(UserMuteStarted),
            tuple(UserMuteStopped),
            tuple(EmulateUserStartedSpeaking),
            tuple(EmulateUserStoppedSpeaking),
            tuple(VADUserStartedSpeaking),
            tuple(VADUserStoppedSpeaking),
            tuple(OutputTransportReady),
            tuple(End),
            tuple(Stop),
            tuple(InterruptionTask),
            tuple(BotInterruption),
            tuple(CancelTask),
            tuple(StopTask),
            tuple(EndTask),
            tuple(FunctionCallsStarted),
            tuple(VisionResponseStart),
            tuple(VisionResponseEnd),
            tuple(ServiceSwitcher)
            ;
            // Struct variants (have `header: FrameHeader` field)
            struc(Metrics),
            struc(Error),
            struc(AudioRawInput),
            struc(AudioRawOutput),
            struc(AudioTts),
            struc(AudioSpeech),
            struc(AudioMix),
            struc(AudioSilence),
            struc(AudioUser),
            struc(Text),
            struc(TextLlm),
            struc(TextAggregated),
            struc(TextTts),
            struc(TextInputRaw),
            struc(Transcription),
            struc(InterimTranscription),
            struc(ImageOutput),
            struc(ImageUrl),
            struc(ImageInput),
            struc(LlmContext),
            struc(LlmMessages),
            struc(LlmToolCall),
            struc(LlmToolResult),
            struc(LlmThoughtText),
            struc(LlmMessagesUpdate),
            struc(LlmMessagesAppend),
            struc(LlmSetTools),
            struc(LlmSetToolChoice),
            struc(LlmConfigureOutput),
            struc(LlmCtxSummaryResult),
            struc(LlmUpdateSettings),
            struc(SttMute),
            struc(SttUpdateSettings),
            struc(SttLanguageUpdate),
            struc(TtsUpdateSettings),
            struc(TtsSpeak),
            struc(Pause),
            struc(Resume),
            struc(PauseUrgent),
            struc(ResumeUrgent),
            struc(UserSpeaking),
            struc(BotSpeaking),
            struc(TransportMessageOut),
            struc(TransportMessageIn),
            struc(TransportMessageBidir),
            struc(DtmfOutput),
            struc(DtmfInput),
            struc(FunctionCallProgress),
            struc(FunctionCallResult),
            struc(FunctionCallCancel),
            struc(VisionText),
            struc(Translation),
            struc(ServiceMetadata),
            struc(ServiceUpdateSettings),
            struc(ServiceSwitchManual),
            struc(FilterUpdateSettings),
            struc(FilterEnable),
            struc(MixerUpdateSettings),
            struc(MixerEnable),
            struc(Custom),
        )
    };
}

impl Frame {
    /// Get a shared reference to the frame header.
    pub fn header(&self) -> &FrameHeader {
        all_variants!(with_header, self)
    }

    /// Get a mutable reference to the frame header.
    pub fn header_mut(&mut self) -> &mut FrameHeader {
        all_variants!(with_header_mut, self)
    }

    /// Frame classification for queue routing.
    ///
    /// - `System` frames use unbounded channels and are never dropped.
    /// - `Control` frames manage pipeline lifecycle.
    /// - `Data` frames carry payload and may be dropped on interruption.
    pub fn classification(&self) -> FrameClass {
        match self {
            // System: high-priority, unbounded queue
            Frame::Start(_)
            | Frame::Cancel(_)
            | Frame::Heartbeat(_)
            | Frame::Interruption(_)
            | Frame::StartInterruption(_)
            | Frame::Metrics { .. }
            | Frame::Error { .. }
            | Frame::AudioRawInput { .. }
            | Frame::UserStartedSpeaking(_)
            | Frame::UserStoppedSpeaking(_) => FrameClass::System,

            // Control: pipeline lifecycle
            Frame::End(_)
            | Frame::Stop(_)
            | Frame::Pause { .. }
            | Frame::Resume { .. } => FrameClass::Control,

            // Everything else is data
            _ => FrameClass::Data,
        }
    }

    /// Whether this frame survives interruption (not dropped from queues).
    pub fn is_uninterruptible(&self) -> bool {
        matches!(self, Frame::End(_) | Frame::Stop(_) | Frame::Cancel(_))
    }

    /// Get the 16-bit type ID for O(1) dispatch table lookups.
    pub fn type_id(&self) -> u16 {
        match self {
            Frame::Start(_) => frame_type::CTRL_START,
            Frame::Cancel(_) => frame_type::CTRL_CANCEL,
            Frame::Heartbeat(_) => frame_type::SYS_HEARTBEAT,
            Frame::Interruption(_) => frame_type::CTRL_INTERRUPT,
            Frame::StartInterruption(_) => frame_type::CTRL_START_INTERRUPT,
            Frame::Metrics { .. } => frame_type::SYS_METRICS,
            Frame::Error { .. } => frame_type::ERROR_GENERAL,

            Frame::AudioRawInput { .. } => frame_type::AUDIO_RAW_INPUT,
            Frame::AudioRawOutput { .. } => frame_type::AUDIO_RAW_OUTPUT,
            Frame::AudioTts { .. } => frame_type::AUDIO_TTS,
            Frame::AudioSpeech { .. } => frame_type::AUDIO_SPEECH,
            Frame::AudioMix { .. } => frame_type::AUDIO_MIX,
            Frame::AudioSilence { .. } => frame_type::AUDIO_SILENCE,
            Frame::AudioUser { .. } => frame_type::AUDIO_USER,

            Frame::Text { .. } => frame_type::TEXT_PLAIN,
            Frame::TextLlm { .. } => frame_type::TEXT_LLM,
            Frame::TextAggregated { .. } => frame_type::TEXT_AGGREGATED,
            Frame::TextTts { .. } => frame_type::TEXT_TTS,
            Frame::TextInputRaw { .. } => frame_type::TEXT_INPUT_RAW,

            Frame::Transcription { .. } => frame_type::TEXT_TRANSCRIPTION,
            Frame::InterimTranscription { .. } => frame_type::TEXT_INTERIM_TRANS,

            Frame::ImageOutput { .. } => frame_type::IMAGE_OUTPUT,
            Frame::ImageUrl { .. } => frame_type::IMAGE_URL,
            Frame::ImageInput { .. } => frame_type::IMAGE_INPUT,

            Frame::LlmContext { .. } => frame_type::LLM_CONTEXT,
            Frame::LlmMessages { .. } => frame_type::LLM_MESSAGES,
            Frame::LlmResponseStart(_) => frame_type::LLM_RESPONSE_START,
            Frame::LlmResponseEnd(_) => frame_type::LLM_RESPONSE_END,
            Frame::LlmRun(_) => frame_type::LLM_RUN,
            Frame::LlmToolCall { .. } => frame_type::LLM_TOOL_CALL,
            Frame::LlmToolResult { .. } => frame_type::LLM_TOOL_RESULT,
            Frame::LlmThoughtText { .. } => frame_type::LLM_THOUGHT_TEXT,
            Frame::LlmThoughtStart(_) => frame_type::LLM_THOUGHT_START,
            Frame::LlmThoughtEnd(_) => frame_type::LLM_THOUGHT_END,
            Frame::LlmMessagesUpdate { .. } => frame_type::LLM_MESSAGES_UPDATE,
            Frame::LlmMessagesAppend { .. } => frame_type::LLM_MESSAGES_APPEND,
            Frame::LlmSetTools { .. } => frame_type::LLM_SET_TOOLS,
            Frame::LlmSetToolChoice { .. } => frame_type::LLM_SET_TOOL_CHOICE,
            Frame::LlmEnableCaching(_) => frame_type::LLM_ENABLE_CACHING,
            Frame::LlmConfigureOutput { .. } => frame_type::LLM_CONFIGURE_OUTPUT,
            Frame::LlmCtxSummaryRequest(_) => frame_type::LLM_CTX_SUMMARY_REQ,
            Frame::LlmCtxSummaryResult { .. } => frame_type::LLM_CTX_SUMMARY_RESULT,
            Frame::LlmUpdateSettings { .. } => frame_type::LLM_UPDATE_SETTINGS,

            Frame::SttMute { .. } => frame_type::STT_MUTE,
            Frame::SttUpdateSettings { .. } => frame_type::STT_UPDATE_SETTINGS,
            Frame::SttLanguageUpdate { .. } => frame_type::STT_LANGUAGE_UPDATE,

            Frame::TtsStarted(_) => frame_type::TTS_STARTED,
            Frame::TtsStopped(_) => frame_type::TTS_STOPPED,
            Frame::TtsUpdateSettings { .. } => frame_type::TTS_UPDATE_SETTINGS,
            Frame::TtsSpeak { .. } => frame_type::TTS_SPEAK,

            Frame::UserStartedSpeaking(_) => frame_type::USER_STARTED_SPEAKING,
            Frame::UserStoppedSpeaking(_) => frame_type::USER_STOPPED_SPEAKING,
            Frame::UserSpeaking { .. } => frame_type::USER_SPEAKING,
            Frame::UserMuteStarted(_) => frame_type::USER_MUTE_STARTED,
            Frame::UserMuteStopped(_) => frame_type::USER_MUTE_STOPPED,
            Frame::EmulateUserStartedSpeaking(_) => frame_type::USER_EMULATE_STARTED,
            Frame::EmulateUserStoppedSpeaking(_) => frame_type::USER_EMULATE_STOPPED,
            Frame::VADUserStartedSpeaking(_) => frame_type::USER_VAD_STARTED,
            Frame::VADUserStoppedSpeaking(_) => frame_type::USER_VAD_STOPPED,

            Frame::BotStartedSpeaking(_) => frame_type::BOT_STARTED_SPEAKING,
            Frame::BotStoppedSpeaking(_) => frame_type::BOT_STOPPED_SPEAKING,
            Frame::BotSpeaking { .. } => frame_type::BOT_SPEAKING,

            Frame::End(_) => frame_type::CTRL_END,
            Frame::Stop(_) => frame_type::CTRL_STOP,
            Frame::Pause { .. } => frame_type::CTRL_PAUSE,
            Frame::Resume { .. } => frame_type::CTRL_RESUME,
            Frame::PauseUrgent { .. } => frame_type::CTRL_PAUSE_URGENT,
            Frame::ResumeUrgent { .. } => frame_type::CTRL_RESUME_URGENT,
            Frame::OutputTransportReady(_) => frame_type::CTRL_OUTPUT_READY,

            Frame::TransportMessageOut { .. } => frame_type::TRANSPORT_MSG_OUT,
            Frame::TransportMessageIn { .. } => frame_type::TRANSPORT_MSG_IN,
            Frame::TransportMessageBidir { .. } => frame_type::TRANSPORT_MSG_BIDIR,

            Frame::InterruptionTask(_) => frame_type::TASK_INTERRUPTION,
            Frame::BotInterruption(_) => frame_type::TASK_BOT_INTERRUPT,
            Frame::CancelTask(_) => frame_type::TASK_CANCEL,
            Frame::StopTask(_) => frame_type::TASK_STOP,
            Frame::EndTask(_) => frame_type::CTRL_END_TASK,

            Frame::DtmfOutput { .. } => frame_type::DTMF_OUTPUT,
            Frame::DtmfInput { .. } => frame_type::DTMF_INPUT,

            Frame::FunctionCallProgress { .. } => frame_type::FUNC_CALL_PROGRESS,
            Frame::FunctionCallResult { .. } => frame_type::FUNC_CALL_RESULT,
            Frame::FunctionCallsStarted(_) => frame_type::FUNC_CALLS_STARTED,
            Frame::FunctionCallCancel { .. } => frame_type::FUNC_CALL_CANCEL,

            Frame::VisionText { .. } => frame_type::VISION_TEXT,
            Frame::VisionResponseStart(_) => frame_type::VISION_RESP_START,
            Frame::VisionResponseEnd(_) => frame_type::VISION_RESP_END,

            Frame::Translation { .. } => frame_type::TEXT_TRANSLATION,

            Frame::ServiceMetadata { .. } => frame_type::SERVICE_METADATA,
            Frame::ServiceUpdateSettings { .. } => frame_type::SERVICE_UPDATE,
            Frame::ServiceSwitcher(_) => frame_type::SERVICE_SWITCHER,
            Frame::ServiceSwitchManual { .. } => frame_type::SERVICE_SWITCH_MANUAL,
            Frame::FilterUpdateSettings { .. } => frame_type::FILTER_UPDATE,
            Frame::FilterEnable { .. } => frame_type::FILTER_ENABLE,
            Frame::MixerUpdateSettings { .. } => frame_type::MIXER_UPDATE,
            Frame::MixerEnable { .. } => frame_type::MIXER_ENABLE,

            Frame::Custom { .. } => frame_type::FRAME,
        }
    }

    /// Human-readable name of the frame variant (useful for logging).
    pub fn name(&self) -> &'static str {
        match self {
            Frame::Start(_) => "Start",
            Frame::Cancel(_) => "Cancel",
            Frame::Heartbeat(_) => "Heartbeat",
            Frame::Interruption(_) => "Interruption",
            Frame::StartInterruption(_) => "StartInterruption",
            Frame::Metrics { .. } => "Metrics",
            Frame::Error { .. } => "Error",
            Frame::AudioRawInput { .. } => "AudioRawInput",
            Frame::AudioRawOutput { .. } => "AudioRawOutput",
            Frame::AudioTts { .. } => "AudioTts",
            Frame::AudioSpeech { .. } => "AudioSpeech",
            Frame::AudioMix { .. } => "AudioMix",
            Frame::AudioSilence { .. } => "AudioSilence",
            Frame::AudioUser { .. } => "AudioUser",
            Frame::Text { .. } => "Text",
            Frame::TextLlm { .. } => "TextLlm",
            Frame::TextAggregated { .. } => "TextAggregated",
            Frame::TextTts { .. } => "TextTts",
            Frame::TextInputRaw { .. } => "TextInputRaw",
            Frame::Transcription { .. } => "Transcription",
            Frame::InterimTranscription { .. } => "InterimTranscription",
            Frame::ImageOutput { .. } => "ImageOutput",
            Frame::ImageUrl { .. } => "ImageUrl",
            Frame::ImageInput { .. } => "ImageInput",
            Frame::LlmContext { .. } => "LlmContext",
            Frame::LlmMessages { .. } => "LlmMessages",
            Frame::LlmResponseStart(_) => "LlmResponseStart",
            Frame::LlmResponseEnd(_) => "LlmResponseEnd",
            Frame::LlmRun(_) => "LlmRun",
            Frame::LlmToolCall { .. } => "LlmToolCall",
            Frame::LlmToolResult { .. } => "LlmToolResult",
            Frame::LlmThoughtText { .. } => "LlmThoughtText",
            Frame::LlmThoughtStart(_) => "LlmThoughtStart",
            Frame::LlmThoughtEnd(_) => "LlmThoughtEnd",
            Frame::LlmMessagesUpdate { .. } => "LlmMessagesUpdate",
            Frame::LlmMessagesAppend { .. } => "LlmMessagesAppend",
            Frame::LlmSetTools { .. } => "LlmSetTools",
            Frame::LlmSetToolChoice { .. } => "LlmSetToolChoice",
            Frame::LlmEnableCaching(_) => "LlmEnableCaching",
            Frame::LlmConfigureOutput { .. } => "LlmConfigureOutput",
            Frame::LlmCtxSummaryRequest(_) => "LlmCtxSummaryRequest",
            Frame::LlmCtxSummaryResult { .. } => "LlmCtxSummaryResult",
            Frame::LlmUpdateSettings { .. } => "LlmUpdateSettings",
            Frame::SttMute { .. } => "SttMute",
            Frame::SttUpdateSettings { .. } => "SttUpdateSettings",
            Frame::SttLanguageUpdate { .. } => "SttLanguageUpdate",
            Frame::TtsStarted(_) => "TtsStarted",
            Frame::TtsStopped(_) => "TtsStopped",
            Frame::TtsUpdateSettings { .. } => "TtsUpdateSettings",
            Frame::TtsSpeak { .. } => "TtsSpeak",
            Frame::UserStartedSpeaking(_) => "UserStartedSpeaking",
            Frame::UserStoppedSpeaking(_) => "UserStoppedSpeaking",
            Frame::UserSpeaking { .. } => "UserSpeaking",
            Frame::UserMuteStarted(_) => "UserMuteStarted",
            Frame::UserMuteStopped(_) => "UserMuteStopped",
            Frame::EmulateUserStartedSpeaking(_) => "EmulateUserStartedSpeaking",
            Frame::EmulateUserStoppedSpeaking(_) => "EmulateUserStoppedSpeaking",
            Frame::VADUserStartedSpeaking(_) => "VADUserStartedSpeaking",
            Frame::VADUserStoppedSpeaking(_) => "VADUserStoppedSpeaking",
            Frame::BotStartedSpeaking(_) => "BotStartedSpeaking",
            Frame::BotStoppedSpeaking(_) => "BotStoppedSpeaking",
            Frame::BotSpeaking { .. } => "BotSpeaking",
            Frame::End(_) => "End",
            Frame::Stop(_) => "Stop",
            Frame::Pause { .. } => "Pause",
            Frame::Resume { .. } => "Resume",
            Frame::PauseUrgent { .. } => "PauseUrgent",
            Frame::ResumeUrgent { .. } => "ResumeUrgent",
            Frame::OutputTransportReady(_) => "OutputTransportReady",
            Frame::TransportMessageOut { .. } => "TransportMessageOut",
            Frame::TransportMessageIn { .. } => "TransportMessageIn",
            Frame::TransportMessageBidir { .. } => "TransportMessageBidir",
            Frame::InterruptionTask(_) => "InterruptionTask",
            Frame::BotInterruption(_) => "BotInterruption",
            Frame::CancelTask(_) => "CancelTask",
            Frame::StopTask(_) => "StopTask",
            Frame::EndTask(_) => "EndTask",
            Frame::DtmfOutput { .. } => "DtmfOutput",
            Frame::DtmfInput { .. } => "DtmfInput",
            Frame::FunctionCallProgress { .. } => "FunctionCallProgress",
            Frame::FunctionCallResult { .. } => "FunctionCallResult",
            Frame::FunctionCallsStarted(_) => "FunctionCallsStarted",
            Frame::FunctionCallCancel { .. } => "FunctionCallCancel",
            Frame::VisionText { .. } => "VisionText",
            Frame::VisionResponseStart(_) => "VisionResponseStart",
            Frame::VisionResponseEnd(_) => "VisionResponseEnd",
            Frame::Translation { .. } => "Translation",
            Frame::ServiceMetadata { .. } => "ServiceMetadata",
            Frame::ServiceUpdateSettings { .. } => "ServiceUpdateSettings",
            Frame::ServiceSwitcher(_) => "ServiceSwitcher",
            Frame::ServiceSwitchManual { .. } => "ServiceSwitchManual",
            Frame::FilterUpdateSettings { .. } => "FilterUpdateSettings",
            Frame::FilterEnable { .. } => "FilterEnable",
            Frame::MixerUpdateSettings { .. } => "MixerUpdateSettings",
            Frame::MixerEnable { .. } => "MixerEnable",
            Frame::Custom { .. } => "Custom",
        }
    }
}

// ─── Send + Sync static assertions ──────────────────────────────────────────

// Static assertions: all core types must be Send + Sync.
fn _assert_send_sync() {
    fn require_send<T: Send>() {}
    fn require_sync<T: Sync>() {}
    require_send::<Frame>();
    require_sync::<Frame>();
    require_send::<FrameHeader>();
    require_sync::<FrameHeader>();
    require_send::<AudioData>();
    require_sync::<AudioData>();
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame_types::{category, category_of, is_audio, is_control, is_text};

    fn make_audio_data() -> AudioData {
        AudioData {
            audio: Bytes::from(vec![0u8; 640]),
            sample_rate: 16000,
            num_channels: 1,
        }
    }

    #[test]
    fn header_new_has_unique_ids() {
        let h1 = FrameHeader::new();
        let h2 = FrameHeader::new();
        assert_ne!(h1.id, h2.id);
    }

    #[test]
    fn header_metadata_lazy_init() {
        let mut h = FrameHeader::new();
        assert!(!h.has_metadata());
        assert!(h.metadata().is_empty());

        h.metadata_mut()
            .insert("key".into(), serde_json::Value::Bool(true));
        assert!(h.has_metadata());
        assert_eq!(h.metadata().len(), 1);
    }

    #[test]
    fn audio_data_num_frames() {
        let ad = make_audio_data();
        // 640 bytes / (1 channel * 2 bytes) = 320 frames
        assert_eq!(ad.num_frames(), 320);
    }

    #[test]
    fn audio_data_duration() {
        let ad = make_audio_data();
        // 320 frames / 16000 Hz = 0.02 seconds
        assert!((ad.duration_secs() - 0.02).abs() < 1e-9);
    }

    #[test]
    fn audio_data_zero_channels() {
        let ad = AudioData {
            audio: Bytes::from(vec![0u8; 100]),
            sample_rate: 16000,
            num_channels: 0,
        };
        assert_eq!(ad.num_frames(), 0);
    }

    #[test]
    fn audio_data_zero_sample_rate() {
        let ad = AudioData {
            audio: Bytes::from(vec![0u8; 100]),
            sample_rate: 0,
            num_channels: 1,
        };
        assert_eq!(ad.duration_secs(), 0.0);
    }

    #[test]
    fn frame_header_access() {
        let frame = Frame::Start(FrameHeader::new());
        let id = frame.header().id;
        assert!(id.as_u64() > 0);
    }

    #[test]
    fn frame_header_mut_access() {
        let mut frame = Frame::Start(FrameHeader::new());
        frame.header_mut().pts = Some(12345);
        assert_eq!(frame.header().pts, Some(12345));
    }

    #[test]
    fn frame_classification_system() {
        let frame = Frame::Start(FrameHeader::new());
        assert_eq!(frame.classification(), FrameClass::System);

        let frame = Frame::Heartbeat(FrameHeader::new());
        assert_eq!(frame.classification(), FrameClass::System);

        let frame = Frame::Error {
            header: FrameHeader::new(),
            error: ErrorData {
                message: "bad".into(),
                fatal: false,
                exception: None,
            },
        };
        assert_eq!(frame.classification(), FrameClass::System);

        let frame = Frame::UserStartedSpeaking(FrameHeader::new());
        assert_eq!(frame.classification(), FrameClass::System);
    }

    #[test]
    fn frame_classification_control() {
        assert_eq!(
            Frame::End(FrameHeader::new()).classification(),
            FrameClass::Control
        );
        assert_eq!(
            Frame::Stop(FrameHeader::new()).classification(),
            FrameClass::Control
        );
        assert_eq!(
            Frame::Pause {
                header: FrameHeader::new(),
                processor_name: "p".into()
            }
            .classification(),
            FrameClass::Control
        );
    }

    #[test]
    fn frame_classification_data() {
        let frame = Frame::Text {
            header: FrameHeader::new(),
            data: TextData {
                text: "hi".into(),
            },
        };
        assert_eq!(frame.classification(), FrameClass::Data);

        let frame = Frame::AudioTts {
            header: FrameHeader::new(),
            audio: make_audio_data(),
            context_id: None,
        };
        assert_eq!(frame.classification(), FrameClass::Data);
    }

    #[test]
    fn frame_uninterruptible() {
        assert!(Frame::End(FrameHeader::new()).is_uninterruptible());
        assert!(Frame::Stop(FrameHeader::new()).is_uninterruptible());
        assert!(Frame::Cancel(FrameHeader::new()).is_uninterruptible());
        assert!(!Frame::Start(FrameHeader::new()).is_uninterruptible());
        assert!(!Frame::Heartbeat(FrameHeader::new()).is_uninterruptible());
    }

    #[test]
    fn frame_type_ids_match_categories() {
        let frame = Frame::AudioRawInput {
            header: FrameHeader::new(),
            audio: make_audio_data(),
        };
        assert!(is_audio(frame.type_id()));

        let frame = Frame::Text {
            header: FrameHeader::new(),
            data: TextData {
                text: "x".into(),
            },
        };
        assert!(is_text(frame.type_id()));

        let frame = Frame::End(FrameHeader::new());
        assert!(is_control(frame.type_id()));
    }

    #[test]
    fn frame_type_id_all_variants_nonzero() {
        // Verify a selection of variants produce valid nonzero type IDs.
        let frames: Vec<Frame> = vec![
            Frame::Start(FrameHeader::new()),
            Frame::Cancel(FrameHeader::new()),
            Frame::Heartbeat(FrameHeader::new()),
            Frame::Interruption(FrameHeader::new()),
            Frame::AudioRawOutput {
                header: FrameHeader::new(),
                audio: make_audio_data(),
            },
            Frame::TextLlm {
                header: FrameHeader::new(),
                data: TextData {
                    text: "hi".into(),
                },
            },
            Frame::LlmResponseStart(FrameHeader::new()),
            Frame::TtsStarted(FrameHeader::new()),
            Frame::UserStartedSpeaking(FrameHeader::new()),
            Frame::BotStartedSpeaking(FrameHeader::new()),
            Frame::DtmfOutput {
                header: FrameHeader::new(),
                keys: "1".into(),
            },
            Frame::InterruptionTask(FrameHeader::new()),
            Frame::FunctionCallProgress {
                header: FrameHeader::new(),
                data: Box::new(FuncCallProgressData {
                    function_name: "f".into(),
                    tool_call_id: "t".into(),
                    arguments: "{}".into(),
                }),
            },
        ];
        for f in &frames {
            assert!(f.type_id() > 0, "type_id should be nonzero for {}", f.name());
        }
    }

    #[test]
    fn frame_name_returns_variant_name() {
        assert_eq!(Frame::Start(FrameHeader::new()).name(), "Start");
        assert_eq!(Frame::End(FrameHeader::new()).name(), "End");
        assert_eq!(
            Frame::AudioRawInput {
                header: FrameHeader::new(),
                audio: make_audio_data()
            }
            .name(),
            "AudioRawInput"
        );
    }

    #[test]
    fn frame_clone() {
        let frame = Frame::Text {
            header: FrameHeader::new(),
            data: TextData {
                text: "hello".into(),
            },
        };
        let cloned = frame.clone();
        // Cloned frame gets the same header ID (Clone, not new)
        assert_eq!(frame.header().id, cloned.header().id);
    }

    #[test]
    fn frame_direction_equality() {
        assert_eq!(FrameDirection::Downstream, FrameDirection::Downstream);
        assert_ne!(FrameDirection::Downstream, FrameDirection::Upstream);
    }

    #[test]
    fn custom_frame_type_id_is_base() {
        let frame = Frame::Custom {
            header: FrameHeader::new(),
            data: Box::new(CustomFrameData {
                type_name: "my_frame".into(),
                data: serde_json::Value::Null,
            }),
        };
        assert_eq!(frame.type_id(), frame_type::FRAME);
        assert_eq!(category_of(frame.type_id()), category::BASE);
    }

    #[test]
    fn llm_context_boxed() {
        let frame = Frame::LlmContext {
            header: FrameHeader::new(),
            context: Box::new(LlmContextData {
                messages: vec![serde_json::json!({"role": "user", "content": "hi"})],
            }),
        };
        if let Frame::LlmContext { context, .. } = &frame {
            assert_eq!(context.messages.len(), 1);
        } else {
            panic!("wrong variant");
        }
    }

    /// Guard against size regressions — these limits ensure the hot/cold split
    /// and variant boxing keep the types cache-friendly.
    #[test]
    fn frame_size_regression() {
        use std::mem::size_of;
        let frame_size = size_of::<Frame>();
        let header_size = size_of::<FrameHeader>();
        assert!(
            header_size <= 56,
            "FrameHeader grew to {header_size} bytes (limit 56)"
        );
        assert!(
            frame_size <= 128,
            "Frame grew to {frame_size} bytes (limit 128)"
        );
    }
}
