//! Frame type registry for mapping between Python frame classes and Rust type IDs.
//!
//! Provides a bidirectional mapping so that frames can be converted between the
//! Python dataclass world and the Rust enum world based on their `type_id`.

use std::collections::HashMap;

use pipecat_core::frame_types::frame_type;

/// Classification of how a frame type is handled at the Python↔Rust boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameTier {
    /// Tier 1: Field-by-field conversion with typed accessors.
    /// These are the hot-path frames (audio, text, control, lifecycle).
    Native,
    /// Tier 2: Known Python frame types that are serialized to `Frame::Custom`
    /// and reconstructed on return via JSON round-trip.
    Opaque,
    /// Tier 3: User-defined frame subclasses registered at runtime.
    Custom,
}

/// Entry in the frame type registry.
#[derive(Debug, Clone)]
#[allow(dead_code)] // Fields are used for debugging and will be used for runtime introspection
pub struct FrameRegistryEntry {
    /// The 16-bit type ID (matches both Python `FrameType` and Rust `frame_type`).
    pub type_id: u16,
    /// Python class name (e.g. "TextFrame", "InputAudioRawFrame").
    pub py_class_name: String,
    /// Rust Frame variant name (e.g. "Text", "AudioRawInput").
    pub rust_variant_name: String,
    /// How this frame type crosses the boundary.
    pub tier: FrameTier,
}

/// Bidirectional registry mapping Python frame classes ↔ Rust type IDs.
pub struct FrameTypeRegistry {
    /// Python class name → registry entry.
    py_to_info: HashMap<String, FrameRegistryEntry>,
    /// Rust type_id → registry entry.
    type_id_to_info: HashMap<u16, FrameRegistryEntry>,
}

impl FrameTypeRegistry {
    /// Build the registry with all known frame types.
    pub fn new() -> Self {
        let mut registry = Self {
            py_to_info: HashMap::new(),
            type_id_to_info: HashMap::new(),
        };

        // ── Tier 1: Native frames (field-by-field conversion) ──────────

        // Control / lifecycle
        registry.register_native("StartFrame", "Start", frame_type::CTRL_START);
        registry.register_native("EndFrame", "End", frame_type::CTRL_END);
        registry.register_native("StopFrame", "Stop", frame_type::CTRL_STOP);
        registry.register_native("CancelFrame", "Cancel", frame_type::CTRL_CANCEL);
        registry.register_native("InterruptionFrame", "Interruption", frame_type::CTRL_INTERRUPT);
        registry.register_native(
            "StartInterruptionFrame",
            "StartInterruption",
            frame_type::CTRL_START_INTERRUPT,
        );
        registry.register_native("HeartbeatFrame", "Heartbeat", frame_type::SYS_HEARTBEAT);
        registry.register_native(
            "FrameProcessorPauseFrame",
            "Pause",
            frame_type::CTRL_PAUSE,
        );
        registry.register_native(
            "FrameProcessorResumeFrame",
            "Resume",
            frame_type::CTRL_RESUME,
        );

        // Audio
        registry.register_native(
            "InputAudioRawFrame",
            "AudioRawInput",
            frame_type::AUDIO_RAW_INPUT,
        );
        registry.register_native(
            "OutputAudioRawFrame",
            "AudioRawOutput",
            frame_type::AUDIO_RAW_OUTPUT,
        );
        registry.register_native("TTSAudioRawFrame", "AudioTts", frame_type::AUDIO_TTS);
        registry.register_native(
            "SpeechOutputAudioRawFrame",
            "AudioSpeech",
            frame_type::AUDIO_SPEECH,
        );
        // AudioMix and AudioSilence have no dedicated Python frame class.
        // They are converted to/from OutputAudioRawFrame with extra fields.
        registry.register_native("OutputAudioRawFrame", "AudioMix", frame_type::AUDIO_MIX);
        registry.register_native(
            "OutputAudioRawFrame",
            "AudioSilence",
            frame_type::AUDIO_SILENCE,
        );
        registry.register_native("UserAudioRawFrame", "AudioUser", frame_type::AUDIO_USER);

        // Text
        registry.register_native("TextFrame", "Text", frame_type::TEXT_PLAIN);
        registry.register_native("LLMTextFrame", "TextLlm", frame_type::TEXT_LLM);
        registry.register_native(
            "AggregatedTextFrame",
            "TextAggregated",
            frame_type::TEXT_AGGREGATED,
        );
        registry.register_native("TTSTextFrame", "TextTts", frame_type::TEXT_TTS);
        registry.register_native("InputTextRawFrame", "TextInputRaw", frame_type::TEXT_INPUT_RAW);

        // Transcription
        registry.register_native(
            "TranscriptionFrame",
            "Transcription",
            frame_type::TEXT_TRANSCRIPTION,
        );
        registry.register_native(
            "InterimTranscriptionFrame",
            "InterimTranscription",
            frame_type::TEXT_INTERIM_TRANS,
        );

        // LLM
        registry.register_native("LLMContextFrame", "LlmContext", frame_type::LLM_CONTEXT);
        registry.register_native("LLMMessagesFrame", "LlmMessages", frame_type::LLM_MESSAGES);
        registry.register_native(
            "LLMFullResponseStartFrame",
            "LlmResponseStart",
            frame_type::LLM_RESPONSE_START,
        );
        registry.register_native(
            "LLMFullResponseEndFrame",
            "LlmResponseEnd",
            frame_type::LLM_RESPONSE_END,
        );
        registry.register_native("LLMRunFrame", "LlmRun", frame_type::LLM_RUN);
        registry.register_native("LLMToolCallFrame", "LlmToolCall", frame_type::LLM_TOOL_CALL);
        registry.register_native(
            "LLMToolResultFrame",
            "LlmToolResult",
            frame_type::LLM_TOOL_RESULT,
        );
        registry.register_native(
            "LLMThoughtTextFrame",
            "LlmThoughtText",
            frame_type::LLM_THOUGHT_TEXT,
        );
        registry.register_native(
            "LLMThoughtStartFrame",
            "LlmThoughtStart",
            frame_type::LLM_THOUGHT_START,
        );
        registry.register_native(
            "LLMThoughtEndFrame",
            "LlmThoughtEnd",
            frame_type::LLM_THOUGHT_END,
        );
        registry.register_native(
            "LLMMessagesUpdateFrame",
            "LlmMessagesUpdate",
            frame_type::LLM_MESSAGES_UPDATE,
        );
        registry.register_native(
            "LLMMessagesAppendFrame",
            "LlmMessagesAppend",
            frame_type::LLM_MESSAGES_APPEND,
        );
        registry.register_native("LLMSetToolsFrame", "LlmSetTools", frame_type::LLM_SET_TOOLS);
        registry.register_native(
            "LLMSetToolChoiceFrame",
            "LlmSetToolChoice",
            frame_type::LLM_SET_TOOL_CHOICE,
        );
        registry.register_native(
            "LLMEnablePromptCachingFrame",
            "LlmEnableCaching",
            frame_type::LLM_ENABLE_CACHING,
        );
        registry.register_native(
            "LLMConfigureOutputFrame",
            "LlmConfigureOutput",
            frame_type::LLM_CONFIGURE_OUTPUT,
        );
        registry.register_native(
            "LLMContextSummaryRequestFrame",
            "LlmCtxSummaryRequest",
            frame_type::LLM_CTX_SUMMARY_REQ,
        );
        registry.register_native(
            "LLMContextSummaryResultFrame",
            "LlmCtxSummaryResult",
            frame_type::LLM_CTX_SUMMARY_RESULT,
        );
        registry.register_native(
            "LLMUpdateSettingsFrame",
            "LlmUpdateSettings",
            frame_type::LLM_UPDATE_SETTINGS,
        );

        // STT
        registry.register_native("STTMuteFrame", "SttMute", frame_type::STT_MUTE);
        registry.register_native(
            "STTUpdateSettingsFrame",
            "SttUpdateSettings",
            frame_type::STT_UPDATE_SETTINGS,
        );
        // STTLanguageUpdateFrame has no dedicated Python class — type_id exists
        // but no frame class uses it. The Rust→Python conversion returns a dict.
        registry.register_opaque("STTLanguageUpdateFrame", frame_type::STT_LANGUAGE_UPDATE);

        // TTS
        registry.register_native("TTSStartedFrame", "TtsStarted", frame_type::TTS_STARTED);
        registry.register_native("TTSStoppedFrame", "TtsStopped", frame_type::TTS_STOPPED);
        registry.register_native(
            "TTSUpdateSettingsFrame",
            "TtsUpdateSettings",
            frame_type::TTS_UPDATE_SETTINGS,
        );
        registry.register_native("TTSSpeakFrame", "TtsSpeak", frame_type::TTS_SPEAK);

        // User / Bot events
        registry.register_native(
            "UserStartedSpeakingFrame",
            "UserStartedSpeaking",
            frame_type::USER_STARTED_SPEAKING,
        );
        registry.register_native(
            "UserStoppedSpeakingFrame",
            "UserStoppedSpeaking",
            frame_type::USER_STOPPED_SPEAKING,
        );
        registry.register_native(
            "BotStartedSpeakingFrame",
            "BotStartedSpeaking",
            frame_type::BOT_STARTED_SPEAKING,
        );
        registry.register_native(
            "BotStoppedSpeakingFrame",
            "BotStoppedSpeaking",
            frame_type::BOT_STOPPED_SPEAKING,
        );

        // Error
        registry.register_native("ErrorFrame", "Error", frame_type::ERROR_GENERAL);

        // Metrics
        registry.register_native("MetricsFrame", "Metrics", frame_type::SYS_METRICS);

        // DTMF
        registry.register_native("OutputDTMFFrame", "DtmfOutput", frame_type::DTMF_OUTPUT);
        registry.register_native("InputDTMFFrame", "DtmfInput", frame_type::DTMF_INPUT);

        // Task
        registry.register_native(
            "InterruptionTaskFrame",
            "InterruptionTask",
            frame_type::TASK_INTERRUPTION,
        );
        registry.register_native(
            "BotInterruptionFrame",
            "BotInterruption",
            frame_type::TASK_BOT_INTERRUPT,
        );
        registry.register_native("CancelTaskFrame", "CancelTask", frame_type::TASK_CANCEL);
        registry.register_native("StopTaskFrame", "StopTask", frame_type::TASK_STOP);
        registry.register_native("EndTaskFrame", "EndTask", frame_type::CTRL_END_TASK);

        // Function calls
        registry.register_native(
            "FunctionCallInProgressFrame",
            "FunctionCallProgress",
            frame_type::FUNC_CALL_PROGRESS,
        );
        registry.register_native(
            "FunctionCallResultFrame",
            "FunctionCallResult",
            frame_type::FUNC_CALL_RESULT,
        );

        // Image
        registry.register_native("OutputImageRawFrame", "ImageOutput", frame_type::IMAGE_OUTPUT);
        registry.register_native("URLImageRawFrame", "ImageUrl", frame_type::IMAGE_URL);
        registry.register_native("InputImageRawFrame", "ImageInput", frame_type::IMAGE_INPUT);

        registry
    }

    fn register_native(&mut self, py_class: &str, rust_variant: &str, type_id: u16) {
        let entry = FrameRegistryEntry {
            type_id,
            py_class_name: py_class.to_string(),
            rust_variant_name: rust_variant.to_string(),
            tier: FrameTier::Native,
        };
        self.py_to_info.insert(py_class.to_string(), entry.clone());
        self.type_id_to_info.insert(type_id, entry);
    }

    /// Register an opaque (Tier 2) frame type at build time.
    #[allow(dead_code)]
    pub fn register_opaque(&mut self, py_class: &str, type_id: u16) {
        let entry = FrameRegistryEntry {
            type_id,
            py_class_name: py_class.to_string(),
            rust_variant_name: "Custom".to_string(),
            tier: FrameTier::Opaque,
        };
        self.py_to_info.insert(py_class.to_string(), entry.clone());
        self.type_id_to_info.insert(type_id, entry);
    }

    /// Register a custom (Tier 3) frame type at runtime.
    #[allow(dead_code)]
    pub fn register_custom(&mut self, py_class: &str, type_id: u16) {
        let entry = FrameRegistryEntry {
            type_id,
            py_class_name: py_class.to_string(),
            rust_variant_name: "Custom".to_string(),
            tier: FrameTier::Custom,
        };
        self.py_to_info.insert(py_class.to_string(), entry.clone());
        self.type_id_to_info.insert(type_id, entry);
    }

    /// Look up a registry entry by Python class name.
    #[allow(dead_code)] // Used in tests; will be used for Python→Rust frame dispatch
    pub fn by_py_class(&self, class_name: &str) -> Option<&FrameRegistryEntry> {
        self.py_to_info.get(class_name)
    }

    /// Look up a registry entry by Rust type ID.
    pub fn by_type_id(&self, type_id: u16) -> Option<&FrameRegistryEntry> {
        self.type_id_to_info.get(&type_id)
    }

    /// Get the Python class name for a Rust type ID.
    #[allow(dead_code)] // Will be used for Rust→Python frame reconstruction
    pub fn py_class_for_type_id(&self, type_id: u16) -> Option<&str> {
        self.type_id_to_info
            .get(&type_id)
            .map(|e| e.py_class_name.as_str())
    }

    /// Get the tier classification for a type ID.
    #[allow(dead_code)] // Used in tests; will be used for tier-based dispatch routing
    pub fn tier_for_type_id(&self, type_id: u16) -> FrameTier {
        self.type_id_to_info
            .get(&type_id)
            .map(|e| e.tier)
            .unwrap_or(FrameTier::Custom)
    }
}

impl Default for FrameTypeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_has_core_types() {
        let reg = FrameTypeRegistry::new();
        assert!(reg.by_py_class("TextFrame").is_some());
        assert!(reg.by_py_class("InputAudioRawFrame").is_some());
        assert!(reg.by_py_class("StartFrame").is_some());
        assert!(reg.by_py_class("EndFrame").is_some());
    }

    #[test]
    fn registry_type_id_lookup() {
        let reg = FrameTypeRegistry::new();
        let entry = reg.by_type_id(frame_type::TEXT_PLAIN).unwrap();
        assert_eq!(entry.py_class_name, "TextFrame");
        assert_eq!(entry.rust_variant_name, "Text");
        assert_eq!(entry.tier, FrameTier::Native);
    }

    #[test]
    fn registry_bidirectional() {
        let reg = FrameTypeRegistry::new();
        let entry = reg.by_py_class("InputAudioRawFrame").unwrap();
        let back = reg.by_type_id(entry.type_id).unwrap();
        assert_eq!(entry.py_class_name, back.py_class_name);
    }

    #[test]
    fn unknown_type_returns_custom_tier() {
        let reg = FrameTypeRegistry::new();
        assert_eq!(reg.tier_for_type_id(0xFFFF), FrameTier::Custom);
    }
}
