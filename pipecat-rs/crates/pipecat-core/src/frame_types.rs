/// 8-bit category prefixes (high byte of type_id).
///
/// Each frame type ID is composed as `(category << 8) | sub_type`, giving
/// 256 categories with up to 256 sub-types each.
pub mod category {
    pub const BASE: u16 = 0x00;
    pub const AUDIO: u16 = 0x01;
    pub const TEXT: u16 = 0x02;
    pub const IMAGE: u16 = 0x03;
    pub const VIDEO: u16 = 0x04;
    pub const LLM: u16 = 0x05;
    pub const STT: u16 = 0x06;
    pub const TTS: u16 = 0x07;
    pub const CONTROL: u16 = 0x08;
    pub const SYSTEM: u16 = 0x09;
    pub const FUNCTION: u16 = 0x0A;
    pub const USER: u16 = 0x0B;
    pub const BOT: u16 = 0x0C;
    pub const METRIC: u16 = 0x0D;
    pub const ERROR: u16 = 0x0E;
    pub const MISC: u16 = 0x0F;
    pub const DTMF: u16 = 0x10;
    pub const TASK: u16 = 0x11;
    pub const SERVICE: u16 = 0x12;
    pub const VISION: u16 = 0x13;
    pub const TRANSPORT: u16 = 0x14;
    pub const FILTER: u16 = 0x15;
    pub const MIXER: u16 = 0x16;
}

/// All known frame type IDs as constants.
pub mod frame_type {
    use super::category;

    /// Helper to compose a type ID from category and sub-type.
    const fn id(cat: u16, sub: u16) -> u16 {
        (cat << 8) | sub
    }

    // ── Base hierarchy ───────────────────────────────────────────────
    pub const FRAME: u16 = id(category::BASE, 0x01);
    pub const SYSTEM_FRAME: u16 = id(category::BASE, 0x02);
    pub const DATA_FRAME: u16 = id(category::BASE, 0x03);
    pub const CONTROL_FRAME: u16 = id(category::BASE, 0x04);

    // ── Audio ────────────────────────────────────────────────────────
    pub const AUDIO_RAW_INPUT: u16 = id(category::AUDIO, 0x01);
    pub const AUDIO_RAW_OUTPUT: u16 = id(category::AUDIO, 0x02);
    pub const AUDIO_TTS: u16 = id(category::AUDIO, 0x03);
    pub const AUDIO_SPEECH: u16 = id(category::AUDIO, 0x04);
    pub const AUDIO_MIX: u16 = id(category::AUDIO, 0x05);
    pub const AUDIO_SILENCE: u16 = id(category::AUDIO, 0x06);
    pub const AUDIO_USER: u16 = id(category::AUDIO, 0x07);
    pub const AUDIO_SPEECH_CTRL: u16 = id(category::AUDIO, 0x08);

    // ── Text ─────────────────────────────────────────────────────────
    pub const TEXT_PLAIN: u16 = id(category::TEXT, 0x01);
    pub const TEXT_LLM: u16 = id(category::TEXT, 0x02);
    pub const TEXT_TRANSCRIPTION: u16 = id(category::TEXT, 0x03);
    pub const TEXT_INTERIM_TRANS: u16 = id(category::TEXT, 0x04);
    pub const TEXT_AGGREGATED: u16 = id(category::TEXT, 0x05);
    pub const TEXT_TTS: u16 = id(category::TEXT, 0x06);
    pub const TEXT_TRANSLATION: u16 = id(category::TEXT, 0x07);
    pub const TEXT_INPUT_RAW: u16 = id(category::TEXT, 0x08);

    // ── Image ────────────────────────────────────────────────────────
    pub const IMAGE_OUTPUT: u16 = id(category::IMAGE, 0x01);
    pub const IMAGE_URL: u16 = id(category::IMAGE, 0x02);
    pub const IMAGE_SPRITE: u16 = id(category::IMAGE, 0x03);
    pub const IMAGE_INPUT: u16 = id(category::IMAGE, 0x04);
    pub const IMAGE_USER: u16 = id(category::IMAGE, 0x05);
    pub const IMAGE_ASSISTANT: u16 = id(category::IMAGE, 0x06);

    // ── LLM ──────────────────────────────────────────────────────────
    pub const LLM_CONTEXT: u16 = id(category::LLM, 0x01);
    pub const LLM_MESSAGES: u16 = id(category::LLM, 0x02);
    pub const LLM_MESSAGES_APPEND: u16 = id(category::LLM, 0x03);
    pub const LLM_RESPONSE_START: u16 = id(category::LLM, 0x04);
    pub const LLM_RESPONSE_END: u16 = id(category::LLM, 0x05);
    pub const LLM_RUN: u16 = id(category::LLM, 0x06);
    pub const LLM_TOOL_CALL: u16 = id(category::LLM, 0x07);
    pub const LLM_TOOL_RESULT: u16 = id(category::LLM, 0x08);
    pub const LLM_THOUGHT_TEXT: u16 = id(category::LLM, 0x09);
    pub const LLM_THOUGHT_START: u16 = id(category::LLM, 0x0A);
    pub const LLM_THOUGHT_END: u16 = id(category::LLM, 0x0B);
    pub const LLM_MESSAGES_UPDATE: u16 = id(category::LLM, 0x0C);
    pub const LLM_SET_TOOLS: u16 = id(category::LLM, 0x0D);
    pub const LLM_SET_TOOL_CHOICE: u16 = id(category::LLM, 0x0E);
    pub const LLM_ENABLE_CACHING: u16 = id(category::LLM, 0x0F);
    pub const LLM_CONFIGURE_OUTPUT: u16 = id(category::LLM, 0x10);
    pub const LLM_CTX_SUMMARY_REQ: u16 = id(category::LLM, 0x11);
    pub const LLM_CTX_SUMMARY_RESULT: u16 = id(category::LLM, 0x12);
    pub const LLM_UPDATE_SETTINGS: u16 = id(category::LLM, 0x13);

    // ── STT ──────────────────────────────────────────────────────────
    pub const STT_MUTE: u16 = id(category::STT, 0x01);
    pub const STT_UPDATE_SETTINGS: u16 = id(category::STT, 0x02);
    pub const STT_LANGUAGE_UPDATE: u16 = id(category::STT, 0x03);
    pub const STT_TRANSCRIPTION_UPDATE: u16 = id(category::STT, 0x04);
    pub const STT_METADATA: u16 = id(category::STT, 0x05);

    // ── TTS ──────────────────────────────────────────────────────────
    pub const TTS_STARTED: u16 = id(category::TTS, 0x01);
    pub const TTS_STOPPED: u16 = id(category::TTS, 0x02);
    pub const TTS_UPDATE_SETTINGS: u16 = id(category::TTS, 0x03);
    pub const TTS_SPEAK: u16 = id(category::TTS, 0x04);

    // ── Control ──────────────────────────────────────────────────────
    pub const CTRL_START: u16 = id(category::CONTROL, 0x01);
    pub const CTRL_END: u16 = id(category::CONTROL, 0x02);
    pub const CTRL_STOP: u16 = id(category::CONTROL, 0x03);
    pub const CTRL_CANCEL: u16 = id(category::CONTROL, 0x04);
    pub const CTRL_INTERRUPT: u16 = id(category::CONTROL, 0x05);
    pub const CTRL_END_TASK: u16 = id(category::CONTROL, 0x06);
    pub const CTRL_START_INTERRUPT: u16 = id(category::CONTROL, 0x07);
    pub const CTRL_PAUSE: u16 = id(category::CONTROL, 0x08);
    pub const CTRL_RESUME: u16 = id(category::CONTROL, 0x09);
    pub const CTRL_PAUSE_URGENT: u16 = id(category::CONTROL, 0x0A);
    pub const CTRL_RESUME_URGENT: u16 = id(category::CONTROL, 0x0B);
    pub const CTRL_OUTPUT_READY: u16 = id(category::CONTROL, 0x0C);
    pub const CTRL_VAD_UPDATE: u16 = id(category::CONTROL, 0x0D);

    // ── System ───────────────────────────────────────────────────────
    pub const SYS_HEARTBEAT: u16 = id(category::SYSTEM, 0x01);
    pub const SYS_METRICS: u16 = id(category::SYSTEM, 0x02);
    pub const SYS_USER_IDLE_TIMEOUT_UPDATE: u16 = id(category::SYSTEM, 0x03);

    // ── Function calls ───────────────────────────────────────────────
    pub const FUNC_CALL_PROGRESS: u16 = id(category::FUNCTION, 0x01);
    pub const FUNC_CALL_RESULT: u16 = id(category::FUNCTION, 0x02);
    pub const FUNC_CALLS_STARTED: u16 = id(category::FUNCTION, 0x03);
    pub const FUNC_CALL_CANCEL: u16 = id(category::FUNCTION, 0x04);

    // ── User events ──────────────────────────────────────────────────
    pub const USER_STARTED_SPEAKING: u16 = id(category::USER, 0x01);
    pub const USER_STOPPED_SPEAKING: u16 = id(category::USER, 0x02);
    pub const USER_MUTE_STARTED: u16 = id(category::USER, 0x03);
    pub const USER_MUTE_STOPPED: u16 = id(category::USER, 0x04);
    pub const USER_SPEAKING: u16 = id(category::USER, 0x05);
    pub const USER_EMULATE_STARTED: u16 = id(category::USER, 0x06);
    pub const USER_EMULATE_STOPPED: u16 = id(category::USER, 0x07);
    pub const USER_VAD_STARTED: u16 = id(category::USER, 0x08);
    pub const USER_VAD_STOPPED: u16 = id(category::USER, 0x09);
    pub const USER_IMAGE_REQUEST: u16 = id(category::USER, 0x0A);

    // ── Bot events ───────────────────────────────────────────────────
    pub const BOT_STARTED_SPEAKING: u16 = id(category::BOT, 0x01);
    pub const BOT_STOPPED_SPEAKING: u16 = id(category::BOT, 0x02);
    pub const BOT_SPEAKING: u16 = id(category::BOT, 0x03);

    // ── Error ────────────────────────────────────────────────────────
    pub const ERROR_GENERAL: u16 = id(category::ERROR, 0x01);
    pub const ERROR_FATAL: u16 = id(category::ERROR, 0x02);

    // ── DTMF ─────────────────────────────────────────────────────────
    pub const DTMF_OUTPUT: u16 = id(category::DTMF, 0x01);
    pub const DTMF_INPUT: u16 = id(category::DTMF, 0x02);
    pub const DTMF_OUTPUT_URGENT: u16 = id(category::DTMF, 0x03);

    // ── Task ─────────────────────────────────────────────────────────
    pub const TASK_FRAME: u16 = id(category::TASK, 0x01);
    pub const TASK_CANCEL: u16 = id(category::TASK, 0x02);
    pub const TASK_STOP: u16 = id(category::TASK, 0x03);
    pub const TASK_INTERRUPTION: u16 = id(category::TASK, 0x04);
    pub const TASK_BOT_INTERRUPT: u16 = id(category::TASK, 0x05);

    // ── Service ──────────────────────────────────────────────────────
    pub const SERVICE_METADATA: u16 = id(category::SERVICE, 0x01);
    pub const SERVICE_UPDATE: u16 = id(category::SERVICE, 0x02);
    pub const SERVICE_SWITCHER: u16 = id(category::SERVICE, 0x03);
    pub const SERVICE_SWITCH_MANUAL: u16 = id(category::SERVICE, 0x04);
    pub const SERVICE_SWITCHER_META: u16 = id(category::SERVICE, 0x05);

    // ── Vision ───────────────────────────────────────────────────────
    pub const VISION_TEXT: u16 = id(category::VISION, 0x01);
    pub const VISION_RESP_START: u16 = id(category::VISION, 0x02);
    pub const VISION_RESP_END: u16 = id(category::VISION, 0x03);

    // ── Transport ────────────────────────────────────────────────────
    pub const TRANSPORT_MSG_OUT: u16 = id(category::TRANSPORT, 0x01);
    pub const TRANSPORT_MSG_BIDIR: u16 = id(category::TRANSPORT, 0x02);
    pub const TRANSPORT_MSG_IN: u16 = id(category::TRANSPORT, 0x03);
    pub const TRANSPORT_MSG_IN_URGENT: u16 = id(category::TRANSPORT, 0x04);
    pub const TRANSPORT_MSG_OUT_URGENT: u16 = id(category::TRANSPORT, 0x05);
    pub const TRANSPORT_MSG_BIDIR_URGENT: u16 = id(category::TRANSPORT, 0x06);

    // ── Filter ───────────────────────────────────────────────────────
    pub const FILTER_CONTROL: u16 = id(category::FILTER, 0x01);
    pub const FILTER_UPDATE: u16 = id(category::FILTER, 0x02);
    pub const FILTER_ENABLE: u16 = id(category::FILTER, 0x03);

    // ── Mixer ────────────────────────────────────────────────────────
    pub const MIXER_CONTROL: u16 = id(category::MIXER, 0x01);
    pub const MIXER_UPDATE: u16 = id(category::MIXER, 0x02);
    pub const MIXER_ENABLE: u16 = id(category::MIXER, 0x03);
}

/// Extract the category (high byte) from a type ID.
#[inline]
pub fn category_of(type_id: u16) -> u16 {
    (type_id & 0xFF00) >> 8
}

/// Returns true if the type ID belongs to the audio category.
#[inline]
pub fn is_audio(type_id: u16) -> bool {
    category_of(type_id) == category::AUDIO
}

/// Returns true if the type ID belongs to the text category.
#[inline]
pub fn is_text(type_id: u16) -> bool {
    category_of(type_id) == category::TEXT
}

/// Returns true if the type ID belongs to the control category.
#[inline]
pub fn is_control(type_id: u16) -> bool {
    category_of(type_id) == category::CONTROL
}

/// Returns true if the type ID belongs to the system category.
#[inline]
pub fn is_system(type_id: u16) -> bool {
    category_of(type_id) == category::SYSTEM
}

/// Returns true if the type ID belongs to the user event category.
#[inline]
pub fn is_user_event(type_id: u16) -> bool {
    category_of(type_id) == category::USER
}

/// Returns true if the type ID belongs to the bot event category.
#[inline]
pub fn is_bot_event(type_id: u16) -> bool {
    category_of(type_id) == category::BOT
}

/// Returns true if the type ID belongs to the LLM category.
#[inline]
pub fn is_llm(type_id: u16) -> bool {
    category_of(type_id) == category::LLM
}

/// Returns true if the type ID belongs to the image category.
#[inline]
pub fn is_image(type_id: u16) -> bool {
    category_of(type_id) == category::IMAGE
}

/// Returns true if the type ID belongs to the error category.
#[inline]
pub fn is_error(type_id: u16) -> bool {
    category_of(type_id) == category::ERROR
}

/// Returns true if the type ID belongs to the task category.
#[inline]
pub fn is_task(type_id: u16) -> bool {
    category_of(type_id) == category::TASK
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn category_encoding() {
        assert_eq!(category_of(frame_type::AUDIO_RAW_INPUT), category::AUDIO);
        assert_eq!(category_of(frame_type::TEXT_PLAIN), category::TEXT);
        assert_eq!(category_of(frame_type::CTRL_START), category::CONTROL);
        assert_eq!(category_of(frame_type::SYS_HEARTBEAT), category::SYSTEM);
        assert_eq!(category_of(frame_type::LLM_CONTEXT), category::LLM);
        assert_eq!(category_of(frame_type::ERROR_GENERAL), category::ERROR);
    }

    #[test]
    fn category_helpers() {
        assert!(is_audio(frame_type::AUDIO_RAW_INPUT));
        assert!(is_audio(frame_type::AUDIO_TTS));
        assert!(!is_audio(frame_type::TEXT_PLAIN));

        assert!(is_text(frame_type::TEXT_PLAIN));
        assert!(is_text(frame_type::TEXT_LLM));
        assert!(!is_text(frame_type::AUDIO_RAW_INPUT));

        assert!(is_control(frame_type::CTRL_START));
        assert!(is_control(frame_type::CTRL_END));
        assert!(!is_control(frame_type::SYS_HEARTBEAT));

        assert!(is_system(frame_type::SYS_HEARTBEAT));
        assert!(is_system(frame_type::SYS_METRICS));
        assert!(!is_system(frame_type::CTRL_START));

        assert!(is_user_event(frame_type::USER_STARTED_SPEAKING));
        assert!(is_bot_event(frame_type::BOT_STOPPED_SPEAKING));
        assert!(is_llm(frame_type::LLM_TOOL_CALL));
        assert!(is_image(frame_type::IMAGE_OUTPUT));
        assert!(is_error(frame_type::ERROR_FATAL));
        assert!(is_task(frame_type::TASK_INTERRUPTION));
    }

    #[test]
    fn sub_types_are_distinct_within_category() {
        assert_ne!(frame_type::AUDIO_RAW_INPUT, frame_type::AUDIO_RAW_OUTPUT);
        assert_ne!(frame_type::TEXT_PLAIN, frame_type::TEXT_LLM);
        assert_ne!(frame_type::CTRL_START, frame_type::CTRL_END);
    }

    #[test]
    fn type_id_layout() {
        // AUDIO_RAW_INPUT = (0x01 << 8) | 0x01 = 0x0101
        assert_eq!(frame_type::AUDIO_RAW_INPUT, 0x0101);
        // CTRL_START = (0x08 << 8) | 0x01 = 0x0801
        assert_eq!(frame_type::CTRL_START, 0x0801);
        // ERROR_FATAL = (0x0E << 8) | 0x02 = 0x0E02
        assert_eq!(frame_type::ERROR_FATAL, 0x0E02);
    }
}
