//! Frame conversion between Python dataclass frames and Rust Frame enum.
//!
//! Tier 1 (Native): Field-by-field extraction/construction.
//! Tier 2/3 (Opaque/Custom): Serialize to JSON, wrap in `Frame::Custom`.

use bytes::Bytes;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

use pipecat_core::frame_types::frame_type;
use pipecat_core::{
    AudioData, ErrorData, Frame, FrameHeader, ImageData, LlmContextData,
    TextData, TranscriptionData,
};
use pipecat_core::frames::{
    CustomFrameData, FuncCallProgressData, FuncCallResultData,
    LlmToolCallData, LlmToolResultData,
};

use crate::interruption::{FrameSidecar, SidecarKind};
use crate::registry::{FrameTier, FrameTypeRegistry};

// ── Python → Rust ─────────────────────────────────────────────────────────

/// Convert a Python frame object to a Rust Frame.
///
/// For Tier 1 frames, extracts fields directly from the Python object.
/// For Tier 2/3 frames, serializes `__dict__` to JSON and wraps in `Frame::Custom`.
pub fn py_frame_to_rust(
    py: Python<'_>,
    py_frame: &Bound<'_, PyAny>,
    registry: &FrameTypeRegistry,
) -> PyResult<Frame> {
    py_frame_to_rust_with_sidecar(py, py_frame, registry, None)
}

/// Convert a Python frame to Rust, with optional sidecar for non-serializable objects.
///
/// When a `sidecar` is provided, non-serializable Python objects (like asyncio.Event
/// on InterruptionFrame) are stored in the sidecar keyed by frame ID, and can be
/// retrieved when the frame is converted back to Python.
pub fn py_frame_to_rust_with_sidecar(
    py: Python<'_>,
    py_frame: &Bound<'_, PyAny>,
    registry: &FrameTypeRegistry,
    sidecar: Option<&FrameSidecar>,
) -> PyResult<Frame> {
    let type_id: u16 = py_frame
        .getattr("type_id")?
        .extract()?;

    let entry = registry.by_type_id(type_id);
    let tier = entry.map(|e| e.tier).unwrap_or(FrameTier::Custom);

    let frame = match tier {
        FrameTier::Native => py_frame_to_rust_native(py, py_frame, type_id),
        FrameTier::Opaque | FrameTier::Custom => py_frame_to_rust_opaque(py, py_frame, type_id, registry),
    }?;

    // If a sidecar is provided and this is an InterruptionFrame with an event,
    // store the event for later retrieval when the frame is converted back.
    if let Some(sc) = sidecar {
        if type_id == frame_type::CTRL_INTERRUPT {
            if let Ok(event_obj) = py_frame.getattr("event") {
                if !event_obj.is_none() {
                    let frame_id = frame.header().id.as_u64();
                    sc.store(frame_id, event_obj.unbind(), SidecarKind::InterruptionEvent);
                }
            }
        }
    }

    Ok(frame)
}

/// Convert a Tier 1 Python frame to a Rust Frame by extracting fields.
fn py_frame_to_rust_native(
    py: Python<'_>,
    py_frame: &Bound<'_, PyAny>,
    type_id: u16,
) -> PyResult<Frame> {
    let header = extract_header(py_frame)?;

    match type_id {
        // ── Control / Lifecycle ───────────────────────────────────────
        frame_type::CTRL_START => Ok(Frame::Start(header)),
        frame_type::CTRL_END => Ok(Frame::End(header)),
        frame_type::CTRL_STOP => Ok(Frame::Stop(header)),
        frame_type::CTRL_CANCEL => Ok(Frame::Cancel(header)),
        frame_type::CTRL_INTERRUPT => Ok(Frame::Interruption(header)),
        frame_type::CTRL_START_INTERRUPT => Ok(Frame::StartInterruption(header)),
        frame_type::SYS_HEARTBEAT => Ok(Frame::Heartbeat(header)),
        frame_type::CTRL_PAUSE => {
            let processor_name: String = py_frame.getattr("processor_name")?.extract()?;
            Ok(Frame::Pause {
                header,
                processor_name,
            })
        }
        frame_type::CTRL_RESUME => {
            let processor_name: String = py_frame.getattr("processor_name")?.extract()?;
            Ok(Frame::Resume {
                header,
                processor_name,
            })
        }

        // ── Audio ─────────────────────────────────────────────────────
        frame_type::AUDIO_RAW_INPUT => {
            let audio = extract_audio_data(py, py_frame)?;
            Ok(Frame::AudioRawInput { header, audio })
        }
        frame_type::AUDIO_RAW_OUTPUT => {
            let audio = extract_audio_data(py, py_frame)?;
            Ok(Frame::AudioRawOutput { header, audio })
        }
        frame_type::AUDIO_TTS => {
            let audio = extract_audio_data(py, py_frame)?;
            let context_id: Option<String> = py_frame
                .getattr("context_id")
                .ok()
                .and_then(|v| v.extract().ok());
            Ok(Frame::AudioTts {
                header,
                audio,
                context_id,
            })
        }
        frame_type::AUDIO_SPEECH => {
            let audio = extract_audio_data(py, py_frame)?;
            Ok(Frame::AudioSpeech { header, audio })
        }
        frame_type::AUDIO_MIX => {
            let audio = extract_audio_data(py, py_frame)?;
            let mix_handle_id: String = py_frame.getattr("mix_handle_id")?.extract()?;
            Ok(Frame::AudioMix {
                header,
                audio,
                mix_handle_id,
            })
        }
        frame_type::AUDIO_SILENCE => {
            let audio = extract_audio_data(py, py_frame)?;
            Ok(Frame::AudioSilence { header, audio })
        }
        frame_type::AUDIO_USER => {
            let audio = extract_audio_data(py, py_frame)?;
            Ok(Frame::AudioUser { header, audio })
        }

        // ── Text ──────────────────────────────────────────────────────
        frame_type::TEXT_PLAIN => {
            let text: String = py_frame.getattr("text")?.extract()?;
            Ok(Frame::Text {
                header,
                data: TextData { text },
            })
        }
        frame_type::TEXT_LLM => {
            let text: String = py_frame.getattr("text")?.extract()?;
            Ok(Frame::TextLlm {
                header,
                data: TextData { text },
            })
        }
        frame_type::TEXT_AGGREGATED => {
            let text: String = py_frame.getattr("text")?.extract()?;
            Ok(Frame::TextAggregated {
                header,
                data: TextData { text },
            })
        }
        frame_type::TEXT_TTS => {
            let text: String = py_frame.getattr("text")?.extract()?;
            Ok(Frame::TextTts {
                header,
                data: TextData { text },
            })
        }
        frame_type::TEXT_INPUT_RAW => {
            let text: String = py_frame.getattr("text")?.extract()?;
            Ok(Frame::TextInputRaw {
                header,
                data: TextData { text },
            })
        }

        // ── Transcription ─────────────────────────────────────────────
        frame_type::TEXT_TRANSCRIPTION => {
            let data = extract_transcription_data(py_frame)?;
            Ok(Frame::Transcription {
                header,
                data: Box::new(data),
            })
        }
        frame_type::TEXT_INTERIM_TRANS => {
            let data = extract_transcription_data(py_frame)?;
            Ok(Frame::InterimTranscription {
                header,
                data: Box::new(data),
            })
        }

        // ── LLM ───────────────────────────────────────────────────────
        frame_type::LLM_RESPONSE_START => Ok(Frame::LlmResponseStart(header)),
        frame_type::LLM_RESPONSE_END => Ok(Frame::LlmResponseEnd(header)),
        frame_type::LLM_RUN => Ok(Frame::LlmRun(header)),
        frame_type::LLM_THOUGHT_START => Ok(Frame::LlmThoughtStart(header)),
        frame_type::LLM_THOUGHT_END => Ok(Frame::LlmThoughtEnd(header)),
        frame_type::LLM_ENABLE_CACHING => Ok(Frame::LlmEnableCaching(header)),
        frame_type::LLM_CTX_SUMMARY_REQ => Ok(Frame::LlmCtxSummaryRequest(header)),
        frame_type::LLM_THOUGHT_TEXT => {
            let text: String = py_frame.getattr("text")?.extract()?;
            Ok(Frame::LlmThoughtText {
                header,
                data: TextData { text },
            })
        }
        frame_type::LLM_CTX_SUMMARY_RESULT => {
            let summary: String = py_frame.getattr("summary")?.extract()?;
            Ok(Frame::LlmCtxSummaryResult { header, summary })
        }
        frame_type::LLM_CONTEXT => {
            let messages_json = py_obj_to_json(py, &py_frame.getattr("messages")?)?;
            let messages = match messages_json {
                serde_json::Value::Array(arr) => arr,
                _ => vec![],
            };
            Ok(Frame::LlmContext {
                header,
                context: Box::new(LlmContextData { messages }),
            })
        }
        frame_type::LLM_MESSAGES | frame_type::LLM_MESSAGES_UPDATE | frame_type::LLM_MESSAGES_APPEND => {
            let messages_json = py_obj_to_json(py, &py_frame.getattr("messages")?)?;
            let messages = match messages_json {
                serde_json::Value::Array(arr) => arr,
                _ => vec![],
            };
            match type_id {
                frame_type::LLM_MESSAGES => Ok(Frame::LlmMessages {
                    header,
                    messages: Box::new(messages),
                }),
                frame_type::LLM_MESSAGES_UPDATE => Ok(Frame::LlmMessagesUpdate {
                    header,
                    messages: Box::new(messages),
                }),
                _ => Ok(Frame::LlmMessagesAppend {
                    header,
                    messages: Box::new(messages),
                }),
            }
        }
        frame_type::LLM_TOOL_CALL => {
            let tool_name: String = py_frame.getattr("tool_name")?.extract()?;
            let arguments: String = py_frame.getattr("arguments")?.extract()?;
            let tool_call_id: Option<String> = py_frame
                .getattr("tool_call_id")
                .ok()
                .and_then(|v| v.extract().ok());
            let run_llm: bool = py_frame
                .getattr("run_llm")
                .ok()
                .and_then(|v| v.extract().ok())
                .unwrap_or(true);
            Ok(Frame::LlmToolCall {
                header,
                data: Box::new(LlmToolCallData {
                    tool_name,
                    arguments,
                    tool_call_id,
                    run_llm,
                }),
            })
        }
        frame_type::LLM_TOOL_RESULT => {
            let tool_name: String = py_frame.getattr("tool_name")?.extract()?;
            let tool_call_id: String = py_frame.getattr("tool_call_id")?.extract()?;
            let result = py_obj_to_json(py, &py_frame.getattr("result")?)?;
            let run_llm: bool = py_frame
                .getattr("run_llm")
                .ok()
                .and_then(|v| v.extract().ok())
                .unwrap_or(true);
            Ok(Frame::LlmToolResult {
                header,
                data: Box::new(LlmToolResultData {
                    tool_name,
                    tool_call_id,
                    result,
                    run_llm,
                }),
            })
        }
        frame_type::LLM_SET_TOOLS => {
            let tools = py_obj_to_json(py, &py_frame.getattr("tools")?)?;
            Ok(Frame::LlmSetTools { header, tools })
        }
        frame_type::LLM_SET_TOOL_CHOICE => {
            let tool_choice = py_obj_to_json(py, &py_frame.getattr("tool_choice")?)?;
            Ok(Frame::LlmSetToolChoice { header, tool_choice })
        }
        frame_type::LLM_CONFIGURE_OUTPUT => {
            let config = py_obj_to_json(py, &py_frame.getattr("config")?)?;
            Ok(Frame::LlmConfigureOutput { header, config })
        }
        frame_type::LLM_UPDATE_SETTINGS => {
            let settings = py_obj_to_json(py, &py_frame.getattr("settings")?)?;
            Ok(Frame::LlmUpdateSettings { header, settings })
        }

        // ── STT ───────────────────────────────────────────────────────
        frame_type::STT_MUTE => {
            let mute: bool = py_frame.getattr("mute")?.extract()?;
            Ok(Frame::SttMute { header, mute })
        }
        frame_type::STT_UPDATE_SETTINGS => {
            let settings = py_obj_to_json(py, &py_frame.getattr("settings")?)?;
            Ok(Frame::SttUpdateSettings { header, settings })
        }
        frame_type::STT_LANGUAGE_UPDATE => {
            let language: String = py_frame.getattr("language")?.extract()?;
            Ok(Frame::SttLanguageUpdate { header, language })
        }

        // ── TTS ───────────────────────────────────────────────────────
        frame_type::TTS_STARTED => Ok(Frame::TtsStarted(header)),
        frame_type::TTS_STOPPED => Ok(Frame::TtsStopped(header)),
        frame_type::TTS_UPDATE_SETTINGS => {
            let settings = py_obj_to_json(py, &py_frame.getattr("settings")?)?;
            Ok(Frame::TtsUpdateSettings { header, settings })
        }
        frame_type::TTS_SPEAK => {
            let text: String = py_frame.getattr("text")?.extract()?;
            Ok(Frame::TtsSpeak { header, text })
        }

        // ── User / Bot events ─────────────────────────────────────────
        frame_type::USER_STARTED_SPEAKING => Ok(Frame::UserStartedSpeaking(header)),
        frame_type::USER_STOPPED_SPEAKING => Ok(Frame::UserStoppedSpeaking(header)),
        frame_type::USER_SPEAKING => {
            let speaking: bool = py_frame.getattr("speaking")
                .ok().and_then(|v| v.extract().ok()).unwrap_or(false);
            Ok(Frame::UserSpeaking { header, speaking })
        }
        frame_type::USER_MUTE_STARTED => Ok(Frame::UserMuteStarted(header)),
        frame_type::USER_MUTE_STOPPED => Ok(Frame::UserMuteStopped(header)),
        frame_type::USER_EMULATE_STARTED => Ok(Frame::EmulateUserStartedSpeaking(header)),
        frame_type::USER_EMULATE_STOPPED => Ok(Frame::EmulateUserStoppedSpeaking(header)),
        frame_type::USER_VAD_STARTED => Ok(Frame::VADUserStartedSpeaking(header)),
        frame_type::USER_VAD_STOPPED => Ok(Frame::VADUserStoppedSpeaking(header)),
        frame_type::BOT_STARTED_SPEAKING => Ok(Frame::BotStartedSpeaking(header)),
        frame_type::BOT_STOPPED_SPEAKING => Ok(Frame::BotStoppedSpeaking(header)),
        frame_type::BOT_SPEAKING => {
            let speaking: bool = py_frame.getattr("speaking")
                .ok().and_then(|v| v.extract().ok()).unwrap_or(false);
            Ok(Frame::BotSpeaking { header, speaking })
        }

        // ── Urgent control ───────────────────────────────────────────
        frame_type::CTRL_PAUSE_URGENT => {
            let processor_name: String = py_frame.getattr("processor_name")?.extract()?;
            Ok(Frame::PauseUrgent { header, processor_name })
        }
        frame_type::CTRL_RESUME_URGENT => {
            let processor_name: String = py_frame.getattr("processor_name")?.extract()?;
            Ok(Frame::ResumeUrgent { header, processor_name })
        }
        frame_type::CTRL_OUTPUT_READY => Ok(Frame::OutputTransportReady(header)),

        // ── Transport messages ───────────────────────────────────────
        frame_type::TRANSPORT_MSG_OUT | frame_type::TRANSPORT_MSG_OUT_URGENT => {
            let msg = py_obj_to_json(py, &py_frame.getattr("message")?)?;
            Ok(Frame::TransportMessageOut { header, message: msg })
        }
        frame_type::TRANSPORT_MSG_IN | frame_type::TRANSPORT_MSG_IN_URGENT => {
            let msg = py_obj_to_json(py, &py_frame.getattr("message")?)?;
            Ok(Frame::TransportMessageIn { header, message: msg })
        }
        frame_type::TRANSPORT_MSG_BIDIR | frame_type::TRANSPORT_MSG_BIDIR_URGENT => {
            let msg = py_obj_to_json(py, &py_frame.getattr("message")?)?;
            Ok(Frame::TransportMessageBidir { header, message: msg })
        }

        // ── Error ─────────────────────────────────────────────────────
        frame_type::ERROR_GENERAL => {
            let message: String = py_frame
                .getattr("error")
                .and_then(|e| e.extract())
                .unwrap_or_default();
            let fatal: bool = py_frame
                .getattr("fatal")
                .ok()
                .and_then(|v| v.extract().ok())
                .unwrap_or(false);
            Ok(Frame::Error {
                header,
                error: ErrorData {
                    message,
                    fatal,
                    exception: None,
                },
            })
        }

        // ── DTMF ─────────────────────────────────────────────────────
        frame_type::DTMF_OUTPUT => {
            let keys: String = py_frame.getattr("keys")?.extract()?;
            Ok(Frame::DtmfOutput { header, keys })
        }
        frame_type::DTMF_INPUT => {
            let keys: String = py_frame.getattr("keys")?.extract()?;
            Ok(Frame::DtmfInput { header, keys })
        }

        // ── Task frames ───────────────────────────────────────────────
        frame_type::TASK_INTERRUPTION => Ok(Frame::InterruptionTask(header)),
        frame_type::TASK_BOT_INTERRUPT => Ok(Frame::BotInterruption(header)),
        frame_type::TASK_CANCEL => Ok(Frame::CancelTask(header)),
        frame_type::TASK_STOP => Ok(Frame::StopTask(header)),
        frame_type::CTRL_END_TASK => Ok(Frame::EndTask(header)),

        // ── Function calls ────────────────────────────────────────────
        frame_type::FUNC_CALL_PROGRESS => {
            let function_name: String = py_frame.getattr("function_name")?.extract()?;
            let tool_call_id: String = py_frame.getattr("tool_call_id")?.extract()?;
            let arguments: String = py_frame.getattr("arguments")?.extract()?;
            Ok(Frame::FunctionCallProgress {
                header,
                data: Box::new(FuncCallProgressData {
                    function_name,
                    tool_call_id,
                    arguments,
                }),
            })
        }
        frame_type::FUNC_CALL_RESULT => {
            let function_name: String = py_frame.getattr("function_name")?.extract()?;
            let tool_call_id: String = py_frame.getattr("tool_call_id")?.extract()?;
            let result = py_obj_to_json(py, &py_frame.getattr("result")?)?;
            let run_llm: bool = py_frame
                .getattr("run_llm")
                .ok()
                .and_then(|v| v.extract().ok())
                .unwrap_or(true);
            Ok(Frame::FunctionCallResult {
                header,
                data: Box::new(FuncCallResultData {
                    function_name,
                    tool_call_id,
                    result,
                    run_llm,
                }),
            })
        }

        frame_type::FUNC_CALLS_STARTED => Ok(Frame::FunctionCallsStarted(header)),
        frame_type::FUNC_CALL_CANCEL => {
            let function_name: String = py_frame.getattr("function_name")?.extract()?;
            Ok(Frame::FunctionCallCancel { header, function_name })
        }

        // ── Vision frames ───────────────────────────────────────────────
        frame_type::VISION_TEXT => {
            let text: String = py_frame.getattr("text")?.extract()?;
            Ok(Frame::VisionText { header, data: TextData { text } })
        }
        frame_type::VISION_RESP_START => Ok(Frame::VisionResponseStart(header)),
        frame_type::VISION_RESP_END => Ok(Frame::VisionResponseEnd(header)),

        // ── Translation ─────────────────────────────────────────────────
        frame_type::TEXT_TRANSLATION => {
            let text: String = py_frame.getattr("text")?.extract()?;
            let language: Option<String> = py_frame.getattr("language")
                .ok().and_then(|v| v.extract().ok());
            Ok(Frame::Translation { header, data: TextData { text }, language })
        }

        // ── Image frames ──────────────────────────────────────────────
        frame_type::IMAGE_OUTPUT => {
            let image = extract_image_data(py, py_frame)?;
            Ok(Frame::ImageOutput {
                header,
                image: Box::new(image),
            })
        }
        frame_type::IMAGE_INPUT => {
            let image = extract_image_data(py, py_frame)?;
            Ok(Frame::ImageInput {
                header,
                image: Box::new(image),
            })
        }
        frame_type::IMAGE_URL => {
            let image = extract_image_data(py, py_frame)?;
            let url: Option<String> = py_frame
                .getattr("url")
                .ok()
                .and_then(|v| v.extract().ok());
            Ok(Frame::ImageUrl {
                header,
                image: Box::new(image),
                url,
            })
        }

        // ── Metrics ──────────────────────────────────────────────────
        frame_type::SYS_METRICS => {
            // MetricsFrame.data is a list of metric dicts — serialize via JSON
            let data_obj = py_frame.getattr("data")?;
            let json_val = py_obj_to_json(py, &data_obj)?;
            let data: Vec<pipecat_core::metrics::MetricsData> =
                serde_json::from_value(json_val).unwrap_or_default();
            Ok(Frame::Metrics { header, data })
        }

        // ── Service frames ───────────────────────────────────────────
        frame_type::SERVICE_METADATA => {
            let service_name: String = py_frame.getattr("service_name")
                .ok().and_then(|v| v.extract().ok()).unwrap_or_default();
            Ok(Frame::ServiceMetadata { header, service_name })
        }
        frame_type::SERVICE_UPDATE => {
            let settings = py_obj_to_json(py, &py_frame.getattr("settings")?)?;
            Ok(Frame::ServiceUpdateSettings { header, settings })
        }
        frame_type::SERVICE_SWITCHER => Ok(Frame::ServiceSwitcher(header)),
        frame_type::SERVICE_SWITCH_MANUAL => {
            let service_name: String = py_frame.getattr("service_name")
                .ok().and_then(|v| v.extract().ok()).unwrap_or_default();
            Ok(Frame::ServiceSwitchManual { header, service_name })
        }

        // ── Filter frames ───────────────────────────────────────────
        frame_type::FILTER_UPDATE => {
            let settings = py_obj_to_json(py, &py_frame.getattr("settings")?)?;
            Ok(Frame::FilterUpdateSettings { header, settings })
        }
        frame_type::FILTER_ENABLE => {
            let enable: bool = py_frame.getattr("enable")
                .ok().and_then(|v| v.extract().ok()).unwrap_or(true);
            Ok(Frame::FilterEnable { header, enable })
        }

        // ── Mixer frames ────────────────────────────────────────────
        frame_type::MIXER_UPDATE => {
            let settings = py_obj_to_json(py, &py_frame.getattr("settings")?)?;
            Ok(Frame::MixerUpdateSettings { header, settings })
        }
        frame_type::MIXER_ENABLE => {
            let enable: bool = py_frame.getattr("enable")
                .ok().and_then(|v| v.extract().ok()).unwrap_or(true);
            Ok(Frame::MixerEnable { header, enable })
        }

        // ── Fallback to opaque ────────────────────────────────────────
        _ => py_frame_to_rust_opaque(py, py_frame, type_id, &FrameTypeRegistry::new()),
    }
}

/// Convert a Tier 2/3 Python frame to `Frame::Custom` via JSON serialization.
fn py_frame_to_rust_opaque(
    py: Python<'_>,
    py_frame: &Bound<'_, PyAny>,
    _type_id: u16,
    _registry: &FrameTypeRegistry,
) -> PyResult<Frame> {
    let header = extract_header(py_frame)?;

    // Get the class name for reconstruction
    let class_name: String = py_frame
        .get_type()
        .name()?
        .to_string();

    // Serialize __dict__ to JSON
    let dict = py_frame.getattr("__dict__")?;
    let data = py_obj_to_json(py, &dict)?;

    Ok(Frame::Custom {
        header,
        data: Box::new(CustomFrameData {
            type_name: class_name,
            data,
        }),
    })
}

// ── Rust → Python ─────────────────────────────────────────────────────────

/// Convert a Rust Frame to a Python frame object.
///
/// For Tier 1 frames, constructs the Python dataclass with explicit kwargs.
/// For Custom frames, looks up the Python class by type_name and deserializes.
pub fn rust_frame_to_py(
    py: Python<'_>,
    frame: Frame,
    _registry: &FrameTypeRegistry,
) -> PyResult<PyObject> {
    rust_frame_to_py_with_sidecar(py, frame, _registry, None)
}

/// Convert a Rust Frame to a Python frame object, with optional sidecar for event reattachment.
pub fn rust_frame_to_py_with_sidecar(
    py: Python<'_>,
    frame: Frame,
    _registry: &FrameTypeRegistry,
    sidecar: Option<&FrameSidecar>,
) -> PyResult<PyObject> {
    // Import pipecat.frames.frames module
    let frames_mod = py.import_bound("pipecat.frames.frames")?;

    match frame {
        // ── Control / Lifecycle ───────────────────────────────────────
        Frame::Start(h) => construct_header_only_frame(py, &frames_mod, "StartFrame", &h),
        Frame::End(h) => construct_header_only_frame(py, &frames_mod, "EndFrame", &h),
        Frame::Stop(h) => construct_header_only_frame(py, &frames_mod, "StopFrame", &h),
        Frame::Cancel(h) => construct_header_only_frame(py, &frames_mod, "CancelFrame", &h),
        Frame::Interruption(h) => {
            let obj = construct_header_only_frame(py, &frames_mod, "InterruptionFrame", &h)?;
            // Reattach asyncio.Event from sidecar if available
            if let Some(sc) = sidecar {
                if let Some((py_event, SidecarKind::InterruptionEvent)) = sc.take(h.id.as_u64()) {
                    let bound = obj.bind(py);
                    let _ = bound.setattr("event", py_event);
                }
            }
            Ok(obj)
        }
        Frame::StartInterruption(h) => {
            construct_header_only_frame(py, &frames_mod, "StartInterruptionFrame", &h)
        }
        Frame::Heartbeat(h) => {
            construct_header_only_frame(py, &frames_mod, "HeartbeatFrame", &h)
        }

        // ── Audio ─────────────────────────────────────────────────────
        Frame::AudioRawInput { header, audio } => {
            construct_audio_frame(py, &frames_mod, "InputAudioRawFrame", &header, &audio)
        }
        Frame::AudioRawOutput { header, audio } => {
            construct_audio_frame(py, &frames_mod, "OutputAudioRawFrame", &header, &audio)
        }
        Frame::AudioTts { header, audio, .. } => {
            construct_audio_frame(py, &frames_mod, "TTSAudioRawFrame", &header, &audio)
        }
        Frame::AudioSpeech { header, audio } => {
            construct_audio_frame(py, &frames_mod, "SpeechOutputAudioRawFrame", &header, &audio)
        }
        Frame::AudioSilence { header, audio } => {
            construct_audio_frame(py, &frames_mod, "SilenceAudioRawFrame", &header, &audio)
        }
        Frame::AudioUser { header, audio } => {
            construct_audio_frame(py, &frames_mod, "UserAudioRawFrame", &header, &audio)
        }

        // ── Text ──────────────────────────────────────────────────────
        Frame::Text { header, data } => {
            construct_text_frame(py, &frames_mod, "TextFrame", &header, &data.text)
        }
        Frame::TextLlm { header, data } => {
            construct_text_frame(py, &frames_mod, "LLMTextFrame", &header, &data.text)
        }
        Frame::TextAggregated { header, data } => {
            construct_text_frame(py, &frames_mod, "AggregatedTextFrame", &header, &data.text)
        }
        Frame::TextTts { header, data } => {
            construct_text_frame(py, &frames_mod, "TTSTextFrame", &header, &data.text)
        }
        Frame::TextInputRaw { header, data } => {
            construct_text_frame(py, &frames_mod, "InputTextRawFrame", &header, &data.text)
        }

        // ── Transcription ─────────────────────────────────────────────
        Frame::Transcription { header, data } => {
            construct_transcription_frame(py, &frames_mod, "TranscriptionFrame", &header, &data)
        }
        Frame::InterimTranscription { header, data } => construct_transcription_frame(
            py,
            &frames_mod,
            "InterimTranscriptionFrame",
            &header,
            &data,
        ),

        // ── LLM ───────────────────────────────────────────────────────
        Frame::LlmResponseStart(h) => {
            construct_header_only_frame(py, &frames_mod, "LLMFullResponseStartFrame", &h)
        }
        Frame::LlmResponseEnd(h) => {
            construct_header_only_frame(py, &frames_mod, "LLMFullResponseEndFrame", &h)
        }
        Frame::LlmRun(h) => construct_header_only_frame(py, &frames_mod, "LLMRunFrame", &h),
        Frame::LlmThoughtStart(h) => {
            construct_header_only_frame(py, &frames_mod, "LLMThoughtStartFrame", &h)
        }
        Frame::LlmThoughtEnd(h) => {
            construct_header_only_frame(py, &frames_mod, "LLMThoughtEndFrame", &h)
        }
        Frame::LlmEnableCaching(h) => {
            construct_header_only_frame(py, &frames_mod, "LLMEnablePromptCachingFrame", &h)
        }
        Frame::LlmCtxSummaryRequest(h) => {
            construct_header_only_frame(py, &frames_mod, "LLMContextSummaryRequestFrame", &h)
        }
        Frame::LlmThoughtText { header, data } => {
            construct_text_frame(py, &frames_mod, "LLMThoughtTextFrame", &header, &data.text)
        }

        // ── User / Bot events ─────────────────────────────────────────
        Frame::UserStartedSpeaking(h) => {
            construct_header_only_frame(py, &frames_mod, "UserStartedSpeakingFrame", &h)
        }
        Frame::UserStoppedSpeaking(h) => {
            construct_header_only_frame(py, &frames_mod, "UserStoppedSpeakingFrame", &h)
        }
        Frame::UserSpeaking { header, speaking } => {
            let cls = frames_mod.getattr("UserSpeakingFrame")?;
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("speaking", speaking)?;
            let obj = cls.call((), Some(&kwargs))?;
            apply_header(py, &obj, &header)?;
            Ok(obj.unbind())
        }
        Frame::UserMuteStarted(h) => {
            construct_header_only_frame(py, &frames_mod, "UserMuteStartedFrame", &h)
        }
        Frame::UserMuteStopped(h) => {
            construct_header_only_frame(py, &frames_mod, "UserMuteStoppedFrame", &h)
        }
        Frame::EmulateUserStartedSpeaking(h) => {
            construct_header_only_frame(py, &frames_mod, "EmulateUserStartedSpeakingFrame", &h)
        }
        Frame::EmulateUserStoppedSpeaking(h) => {
            construct_header_only_frame(py, &frames_mod, "EmulateUserStoppedSpeakingFrame", &h)
        }
        Frame::VADUserStartedSpeaking(h) => {
            construct_header_only_frame(py, &frames_mod, "VADUserStartedSpeakingFrame", &h)
        }
        Frame::VADUserStoppedSpeaking(h) => {
            construct_header_only_frame(py, &frames_mod, "VADUserStoppedSpeakingFrame", &h)
        }
        Frame::BotStartedSpeaking(h) => {
            construct_header_only_frame(py, &frames_mod, "BotStartedSpeakingFrame", &h)
        }
        Frame::BotStoppedSpeaking(h) => {
            construct_header_only_frame(py, &frames_mod, "BotStoppedSpeakingFrame", &h)
        }
        Frame::BotSpeaking { header, speaking } => {
            let cls = frames_mod.getattr("BotSpeakingFrame")?;
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("speaking", speaking)?;
            let obj = cls.call((), Some(&kwargs))?;
            apply_header(py, &obj, &header)?;
            Ok(obj.unbind())
        }

        // ── Urgent control ──────────────────────────────────────────
        Frame::PauseUrgent { header, processor_name } => {
            let cls = frames_mod.getattr("FrameProcessorPauseUrgentFrame")?;
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("processor", py.None())?;
            let obj = cls.call((), Some(&kwargs))?;
            let _ = obj.setattr("_processor_name", &processor_name);
            apply_header(py, &obj, &header)?;
            Ok(obj.unbind())
        }
        Frame::ResumeUrgent { header, processor_name } => {
            let cls = frames_mod.getattr("FrameProcessorResumeUrgentFrame")?;
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("processor", py.None())?;
            let obj = cls.call((), Some(&kwargs))?;
            let _ = obj.setattr("_processor_name", &processor_name);
            apply_header(py, &obj, &header)?;
            Ok(obj.unbind())
        }
        Frame::OutputTransportReady(h) => {
            construct_header_only_frame(py, &frames_mod, "OutputTransportReadyFrame", &h)
        }

        // ── Transport messages ──────────────────────────────────────
        Frame::TransportMessageOut { header, message } => {
            let cls = frames_mod.getattr("OutputTransportMessageFrame")?;
            let kwargs = PyDict::new_bound(py);
            let py_msg = json_to_py(py, &message)?;
            kwargs.set_item("message", py_msg)?;
            let obj = cls.call((), Some(&kwargs))?;
            apply_header(py, &obj, &header)?;
            Ok(obj.unbind())
        }
        Frame::TransportMessageIn { header, message } => {
            let cls = frames_mod.getattr("InputTransportMessageFrame")?;
            let kwargs = PyDict::new_bound(py);
            let py_msg = json_to_py(py, &message)?;
            kwargs.set_item("message", py_msg)?;
            let obj = cls.call((), Some(&kwargs))?;
            apply_header(py, &obj, &header)?;
            Ok(obj.unbind())
        }
        Frame::TransportMessageBidir { header, message } => {
            let cls = frames_mod.getattr("TransportMessageFrame")?;
            let kwargs = PyDict::new_bound(py);
            let py_msg = json_to_py(py, &message)?;
            kwargs.set_item("message", py_msg)?;
            let obj = cls.call((), Some(&kwargs))?;
            apply_header(py, &obj, &header)?;
            Ok(obj.unbind())
        }

        // ── TTS ───────────────────────────────────────────────────────
        Frame::TtsStarted(h) => {
            construct_header_only_frame(py, &frames_mod, "TTSStartedFrame", &h)
        }
        Frame::TtsStopped(h) => {
            construct_header_only_frame(py, &frames_mod, "TTSStoppedFrame", &h)
        }
        Frame::TtsSpeak { header, text } => {
            let cls = frames_mod.getattr("TTSSpeakFrame")?;
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("text", &text)?;
            let obj = cls.call((), Some(&kwargs))?;
            apply_header(py, &obj, &header)?;
            Ok(obj.unbind())
        }
        Frame::TtsUpdateSettings { header, settings } => {
            construct_json_settings_frame(py, &frames_mod, "TTSUpdateSettingsFrame", &header, &settings)
        }

        // ── AudioMix ─────────────────────────────────────────────────
        Frame::AudioMix { header, audio, mix_handle_id } => {
            // AudioMixFrame may not exist in Python; use audio frame with extra field
            let obj = construct_audio_frame(py, &frames_mod, "OutputAudioRawFrame", &header, &audio)?;
            let _ = py.eval_bound("None", None, None)
                .and_then(|_| {
                    let bound = obj.bind(py);
                    bound.setattr("mix_handle_id", &mix_handle_id)
                });
            Ok(obj)
        }

        // ── Task frames ──────────────────────────────────────────────
        Frame::InterruptionTask(h) => {
            construct_header_only_frame(py, &frames_mod, "InterruptionTaskFrame", &h)
        }
        Frame::BotInterruption(h) => {
            construct_header_only_frame(py, &frames_mod, "BotInterruptionFrame", &h)
        }
        Frame::CancelTask(h) => {
            construct_header_only_frame(py, &frames_mod, "CancelTaskFrame", &h)
        }
        Frame::StopTask(h) => {
            construct_header_only_frame(py, &frames_mod, "StopTaskFrame", &h)
        }
        Frame::EndTask(h) => {
            construct_header_only_frame(py, &frames_mod, "EndTaskFrame", &h)
        }

        // ── Pause / Resume ───────────────────────────────────────────
        Frame::Pause { header, processor_name } => {
            // FrameProcessorPauseFrame requires a `processor` object.
            // We only have the name, so pass None and set the name attribute.
            let cls = frames_mod.getattr("FrameProcessorPauseFrame")?;
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("processor", py.None())?;
            let obj = cls.call((), Some(&kwargs))?;
            let _ = obj.setattr("_processor_name", &processor_name);
            apply_header(py, &obj, &header)?;
            Ok(obj.unbind())
        }
        Frame::Resume { header, processor_name } => {
            let cls = frames_mod.getattr("FrameProcessorResumeFrame")?;
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("processor", py.None())?;
            let obj = cls.call((), Some(&kwargs))?;
            let _ = obj.setattr("_processor_name", &processor_name);
            apply_header(py, &obj, &header)?;
            Ok(obj.unbind())
        }

        // ── DTMF ─────────────────────────────────────────────────────
        Frame::DtmfOutput { header, keys } => {
            let dtmf_types = py.import_bound("pipecat.audio.dtmf.types")?;
            let keypad_cls = dtmf_types.getattr("KeypadEntry")?;
            let button = keypad_cls.call1((&keys,))?;
            let cls = frames_mod.getattr("OutputDTMFFrame")?;
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("button", button)?;
            let obj = cls.call((), Some(&kwargs))?;
            apply_header(py, &obj, &header)?;
            Ok(obj.unbind())
        }
        Frame::DtmfInput { header, keys } => {
            let dtmf_types = py.import_bound("pipecat.audio.dtmf.types")?;
            let keypad_cls = dtmf_types.getattr("KeypadEntry")?;
            let button = keypad_cls.call1((&keys,))?;
            let cls = frames_mod.getattr("InputDTMFFrame")?;
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("button", button)?;
            let obj = cls.call((), Some(&kwargs))?;
            apply_header(py, &obj, &header)?;
            Ok(obj.unbind())
        }

        // ── STT control ──────────────────────────────────────────────
        Frame::SttMute { header, mute } => {
            let cls = frames_mod.getattr("STTMuteFrame")?;
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("mute", mute)?;
            let obj = cls.call((), Some(&kwargs))?;
            apply_header(py, &obj, &header)?;
            Ok(obj.unbind())
        }
        Frame::SttUpdateSettings { header, settings } => {
            construct_json_settings_frame(py, &frames_mod, "STTUpdateSettingsFrame", &header, &settings)
        }
        Frame::SttLanguageUpdate { header, language } => {
            // No dedicated Python frame class; create a generic dict
            let dict = PyDict::new_bound(py);
            dict.set_item("__rust_frame_name__", "SttLanguageUpdate")?;
            dict.set_item("language", &language)?;
            dict.set_item("id", header.id.as_u64())?;
            Ok(dict.into_any().unbind())
        }

        // ── LLM (structured) ────────────────────────────────────────
        Frame::LlmMessages { header, messages } => {
            let cls = frames_mod.getattr("LLMMessagesFrame")?;
            let kwargs = PyDict::new_bound(py);
            let py_messages = json_array_to_py_list(py, &messages)?;
            kwargs.set_item("messages", py_messages)?;
            let obj = cls.call((), Some(&kwargs))?;
            apply_header(py, &obj, &header)?;
            Ok(obj.unbind())
        }
        Frame::LlmMessagesUpdate { header, messages } => {
            let cls = frames_mod.getattr("LLMMessagesUpdateFrame")?;
            let kwargs = PyDict::new_bound(py);
            let py_messages = json_array_to_py_list(py, &messages)?;
            kwargs.set_item("messages", py_messages)?;
            let obj = cls.call((), Some(&kwargs))?;
            apply_header(py, &obj, &header)?;
            Ok(obj.unbind())
        }
        Frame::LlmMessagesAppend { header, messages } => {
            let cls = frames_mod.getattr("LLMMessagesAppendFrame")?;
            let kwargs = PyDict::new_bound(py);
            let py_messages = json_array_to_py_list(py, &messages)?;
            kwargs.set_item("messages", py_messages)?;
            let obj = cls.call((), Some(&kwargs))?;
            apply_header(py, &obj, &header)?;
            Ok(obj.unbind())
        }
        Frame::LlmContext { header, context } => {
            // LLMContextFrame expects a Python LLMContext object.
            // Construct one with the messages from our Rust data.
            let ctx_mod = py.import_bound("pipecat.processors.aggregators.llm_context")?;
            let llm_ctx_cls = ctx_mod.getattr("LLMContext")?;
            let kwargs = PyDict::new_bound(py);
            let py_messages = json_array_to_py_list(py, &context.messages)?;
            kwargs.set_item("messages", py_messages)?;
            let llm_ctx = llm_ctx_cls.call((), Some(&kwargs))?;

            let cls = frames_mod.getattr("LLMContextFrame")?;
            let frame_kwargs = PyDict::new_bound(py);
            frame_kwargs.set_item("context", &llm_ctx)?;
            let obj = cls.call((), Some(&frame_kwargs))?;
            apply_header(py, &obj, &header)?;
            Ok(obj.unbind())
        }
        Frame::LlmToolCall { header, data } => {
            // No Python LLMToolCallFrame class — use FunctionCallInProgressFrame
            // as the closest equivalent (function_name maps to tool_name).
            let cls = frames_mod.getattr("FunctionCallInProgressFrame")?;
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("function_name", &data.tool_name)?;
            if let Some(ref id) = data.tool_call_id {
                kwargs.set_item("tool_call_id", id)?;
            } else {
                kwargs.set_item("tool_call_id", "")?;
            }
            kwargs.set_item("arguments", &data.arguments)?;
            let obj = cls.call((), Some(&kwargs))?;
            apply_header(py, &obj, &header)?;
            Ok(obj.unbind())
        }
        Frame::LlmToolResult { header, data } => {
            // No Python LLMToolResultFrame class — use FunctionCallResultFrame
            // as the closest equivalent.
            let cls = frames_mod.getattr("FunctionCallResultFrame")?;
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("function_name", &data.tool_name)?;
            kwargs.set_item("tool_call_id", &data.tool_call_id)?;
            kwargs.set_item("arguments", "")?;
            let py_result = json_to_py(py, &data.result)?;
            kwargs.set_item("result", py_result)?;
            kwargs.set_item("run_llm", data.run_llm)?;
            let obj = cls.call((), Some(&kwargs))?;
            apply_header(py, &obj, &header)?;
            Ok(obj.unbind())
        }
        Frame::LlmSetTools { header, tools } => {
            let cls = frames_mod.getattr("LLMSetToolsFrame")?;
            let kwargs = PyDict::new_bound(py);
            let py_tools = json_to_py(py, &tools)?;
            kwargs.set_item("tools", py_tools)?;
            let obj = cls.call((), Some(&kwargs))?;
            apply_header(py, &obj, &header)?;
            Ok(obj.unbind())
        }
        Frame::LlmSetToolChoice { header, tool_choice } => {
            let cls = frames_mod.getattr("LLMSetToolChoiceFrame")?;
            let kwargs = PyDict::new_bound(py);
            let py_choice = json_to_py(py, &tool_choice)?;
            kwargs.set_item("tool_choice", py_choice)?;
            let obj = cls.call((), Some(&kwargs))?;
            apply_header(py, &obj, &header)?;
            Ok(obj.unbind())
        }
        Frame::LlmConfigureOutput { header, config } => {
            let cls = frames_mod.getattr("LLMConfigureOutputFrame")?;
            let kwargs = PyDict::new_bound(py);
            // config is a JSON value; try to extract skip_tts from it
            if let Some(skip) = config.get("skip_tts").and_then(|v| v.as_bool()) {
                kwargs.set_item("skip_tts", skip)?;
            }
            let obj = cls.call((), Some(&kwargs))?;
            apply_header(py, &obj, &header)?;
            Ok(obj.unbind())
        }
        Frame::LlmCtxSummaryResult { header, summary } => {
            let cls = frames_mod.getattr("LLMContextSummaryResultFrame")?;
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("summary", &summary)?;
            kwargs.set_item("request_id", "")?;
            kwargs.set_item("last_summarized_index", 0)?;
            let obj = cls.call((), Some(&kwargs))?;
            apply_header(py, &obj, &header)?;
            Ok(obj.unbind())
        }
        Frame::LlmUpdateSettings { header, settings } => {
            construct_json_settings_frame(py, &frames_mod, "LLMUpdateSettingsFrame", &header, &settings)
        }

        // ── Function calls ───────────────────────────────────────────
        Frame::FunctionCallProgress { header, data } => {
            let cls = frames_mod.getattr("FunctionCallInProgressFrame")?;
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("function_name", &data.function_name)?;
            kwargs.set_item("tool_call_id", &data.tool_call_id)?;
            kwargs.set_item("arguments", &data.arguments)?;
            let obj = cls.call((), Some(&kwargs))?;
            apply_header(py, &obj, &header)?;
            Ok(obj.unbind())
        }
        Frame::FunctionCallResult { header, data } => {
            let cls = frames_mod.getattr("FunctionCallResultFrame")?;
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("function_name", &data.function_name)?;
            kwargs.set_item("tool_call_id", &data.tool_call_id)?;
            // Python FunctionCallResultFrame requires `arguments` but Rust
            // FuncCallResultData doesn't store it — pass empty string.
            kwargs.set_item("arguments", "")?;
            let py_result = json_to_py(py, &data.result)?;
            kwargs.set_item("result", py_result)?;
            kwargs.set_item("run_llm", data.run_llm)?;
            let obj = cls.call((), Some(&kwargs))?;
            apply_header(py, &obj, &header)?;
            Ok(obj.unbind())
        }

        Frame::FunctionCallsStarted(h) => {
            construct_header_only_frame(py, &frames_mod, "FunctionCallsStartedFrame", &h)
        }
        Frame::FunctionCallCancel { header, function_name } => {
            let cls = frames_mod.getattr("FunctionCallCancelFrame")?;
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("function_name", &function_name)?;
            let obj = cls.call((), Some(&kwargs))?;
            apply_header(py, &obj, &header)?;
            Ok(obj.unbind())
        }

        // ── Vision ──────────────────────────────────────────────────
        Frame::VisionText { header, data } => {
            construct_text_frame(py, &frames_mod, "VisionTextFrame", &header, &data.text)
        }
        Frame::VisionResponseStart(h) => {
            construct_header_only_frame(py, &frames_mod, "VisionFullResponseStartFrame", &h)
        }
        Frame::VisionResponseEnd(h) => {
            construct_header_only_frame(py, &frames_mod, "VisionFullResponseEndFrame", &h)
        }

        // ── Translation ─────────────────────────────────────────────
        Frame::Translation { header, data, language } => {
            let cls = frames_mod.getattr("TranslationFrame")?;
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("text", &data.text)?;
            if let Some(ref lang) = language {
                kwargs.set_item("language", lang)?;
            }
            let obj = cls.call((), Some(&kwargs))?;
            apply_header(py, &obj, &header)?;
            Ok(obj.unbind())
        }

        // ── Image frames ─────────────────────────────────────────────
        Frame::ImageOutput { header, image } => {
            construct_image_frame(py, &frames_mod, "OutputImageRawFrame", &header, &image)
        }
        Frame::ImageInput { header, image } => {
            construct_image_frame(py, &frames_mod, "InputImageRawFrame", &header, &image)
        }
        Frame::ImageUrl { header, image, url } => {
            let obj = construct_image_frame(py, &frames_mod, "URLImageRawFrame", &header, &image)?;
            if let Some(ref u) = url {
                let _ = obj.bind(py).setattr("url", u);
            }
            Ok(obj)
        }

        // ── Metrics ──────────────────────────────────────────────────
        Frame::Metrics { header, data } => {
            // MetricsData is an enum (Ttfb, Processing, LlmUsage) — serialize
            // each variant to JSON via serde, then convert to Python dicts.
            let list = PyList::empty_bound(py);
            for m in &data {
                let json_val = serde_json::to_value(m).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Metrics serialization error: {e}"
                    ))
                })?;
                let py_val = json_to_py(py, &json_val)?;
                list.append(py_val)?;
            }
            let cls = frames_mod.getattr("MetricsFrame")?;
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("data", list)?;
            let obj = cls.call((), Some(&kwargs))?;
            apply_header(py, &obj, &header)?;
            Ok(obj.unbind())
        }

        // ── Error ─────────────────────────────────────────────────────
        Frame::Error { header, error } => {
            let cls = frames_mod.getattr("ErrorFrame")?;
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("error", &error.message)?;
            kwargs.set_item("fatal", error.fatal)?;
            let obj = cls.call((), Some(&kwargs))?;
            apply_header(py, &obj, &header)?;
            Ok(obj.unbind())
        }

        // ── Service ──────────────────────────────────────────────────
        Frame::ServiceMetadata { header, service_name } => {
            let cls = frames_mod.getattr("ServiceMetadataFrame")?;
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("service_name", &service_name)?;
            let obj = cls.call((), Some(&kwargs))?;
            apply_header(py, &obj, &header)?;
            Ok(obj.unbind())
        }
        Frame::ServiceUpdateSettings { header, settings } => {
            construct_json_settings_frame(py, &frames_mod, "ServiceUpdateSettingsFrame", &header, &settings)
        }
        Frame::ServiceSwitcher(h) => {
            construct_header_only_frame(py, &frames_mod, "ServiceSwitcherFrame", &h)
        }
        Frame::ServiceSwitchManual { header, service_name } => {
            let cls = frames_mod.getattr("ManuallySwitchServiceFrame")?;
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("service_name", &service_name)?;
            let obj = cls.call((), Some(&kwargs))?;
            apply_header(py, &obj, &header)?;
            Ok(obj.unbind())
        }

        // ── Filter ──────────────────────────────────────────────────
        Frame::FilterUpdateSettings { header, settings } => {
            construct_json_settings_frame(py, &frames_mod, "FilterUpdateSettingsFrame", &header, &settings)
        }
        Frame::FilterEnable { header, enable } => {
            let cls = frames_mod.getattr("FilterEnableFrame")?;
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("enable", enable)?;
            let obj = cls.call((), Some(&kwargs))?;
            apply_header(py, &obj, &header)?;
            Ok(obj.unbind())
        }

        // ── Mixer ───────────────────────────────────────────────────
        Frame::MixerUpdateSettings { header, settings } => {
            construct_json_settings_frame(py, &frames_mod, "MixerUpdateSettingsFrame", &header, &settings)
        }
        Frame::MixerEnable { header, enable } => {
            let cls = frames_mod.getattr("MixerEnableFrame")?;
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("enable", enable)?;
            let obj = cls.call((), Some(&kwargs))?;
            apply_header(py, &obj, &header)?;
            Ok(obj.unbind())
        }

        // ── Custom (Tier 2/3) ─────────────────────────────────────────
        Frame::Custom { header, data } => {
            // Try to find the Python class by name
            match frames_mod.getattr(data.type_name.as_str()) {
                Ok(cls) => {
                    // Reconstruct from JSON data
                    let obj = cls.call0()?;
                    if let serde_json::Value::Object(map) = &data.data {
                        for (k, v) in map {
                            if k == "id" || k == "pts" || k == "_name" || k == "_metadata" {
                                continue;
                            }
                            let py_val = json_to_py(py, v)?;
                            let _ = obj.setattr(k.as_str(), py_val);
                        }
                    }
                    apply_header(py, &obj, &header)?;
                    Ok(obj.unbind())
                }
                Err(_) => {
                    // Unknown frame type — create a generic dict-like object
                    let dict = PyDict::new_bound(py);
                    dict.set_item("__type_name__", &data.type_name)?;
                    dict.set_item("__frame_id__", header.id.as_u64())?;
                    if let serde_json::Value::Object(map) = &data.data {
                        for (k, v) in map {
                            let py_val = json_to_py(py, v)?;
                            dict.set_item(k.as_str(), py_val)?;
                        }
                    }
                    Ok(dict.into_any().unbind())
                }
            }
        }
    }
}

// ── Header helpers ────────────────────────────────────────────────────────

/// Extract a FrameHeader from a Python frame object.
///
/// Preserves the Python frame's `id` — does NOT generate a new one.
/// This ensures frame identity is maintained across Python↔Rust round-trips.
fn extract_header(py_frame: &Bound<'_, PyAny>) -> PyResult<FrameHeader> {
    let mut header = FrameHeader::new();

    // CRITICAL: Preserve the Python frame's ID instead of using the
    // auto-generated one from FrameHeader::new(). Without this, every
    // frame crossing the bridge gets a new ID, breaking observers,
    // deduplication, and any logic that tracks frames by ID.
    if let Ok(py_id) = py_frame.getattr("id") {
        if let Ok(id_val) = py_id.extract::<u64>() {
            header.id = pipecat_core::frame_id::FrameId::from_raw(id_val);
        }
    }

    // Extract pts if present (Python uses nanoseconds, Rust uses microseconds)
    if let Ok(pts_obj) = py_frame.getattr("pts") {
        if let Ok(pts_ns) = pts_obj.extract::<u64>() {
            header.pts = Some(pts_ns / 1000); // ns → µs
        } else if !pts_obj.is_none() {
            if let Ok(Some(pts_ns)) = pts_obj.extract::<Option<u64>>() {
                header.pts = Some(pts_ns / 1000);
            }
        }
    }

    // Extract broadcast_sibling_id
    if let Ok(bsi) = py_frame.getattr("broadcast_sibling_id") {
        if let Ok(Some(id)) = bsi.extract::<Option<u64>>() {
            header.broadcast_sibling_id = Some(pipecat_core::frame_id::FrameId::from_raw(id));
        }
    }

    // Extract transport_source
    if let Ok(src) = py_frame.getattr("transport_source") {
        if let Ok(Some(s)) = src.extract::<Option<String>>() {
            header.set_transport_source(s);
        }
    }

    // Extract transport_destination
    if let Ok(dst) = py_frame.getattr("transport_destination") {
        if let Ok(Some(s)) = dst.extract::<Option<String>>() {
            header.set_transport_destination(s);
        }
    }

    // Extract _metadata if present (Python Dict[str, Any] → Rust HashMap<String, Value>)
    if let Ok(meta_obj) = py_frame.getattr("_metadata") {
        if !meta_obj.is_none() {
            let py = py_frame.py();
            if let Ok(serde_json::Value::Object(map)) = py_obj_to_json(py, &meta_obj) {
                let meta = header.metadata_mut();
                for (k, v) in map {
                    meta.insert(k, v);
                }
            }
        }
    }

    Ok(header)
}

/// Apply a FrameHeader's values back to a Python frame object.
///
/// Writes the Rust header's ID back to the Python frame so that frame
/// identity is preserved across Rust→Python conversion. Also restores
/// pts, broadcast_sibling_id, transport routing, and metadata.
fn apply_header(
    py: Python<'_>,
    py_frame: &Bound<'_, PyAny>,
    header: &FrameHeader,
) -> PyResult<()> {
    // CRITICAL: Write the Rust header ID back to the Python frame.
    // This preserves frame identity across the Rust→Python bridge.
    let _ = py_frame.setattr("id", header.id.as_u64());

    // Set pts (µs → ns)
    if let Some(pts_us) = header.pts {
        let _ = py_frame.setattr("pts", pts_us * 1000);
    }

    // Set broadcast_sibling_id
    if let Some(bsi) = header.broadcast_sibling_id {
        let _ = py_frame.setattr("broadcast_sibling_id", bsi.as_u64());
    }

    // Set transport_source
    if let Some(src) = header.transport_source() {
        let _ = py_frame.setattr("transport_source", src);
    }

    // Set transport_destination
    if let Some(dst) = header.transport_destination() {
        let _ = py_frame.setattr("transport_destination", dst);
    }

    // Restore metadata if present
    if header.has_metadata() {
        let meta = header.metadata();
        let py_dict = pyo3::types::PyDict::new_bound(py);
        for (k, v) in meta {
            let py_val = json_to_py(py, v)?;
            py_dict.set_item(k, py_val)?;
        }
        let _ = py_frame.setattr("_metadata", py_dict);
    }

    Ok(())
}

// ── Audio helpers ─────────────────────────────────────────────────────────

/// Extract AudioData from a Python audio frame.
fn extract_audio_data(_py: Python<'_>, py_frame: &Bound<'_, PyAny>) -> PyResult<AudioData> {
    let audio_obj = py_frame.getattr("audio")?;
    let audio_bytes: Vec<u8> = audio_obj.extract()?;
    let sample_rate: u32 = py_frame.getattr("sample_rate")?.extract()?;
    let num_channels: u16 = py_frame
        .getattr("num_channels")
        .ok()
        .and_then(|v| v.extract().ok())
        .unwrap_or(1);

    Ok(AudioData {
        audio: Bytes::from(audio_bytes),
        sample_rate,
        num_channels,
    })
}

/// Extract ImageData from a Python image frame.
fn extract_image_data(_py: Python<'_>, py_frame: &Bound<'_, PyAny>) -> PyResult<ImageData> {
    let image_obj = py_frame.getattr("image")?;
    let image_bytes: Vec<u8> = image_obj.extract()?;
    let size: (u32, u32) = py_frame.getattr("size")?.extract()?;
    let format: Option<String> = py_frame
        .getattr("format")
        .ok()
        .and_then(|v| v.extract().ok());

    Ok(ImageData {
        image: Bytes::from(image_bytes),
        size,
        format,
    })
}

/// Construct a Python audio frame.
fn construct_audio_frame(
    py: Python<'_>,
    frames_mod: &Bound<'_, PyAny>,
    class_name: &str,
    header: &FrameHeader,
    audio: &AudioData,
) -> PyResult<PyObject> {
    let cls = frames_mod.getattr(class_name)?;
    let kwargs = PyDict::new_bound(py);
    kwargs.set_item("audio", PyBytes::new_bound(py, &audio.audio))?;
    kwargs.set_item("sample_rate", audio.sample_rate)?;
    kwargs.set_item("num_channels", audio.num_channels)?;
    let obj = cls.call((), Some(&kwargs))?;
    apply_header(py, &obj, header)?;
    Ok(obj.unbind())
}

// ── Text helpers ──────────────────────────────────────────────────────────

/// Construct a Python text frame.
fn construct_text_frame(
    py: Python<'_>,
    frames_mod: &Bound<'_, PyAny>,
    class_name: &str,
    header: &FrameHeader,
    text: &str,
) -> PyResult<PyObject> {
    let cls = frames_mod.getattr(class_name)?;
    let kwargs = PyDict::new_bound(py);
    kwargs.set_item("text", text)?;
    let obj = cls.call((), Some(&kwargs))?;
    apply_header(py, &obj, header)?;
    Ok(obj.unbind())
}

/// Construct a header-only frame (no payload fields).
fn construct_header_only_frame(
    py: Python<'_>,
    frames_mod: &Bound<'_, PyAny>,
    class_name: &str,
    header: &FrameHeader,
) -> PyResult<PyObject> {
    let cls = frames_mod.getattr(class_name)?;
    let obj = cls.call0()?;
    apply_header(py, &obj, header)?;
    Ok(obj.unbind())
}

// ── Transcription helpers ─────────────────────────────────────────────────

fn extract_transcription_data(py_frame: &Bound<'_, PyAny>) -> PyResult<TranscriptionData> {
    let text: String = py_frame.getattr("text")?.extract()?;
    let user_id: Option<String> = py_frame
        .getattr("user_id")
        .ok()
        .and_then(|v| v.extract().ok());
    let timestamp: Option<String> = py_frame
        .getattr("timestamp")
        .ok()
        .and_then(|v| v.extract().ok());
    let language: Option<String> = py_frame
        .getattr("language")
        .ok()
        .and_then(|v| v.extract().ok());

    Ok(TranscriptionData {
        text,
        user_id,
        timestamp,
        language,
    })
}

fn construct_transcription_frame(
    py: Python<'_>,
    frames_mod: &Bound<'_, PyAny>,
    class_name: &str,
    header: &FrameHeader,
    data: &TranscriptionData,
) -> PyResult<PyObject> {
    let cls = frames_mod.getattr(class_name)?;
    let kwargs = PyDict::new_bound(py);
    kwargs.set_item("text", &data.text)?;
    if let Some(ref uid) = data.user_id {
        kwargs.set_item("user_id", uid)?;
    }
    if let Some(ref ts) = data.timestamp {
        kwargs.set_item("timestamp", ts)?;
    }
    if let Some(ref lang) = data.language {
        kwargs.set_item("language", lang)?;
    }
    let obj = cls.call((), Some(&kwargs))?;
    apply_header(py, &obj, header)?;
    Ok(obj.unbind())
}

// ── JSON conversion helpers ───────────────────────────────────────────────

/// Convert a Python object to a serde_json::Value.
fn py_obj_to_json(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
    if obj.is_none() {
        return Ok(serde_json::Value::Null);
    }

    // Use Python's json module for reliable serialization
    let json_mod = py.import_bound("json")?;
    let json_str: String = json_mod
        .call_method1("dumps", (obj,))?
        .extract()?;

    serde_json::from_str(&json_str)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("JSON parse error: {e}")))
}

/// Convert a Vec<serde_json::Value> to a Python list.
fn json_array_to_py_list(
    py: Python<'_>,
    values: &[serde_json::Value],
) -> PyResult<PyObject> {
    let list = PyList::empty_bound(py);
    for item in values {
        list.append(json_to_py(py, item)?)?;
    }
    Ok(list.into_any().unbind())
}

/// Construct a Python settings frame (settings as a JSON dict).
fn construct_json_settings_frame(
    py: Python<'_>,
    frames_mod: &Bound<'_, PyAny>,
    class_name: &str,
    header: &FrameHeader,
    settings: &serde_json::Value,
) -> PyResult<PyObject> {
    let cls = frames_mod.getattr(class_name)?;
    let kwargs = PyDict::new_bound(py);
    let py_settings = json_to_py(py, settings)?;
    kwargs.set_item("settings", py_settings)?;
    let obj = cls.call((), Some(&kwargs))?;
    apply_header(py, &obj, header)?;
    Ok(obj.unbind())
}

/// Construct a Python image frame from ImageData.
fn construct_image_frame(
    py: Python<'_>,
    frames_mod: &Bound<'_, PyAny>,
    class_name: &str,
    header: &FrameHeader,
    image: &ImageData,
) -> PyResult<PyObject> {
    let cls = frames_mod.getattr(class_name)?;
    let kwargs = PyDict::new_bound(py);
    kwargs.set_item("image", PyBytes::new_bound(py, &image.image))?;
    kwargs.set_item("size", (image.size.0, image.size.1))?;
    if let Some(ref fmt) = image.format {
        kwargs.set_item("format", fmt)?;
    }
    let obj = cls.call((), Some(&kwargs))?;
    apply_header(py, &obj, header)?;
    Ok(obj.unbind())
}

/// Convert a serde_json::Value to a Python object.
fn json_to_py(py: Python<'_>, value: &serde_json::Value) -> PyResult<PyObject> {
    match value {
        serde_json::Value::Null => Ok(py.None()),
        serde_json::Value::Bool(b) => Ok(b.into_py(py)),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_py(py))
            } else if let Some(f) = n.as_f64() {
                Ok(f.into_py(py))
            } else {
                Ok(py.None())
            }
        }
        serde_json::Value::String(s) => Ok(s.into_py(py)),
        serde_json::Value::Array(arr) => {
            let list = PyList::empty_bound(py);
            for item in arr {
                list.append(json_to_py(py, item)?)?;
            }
            Ok(list.into_any().unbind())
        }
        serde_json::Value::Object(map) => {
            let dict = PyDict::new_bound(py);
            for (k, v) in map {
                dict.set_item(k, json_to_py(py, v)?)?;
            }
            Ok(dict.into_any().unbind())
        }
    }
}
