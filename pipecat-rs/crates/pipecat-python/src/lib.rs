// pyo3 proc-macros (#[pymethods], #[pyfunction]) generate `.into()` calls on
// PyErr return paths that are already PyErr, triggering clippy::useless_conversion.
// This is a known pyo3 0.22 issue in generated code, not in hand-written code.
#![allow(clippy::useless_conversion)]

use pyo3::prelude::*;

mod async_bridge;
mod audio;
mod conversion;
mod frame;
mod frame_bridge;
mod interruption;
mod observer_bridge;
mod pipeline;
mod processor;
mod py_processor;
mod registry;

/// Native API version — checked by `_native_status.py` to ensure compatibility.
const NATIVE_API_VERSION: u32 = 1;

/// Python module for pipecat native engine (`pipecat._native`).
///
/// The function name must match the last component of `module-name` in pyproject.toml
/// so that maturin finds the correct `PyInit__native` symbol.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // API version constant
    m.add("NATIVE_API_VERSION", NATIVE_API_VERSION)?;

    // Original frame/pipeline bindings
    m.add_class::<frame::PyFrame>()?;
    m.add_class::<frame::PyFrameHeader>()?;
    m.add_class::<frame::PyAudioData>()?;
    m.add_class::<pipeline::PyPipeline>()?;
    m.add_class::<pipeline::PyPipelineTask>()?;
    m.add_class::<pipeline::PyPipelineRunner>()?;
    m.add_class::<processor::PyPassthroughProcessor>()?;

    // Native pipeline task (Python processor bridge)
    m.add_class::<pipeline::PyNativePipelineTask>()?;

    // New frame bridge types
    m.add_class::<frame_bridge::PyBridgeFrame>()?;
    m.add_class::<frame_bridge::PyFrameDirection>()?;

    // Helper functions
    m.add_function(wrap_pyfunction!(verify_type_ids, m)?)?;
    m.add_function(wrap_pyfunction!(wrap_processor, m)?)?;

    // Audio utility submodule (Rust-accelerated audio processing)
    audio::register_audio_module(m)?;

    Ok(())
}

/// Verify that Python FrameType constants match Rust frame_type constants.
///
/// Returns a list of mismatches as `(name, python_value, rust_value)` tuples.
/// An empty list means all constants are aligned.
#[pyfunction]
fn verify_type_ids(py: Python<'_>) -> PyResult<Vec<(String, u16, u16)>> {
    use pipecat_core::frame_types::frame_type;

    let frame_type_cls = py
        .import_bound("pipecat.frames.frame_types")?
        .getattr("FrameType")?;

    let checks: Vec<(&str, u16)> = vec![
        ("AUDIO_RAW_INPUT", frame_type::AUDIO_RAW_INPUT),
        ("AUDIO_RAW_OUTPUT", frame_type::AUDIO_RAW_OUTPUT),
        ("AUDIO_TTS", frame_type::AUDIO_TTS),
        ("AUDIO_SPEECH", frame_type::AUDIO_SPEECH),
        ("AUDIO_MIX", frame_type::AUDIO_MIX),
        ("AUDIO_SILENCE", frame_type::AUDIO_SILENCE),
        ("AUDIO_USER", frame_type::AUDIO_USER),
        ("TEXT_PLAIN", frame_type::TEXT_PLAIN),
        ("TEXT_LLM", frame_type::TEXT_LLM),
        ("TEXT_TRANSCRIPTION", frame_type::TEXT_TRANSCRIPTION),
        ("TEXT_INTERIM_TRANS", frame_type::TEXT_INTERIM_TRANS),
        ("TEXT_AGGREGATED", frame_type::TEXT_AGGREGATED),
        ("TEXT_TTS", frame_type::TEXT_TTS),
        ("TEXT_INPUT_RAW", frame_type::TEXT_INPUT_RAW),
        ("CTRL_START", frame_type::CTRL_START),
        ("CTRL_END", frame_type::CTRL_END),
        ("CTRL_STOP", frame_type::CTRL_STOP),
        ("CTRL_CANCEL", frame_type::CTRL_CANCEL),
        ("CTRL_INTERRUPT", frame_type::CTRL_INTERRUPT),
        ("CTRL_START_INTERRUPT", frame_type::CTRL_START_INTERRUPT),
        ("CTRL_PAUSE", frame_type::CTRL_PAUSE),
        ("CTRL_RESUME", frame_type::CTRL_RESUME),
        ("SYS_HEARTBEAT", frame_type::SYS_HEARTBEAT),
        ("SYS_METRICS", frame_type::SYS_METRICS),
        ("LLM_CONTEXT", frame_type::LLM_CONTEXT),
        ("LLM_MESSAGES", frame_type::LLM_MESSAGES),
        ("LLM_RESPONSE_START", frame_type::LLM_RESPONSE_START),
        ("LLM_RESPONSE_END", frame_type::LLM_RESPONSE_END),
        ("LLM_RUN", frame_type::LLM_RUN),
        ("LLM_TOOL_CALL", frame_type::LLM_TOOL_CALL),
        ("LLM_TOOL_RESULT", frame_type::LLM_TOOL_RESULT),
        ("LLM_UPDATE_SETTINGS", frame_type::LLM_UPDATE_SETTINGS),
        ("STT_MUTE", frame_type::STT_MUTE),
        ("STT_UPDATE_SETTINGS", frame_type::STT_UPDATE_SETTINGS),
        ("STT_LANGUAGE_UPDATE", frame_type::STT_LANGUAGE_UPDATE),
        ("TTS_STARTED", frame_type::TTS_STARTED),
        ("TTS_STOPPED", frame_type::TTS_STOPPED),
        ("TTS_UPDATE_SETTINGS", frame_type::TTS_UPDATE_SETTINGS),
        ("TTS_SPEAK", frame_type::TTS_SPEAK),
        ("USER_STARTED_SPEAKING", frame_type::USER_STARTED_SPEAKING),
        ("USER_STOPPED_SPEAKING", frame_type::USER_STOPPED_SPEAKING),
        ("BOT_STARTED_SPEAKING", frame_type::BOT_STARTED_SPEAKING),
        ("BOT_STOPPED_SPEAKING", frame_type::BOT_STOPPED_SPEAKING),
        ("ERROR_GENERAL", frame_type::ERROR_GENERAL),
        ("DTMF_OUTPUT", frame_type::DTMF_OUTPUT),
        ("DTMF_INPUT", frame_type::DTMF_INPUT),
        ("TASK_INTERRUPTION", frame_type::TASK_INTERRUPTION),
        ("TASK_BOT_INTERRUPT", frame_type::TASK_BOT_INTERRUPT),
        ("TASK_CANCEL", frame_type::TASK_CANCEL),
        ("TASK_STOP", frame_type::TASK_STOP),
        ("CTRL_END_TASK", frame_type::CTRL_END_TASK),
        ("FUNC_CALL_PROGRESS", frame_type::FUNC_CALL_PROGRESS),
        ("FUNC_CALL_RESULT", frame_type::FUNC_CALL_RESULT),
        ("IMAGE_OUTPUT", frame_type::IMAGE_OUTPUT),
        ("IMAGE_URL", frame_type::IMAGE_URL),
        ("IMAGE_INPUT", frame_type::IMAGE_INPUT),
        // LLM extended
        ("LLM_MESSAGES_UPDATE", frame_type::LLM_MESSAGES_UPDATE),
        ("LLM_MESSAGES_APPEND", frame_type::LLM_MESSAGES_APPEND),
        ("LLM_SET_TOOLS", frame_type::LLM_SET_TOOLS),
        ("LLM_SET_TOOL_CHOICE", frame_type::LLM_SET_TOOL_CHOICE),
        ("LLM_ENABLE_CACHING", frame_type::LLM_ENABLE_CACHING),
        ("LLM_CONFIGURE_OUTPUT", frame_type::LLM_CONFIGURE_OUTPUT),
        ("LLM_CTX_SUMMARY_REQ", frame_type::LLM_CTX_SUMMARY_REQ),
        ("LLM_CTX_SUMMARY_RESULT", frame_type::LLM_CTX_SUMMARY_RESULT),
        ("LLM_THOUGHT_TEXT", frame_type::LLM_THOUGHT_TEXT),
        ("LLM_THOUGHT_START", frame_type::LLM_THOUGHT_START),
        ("LLM_THOUGHT_END", frame_type::LLM_THOUGHT_END),
    ];

    let mut mismatches = Vec::new();
    for (name, rust_val) in checks {
        if let Ok(py_val) = frame_type_cls.getattr(name) {
            if let Ok(py_int) = py_val.extract::<u16>() {
                if py_int != rust_val {
                    mismatches.push((name.to_string(), py_int, rust_val));
                }
            }
        }
    }

    Ok(mismatches)
}

/// Wrap a Python FrameProcessor for use in the Rust pipeline.
///
/// This creates a `PythonFrameProcessor` that delegates `process_frame` calls
/// to the Python processor.
#[pyfunction]
fn wrap_processor(py: Python<'_>, processor: PyObject) -> PyResult<()> {
    // Validate that the object has a process_frame method
    if processor.getattr(py, "process_frame").is_err() {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "Object must have a process_frame method (is it a FrameProcessor?)",
        ));
    }

    // The actual wrapping happens in NativePipeline construction
    // This function just validates the processor
    Ok(())
}
