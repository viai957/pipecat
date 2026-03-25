//! PyFrame: opaque handle to a Rust Frame with typed Python accessors.
//!
//! Frame data stays in Rust memory. Only fields actually accessed by Python
//! code are extracted across the FFI boundary. For `isinstance()` compatibility,
//! `as_python_frame()` lazily materializes the full Python dataclass and caches it.

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use pipecat_core::{AudioData, Frame, FrameDirection};

use crate::conversion;
use crate::registry::FrameTypeRegistry;

/// Opaque handle to a Rust Frame, exposed to Python with typed accessors.
///
/// The frame lives in Rust memory. Python only pays for fields it actually reads.
/// For full `isinstance()` compatibility, call `as_python_frame()` which lazily
/// materializes the complete Python dataclass (expensive, cached).
#[pyclass(name = "RustFrame")]
pub struct PyBridgeFrame {
    /// The Rust frame. `Option` allows `take()` when transferring ownership.
    inner: Option<Frame>,
    /// Lazily-materialized Python dataclass (cached after first call).
    py_frame_cache: Option<PyObject>,
}

impl PyBridgeFrame {
    /// Wrap a Rust Frame for Python access.
    pub fn new(frame: Frame) -> Self {
        Self {
            inner: Some(frame),
            py_frame_cache: None,
        }
    }

    /// Take ownership of the inner frame (consuming this bridge).
    pub fn take_frame(&mut self) -> Option<Frame> {
        self.inner.take()
    }

    /// Borrow the inner frame.
    pub fn frame_ref(&self) -> Option<&Frame> {
        self.inner.as_ref()
    }
}

#[pymethods]
impl PyBridgeFrame {
    /// The 16-bit type ID for O(1) dispatch.
    #[getter]
    fn type_id(&self) -> u16 {
        self.inner
            .as_ref()
            .map(|f| f.type_id())
            .unwrap_or(0)
    }

    /// Human-readable frame variant name.
    #[getter]
    fn name(&self) -> &str {
        self.inner
            .as_ref()
            .map(|f| f.name())
            .unwrap_or("Consumed")
    }

    /// Unique frame ID.
    #[getter]
    fn id(&self) -> u64 {
        self.inner
            .as_ref()
            .map(|f| f.header().id.as_u64())
            .unwrap_or(0)
    }

    /// Presentation timestamp in microseconds.
    #[getter]
    fn pts(&self) -> Option<u64> {
        self.inner.as_ref().and_then(|f| f.header().pts)
    }

    /// Broadcast sibling ID (for parallel pipeline fan-out).
    #[getter]
    fn broadcast_sibling_id(&self) -> Option<u64> {
        self.inner
            .as_ref()
            .and_then(|f| f.header().broadcast_sibling_id.map(|id| id.as_u64()))
    }

    /// Extract text content (only for text-carrying frames).
    fn text(&self) -> Option<String> {
        match self.inner.as_ref()? {
            Frame::Text { data, .. }
            | Frame::TextLlm { data, .. }
            | Frame::TextAggregated { data, .. }
            | Frame::TextTts { data, .. }
            | Frame::TextInputRaw { data, .. }
            | Frame::LlmThoughtText { data, .. } => Some(data.text.clone()),
            Frame::Transcription { data, .. } | Frame::InterimTranscription { data, .. } => {
                Some(data.text.clone())
            }
            Frame::TtsSpeak { text, .. }
            | Frame::SttLanguageUpdate { language: text, .. }
            | Frame::LlmCtxSummaryResult { summary: text, .. } => Some(text.clone()),
            _ => None,
        }
    }

    /// Extract raw audio bytes (only for audio frames).
    fn audio_bytes<'py>(&self, py: Python<'py>) -> Option<PyObject> {
        let audio = self.audio_data()?;
        Some(
            PyBytes::new_bound(py, &audio.audio)
                .into_any()
                .unbind(),
        )
    }

    /// Sample rate of audio frame.
    fn sample_rate(&self) -> Option<u32> {
        self.audio_data().map(|a| a.sample_rate)
    }

    /// Number of audio channels.
    fn num_channels(&self) -> Option<u16> {
        self.audio_data().map(|a| a.num_channels)
    }

    /// Frame metadata as a Python dict (only allocated when accessed).
    fn metadata<'py>(&self, py: Python<'py>) -> PyResult<Option<PyObject>> {
        let frame = match self.inner.as_ref() {
            Some(f) => f,
            None => return Ok(None),
        };
        let header = frame.header();
        if !header.has_metadata() {
            return Ok(None);
        }
        let dict = PyDict::new_bound(py);
        for (k, v) in header.metadata() {
            let json_str = v.to_string();
            dict.set_item(k, json_str)?;
        }
        Ok(Some(dict.into_any().unbind()))
    }

    /// Materialize the full Python dataclass frame (expensive, cached).
    ///
    /// This creates the complete Python frame object with all fields,
    /// enabling `isinstance()` checks in user code.
    fn as_python_frame<'py>(&mut self, py: Python<'py>) -> PyResult<PyObject> {
        if let Some(ref cached) = self.py_frame_cache {
            return Ok(cached.clone_ref(py));
        }
        let frame = self
            .inner
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("frame already consumed"))?;

        let registry = FrameTypeRegistry::new();
        let py_frame = conversion::rust_frame_to_py(py, frame.clone(), &registry)?;
        self.py_frame_cache = Some(py_frame.clone_ref(py));
        Ok(py_frame)
    }

    fn __repr__(&self) -> String {
        match self.inner.as_ref() {
            Some(f) => format!("RustFrame({}, id={})", f.name(), f.header().id.as_u64()),
            None => "RustFrame(<consumed>)".to_string(),
        }
    }
}

impl PyBridgeFrame {
    /// Helper to extract AudioData from any audio frame variant.
    fn audio_data(&self) -> Option<&AudioData> {
        match self.inner.as_ref()? {
            Frame::AudioRawInput { audio, .. }
            | Frame::AudioRawOutput { audio, .. }
            | Frame::AudioTts { audio, .. }
            | Frame::AudioSpeech { audio, .. }
            | Frame::AudioMix { audio, .. }
            | Frame::AudioSilence { audio, .. }
            | Frame::AudioUser { audio, .. } => Some(audio),
            _ => None,
        }
    }
}

/// Python wrapper for FrameDirection.
#[pyclass(name = "FrameDirection")]
#[derive(Clone, Copy)]
pub struct PyFrameDirection {
    pub inner: FrameDirection,
}

#[pymethods]
impl PyFrameDirection {
    /// Downstream direction constant.
    #[classattr]
    const DOWNSTREAM: u8 = 0;
    /// Upstream direction constant.
    #[classattr]
    const UPSTREAM: u8 = 1;

    #[new]
    fn new(value: u8) -> PyResult<Self> {
        match value {
            0 => Ok(Self {
                inner: FrameDirection::Downstream,
            }),
            1 => Ok(Self {
                inner: FrameDirection::Upstream,
            }),
            _ => Err(pyo3::exceptions::PyValueError::new_err(
                "direction must be 0 (downstream) or 1 (upstream)",
            )),
        }
    }

    fn __repr__(&self) -> &str {
        match self.inner {
            FrameDirection::Downstream => "FrameDirection.DOWNSTREAM",
            FrameDirection::Upstream => "FrameDirection.UPSTREAM",
        }
    }
}

/// Convert Rust FrameDirection to Python int.
pub fn rust_direction_to_py(dir: FrameDirection) -> u8 {
    match dir {
        FrameDirection::Downstream => 0,
        FrameDirection::Upstream => 1,
    }
}
