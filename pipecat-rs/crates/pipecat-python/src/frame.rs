use bytes::Bytes;
use pyo3::prelude::*;

use pipecat_core::{AudioData, Frame, FrameHeader, TextData};

/// Python wrapper for FrameHeader
#[pyclass(name = "FrameHeader")]
#[derive(Clone)]
pub struct PyFrameHeader {
    pub(crate) inner: FrameHeader,
}

#[pymethods]
impl PyFrameHeader {
    #[new]
    fn new() -> Self {
        Self {
            inner: FrameHeader::new(),
        }
    }

    #[getter]
    fn id(&self) -> u64 {
        self.inner.id.as_u64()
    }

    #[getter]
    fn pts(&self) -> Option<u64> {
        self.inner.pts
    }

    #[setter]
    fn set_pts(&mut self, pts: Option<u64>) {
        self.inner.pts = pts;
    }
}

/// Python wrapper for AudioData
#[pyclass(name = "AudioData")]
#[derive(Clone)]
pub struct PyAudioData {
    pub(crate) inner: AudioData,
}

#[pymethods]
impl PyAudioData {
    #[new]
    #[pyo3(signature = (audio, sample_rate=16000, num_channels=1))]
    fn new(audio: Vec<u8>, sample_rate: u32, num_channels: u16) -> Self {
        Self {
            inner: AudioData {
                audio: Bytes::from(audio),
                sample_rate,
                num_channels,
            },
        }
    }

    #[getter]
    fn sample_rate(&self) -> u32 {
        self.inner.sample_rate
    }

    #[getter]
    fn num_channels(&self) -> u16 {
        self.inner.num_channels
    }

    #[getter]
    fn num_frames(&self) -> usize {
        self.inner.num_frames()
    }

    #[getter]
    fn duration_secs(&self) -> f64 {
        self.inner.duration_secs()
    }

    /// Get audio bytes as Python bytes object
    fn audio_bytes<'py>(&self, py: Python<'py>) -> PyObject {
        pyo3::types::PyBytes::new_bound(py, &self.inner.audio)
            .into_any()
            .unbind()
    }
}

/// Python wrapper for Frame (simplified enum representation)
#[pyclass(name = "Frame")]
#[derive(Clone)]
pub struct PyFrame {
    pub(crate) inner: Frame,
}

#[pymethods]
impl PyFrame {
    /// Create a StartFrame
    #[staticmethod]
    fn start() -> Self {
        Self {
            inner: Frame::Start(FrameHeader::new()),
        }
    }

    /// Create an EndFrame
    #[staticmethod]
    fn end() -> Self {
        Self {
            inner: Frame::End(FrameHeader::new()),
        }
    }

    /// Create a CancelFrame
    #[staticmethod]
    fn cancel() -> Self {
        Self {
            inner: Frame::Cancel(FrameHeader::new()),
        }
    }

    /// Create a TextFrame
    #[staticmethod]
    fn text(text: String) -> Self {
        Self {
            inner: Frame::Text {
                header: FrameHeader::new(),
                data: TextData { text },
            },
        }
    }

    /// Create an AudioRawInputFrame
    #[staticmethod]
    fn audio_raw_input(audio: PyAudioData) -> Self {
        Self {
            inner: Frame::AudioRawInput {
                header: FrameHeader::new(),
                audio: audio.inner,
            },
        }
    }

    /// Create an AudioRawOutputFrame
    #[staticmethod]
    fn audio_raw_output(audio: PyAudioData) -> Self {
        Self {
            inner: Frame::AudioRawOutput {
                header: FrameHeader::new(),
                audio: audio.inner,
            },
        }
    }

    /// Get the frame's name
    fn name(&self) -> &str {
        self.inner.name()
    }

    /// Get the frame's header
    fn header(&self) -> PyFrameHeader {
        PyFrameHeader {
            inner: self.inner.header().clone(),
        }
    }

    /// Get the frame's type_id
    fn type_id(&self) -> u16 {
        self.inner.type_id()
    }

    /// Get text content (returns None if not a text frame)
    fn text_content(&self) -> Option<String> {
        match &self.inner {
            Frame::Text { data, .. } => Some(data.text.clone()),
            Frame::TextLlm { data, .. } => Some(data.text.clone()),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        format!("Frame({})", self.inner.name())
    }
}
