use std::sync::Arc;
use std::time::Duration;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use tokio::sync::mpsc;

use pipecat_core::{Frame, FrameDirection, FrameHeader};
use pipecat_pipeline::{
    FrameEnvelope, FrameProcessor, Pipeline, PipelineRunner, PipelineTask, PipelineTaskParams,
};

use crate::conversion;
use crate::frame::PyFrame;
use crate::interruption::FrameSidecar;
use crate::observer_bridge::PythonObserverBridge;
use crate::processor::PyPassthroughProcessor;
use crate::py_processor::PythonFrameProcessor;
use crate::registry::FrameTypeRegistry;

/// Python wrapper for Pipeline
#[pyclass(name = "Pipeline")]
pub struct PyPipeline {
    processors: Option<Vec<Box<dyn FrameProcessor>>>,
}

#[pymethods]
impl PyPipeline {
    /// Create a new pipeline from a list of PassthroughProcessor instances.
    #[new]
    fn new(processors: Vec<Py<PyPassthroughProcessor>>) -> PyResult<Self> {
        let boxed: Vec<Box<dyn FrameProcessor>> = Python::with_gil(|py| {
            processors
                .into_iter()
                .map(|p| {
                    let proc = p.borrow(py);
                    let name = proc.name.clone();
                    drop(proc);
                    Box::new(pipecat_pipeline::PassthroughProcessor::new(&name))
                        as Box<dyn FrameProcessor>
                })
                .collect()
        });
        Ok(Self {
            processors: Some(boxed),
        })
    }
}

/// Python wrapper for PipelineTask
#[pyclass(name = "PipelineTask")]
pub struct PyPipelineTask {
    task: Option<PipelineTask>,
}

#[pymethods]
impl PyPipelineTask {
    #[new]
    #[pyo3(signature = (pipeline, heartbeat_secs=None))]
    fn new(pipeline: &mut PyPipeline, heartbeat_secs: Option<f64>) -> PyResult<Self> {
        let processors = pipeline
            .processors
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("pipeline already consumed"))?;
        let params = PipelineTaskParams {
            heartbeat_interval: heartbeat_secs.map(std::time::Duration::from_secs_f64),
            idle_timeout: None,
        };
        let pipe = Pipeline::new(processors);
        Ok(Self {
            task: Some(PipelineTask::new(pipe, params)),
        })
    }

    /// Queue a frame for injection into the pipeline
    fn queue_frame(&self, frame: PyFrame) -> PyResult<()> {
        if let Some(task) = &self.task {
            task.queue_frame(frame.inner)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        } else {
            Err(PyRuntimeError::new_err("task already consumed"))
        }
    }

    /// Cancel the pipeline
    fn cancel(&self) -> PyResult<()> {
        if let Some(task) = &self.task {
            task.cancel()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        } else {
            Err(PyRuntimeError::new_err("task already consumed"))
        }
    }
}

/// Python wrapper for PipelineRunner
#[pyclass(name = "PipelineRunner")]
pub struct PyPipelineRunner;

#[pymethods]
impl PyPipelineRunner {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Run a pipeline task to completion (blocking).
    ///
    /// This creates a new tokio runtime and blocks until the pipeline finishes.
    fn run(&self, task: &mut PyPipelineTask) -> PyResult<()> {
        let mut inner_task = task
            .task
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("task already consumed"))?;

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("failed to create runtime: {e}")))?;

        rt.block_on(async { PipelineRunner::run(&mut inner_task).await })
            .map(|_reason| ())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
}

// ── Native Pipeline Task (Python processor bridge) ───────────────────────

/// Pipeline task that wraps Python FrameProcessor instances in Rust's
/// pipeline engine. This is the primary entry point for the native engine.
///
/// The task exposes `queue_frame()` and `cancel()` so that Python can inject
/// frames into the Rust pipeline while it is running. The external frame
/// sender is captured *before* `run_async()` moves the task, ensuring the
/// handle remains valid for the task's lifetime.
///
/// Usage from Python::
///
///     task = NativePipelineTask(
///         processors=[p1, p2, p3],
///         observers=[obs],
///         heartbeat_secs=5.0,
///     )
///     # queue_frame / cancel are available before AND during run_async
///     await task.run_async()
#[pyclass(name = "NativePipelineTask")]
pub struct PyNativePipelineTask {
    py_processors: Vec<PyObject>,
    py_observers: Vec<PyObject>,
    heartbeat_secs: Option<f64>,
    /// External frame sender — cloned from PipelineTask before run() consumes it.
    frame_tx: Option<mpsc::UnboundedSender<FrameEnvelope>>,
    /// Shared frame type registry for Python↔Rust conversions.
    registry: Arc<FrameTypeRegistry>,
    /// Whether run_async() has already been called.
    running: bool,
    /// Optional Python callback invoked when a terminal frame reaches the sink.
    /// Signature: `callback(frame_name: str)` where frame_name is "End", "Stop", or "Cancel".
    /// This enables hybrid delegation: Python lifecycle + Rust routing.
    sink_callback: Option<PyObject>,
    /// Shared sidecar for non-serializable Python objects (asyncio.Event, etc.).
    /// Shared between all PythonFrameProcessors and the sink monitor.
    sidecar: Arc<FrameSidecar>,
}

#[pymethods]
impl PyNativePipelineTask {
    #[new]
    #[pyo3(signature = (processors, observers=None, heartbeat_secs=None, sink_callback=None))]
    fn new(
        processors: Vec<PyObject>,
        observers: Option<Vec<PyObject>>,
        heartbeat_secs: Option<f64>,
        sink_callback: Option<PyObject>,
    ) -> Self {
        Self {
            py_processors: processors,
            py_observers: observers.unwrap_or_default(),
            heartbeat_secs,
            frame_tx: None,
            registry: Arc::new(FrameTypeRegistry::new()),
            running: false,
            sink_callback,
            sidecar: Arc::new(FrameSidecar::new()),
        }
    }

    /// Set the sink callback. Called when a terminal frame reaches the pipeline sink.
    ///
    /// The callback receives the terminal frame's name as a string ("End", "Stop",
    /// "Cancel", or "FatalError"). This enables Python to coordinate lifecycle
    /// events while Rust owns the frame routing loop.
    #[setter]
    fn set_sink_callback(&mut self, callback: Option<PyObject>) {
        self.sink_callback = callback;
    }

    /// Queue a Python frame for injection into the Rust pipeline.
    ///
    /// Converts the Python frame to a Rust Frame and sends it downstream
    /// via the pipeline's external channel. Can be called before or during
    /// `run_async()`.
    fn queue_frame(&self, py: Python<'_>, py_frame: &Bound<'_, PyAny>) -> PyResult<()> {
        let tx = self.frame_tx.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err(
                "Cannot queue frame: task not initialized (call run_async first, or task already finished)",
            )
        })?;

        let frame = conversion::py_frame_to_rust(py, py_frame, &self.registry)?;
        tx.send(FrameEnvelope {
            frame,
            direction: FrameDirection::Downstream,
        })
        .map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to queue frame: {}", e.0.frame.name()))
        })?;

        Ok(())
    }

    /// Queue a Python frame traveling upstream into the pipeline sink.
    fn queue_upstream_frame(&self, py: Python<'_>, py_frame: &Bound<'_, PyAny>) -> PyResult<()> {
        let tx = self.frame_tx.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Cannot queue frame: task not initialized")
        })?;

        let frame = conversion::py_frame_to_rust(py, py_frame, &self.registry)?;
        tx.send(FrameEnvelope {
            frame,
            direction: FrameDirection::Upstream,
        })
        .map_err(|e| {
            PyRuntimeError::new_err(format!(
                "Failed to queue upstream frame: {}",
                e.0.frame.name()
            ))
        })?;

        Ok(())
    }

    /// Cancel the pipeline by injecting a CancelFrame.
    fn cancel(&self) -> PyResult<()> {
        let tx = self.frame_tx.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Cannot cancel: task not initialized")
        })?;

        tx.send(FrameEnvelope {
            frame: Frame::Cancel(FrameHeader::new()),
            direction: FrameDirection::Downstream,
        })
        .map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to send cancel: {}", e.0.frame.name()))
        })?;

        Ok(())
    }

    /// Whether the pipeline is currently running.
    #[getter]
    fn is_running(&self) -> bool {
        self.running
    }

    /// Run the native pipeline to completion.
    ///
    /// Returns a Python awaitable that completes when the pipeline finishes
    /// (EndFrame/StopFrame/CancelFrame reaches the sink).
    ///
    /// This bridges tokio's pipeline loop to Python's asyncio event loop:
    /// - Rust owns the pipeline loop (frame routing, backpressure, heartbeat)
    /// - Python processors are called via GIL-protected callbacks
    /// - The GIL is released during Rust-side work, acquired only for Python calls
    ///
    /// The `frame_tx` sender is captured before moving the task, so
    /// `queue_frame()` / `cancel()` remain usable while the pipeline runs.
    fn run_async<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if self.running {
            return Err(PyRuntimeError::new_err("Task is already running"));
        }

        // Wrap each Python processor in a PythonFrameProcessor with shared sidecar.
        // The shared sidecar allows asyncio.Event objects (on InterruptionFrame) to be
        // stored during frame conversion and retrieved at the sink for completion.
        let mut boxed_processors: Vec<Box<dyn FrameProcessor>> = Vec::new();
        for py_proc in &self.py_processors {
            let wrapper = PythonFrameProcessor::new_with_sidecar(
                py,
                py_proc.clone_ref(py),
                Arc::clone(&self.sidecar),
            )
                .map_err(|e| {
                    PyRuntimeError::new_err(format!(
                        "Failed to wrap processor: {e}"
                    ))
                })?;
            boxed_processors.push(Box::new(wrapper));
        }

        // Create observer bridge
        let observer_bridge = PythonObserverBridge::new(
            self.py_observers.iter().map(|o| o.clone_ref(py)).collect(),
        );

        // Build Rust pipeline and task
        let pipeline = Pipeline::new(boxed_processors);
        let params = PipelineTaskParams {
            heartbeat_interval: self.heartbeat_secs.map(Duration::from_secs_f64),
            idle_timeout: None,
        };
        let mut task = PipelineTask::new(pipeline, params);

        if observer_bridge.has_observers() {
            task.add_observer(Box::new(observer_bridge));
        }

        // Capture the frame sender BEFORE moving task into the async closure.
        // This is the key fix: Python can now call queue_frame()/cancel() while
        // the Rust pipeline loop is running.
        self.frame_tx = Some(task.frame_sender());
        self.running = true;

        // Capture the sink callback and sidecar for the async closure.
        let sink_cb = self.sink_callback.as_ref().map(|cb| cb.clone_ref(py));
        let sidecar = Arc::clone(&self.sidecar);

        // Bridge the tokio future to a Python awaitable.
        // `future_into_py` ensures the GIL is released while tokio runs,
        // and re-acquired when Python callbacks (process_frame) need it.
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let end_reason = task.run()
                .await
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            // Clean up any remaining sidecar entries (uncompleted interruption events).
            sidecar.clear();

            // If a sink callback was provided, invoke it with the end reason
            // so Python can coordinate lifecycle events (cleanup, event handlers).
            if let Some(cb) = sink_cb {
                let reason_str = end_reason.as_str().to_string();
                Python::with_gil(|py| {
                    let _ = cb.call1(py, (reason_str,));
                });
            }

            Ok(())
        })
    }
}
