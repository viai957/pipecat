//! Observer bridge: dispatches Rust pipeline observer events to Python
//! BaseObserver instances.
//!
//! Python observers implement async `on_process_frame(data)` and
//! `on_push_frame(data)`. This bridge converts the Rust `FrameProcessed`/
//! `FramePushed` data structures to Python-compatible `FrameProcessed`/
//! `FramePushed` dataclass instances and properly awaits the async callbacks.

use async_trait::async_trait;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use pipecat_pipeline::observer::{FrameProcessed, FramePushed, Observer};

use crate::conversion;
use crate::registry::FrameTypeRegistry;

/// Bridges Rust pipeline observer events to Python observer instances.
///
/// Each observer callback creates a Python `FrameProcessed`/`FramePushed`
/// dataclass and calls the Python observer's async method, properly awaiting
/// the returned coroutine via `pyo3_async_runtimes::tokio::into_future()`.
pub struct PythonObserverBridge {
    /// Python BaseObserver instances.
    py_observers: Vec<PyObject>,
    /// Shared frame type registry for conversions.
    registry: std::sync::Arc<FrameTypeRegistry>,
}

impl PythonObserverBridge {
    /// Create a new bridge with the given Python observers.
    pub fn new(py_observers: Vec<PyObject>) -> Self {
        Self {
            py_observers,
            registry: std::sync::Arc::new(FrameTypeRegistry::new()),
        }
    }

    /// Create an empty bridge (no observers).
    #[allow(dead_code)]
    pub fn empty() -> Self {
        Self {
            py_observers: Vec::new(),
            registry: std::sync::Arc::new(FrameTypeRegistry::new()),
        }
    }

    /// Check if there are any observers registered.
    pub fn has_observers(&self) -> bool {
        !self.py_observers.is_empty()
    }
}

#[async_trait]
impl Observer for PythonObserverBridge {
    async fn on_process_frame(&self, data: &FrameProcessed) {
        if self.py_observers.is_empty() {
            return;
        }

        // Prepare coroutines under the GIL, then await them outside.
        let futures: Vec<_> = Python::with_gil(|py| {
            let py_data = match create_py_frame_processed(py, data, &self.registry) {
                Ok(d) => d,
                Err(e) => {
                    tracing::warn!("Observer bridge: failed to create FrameProcessed: {e}");
                    return Vec::new();
                }
            };

            self.py_observers
                .iter()
                .filter_map(|obs| {
                    // Call the async method — returns a coroutine object
                    let coroutine = match obs.call_method1(py, "on_process_frame", (&py_data,)) {
                        Ok(c) => c,
                        Err(e) => {
                            tracing::warn!("Observer on_process_frame call error: {e}");
                            return None;
                        }
                    };

                    // Bridge the coroutine to a tokio future
                    let bound_coro = coroutine.bind(py).clone();
                    match pyo3_async_runtimes::tokio::into_future(bound_coro) {
                        Ok(future) => Some(future),
                        Err(e) => {
                            tracing::warn!("Observer coroutine bridge error: {e}");
                            None
                        }
                    }
                })
                .collect()
        });

        // Await all observer coroutines outside the GIL
        for future in futures {
            if let Err(e) = future.await {
                tracing::warn!("Observer on_process_frame error: {e}");
            }
        }
    }

    async fn on_push_frame(&self, data: &FramePushed) {
        if self.py_observers.is_empty() {
            return;
        }

        let futures: Vec<_> = Python::with_gil(|py| {
            let py_data = match create_py_frame_pushed(py, data, &self.registry) {
                Ok(d) => d,
                Err(e) => {
                    tracing::warn!("Observer bridge: failed to create FramePushed: {e}");
                    return Vec::new();
                }
            };

            self.py_observers
                .iter()
                .filter_map(|obs| {
                    let coroutine = match obs.call_method1(py, "on_push_frame", (&py_data,)) {
                        Ok(c) => c,
                        Err(e) => {
                            tracing::warn!("Observer on_push_frame call error: {e}");
                            return None;
                        }
                    };

                    let bound_coro = coroutine.bind(py).clone();
                    match pyo3_async_runtimes::tokio::into_future(bound_coro) {
                        Ok(future) => Some(future),
                        Err(e) => {
                            tracing::warn!("Observer coroutine bridge error: {e}");
                            None
                        }
                    }
                })
                .collect()
        });

        for future in futures {
            if let Err(e) = future.await {
                tracing::warn!("Observer on_push_frame error: {e}");
            }
        }
    }
}

/// Create a Python `FrameProcessed` dataclass from Rust data.
///
/// Constructs `pipecat.observers.base_observer.FrameProcessed` with the
/// Rust frame converted to a proper Python frame object, preserving
/// frame identity (id, metadata, etc.).
fn create_py_frame_processed(
    py: Python<'_>,
    data: &FrameProcessed,
    registry: &FrameTypeRegistry,
) -> PyResult<PyObject> {
    // Convert the Rust frame to a Python frame object
    let py_frame = conversion::rust_frame_to_py(py, data.frame.clone(), registry)?;

    // Convert direction to Python FrameDirection enum
    let fp_mod = py.import_bound("pipecat.processors.frame_processor")?;
    let direction_cls = fp_mod.getattr("FrameDirection")?;
    let py_direction = match data.direction {
        pipecat_core::FrameDirection::Downstream => direction_cls.getattr("DOWNSTREAM")?,
        pipecat_core::FrameDirection::Upstream => direction_cls.getattr("UPSTREAM")?,
    };

    // Try to construct the Python FrameProcessed dataclass.
    // If it fails (e.g., the processor field expects a FrameProcessor object),
    // fall back to a dict representation.
    let observer_mod = py.import_bound("pipecat.observers.base_observer")?;
    match observer_mod.getattr("FrameProcessed") {
        Ok(cls) => {
            let kwargs = PyDict::new_bound(py);
            // processor field: pass None since we only have the name string
            kwargs.set_item("processor", py.None())?;
            kwargs.set_item("frame", &py_frame)?;
            kwargs.set_item("direction", &py_direction)?;
            kwargs.set_item("timestamp", data.timestamp_us)?;
            let obj = cls.call((), Some(&kwargs))?;
            // Set processor name as an extra attribute for observability
            let _ = obj.setattr("_processor_name", &data.processor_name);
            Ok(obj.unbind())
        }
        Err(_) => {
            // Fallback: plain dict for compatibility
            let dict = PyDict::new_bound(py);
            dict.set_item("processor_name", &data.processor_name)?;
            dict.set_item("frame", &py_frame)?;
            dict.set_item("direction", &py_direction)?;
            dict.set_item("timestamp", data.timestamp_us)?;
            Ok(dict.into_any().unbind())
        }
    }
}

/// Create a Python `FramePushed` dataclass from Rust data.
fn create_py_frame_pushed(
    py: Python<'_>,
    data: &FramePushed,
    registry: &FrameTypeRegistry,
) -> PyResult<PyObject> {
    let py_frame = conversion::rust_frame_to_py(py, data.frame.clone(), registry)?;

    let fp_mod = py.import_bound("pipecat.processors.frame_processor")?;
    let direction_cls = fp_mod.getattr("FrameDirection")?;
    let py_direction = match data.direction {
        pipecat_core::FrameDirection::Downstream => direction_cls.getattr("DOWNSTREAM")?,
        pipecat_core::FrameDirection::Upstream => direction_cls.getattr("UPSTREAM")?,
    };

    let observer_mod = py.import_bound("pipecat.observers.base_observer")?;
    match observer_mod.getattr("FramePushed") {
        Ok(cls) => {
            let kwargs = PyDict::new_bound(py);
            // source/destination: pass None since we only have name strings
            kwargs.set_item("source", py.None())?;
            kwargs.set_item("destination", py.None())?;
            kwargs.set_item("frame", &py_frame)?;
            kwargs.set_item("direction", &py_direction)?;
            kwargs.set_item("timestamp", data.timestamp_us)?;
            let obj = cls.call((), Some(&kwargs))?;
            let _ = obj.setattr("_source_name", &data.source_name);
            let _ = obj.setattr("_destination_name", &data.destination_name);
            Ok(obj.unbind())
        }
        Err(_) => {
            let dict = PyDict::new_bound(py);
            dict.set_item("source_name", &data.source_name)?;
            dict.set_item("destination_name", &data.destination_name)?;
            dict.set_item("frame", &py_frame)?;
            dict.set_item("direction", &py_direction)?;
            dict.set_item("timestamp", data.timestamp_us)?;
            Ok(dict.into_any().unbind())
        }
    }
}
