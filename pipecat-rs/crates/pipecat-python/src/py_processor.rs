//! PythonFrameProcessor: Rust FrameProcessor that delegates to a Python
//! FrameProcessor instance.
//!
//! This is the core bridge between Rust's pipeline engine and Python's processor
//! ecosystem. When Rust drives the pipeline loop:
//!
//! 1. Each Python processor has `enable_direct_mode=True` (bypasses Python's
//!    internal queues).
//! 2. `__internal_push_frame` is replaced with a `RustPushBridge` that routes
//!    frames through Rust's bounded channels via `ProcessorContext::try_push_frame`.
//! 3. Python's event handlers (`on_before/after_process/push_frame`) still fire
//!    normally since we call the Python-level methods.

use std::sync::Arc;

use async_trait::async_trait;
use pyo3::prelude::*;

use pipecat_core::{Frame, FrameDirection, PipecatError, Result};
use pipecat_pipeline::processor::{FrameProcessor, ProcessorContext};
use pipecat_pipeline::FrameEnvelope;

use crate::conversion;
use crate::frame_bridge;
use crate::interruption::FrameSidecar;
use crate::registry::FrameTypeRegistry;

/// A Rust FrameProcessor that wraps a Python FrameProcessor instance.
///
/// The Python processor's `process_frame(frame, direction)` is called from
/// Rust's pipeline loop. Frames pushed by Python are intercepted by the
/// `RustPushBridge` and routed through Rust channels.
pub struct PythonFrameProcessor {
    /// The Python FrameProcessor instance.
    py_processor: PyObject,
    /// Cached processor name (avoid GIL acquisition per frame).
    name: String,
    /// Shared frame type registry.
    registry: Arc<FrameTypeRegistry>,
    /// Sidecar storage for non-serializable Python objects (asyncio.Event, etc.).
    sidecar: Arc<FrameSidecar>,
    /// Whether this processor is paused.
    paused: bool,
    /// Notify for resume after pause.
    pause_notify: Arc<tokio::sync::Notify>,
}

impl PythonFrameProcessor {
    /// Create a new bridge wrapping a Python FrameProcessor.
    #[allow(dead_code)] // Public API: convenience wrapper around new_with_sidecar
    pub fn new(py: Python<'_>, py_processor: PyObject) -> PyResult<Self> {
        Self::new_with_sidecar(py, py_processor, Arc::new(FrameSidecar::new()))
    }

    /// Create a new bridge with a shared sidecar for non-serializable Python objects.
    ///
    /// When multiple processors share a sidecar, asyncio.Event objects stored
    /// during frame conversion in one processor can be retrieved by the sink
    /// monitor or another processor.
    pub fn new_with_sidecar(
        py: Python<'_>,
        py_processor: PyObject,
        sidecar: Arc<FrameSidecar>,
    ) -> PyResult<Self> {
        let name: String = py_processor
            .call_method0(py, "__str__")
            .and_then(|s| s.extract(py))
            .unwrap_or_else(|_| "PythonProcessor".to_string());

        let name = py_processor
            .getattr(py, "name")
            .and_then(|n| n.extract::<String>(py))
            .unwrap_or(name);

        Ok(Self {
            py_processor,
            name,
            registry: Arc::new(FrameTypeRegistry::new()),
            sidecar,
            paused: false,
            pause_notify: Arc::new(tokio::sync::Notify::new()),
        })
    }

    /// Get a reference to the shared sidecar for non-serializable Python objects.
    #[allow(dead_code)] // Public API: will be used when sink monitors query sidecars
    pub fn sidecar(&self) -> &Arc<FrameSidecar> {
        &self.sidecar
    }
}

#[async_trait]
impl FrameProcessor for PythonFrameProcessor {
    async fn process_frame(
        &mut self,
        frame: Frame,
        direction: FrameDirection,
        _ctx: &ProcessorContext,
    ) -> Result<()> {
        // Handle pause/resume at the Rust level
        if self.paused {
            // Check if this is a resume frame for us
            if let Frame::Resume {
                processor_name, ..
            } = &frame
            {
                if processor_name == &self.name {
                    self.paused = false;
                    self.pause_notify.notify_one();
                }
            }
            // If paused and not a system/control frame, wait for resume
            if self.paused && frame.classification() == pipecat_core::FrameClass::Data {
                let notify = self.pause_notify.clone();
                notify.notified().await;
            }
        }

        // Check if this is a pause frame targeting us
        if let Frame::Pause {
            processor_name, ..
        } = &frame
        {
            if processor_name == &self.name {
                self.paused = true;
            }
        }

        // Convert Rust frame to Python frame object (with sidecar for Event reattachment)
        let sidecar_ref = &self.sidecar;
        let py_frame_result = Python::with_gil(|py| {
            conversion::rust_frame_to_py_with_sidecar(py, frame.clone(), &self.registry, Some(sidecar_ref))
        });

        let py_frame = py_frame_result.map_err(|e| PipecatError::Processor {
            message: format!("Frame conversion error: {e}"),
            fatal: false,
        })?;

        // Convert direction to Python int (0=downstream, 1=upstream)
        let py_direction = frame_bridge::rust_direction_to_py(direction);

        // Call Python's process_frame(frame, direction)
        let coroutine = Python::with_gil(|py| {
            let dir_enum = py
                .import_bound("pipecat.processors.frame_processor")
                .and_then(|m| m.getattr("FrameDirection"))
                .and_then(|cls| match py_direction {
                    0 => cls.getattr("DOWNSTREAM"),
                    _ => cls.getattr("UPSTREAM"),
                })
                .map_err(|e| PipecatError::Processor {
                    message: format!("Cannot get FrameDirection: {e}"),
                    fatal: false,
                })?;

            self.py_processor
                .call_method1(py, "process_frame", (py_frame, dir_enum))
                .map_err(|e| PipecatError::Processor {
                    message: format!("Python process_frame error: {e}"),
                    fatal: false,
                })
        })?;

        // Drive the Python coroutine to completion.
        //
        // The coroutine (process_frame) may contain `await push_frame()`
        // which calls the async RustPushBridge. Since the bridge's actual
        // work (try_send) is synchronous, the nested coroutine completes
        // immediately. We drive the full coroutine chain by iteratively
        // sending None and handling yielded sub-coroutines.
        Python::with_gil(|py| {
            let bound_coro = coroutine.bind(py);
            drive_coroutine_sync(py, bound_coro)
        })?;

        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }

    async fn setup(&mut self, ctx: &ProcessorContext) -> Result<()> {
        // Set enable_direct_mode=True on the Python processor.
        // This bypasses Python's internal queue system.
        // Also set __started=True so push_frame() doesn't reject frames
        // with "StartFrame not received yet".
        Python::with_gil(|py| {
            self.py_processor
                .setattr(py, "_enable_direct_mode", true)
                .map_err(|e| PipecatError::Processor {
                    message: format!("Failed to set direct_mode: {e}"),
                    fatal: false,
                })?;

            // Inject the RustPushBridge to replace __internal_push_frame
            let push_bridge = create_push_bridge(py, ctx, self.registry.clone(), self.sidecar.clone())?;
            self.py_processor
                .setattr(
                    py,
                    "_FrameProcessor__internal_push_frame",
                    push_bridge,
                )
                .map_err(|e| PipecatError::Processor {
                    message: format!("Failed to inject push bridge: {e}"),
                    fatal: false,
                })?;

            // Mark the Python processor as started so push_frame() doesn't
            // reject frames with "StartFrame not received yet". In direct mode,
            // the StartFrame goes through process_frame() directly but the
            // internal __started flag isn't set by the normal __start() path.
            self.py_processor
                .setattr(py, "_FrameProcessor__started", true)
                .map_err(|e| PipecatError::Processor {
                    message: format!("Failed to set __started: {e}"),
                    fatal: false,
                })?;

            // Force push_frame() to always go through __internal_push_frame
            // (replaced by RustPushBridge) rather than the ultra-fast path
            // that uses self._next.queue_frame(). In Rust-driven mode,
            // _next/_prev aren't set because processors are chained through
            // Rust's mpsc channels.
            //
            // Setting _has_push_frame_handlers=True ensures push_frame()
            // takes the handler path which calls __internal_push_frame.
            self.py_processor
                .setattr(py, "_has_push_frame_handlers", true)
                .map_err(|e| PipecatError::Processor {
                    message: format!("Failed to set push handler flag: {e}"),
                    fatal: false,
                })?;
            self.py_processor
                .setattr(py, "_push_fast_ready", false)
                .map_err(|e| PipecatError::Processor {
                    message: format!("Failed to disable fast push: {e}"),
                    fatal: false,
                })?;

            Ok(())
        })
    }

    async fn cleanup(&mut self) -> Result<()> {
        // Clean up sidecar entries (asyncio.Event objects, etc.)
        self.sidecar.clear();

        // Call Python processor's cleanup if it exists.
        // Use drive_coroutine_sync (same as process_frame) to properly
        // await the cleanup coroutine without relying on pyo3_async_runtimes.
        Python::with_gil(|py| {
            let cleanup = match self.py_processor.getattr(py, "cleanup") {
                Ok(c) if !c.is_none(py) => c,
                _ => return Ok::<(), PipecatError>(()),
            };
            let coroutine = cleanup.call0(py).map_err(|e| PipecatError::Processor {
                message: format!("Python cleanup error: {e}"),
                fatal: false,
            })?;
            let bound_coro = coroutine.bind(py);
            // Ignore errors during cleanup — processor may already be partially torn down
            let _ = drive_coroutine_sync(py, bound_coro);
            Ok(())
        })?;

        Ok(())
    }
}

/// Drive a Python coroutine to completion synchronously.
///
/// Handles nested `await` chains by recursively driving sub-coroutines.
/// This works for coroutines where all awaited sub-coroutines complete
/// immediately (no I/O suspension). In Pipecat's case, process_frame
/// awaits push_frame which awaits the RustPushBridge (sync try_send),
/// so the entire chain completes without true suspension.
fn drive_coroutine_sync(py: Python<'_>, coro: &Bound<'_, pyo3::PyAny>) -> Result<()> {
    // Use Python helper to drive the coroutine. This handles the full
    // async protocol including nested awaits on sub-coroutines.
    let helper_code = r#"
def _drive_coro(coro):
    """Drive a coroutine to completion by iteratively sending None."""
    result = None
    while True:
        try:
            yielded = coro.send(result)
            # If the yielded value is a coroutine, drive it recursively
            import inspect
            if inspect.iscoroutine(yielded):
                result = _drive_coro(yielded)
            else:
                result = None
        except StopIteration as e:
            return e.value
"#;

    let helper_mod = pyo3::types::PyModule::from_code_bound(
        py,
        helper_code,
        "_coro_driver",
        "_coro_driver",
    )
    .map_err(|e| PipecatError::Processor {
        message: format!("Failed to create coroutine driver: {e}"),
        fatal: false,
    })?;

    helper_mod
        .getattr("_drive_coro")
        .and_then(|f| f.call1((coro,)))
        .map_err(|e| PipecatError::Processor {
            message: format!("Python coroutine error: {e}"),
            fatal: false,
        })?;

    Ok(())
}

fn create_push_bridge(
    py: Python<'_>,
    ctx: &ProcessorContext,
    registry: Arc<FrameTypeRegistry>,
    sidecar: Arc<FrameSidecar>,
) -> Result<PyObject> {
    let ds_sender = ctx.downstream_sender();
    let us_sender = ctx.upstream_sender();

    // Create a Python class that wraps the Rust push function.
    let bridge_code = r#"
class RustPushBridge:
    """Replaces FrameProcessor.__internal_push_frame to route through Rust channels."""

    def __init__(self, push_fn):
        self._push_fn = push_fn

    async def __call__(self, frame, direction):
        self._push_fn(frame, direction)

    def push_sync(self, frame, direction):
        self._push_fn(frame, direction)
"#;

    let bridge_mod = pyo3::types::PyModule::from_code_bound(
        py,
        bridge_code,
        "rust_push_bridge",
        "rust_push_bridge",
    )
    .map_err(|e| PipecatError::Processor {
        message: format!("Failed to create push bridge module: {e}"),
        fatal: false,
    })?;

    let bridge_cls = bridge_mod.getattr("RustPushBridge").map_err(|e| {
        PipecatError::Processor {
            message: format!("Failed to get RustPushBridge class: {e}"),
            fatal: false,
        }
    })?;

    // Create the Rust closure that does the actual frame conversion + channel send.
    // Captures: cloned senders + registry (all Arc-wrapped, Send + 'static).
    let push_fn = pyo3::types::PyCFunction::new_closure_bound(
        py,
        None,
        None,
        move |args: &Bound<'_, pyo3::types::PyTuple>,
              _kwargs: Option<&Bound<'_, pyo3::types::PyDict>>|
              -> PyResult<()> {
            let py = args.py();
            let py_frame = args.get_item(0)?;
            let py_direction = args.get_item(1)?;

            // Python FrameDirection: DOWNSTREAM=1, UPSTREAM=2
            let dir_value: i32 = py_direction.getattr("value")?.extract()?;
            let direction = if dir_value == 1 {
                FrameDirection::Downstream
            } else {
                FrameDirection::Upstream
            };

            // Convert Python frame to Rust Frame (with sidecar for asyncio.Event)
            let frame = conversion::py_frame_to_rust_with_sidecar(py, &py_frame, &registry, Some(&sidecar))?;

            // Send through the appropriate channel
            let envelope = FrameEnvelope { frame, direction };
            let result = match direction {
                FrameDirection::Downstream => ds_sender.try_send(envelope),
                FrameDirection::Upstream => us_sender.try_send(envelope),
            };

            result.map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to push frame through Rust channel: {}",
                    e.frame.name()
                ))
            })?;

            Ok(())
        },
    )
    .map_err(|e| PipecatError::Processor {
        message: format!("Failed to create push closure: {e}"),
        fatal: false,
    })?;

    let bridge = bridge_cls
        .call1((push_fn,))
        .map_err(|e| PipecatError::Processor {
            message: format!("Failed to instantiate RustPushBridge: {e}"),
            fatal: false,
        })?;

    Ok(bridge.unbind())
}
