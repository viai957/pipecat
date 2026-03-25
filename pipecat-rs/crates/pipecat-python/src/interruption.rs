//! Sidecar storage for non-serializable Python objects that need to survive
//! frame round-trips through Rust.
//!
//! Primary use case: `asyncio.Event` attached to `InterruptionTaskFrame`.
//! When the resulting `InterruptionFrame` reaches the pipeline sink,
//! the event must be set to unblock the original caller.

use std::collections::HashMap;
use std::sync::Mutex;

use pyo3::prelude::*;

/// Thread-safe sidecar map keyed by frame ID.
///
/// Stores Python objects that cannot be serialized across the Rust boundary
/// (e.g., `asyncio.Event`, `LLMContext` references).
pub struct FrameSidecar {
    entries: Mutex<HashMap<u64, SidecarEntry>>,
}

/// A sidecar entry with its associated Python object and metadata.
struct SidecarEntry {
    /// The Python object (e.g., asyncio.Event).
    py_object: PyObject,
    /// What kind of sidecar this is (for cleanup/dispatch).
    kind: SidecarKind,
}

/// Classification of sidecar entries for dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SidecarKind {
    /// An asyncio.Event that should be set when the frame reaches the sink.
    InterruptionEvent,
    /// An arbitrary Python object stashed for later retrieval.
    #[allow(dead_code)] // Used in tests; will be used for generic sidecar storage
    Generic,
}

impl FrameSidecar {
    /// Create a new empty sidecar.
    pub fn new() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
        }
    }

    /// Store a Python object keyed by frame ID.
    pub fn store(&self, frame_id: u64, py_object: PyObject, kind: SidecarKind) {
        self.entries.lock().unwrap().insert(
            frame_id,
            SidecarEntry { py_object, kind },
        );
    }

    /// Retrieve and remove a sidecar entry by frame ID.
    pub fn take(&self, frame_id: u64) -> Option<(PyObject, SidecarKind)> {
        self.entries
            .lock()
            .unwrap()
            .remove(&frame_id)
            .map(|e| (e.py_object, e.kind))
    }

    /// Check if a sidecar entry exists for the given frame ID.
    #[allow(dead_code)]
    pub fn contains(&self, frame_id: u64) -> bool {
        self.entries.lock().unwrap().contains_key(&frame_id)
    }

    /// Set an asyncio.Event stored in the sidecar (for interruption completion).
    ///
    /// Acquires the GIL, retrieves the event, and calls `event.set()`.
    #[allow(dead_code)] // Will be called when sink completion is wired up
    pub fn complete_interruption_event(&self, frame_id: u64) {
        if let Some((py_event, kind)) = self.take(frame_id) {
            if kind == SidecarKind::InterruptionEvent {
                Python::with_gil(|py| {
                    let _ = py_event.call_method0(py, "set");
                });
            }
        }
    }

    /// Clean up all entries. Called when the pipeline ends or is cancelled.
    pub fn clear(&self) {
        self.entries.lock().unwrap().clear();
    }

    /// Number of entries currently stored.
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.entries.lock().unwrap().len()
    }
}

impl Default for FrameSidecar {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sidecar_store_and_take() {
        pyo3::prepare_freethreaded_python();

        let sidecar = FrameSidecar::new();

        Python::with_gil(|py| {
            let obj = py.None();
            sidecar.store(42, obj, SidecarKind::Generic);
            assert!(sidecar.contains(42));

            let (taken, kind) = sidecar.take(42).unwrap();
            assert_eq!(kind, SidecarKind::Generic);
            assert!(!sidecar.contains(42));
        });
    }

    #[test]
    fn sidecar_clear() {
        pyo3::prepare_freethreaded_python();

        let sidecar = FrameSidecar::new();

        Python::with_gil(|py| {
            sidecar.store(1, py.None(), SidecarKind::Generic);
            sidecar.store(2, py.None(), SidecarKind::InterruptionEvent);
            assert_eq!(sidecar.len(), 2);

            sidecar.clear();
            assert_eq!(sidecar.len(), 0);
        });
    }
}
