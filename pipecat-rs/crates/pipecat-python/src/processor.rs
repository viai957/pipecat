use pyo3::prelude::*;

/// Python wrapper for PassthroughProcessor
#[pyclass(name = "PassthroughProcessor")]
pub struct PyPassthroughProcessor {
    pub(crate) name: String,
}

#[pymethods]
impl PyPassthroughProcessor {
    #[new]
    fn new(name: String) -> Self {
        Self { name }
    }

    #[getter]
    fn name(&self) -> &str {
        &self.name
    }
}
