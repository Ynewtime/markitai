use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::panic::{AssertUnwindSafe, catch_unwind};

#[pyfunction]
fn convert_json(py: Python<'_>, request: String) -> PyResult<String> {
    py.detach(move || {
        catch_unwind(AssertUnwindSafe(|| markitai_core::convert_json(&request)))
            .map_err(|_| PyRuntimeError::new_err("Native conversion failed unexpectedly"))
    })
}

#[pyfunction]
#[pyo3(signature = (overrides, *, model = "MarkitaiConfig", schema = false))]
fn config_json(overrides: &str, model: &str, schema: bool) -> PyResult<String> {
    catch_unwind(AssertUnwindSafe(|| {
        if schema {
            return Ok(markitai_core::config::schema().to_string());
        }
        let overrides: serde_json::Value = serde_json::from_str(overrides)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        if !overrides.is_object() {
            return Err(PyValueError::new_err("config must be an object"));
        }
        let config = markitai_core::config::normalize_model(model, &overrides)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(config.to_string())
    }))
    .map_err(|_| PyRuntimeError::new_err("Native configuration failed unexpectedly"))?
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(convert_json, module)?)?;
    module.add_function(wrap_pyfunction!(config_json, module)?)?;
    module.add("__version__", markitai_core::VERSION)?;
    Ok(())
}
