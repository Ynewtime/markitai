use pyo3::prelude::*;

/// A panic is answered with the shared `internal_error` envelope, which the
/// Python wrapper raises as a `ConversionError` like any other code.
#[pyfunction]
fn convert_json(py: Python<'_>, request: String) -> PyResult<String> {
    Ok(py.detach(move || markitai_core::convert_json_caught(&request)))
}

/// Answers with the same envelope as `convert_json`, a panic included, so the
/// wrapper raises configuration failures as it raises a conversion's.
#[pyfunction]
#[pyo3(signature = (overrides, *, model = "MarkitaiConfig", schema = false))]
fn config_json(overrides: &str, model: &str, schema: bool) -> String {
    markitai_core::config_json_caught(overrides, model, schema)
}

/// Runs the command line `arguments` (the program name first: `markitai`,
/// `mkai` or `markitai-mcp`) as the `markitai` executable does, and ends the
/// process with its exit status, as the executable does: the wheel's entry
/// points call it and nothing of the interpreter runs after it.
#[pyfunction]
fn run_cli(py: Python<'_>, arguments: Vec<std::ffi::OsString>) {
    py.detach(move || std::process::exit(markitai_cli::run(arguments)))
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(convert_json, module)?)?;
    module.add_function(wrap_pyfunction!(config_json, module)?)?;
    module.add_function(wrap_pyfunction!(run_cli, module)?)?;
    module.add("__version__", markitai_core::VERSION)?;
    Ok(())
}
