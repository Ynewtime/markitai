use napi::bindgen_prelude::*;
use napi_derive::napi;

/// A panic is answered with the shared `internal_error` envelope, which the
/// JavaScript wrapper turns into a `ConversionError` like any other code.
fn run(request: &str) -> Result<String> {
    Ok(markitai_core::convert_json_caught(request))
}

pub struct ConvertTask {
    request: String,
}

impl Task for ConvertTask {
    type Output = String;
    type JsValue = String;

    fn compute(&mut self) -> Result<Self::Output> {
        run(&self.request)
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(output)
    }
}

#[napi]
pub fn convert_json(request: String) -> AsyncTask<ConvertTask> {
    AsyncTask::new(ConvertTask { request })
}

#[napi]
pub fn convert_json_sync(request: String) -> Result<String> {
    run(&request)
}

#[napi]
pub fn version() -> &'static str {
    markitai_core::VERSION
}
