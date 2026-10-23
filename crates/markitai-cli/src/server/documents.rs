//! The document JSON of each finished item (docs/json-output.md).
//!
//! A conversion's result object exists only while it runs, so the document
//! is rendered then and kept in the job folder, outside `out/`: the result
//! files, their retry transactions and archives stay exactly what they were.
//! A retry that publishes a new result replaces it; one that fails keeps the
//! previous result and so its document.

use super::{
    State, jobs,
    types::{ApiError, ApiResult},
};
use axum::{
    extract::{Path, State as ExtractState},
    http::{HeaderValue, header},
    response::{IntoResponse, Response},
};
use std::{path::Path as FsPath, path::PathBuf, sync::Arc};

fn path(folder: &FsPath, item_id: &str) -> Option<PathBuf> {
    // Item IDs are generated, but the name is checked before it names a file.
    (!item_id.is_empty()
        && item_id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'))
    .then(|| folder.join("documents").join(format!("{item_id}.json")))
}

/// The document of `result`, converted for the item `name` into `out`.
pub(super) fn render(
    name: &str,
    result: &markitai_core::ConversionOutput,
    out: &FsPath,
    duration_s: Option<f64>,
) -> String {
    markitai_core::document_json(
        result,
        &markitai_core::DocumentJsonContext {
            source_label: Some(name),
            base_dir: Some(out),
            duration_s,
        },
    )
    .render()
}

/// Keep the item's `document` in the job folder. A failure to keep it only
/// means the document is not available.
pub(super) fn keep(folder: &FsPath, item_id: &str, document: &str) {
    let Some(target) = path(folder, item_id) else {
        return;
    };
    let kept = target
        .parent()
        .ok_or(())
        .and_then(|parent| markitai_core::output::create_private_dir(parent).map_err(|_| ()))
        .and_then(|_| {
            let mut file =
                tempfile::NamedTempFile::new_in(target.parent().unwrap()).map_err(|_| ())?;
            std::io::Write::write_all(&mut file, document.as_bytes()).map_err(|_| ())?;
            file.persist(&target).map(drop).map_err(|_| ())
        });
    if kept.is_err() {
        let _ = std::fs::remove_file(&target);
    }
}

/// v1.20 `format: "json"` (`-f json`): whether a result's document is also
/// published beside its Markdown. A skip or a result without Markdown has none.
pub(super) fn wanted(cfg: &serde_json::Value, result: &markitai_core::ConversionOutput) -> bool {
    cfg["output"]["format"] == "json"
        && result.skip_reason.is_none()
        && (result.output_path.is_some() || result.llm_output_path.is_some())
}

/// The name the document of the item whose Markdown is `<base>.md` has in `out/`.
pub(super) fn beside_name(base: &str) -> String {
    format!("{base}.json")
}

/// Publish `document` as `<base>.json` in `out` (the name `markitai -f json`
/// gives it beside `<base>.md`); returns its name there.
pub(super) fn write_beside(out: &FsPath, base: &str, document: &str) -> std::io::Result<String> {
    let name = beside_name(base);
    let mut file = markitai_core::output::deliverable_builder()
        .prefix(".markitai-json-")
        .tempfile_in(out)?;
    std::io::Write::write_all(&mut file, document.as_bytes())?;
    file.persist(out.join(&name)).map_err(|error| error.error)?;
    Ok(name)
}

/// The item warning a document that could not be published leaves.
pub(super) fn unwritten(error: std::io::Error) -> String {
    format!("The JSON document could not be written: {error}")
}

/// Forget the document of a deleted item, or of one whose new attempt has
/// no document.
pub(super) fn forget(folder: &FsPath, item_id: &str) {
    if let Some(target) = path(folder, item_id) {
        let _ = std::fs::remove_file(target);
    }
}

/// `GET /api/jobs/{job_id}/items/{item_id}/document`: the item's document
/// JSON, 404 `result_unavailable` when the item has none (not finished, a
/// failure, or converted before documents were kept).
pub(super) async fn document(
    ExtractState(state): ExtractState<Arc<State>>,
    Path((id, item_id)): Path<(String, String)>,
) -> ApiResult<Response> {
    let job = jobs::get(&state, &id)?;
    crate::task::blocking(move || {
        let _guard = job.access.lock().unwrap();
        let done = {
            let data = job.data.lock().unwrap();
            let item = data
                .items
                .iter()
                .find(|item| item.item_id == item_id)
                .ok_or_else(|| ApiError::new(404, "item_not_found", "item not found"))?;
            item.status == "done"
        };
        let unavailable = || ApiError::new(404, "result_unavailable", "item result not available");
        let target = path(&job.folder, &item_id)
            .filter(|_| done)
            .ok_or_else(unavailable)?;
        let bytes = std::fs::read(target).map_err(|_| unavailable())?;
        let mut response = bytes.into_response();
        response.headers_mut().insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/json; charset=utf-8"),
        );
        Ok(response)
    })
    .await
    .map_err(ApiError::internal)?
}
