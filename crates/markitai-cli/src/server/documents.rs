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

/// Keep the document of `result`, converted for the item `name` into `out`.
/// A failure to keep it only means the document is not available.
pub(super) fn keep(
    folder: &FsPath,
    item_id: &str,
    name: &str,
    result: &markitai_core::ConversionOutput,
    out: &FsPath,
    duration_s: Option<f64>,
) {
    let Some(target) = path(folder, item_id) else {
        return;
    };
    let document = markitai_core::document_json(
        result,
        &markitai_core::DocumentJsonContext {
            source_label: Some(name),
            base_dir: Some(out),
            duration_s,
        },
    )
    .render();
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
