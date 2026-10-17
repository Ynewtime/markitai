//! `--format json`: the document JSON on stdout, beside each published
//! Markdown file, and the batch index (docs/json-output.md).
//!
//! The Markdown files stay the published result that claims, receipts,
//! resume, reports and history own; the JSON is derived from the same
//! conversion once it is published, under the same name (`report.pdf.md`
//! and `report.pdf.json`, `report.pdf.v2.md` and `report.pdf.v2.json`).

use super::*;
use std::collections::BTreeMap;

/// The batch index beside the per-file JSON documents.
pub(super) const INDEX_NAME: &str = "markitai.index.json";

pub(super) fn wanted(cfg: &Value) -> bool {
    cfg["output"]["format"] == "json"
}

/// The JSON document a result is printed as on stdout.
pub(super) fn render_stdout(result: &ConversionOutput) -> String {
    markitai_core::document_json(result, &markitai_core::DocumentJsonContext::default()).render()
}

/// `<stem>.json` beside the Markdown a result published, or `None` when it
/// published no Markdown (a skip, a screenshot-only page).
pub(super) fn path_beside(result: &ConversionOutput) -> Option<PathBuf> {
    if let Some(base) = &result.output_path {
        let name = base.file_name()?.to_str()?.strip_suffix(".md")?;
        return Some(base.with_file_name(format!("{name}.json")));
    }
    let enhanced = result.llm_output_path.as_ref()?;
    let name = enhanced.file_name()?.to_str()?;
    let stem = name
        .strip_suffix(".llm.md")
        .or_else(|| name.strip_suffix(".md"))?;
    Some(enhanced.with_file_name(format!("{stem}.json")))
}

/// The JSON document of a result whose Markdown is published under `directory`.
pub(super) fn write_beside(
    result: &ConversionOutput,
    directory: &Path,
    cfg: &Value,
) -> Result<Option<PathBuf>, String> {
    let Some(path) = path_beside(result) else {
        return Ok(None);
    };
    let directory = path.parent().unwrap_or(directory);
    let document = markitai_core::document_json(
        result,
        &markitai_core::DocumentJsonContext {
            base_dir: Some(directory),
            ..Default::default()
        },
    );
    write_atomic(&path, document.render().as_bytes(), cfg)
        .map_err(|error| format!("Cannot write the JSON document {}: {error}", path.display()))?;
    Ok(Some(path))
}

fn write_atomic(path: &Path, bytes: &[u8], cfg: &Value) -> Result<(), String> {
    markitai_core::output::check_user_path(path, config::enabled(cfg, "/output/allow_symlinks"))
        .map_err(|error| error.to_string())?;
    let parent = path.parent().unwrap_or(Path::new("."));
    let mut file = markitai_core::output::deliverable_builder()
        .prefix(".markitai-json-")
        .tempfile_in(parent)
        .map_err(|error| error.to_string())?;
    file.write_all(bytes).map_err(|error| error.to_string())?;
    file.as_file()
        .sync_all()
        .map_err(|error| error.to_string())?;
    file.persist(path)
        .map_err(|error| error.error.to_string())?;
    Ok(())
}

/// One item of the batch index.
#[derive(serde::Serialize, serde::Deserialize)]
struct IndexItem {
    source: String,
    status: String,
    document: Option<String>,
    markdown: Option<String>,
    error: Option<String>,
    skip_reason: Option<String>,
}

#[derive(serde::Serialize)]
struct Index<'a> {
    schema_version: &'static str,
    tool: IndexTool,
    items: Vec<&'a IndexItem>,
}

#[derive(serde::Serialize)]
struct IndexTool {
    name: &'static str,
    version: &'static str,
    commit: &'static str,
}

fn relative(path: &Path, directory: &Path) -> String {
    match path.strip_prefix(directory) {
        Ok(relative) => relative
            .components()
            .map(|part| part.as_os_str().to_string_lossy())
            .collect::<Vec<_>>()
            .join("/"),
        Err(_) => path.to_string_lossy().into_owned(),
    }
}

/// Write `markitai.index.json` for the items of this run. Items of an
/// earlier run in the same folder (a resumed batch) that this run did not
/// touch stay listed; every item is ordered by source.
pub(super) fn write_index(
    directory: &Path,
    records: &[RunItem],
    cfg: &Value,
) -> Result<(), String> {
    let directory = config::expand_home(directory);
    let path = directory.join(INDEX_NAME);
    let mut items = BTreeMap::new();
    if let Ok(existing) = std::fs::read(&path)
        && let Ok(value) = serde_json::from_slice::<Value>(&existing)
        && let Some(listed) = value.get("items").and_then(Value::as_array)
    {
        for item in listed {
            if let Ok(item) = serde_json::from_value::<IndexItem>(item.clone()) {
                items.insert(item.source.clone(), item);
            }
        }
    }
    for record in records {
        let markdown = record
            .output
            .as_ref()
            .filter(|path| path.extension().is_some_and(|extension| extension == "md"));
        let document = (record.status == ItemStatus::Completed)
            .then(|| {
                markdown.and_then(|path| {
                    let mut result = ConversionOutput::default();
                    if path.to_string_lossy().ends_with(".llm.md") {
                        result.llm_output_path = Some(path.clone());
                    } else {
                        result.output_path = Some(path.clone());
                    }
                    path_beside(&result).filter(|json| json.is_file())
                })
            })
            .flatten();
        items.insert(
            record.report_key.clone(),
            IndexItem {
                source: record.report_key.clone(),
                status: match record.status {
                    ItemStatus::Completed => "completed",
                    ItemStatus::Skipped => "skipped",
                    ItemStatus::Failed => "failed",
                    ItemStatus::Pending => "pending",
                }
                .into(),
                document: document.as_deref().map(|path| relative(path, &directory)),
                markdown: markdown.map(|path| relative(path, &directory)),
                error: record.error.clone(),
                skip_reason: record.skip_reason.clone(),
            },
        );
    }
    let index = Index {
        schema_version: markitai_core::document_json::SCHEMA_VERSION,
        tool: IndexTool {
            name: "markitai",
            version: markitai_core::VERSION,
            commit: markitai_core::build_info::COMMIT,
        },
        items: items.values().collect(),
    };
    let mut text = serde_json::to_string_pretty(&index).map_err(|error| error.to_string())?;
    text.push('\n');
    write_atomic(&path, text.as_bytes(), cfg)
        .map_err(|error| format!("Cannot write the batch index {}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_json_takes_the_name_of_the_markdown_it_sits_beside() {
        let mut result = ConversionOutput::default();
        result.output_path = Some(PathBuf::from("out/a.pdf.v2.md"));
        result.llm_output_path = Some(PathBuf::from("out/a.pdf.v2.llm.md"));
        assert_eq!(
            path_beside(&result),
            Some(PathBuf::from("out/a.pdf.v2.json"))
        );
        result.output_path = None;
        assert_eq!(
            path_beside(&result),
            Some(PathBuf::from("out/a.pdf.v2.json"))
        );
        // An exact name without the base file: `-o out/report.md`.
        result.llm_output_path = Some(PathBuf::from("out/report.md"));
        assert_eq!(path_beside(&result), Some(PathBuf::from("out/report.json")));
        assert_eq!(path_beside(&ConversionOutput::default()), None);
    }
}
