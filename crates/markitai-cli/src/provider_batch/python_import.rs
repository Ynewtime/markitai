//! A batch that Python markitai 1.2.x submitted and left pending becomes a
//! native job here, so `--llm-batch-collect ID` and `--llm-batch --resume`
//! collect it after upgrading. Python's files under
//! `.markitai/batch-<id>/` are only read; its request rows are copied into
//! the job as submitted, and each base Markdown Python wrote is adopted into
//! this build's output ownership with its bytes unchanged.
use super::*;
use std::collections::HashSet;

const STATE_LIMIT: u64 = 16 * 1024 * 1024;
const TEXT_LIMIT: u64 = 8 * 1024 * 1024;
const REQUESTS_LIMIT: u64 = 200_000_000;

pub(super) enum Outcome {
    /// No Python state for this batch.
    Absent,
    /// Python already collected it.
    Collected,
    Imported(Box<store::Store>),
}

fn read(path: &Path, limit: u64) -> CliResult<Vec<u8>> {
    let file = markitai_core::platform::open_read(path, false).map_err(runtime)?;
    let metadata = file.metadata().map_err(runtime)?;
    if !metadata.is_file() || metadata.len() > limit {
        return Err(runtime(format!(
            "Python batch file is not a bounded regular file: {}",
            path.display()
        )));
    }
    markitai_core::platform::read_limited(file, limit)
        .map_err(runtime)?
        .ok_or_else(|| runtime("Python batch file exceeds its byte limit"))
}

fn text(path: &Path, limit: u64) -> CliResult<String> {
    String::from_utf8(read(path, limit)?).map_err(|_| {
        runtime(format!(
            "Python batch file is not UTF-8: {}",
            path.display()
        ))
    })
}

/// Python state directories in `output` that nobody has collected yet and
/// that are not already jobs here, by batch ID.
pub(super) fn pending_states(output: &Path, cfg: &Value) -> CliResult<Vec<String>> {
    let allow = config::enabled(cfg, "/output/allow_symlinks");
    let meta = output.join(".markitai");
    let Ok(entries) = fs::read_dir(&meta) else {
        return Ok(Vec::new());
    };
    let imported =
        store::Store::imported_states(output, allow, Default::default()).map_err(runtime)?;
    let mut found = Vec::new();
    for entry in entries {
        let entry = entry.map_err(runtime)?;
        let name = entry.file_name();
        let Some(id) = name.to_str().and_then(|name| name.strip_prefix("batch-")) else {
            continue;
        };
        if !provider::batch_id(id)
            || imported.contains(&Path::new(".markitai").join(name.as_os_str()))
        {
            continue;
        }
        let state = entry.path().join("state.json");
        if !state.is_file() {
            continue;
        }
        let parsed = provider::PythonState::parse(&read(&state, STATE_LIMIT)?).map_err(runtime)?;
        if !parsed.collected && parsed.ended_status.is_none() {
            found.push(id.to_owned());
        }
    }
    crate::sort::by(&mut found, |left, right| left.cmp(right));
    Ok(found)
}

/// Convert the Python state of `batch_id` in `output` into a native job.
pub(super) fn import(output: &Path, batch_id: &str, cfg: &Value) -> CliResult<Outcome> {
    let allow = config::enabled(cfg, "/output/allow_symlinks");
    let output = crate::report_store::resolve_path(output).map_err(runtime)?;
    if !provider::batch_id(batch_id) {
        return Ok(Outcome::Absent);
    }
    let relative_state = Path::new(".markitai").join(format!("batch-{batch_id}"));
    let directory = output.join(&relative_state);
    let state_path = directory.join("state.json");
    if !state_path.is_file() {
        return Ok(Outcome::Absent);
    }
    let state_bytes = read(&state_path, STATE_LIMIT)?;
    let state = provider::PythonState::parse(&state_bytes).map_err(runtime)?;
    if state.batch_id != batch_id {
        return Err(runtime("Python batch state names another batch"));
    }
    if state.collected {
        return Ok(Outcome::Collected);
    }
    if let Some(status) = &state.ended_status {
        return Err(runtime(format!(
            "Python batch {batch_id} ended with status {status}; there are no results to collect. Run the conversion again to submit a new batch"
        )));
    }
    let session = provider::Session::for_collection(cfg).map_err(runtime)?;
    let endpoint = session.identity().clone();
    if endpoint.provider != state.provider || endpoint.model != state.model {
        return Err(runtime(format!(
            "Python batch {batch_id} was submitted to {}/{}; configure that model to collect it",
            state.provider, state.model
        )));
    }
    let requests = read(&directory.join("requests.jsonl"), REQUESTS_LIMIT)?;
    let mut rows = HashMap::new();
    for line in requests.split(|byte| *byte == b'\n') {
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        let row: Value = serde_json::from_slice(line)
            .map_err(|_| runtime("Python batch requests are not valid JSONL"))?;
        let id = row
            .get("custom_id")
            .and_then(Value::as_str)
            .ok_or_else(|| runtime("Python batch request has no custom_id"))?
            .to_owned();
        if rows.insert(id, row).is_some() {
            return Err(runtime("Python batch requests repeat a custom_id"));
        }
    }
    let documents: Vec<_> = state
        .items
        .iter()
        .filter(|item| item.kind != provider::PythonKind::Image)
        .collect();
    if documents.is_empty() {
        return Err(runtime(
            "Python batch has no document request to collect (only image analyses of documents Python enhanced live)",
        ));
    }
    let sent: HashSet<&str> = state
        .items
        .iter()
        .filter(|item| item.answer.is_none())
        .map(|item| item.custom_id.as_str())
        .collect();
    if sent.len() != rows.len() || rows.keys().any(|id| !sent.contains(id.as_str())) {
        return Err(runtime(
            "Python batch requests do not match its state's items",
        ));
    }
    // The provider's own record of the batch: its input identity, and proof
    // that the configured credentials can read it.
    let batch = session
        .client()
        .map_err(runtime)?
        .retrieve(batch_id)
        .map_err(runtime)?;
    let generation = uuid::Uuid::new_v4().to_string();
    let orphans = state.orphan_images();
    let mut items = Vec::with_capacity(documents.len());
    for (position, document) in documents.iter().enumerate() {
        let images: Vec<_> = state
            .items
            .iter()
            .filter(|item| {
                item.kind == provider::PythonKind::Image && item.base_md == document.base_md
            })
            .map(|item| (item, rows.get(&item.custom_id)))
            .collect();
        // Image requests of documents Python enhanced live ride with the
        // first document so their paid usage is recorded; they are not applied.
        let detached: Vec<_> = if position == 0 {
            orphans
                .iter()
                .map(|item| (*item, rows.get(&item.custom_id)))
                .collect()
        } else {
            Vec::new()
        };
        let input = text(&directory.join(&document.input_md), TEXT_LIMIT)?;
        let base_path = output.join(&document.base_md);
        let base = text(&base_path, TEXT_LIMIT)?;
        let row = rows
            .get(&document.custom_id)
            .ok_or_else(|| runtime("Python batch document has no request"))?;
        let plan = provider::PythonPlan::new(
            &state, document, row, &input, &base, &images, &detached, &endpoint, cfg,
        )
        .map_err(runtime)?;
        let attached = plan.image_requests();
        let key = document.source.clone();
        let enhanced = Path::new(&document.base_md).with_extension("llm.md");
        let owner = Owner {
            generation: generation.clone(),
            mode: "directory".into(),
            input: output.clone(),
            output: output.clone(),
            kind: "file".into(),
            key: key.clone(),
        };
        adopt(&base_path, base.as_bytes(), &owner, allow)?;
        items.push(store::ImportItem {
            item: store::NewItem {
                custom_id: document.custom_id.clone(),
                source: document.source.clone(),
                key,
                base: document.base_md.clone().into(),
                enhanced,
                base_sha256: sha256_hex(base.as_bytes()),
                owner,
                plan: serde_json::to_value(&plan).map_err(runtime)?,
            },
            attached,
        });
    }
    let expected: Vec<String> = items
        .iter()
        .flat_map(|item| std::iter::once(item.item.custom_id.clone()).chain(item.attached.clone()))
        .collect();
    if expected.len() != rows.len() {
        return Err(runtime(
            "Python batch requests do not match its state's items",
        ));
    }
    let store = store::Store::import(
        &output,
        store::ImportJob {
            id: generation,
            endpoint: store::Endpoint {
                provider: endpoint.provider.clone(),
                api_base: endpoint.api_base.clone(),
                model: endpoint.model.clone(),
            },
            items,
            uploaded: provider::UploadedInput {
                file_id: batch.input_file_id.clone(),
                model: endpoint.model.clone(),
                custom_ids: expected,
                bytes: requests.len() as u64,
                sha256: sha256_hex(&requests),
            },
            requests,
            batch,
            imported: store::Imported {
                python_state: relative_state,
                state_sha256: sha256_hex(&state_bytes),
            },
        },
        allow,
        Default::default(),
    )
    .map_err(runtime)?;
    logging::shaped(
        Shape::item(Mark::Info),
        &format!(
            "Imported Python markitai batch {batch_id} ({} documents); its state files were left unchanged.",
            store.state().items.len()
        ),
        None,
    );
    Ok(Outcome::Imported(Box::new(store)))
}

/// Make a Python-written base Markdown this job's output, keeping its bytes:
/// the publication records a native receipt for the same content.
fn adopt(path: &Path, bytes: &[u8], owner: &Owner, allow: bool) -> CliResult<()> {
    let parent = path
        .parent()
        .ok_or_else(|| runtime("Python batch base has no parent directory"))?;
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| runtime("Python batch base name is not Unicode"))?;
    let enhanced = format!(
        "{}.llm.md",
        name.strip_suffix(".md")
            .ok_or_else(|| runtime("Python batch base is not Markdown"))?
    );
    let leases =
        MemberLeases::acquire(parent, &[name.to_owned(), enhanced], allow).map_err(runtime)?;
    // The bytes that were read must still be the file's before it is adopted.
    if member(path)? != bytes {
        return Err(runtime(
            "A Python batch base changed while it was imported; nothing was adopted",
        ));
    }
    let claim =
        Claim::new(leases, Some(owner.clone()), Policy::Overwrite, false).map_err(runtime)?;
    claim.publish(path, bytes).map_err(runtime)?;
    Ok(())
}
