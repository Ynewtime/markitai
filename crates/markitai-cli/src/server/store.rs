use super::{
    jobs::{Job, JobData},
    types::{ApiError, ApiResult, Item, JobOptions, now},
};
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    fs::{self, File},
    io::Write,
    path::{Component, Path, PathBuf},
    sync::{Arc, Mutex},
};

pub(super) fn private_dir(path: &Path) -> std::io::Result<()> {
    markitai_core::output::check_path(path, false).map_err(std::io::Error::other)?;
    fs::create_dir_all(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}
pub(super) fn safe_file(root: &Path, name: &str) -> ApiResult<PathBuf> {
    let relative = Path::new(name);
    if name.is_empty()
        || name.contains('\\')
        || relative
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
    {
        return Err(ApiError::new(404, "file not found"));
    }
    let path = root.join(relative);
    markitai_core::output::check_path(&path, false)
        .map_err(|_| ApiError::new(404, "file not found"))?;
    if !path.is_file() {
        return Err(ApiError::new(404, "file not found"));
    }
    let canonical = path
        .canonicalize()
        .map_err(|_| ApiError::new(404, "file not found"))?;
    if !canonical.starts_with(root.canonicalize().map_err(ApiError::internal)?) {
        return Err(ApiError::new(404, "file not found"));
    }
    Ok(canonical)
}

pub(super) fn files(root: &Path) -> std::io::Result<Vec<(String, PathBuf)>> {
    fn walk(
        root: &Path,
        path: &Path,
        result: &mut Vec<(String, PathBuf)>,
        depth: usize,
    ) -> std::io::Result<()> {
        if depth > 32 || result.len() > 100_000 {
            return Err(std::io::Error::other("archive exceeds structural limit"));
        }
        for entry in fs::read_dir(path)? {
            let path = entry?.path();
            let metadata = fs::symlink_metadata(&path)?;
            if metadata.is_dir() {
                walk(root, &path, result, depth + 1)?;
            } else if metadata.is_file() {
                if path.file_name().is_some_and(|s| {
                    s.to_string_lossy().starts_with('.') && s.to_string_lossy().ends_with(".tmp")
                }) {
                    continue;
                }
                result.push((
                    path.strip_prefix(root)
                        .unwrap()
                        .to_string_lossy()
                        .replace('\\', "/"),
                    path,
                ));
            } else {
                return Err(std::io::Error::other("nonregular archive member"));
            }
        }
        Ok(())
    }
    markitai_core::output::check_path(root, false).map_err(std::io::Error::other)?;
    let mut result = Vec::new();
    walk(root, root, &mut result, 0)?;
    crate::sort::by(&mut result, |a, b| a.0.cmp(&b.0));
    Ok(result)
}

pub(super) fn persist(folder: &Path, data: &JobData) -> std::io::Result<()> {
    let mut items = serde_json::to_value(&data.items).map_err(std::io::Error::other)?;
    for item in items.as_array_mut().unwrap() {
        item["options"] = data
            .item_options
            .get(item["item_id"].as_str().unwrap_or(""))
            .unwrap_or(&data.options)
            .clone();
    }
    let meta = json!({"job_id":data.id,"created_at":data.created_at,"finished_at":data.finished_at,
        "status":data.status,"options":data.options,"dir_size_bytes":data.size,"version":2,"items":items,
        "native_bases":data.bases,"native_assets":data.assets,"native_transactions":data.transactions});
    let mut temporary = tempfile::NamedTempFile::new_in(folder)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        temporary
            .as_file()
            .set_permissions(fs::Permissions::from_mode(0o600))?;
    }
    serde_json::to_writer_pretty(&mut temporary, &meta).map_err(std::io::Error::other)?;
    temporary.write_all(b"\n")?;
    temporary.as_file().sync_all()?;
    temporary
        .persist(folder.join("meta.json"))
        .map_err(|error| error.error)?;
    File::open(folder)?.sync_all()?;
    Ok(())
}

pub(super) fn rehydrate(
    root: &Path,
    known: &Mutex<HashMap<String, Arc<Job>>>,
) -> std::io::Result<()> {
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let id = entry.file_name().to_string_lossy().into_owned();
        if id.len() != 12
            || !id
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            continue;
        }
        if known.lock().unwrap().contains_key(&id) {
            continue;
        }
        let folder = entry.path();
        if !entry.file_type()?.is_dir()
            || markitai_core::output::check_path(&folder, false).is_err()
        {
            continue;
        }
        super::transaction::recover(&folder)?;
        let Ok(path) = safe_file(&folder, "meta.json") else {
            continue;
        };
        if fs::metadata(&path)?.len() > 16 * 1024 * 1024 {
            continue;
        }
        let value: Value = match fs::read(&path)
            .ok()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        {
            Some(value) => value,
            None => continue,
        };
        if value["status"] != "done" {
            continue;
        }
        let Some(raw_items) = value["items"].as_array() else {
            continue;
        };
        if raw_items.len() > 100_000 {
            continue;
        }
        let bases: HashMap<String, String> =
            serde_json::from_value(value["native_bases"].clone()).unwrap_or_default();
        let mut items = Vec::new();
        for (index, raw) in raw_items.iter().enumerate() {
            let mut object =
                serde_json::to_value(Item::new(index + 1, String::new(), "file", None)).unwrap();
            if let (Some(target), Some(source)) = (object.as_object_mut(), raw.as_object()) {
                target.extend(source.clone());
            }
            if let Some(diagnostics) = object.get("diagnostics")
                && !diagnostics.is_null()
                && crate::diagnostics::AttemptDiagnostics::from_value(diagnostics).is_err()
            {
                // An invalid optional observation does not erase the retained output row.
                object.as_object_mut().unwrap().remove("diagnostics");
                eprintln!("Serve: ignored invalid stored attempt diagnostics");
            }
            if let Ok(mut item) = serde_json::from_value::<Item>(object) {
                // Older history writers saved the actual enhanced name in both
                // fields. Adapt the public base name without renaming any file.
                // Native indexes disambiguate a literal source named notes.llm.
                if !bases.contains_key(&item.item_id)
                    && let Some(stem) = item
                        .output_name
                        .as_deref()
                        .and_then(|name| name.strip_suffix(".llm.md"))
                {
                    item.output_name = Some(format!("{stem}.md"));
                }
                if raw.get("llm_enhanced").is_none() {
                    item.llm_enhanced = item.output.as_deref().is_some_and(|output| {
                        bases.get(&item.item_id).map_or_else(
                            || output.ends_with(".llm.md"),
                            |base| output == format!("{base}.llm.md"),
                        )
                    });
                }
                if raw.get("retryable").is_none()
                    && value["options"]["origin"] == "cli"
                    && item.kind == "file"
                {
                    item.retryable = false;
                }
                if item.finished_at.is_none() {
                    item.finished_at = value["finished_at"].as_str().map(str::to_owned);
                }
                items.push(item);
            }
        }
        let mut options = serde_json::to_value(JobOptions::default()).unwrap();
        if let (Some(target), Some(source)) =
            (options.as_object_mut(), value["options"].as_object())
        {
            target.extend(source.clone());
        }
        let size = value["dir_size_bytes"].as_u64().unwrap_or_else(|| {
            files(&folder.join("out"))
                .unwrap_or_default()
                .iter()
                .filter_map(|(_, p)| fs::metadata(p).ok().map(|m| m.len()))
                .sum()
        });
        let data = JobData {
            id: id.clone(),
            created_at: value["created_at"].as_str().unwrap_or("").into(),
            finished_at: value["finished_at"].as_str().map(str::to_owned),
            status: "done".into(),
            persistence_error: None,
            options,
            items,
            size,
            bases,
            assets: serde_json::from_value(value["native_assets"].clone()).unwrap_or_default(),
            item_options: raw_items
                .iter()
                .filter_map(|item| {
                    Some((
                        item["item_id"].as_str()?.to_owned(),
                        item.get("options")?
                            .as_object()
                            .map(|value| Value::Object(value.clone()))?,
                    ))
                })
                .collect(),
            transactions: Vec::new(),
        };
        known
            .lock()
            .unwrap()
            .entry(id)
            .or_insert_with(|| Arc::new(Job::new(folder, data)));
    }
    Ok(())
}

pub(super) fn measure(folder: &Path) -> std::io::Result<u64> {
    let mut size = 0u64;
    for name in ["out", "uploads"] {
        let path = folder.join(name);
        if !path.try_exists()? {
            continue;
        }
        for (_, file) in files(&path)? {
            size = size
                .checked_add(fs::metadata(file)?.len())
                .ok_or_else(|| std::io::Error::other("job size overflow"))?;
        }
    }
    Ok(size)
}
pub(super) fn finish(job: &Job) -> std::io::Result<()> {
    let calculated = if job.data.lock().unwrap().persistence_error.is_some() {
        Err(std::io::Error::other("job requires recovery"))
    } else {
        measure(&job.folder)
    };
    let mut data = job.data.lock().unwrap();
    data.finished_at = Some(now());
    let result = calculated.and_then(|size| {
        data.size = size;
        data.status = "done".into();
        persist(&job.folder, &data)
    });
    if let Err(error) = &result {
        eprintln!("Serve: history persistence failed: {error}");
        data.status = "error".into();
        data.persistence_error = Some(
            "history could not be persisted; completed artifacts remain available until shutdown"
                .into(),
        );
    }
    if result.is_ok() {
        super::transaction::committed(&job.folder, &mut data);
    }
    let mut progress = data.progress();
    if let Some(error) = &data.persistence_error {
        progress["persistence_error"] = json!(error);
    }
    let _ = job.events.send(("job", progress));
    result
}
