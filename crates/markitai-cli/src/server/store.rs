use super::{
    jobs::{Job, JobData},
    types::{ApiError, ApiResult, Item, JobOptions, now},
};
use markitai_core::platform;
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
/// One service owns recovery and temporary uploads for the lifetime of this handle.
/// Do not unlink the lock: another process must observe the same locked inode.
pub(super) struct ServiceLock(File);
impl Drop for ServiceLock {
    fn drop(&mut self) {
        let _ = self.0.unlock();
    }
}
pub(super) fn service_lock(root: &Path) -> std::io::Result<ServiceLock> {
    markitai_core::output::check_path(root, false).map_err(std::io::Error::other)?;
    if !root.try_exists()? {
        private_dir(root)?;
    }
    let directory = platform::status(root)?;
    if !directory.metadata().is_dir() || !directory.owned_by_current_user() {
        return Err(std::io::Error::other(
            "serve jobs directory must be owned by the current user",
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if directory.metadata().permissions().mode() & 0o022 != 0 {
            return Err(std::io::Error::other(
                "serve jobs directory must not be writable by other users",
            ));
        }
    }
    let path = root.join(".serve.lock");
    let mut options = fs::OpenOptions::new();
    options.read(true).write(true).create(true).truncate(false);
    platform::private_file(&mut options);
    let file = platform::open_no_follow(&options, &path)?;
    let opened = platform::file_status(&file)?;
    let named = platform::status(&path)?;
    if !opened.metadata().is_file()
        || !opened.private()
        || !opened.owned_by_current_user()
        || opened.links() != 1
        || opened.id() != named.id()
        || opened.id().volume != directory.id().volume
    {
        return Err(std::io::Error::other("invalid serve lifetime lock"));
    }
    file.try_lock().map_err(|_| std::io::Error::other(
        "another serve instance is using this MARKITAI_HOME; stop it before starting a new server",
    ))?;
    let held = ServiceLock(file);
    if platform::status(root)?.id() != directory.id()
        || platform::status(&path)?.id() != opened.id()
    {
        return Err(std::io::Error::other("serve lock path changed"));
    }
    // Tighten a legacy directory only after exclusion has been obtained.
    private_dir(root)?;
    Ok(held)
}

const UPLOAD_MARKER: &str = ".markitai-upload-v1";
const UPLOAD_OWNER: &[u8] = b"markitai serve upload v1\n";
pub(super) fn mark_upload(stage: &Path) -> std::io::Result<()> {
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    platform::private_file(&mut options);
    let mut marker = platform::open_no_follow(&options, &stage.join(UPLOAD_MARKER))?;
    marker.write_all(UPLOAD_OWNER)?;
    marker.sync_all()?;
    platform::sync_directory(stage)
}
pub(super) fn unmark_upload(folder: &Path) -> std::io::Result<()> {
    fs::remove_file(folder.join(UPLOAD_MARKER))
}

/// Only called at startup while holding the service lifetime lock. Unknown old
/// temporary directories are retained: their name alone does not prove ownership.
pub(super) fn clean_uploads(root: &Path, _owner: &ServiceLock) -> std::io::Result<()> {
    clean_uploads_at(root, std::time::SystemTime::now())
}
fn clean_uploads_at(root: &Path, now: std::time::SystemTime) -> std::io::Result<()> {
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        if !entry.file_name().to_string_lossy().starts_with(".upload-") {
            continue;
        }
        let folder = entry.path();
        if !entry.file_type()?.is_dir()
            || markitai_core::output::check_path(&folder, false).is_err()
        {
            continue;
        }
        let marker = folder.join(UPLOAD_MARKER);
        let Ok(status) = platform::status(&marker) else {
            continue;
        };
        if !status.metadata().is_file()
            || !status.private()
            || !status.owned_by_current_user()
            || status.links() != 1
            || status.metadata().len() != UPLOAD_OWNER.len() as u64
            || status
                .metadata()
                .modified()
                .ok()
                .and_then(|time| now.duration_since(time).ok())
                .is_none_or(|age| age < std::time::Duration::from_secs(24 * 60 * 60))
        {
            continue;
        }
        let Ok(mut file) = platform::open_read(&marker, false) else {
            continue;
        };
        if platform::file_status(&file)?.id() != status.id() {
            continue;
        }
        let mut bytes = vec![0; UPLOAD_OWNER.len() + 1];
        use std::io::Read;
        let length = file.read(&mut bytes)?;
        if &bytes[..length] != UPLOAD_OWNER {
            continue;
        }
        let mut count = 0;
        if !owned_upload_tree(&folder, 0, &mut count)? {
            continue;
        }
        if platform::status(&marker)?.id() != status.id() {
            continue;
        }
        fs::remove_dir_all(&folder)?;
    }
    Ok(())
}
fn owned_upload_tree(path: &Path, depth: usize, count: &mut usize) -> std::io::Result<bool> {
    *count += 1;
    if depth > 32 || *count > 100_000 {
        return Ok(false);
    }
    let status = platform::status(path)?;
    if status.metadata().file_type().is_symlink()
        || !status.owned_by_current_user()
        || markitai_core::output::check_path(path, false).is_err()
    {
        return Ok(false);
    }
    if status.metadata().is_file() {
        return Ok(status.links() == 1);
    }
    if !status.metadata().is_dir() || !status.private() {
        return Ok(false);
    }
    for entry in fs::read_dir(path)? {
        if !owned_upload_tree(&entry?.path(), depth + 1, count)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Make a freshly staged job's uploads durable before the job is published.
///
/// The uploads are kept so an item can be retried, so they must outlive a power
/// loss once the job has been reported created. Asking the drive to persist every
/// file separately is what made a 1,000-file job take seconds to create: on macOS
/// `File::sync_all` is a full cache flush (`F_FULLFSYNC`), several milliseconds
/// each. The guarantee needs only one ordering: every byte has been handed to
/// the drive, and one full flush then covers all of it. So each file gets an
/// ordinary `fsync`, and a single full flush of the directory follows; the
/// metadata and parent-directory syncs that publish the job come after that, as
/// before. Where `fsync` itself already includes the drive flush (Linux,
/// Windows), each file is synced once, here, instead of as it arrives. Windows
/// cannot flush the directory; the job's metadata, flushed after it is
/// renamed into place, commits the new names.
pub(super) fn sync_uploads(uploads: &Path, names: &[&str]) -> std::io::Result<()> {
    for name in names {
        let file = fs::OpenOptions::new()
            .write(true)
            .open(uploads.join(name))?;
        hand_to_drive(&file)?;
    }
    platform::sync_directory(uploads)
}

#[cfg(target_vendor = "apple")]
fn hand_to_drive(file: &File) -> std::io::Result<()> {
    use std::os::fd::AsRawFd;
    // SAFETY: the descriptor stays open for the duration of the call.
    if unsafe { libc::fsync(file.as_raw_fd()) } == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

#[cfg(not(target_vendor = "apple"))]
fn hand_to_drive(file: &File) -> std::io::Result<()> {
    file.sync_all()
}

pub(super) fn safe_file(root: &Path, name: &str) -> ApiResult<PathBuf> {
    let relative = Path::new(name);
    if name.is_empty()
        || name.contains('\\')
        || relative
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
    {
        return Err(ApiError::new(404, "file_not_found", "file not found"));
    }
    let path = root.join(relative);
    markitai_core::output::check_path(&path, false)
        .map_err(|_| ApiError::new(404, "file_not_found", "file not found"))?;
    if !path.is_file() {
        return Err(ApiError::new(404, "file_not_found", "file not found"));
    }
    let canonical = platform::canonicalize(&path)
        .map_err(|_| ApiError::new(404, "file_not_found", "file not found"))?;
    if !canonical.starts_with(platform::canonicalize(root).map_err(ApiError::internal)?) {
        return Err(ApiError::new(404, "file_not_found", "file not found"));
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
    let installed =
        platform::persist(temporary, &folder.join("meta.json")).map_err(|error| error.error)?;
    platform::sync_renamed(&installed)?;
    platform::sync_directory(folder)?;
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
        let interrupted = value["status"] == "running";
        if value["status"] != "done" && !interrupted {
            continue;
        }
        if interrupted && (value["job_id"] != id || value["version"] != 2) {
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
            if let Some(failure) = object.get("rerun_failure")
                && !failure.is_null()
                && (super::types::RerunFailure::from_value(failure).is_err()
                    || object["status"] != "done"
                    || object["skipped"] == true
                    || !object["output"].is_string())
            {
                // A damaged optional outcome must not hide valid retained output.
                object.as_object_mut().unwrap().remove("rerun_failure");
                eprintln!("Serve: ignored invalid stored rerun failure");
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
                if interrupted && matches!(item.status.as_str(), "queued" | "running") {
                    item.status = "error".into();
                    item.error_code = Some("interrupted".into());
                    item.error = Some(
                        "conversion interrupted when the server stopped; retry this item".into(),
                    );
                    item.finished_at = Some(now());
                }
                items.push(item);
            }
        }
        if interrupted && items.len() != raw_items.len() {
            // Do not rewrite malformed history while silently dropping rows.
            continue;
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
        let mut data = JobData {
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
        if interrupted {
            data.finished_at = Some(now());
            data.size = measure(&folder)?;
            persist(&folder, &data)?;
        }
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

#[cfg(test)]
mod upload_sync_tests {
    use super::*;

    #[test]
    fn staged_uploads_are_synced_as_one_batch_and_a_missing_one_is_reported() {
        let temporary = tempfile::tempdir().unwrap();
        let uploads = temporary.path().join("uploads");
        fs::create_dir(&uploads).unwrap();
        for name in ["a.txt", "b with space.md", "ü.html"] {
            fs::write(uploads.join(name), name).unwrap();
        }
        sync_uploads(&uploads, &["a.txt", "b with space.md", "ü.html"]).unwrap();
        // The bytes are untouched and the batch can run again on the same files.
        sync_uploads(&uploads, &["a.txt"]).unwrap();
        assert_eq!(
            fs::read(uploads.join("ü.html")).unwrap(),
            "ü.html".as_bytes()
        );
        sync_uploads(&uploads, &[]).unwrap();
        let error = sync_uploads(&uploads, &["gone.txt"]).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::NotFound);
    }
}

#[cfg(test)]
mod recovery_tests {
    use super::*;

    #[test]
    fn lifetime_lock_refuses_a_second_server_and_releases_on_drop() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("jobs");
        let first = service_lock(&root).unwrap();
        assert!(service_lock(&root).is_err());
        drop(first);
        let _next = service_lock(&root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn lifetime_lock_tightens_owned_legacy_root_but_rejects_shared_writes() {
        use std::os::unix::fs::PermissionsExt;
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("jobs");
        fs::create_dir(&root).unwrap();
        fs::set_permissions(&root, fs::Permissions::from_mode(0o755)).unwrap();
        let guard = service_lock(&root).unwrap();
        assert_eq!(
            fs::metadata(&root).unwrap().permissions().mode() & 0o777,
            0o700
        );
        drop(guard);
        fs::set_permissions(&root, fs::Permissions::from_mode(0o770)).unwrap();
        assert!(service_lock(&root).is_err());
        assert_eq!(
            fs::metadata(&root).unwrap().permissions().mode() & 0o777,
            0o770
        );
    }

    #[test]
    fn interrupted_history_keeps_outputs_and_completed_items_and_can_reload() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("jobs");
        let _owner = service_lock(&root).unwrap();
        let folder = root.join("012345abcdef");
        private_dir(&folder.join("out")).unwrap();
        private_dir(&folder.join("uploads")).unwrap();
        fs::write(folder.join("out/kept.md"), "retained output").unwrap();
        fs::write(folder.join("uploads/pending.txt"), "retry input").unwrap();
        let mut kept = Item::new(1, "kept.txt".into(), "file", Some("kept.md".into()));
        kept.status = "done".into();
        kept.output = Some("kept.md".into());
        kept.cost_usd = Some(0.25);
        let mut active = Item::new(2, "pending.txt".into(), "file", None);
        active.status = "running".into();
        let queued = Item::new(3, "https://example.invalid/".into(), "url", None);
        let mut failed = Item::new(4, "failed.txt".into(), "file", None);
        failed.status = "error".into();
        failed.error_code = Some("unsupported".into());
        let data = JobData {
            id: "012345abcdef".into(),
            created_at: now(),
            finished_at: None,
            status: "running".into(),
            persistence_error: None,
            options: json!({}),
            items: vec![kept.clone(), active, queued, failed.clone()],
            size: 0,
            bases: HashMap::new(),
            assets: HashMap::new(),
            item_options: HashMap::new(),
            transactions: Vec::new(),
        };
        persist(&folder, &data).unwrap();
        let known = Mutex::new(HashMap::new());
        rehydrate(&root, &known).unwrap();
        {
            let registry = known.lock().unwrap();
            let restored = registry["012345abcdef"].data.lock().unwrap();
            assert_eq!(restored.status, "done");
            assert!(restored.finished_at.is_some());
            assert_eq!(restored.items[0].output, kept.output);
            assert_eq!(restored.items[0].cost_usd, Some(0.25));
            assert_eq!(restored.items[3].error_code, failed.error_code);
            for index in [1, 2] {
                assert_eq!(restored.items[index].status, "error");
                assert_eq!(
                    restored.items[index].error_code.as_deref(),
                    Some("interrupted")
                );
                assert!(restored.items[index].retryable);
            }
        }
        assert_eq!(
            fs::read_to_string(folder.join("out/kept.md")).unwrap(),
            "retained output"
        );
        assert_eq!(
            fs::read_to_string(folder.join("uploads/pending.txt")).unwrap(),
            "retry input"
        );
        let before = fs::read(folder.join("meta.json")).unwrap();
        rehydrate(&root, &Mutex::new(HashMap::new())).unwrap();
        assert_eq!(fs::read(folder.join("meta.json")).unwrap(), before);
    }

    #[test]
    fn cleanup_only_removes_expired_marked_private_stages() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("jobs");
        let _owner = service_lock(&root).unwrap();
        let marked = root.join(".upload-owned");
        let legacy = root.join(".upload-legacy");
        for stage in [&marked, &legacy] {
            private_dir(stage).unwrap();
        }
        mark_upload(&marked).unwrap();
        let now = std::time::SystemTime::now();
        clean_uploads_at(&root, now).unwrap();
        assert!(marked.exists());
        clean_uploads_at(&root, now + std::time::Duration::from_secs(25 * 60 * 60)).unwrap();
        assert!(!marked.exists());
        assert!(legacy.exists());
    }

    #[cfg(unix)]
    #[test]
    fn cleanup_keeps_a_marked_stage_containing_a_link_and_its_target() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("jobs");
        let _owner = service_lock(&root).unwrap();
        let stage = root.join(".upload-linked");
        private_dir(&stage).unwrap();
        mark_upload(&stage).unwrap();
        let outside = tmp.path().join("outside.txt");
        fs::write(&outside, "untouched").unwrap();
        std::os::unix::fs::symlink(&outside, stage.join("link")).unwrap();
        clean_uploads_at(
            &root,
            std::time::SystemTime::now() + std::time::Duration::from_secs(25 * 60 * 60),
        )
        .unwrap();
        assert!(stage.exists());
        assert_eq!(fs::read_to_string(outside).unwrap(), "untouched");
    }
}
