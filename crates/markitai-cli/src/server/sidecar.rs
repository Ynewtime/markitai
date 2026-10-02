//! Rebase staged image metadata and preserve sibling entries under the core lock.
//! Writers hold the sidecar `.images.lock`, never `images.json`, so a Windows
//! (mandatory) lock never refuses a reader of the index.
use super::store;
use markitai_core::platform;
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    fs::{self, File, OpenOptions, TryLockError},
    io::{self, Read, Write},
    path::Path,
    time::{Duration, Instant},
};
const LIMIT: u64 = 16 * 1024 * 1024;
fn read(path: &Path) -> io::Result<Value> {
    markitai_core::output::check_path(path, false).map_err(io::Error::other)?;
    match fs::symlink_metadata(path) {
        Ok(metadata) if !metadata.is_file() => {
            return Err(io::Error::other("image metadata is not a regular file"));
        }
        Ok(_) => {}
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(json!({})),
        Err(e) => return Err(e),
    }
    let file = match platform::open_read(path, false) {
        Ok(file) => file,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(json!({})),
        Err(e) => return Err(e),
    };
    if !file.metadata()?.is_file() || file.metadata()?.len() > LIMIT {
        return Err(io::Error::other("invalid image metadata size"));
    }
    let mut bytes = Vec::new();
    file.take(LIMIT + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > LIMIT {
        return Err(io::Error::other("image metadata exceeds limit"));
    }
    let value: Value = serde_json::from_slice(&bytes).map_err(io::Error::other)?;
    if !value.is_object()
        || ["images", "assets"]
            .iter()
            .any(|key| value.get(key).is_some_and(|v| !v.is_array()))
    {
        return Err(io::Error::other("invalid image metadata shape"));
    }
    Ok(value)
}
pub(super) struct ImageMetadataLock(File);
impl Drop for ImageMetadataLock {
    fn drop(&mut self) {
        let _ = self.0.unlock();
    }
}

fn lock(directory: &Path) -> io::Result<ImageMetadataLock> {
    store::private_dir(directory)?;
    let path = directory.join(".images.lock");
    markitai_core::output::check_path(&path, false).map_err(io::Error::other)?;
    let mut options = OpenOptions::new();
    options.read(true).write(true).create(true).truncate(false);
    platform::private_file(&mut options);
    let file = platform::open_no_follow(&options, &path)?;
    let opened = platform::file_status(&file)?;
    if !opened.metadata().is_file() {
        return Err(io::Error::other("invalid image metadata lock"));
    }
    if opened.links() != 1 || !opened.private() {
        return Err(io::Error::other("image metadata lock is not private"));
    }
    let start = Instant::now();
    loop {
        match file.try_lock() {
            Ok(()) => break,
            Err(TryLockError::WouldBlock) if start.elapsed() < Duration::from_secs(5) => {
                std::thread::sleep(Duration::from_millis(10))
            }
            Err(TryLockError::WouldBlock) => {
                return Err(io::Error::other("image metadata is busy"));
            }
            Err(TryLockError::Error(e)) => return Err(e),
        }
    }
    let held = ImageMetadataLock(file);
    let current = platform::status(&path)?;
    if !current.metadata().is_file() || current.id() != opened.id() {
        return Err(io::Error::other("image metadata lock changed"));
    }
    Ok(held)
}
fn rows(value: &Value) -> impl Iterator<Item = &Value> {
    value
        .get("images")
        .and_then(Value::as_array)
        .filter(|v| !v.is_empty())
        .or_else(|| value.get("assets").and_then(Value::as_array))
        .into_iter()
        .flatten()
}
pub(super) fn prepare(staged: &Path, final_out: &Path) -> io::Result<Vec<ImageMetadataLock>> {
    let mut locks = Vec::new();
    for prefix in [".markitai/assets", "assets"] {
        let path = staged.join(prefix).join("images.json");
        if !path.try_exists()? {
            continue;
        }
        let new = read(&path)?;
        let directory = final_out.join(prefix);
        locks.push(lock(&directory)?);
        let old_path = directory.join("images.json");
        let old = if old_path.try_exists()? {
            read(&old_path)?
        } else {
            read(&directory.join("assets.json"))?
        };
        let mut images = Vec::new();
        let mut indexes = HashMap::new();
        for row in rows(&old) {
            let Some(path) = row
                .get("path")
                .or_else(|| row.get("asset"))
                .and_then(Value::as_str)
            else {
                continue;
            };
            let mut row = row.clone();
            row["path"] = json!(path);
            row.as_object_mut()
                .ok_or_else(|| io::Error::other("invalid image metadata row"))?
                .remove("asset");
            if let Some(index) = indexes.get(path) {
                images[*index] = row;
            } else {
                indexes.insert(path.to_owned(), images.len());
                images.push(row);
            }
        }
        for row in rows(&new) {
            let raw = row
                .get("path")
                .and_then(Value::as_str)
                .ok_or_else(|| io::Error::other("image metadata lacks path"))?;
            let relative = Path::new(raw)
                .strip_prefix(staged)
                .map_err(|_| io::Error::other("staged image metadata path escaped output"))?;
            let final_path = final_out.join(relative).to_string_lossy().into_owned();
            let mut row = row.clone();
            row["path"] = json!(final_path);
            if let Some(index) = indexes.get(&final_path) {
                images[*index] = row;
            } else {
                indexes.insert(final_path, images.len());
                images.push(row);
            }
        }
        let value = json!({"version":"1.0","created":old.get("created").or_else(||new.get("created")),"updated":new.get("updated"),"images":images});
        let bytes = serde_json::to_vec_pretty(&value).map_err(io::Error::other)?;
        if bytes.len() as u64 > LIMIT {
            return Err(io::Error::other("merged image metadata exceeds limit"));
        }
        let mut file = tempfile::NamedTempFile::new_in(path.parent().unwrap())?;
        file.write_all(&bytes)?;
        file.write_all(b"\n")?;
        file.as_file().sync_all()?;
        platform::persist(file, &path).map_err(|e| e.error)?;
    }
    Ok(locks)
}

/// Stage index pruning for files that this deletion will actually remove. A
/// shared asset is absent from `removed`, so its metadata survives unchanged.
pub(super) fn prune(
    staged: &Path,
    final_out: &Path,
    removed: &std::collections::HashSet<String>,
) -> io::Result<Vec<ImageMetadataLock>> {
    let mut locks = Vec::new();
    for prefix in [".markitai/assets", "assets"] {
        let directory = final_out.join(prefix);
        let path = directory.join("images.json");
        if !path.try_exists()? {
            continue;
        }
        locks.push(lock(&directory)?);
        let mut value = read(&path)?;
        let mut changed = false;
        for key in ["images", "assets"] {
            if let Some(rows) = value.get_mut(key).and_then(Value::as_array_mut) {
                rows.retain(|row| {
                    let delete = row
                        .get("path")
                        .or_else(|| row.get("asset"))
                        .and_then(Value::as_str)
                        .is_some_and(|raw| {
                            let path = Path::new(raw);
                            let relative = if path.is_absolute() {
                                path.strip_prefix(final_out).ok()
                            } else {
                                Some(path)
                            };
                            relative.is_some_and(|relative| {
                                removed.contains(relative.to_string_lossy().as_ref())
                            })
                        });
                    changed |= delete;
                    !delete
                });
            }
        }
        if changed {
            let target = staged.join(prefix).join("images.json");
            store::private_dir(target.parent().unwrap())?;
            let bytes = serde_json::to_vec_pretty(&value).map_err(io::Error::other)?;
            if bytes.len() as u64 > LIMIT {
                return Err(io::Error::other("image metadata exceeds limit"));
            }
            let mut file = tempfile::NamedTempFile::new_in(target.parent().unwrap())?;
            file.write_all(&bytes)?;
            file.write_all(b"\n")?;
            file.as_file().sync_all()?;
            platform::persist(file, &target).map_err(|e| e.error)?;
        }
    }
    Ok(locks)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn staged_absolute_paths_become_final_and_other_items_survive() {
        let root = tempfile::tempdir().unwrap();
        let stage = root.path().join("stage");
        let final_out = root.path().join("out");
        fs::create_dir_all(stage.join("assets")).unwrap();
        fs::create_dir_all(final_out.join("assets")).unwrap();
        let old = json!({"created":"first","images":[{"path":final_out.join("assets/other.png"),"desc":"other item"}]});
        fs::write(final_out.join("assets/images.json"), old.to_string()).unwrap();
        fs::write(stage.join("assets/images.json"),json!({"created":"new","updated":"now","images":[{"path":stage.join("assets/a.png"),"desc":"fresh"}]}).to_string()).unwrap();
        let _locks = prepare(&stage, &final_out).unwrap();
        let merged = read(&stage.join("assets/images.json")).unwrap();
        assert_eq!(merged["created"], "first");
        assert_eq!(merged["images"][0], old["images"][0]);
        assert_eq!(
            merged["images"][1]["path"],
            final_out.join("assets/a.png").to_string_lossy().as_ref()
        );
        assert!(!merged.to_string().contains("/stage/"));
        assert!(!super::super::files::public_member("assets/.images.lock"));
    }
    #[test]
    fn corrupt_final_metadata_is_not_overwritten() {
        let root = tempfile::tempdir().unwrap();
        let stage = root.path().join("stage");
        let out = root.path().join("out");
        fs::create_dir_all(stage.join("assets")).unwrap();
        fs::create_dir_all(out.join("assets")).unwrap();
        fs::write(
            stage.join("assets/images.json"),
            json!({"images":[]}).to_string(),
        )
        .unwrap();
        fs::write(out.join("assets/images.json"), b"bad JSON").unwrap();
        assert!(prepare(&stage, &out).is_err());
        assert_eq!(
            fs::read(out.join("assets/images.json")).unwrap(),
            b"bad JSON"
        );
    }
}
