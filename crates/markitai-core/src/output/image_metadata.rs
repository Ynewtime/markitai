//! Merge successful image descriptions while serializing cooperating processes.
//! Writers hold the sidecar `.images.lock`, never `images.json` itself: a
//! Windows lock is mandatory, and readers of the index must not be refused.
use super::*;
use crate::platform;
use std::fs::{File, OpenOptions, TryLockError};
use std::io::Read;
use std::time::{Duration, Instant};

const MAX_BYTES: u64 = 16 * 1024 * 1024;

fn invalid(message: &str) -> Error {
    Error::Conversion(message.into())
}

struct ImageMetadataLock(File);
impl Drop for ImageMetadataLock {
    fn drop(&mut self) {
        let _ = self.0.unlock();
    }
}

fn lock(directory: &Path, allow_symlinks: bool) -> Result<ImageMetadataLock> {
    let path = directory.join(".images.lock");
    check_path(&path, allow_symlinks)?;
    if std::fs::symlink_metadata(&path).is_ok_and(|metadata| metadata.file_type().is_symlink()) {
        return Err(invalid("Image metadata lock cannot be a symbolic link"));
    }
    let mut options = OpenOptions::new();
    options.read(true).write(true).create(true).truncate(false);
    platform::private_file(&mut options);
    let file = platform::open_no_follow(&options, &path)?;
    let opened = platform::file_status(&file)?;
    if !opened.metadata().is_file() {
        return Err(invalid("Image metadata lock is not a regular file"));
    }
    if opened.links() != 1 || !opened.private() {
        return Err(invalid(
            "Image metadata lock must be private and have one link",
        ));
    }
    let started = Instant::now();
    loop {
        match file.try_lock() {
            Ok(()) => break,
            Err(TryLockError::WouldBlock) if started.elapsed() < Duration::from_secs(5) => {
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(TryLockError::WouldBlock) => {
                return Err(invalid("Image metadata publication is busy"));
            }
            Err(TryLockError::Error(error)) => return Err(error.into()),
        }
    }
    let held = ImageMetadataLock(file);
    let current = platform::status(&path)?;
    if !current.metadata().is_file() || current.id() != opened.id() {
        return Err(invalid("Image metadata lock changed during publication"));
    }
    Ok(held)
}

fn read(path: &Path, allow_symlinks: bool) -> Result<Option<Value>> {
    check_path(path, allow_symlinks)?;
    let file = match platform::open_read(path, allow_symlinks) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.len() > MAX_BYTES {
        return Err(invalid(
            "Existing image metadata is not a regular file within the 16 MiB limit",
        ));
    }
    let mut bytes = Vec::new();
    file.take(MAX_BYTES + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > MAX_BYTES {
        return Err(invalid("Image metadata exceeds 16 MiB"));
    }
    let value: Value = serde_json::from_slice(&bytes)
        .map_err(|_| invalid("Existing image metadata is invalid JSON; its bytes were retained"))?;
    if !value.is_object() {
        return Err(invalid(
            "Existing image metadata must be an object; its bytes were retained",
        ));
    }
    for key in ["images", "assets"] {
        if value.get(key).is_some_and(|value| !value.is_array()) {
            return Err(invalid(
                "Existing image metadata entries must be an array; its bytes were retained",
            ));
        }
    }
    Ok(Some(value))
}

pub(super) fn publish(
    directory: &Path,
    entries: &[Value],
    source: &str,
    allow_symlinks: bool,
) -> Result<()> {
    if entries.is_empty() {
        return Ok(());
    }
    check_path(directory, allow_symlinks)?;
    std::fs::create_dir_all(directory)?;
    let _lock = lock(directory, allow_symlinks)?;
    let path = directory.join("images.json");
    let old = match read(&path, allow_symlinks)? {
        Some(old) => old,
        None => read(&directory.join("assets.json"), allow_symlinks)?.unwrap_or_else(|| json!({})),
    };
    let existing = old
        .get("images")
        .and_then(Value::as_array)
        .filter(|items| !items.is_empty())
        .or_else(|| old.get("assets").and_then(Value::as_array));
    let mut images = Vec::<Value>::new();
    let mut indexes = HashMap::<String, usize>::new();
    if let Some(existing) = existing {
        for entry in existing {
            let Some(key) = entry
                .get("path")
                .or_else(|| entry.get("asset"))
                .and_then(Value::as_str)
                .filter(|key| !key.is_empty())
            else {
                continue;
            };
            let mut entry = entry.clone();
            entry["path"] = key.into();
            let key = key.to_owned();
            entry.as_object_mut().unwrap().remove("asset");
            if let Some(index) = indexes.get(&key) {
                images[*index] = entry;
            } else {
                indexes.insert(key, images.len());
                images.push(entry);
            }
        }
    }
    for entry in entries {
        let Some(key) = entry
            .get("asset")
            .and_then(Value::as_str)
            .filter(|key| !key.is_empty())
        else {
            return Err(invalid("Image metadata has no published asset path"));
        };
        let mut entry = entry.clone();
        entry["path"] = key.into();
        entry["source"] = source.into();
        let map = entry.as_object_mut().unwrap();
        map.remove("asset");
        map.remove("llm_usage");
        if let Some(index) = indexes.get(key) {
            images[*index] = entry;
        } else {
            indexes.insert(key.into(), images.len());
            images.push(entry);
        }
    }
    let now = Local::now().to_rfc3339_opts(SecondsFormat::Millis, false);
    #[derive(serde::Serialize)]
    struct Index<'a> {
        version: &'a str,
        created: Value,
        updated: &'a str,
        images: Vec<Value>,
    }
    let index = Index {
        version: "1.0",
        created: old
            .get("created")
            .cloned()
            .unwrap_or_else(|| now.clone().into()),
        updated: &now,
        images,
    };
    let mut bytes = serde_json::to_vec_pretty(&index)?;
    bytes.push(b'\n');
    if bytes.len() as u64 > MAX_BYTES {
        return Err(invalid(
            "Merged image metadata exceeds 16 MiB; previous bytes were retained",
        ));
    }
    check_path(&path, allow_symlinks)?;
    atomic_write(&path, &bytes, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_merge_preserves_other_entries_and_creation_without_internal_usage() {
        let dir = tempfile::tempdir().unwrap();
        let legacy = dir.path().join("assets.json");
        let old = br#"{"created":"first","assets":[{"asset":"one.png","desc":"older"},{"asset":"two.png","custom":true}]}"#;
        std::fs::write(&legacy, old).unwrap();
        publish(dir.path(), &[json!({"asset":"one.png","alt":"new","desc":"description","text":"letters","created":"entry","llm_usage":{"mock":{"requests":1}}})], "source.pdf", false).unwrap();
        let result: Value =
            serde_json::from_slice(&std::fs::read(dir.path().join("images.json")).unwrap())
                .unwrap();
        assert_eq!(result["version"], "1.0");
        assert_eq!(result["created"], "first");
        assert_eq!(result["images"].as_array().unwrap().len(), 2);
        assert_eq!(result["images"][0]["source"], "source.pdf");
        assert_eq!(result["images"][0]["path"], "one.png");
        assert!(result["images"][0].get("llm_usage").is_none());
        assert!(result["images"][0].get("asset").is_none());
        assert_eq!(result["images"][1]["custom"], true);
        assert_eq!(std::fs::read(legacy).unwrap(), old);
    }

    #[test]
    fn corrupt_metadata_is_retained_on_failed_merge() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("images.json");
        for bytes in [
            b"user-maintained malformed metadata".as_slice(),
            br#"{"images":{}}"#,
        ] {
            std::fs::write(&path, bytes).unwrap();
            assert!(publish(dir.path(), &[json!({"asset":"a.png"})], "source", false).is_err());
            assert_eq!(std::fs::read(&path).unwrap(), bytes);
        }
    }

    #[test]
    fn concurrent_processes_merge_without_lost_entries() {
        const CHILD: &str = "MARKITAI_TEST_IMAGE_METADATA_CHILD";
        if let Ok(directory) = std::env::var(CHILD) {
            let id = std::env::var("MARKITAI_TEST_IMAGE_METADATA_ID").unwrap();
            let ready = Path::new(&directory).join(format!("ready-{id}"));
            std::fs::write(ready, []).unwrap();
            let started = Instant::now();
            while !Path::new(&directory).join("go").exists() {
                assert!(started.elapsed() < Duration::from_secs(10));
                std::thread::sleep(Duration::from_millis(5));
            }
            for index in 0..12 {
                publish(
                    Path::new(&directory),
                    &[json!({"asset":format!("{id}-{index}.png"),"desc":id})],
                    &id,
                    false,
                )
                .unwrap();
            }
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let mut children = Vec::new();
        for id in ["left", "right"] {
            children.push(std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "output::image_metadata::tests::concurrent_processes_merge_without_lost_entries", "--nocapture"])
                .env(CHILD, dir.path()).env("MARKITAI_TEST_IMAGE_METADATA_ID", id)
                .stdout(std::process::Stdio::null()).spawn().unwrap());
        }
        let started = Instant::now();
        while !["left", "right"]
            .iter()
            .all(|id| dir.path().join(format!("ready-{id}")).exists())
        {
            assert!(started.elapsed() < Duration::from_secs(10));
            std::thread::sleep(Duration::from_millis(5));
        }
        std::fs::write(dir.path().join("go"), []).unwrap();
        for child in &mut children {
            assert!(child.wait().unwrap().success());
        }
        let value: Value =
            serde_json::from_slice(&std::fs::read(dir.path().join("images.json")).unwrap())
                .unwrap();
        assert_eq!(value["images"].as_array().unwrap().len(), 24);
    }

    #[test]
    #[cfg(unix)]
    fn symlink_metadata_or_lock_never_redirects_a_write() {
        use std::os::unix::fs::symlink;
        let dir = tempfile::tempdir().unwrap();
        let outside = dir.path().join("outside");
        std::fs::write(&outside, b"protected").unwrap();
        for target in ["images.json", ".images.lock"] {
            let folder = dir.path().join(target.replace('.', "-"));
            std::fs::create_dir(&folder).unwrap();
            symlink(&outside, folder.join(target)).unwrap();
            assert!(publish(&folder, &[json!({"asset":"a.png"})], "source", false).is_err());
            assert_eq!(std::fs::read(&outside).unwrap(), b"protected");
        }
    }
}
