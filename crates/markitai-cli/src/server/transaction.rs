//! Reversible file publication. The metadata commit decides recovery after restart.
//!
//! Directory synchronization below is `fsync` on Unix. Windows cannot flush a
//! directory: each published file is flushed after its rename instead, which
//! commits the NTFS log records of every earlier rename and removal too.
use super::{jobs::JobData, store};
use markitai_core::platform;
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File},
    io::{self, Read, Write},
    path::{Component, Path},
};

const LIMIT: u64 = 5 * 1024 * 1024 * 1024;
#[derive(Serialize, Deserialize)]
struct Entry {
    path: String,
    previous: bool,
}
#[derive(Serialize, Deserialize)]
struct Journal {
    id: String,
    sequence: u64,
    entries: Vec<Entry>,
}

#[derive(Debug)]
struct RecoveryRequired(io::Error);
impl std::fmt::Display for RecoveryRequired {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "output recovery is required: {}", self.0)
    }
}
impl std::error::Error for RecoveryRequired {}
pub(super) fn requires_recovery(error: &io::Error) -> bool {
    error.get_ref().is_some_and(|e| e.is::<RecoveryRequired>())
}
fn valid(path: &str) -> bool {
    !path.is_empty()
        && !path.contains('\\')
        && Path::new(path)
            .components()
            .all(|p| matches!(p, Component::Normal(_)))
        && (path.starts_with("out/") || path.starts_with("uploads/") || path == "archive.zip")
}
fn copy(source: &Path, target: &Path, total: &mut u64) -> io::Result<()> {
    markitai_core::output::check_path(source, false).map_err(io::Error::other)?;
    if !fs::metadata(source)?.is_file() {
        return Err(io::Error::other("nonregular transaction member"));
    }
    let input = File::open(source)?;
    let size = input.metadata()?.len();
    *total = total
        .checked_add(size)
        .filter(|v| *v <= LIMIT)
        .ok_or_else(|| io::Error::other("transaction exceeds size limit"))?;
    let parent = target
        .parent()
        .ok_or_else(|| io::Error::other("invalid transaction path"))?;
    store::private_dir(parent)?;
    let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
    let count = io::copy(&mut input.take(size + 1), &mut temporary)?;
    if count != size {
        return Err(io::Error::other("transaction member changed while copying"));
    }
    temporary.as_file().sync_all()?;
    let installed = platform::persist(temporary, target).map_err(|e| e.error)?;
    platform::sync_renamed(&installed)?;
    platform::sync_directory(parent)
}
fn sync_chain(mut path: &Path, root: &Path) -> io::Result<()> {
    loop {
        platform::sync_directory(path)?;
        if path == root {
            return Ok(());
        }
        path = path
            .parent()
            .filter(|p| p.starts_with(root))
            .ok_or_else(|| io::Error::other("transaction directory escaped job"))?;
    }
}
fn rollback(folder: &Path, stage: &Path, journal: &Journal) -> io::Result<()> {
    let mut total = 0;
    for (index, entry) in journal.entries.iter().enumerate().rev() {
        if !valid(&entry.path) {
            return Err(io::Error::other("invalid transaction member"));
        }
        let target = folder.join(&entry.path);
        markitai_core::output::check_path(&target, false).map_err(io::Error::other)?;
        if entry.previous {
            copy(
                &stage.join("backup").join(index.to_string()),
                &target,
                &mut total,
            )?;
        } else {
            match fs::remove_file(&target) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::NotFound => {}
                Err(e) => return Err(e),
            }
        }
    }
    platform::sync_directory(folder)
}

/// Caller holds job.access; staged replacements have paths relative to the job.
/// A persisted native_transactions ID commits this journal. Until then restart
/// restores every prior file, including assets and absent Markdown variants.
pub(super) fn publish(
    folder: &Path,
    stage: tempfile::TempDir,
    sequence: u64,
    replacements: Vec<(String, std::path::PathBuf)>,
    removals: Vec<String>,
) -> io::Result<String> {
    let id = stage
        .path()
        .file_name()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    let mut entries = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut total = 0;
    for name in replacements.iter().map(|v| &v.0).chain(removals.iter()) {
        if !valid(name) || !seen.insert(name.clone()) {
            return Err(io::Error::other("invalid or repeated transaction member"));
        }
        let target = folder.join(name);
        markitai_core::output::check_path(&target, false).map_err(io::Error::other)?;
        let previous = target.try_exists()?;
        if previous {
            copy(
                &target,
                &stage.path().join("backup").join(entries.len().to_string()),
                &mut total,
            )?;
        }
        entries.push(Entry {
            path: name.clone(),
            previous,
        });
    }
    if entries.len() > 100_000 {
        return Err(io::Error::other("transaction has too many members"));
    }
    let journal = Journal {
        id: id.clone(),
        sequence,
        entries,
    };
    let encoded = serde_json::to_vec(&journal).map_err(io::Error::other)?;
    if encoded.len() > 16 * 1024 * 1024 {
        return Err(io::Error::other("transaction metadata exceeds limit"));
    }
    let mut manifest = tempfile::NamedTempFile::new_in(stage.path())?;
    manifest.write_all(&encoded)?;
    manifest.as_file().sync_all()?;
    let recorded =
        platform::persist(manifest, &stage.path().join("journal.json")).map_err(|e| e.error)?;
    platform::sync_renamed(&recorded)?;
    drop(recorded);
    platform::sync_directory(stage.path())?;
    let path = stage.keep();
    platform::sync_directory(folder)?;
    let result = (|| {
        for (name, source) in replacements {
            let target = folder.join(name);
            let parent = target.parent().unwrap();
            store::private_dir(parent)?;
            platform::sync_file(&source)?;
            platform::rename(&source, &target)?;
            platform::sync_renamed_path(&target)?;
            sync_chain(parent, folder)?;
        }
        for name in removals {
            let target = folder.join(name);
            if target.try_exists()? {
                fs::remove_file(&target)?;
                platform::sync_directory(target.parent().unwrap())?;
            }
        }
        Ok::<_, io::Error>(())
    })();
    if let Err(error) = result {
        rollback(folder, &path, &journal).map_err(|e| io::Error::other(RecoveryRequired(e)))?;
        fs::remove_dir_all(path)?;
        return Err(error);
    }
    Ok(id)
}

pub(super) fn stage(folder: &Path) -> io::Result<tempfile::TempDir> {
    let stage = tempfile::Builder::new()
        .prefix(".retry-")
        .tempdir_in(folder)?;
    store::private_dir(stage.path())?;
    Ok(stage)
}

pub(super) fn clean(folder: &Path, ids: &[String]) -> io::Result<()> {
    for id in ids {
        if !id.starts_with(".retry-") || Path::new(id).components().count() != 1 {
            return Err(io::Error::other("invalid transaction identifier"));
        }
        let path = folder.join(id);
        markitai_core::output::check_path(&path, false).map_err(io::Error::other)?;
        match fs::remove_dir_all(path) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(e),
        }
    }
    platform::sync_directory(folder)
}

pub(super) fn recover(folder: &Path) -> io::Result<()> {
    markitai_core::output::check_path(folder, false).map_err(io::Error::other)?;
    let committed = store::safe_file(folder, "meta.json")
        .ok()
        .and_then(|p| (fs::metadata(&p).ok()?.len() <= 16 * 1024 * 1024).then_some(p))
        .and_then(|p| fs::read(p).ok())
        .and_then(|v| serde_json::from_slice::<serde_json::Value>(&v).ok())
        .and_then(|v| serde_json::from_value::<Vec<String>>(v["native_transactions"].clone()).ok())
        .unwrap_or_default();
    let mut pending = Vec::new();
    for entry in fs::read_dir(folder)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name.starts_with(".retry-") {
            continue;
        }
        let stage = entry.path();
        markitai_core::output::check_path(&stage, false).map_err(io::Error::other)?;
        let journal = stage.join("journal.json");
        if !journal.try_exists()? {
            fs::remove_dir_all(stage)?;
            continue;
        }
        markitai_core::output::check_path(&journal, false).map_err(io::Error::other)?;
        if fs::metadata(&journal)?.len() > 16 * 1024 * 1024 {
            return Err(io::Error::other("transaction metadata exceeds limit"));
        }
        let record: Journal =
            serde_json::from_slice(&fs::read(journal)?).map_err(io::Error::other)?;
        if record.id != name || record.entries.len() > 100_000 {
            return Err(io::Error::other("invalid transaction journal"));
        }
        pending.push((stage, record));
    }
    crate::sort::by_key(&mut pending, |(_, record)| {
        std::cmp::Reverse(record.sequence)
    });
    for (stage, record) in pending {
        if !committed.contains(&record.id) {
            rollback(folder, &stage, &record)?;
        }
        fs::remove_dir_all(stage)?;
    }
    Ok(())
}

pub(super) fn committed(folder: &Path, data: &mut JobData) {
    if clean(folder, &data.transactions).is_ok() {
        data.transactions.clear();
    }
}

pub(super) fn abort(folder: &Path, id: &str) -> io::Result<()> {
    if !id.starts_with(".retry-") || Path::new(id).components().count() != 1 {
        return Err(io::Error::other("invalid transaction identifier"));
    }
    let stage = folder.join(id);
    let path = stage.join("journal.json");
    markitai_core::output::check_path(&path, false).map_err(io::Error::other)?;
    if fs::metadata(&path)?.len() > 16 * 1024 * 1024 {
        return Err(io::Error::other("transaction metadata exceeds limit"));
    }
    let journal: Journal = serde_json::from_slice(&fs::read(path)?).map_err(io::Error::other)?;
    if journal.id != id {
        return Err(io::Error::other("invalid transaction identifier"));
    }
    rollback(folder, &stage, &journal)?;
    fs::remove_dir_all(stage)?;
    platform::sync_directory(folder)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> tempfile::TempDir {
        let root = tempfile::tempdir().unwrap();
        store::private_dir(&root.path().join("out")).unwrap();
        fs::write(root.path().join("out/document.md"), b"old body").unwrap();
        root
    }
    fn replacement(root: &Path, value: &[u8], sequence: u64) -> String {
        let staged = stage(root).unwrap();
        let path = staged.path().join("new.md");
        fs::write(&path, value).unwrap();
        publish(
            root,
            staged,
            sequence,
            vec![("out/document.md".into(), path)],
            vec![],
        )
        .unwrap()
    }
    #[test]
    fn restart_restores_published_but_uncommitted_bytes_in_reverse_order() {
        let root = fixture();
        replacement(root.path(), b"second", 1);
        replacement(root.path(), b"third", 2);
        assert_eq!(
            fs::read(root.path().join("out/document.md")).unwrap(),
            b"third"
        );
        recover(root.path()).unwrap();
        assert_eq!(
            fs::read(root.path().join("out/document.md")).unwrap(),
            b"old body"
        );
        assert_eq!(fs::read_dir(root.path()).unwrap().count(), 1);
    }
    #[test]
    fn metadata_commit_preserves_new_bytes_and_removes_recovery_material() {
        let root = fixture();
        let id = replacement(root.path(), b"new body", 1);
        fs::write(
            root.path().join("meta.json"),
            serde_json::json!({"native_transactions":[id]}).to_string(),
        )
        .unwrap();
        recover(root.path()).unwrap();
        assert_eq!(
            fs::read(root.path().join("out/document.md")).unwrap(),
            b"new body"
        );
        assert_eq!(fs::read_dir(root.path()).unwrap().count(), 2);
    }
    #[test]
    fn delete_recovery_restores_upload_and_shared_output_bytes() {
        let root = fixture();
        store::private_dir(&root.path().join("uploads")).unwrap();
        fs::write(root.path().join("uploads/source.txt"), b"source").unwrap();
        publish(
            root.path(),
            stage(root.path()).unwrap(),
            0,
            vec![],
            vec!["out/document.md".into(), "uploads/source.txt".into()],
        )
        .unwrap();
        assert!(!root.path().join("out/document.md").exists());
        recover(root.path()).unwrap();
        assert_eq!(
            fs::read(root.path().join("uploads/source.txt")).unwrap(),
            b"source"
        );
        assert_eq!(
            fs::read(root.path().join("out/document.md")).unwrap(),
            b"old body"
        );
    }
    #[test]
    fn invalid_member_cannot_publish_outside_the_job() {
        let root = fixture();
        let staged = stage(root.path()).unwrap();
        let path = staged.path().join("new");
        fs::write(&path, b"new").unwrap();
        assert!(
            publish(
                root.path(),
                staged,
                0,
                vec![("out/../escape".into(), path)],
                vec![]
            )
            .is_err()
        );
        assert_eq!(
            fs::read(root.path().join("out/document.md")).unwrap(),
            b"old body"
        );
        assert!(!root.path().join("escape").exists());
    }
}
