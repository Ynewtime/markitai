//! Filesystem container checks; semantic decoding is shared with ZIP packages.

use super::{Budget, MAX_ENTRIES, MAX_PACKAGE, MAX_PART, Result, error, finish, validate_iwa};
use std::fs::{self, File, Metadata, OpenOptions};
use std::io::Read;
use std::path::{Path, PathBuf};

const MAX_DEPTH: usize = 8;

struct Entry {
    name: String,
    metadata: Metadata,
}

struct Inventory {
    root: PathBuf,
    root_metadata: Metadata,
    entries: Vec<Entry>,
}

pub(super) fn open(path: &Path) -> Result<(iwork::Document, usize)> {
    let inventory = inventory(path)?;
    let mut entries = Vec::new();
    let mut budget = Budget::default();
    for entry in &inventory.entries {
        if entry.metadata.is_dir() || entry.name.rsplit('/').next() == Some(".DS_Store") {
            continue;
        }
        let data = read_entry(&inventory.root, entry)?;
        if entry.name.ends_with(".iwa") {
            validate_iwa(&data, &mut budget)?;
        }
        entries.push((entry.name.clone(), data));
    }
    // Catch ordinary edits and replacements during a read. This is not an
    // atomic snapshot against an adversary changing ancestor directories.
    verify_metadata(&inventory.root, &inventory.root_metadata)?;
    for entry in &inventory.entries {
        verify_metadata(&inventory.root.join(&entry.name), &entry.metadata)?;
    }
    finish(entries, iwork::package::Form::Directory, budget)
}

fn inventory(path: &Path) -> Result<Inventory> {
    // The public conversion boundary checks the supplied spelling against its
    // outer symlink policy. Internal package links are always rejected here.
    let root = path
        .canonicalize()
        .map_err(|_| error("package directory cannot be resolved"))?;
    let root_metadata =
        fs::symlink_metadata(&root).map_err(|_| error("package directory cannot be inspected"))?;
    if !root_metadata.is_dir() {
        return Err(error("package path is not a directory"));
    }
    let mut entries = Vec::new();
    let mut pending = vec![(String::new(), 0usize)];
    let mut total = 0u64;
    while let Some((relative, depth)) = pending.pop() {
        let directory = root.join(&relative);
        crate::output::check_path(&directory, false)
            .map_err(|_| error("package contains a symbolic link or changed directory"))?;
        let children = fs::read_dir(&directory)
            .map_err(|_| error("package directory cannot be enumerated"))?;
        for child in children {
            if entries.len() >= MAX_ENTRIES {
                return Err(error("directory package exceeds the 4096-node limit"));
            }
            if depth >= MAX_DEPTH {
                return Err(error("directory package exceeds the 8-level depth limit"));
            }
            let child = child.map_err(|_| error("package entry cannot be enumerated"))?;
            let component = child
                .file_name()
                .into_string()
                .map_err(|_| error("package entry name is not UTF-8"))?;
            if component.is_empty()
                || component == "."
                || component == ".."
                || component.contains(['/', '\\', '\0'])
            {
                return Err(error("invalid package entry name"));
            }
            let name = if relative.is_empty() {
                component.clone()
            } else {
                format!("{relative}/{component}")
            };
            if name.len() > 4096 {
                return Err(error("package entry name exceeds the size limit"));
            }
            let metadata = fs::symlink_metadata(child.path())
                .map_err(|_| error("package entry cannot be inspected"))?;
            if metadata.file_type().is_symlink() {
                return Err(error("directory package contains a symbolic link"));
            }
            if !metadata.is_file() && !metadata.is_dir() {
                return Err(error("directory package contains a non-regular entry"));
            }
            if name == ".iwpv2" {
                return Err(crate::Error::Unsupported(
                    "Encrypted Numbers documents are not supported".into(),
                ));
            }
            if metadata.is_dir() {
                pending.push((name.clone(), depth + 1));
            } else {
                if metadata.len() > MAX_PART as u64 {
                    return Err(error("directory package file exceeds the 32 MiB limit"));
                }
                total = total
                    .checked_add(metadata.len())
                    .filter(|total| *total <= MAX_PACKAGE as u64)
                    .ok_or_else(|| error("directory package exceeds the 128 MiB limit"))?;
            }
            entries.push(Entry { name, metadata });
        }
    }
    // Names are distinct paths, so the stable order is the only one.
    crate::sort::by(&mut entries, |left, right| left.name.cmp(&right.name));
    Ok(Inventory {
        root,
        root_metadata,
        entries,
    })
}

fn read_entry(root: &Path, entry: &Entry) -> Result<Vec<u8>> {
    let path = root.join(&entry.name);
    crate::output::check_path(&path, false)
        .map_err(|_| error("package contains a symbolic link or changed directory"))?;
    verify_metadata(&path, &entry.metadata)?;
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let mut file = options
        .open(&path)
        .map_err(|_| error("package file cannot be read"))?;
    verify_open_file(&file, &entry.metadata)?;
    let mut data = Vec::with_capacity(entry.metadata.len() as usize);
    (&mut file)
        .take(entry.metadata.len() + 1)
        .read_to_end(&mut data)
        .map_err(|_| error("package file cannot be read"))?;
    if data.len() as u64 != entry.metadata.len() {
        return Err(error("package changed while it was being read"));
    }
    verify_open_file(&file, &entry.metadata)?;
    verify_metadata(&path, &entry.metadata)?;
    Ok(data)
}

fn verify_open_file(file: &File, expected: &Metadata) -> Result<()> {
    let actual = file
        .metadata()
        .map_err(|_| error("package file cannot be inspected"))?;
    if !actual.is_file() || !same_metadata(expected, &actual) {
        return Err(error("package changed while it was being read"));
    }
    Ok(())
}

fn verify_metadata(path: &Path, expected: &Metadata) -> Result<()> {
    let actual =
        fs::symlink_metadata(path).map_err(|_| error("package changed while it was being read"))?;
    if !same_metadata(expected, &actual) {
        return Err(error("package changed while it was being read"));
    }
    Ok(())
}

fn same_metadata(expected: &Metadata, actual: &Metadata) -> bool {
    if expected.file_type() != actual.file_type()
        || expected.len() != actual.len()
        || expected.modified().ok() != actual.modified().ok()
    {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if expected.dev() != actual.dev()
            || expected.ino() != actual.ino()
            || expected.ctime() != actual.ctime()
            || expected.ctime_nsec() != actual.ctime_nsec()
        {
            return false;
        }
    }
    true
}

#[cfg(test)]
#[path = "directory_tests.rs"]
mod tests;
