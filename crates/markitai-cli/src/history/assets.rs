use caseless::default_case_fold_str as fold;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::{self, File, Metadata, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

const CHUNK: usize = 64 * 1024;
const ASSET_DIRS: [&str; 3] = [".markitai/assets", ".markitai/screenshots", "assets"];

#[derive(Clone, Copy)]
struct Limits {
    file_bytes: u64,
    archive_bytes: u64,
    entries: usize,
    depth: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            file_bytes: 256 * 1024 * 1024,
            archive_bytes: 1024 * 1024 * 1024,
            entries: 100_000,
            depth: 64,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Fingerprint {
    bytes: u64,
    sha256: [u8; 32],
}

#[derive(Clone)]
struct Name {
    spelling: String,
    directory: bool,
}

#[derive(Default)]
struct Directory {
    names: HashMap<String, Name>,
    next_suffix: HashMap<String, u64>,
    // Hashes narrow candidate lookup; an actual byte comparison proves reuse.
    variants: HashMap<(String, Fingerprint), Vec<String>>,
}

#[derive(Default)]
pub(super) struct Budget {
    limits: Limits,
    bytes: u64,
    files: usize,
    visited: usize,
    directories: HashMap<PathBuf, Directory>,
    fingerprints: HashMap<PathBuf, Fingerprint>,
    image_indexes: Vec<(PathBuf, PathBuf)>,
}

impl Budget {
    pub(super) fn take_image_indexes(&mut self) -> Vec<(PathBuf, PathBuf)> {
        std::mem::take(&mut self.image_indexes)
    }

    pub(super) fn bytes(&self) -> u64 {
        self.bytes
    }

    pub(super) fn replace_bytes(&mut self, old: u64, new: u64) -> io::Result<()> {
        let total = self
            .bytes
            .checked_sub(old)
            .and_then(|value| value.checked_add(new))
            .ok_or_else(limit_error)?;
        if new > self.limits.file_bytes || total > self.limits.archive_bytes {
            return Err(limit_error());
        }
        self.bytes = total;
        Ok(())
    }

    fn admit(&self, bytes: u64) -> io::Result<()> {
        if bytes > self.limits.file_bytes
            || self.files >= self.limits.entries
            || self
                .bytes
                .checked_add(bytes)
                .is_none_or(|total| total > self.limits.archive_bytes)
        {
            return Err(limit_error());
        }
        Ok(())
    }

    fn visit(&mut self) -> io::Result<()> {
        if self.visited >= self.limits.entries {
            return Err(limit_error());
        }
        self.visited += 1;
        Ok(())
    }
}

fn limit_error() -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, "History archive limit exceeded")
}

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn policy(path: &Path, allow_symlinks: bool) -> io::Result<()> {
    markitai_core::output::check_path(path, allow_symlinks)
        .map_err(|_| invalid("History path violates the symlink policy"))
}

fn regular(metadata: &Metadata) -> io::Result<()> {
    if !metadata.file_type().is_file() {
        return Err(invalid("History source must be a regular file"));
    }
    Ok(())
}

fn unchanged(before: &Metadata, after: &Metadata) -> bool {
    let common = before.file_type() == after.file_type()
        && before.len() == after.len()
        && before.modified().ok() == after.modified().ok();
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        common
            && before.dev() == after.dev()
            && before.ino() == after.ino()
            && before.ctime() == after.ctime()
            && before.ctime_nsec() == after.ctime_nsec()
    }
    #[cfg(not(unix))]
    {
        common
    }
}

fn open_source(path: &Path, limit: u64, allow_symlinks: bool) -> io::Result<(File, Metadata)> {
    policy(path, allow_symlinks)?;
    let before = fs::symlink_metadata(path)?;
    regular(&before)?;
    if before.len() > limit {
        return Err(limit_error());
    }
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let file = options.open(path)?;
    let opened = file.metadata()?;
    regular(&opened)?;
    if !unchanged(&before, &opened) {
        return Err(invalid("History source changed while opening"));
    }
    Ok((file, opened))
}

fn confirm(path: &Path, file: &File, before: &Metadata) -> io::Result<()> {
    if !unchanged(before, &file.metadata()?) || !unchanged(before, &fs::symlink_metadata(path)?) {
        return Err(invalid("History source changed while copying"));
    }
    Ok(())
}

fn stream(
    input: &mut File,
    mut output: Option<&mut File>,
    expected: u64,
) -> io::Result<Fingerprint> {
    let mut buffer = [0_u8; CHUNK];
    let mut bytes = 0_u64;
    let mut hash = Sha256::new();
    loop {
        // Read at most the observed length plus one byte to detect growth.
        let read_limit = (expected - bytes).min(CHUNK as u64 - 1) as usize + 1;
        let count = input.read(&mut buffer[..read_limit])?;
        if count == 0 {
            break;
        }
        bytes += count as u64;
        if bytes > expected {
            return Err(invalid("History source grew while copying"));
        }
        if let Some(file) = output.as_deref_mut() {
            file.write_all(&buffer[..count])?;
        }
        hash.update(&buffer[..count]);
    }
    if bytes != expected {
        return Err(invalid("History source shrank while copying"));
    }
    Ok(Fingerprint {
        bytes,
        sha256: hash.finalize().into(),
    })
}

/// The caller owns the private staging directory. Existing targets are never removed.
pub(super) fn copy_file(
    source: &Path,
    target: &Path,
    budget: &mut Budget,
    allow_symlinks: bool,
) -> io::Result<()> {
    let (input, before) = open_source(source, budget.limits.file_bytes, allow_symlinks)?;
    copy_opened(source, input, before, target, budget, allow_symlinks)
}

fn copy_opened(
    source: &Path,
    mut input: File,
    before: Metadata,
    target: &Path,
    budget: &mut Budget,
    allow_symlinks: bool,
) -> io::Result<()> {
    budget.admit(before.len())?;
    policy(target, allow_symlinks)?;
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600).custom_flags(libc::O_NOFOLLOW);
    }
    let mut output = options.open(target)?;
    let result = (|| {
        let fingerprint = stream(&mut input, Some(&mut output), before.len())?;
        confirm(source, &input, &before)?;
        // Synchronized with the whole stage before the job is named.
        Ok(fingerprint)
    })();
    drop(output);
    let fingerprint = match result {
        Ok(value) => value,
        Err(error) => {
            let _ = fs::remove_file(target);
            return Err(error);
        }
    };
    budget.bytes += fingerprint.bytes;
    budget.files += 1;
    budget.fingerprints.insert(target.to_owned(), fingerprint);
    if let (Some(parent), Some(name)) =
        (target.parent(), target.file_name().and_then(|v| v.to_str()))
        && let Some(index) = budget.directories.get_mut(parent)
    {
        index.names.insert(
            fold(name),
            Name {
                spelling: name.to_owned(),
                directory: false,
            },
        );
    }
    Ok(())
}

fn directory(path: &Path, allow_symlinks: bool) -> io::Result<bool> {
    policy(path, allow_symlinks)?;
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_dir() => Ok(true),
        Ok(_) => Err(invalid(
            "History asset directory is not a regular directory",
        )),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error),
    }
}

pub(super) fn find_root(
    output: &Path,
    ascend: usize,
    boundary: &Path,
    allow_symlinks: bool,
) -> io::Result<Option<PathBuf>> {
    let Some(parent) = output.parent() else {
        return Ok(None);
    };
    for path in [parent, boundary] {
        // These paths are ancestors of the output, not selected asset leaves.
        policy(path, allow_symlinks)?;
        match fs::metadata(path) {
            Ok(metadata) if metadata.is_dir() => {}
            Ok(_) => return Err(invalid("History output boundary is not a directory")),
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error),
        }
    }
    let boundary = boundary.canonicalize()?;
    let mut level = parent.canonicalize()?;
    if !level.starts_with(&boundary) {
        return Ok(None);
    }
    for _ in 0..=ascend.min(Limits::default().depth) {
        if directory(&level.join(".markitai"), allow_symlinks)?
            || directory(&level.join("assets"), allow_symlinks)?
        {
            return Ok(Some(level));
        }
        if level == boundary || !level.pop() || !level.starts_with(&boundary) {
            break;
        }
    }
    Ok(None)
}

fn names(path: &Path, budget: &mut Budget) -> io::Result<Vec<String>> {
    let mut result = Vec::new();
    for entry in fs::read_dir(path)? {
        budget.visit()?;
        let name = entry?
            .file_name()
            .into_string()
            .map_err(|_| invalid("History asset names must be Unicode"))?;
        result.push(name);
    }
    crate::sort::by(&mut result, String::cmp);
    Ok(result)
}

fn index(path: &Path, budget: &mut Budget, allow_symlinks: bool) -> io::Result<()> {
    if budget.directories.contains_key(path) {
        return Ok(());
    }
    if !directory(path, allow_symlinks)? {
        return Err(invalid("History destination directory is missing"));
    }
    let mut index = Directory::default();
    for name in names(path, budget)? {
        let metadata = fs::symlink_metadata(path.join(&name))?;
        if !metadata.is_dir() && !metadata.is_file() {
            return Err(invalid("History destination contains a nonregular entry"));
        }
        let previous = index.names.insert(
            fold(&name),
            Name {
                spelling: name,
                directory: metadata.is_dir(),
            },
        );
        if previous.is_some() {
            return Err(invalid("History destination has ambiguous Unicode names"));
        }
    }
    budget.directories.insert(path.to_owned(), index);
    Ok(())
}

fn suffix(name: &str, counter: u64) -> String {
    // Match splitext: a leading dot alone does not introduce an extension.
    let split = name
        .rfind('.')
        .filter(|&position| name[..position].chars().any(|character| character != '.'));
    match split {
        Some(position) => format!("{}-{counter}{}", &name[..position], &name[position..]),
        None => format!("{name}-{counter}"),
    }
}

fn available_name(parent: &Path, name: &str, budget: &mut Budget) -> io::Result<String> {
    let family = fold(name);
    let index = budget.directories.get_mut(parent).expect("indexed parent");
    let mut next = *index.next_suffix.get(&family).unwrap_or(&2);
    loop {
        let candidate = suffix(name, next);
        next = next.checked_add(1).ok_or_else(limit_error)?;
        if !index.names.contains_key(&fold(&candidate)) {
            match fs::symlink_metadata(parent.join(&candidate)) {
                Err(error) if error.kind() == io::ErrorKind::NotFound => {
                    index.next_suffix.insert(family, next);
                    return Ok(candidate);
                }
                Ok(_) => {}
                Err(error) => return Err(error),
            }
        }
        if next > budget.limits.entries as u64 + 2 {
            return Err(limit_error());
        }
    }
}

fn mkdir(path: &Path) -> io::Result<()> {
    #[cfg_attr(not(unix), allow(unused_mut))] // Only Unix sets a mode.
    let mut options = fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        options.mode(0o700);
    }
    options.create(path)
}

fn target_directory(
    parent: &Path,
    name: &str,
    budget: &mut Budget,
    allow_symlinks: bool,
) -> io::Result<PathBuf> {
    index(parent, budget, allow_symlinks)?;
    let existing = budget.directories[parent].names.get(&fold(name)).cloned();
    if let Some(existing) = &existing
        && existing.directory
    {
        let path = parent.join(&existing.spelling);
        if directory(&path, allow_symlinks)? {
            return Ok(path);
        }
        return Err(invalid("History destination directory disappeared"));
    }
    let spelling = if existing.is_some() || fs::symlink_metadata(parent.join(name)).is_ok() {
        available_name(parent, name, budget)?
    } else {
        name.to_owned()
    };
    let path = parent.join(&spelling);
    mkdir(&path)?;
    budget
        .directories
        .get_mut(parent)
        .expect("indexed parent")
        .names
        .insert(
            fold(&spelling),
            Name {
                spelling,
                directory: true,
            },
        );
    budget
        .directories
        .insert(path.clone(), Directory::default());
    Ok(path)
}

fn fingerprint(path: &Path, limit: u64, allow_symlinks: bool) -> io::Result<Fingerprint> {
    let (mut file, before) = open_source(path, limit, allow_symlinks)?;
    let result = stream(&mut file, None, before.len())?;
    confirm(path, &file, &before)?;
    Ok(result)
}

fn equal_bytes(left: &Path, right: &Path, limit: u64, allow_symlinks: bool) -> io::Result<bool> {
    let (mut a, a_before) = open_source(left, limit, allow_symlinks)?;
    let (mut b, b_before) = open_source(right, limit, allow_symlinks)?;
    if a_before.len() != b_before.len() {
        return Ok(false);
    }
    let mut a_buffer = [0_u8; CHUNK];
    let mut b_buffer = [0_u8; CHUNK];
    let mut remaining = a_before.len();
    let mut same = true;
    while remaining != 0 {
        let count = remaining.min(CHUNK as u64) as usize;
        a.read_exact(&mut a_buffer[..count])?;
        b.read_exact(&mut b_buffer[..count])?;
        if a_buffer[..count] != b_buffer[..count] {
            same = false;
            break;
        }
        remaining -= count as u64;
    }
    confirm(left, &a, &a_before)?;
    confirm(right, &b, &b_before)?;
    Ok(same)
}

fn target_file(
    source: &Path,
    parent: &Path,
    name: &str,
    budget: &mut Budget,
    allow_symlinks: bool,
) -> io::Result<PathBuf> {
    index(parent, budget, allow_symlinks)?;
    let family = fold(name);
    let existing = budget.directories[parent].names.get(&family).cloned();
    if let Some(existing) = &existing
        && !existing.directory
    {
        let path = parent.join(&existing.spelling);
        let value = match budget.fingerprints.get(&path) {
            Some(value) => *value,
            None => {
                let value = fingerprint(&path, budget.limits.file_bytes, allow_symlinks)?;
                budget.fingerprints.insert(path, value);
                value
            }
        };
        let variants = &mut budget
            .directories
            .get_mut(parent)
            .expect("indexed parent")
            .variants;
        let entries = variants.entry((family.clone(), value)).or_default();
        if !entries.contains(&existing.spelling) {
            entries.push(existing.spelling.clone());
        }
    }
    if existing.is_some() {
        let value = fingerprint(source, budget.limits.file_bytes, allow_symlinks)?;
        if let Some(candidates) = budget.directories[parent]
            .variants
            .get(&(family.clone(), value))
        {
            for candidate in candidates {
                let path = parent.join(candidate);
                if equal_bytes(source, &path, budget.limits.file_bytes, allow_symlinks)? {
                    return Ok(path);
                }
            }
        }
    }
    let spelling = if existing.is_some() || fs::symlink_metadata(parent.join(name)).is_ok() {
        available_name(parent, name, budget)?
    } else {
        name.to_owned()
    };
    let path = parent.join(&spelling);
    copy_file(source, &path, budget, allow_symlinks)?;
    let value = budget.fingerprints[&path];
    budget
        .directories
        .get_mut(parent)
        .expect("indexed parent")
        .variants
        .entry((family, value))
        .or_default()
        .push(spelling);
    Ok(path)
}

struct Merge<'a> {
    source: &'a Path,
    out: &'a Path,
    budget: &'a mut Budget,
    allow_symlinks: bool,
    replacements: HashMap<String, String>,
}

impl Merge<'_> {
    fn walk(&mut self, source: &Path, target: &Path, depth: usize) -> io::Result<()> {
        if depth > self.budget.limits.depth {
            return Err(limit_error());
        }
        for name in names(source, self.budget)? {
            let path = source.join(&name);
            let metadata = fs::symlink_metadata(&path)?;
            let old = relative(&path, self.source)?;
            if matches!(
                old.as_str(),
                ".markitai/assets/.images.lock" | "assets/.images.lock"
            ) {
                regular(&metadata)?;
                continue;
            }
            if matches!(
                old.as_str(),
                ".markitai/assets/images.json" | "assets/images.json"
            ) {
                regular(&metadata)?;
                self.budget.image_indexes.push((path, target.join(name)));
                continue;
            }
            if metadata.file_type().is_dir() {
                let destination =
                    target_directory(target, &name, self.budget, self.allow_symlinks)?;
                self.walk(&path, &destination, depth + 1)?;
            } else if metadata.file_type().is_file() {
                let destination =
                    target_file(&path, target, &name, self.budget, self.allow_symlinks)?;
                let old = relative(&path, self.source)?;
                let new = relative(&destination, self.out)?;
                self.replacements.insert(old, new);
            } else {
                return Err(invalid(
                    "History assets contain a symlink or nonregular file",
                ));
            }
        }
        Ok(())
    }
}

fn relative(path: &Path, root: &Path) -> io::Result<String> {
    let path = path
        .strip_prefix(root)
        .map_err(|_| invalid("History asset escaped its root"))?;
    path.iter()
        .map(|component| {
            component
                .to_str()
                .ok_or_else(|| invalid("History asset names must be Unicode"))
        })
        .collect::<io::Result<Vec<_>>>()
        .map(|components| components.join("/"))
}

/// An error invalidates the whole private archive stage, including earlier copies.
pub(super) fn merge_root(
    source: &Path,
    out: &Path,
    budget: &mut Budget,
    allow_symlinks: bool,
) -> io::Result<HashMap<String, String>> {
    if !directory(source, allow_symlinks)? || !directory(out, allow_symlinks)? {
        return Err(invalid("History asset root is missing"));
    }
    let source = source.canonicalize()?;
    let out = out.canonicalize()?;
    if source == out {
        return Err(invalid("History asset source and destination overlap"));
    }
    let mut merge = Merge {
        source: &source,
        out: &out,
        budget,
        allow_symlinks,
        replacements: HashMap::new(),
    };
    for rel in ASSET_DIRS {
        // Check each selected component, even when ancestor symlinks are allowed.
        let mut input = source.clone();
        let mut present = true;
        for component in Path::new(rel) {
            input.push(component);
            if !directory(&input, allow_symlinks)? {
                present = false;
                break;
            }
        }
        if !present {
            continue;
        }
        // The job store may live below the output root (for example, a home
        // directory). Only overlap with a tree actually copied is recursive.
        let input = input.canonicalize()?;
        if input == out || out.starts_with(&input) || input.starts_with(&out) {
            return Err(invalid("History asset source and destination overlap"));
        }
        let mut target = out.clone();
        for component in rel.split('/') {
            target = target_directory(&target, component, merge.budget, allow_symlinks)?;
        }
        merge.walk(&input, &target, 0)?;
    }
    Ok(merge.replacements)
}

/// Read an index under the same no-follow and change-detection policy as assets.
pub(super) fn read_index(path: &Path, limit: u64, allow_symlinks: bool) -> io::Result<Vec<u8>> {
    let (mut file, before) = open_source(path, limit, allow_symlinks)?;
    let mut bytes = Vec::new();
    (&mut file).take(before.len() + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 != before.len() {
        return Err(invalid("History image index changed while reading"));
    }
    confirm(path, &file, &before)?;
    Ok(bytes)
}

pub(super) fn write_index(
    path: &Path,
    bytes: &[u8],
    budget: &mut Budget,
    allow_symlinks: bool,
) -> io::Result<()> {
    budget.admit(bytes.len() as u64)?;
    policy(path, allow_symlinks)?;
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600).custom_flags(libc::O_NOFOLLOW);
    }
    let mut file = options.open(path)?;
    file.write_all(bytes)?;
    // Synchronized with the whole stage before the job is named.
    budget.bytes += bytes.len() as u64;
    budget.files += 1;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn write(root: &Path, relative: &str, bytes: &[u8]) -> PathBuf {
        let path = root.join(relative);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, bytes).unwrap();
        path
    }

    fn limited_budget(file_bytes: u64, archive_bytes: u64, entries: usize, depth: usize) -> Budget {
        Budget {
            limits: Limits {
                file_bytes,
                archive_bytes,
                entries,
                depth,
            },
            ..Budget::default()
        }
    }

    #[test]
    fn copy_is_exclusive_private_streamed_and_charged_once() {
        let temp = TempDir::new().unwrap();
        let contents = vec![73; CHUNK * 3 + 17];
        let source = write(temp.path(), "source", &contents);
        let target = temp.path().join("target");
        let mut budget = Budget::default();
        copy_file(&source, &target, &mut budget, false).unwrap();
        assert_eq!(fs::read(&target).unwrap(), contents);
        assert_eq!(budget.bytes(), contents.len() as u64);
        assert_eq!(budget.files, 1);
        assert!(copy_file(&source, &target, &mut budget, false).is_err());
        assert_eq!(fs::read(&target).unwrap(), contents);
        assert_eq!(budget.files, 1);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(target).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
    }

    #[test]
    fn file_total_count_and_rewrite_limits_are_atomic() {
        let temp = TempDir::new().unwrap();
        let source = write(temp.path(), "source", b"1234");
        let mut budget = limited_budget(4, 6, 2, 64);
        copy_file(&source, &temp.path().join("one"), &mut budget, false).unwrap();
        assert!(copy_file(&source, &temp.path().join("two"), &mut budget, false).is_err());
        assert!(!temp.path().join("two").exists());
        assert_eq!((budget.bytes(), budget.files), (4, 1));
        assert!(budget.replace_bytes(4, 5).is_err());
        assert!(budget.replace_bytes(5, 1).is_err());
        assert_eq!(budget.bytes(), 4);
        budget.replace_bytes(4, 2).unwrap();
        copy_file(&source, &temp.path().join("two"), &mut budget, false).unwrap();
        assert_eq!((budget.bytes(), budget.files), (6, 2));
        let empty = write(temp.path(), "empty", b"");
        assert!(copy_file(&empty, &temp.path().join("three"), &mut budget, false).is_err());
        assert!(!temp.path().join("three").exists());
        let mut small = limited_budget(3, 100, 10, 64);
        assert!(copy_file(&source, &temp.path().join("large"), &mut small, false).is_err());
        assert_eq!(small.bytes(), 0);
    }

    #[test]
    fn mutation_after_open_cleans_partial_file_without_charging_budget() {
        let temp = TempDir::new().unwrap();
        let source = write(temp.path(), "source", &vec![7; CHUNK * 2]);
        let target = temp.path().join("copy");
        let (input, before) = open_source(&source, 1_000_000, false).unwrap();
        OpenOptions::new()
            .append(true)
            .open(&source)
            .unwrap()
            .write_all(b"growth")
            .unwrap();
        let mut budget = Budget::default();
        assert!(copy_opened(&source, input, before, &target, &mut budget, false).is_err());
        assert!(!target.exists());
        assert_eq!((budget.bytes(), budget.files), (0, 0));
    }

    #[test]
    fn nearest_root_stops_at_bookkeeping_and_never_crosses_boundary() {
        let temp = TempDir::new().unwrap();
        let outer = temp.path();
        write(outer, "assets/global.bin", b"outer");
        let boundary = outer.join("batch");
        let output = write(&boundary, "nested/doc.md", b"doc");
        assert_eq!(find_root(&output, 99, &boundary, false).unwrap(), None);
        write(&boundary, "assets/local.bin", b"local");
        assert_eq!(find_root(&output, 0, &boundary, false).unwrap(), None);
        assert_eq!(
            find_root(&output, 1, &boundary, false).unwrap(),
            Some(boundary.canonicalize().unwrap())
        );
        fs::create_dir_all(boundary.join("nested/.markitai/states")).unwrap();
        assert_eq!(
            find_root(&output, 99, &boundary, false).unwrap(),
            Some(boundary.join("nested").canonicalize().unwrap())
        );
        assert_eq!(
            find_root(&outer.join("missing/doc.md"), 99, &boundary, false).unwrap(),
            None
        );
    }

    #[test]
    fn merge_copies_only_asset_trees_and_maps_unchanged_nested_paths() {
        let temp = TempDir::new().unwrap();
        let source = temp.path().join("source");
        let out = temp.path().join("out");
        fs::create_dir(&out).unwrap();
        write(&source, ".markitai/assets/nested/p.png", b"image");
        write(&source, ".markitai/screenshots/page.jpg", b"screen");
        write(&source, "assets/plain.svg", b"visible");
        write(&source, ".markitai/states/state.json", b"private");
        write(&source, ".markitai/reports/report.json", b"private");
        write(&source, "unrelated.bin", b"private");
        let mut budget = Budget::default();
        let mapping = merge_root(&source, &out, &mut budget, false).unwrap();
        assert_eq!(mapping.len(), 3);
        for key in [
            ".markitai/assets/nested/p.png",
            ".markitai/screenshots/page.jpg",
            "assets/plain.svg",
        ] {
            assert_eq!(mapping[key], key);
            assert_eq!(
                fs::read(source.join(key)).unwrap(),
                fs::read(out.join(key)).unwrap()
            );
        }
        assert!(!out.join(".markitai/states").exists());
        assert!(!out.join(".markitai/reports").exists());
        assert!(!out.join("unrelated.bin").exists());
        assert_eq!((budget.bytes(), budget.files), (18, 3));
    }

    #[test]
    fn unicode_aliases_dedup_exact_bytes_and_reuse_renamed_variants() {
        let temp = TempDir::new().unwrap();
        let out = temp.path().join("out");
        fs::create_dir(&out).unwrap();
        let roots: Vec<_> = (0..4)
            .map(|index| temp.path().join(index.to_string()))
            .collect();
        write(&roots[0], "assets/Straße.png", b"AAAA");
        write(&roots[1], "assets/STRASSE.png", b"AAAA");
        write(&roots[2], "assets/strasse.png", b"BBBB");
        write(&roots[3], "assets/STRASSE.png", b"BBBB");
        let mut budget = Budget::default();
        merge_root(&roots[0], &out, &mut budget, false).unwrap();
        let identical = merge_root(&roots[1], &out, &mut budget, false).unwrap();
        assert_eq!(identical["assets/STRASSE.png"], "assets/Straße.png");
        let different = merge_root(&roots[2], &out, &mut budget, false).unwrap();
        assert_eq!(different["assets/strasse.png"], "assets/strasse-2.png");
        let repeat = merge_root(&roots[3], &out, &mut budget, false).unwrap();
        assert_eq!(repeat["assets/STRASSE.png"], "assets/strasse-2.png");
        assert_eq!((budget.bytes(), budget.files), (8, 2));
        assert_eq!(fs::read(out.join("assets/Straße.png")).unwrap(), b"AAAA");
    }

    #[test]
    fn sorted_names_and_cached_suffixes_preserve_reserved_candidates() {
        let temp = TempDir::new().unwrap();
        let out = temp.path().join("out");
        fs::create_dir(&out).unwrap();
        let first = temp.path().join("first");
        write(&first, "assets/p-2.png", b"reserved");
        write(&first, "assets/p.png", b"original");
        let mut budget = Budget::default();
        merge_root(&first, &out, &mut budget, false).unwrap();
        for number in 3..20 {
            let source = temp.path().join(number.to_string());
            write(&source, "assets/p.png", number.to_string().as_bytes());
            let mapping = merge_root(&source, &out, &mut budget, false).unwrap();
            assert_eq!(mapping["assets/p.png"], format!("assets/p-{number}.png"));
        }
        assert_eq!(fs::read(out.join("assets/p-2.png")).unwrap(), b"reserved");
        let directory = out.canonicalize().unwrap().join("assets");
        assert_eq!(budget.directories[&directory].next_suffix["p.png"], 20);
    }

    #[test]
    fn nested_directory_case_aliases_use_one_physical_spelling() {
        let temp = TempDir::new().unwrap();
        let out = temp.path().join("out");
        fs::create_dir(&out).unwrap();
        let first = temp.path().join("first");
        let second = temp.path().join("second");
        write(&first, "assets/Straße/a.png", b"one");
        write(&second, "assets/STRASSE/b.png", b"two");
        let mut budget = Budget::default();
        merge_root(&first, &out, &mut budget, false).unwrap();
        let mapping = merge_root(&second, &out, &mut budget, false).unwrap();
        assert_eq!(mapping["assets/STRASSE/b.png"], "assets/Straße/b.png");
        assert_eq!(fs::read_dir(out.join("assets")).unwrap().count(), 1);
    }

    #[test]
    fn file_directory_conflicts_are_renamed_without_overwrite() {
        let temp = TempDir::new().unwrap();
        let out = temp.path().join("out");
        fs::create_dir(&out).unwrap();
        let first = temp.path().join("first");
        let second = temp.path().join("second");
        write(&first, "assets/name.png", b"file");
        write(&second, "assets/name.png/inside", b"nested");
        let mut budget = Budget::default();
        merge_root(&first, &out, &mut budget, false).unwrap();
        let mapping = merge_root(&second, &out, &mut budget, false).unwrap();
        assert_eq!(
            mapping["assets/name.png/inside"],
            "assets/name-2.png/inside"
        );
        assert_eq!(fs::read(out.join("assets/name.png")).unwrap(), b"file");
    }

    #[test]
    fn traversal_and_depth_limits_reject_incomplete_archives() {
        let temp = TempDir::new().unwrap();
        let source = temp.path().join("source");
        write(&source, "assets/nested/deeper/file", b"bytes");
        let out = temp.path().join("out");
        fs::create_dir(&out).unwrap();
        let mut shallow = limited_budget(100, 100, 100, 1);
        assert!(merge_root(&source, &out, &mut shallow, false).is_err());
        assert_eq!(shallow.bytes(), 0);
        let out2 = temp.path().join("out2");
        fs::create_dir(&out2).unwrap();
        let mut few = limited_budget(100, 100, 1, 64);
        assert!(merge_root(&source, &out2, &mut few, false).is_err());
        assert_eq!(few.bytes(), 0);
    }

    #[test]
    fn nested_job_store_outside_selected_asset_trees_is_allowed() {
        let temp = TempDir::new().unwrap();
        write(temp.path(), "assets/a", b"a");
        let out = temp.path().join(".markitai/serve/jobs/stage/out");
        fs::create_dir_all(&out).unwrap();
        let mapping = merge_root(temp.path(), &out, &mut Budget::default(), false).unwrap();
        assert_eq!(mapping["assets/a"], "assets/a");
        assert_eq!(fs::read(out.join("assets/a")).unwrap(), b"a");
        assert!(!out.join(".markitai").exists());
    }

    #[test]
    fn stage_inside_a_selected_asset_tree_is_rejected_before_walking() {
        let temp = TempDir::new().unwrap();
        write(temp.path(), "assets/a", b"a");
        let out = temp.path().join("assets/archive/out");
        fs::create_dir_all(&out).unwrap();
        assert!(merge_root(temp.path(), &out, &mut Budget::default(), false).is_err());
        assert_eq!(fs::read_dir(&out).unwrap().count(), 0);
        assert!(merge_root(temp.path(), temp.path(), &mut Budget::default(), false).is_err());
    }

    #[test]
    fn extension_placement_matches_hidden_file_rules() {
        assert_eq!(suffix("a.tar.gz", 2), "a.tar-2.gz");
        assert_eq!(suffix(".hidden", 2), ".hidden-2");
        assert_eq!(suffix("...hidden", 2), "...hidden-2");
        assert_eq!(suffix(".hidden.png", 2), ".hidden-2.png");
        assert_eq!(suffix("a.", 2), "a-2.");
    }

    #[cfg(unix)]
    #[test]
    fn leaf_links_and_nonregular_files_are_rejected_even_when_allowed() {
        use std::os::unix::{fs::symlink, net::UnixListener};
        let temp = TempDir::new().unwrap();
        let actual = write(temp.path(), "actual", b"secret");
        symlink(&actual, temp.path().join("link")).unwrap();
        let mut budget = Budget::default();
        assert!(
            copy_file(
                &temp.path().join("link"),
                &temp.path().join("copy"),
                &mut budget,
                true
            )
            .is_err()
        );
        assert!(!temp.path().join("copy").exists());
        let _socket = UnixListener::bind(temp.path().join("socket")).unwrap();
        assert!(
            copy_file(
                &temp.path().join("socket"),
                &temp.path().join("copy"),
                &mut budget,
                true
            )
            .is_err()
        );
        let source = temp.path().join("source");
        fs::create_dir_all(source.join("assets")).unwrap();
        symlink(&actual, source.join("assets/link")).unwrap();
        let out = temp.path().join("out");
        fs::create_dir(&out).unwrap();
        assert!(merge_root(&source, &out, &mut budget, true).is_err());
        assert_eq!((budget.bytes(), budget.files), (0, 0));
    }

    #[cfg(unix)]
    #[test]
    fn ancestor_policy_is_separate_from_asset_directory_link_rejection() {
        use std::os::unix::fs::symlink;
        let temp = TempDir::new().unwrap();
        let actual = temp.path().join("actual");
        write(&actual, "assets/a", b"asset");
        write(&actual, "document.md", b"doc");
        let alias = temp.path().join("alias");
        symlink(&actual, &alias).unwrap();
        assert!(
            copy_file(
                &alias.join("document.md"),
                &temp.path().join("copy"),
                &mut Budget::default(),
                false
            )
            .is_err()
        );
        copy_file(
            &alias.join("document.md"),
            &temp.path().join("copy"),
            &mut Budget::default(),
            true,
        )
        .unwrap();
        assert_eq!(
            find_root(&alias.join("document.md"), 0, &alias, true).unwrap(),
            Some(actual.canonicalize().unwrap())
        );
        let root = temp.path().join("root");
        fs::create_dir(&root).unwrap();
        symlink(actual.join("assets"), root.join("assets")).unwrap();
        assert!(find_root(&root.join("doc.md"), 0, &root, true).is_err());
    }
}
