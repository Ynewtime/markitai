use md5::{Digest, Md5};
use serde_json::Value;
use std::collections::HashSet;
use std::ffi::OsStr;
use std::fmt::Write as _;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Component, Path, PathBuf};
use tempfile::NamedTempFile;

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Publication {
    Written(PathBuf),
    SkippedExisting(PathBuf),
}

/// Resolve existing symlinks before simplifying `..`, retaining missing tails.
/// Planning never creates a directory or requires the final path to exist.
pub(crate) fn resolve_path(path: &Path) -> io::Result<PathBuf> {
    let absolute = std::path::absolute(path)?;
    let mut resolved = PathBuf::new();
    resolve_components(&absolute, &mut resolved, &mut HashSet::new(), 0)?;
    Ok(resolved)
}

fn resolve_components(
    path: &Path,
    resolved: &mut PathBuf,
    active: &mut HashSet<PathBuf>,
    depth: usize,
) -> io::Result<()> {
    if depth > 256 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Report path has too many nested symbolic links",
        ));
    }
    for component in path.components() {
        match component {
            Component::Prefix(_) | Component::RootDir => resolved.push(component),
            Component::CurDir => (),
            Component::ParentDir => {
                resolved.pop();
            }
            Component::Normal(name) => {
                let candidate = resolved.join(name);
                let metadata = match fs::symlink_metadata(&candidate) {
                    Ok(metadata) => Some(metadata),
                    // Python's non-strict resolver also retains inaccessible
                    // components; publication separately checks actual access.
                    Err(error)
                        if matches!(
                            error.kind(),
                            io::ErrorKind::NotFound
                                | io::ErrorKind::NotADirectory
                                | io::ErrorKind::PermissionDenied
                        ) =>
                    {
                        None
                    }
                    Err(error) => return Err(error),
                };
                if metadata.is_some_and(|metadata| metadata.file_type().is_symlink()) {
                    if !active.insert(candidate.clone()) {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            "Report path contains a symbolic link loop",
                        ));
                    }
                    let target = fs::read_link(&candidate)?;
                    resolve_components(&target, resolved, active, depth + 1)?;
                    active.remove(&candidate);
                } else {
                    resolved.push(name);
                }
            }
        }
    }
    Ok(())
}

pub(crate) fn task_hash(
    input: &Path,
    output: &Path,
    selected_options: &Value,
) -> io::Result<String> {
    hash_resolved(
        &resolve_path(input)?,
        &resolve_path(output)?,
        selected_options,
    )
}

fn hash_resolved(input: &Path, output: &Path, options: &Value) -> io::Result<String> {
    let mut json = String::from("{\"input\": ");
    write_path(&mut json, input.as_os_str());
    json.push_str(", \"options\": ");
    write_json(&mut json, options)?;
    json.push_str(", \"output\": ");
    write_path(&mut json, output.as_os_str());
    json.push('}');
    let digest = Md5::digest(json.as_bytes());
    Ok(format!(
        "{:02x}{:02x}{:02x}",
        digest[0], digest[1], digest[2]
    ))
}

// Task identities use Python json.dumps defaults, including ASCII escaping and
// spaces after separators. Serde's compact representation changes the identity.
fn write_json(output: &mut String, value: &Value) -> io::Result<()> {
    match value {
        Value::Null => output.push_str("null"),
        Value::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
        Value::Number(value) if value.is_i64() || value.is_u64() => {
            output.push_str(&value.to_string());
        }
        Value::Number(_) => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Report identity options must use integer numbers",
            ));
        }
        Value::String(value) => write_string(output, value),
        Value::Array(values) => {
            output.push('[');
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    output.push_str(", ");
                }
                write_json(output, value)?;
            }
            output.push(']');
        }
        Value::Object(values) => {
            output.push('{');
            let mut entries: Vec<_> = values.iter().collect();
            // A map's keys are distinct: the stable order is the unstable one.
            crate::sort::by(&mut entries, |left, right| left.0.cmp(right.0));
            for (index, (key, value)) in entries.into_iter().enumerate() {
                if index != 0 {
                    output.push_str(", ");
                }
                write_string(output, key);
                output.push_str(": ");
                write_json(output, value)?;
            }
            output.push('}');
        }
    }
    Ok(())
}

fn write_string(output: &mut String, value: &str) {
    output.push('"');
    for unit in value.encode_utf16() {
        write_unit(output, unit);
    }
    output.push('"');
}

fn write_unit(output: &mut String, unit: u16) {
    match unit {
        0x08 => output.push_str("\\b"),
        0x0c => output.push_str("\\f"),
        0x0a => output.push_str("\\n"),
        0x0d => output.push_str("\\r"),
        0x09 => output.push_str("\\t"),
        0x22 => output.push_str("\\\""),
        0x5c => output.push_str("\\\\"),
        0x20..=0x7e => output.push(char::from_u32(u32::from(unit)).unwrap()),
        _ => write!(output, "\\u{unit:04x}").unwrap(),
    }
}

#[cfg(unix)]
fn write_path(output: &mut String, value: &OsStr) {
    use std::os::unix::ffi::OsStrExt;
    output.push('"');
    let mut remaining = value.as_bytes();
    while !remaining.is_empty() {
        match std::str::from_utf8(remaining) {
            Ok(valid) => {
                for unit in valid.encode_utf16() {
                    write_unit(output, unit);
                }
                break;
            }
            Err(error) => {
                let (valid, invalid) = remaining.split_at(error.valid_up_to());
                for unit in std::str::from_utf8(valid).unwrap().encode_utf16() {
                    write_unit(output, unit);
                }
                let invalid_len = error.error_len().unwrap_or(invalid.len());
                // Python decodes undecodable filesystem bytes with surrogateescape.
                for byte in &invalid[..invalid_len] {
                    write_unit(output, 0xdc00 | u16::from(*byte));
                }
                remaining = &invalid[invalid_len..];
            }
        }
    }
    output.push('"');
}

#[cfg(windows)]
fn write_path(output: &mut String, value: &OsStr) {
    use std::os::windows::ffi::OsStrExt;
    output.push('"');
    for unit in value.encode_wide() {
        write_unit(output, unit);
    }
    output.push('"');
}

#[cfg(not(any(unix, windows)))]
fn write_path(output: &mut String, value: &OsStr) {
    write_string(output, &value.to_string_lossy());
}

pub(crate) fn publish(
    output_dir: &Path,
    task_hash: &str,
    on_conflict: &str,
    allow_symlinks: bool,
    directory_fallback: bool,
    bytes: &[u8],
) -> io::Result<Publication> {
    if task_hash.len() != 6
        || !task_hash
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Report task hash must contain six lowercase hexadecimal digits",
        ));
    }
    if !matches!(on_conflict, "rename" | "overwrite" | "skip") {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Unknown report conflict policy",
        ));
    }
    let requested = output_dir.join(".markitai/reports");
    check_path(output_dir, allow_symlinks)?;
    check_path(&requested, allow_symlinks)?;
    let reports = resolve_path(&requested)?;
    fs::create_dir_all(&reports)?;
    check_path(&requested, allow_symlinks)?;
    publish_candidates(
        &requested,
        &reports,
        ReportNames::new(task_hash, directory_fallback),
        on_conflict,
        allow_symlinks,
        bytes,
    )
}

fn publish_candidates(
    requested: &Path,
    reports: &Path,
    names: impl Iterator<Item = String>,
    on_conflict: &str,
    allow_symlinks: bool,
    bytes: &[u8],
) -> io::Result<Publication> {
    let mut staged = None;
    for filename in names {
        let path = reports.join(&filename);
        check_path(&requested.join(&filename), allow_symlinks)?;
        check_path(&path, allow_symlinks)?;
        if occupied(&path)? {
            match on_conflict {
                "skip" => return Ok(Publication::SkippedExisting(path)),
                "rename" => continue,
                _ => (),
            }
        }
        let temp = match staged.take() {
            Some(temp) => temp,
            None => stage_report(reports, |file| file.write_all(bytes))?,
        };
        // Recheck policy immediately before publication. The temp file already
        // contains the complete synced report; a failed publish keeps the old one.
        check_path(&requested.join(&filename), allow_symlinks)?;
        check_path(&path, allow_symlinks)?;
        if on_conflict == "overwrite" {
            temp.persist(&path).map_err(|error| error.error)?;
            return Ok(Publication::Written(path));
        }
        match temp.persist_noclobber(&path) {
            Ok(_) => return Ok(Publication::Written(path)),
            Err(error) if error.error.kind() == io::ErrorKind::AlreadyExists => {
                if on_conflict == "skip" {
                    return Ok(Publication::SkippedExisting(path));
                }
                staged = Some(error.file);
            }
            Err(error) => return Err(error.error),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "Unable to reserve a unique report filename",
    ))
}

fn check_path(path: &Path, allow_symlinks: bool) -> io::Result<()> {
    markitai_core::output::check_path(path, allow_symlinks).map_err(io::Error::other)
}

fn occupied(path: &Path) -> io::Result<bool> {
    match fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error),
    }
}

fn stage_report(
    directory: &Path,
    write: impl FnOnce(&mut File) -> io::Result<()>,
) -> io::Result<NamedTempFile> {
    let mut temp = markitai_core::output::deliverable_builder()
        .prefix(".markitai-report-")
        .suffix(".tmp")
        .tempfile_in(directory)?;
    write(temp.as_file_mut())?;
    // Complete bytes must precede the report's name. That name is never
    // synchronized, so the report was not durable on return before either;
    // an ordering fence gives the same crash outcomes without a cache flush.
    let mut fence = crate::output_claims::sync_group::SyncGroup::new();
    fence.stage(temp.as_file())?;
    fence.commit_ordered()?;
    Ok(temp)
}

struct ReportNames<'a> {
    hash: &'a str,
    version: u32,
    fallback_attempt: u8,
    directory: bool,
    timestamp: Option<i64>,
}

impl<'a> ReportNames<'a> {
    fn new(hash: &'a str, directory: bool) -> Self {
        Self {
            hash,
            version: 1,
            fallback_attempt: 0,
            directory,
            timestamp: None,
        }
    }
}

impl Iterator for ReportNames<'_> {
    type Item = String;

    fn next(&mut self) -> Option<String> {
        let suffix = match self.version {
            1 => String::new(),
            2..=9999 => format!(".v{}", self.version),
            _ => {
                if self.fallback_attempt == 64 {
                    return None;
                }
                let attempt = self.fallback_attempt;
                self.fallback_attempt += 1;
                let timestamp = self
                    .timestamp
                    .get_or_insert_with(|| chrono::Utc::now().timestamp());
                if self.directory && attempt == 0 {
                    format!(".{timestamp}")
                } else if self.directory {
                    format!(".{timestamp}.{}", uuid::Uuid::new_v4().simple())
                } else {
                    format!(".{}", uuid::Uuid::new_v4().simple())
                }
            }
        };
        self.version += 1;
        Some(format!("markitai.{}{suffix}.report.json", self.hash))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn written(result: Publication) -> PathBuf {
        match result {
            Publication::Written(path) => path,
            Publication::SkippedExisting(_) => panic!("Expected a new report"),
        }
    }

    #[test]
    fn python_json_spacing_sorting_ascii_and_surrogate_pairs() {
        let value = json!({
            "z": "\0\u{8}\u{c}\n\r\t\u{1f}\u{7f}/\"\\é😀",
            "a": [true, null, {"β": 2, "a": 1}]
        });
        let mut text = String::new();
        write_json(&mut text, &value).unwrap();
        assert_eq!(
            text,
            r#"{"a": [true, null, {"a": 1, "\u03b2": 2}], "z": "\u0000\b\f\n\r\t\u001f\u007f/\"\\\u00e9\ud83d\ude00"}"#
        );
    }

    #[test]
    fn python_md5_goldens_cover_spaces_unicode_and_absence() {
        assert_eq!(
            hash_resolved(
                Path::new("/input with spaces/資料😀.txt"),
                Path::new("/output dir"),
                &json!({"llm": false, "glob_patterns": ["*.文", "**/😀?.txt"], "scan_max_depth": null}),
            ).unwrap(),
            "16cdb3"
        );
        assert_eq!(
            hash_resolved(Path::new("/a"), Path::new("/b"), &json!({})).unwrap(),
            "2a2c9f"
        );
        assert_eq!(
            hash_resolved(Path::new("/a"), Path::new("/b"), &json!({"llm": false})).unwrap(),
            "b53e0c"
        );
        assert!(hash_resolved(Path::new("/a"), Path::new("/b"), &json!({"depth": 1.5})).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn invalid_filesystem_bytes_use_python_surrogateescape() {
        use std::os::unix::ffi::OsStrExt;
        assert_eq!(
            hash_resolved(
                Path::new(OsStr::from_bytes(b"/markitai-\xff")),
                Path::new("/out"),
                &json!({})
            )
            .unwrap(),
            "449256"
        );
    }

    #[test]
    fn resolve_missing_parent_components_without_creating_paths() {
        let dir = tempfile::tempdir().unwrap();
        let root = fs::canonicalize(dir.path()).unwrap();
        assert_eq!(
            resolve_path(&dir.path().join("missing/../future/./file")).unwrap(),
            root.join("future/file")
        );
        assert!(!dir.path().join("missing").exists());
        assert!(!dir.path().join("future").exists());
        let cwd = std::env::current_dir().unwrap();
        assert_eq!(
            resolve_path(Path::new("missing/../future")).unwrap(),
            resolve_path(&cwd.join("future")).unwrap()
        );
        assert_eq!(
            task_hash(
                &dir.path().join("x/../input"),
                &dir.path().join("out"),
                &json!({})
            )
            .unwrap(),
            task_hash(
                &dir.path().join("input"),
                &dir.path().join("out"),
                &json!({})
            )
            .unwrap()
        );
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 0);
    }

    #[cfg(unix)]
    #[test]
    fn resolve_symlinks_before_parent_traversal_and_missing_targets() {
        use std::os::unix::fs::symlink;
        let dir = tempfile::tempdir().unwrap();
        let root = fs::canonicalize(dir.path()).unwrap();
        fs::create_dir_all(root.join("real/nested")).unwrap();
        symlink("real/nested", root.join("alias")).unwrap();
        symlink("absent/../future", root.join("dangling")).unwrap();
        assert_eq!(
            resolve_path(&root.join("alias/../file")).unwrap(),
            root.join("real/file")
        );
        assert_eq!(
            resolve_path(&root.join("dangling/child")).unwrap(),
            root.join("future/child")
        );
        symlink("cycle", root.join("cycle")).unwrap();
        assert!(resolve_path(&root.join("cycle/file")).is_err());
    }

    #[test]
    fn publish_skip_rename_and_overwrite_preserve_correct_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let first =
            written(publish(dir.path(), "a12b34", "rename", false, false, b"first").unwrap());
        assert_eq!(
            publish(dir.path(), "a12b34", "skip", false, false, b"ignored").unwrap(),
            Publication::SkippedExisting(first.clone())
        );
        assert_eq!(fs::read(&first).unwrap(), b"first");
        let second =
            written(publish(dir.path(), "a12b34", "rename", false, false, b"second").unwrap());
        assert_eq!(
            second.file_name().unwrap(),
            "markitai.a12b34.v2.report.json"
        );
        assert_eq!(
            written(
                publish(
                    dir.path(),
                    "a12b34",
                    "overwrite",
                    false,
                    false,
                    b"replacement"
                )
                .unwrap()
            ),
            first
        );
        assert_eq!(fs::read(&first).unwrap(), b"replacement");
        assert_eq!(fs::read(&second).unwrap(), b"second");
        assert_eq!(fs::read_dir(first.parent().unwrap()).unwrap().count(), 2);
    }

    /// Each staged report is fenced for ordering once before its rename; a
    /// skipped existing report stages nothing.
    #[test]
    fn staged_report_bytes_are_ordered_before_publication() {
        use crate::output_claims::sync_group::take_commits;
        let dir = tempfile::tempdir().unwrap();
        take_commits();
        let first =
            written(publish(dir.path(), "a12b34", "rename", false, false, b"first").unwrap());
        assert_eq!(take_commits(), ["ordered"]);
        publish(dir.path(), "a12b34", "skip", false, false, b"ignored").unwrap();
        assert!(take_commits().is_empty());
        written(publish(dir.path(), "a12b34", "overwrite", false, false, b"new").unwrap());
        assert_eq!(take_commits(), ["ordered"]);
        assert_eq!(fs::read(first).unwrap(), b"new");
    }

    #[test]
    fn invalid_names_and_policy_have_no_side_effects() {
        let dir = tempfile::tempdir().unwrap();
        for hash in ["../../", "a12b3/", "abcdefghi", "ABCDEF", "💥"] {
            assert!(publish(dir.path(), hash, "rename", true, false, b"x").is_err());
        }
        assert!(publish(dir.path(), "abcdef", "invalid", false, false, b"x").is_err());
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 0);
    }

    #[test]
    fn failed_staging_removes_partial_temporary_file_and_preserves_report() {
        let dir = tempfile::tempdir().unwrap();
        let old = written(publish(dir.path(), "abcdef", "rename", false, false, b"old").unwrap());
        let error = stage_report(old.parent().unwrap(), |file| {
            file.write_all(b"partial replacement")?;
            Err(io::Error::other("injected write failure"))
        })
        .unwrap_err();
        assert_eq!(error.to_string(), "injected write failure");
        assert_eq!(fs::read(&old).unwrap(), b"old");
        assert_eq!(fs::read_dir(old.parent().unwrap()).unwrap().count(), 1);
    }

    #[test]
    fn failed_publication_preserves_existing_directory_and_cleans_temp() {
        let dir = tempfile::tempdir().unwrap();
        let reports = dir.path().join(".markitai/reports");
        let occupied = reports.join("markitai.abcdef.report.json");
        fs::create_dir_all(&occupied).unwrap();
        fs::write(occupied.join("keep"), b"existing").unwrap();
        assert!(publish(dir.path(), "abcdef", "overwrite", false, false, b"x").is_err());
        assert_eq!(fs::read(occupied.join("keep")).unwrap(), b"existing");
        assert_eq!(fs::read_dir(&reports).unwrap().count(), 1);
        let blocked = dir.path().join("blocked");
        fs::write(&blocked, b"not a directory").unwrap();
        assert!(publish(&blocked, "abcdef", "rename", false, false, b"x").is_err());
        assert_eq!(fs::read(blocked).unwrap(), b"not a directory");
    }

    #[test]
    fn concurrent_renames_do_not_clobber_any_report() {
        let dir = tempfile::tempdir().unwrap();
        let barrier = std::sync::Barrier::new(8);
        let paths = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..8)
                .map(|index| {
                    let barrier = &barrier;
                    let path = dir.path();
                    scope.spawn(move || {
                        barrier.wait();
                        let bytes = index.to_string();
                        (
                            written(
                                publish(path, "abcdef", "rename", false, false, bytes.as_bytes())
                                    .unwrap(),
                            ),
                            bytes,
                        )
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
        for (path, bytes) in &paths {
            assert_eq!(fs::read(path).unwrap(), bytes.as_bytes());
        }
        assert_eq!(
            paths
                .iter()
                .map(|(path, _)| path)
                .collect::<HashSet<_>>()
                .len(),
            8
        );
        assert_eq!(
            fs::read_dir(paths[0].0.parent().unwrap()).unwrap().count(),
            8
        );
    }

    #[test]
    fn version_limit_and_fallback_names_remain_bounded_and_distinct() {
        let mut names = ReportNames::new("abcdef", true);
        names.version = 9999;
        names.timestamp = Some(1_800_000_000);
        assert_eq!(names.next().unwrap(), "markitai.abcdef.v9999.report.json");
        assert_eq!(
            names.next().unwrap(),
            "markitai.abcdef.1800000000.report.json"
        );
        let rest: HashSet<_> = names.collect();
        assert_eq!(rest.len(), 63);
        assert!(
            rest.iter()
                .all(|name| name.starts_with("markitai.abcdef.1800000000.")
                    && name.ends_with(".report.json"))
        );
        let mut shared = ReportNames::new("abcdef", false);
        shared.version = 10000;
        let names: HashSet<_> = shared.collect();
        assert_eq!(names.len(), 64);
        for name in names {
            let uuid = name
                .strip_prefix("markitai.abcdef.")
                .unwrap()
                .strip_suffix(".report.json")
                .unwrap();
            assert_eq!(uuid.len(), 32);
            assert!(uuid::Uuid::parse_str(uuid).is_ok());
        }
    }

    #[test]
    fn timestamp_fallback_collision_keeps_both_existing_reports() {
        let dir = tempfile::tempdir().unwrap();
        let reports = fs::canonicalize(dir.path()).unwrap();
        let versioned = reports.join("markitai.abcdef.v9999.report.json");
        let timestamp = reports.join("markitai.abcdef.1800000000.report.json");
        fs::write(&versioned, b"versioned").unwrap();
        fs::write(&timestamp, b"timestamp").unwrap();
        let mut names = ReportNames::new("abcdef", true);
        names.version = 9999;
        names.timestamp = Some(1_800_000_000);
        let path = written(
            publish_candidates(&reports, &reports, names, "rename", false, b"new").unwrap(),
        );
        assert!(
            path.file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .starts_with("markitai.abcdef.1800000000.")
        );
        assert_eq!(fs::read(path).unwrap(), b"new");
        assert_eq!(fs::read(versioned).unwrap(), b"versioned");
        assert_eq!(fs::read(timestamp).unwrap(), b"timestamp");
        assert_eq!(fs::read_dir(reports).unwrap().count(), 3);
    }

    #[cfg(unix)]
    #[test]
    fn symlink_policy_applies_to_directories_and_report_entries() {
        use std::os::unix::fs::symlink;
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("target");
        fs::create_dir(&target).unwrap();
        let alias = dir.path().join("alias");
        symlink(&target, &alias).unwrap();
        assert!(publish(&alias, "abcdef", "rename", false, false, b"x").is_err());
        assert!(!target.join(".markitai").exists());
        let first = written(publish(&alias, "abcdef", "rename", true, false, b"first").unwrap());
        let external = dir.path().join("external");
        fs::write(&external, b"external").unwrap();
        fs::remove_file(&first).unwrap();
        symlink(&external, &first).unwrap();
        assert!(publish(&target, "abcdef", "overwrite", false, false, b"x").is_err());
        assert_eq!(fs::read(&external).unwrap(), b"external");
        assert_eq!(
            publish(&target, "abcdef", "skip", true, false, b"ignored").unwrap(),
            Publication::SkippedExisting(first.clone())
        );
        written(publish(&target, "abcdef", "overwrite", true, false, b"replacement").unwrap());
        assert!(!first.is_symlink());
        assert_eq!(fs::read(first).unwrap(), b"replacement");
        assert_eq!(fs::read(external).unwrap(), b"external");
    }
}
