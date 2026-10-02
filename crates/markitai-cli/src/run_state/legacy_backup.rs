//! Retain the original legacy pair before publishing a native checkpoint.
//! These bytes permit state-format rollback; they do not undo output or model work.
use super::{Error, Limits, Result, sync_directory};
use markitai_core::platform::{self, FileId, Status};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

#[derive(Debug, PartialEq, Eq)]
struct Identity {
    bytes: u64,
    modified: Option<SystemTime>,
    id: FileId,
}
impl Identity {
    fn new(status: &Status) -> Self {
        Self {
            bytes: status.metadata().len(),
            modified: status.metadata().modified().ok(),
            id: status.id(),
        }
    }
}

#[derive(Serialize)]
struct Entry {
    name: String,
    bytes: u64,
    sha256: String,
}
struct Copy {
    entry: Entry,
    identity: Identity,
}
#[derive(Serialize)]
struct Manifest<'a> {
    version: u32,
    kind: &'static str,
    generation: &'a str,
    base: &'a Entry,
    // Null preserves absence; an existing zero-byte journal has a normal entry.
    journal: Option<&'a Entry>,
}

pub(super) fn preserve(
    directory: &Path,
    base: &Path,
    journal: &Path,
    generation: &str,
    limits: Limits,
) -> Result<PathBuf> {
    preserve_with(directory, base, journal, generation, limits, || Ok(()))
}

fn preserve_with(
    directory: &Path,
    base: &Path,
    journal: &Path,
    generation: &str,
    limits: Limits,
    before_verify: impl FnOnce() -> Result<()>,
) -> Result<PathBuf> {
    if uuid::Uuid::parse_str(generation).is_err() {
        return Err(Error::Invalid("invalid legacy backup generation".into()));
    }
    let mut builder = tempfile::Builder::new();
    builder.prefix(".markitai-legacy-");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        builder.permissions(fs::Permissions::from_mode(0o700));
    }
    let staged = builder.tempdir_in(directory)?;
    let saved_base = copy(base, staged.path(), limits.base_bytes)?
        .ok_or_else(|| Error::Invalid("legacy base disappeared before backup".into()))?;
    let saved_journal = copy(journal, staged.path(), limits.journal_bytes)?;
    before_verify()?;
    verify(base, Some(&saved_base), limits.base_bytes)?;
    verify(journal, saved_journal.as_ref(), limits.journal_bytes)?;
    let manifest = Manifest {
        version: 1,
        kind: "legacy-recovery-pair",
        generation,
        base: &saved_base.entry,
        journal: saved_journal.as_ref().map(|saved| &saved.entry),
    };
    let mut file = private_file(&staged.path().join("manifest.json"))?;
    serde_json::to_writer_pretty(&mut file, &manifest)
        .map_err(|_| Error::Invalid("legacy backup manifest could not be written".into()))?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    // Windows cannot rename a directory while a file in it is open.
    drop(file);
    sync_directory(staged.path())?;
    let name = base
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| Error::Invalid("legacy base filename is not UTF-8".into()))?;
    let target = directory.join(format!("{name}.legacy.{generation}"));
    match fs::symlink_metadata(&target) {
        Err(error) if error.kind() == io::ErrorKind::NotFound => (),
        Err(error) => return Err(error.into()),
        Ok(_) => {
            return Err(Error::Invalid(
                "legacy backup destination already exists".into(),
            ));
        }
    }
    platform::rename(staged.path(), &target)?;
    // Drop only knows the old temporary path. A failed parent sync leaves the
    // complete backup in place and returns no permission to replace the base.
    // Windows has no directory flush; the manifest's flush after the rename
    // commits it instead.
    platform::sync_renamed_path(&target.join("manifest.json"))?;
    sync_directory(directory)?;
    Ok(target)
}

fn private_file(path: &Path) -> Result<File> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    Ok(platform::private_file(&mut options).open(path)?)
}

fn open_regular(path: &Path, maximum: usize) -> Result<Option<(File, Identity)>> {
    let status = match platform::status(path) {
        Ok(status) => status,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    let metadata = status.metadata();
    if !metadata.is_file() || metadata.file_type().is_symlink() {
        return Err(Error::Invalid(
            "legacy backup source must be a regular file without symbolic links".into(),
        ));
    }
    if metadata.len() > maximum as u64 {
        return Err(Error::Limit("legacy backup bytes"));
    }
    let expected = Identity::new(&status);
    let file = platform::open_read(path, false)?;
    let opened = platform::file_status(&file)?;
    if !opened.metadata().is_file() || Identity::new(&opened) != expected {
        return Err(changed());
    }
    Ok(Some((file, expected)))
}

fn fingerprint(
    mut file: &File,
    mut target: Option<&mut File>,
    maximum: usize,
) -> Result<(u64, String)> {
    let mut digest = Sha256::new();
    let mut bytes = 0u64;
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        bytes = bytes
            .checked_add(read as u64)
            .ok_or(Error::Limit("legacy backup bytes"))?;
        if bytes > maximum as u64 {
            return Err(Error::Limit("legacy backup bytes"));
        }
        digest.update(&buffer[..read]);
        if let Some(output) = target.as_mut() {
            output.write_all(&buffer[..read])?;
        }
    }
    Ok((bytes, markitai_core::hex(digest.finalize())))
}

fn copy(source: &Path, directory: &Path, maximum: usize) -> Result<Option<Copy>> {
    let Some((input, identity)) = open_regular(source, maximum)? else {
        return Ok(None);
    };
    let name = source
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| Error::Invalid("legacy backup filename is not UTF-8".into()))?;
    let mut output = private_file(&directory.join(name))?;
    let (bytes, sha256) = fingerprint(&input, Some(&mut output), maximum)?;
    if bytes != identity.bytes || Identity::new(&platform::file_status(&input)?) != identity {
        return Err(changed());
    }
    output.sync_all()?;
    Ok(Some(Copy {
        entry: Entry {
            name: name.into(),
            bytes,
            sha256,
        },
        identity,
    }))
}

fn verify(path: &Path, saved: Option<&Copy>, maximum: usize) -> Result<()> {
    match (open_regular(path, maximum)?, saved) {
        (None, None) => Ok(()),
        (Some((file, identity)), Some(saved)) if identity == saved.identity => {
            let (bytes, sha256) = fingerprint(&file, None, maximum)?;
            if bytes != saved.entry.bytes
                || sha256 != saved.entry.sha256
                || Identity::new(&platform::file_status(&file)?) != identity
            {
                return Err(changed());
            }
            // Reopening checked identity before hashing; ensure the path still
            // names that regular file after the bounded read.
            let status = platform::status(path)?;
            if !status.metadata().is_file() || Identity::new(&status) != identity {
                return Err(changed());
            }
            Ok(())
        }
        _ => Err(changed()),
    }
}

fn changed() -> Error {
    Error::Invalid("legacy recovery files changed while preserving their original bytes".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn pair(root: &Path) -> (PathBuf, PathBuf) {
        (
            root.join("markitai.abc123.state.json"),
            root.join("markitai.abc123.state.jsonl"),
        )
    }
    fn generation() -> String {
        uuid::Uuid::new_v4().to_string()
    }
    fn manifest(path: &Path) -> Value {
        serde_json::from_slice(&fs::read(path.join("manifest.json")).unwrap()).unwrap()
    }

    #[test]
    fn exact_raw_pair_retains_unknown_fields_whitespace_bad_lines_and_private_modes() {
        let dir = tempfile::tempdir().unwrap();
        let (base, journal) = pair(dir.path());
        let original = " { \"version\":\"1.0\", \"extra\":\"私人\", \"documents\": {} } \n";
        let lines = b"{bad json\n\n{\"type\":\"unknown\",\"data\":{\"x\":1}}\n";
        fs::write(&base, original).unwrap();
        fs::write(&journal, lines).unwrap();
        let target = preserve(
            dir.path(),
            &base,
            &journal,
            &generation(),
            Limits::default(),
        )
        .unwrap();
        for source in [&base, &journal] {
            assert_eq!(
                fs::read(target.join(source.file_name().unwrap())).unwrap(),
                fs::read(source).unwrap()
            );
        }
        let record = manifest(&target);
        assert_eq!(
            record["base"]["sha256"],
            markitai_core::hex(Sha256::digest(original.as_bytes()))
        );
        assert_eq!(
            record["journal"]["sha256"],
            markitai_core::hex(Sha256::digest(lines))
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(&target).unwrap().permissions().mode() & 0o777,
                0o700
            );
            for entry in fs::read_dir(&target).unwrap() {
                assert_eq!(
                    entry.unwrap().metadata().unwrap().permissions().mode() & 0o777,
                    0o600
                );
            }
        }
    }

    #[test]
    fn missing_journal_and_empty_journal_remain_distinct() {
        let dir = tempfile::tempdir().unwrap();
        let (base, journal) = pair(dir.path());
        fs::write(&base, b"{}").unwrap();
        let absent = preserve(
            dir.path(),
            &base,
            &journal,
            &generation(),
            Limits::default(),
        )
        .unwrap();
        assert!(manifest(&absent)["journal"].is_null());
        assert!(!absent.join(journal.file_name().unwrap()).exists());
        fs::write(&journal, []).unwrap();
        let empty = preserve(
            dir.path(),
            &base,
            &journal,
            &generation(),
            Limits::default(),
        )
        .unwrap();
        assert_eq!(manifest(&empty)["journal"]["bytes"], 0);
        assert_eq!(
            fs::read(empty.join(journal.file_name().unwrap())).unwrap(),
            b""
        );
    }

    #[test]
    fn oversized_or_changed_sources_and_failed_staging_preserve_original_pair() {
        let dir = tempfile::tempdir().unwrap();
        let (base, journal) = pair(dir.path());
        fs::write(&base, b"base").unwrap();
        fs::write(&journal, b"journal").unwrap();
        let limited = Limits {
            journal_bytes: 3,
            ..Limits::default()
        };
        assert!(preserve(dir.path(), &base, &journal, &generation(), limited).is_err());
        assert!(
            preserve_with(
                dir.path(),
                &base,
                &journal,
                &generation(),
                Limits::default(),
                || { Err(io::Error::other("injected backup staging failure").into()) }
            )
            .is_err()
        );
        assert_eq!(fs::read(&base).unwrap(), b"base");
        assert_eq!(fs::read(&journal).unwrap(), b"journal");
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 2);
        assert!(
            preserve_with(
                dir.path(),
                &base,
                &journal,
                &generation(),
                Limits::default(),
                || {
                    fs::write(&journal, b"changed")?;
                    Ok(())
                }
            )
            .is_err()
        );
        assert_eq!(fs::read(&journal).unwrap(), b"changed");
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 2);
    }

    #[cfg(unix)]
    #[test]
    fn links_and_special_files_are_rejected_without_opening_them() {
        use std::os::unix::fs::symlink;
        let dir = tempfile::tempdir().unwrap();
        let (base, journal) = pair(dir.path());
        fs::write(&base, b"{}").unwrap();
        symlink(&base, &journal).unwrap();
        assert!(
            preserve(
                dir.path(),
                &base,
                &journal,
                &generation(),
                Limits::default()
            )
            .is_err()
        );
        fs::remove_file(&journal).unwrap();
        let path = std::ffi::CString::new(journal.as_os_str().as_encoded_bytes()).unwrap();
        assert_eq!(unsafe { libc::mkfifo(path.as_ptr(), 0o600) }, 0);
        assert!(
            preserve(
                dir.path(),
                &base,
                &journal,
                &generation(),
                Limits::default()
            )
            .is_err()
        );
        assert_eq!(fs::read(&base).unwrap(), b"{}");
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 2);
    }
}
