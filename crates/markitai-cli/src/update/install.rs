//! Replacing a setup-script install in place, as the setup scripts would:
//! the release's SHA256SUMS names the archive and its digest; the archive is
//! downloaded into the install folder (so the final step is a rename on one
//! volume), checked, unpacked and dry-started before anything installed is
//! touched; then the files are swapped, and on any error put back.
//!
//! `setup.sh` installs keep `bin_dir/markitai` (its `mkai` and `markitai-mcp`
//! links stay) and `~/.local/share/markitai/current` (notices, guides,
//! receipt). The program is replaced by one atomic rename; the old one is
//! kept beside it as `.markitai.old-<id>` until the update is committed.
//! `setup.ps1` installs are one folder: every name its receipt lists is
//! renamed aside (a running EXE can be renamed, not deleted), the new names
//! renamed in, and on a failure everything goes back. The planned renames
//! are written first (`.markitai-swap-<id>`): a run killed between them is
//! undone by the next update, server start or setup.ps1.
use super::check::{self, Base, Failure, Timeouts};
use super::method::{Receipt, Script, SetupInstall};
use super::version::Version;
use sha2::{Digest, Sha256};
use std::ffi::OsString;
use std::io::{self, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::time::Duration;

/// An unpacked release is about 100 MB; anything far bigger is not one.
const UNPACKED_LIMIT: u64 = 1024 * 1024 * 1024;
const ENTRY_LIMIT: usize = 20_000;
const ARCHIVE_LIMIT: u64 = 512 * 1024 * 1024;

/// What an update failed on, and therefore what it left.
#[derive(Debug)]
pub(crate) enum Error {
    Fetch(Failure),
    Checksum {
        expected: String,
        actual: String,
    },
    DiskFull {
        dir: PathBuf,
        needed: u64,
    },
    NotWritable {
        dir: PathBuf,
        root_owned: bool,
    },
    /// Windows: a file another program holds open; everything was put back.
    FileBusy {
        name: String,
    },
    /// The new program did not start; `rolled_back` when it had been installed.
    StartFailed {
        reason: String,
        rolled_back: bool,
    },
    /// The install folder holds files of the release's names that are not
    /// this install's.
    Foreign {
        dir: PathBuf,
        names: Vec<String>,
    },
    Archive(String),
    Io {
        what: String,
        error: io::Error,
        rolled_back: bool,
    },
    Cancelled,
    /// Another update of this home runs (`markitai update`, or a server's).
    Busy,
}

impl Error {
    /// The API's error code.
    pub(crate) fn code(&self) -> &'static str {
        match self {
            Self::Cancelled => "cancelled",
            Self::Busy => "update_running",
            Self::Fetch(_) => "offline",
            Self::Checksum { .. } | Self::Archive(_) => "checksum",
            Self::DiskFull { .. } => "disk_full",
            Self::NotWritable { .. } | Self::Foreign { .. } => "not_writable",
            Self::FileBusy { .. } => "file_busy",
            Self::StartFailed { .. } => "start_failed",
            Self::Io { .. } => "install_failed",
        }
    }

    /// Whether the files had been replaced and were put back (as opposed
    /// to nothing having changed).
    pub(crate) fn rolled_back(&self) -> bool {
        match self {
            Self::FileBusy { .. } => true,
            Self::StartFailed { rolled_back, .. } | Self::Io { rolled_back, .. } => *rolled_back,
            _ => false,
        }
    }

    /// The English reason, for logs and the API.
    pub(crate) fn reason(&self) -> String {
        match self {
            Self::Fetch(failure) => fetch_reason(failure),
            Self::Checksum { expected, actual } => format!(
                "SHA-256 mismatch; the download was deleted ({} != {})",
                short_digest(actual),
                short_digest(expected)
            ),
            Self::DiskFull { dir, needed } => format!(
                "Not enough disk space: {} needs {}",
                dir.display(),
                megabytes(*needed)
            ),
            Self::NotWritable { dir, .. } => {
                format!("Can't write to {}: permission denied", dir.display())
            }
            Self::FileBusy { name } => format!("Can't replace {name}: another program is using it"),
            Self::StartFailed { reason, .. } => format!("The new version didn't start: {reason}"),
            Self::Foreign { dir, names } => format!(
                "{} holds files this install did not create: {}",
                dir.display(),
                names.join(", ")
            ),
            Self::Archive(reason) => format!("The release archive is unusable: {reason}"),
            Self::Io { what, error, .. } => format!("{what}: {error}"),
            Self::Cancelled => "Cancelled".into(),
            Self::Busy => "Another update is running".into(),
        }
    }
}

/// Held while an update runs: one at a time per Markitai home.
#[derive(Debug)]
pub(crate) struct Exclusive(#[allow(dead_code)] std::fs::File);

pub(crate) fn exclusive(home: &Path) -> Result<Exclusive, Error> {
    let io = |error| Error::Io {
        what: "Can't take the update lock".into(),
        error,
        rolled_back: false,
    };
    markitai_core::output::create_private_dir(home).map_err(io)?;
    let file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(home.join("update.lock"))
        .map_err(io)?;
    match file.try_lock() {
        Ok(()) => Ok(Exclusive(file)),
        Err(std::fs::TryLockError::WouldBlock) => Err(Error::Busy),
        Err(std::fs::TryLockError::Error(error)) => Err(io(error)),
    }
}

pub(crate) fn fetch_reason(failure: &Failure) -> String {
    match failure {
        Failure::Offline { host, reason } => format!(
            "Can't reach {host}: {}",
            match *reason {
                "timeout" => "connection timed out",
                "refused" => "connection refused",
                _ => "no connection",
            }
        ),
        Failure::Status { host, code } => format!("{host} answered HTTP {code}"),
        Failure::Missing { what } => format!("No such release: {what}"),
        Failure::Malformed { what } => format!("Unreadable answer: {what}"),
        Failure::Setup(reason) => reason.clone(),
    }
}

/// `3f9a…c2`: enough of a digest to tell two apart.
pub(crate) fn short_digest(digest: &str) -> String {
    if digest.len() < 8 {
        return digest.to_owned();
    }
    format!("{}…{}", &digest[..4], &digest[digest.len() - 2..])
}

/// `31.2 MB` (decimal megabytes, as download sizes are spoken of).
pub(crate) fn megabytes(bytes: u64) -> String {
    if bytes < 100_000 {
        return format!("{} KB", bytes.div_ceil(1000));
    }
    let value = bytes as f64 / 1_000_000.0;
    if value >= 10.0 || bytes == 0 {
        format!("{value:.1} MB").replace(".0 MB", " MB")
    } else {
        format!("{value:.1} MB")
    }
}

/// The release to install: its version, its archive and that archive's digest.
#[derive(Clone, Debug)]
pub(crate) struct Release {
    pub(crate) version: Version,
    pub(crate) base: Base,
    pub(crate) archive: String,
    pub(crate) sha256: String,
}

impl Release {
    pub(crate) fn url(&self) -> String {
        self.base.asset(&self.version, &self.archive)
    }
}

pub(crate) fn resolve(
    base: &Base,
    version: &Version,
    target: &str,
    timeouts: Timeouts,
) -> Result<Release, Error> {
    let sums = check::sums(base, version, timeouts).map_err(Error::Fetch)?;
    let (sha256, archive) = check::archive_line(&sums, version, target).ok_or_else(|| {
        Error::Fetch(Failure::Missing {
            what: format!("markitai-v{version}-{target}"),
        })
    })?;
    Ok(Release {
        version: version.clone(),
        base: base.clone(),
        archive,
        sha256,
    })
}

/// Thirty-two random hexadecimal digits: the names setup.ps1 cleans up
/// (`*.old-<id>`, `.markitai-new-<id>`) use this form.
pub(crate) fn new_id() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}

#[cfg(test)]
thread_local! {
    /// A step name where tests make the next operation fail.
    pub(crate) static FAIL_AT: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
}

/// A failure injected by a test at `step`.
fn fault(step: &str) -> io::Result<()> {
    #[cfg(test)]
    {
        let hit = FAIL_AT.with(|fail| fail.borrow().as_deref() == Some(step));
        if hit {
            return Err(io::Error::from_raw_os_error(if step.starts_with("busy:") {
                32
            } else {
                28
            }));
        }
    }
    let _ = step;
    Ok(())
}

/// The work folder inside the install folder; removed when dropped unless kept.
#[derive(Debug)]
pub(crate) struct Work {
    pub(crate) path: PathBuf,
}

impl Drop for Work {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

fn not_writable(dir: &Path, error: &io::Error) -> Option<Error> {
    (error.kind() == io::ErrorKind::PermissionDenied || error.raw_os_error() == Some(30)).then(
        || Error::NotWritable {
            dir: dir.to_owned(),
            root_owned: owned_by_root(dir),
        },
    )
}

fn owned_by_root(dir: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        std::fs::metadata(dir).is_ok_and(|metadata| metadata.uid() == 0)
    }
    #[cfg(not(unix))]
    {
        let _ = dir;
        false
    }
}

/// The folders an update writes: the program's, and on `setup.sh` installs
/// the one holding `current`.
fn folders(install: &SetupInstall) -> Vec<PathBuf> {
    let mut folders = vec![install.dir.clone()];
    if install.script == Script::Shell
        && let Some(share) = install.receipt_dir.parent()
    {
        folders.push(share.to_owned());
    }
    folders
}

/// Create the work folder, which proves the install folder is writable, and
/// check the other folder can take files too. Nothing is downloaded yet.
pub(crate) fn prepare(install: &SetupInstall) -> Result<Work, Error> {
    for folder in folders(install).iter().skip(1) {
        match tempfile::tempfile_in(folder) {
            Ok(_) => {}
            Err(error) => {
                return Err(not_writable(folder, &error).unwrap_or(Error::Io {
                    what: format!("Can't write to {}", folder.display()),
                    error,
                    rolled_back: false,
                }));
            }
        }
    }
    let path = install.dir.join(format!(".markitai-new-{}", new_id()));
    if let Err(error) = std::fs::create_dir(&path).and_then(|()| fault("prepare")) {
        let _ = std::fs::remove_dir_all(&path);
        return Err(not_writable(&install.dir, &error).unwrap_or(Error::Io {
            what: format!("Can't write to {}", install.dir.display()),
            error,
            rolled_back: false,
        }));
    }
    Ok(Work { path })
}

/// Free bytes on the volume holding `dir`, when the system says.
pub(crate) fn available_space(dir: &Path) -> Option<u64> {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let path = std::ffi::CString::new(dir.as_os_str().as_bytes()).ok()?;
        let mut stats: libc::statvfs = unsafe { std::mem::zeroed() };
        // SAFETY: `path` is NUL-terminated and `stats` is a valid out pointer.
        if unsafe { libc::statvfs(path.as_ptr(), &mut stats) } != 0 {
            return None;
        }
        #[allow(clippy::unnecessary_cast)]
        Some(stats.f_bavail as u64 * stats.f_frsize as u64)
    }
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        let wide: Vec<u16> = dir.as_os_str().encode_wide().chain([0]).collect();
        let mut free = 0u64;
        // SAFETY: `wide` is NUL-terminated; the other outputs may be null.
        let ok = unsafe {
            windows_sys::Win32::Storage::FileSystem::GetDiskFreeSpaceExW(
                wide.as_ptr(),
                &mut free,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        (ok != 0).then_some(free)
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = dir;
        None
    }
}

/// Room an update of an archive this size needs: the archive, what it
/// unpacks to, and the old program kept aside.
pub(crate) fn space_needed(archive: u64) -> u64 {
    archive.saturating_mul(4).max(16 * 1024 * 1024)
}

#[derive(Debug)]
pub(crate) struct Downloaded {
    pub(crate) archive: PathBuf,
    pub(crate) size: u64,
}

/// Download the archive into the work folder, checking room before the body
/// and the digest after it. `progress(done, total)` follows the bytes;
/// `cancelled()` is asked between reads.
pub(crate) fn download(
    work: &Work,
    release: &Release,
    timeouts: Timeouts,
    progress: &mut dyn FnMut(u64, Option<u64>),
    cancelled: &dyn Fn() -> bool,
) -> Result<Downloaded, Error> {
    let mut response =
        check::open(&release.base, &release.url(), timeouts).map_err(Error::Fetch)?;
    let total = response.content_length();
    let dir = work.path.parent().unwrap_or(&work.path).to_owned();
    if let Some(total) = total {
        if total > ARCHIVE_LIMIT {
            return Err(Error::Archive("larger than any release".into()));
        }
        let needed = space_needed(total);
        if available_space(&dir).is_some_and(|free| free < needed) {
            return Err(Error::DiskFull { dir, needed });
        }
    }
    let path = work.path.join(&release.archive);
    let mut file = std::fs::File::create(&path).map_err(|error| Error::Io {
        what: "Can't save the download".into(),
        error,
        rolled_back: false,
    })?;
    let mut hash = Sha256::new();
    let mut done = 0u64;
    let mut buffer = vec![0u8; 64 * 1024];
    let host = release.base.host();
    loop {
        if cancelled() {
            return Err(Error::Cancelled);
        }
        let read = match response.read(&mut buffer) {
            Ok(read) => read,
            Err(_) if cancelled() => return Err(Error::Cancelled),
            Err(error) => {
                return Err(Error::Fetch(Failure::Offline {
                    host,
                    reason: if error.kind() == io::ErrorKind::TimedOut
                        || error.to_string().contains("timed out")
                    {
                        "timeout"
                    } else {
                        "unreachable"
                    },
                }));
            }
        };
        if read == 0 {
            break;
        }
        done += read as u64;
        if done > ARCHIVE_LIMIT {
            return Err(Error::Archive("larger than any release".into()));
        }
        hash.update(&buffer[..read]);
        if let Err(error) = file
            .write_all(&buffer[..read])
            .and_then(|()| fault("write"))
        {
            if error.raw_os_error() == Some(28) || error.kind() == io::ErrorKind::StorageFull {
                return Err(Error::DiskFull {
                    dir,
                    needed: space_needed(total.unwrap_or(done)),
                });
            }
            return Err(Error::Io {
                what: "Can't save the download".into(),
                error,
                rolled_back: false,
            });
        }
        progress(done, total);
    }
    drop(file);
    let actual = markitai_core::hex(hash.finalize());
    if actual != release.sha256 {
        let _ = std::fs::remove_file(&path);
        return Err(Error::Checksum {
            expected: release.sha256.clone(),
            actual,
        });
    }
    Ok(Downloaded {
        archive: path,
        size: done,
    })
}

/// A relative archive path made only of plain names.
fn safe_relative(path: &Path) -> Option<PathBuf> {
    let mut clean = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Normal(name) => clean.push(name),
            Component::CurDir => {}
            _ => return None,
        }
    }
    (!clean.as_os_str().is_empty()).then_some(clean)
}

fn archive_error(error: impl std::fmt::Display) -> Error {
    Error::Archive(error.to_string())
}

/// Unpack the archive into `work/unpacked`: plain files and folders only
/// (the Unix archive's `mkai`/`markitai-mcp` links are left out: the install
/// keeps its own), within size limits. Returns the unpacked folder.
pub(crate) fn unpack(work: &Work, downloaded: &Downloaded) -> Result<PathBuf, Error> {
    let destination = work.path.join("unpacked");
    // A full disk is said as such; other write errors as an unusable archive.
    let written_error = |error: io::Error| {
        if error.raw_os_error() == Some(28) || error.kind() == io::ErrorKind::StorageFull {
            Error::DiskFull {
                dir: work.path.parent().unwrap_or(&work.path).to_owned(),
                needed: space_needed(downloaded.size),
            }
        } else {
            archive_error(error)
        }
    };
    std::fs::create_dir(&destination).map_err(&written_error)?;
    let file = std::fs::File::open(&downloaded.archive).map_err(archive_error)?;
    let mut written = 0u64;
    let mut count = 0usize;
    let mut account = |size: u64| -> Result<(), Error> {
        written = written.saturating_add(size);
        count += 1;
        if written > UNPACKED_LIMIT || count > ENTRY_LIMIT {
            return Err(Error::Archive("unpacks to more than any release".into()));
        }
        Ok(())
    };
    if downloaded.archive.to_string_lossy().ends_with(".zip") {
        let mut archive = zip::ZipArchive::new(file).map_err(archive_error)?;
        for index in 0..archive.len() {
            let mut entry = archive.by_index(index).map_err(archive_error)?;
            let shown = entry
                .name()
                .map(|name| name.into_owned())
                .unwrap_or_default();
            let name = entry
                .enclosed_name()
                .and_then(|name| safe_relative(&name))
                .ok_or_else(|| Error::Archive(format!("unsafe member {shown}")))?;
            if entry.is_symlink() {
                return Err(Error::Archive(format!("link member {shown}")));
            }
            let target = destination.join(&name);
            if entry.is_dir() {
                std::fs::create_dir_all(&target).map_err(&written_error)?;
                continue;
            }
            account(entry.size())?;
            if let Some(parent) = target.parent() {
                std::fs::create_dir_all(parent).map_err(&written_error)?;
            }
            let mut output = std::fs::File::create(&target)
                .and_then(|output| fault("unpack").map(|()| output))
                .map_err(&written_error)?;
            let copied = io::copy(&mut (&mut entry).take(UNPACKED_LIMIT), &mut output)
                .map_err(&written_error)?;
            if copied != entry.size() {
                return Err(Error::Archive(format!("short member {shown}")));
            }
        }
    } else {
        let mut archive = tar::Archive::new(flate2::read::GzDecoder::new(file));
        for entry in archive.entries().map_err(archive_error)? {
            let mut entry = entry.map_err(archive_error)?;
            let raw = entry.path().map_err(archive_error)?.into_owned();
            let name = safe_relative(&raw)
                .ok_or_else(|| Error::Archive(format!("unsafe member {}", raw.display())))?;
            let target = destination.join(&name);
            match entry.header().entry_type() {
                tar::EntryType::Directory => {
                    std::fs::create_dir_all(&target).map_err(&written_error)?;
                }
                tar::EntryType::Regular | tar::EntryType::Continuous => {
                    account(entry.size())?;
                    if let Some(parent) = target.parent() {
                        std::fs::create_dir_all(parent).map_err(&written_error)?;
                    }
                    let mut output = std::fs::File::create(&target)
                        .and_then(|output| fault("unpack").map(|()| output))
                        .map_err(&written_error)?;
                    io::copy(&mut (&mut entry).take(UNPACKED_LIMIT), &mut output)
                        .map_err(&written_error)?;
                    #[cfg(unix)]
                    {
                        use std::os::unix::fs::PermissionsExt;
                        let mode = entry.header().mode().unwrap_or(0o644);
                        let mode = if mode & 0o111 != 0 { 0o755 } else { 0o644 };
                        std::fs::set_permissions(&target, std::fs::Permissions::from_mode(mode))
                            .map_err(archive_error)?;
                    }
                }
                tar::EntryType::Symlink
                    if matches!(name.to_str(), Some("mkai" | "markitai-mcp"))
                        && entry
                            .link_name()
                            .ok()
                            .flatten()
                            .is_some_and(|link| link.as_os_str() == "markitai") => {}
                other => {
                    return Err(Error::Archive(format!(
                        "member {} of type {other:?}",
                        raw.display()
                    )));
                }
            }
        }
    }
    let _ = std::fs::remove_file(&downloaded.archive);
    Ok(destination)
}

/// The program inside an unpacked release.
pub(crate) fn program(unpacked: &Path, script: Script) -> Result<PathBuf, Error> {
    let required: &[&str] = match script {
        Script::Shell => &["markitai"],
        Script::PowerShell => &["markitai.exe", "mkai.exe", "markitai-mcp.exe"],
    };
    for name in required {
        if !unpacked.join(name).is_file() {
            return Err(Error::Archive(format!("{name} is missing")));
        }
    }
    Ok(unpacked.join(required[0]))
}

/// Run the unpacked program with `arguments` (`--version`, or `serve
/// --preflight …`) and require success within a minute. `expect` must
/// appear in what it prints, when given.
pub(crate) fn dry_start(
    program: &Path,
    arguments: &[OsString],
    expect: Option<&str>,
) -> Result<(), Error> {
    let started = std::process::Command::new(program)
        .args(arguments)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .env_remove("MARKITAI_TEST_UPDATE_AS")
        .spawn();
    let mut child = started.map_err(|error| Error::StartFailed {
        reason: error.to_string(),
        rolled_back: false,
    })?;
    let deadline = std::time::Instant::now() + Duration::from_secs(60);
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if std::time::Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(20));
            }
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(Error::StartFailed {
                    reason: "it did not finish within a minute".into(),
                    rolled_back: false,
                });
            }
        }
    };
    let mut stdout = String::new();
    let mut stderr = String::new();
    if let Some(mut out) = child.stdout.take() {
        let _ = (&mut out).take(64 * 1024).read_to_string(&mut stdout);
    }
    if let Some(mut err) = child.stderr.take() {
        let _ = (&mut err).take(64 * 1024).read_to_string(&mut stderr);
    }
    if !status.success() {
        let reason = stderr
            .lines()
            .chain(stdout.lines())
            .map(str::trim)
            .rfind(|line| !line.is_empty())
            .map(|line| line.trim_start_matches(['✗', 'x', ' ']).to_owned())
            .unwrap_or_else(|| format!("exit status {}", status.code().unwrap_or(-1)));
        return Err(Error::StartFailed {
            reason,
            rolled_back: false,
        });
    }
    if let Some(expect) = expect
        && !stdout.contains(expect)
    {
        return Err(Error::StartFailed {
            reason: format!("it reports {}", stdout.trim()),
            rolled_back: false,
        });
    }
    Ok(())
}

/// The receipt the new release gets: the old one's fields, with the new
/// version, the base it came from and the receipt format.
fn new_receipt(install: &SetupInstall, release: &Release, names: Option<&[String]>) -> Receipt {
    let mut receipt = install.receipt.clone();
    receipt.set("version", &release.version.to_string());
    receipt.set("target", &install.target());
    if install.script == Script::Shell {
        receipt.set("bin_dir", &install.dir.display().to_string());
        if receipt.get("installer").is_none() {
            receipt.set("installer", "setup.sh");
        }
    }
    receipt.set("base", release.base.as_str());
    receipt.set("format", "1");
    if let Some(names) = names {
        receipt.set_all("file", names);
    }
    receipt
}

/// What a swap moved, to commit or undo it.
#[derive(Debug, Default, serde::Serialize, serde::Deserialize, PartialEq, Eq, Clone)]
pub(crate) struct Swapped {
    /// (installed path, where the old one went).
    pub(crate) aside: Vec<(PathBuf, PathBuf)>,
    /// New paths placed where nothing was before.
    pub(crate) placed: Vec<PathBuf>,
    /// The old program, kept for a rollback until committed.
    pub(crate) old_program: Option<PathBuf>,
}

fn rename(from: &Path, to: &Path, step: &str) -> io::Result<()> {
    fault(step)?;
    markitai_core::platform::rename(from, to)
}

fn remove_any(path: &Path) -> io::Result<()> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_dir() => std::fs::remove_dir_all(path),
        Ok(_) => std::fs::remove_file(path),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

impl Swapped {
    /// Put every moved file back: new ones removed, old ones renamed back.
    pub(crate) fn rollback(&self) -> io::Result<()> {
        let mut first = None;
        for path in self.placed.iter().rev() {
            if let Err(error) = remove_any(path) {
                first.get_or_insert(error);
            }
        }
        for (installed, aside) in self.aside.iter().rev() {
            let restored = if std::fs::symlink_metadata(aside).is_ok() {
                // A file is renamed over the new one; a folder must go first.
                if std::fs::symlink_metadata(installed).is_ok_and(|metadata| metadata.is_dir()) {
                    let _ = remove_any(installed);
                }
                markitai_core::platform::rename(aside, installed)
            } else {
                Ok(())
            };
            if let Err(error) = restored {
                first.get_or_insert(error);
            }
        }
        first.map_or(Ok(()), Err)
    }

    /// Delete what was set aside. A running Windows EXE cannot be deleted
    /// yet; its `.old-<id>` name stays for the next update (or setup) to
    /// remove. Returns whether something stayed.
    pub(crate) fn commit(&self) -> bool {
        let mut left = false;
        for (_, aside) in &self.aside {
            if remove_any(aside).is_err() {
                left = true;
            }
        }
        left
    }
}

/// Install the unpacked release over `install`. On an error everything
/// is back as it was. The old program stays aside until `commit`.
pub(crate) fn swap(
    install: &SetupInstall,
    release: &Release,
    unpacked: &Path,
) -> Result<Swapped, Error> {
    let id = new_id();
    match install.script {
        Script::Shell => swap_split(install, release, unpacked, &id),
        Script::PowerShell => swap_folder(install, release, unpacked, &id),
    }
}

fn io_error(what: String, error: io::Error, rolled_back: bool) -> Error {
    Error::Io {
        what,
        error,
        rolled_back,
    }
}

/// `setup.sh`: the program by one rename (the old one hard-linked aside
/// first, so `bin_dir/markitai` never disappears), then `current` replaced
/// by renames like setup.sh does.
fn swap_split(
    install: &SetupInstall,
    release: &Release,
    unpacked: &Path,
    id: &str,
) -> Result<Swapped, Error> {
    let program = install.dir.join("markitai");
    let kept = install.dir.join(format!(".markitai.old-{id}"));
    let mut swapped = Swapped::default();
    if program.exists() {
        let linked = fault("keep").and_then(|()| std::fs::hard_link(&program, &kept));
        if linked.is_err() {
            std::fs::copy(&program, &kept)
                .map_err(|error| io_error("Can't keep the old program".into(), error, false))?;
        }
        swapped.aside.push((program.clone(), kept.clone()));
        swapped.old_program = Some(kept.clone());
    }
    let undo = |swapped: &Swapped, error: Error| {
        let _ = swapped.rollback();
        error
    };
    if let Err(error) = rename(&unpacked.join("markitai"), &program, "program") {
        let error = not_writable(&install.dir, &error).unwrap_or_else(|| {
            io_error(format!("Can't replace {}", program.display()), error, false)
        });
        let _ = remove_any(&kept);
        return Err(error);
    }
    // The links setup made stay; missing ones come back.
    #[cfg(unix)]
    for name in ["mkai", "markitai-mcp"] {
        let link = install.dir.join(name);
        if std::fs::symlink_metadata(&link).is_err() {
            let _ = std::os::unix::fs::symlink("markitai", &link);
        }
    }
    let current = install.receipt_dir.clone();
    let Some(share) = current.parent().map(Path::to_owned) else {
        return Ok(swapped);
    };
    let fresh = share.join("current.new");
    let old = share.join("current.old");
    let mut share_step = || -> io::Result<()> {
        let _ = remove_any(&fresh);
        let _ = remove_any(&old);
        for name in ["markitai", "mkai", "markitai-mcp"] {
            let _ = remove_any(&unpacked.join(name));
        }
        fault("share")?;
        if markitai_core::platform::rename(unpacked, &fresh).is_err() {
            copy_tree(unpacked, &fresh)?;
        }
        let receipt = new_receipt(install, release, None);
        std::fs::write(fresh.join("receipt"), receipt.render())?;
        if current.exists() {
            rename(&current, &old, "current")?;
            swapped.aside.push((current.clone(), old.clone()));
        } else {
            swapped.placed.push(current.clone());
        }
        rename(&fresh, &current, "current.new")
    };
    if let Err(error) = share_step() {
        let _ = remove_any(&fresh);
        return Err(undo(
            &swapped,
            io_error(format!("Can't replace {}", current.display()), error, true),
        ));
    }
    Ok(swapped)
}

fn copy_tree(from: &Path, to: &Path) -> io::Result<()> {
    std::fs::create_dir_all(to)?;
    for entry in std::fs::read_dir(from)? {
        let entry = entry?;
        let target = to.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_tree(&entry.path(), &target)?;
        } else {
            std::fs::copy(entry.path(), target)?;
        }
    }
    Ok(())
}

/// Sharing violation, lock violation or access denied after the folder
/// proved writable: another program holds the file open.
fn busy(error: &io::Error) -> bool {
    (cfg!(windows) || cfg!(test)) && matches!(error.raw_os_error(), Some(5 | 32 | 33))
}

/// `setup.ps1`: every name the old receipt lists and the release brings is
/// renamed aside, then the new ones are renamed in from the work folder.
fn swap_folder(
    install: &SetupInstall,
    release: &Release,
    unpacked: &Path,
    id: &str,
) -> Result<Swapped, Error> {
    let dir = &install.dir;
    let mut names: Vec<String> = std::fs::read_dir(unpacked)
        .map_err(|error| io_error("Can't read the release".into(), error, false))?
        .filter_map(|entry| entry.ok()?.file_name().into_string().ok())
        .filter(|name| name != "receipt")
        .collect();
    names.sort();
    let owned: Vec<String> = install
        .receipt
        .all("file")
        .into_iter()
        .map(str::to_owned)
        .collect();
    let owns = |name: &str| owned.iter().any(|own| own.eq_ignore_ascii_case(name));
    let foreign: Vec<String> = names
        .iter()
        .filter(|name| !owns(name) && dir.join(name).exists())
        .cloned()
        .collect();
    if !foreign.is_empty() {
        return Err(Error::Foreign {
            dir: dir.clone(),
            names: foreign,
        });
    }
    let receipt = new_receipt(install, release, Some(&names));
    std::fs::write(unpacked.join("receipt"), receipt.render())
        .map_err(|error| io_error("Can't write the receipt".into(), error, false))?;
    let mut moving: Vec<String> = owned
        .iter()
        .filter(|name| {
            safe_relative(Path::new(name)).is_some_and(|path| path.components().count() == 1)
        })
        .cloned()
        .collect();
    for name in names.iter().chain(std::iter::once(&"receipt".to_owned())) {
        if !moving.iter().any(|known| known.eq_ignore_ascii_case(name)) {
            moving.push(name.clone());
        }
    }
    let present: Vec<&str> = moving
        .iter()
        .filter(|name| std::fs::symlink_metadata(dir.join(name)).is_ok())
        .map(String::as_str)
        .collect();
    let placing: Vec<&str> = names
        .iter()
        .map(String::as_str)
        .chain(std::iter::once("receipt"))
        .collect();
    let journal = write_journal(dir, id, &present, &placing)
        .map_err(|error| io_error("Can't record the update".into(), error, false))?;
    let mut swapped = Swapped::default();
    let fail = |swapped: &Swapped, name: &str, error: io::Error| {
        // Kept when something stayed: the next start finishes the undo.
        if swapped.rollback().is_ok() {
            let _ = std::fs::remove_file(&journal);
        }
        if busy(&error) {
            Error::FileBusy {
                name: name.to_owned(),
            }
        } else {
            io_error(format!("Can't replace {name}"), error, true)
        }
    };
    for name in present {
        let path = dir.join(name);
        let aside = dir.join(format!("{name}.old-{id}"));
        if let Err(error) = rename(&path, &aside, &format!("busy:{name}")) {
            return Err(fail(&swapped, name, error));
        }
        if name.eq_ignore_ascii_case("markitai.exe") {
            swapped.old_program = Some(aside.clone());
        }
        swapped.aside.push((path, aside));
    }
    // The tests' kill between the two loops: nothing is put back here.
    fault("killed").map_err(|error| io_error("Killed".into(), error, false))?;
    for name in placing {
        let path = dir.join(name);
        if let Err(error) = rename(&unpacked.join(name), &path, &format!("place:{name}")) {
            return Err(fail(&swapped, name, error));
        }
        swapped.placed.push(path);
    }
    // A journal left behind would undo this install later: then undo it now.
    if let Err(error) = std::fs::remove_file(&journal) {
        return Err(fail(&swapped, "receipt", error));
    }
    Ok(swapped)
}

/// A folder swap's journal: `.markitai-swap-<id>`.
const JOURNAL: &str = ".markitai-swap-";

fn is_id(id: &str) -> bool {
    id.len() == 32 && id.bytes().all(|byte| byte.is_ascii_hexdigit())
}

/// Record a folder swap before its first rename: the names set aside as
/// `<name>.old-<id>` (`aside=`) and the names placed (`place=`).
fn write_journal(dir: &Path, id: &str, aside: &[&str], place: &[&str]) -> io::Result<PathBuf> {
    let path = dir.join(format!("{JOURNAL}{id}"));
    let mut text = String::from("format=1\n");
    for name in aside {
        text.push_str(&format!("aside={name}\n"));
    }
    for name in place {
        text.push_str(&format!("place={name}\n"));
    }
    let mut file = std::fs::File::create(&path)?;
    file.write_all(text.as_bytes())?;
    file.sync_all()?;
    Ok(path)
}

/// Undo the folder swaps a killed run left half done: the names it placed
/// go, the copies it set aside come back. Run under the update lock (or by
/// setup.ps1). Returns the ids of journals that could not be undone whole;
/// their copies are not leftovers.
pub(crate) fn recover(dir: &Path) -> Vec<String> {
    let mut stuck = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return stuck;
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some(id) = name.strip_prefix(JOURNAL).filter(|id| is_id(id)) else {
            continue;
        };
        let undone = std::fs::read_to_string(entry.path())
            .and_then(|text| undo_journal(dir, id, &text))
            .and_then(|()| std::fs::remove_file(entry.path()));
        if undone.is_err() {
            stuck.push(id.to_owned());
        }
    }
    stuck
}

fn undo_journal(dir: &Path, id: &str, text: &str) -> io::Result<()> {
    let list = |key: &str| -> Vec<&str> {
        text.lines()
            .filter_map(|line| line.strip_prefix(key)?.strip_prefix('='))
            .filter(|name| {
                safe_relative(Path::new(name)).is_some_and(|path| path.components().count() == 1)
            })
            .collect()
    };
    let (aside, place) = (list("aside"), list("place"));
    let mut first = None;
    // A placed name nothing was set aside for did not exist before.
    for name in place
        .iter()
        .filter(|name| !aside.iter().any(|old| old.eq_ignore_ascii_case(name)))
    {
        if let Err(error) = remove_any(&dir.join(name)) {
            first.get_or_insert(error);
        }
    }
    // A name never set aside is still the old one.
    for name in aside {
        let kept = dir.join(format!("{name}.old-{id}"));
        if std::fs::symlink_metadata(&kept).is_err() {
            continue;
        }
        let path = dir.join(name);
        let _ = remove_any(&path);
        if let Err(error) = markitai_core::platform::rename(&kept, &path) {
            first.get_or_insert(error);
        }
    }
    first.map_or(Ok(()), Err)
}

/// Remove what earlier updates left in the install folder: work folders and
/// old copies (`.markitai-new-<id>`, `*.old-<id>`), except `keep`. A swap a
/// killed run left half done is undone first.
pub(crate) fn remove_leftovers(install: &SetupInstall, keep: &[PathBuf]) {
    let stuck = recover(&install.dir);
    let Ok(entries) = std::fs::read_dir(&install.dir) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let leftover = name
            .strip_prefix(".markitai-new-")
            .or_else(|| name.strip_prefix(".markitai.old-"))
            .or_else(|| name.rsplit_once(".old-").map(|(_, id)| id))
            .is_some_and(|id| is_id(id) && !stuck.iter().any(|stuck| stuck == id));
        if leftover && !keep.contains(&entry.path()) {
            let _ = remove_any(&entry.path());
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::update::method::Receipt;

    pub(crate) fn tar_gz(files: &[(&str, &[u8], u32)], links: &[(&str, &str)]) -> Vec<u8> {
        let mut builder = tar::Builder::new(flate2::write::GzEncoder::new(
            Vec::new(),
            flate2::Compression::fast(),
        ));
        for (name, bytes, mode) in files {
            let mut header = tar::Header::new_gnu();
            header.set_size(bytes.len() as u64);
            header.set_mode(*mode);
            header.set_entry_type(tar::EntryType::Regular);
            // Written raw: the builder itself refuses the unsafe names tests need.
            header.as_old_mut().name[..name.len()].copy_from_slice(name.as_bytes());
            header.set_cksum();
            builder.append(&header, *bytes).unwrap();
        }
        for (name, target) in links {
            let mut header = tar::Header::new_gnu();
            header.set_size(0);
            header.set_entry_type(tar::EntryType::Symlink);
            builder.append_link(&mut header, name, target).unwrap();
        }
        builder.into_inner().unwrap().finish().unwrap()
    }

    pub(crate) fn zip_bytes(files: &[(&str, &[u8])]) -> Vec<u8> {
        let mut writer = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
        for (name, bytes) in files {
            writer
                .start_file(*name, zip::write::SimpleFileOptions::default())
                .unwrap();
            writer.write_all(bytes).unwrap();
        }
        writer.finish().unwrap().into_inner()
    }

    fn release(version: &str, archive: &str, sha256: &str) -> Release {
        Release {
            version: Version::parse(version).unwrap(),
            base: Base::parse("https://github.com/Ynewtime/markitai/releases").unwrap(),
            archive: archive.into(),
            sha256: sha256.into(),
        }
    }

    fn fail_at(step: Option<&str>) {
        FAIL_AT.with(|fail| *fail.borrow_mut() = step.map(str::to_owned));
    }

    /// A setup.sh install of 2.1.0 under `root`.
    fn shell_install(root: &Path) -> SetupInstall {
        let bin = root.join("bin");
        let current = root.join("share/markitai/current");
        std::fs::create_dir_all(&bin).unwrap();
        std::fs::create_dir_all(current.join("licenses")).unwrap();
        std::fs::write(bin.join("markitai"), "old program").unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink("markitai", bin.join("mkai")).unwrap();
        std::fs::write(current.join("LICENSE"), "old licence").unwrap();
        let receipt = format!(
            "version=2.1.0\ntarget=x86_64-unknown-linux-gnu\nbin_dir={}\n",
            bin.display()
        );
        std::fs::write(current.join("receipt"), &receipt).unwrap();
        SetupInstall {
            script: Script::Shell,
            exe: bin.join("markitai"),
            dir: bin,
            receipt_dir: current,
            receipt: Receipt::parse(&receipt),
        }
    }

    fn unpacked_release(work: &Path, files: &[(&str, &str)]) -> PathBuf {
        let unpacked = work.join("unpacked");
        for (name, text) in files {
            let path = unpacked.join(name);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, text).unwrap();
        }
        unpacked
    }

    fn read(path: &Path) -> String {
        std::fs::read_to_string(path).unwrap()
    }

    #[test]
    fn a_shell_install_is_swapped_by_renames_and_its_receipt_rewritten() {
        let root = tempfile::tempdir().unwrap();
        let install = shell_install(root.path());
        let work = prepare(&install).unwrap();
        let unpacked = unpacked_release(
            &work.path,
            &[("markitai", "new program"), ("LICENSE", "new licence")],
        );
        let release = release(
            "2.2.0",
            "markitai-v2.2.0-x86_64-unknown-linux-gnu.tar.gz",
            "",
        );
        let swapped = swap(&install, &release, &unpacked).unwrap();
        assert_eq!(read(&install.dir.join("markitai")), "new program");
        assert_eq!(read(&install.receipt_dir.join("LICENSE")), "new licence");
        let receipt = Receipt::parse(&read(&install.receipt_dir.join("receipt")));
        assert_eq!(receipt.get("version"), Some("2.2.0"));
        assert_eq!(receipt.get("installer"), Some("setup.sh"));
        assert_eq!(receipt.get("format"), Some("1"));
        assert_eq!(
            receipt.get("base"),
            Some("https://github.com/Ynewtime/markitai/releases")
        );
        assert_eq!(
            receipt.get("bin_dir").map(PathBuf::from),
            Some(install.dir.clone())
        );
        // The old program waits aside for a rollback until committed.
        let kept = swapped.old_program.clone().unwrap();
        assert_eq!(read(&kept), "old program");
        #[cfg(unix)]
        assert!(std::fs::symlink_metadata(install.dir.join("markitai-mcp")).is_ok());
        // A rollback after the swap (the new server did not start).
        swapped.rollback().unwrap();
        assert_eq!(read(&install.dir.join("markitai")), "old program");
        assert_eq!(read(&install.receipt_dir.join("LICENSE")), "old licence");
        assert_eq!(
            Receipt::parse(&read(&install.receipt_dir.join("receipt"))).get("version"),
            Some("2.1.0")
        );
    }

    #[test]
    fn a_failure_in_the_middle_of_a_shell_swap_puts_everything_back() {
        for step in ["program", "share", "current", "current.new"] {
            let root = tempfile::tempdir().unwrap();
            let install = shell_install(root.path());
            let work = prepare(&install).unwrap();
            let unpacked = unpacked_release(
                &work.path,
                &[("markitai", "new program"), ("LICENSE", "new licence")],
            );
            fail_at(Some(step));
            let error = swap(&install, &release("2.2.0", "a.tar.gz", ""), &unpacked).unwrap_err();
            fail_at(None);
            assert_eq!(read(&install.dir.join("markitai")), "old program", "{step}");
            assert_eq!(
                read(&install.receipt_dir.join("LICENSE")),
                "old licence",
                "{step}"
            );
            assert_eq!(
                Receipt::parse(&read(&install.receipt_dir.join("receipt"))).get("version"),
                Some("2.1.0"),
                "{step}"
            );
            assert_eq!(error.rolled_back(), step != "program", "{step}: {error:?}");
            // Nothing is left beside the program but the work folder.
            drop(work);
            let names: Vec<String> = std::fs::read_dir(&install.dir)
                .unwrap()
                .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
                .collect();
            assert!(
                names
                    .iter()
                    .all(|name| ["markitai", "mkai", "markitai-mcp"].contains(&name.as_str())),
                "{step}: {names:?}"
            );
        }
    }

    /// A setup.ps1 install of 2.1.0 (the same renames work on every system).
    fn folder_install(root: &Path) -> SetupInstall {
        let dir = root.join("Programs/markitai");
        std::fs::create_dir_all(dir.join("docs")).unwrap();
        for name in ["markitai.exe", "mkai.exe", "markitai-mcp.exe", "LICENSE"] {
            std::fs::write(dir.join(name), format!("old {name}")).unwrap();
        }
        std::fs::write(dir.join("docs/index.md"), "old docs").unwrap();
        std::fs::write(dir.join("notes.txt"), "the user's own file").unwrap();
        let receipt = "version=2.1.0\ntarget=x86_64-pc-windows-msvc\npath_added=1\nfile=LICENSE\nfile=docs\nfile=markitai-mcp.exe\nfile=markitai.exe\nfile=mkai.exe\n";
        std::fs::write(dir.join("receipt"), receipt).unwrap();
        SetupInstall {
            script: Script::PowerShell,
            exe: dir.join("markitai.exe"),
            dir: dir.clone(),
            receipt_dir: dir,
            receipt: Receipt::parse(receipt),
        }
    }

    fn new_folder_release(work: &Path) -> PathBuf {
        unpacked_release(
            work,
            &[
                ("markitai.exe", "new markitai.exe"),
                ("mkai.exe", "new mkai.exe"),
                ("markitai-mcp.exe", "new markitai-mcp.exe"),
                ("LICENSE", "new LICENSE"),
                ("docs/index.md", "new docs"),
                ("NOTICE", "new NOTICE"),
            ],
        )
    }

    #[test]
    fn a_folder_install_renames_aside_then_in_and_keeps_the_users_files() {
        let root = tempfile::tempdir().unwrap();
        let install = folder_install(root.path());
        let work = prepare(&install).unwrap();
        let unpacked = new_folder_release(&work.path);
        let swapped = swap(
            &install,
            &release("2.2.0", "markitai-v2.2.0-x86_64-pc-windows-msvc.zip", ""),
            &unpacked,
        )
        .unwrap();
        assert_eq!(
            read(&install.dir.join("markitai-mcp.exe")),
            "new markitai-mcp.exe"
        );
        assert_eq!(read(&install.dir.join("NOTICE")), "new NOTICE");
        assert_eq!(read(&install.dir.join("docs/index.md")), "new docs");
        assert_eq!(read(&install.dir.join("notes.txt")), "the user's own file");
        let receipt = Receipt::parse(&read(&install.dir.join("receipt")));
        assert_eq!(receipt.get("path_added"), Some("1"));
        assert_eq!(receipt.get("version"), Some("2.2.0"));
        assert_eq!(receipt.get("format"), Some("1"));
        assert_eq!(
            receipt.all("file"),
            [
                "LICENSE",
                "NOTICE",
                "docs",
                "markitai-mcp.exe",
                "markitai.exe",
                "mkai.exe"
            ]
        );
        assert!(swapped.old_program.as_ref().unwrap().exists());
        assert!(!swapped.commit());
        drop(work);
        remove_leftovers(&install, &[]);
        let mut names: Vec<String> = std::fs::read_dir(&install.dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        assert_eq!(
            names,
            [
                "LICENSE",
                "NOTICE",
                "docs",
                "markitai-mcp.exe",
                "markitai.exe",
                "mkai.exe",
                "notes.txt",
                "receipt"
            ]
        );
    }

    #[test]
    fn a_file_in_use_rolls_the_folder_back_whole() {
        for step in ["busy:markitai-mcp.exe", "place:mkai.exe", "place:receipt"] {
            let root = tempfile::tempdir().unwrap();
            let install = folder_install(root.path());
            let work = prepare(&install).unwrap();
            let unpacked = new_folder_release(&work.path);
            fail_at(Some(step));
            let error = swap(&install, &release("2.2.0", "a.zip", ""), &unpacked).unwrap_err();
            fail_at(None);
            if step.starts_with("busy:") {
                assert!(
                    matches!(&error, Error::FileBusy { name } if name == "markitai-mcp.exe"),
                    "{error:?}"
                );
                assert_eq!(error.code(), "file_busy");
            }
            assert!(error.rolled_back(), "{step}");
            for name in ["markitai.exe", "mkai.exe", "markitai-mcp.exe", "LICENSE"] {
                assert_eq!(
                    read(&install.dir.join(name)),
                    format!("old {name}"),
                    "{step}"
                );
            }
            assert_eq!(read(&install.dir.join("docs/index.md")), "old docs");
            assert!(!install.dir.join("NOTICE").exists(), "{step}");
            assert_eq!(
                Receipt::parse(&read(&install.dir.join("receipt"))).get("version"),
                Some("2.1.0")
            );
        }
    }

    #[test]
    fn a_swap_killed_between_its_renames_is_undone_by_the_next_start() {
        let root = tempfile::tempdir().unwrap();
        let install = folder_install(root.path());
        let work = prepare(&install).unwrap();
        let unpacked = new_folder_release(&work.path);
        // Killed after every old name is aside and before any new one is in.
        fail_at(Some("killed"));
        let error = swap(&install, &release("2.2.0", "a.zip", ""), &unpacked).unwrap_err();
        fail_at(None);
        assert!(!error.rolled_back());
        assert!(!install.dir.join("markitai.exe").exists());
        // As a killed run would: the work folder stays, half moved out.
        std::fs::rename(unpacked.join("NOTICE"), install.dir.join("NOTICE")).unwrap();
        let journal: Vec<String> = std::fs::read_dir(&install.dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|name| name.starts_with(JOURNAL))
            .collect();
        assert_eq!(journal.len(), 1, "{journal:?}");
        std::mem::forget(work);
        // The next update's cleanup puts the old files back, not deletes them.
        remove_leftovers(&install, &[]);
        for name in ["markitai.exe", "mkai.exe", "markitai-mcp.exe", "LICENSE"] {
            assert_eq!(read(&install.dir.join(name)), format!("old {name}"));
        }
        assert_eq!(read(&install.dir.join("docs/index.md")), "old docs");
        assert_eq!(
            Receipt::parse(&read(&install.dir.join("receipt"))).get("version"),
            Some("2.1.0")
        );
        let mut names: Vec<String> = std::fs::read_dir(&install.dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        assert_eq!(
            names,
            [
                "LICENSE",
                "docs",
                "markitai-mcp.exe",
                "markitai.exe",
                "mkai.exe",
                "notes.txt",
                "receipt"
            ]
        );
    }

    #[test]
    fn a_journal_that_cannot_be_undone_keeps_its_copies() {
        let root = tempfile::tempdir().unwrap();
        let install = folder_install(root.path());
        let id = "0123456789abcdef0123456789abcdef";
        std::fs::rename(
            install.dir.join("LICENSE"),
            install.dir.join(format!("LICENSE.old-{id}")),
        )
        .unwrap();
        // Its old copy cannot go back over a folder that will not go.
        std::fs::create_dir(install.dir.join("LICENSE")).unwrap();
        write_journal(&install.dir, id, &["LICENSE"], &["LICENSE"]).unwrap();
        // A folder in the way of the rename back: it holds a file and is
        // read-only, so it cannot be removed.
        let blocked = install.dir.join("LICENSE");
        std::fs::write(blocked.join("x"), "x").unwrap();
        let mut permissions = std::fs::metadata(&blocked).unwrap().permissions();
        permissions.set_readonly(true);
        std::fs::set_permissions(&blocked, permissions.clone()).unwrap();
        let stuck = recover(&install.dir);
        if !stuck.is_empty() {
            assert_eq!(stuck, [id]);
            remove_leftovers(&install, &[]);
            assert!(install.dir.join(format!("LICENSE.old-{id}")).exists());
        }
        #[allow(clippy::permissions_set_readonly_false)]
        permissions.set_readonly(false);
        std::fs::set_permissions(&blocked, permissions).unwrap();
        // Once it can, the next start finishes the undo.
        remove_leftovers(&install, &[]);
        assert_eq!(read(&install.dir.join("LICENSE")), "old LICENSE");
    }

    #[test]
    fn a_folder_holding_someone_elses_file_of_a_release_name_is_refused() {
        let root = tempfile::tempdir().unwrap();
        let install = folder_install(root.path());
        std::fs::write(install.dir.join("NOTICE"), "another program's notice").unwrap();
        let work = prepare(&install).unwrap();
        let unpacked = new_folder_release(&work.path);
        let error = swap(&install, &release("2.2.0", "a.zip", ""), &unpacked).unwrap_err();
        assert!(
            matches!(&error, Error::Foreign { names, .. } if names == &["NOTICE"]),
            "{error:?}"
        );
        assert_eq!(read(&install.dir.join("markitai.exe")), "old markitai.exe");
    }

    #[test]
    fn archives_unpack_plain_files_only_and_the_digest_must_match() {
        let root = tempfile::tempdir().unwrap();
        let install = shell_install(root.path());
        let work = prepare(&install).unwrap();
        let bytes = tar_gz(
            &[
                ("markitai", b"#!/bin/sh\n", 0o755),
                ("docs/index.md", b"docs", 0o644),
            ],
            &[("mkai", "markitai"), ("markitai-mcp", "markitai")],
        );
        let archive = work.path.join("a.tar.gz");
        std::fs::write(&archive, &bytes).unwrap();
        // A full disk while unpacking is said as such.
        fail_at(Some("unpack"));
        let full = unpack(
            &work,
            &Downloaded {
                archive: archive.clone(),
                size: bytes.len() as u64,
            },
        )
        .unwrap_err();
        fail_at(None);
        assert_eq!(full.code(), "disk_full", "{full:?}");
        std::fs::remove_dir_all(work.path.join("unpacked")).unwrap();
        let unpacked = unpack(
            &work,
            &Downloaded {
                archive,
                size: bytes.len() as u64,
            },
        )
        .unwrap();
        assert_eq!(
            program(&unpacked, Script::Shell).unwrap(),
            unpacked.join("markitai")
        );
        assert!(!unpacked.join("mkai").exists());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(unpacked.join("markitai"))
                .unwrap()
                .permissions()
                .mode();
            assert_eq!(mode & 0o777, 0o755);
        }
        // A link anywhere else, or a path out of the folder, is refused.
        for (files, links) in [
            (vec![], vec![("docs", "/etc")]),
            (vec![("../escape", b"x".as_slice(), 0o644)], vec![]),
            (vec![("/abs", b"x".as_slice(), 0o644)], vec![]),
        ] {
            let work = prepare(&install).unwrap();
            let archive = work.path.join("bad.tar.gz");
            std::fs::write(&archive, tar_gz(&files, &links)).unwrap();
            let unpacked = unpack(&work, &Downloaded { archive, size: 0 });
            assert!(matches!(unpacked, Err(Error::Archive(_))), "{unpacked:?}");
        }
        let work = prepare(&install).unwrap();
        let archive = work.path.join("a.zip");
        std::fs::write(
            &archive,
            zip_bytes(&[
                ("markitai.exe", b"MZ"),
                ("mkai.exe", b"MZ"),
                ("markitai-mcp.exe", b"MZ"),
                ("docs/cli.md", b"x"),
            ]),
        )
        .unwrap();
        let unpacked = unpack(&work, &Downloaded { archive, size: 0 }).unwrap();
        assert!(program(&unpacked, Script::PowerShell).is_ok());
        assert!(matches!(
            program(&unpacked, Script::Shell),
            Err(Error::Archive(_))
        ));
    }

    #[test]
    fn a_download_is_checked_for_room_and_digest_and_cancels_cleanly() {
        let body = b"archive bytes".to_vec();
        let digest = markitai_core::hex(Sha256::digest(&body));
        let (base, _) = crate::update::check::tests::serve(vec![(
            "/releases/download/v2.2.0/a.tar.gz",
            200,
            vec![],
            body.clone(),
        )]);
        let root = tempfile::tempdir().unwrap();
        let install = shell_install(root.path());
        let mut good = release("2.2.0", "a.tar.gz", &digest);
        good.base = Base::parse(&base).unwrap();
        let work = prepare(&install).unwrap();
        let mut seen = Vec::new();
        let downloaded = download(
            &work,
            &good,
            check::NOTICE,
            &mut |done, total| seen.push((done, total)),
            &|| false,
        )
        .unwrap();
        assert_eq!(downloaded.size, body.len() as u64);
        assert_eq!(
            seen.last(),
            Some(&(body.len() as u64, Some(body.len() as u64)))
        );
        let mut bad = good.clone();
        bad.sha256 = "0".repeat(64);
        let work = prepare(&install).unwrap();
        let error = download(&work, &bad, check::NOTICE, &mut |_, _| {}, &|| false).unwrap_err();
        assert_eq!(error.code(), "checksum");
        assert!(
            !work.path.join("a.tar.gz").exists(),
            "the mismatching download is deleted"
        );
        assert!(error.reason().contains("0000…00"), "{}", error.reason());
        let work = prepare(&install).unwrap();
        let error = download(&work, &good, check::NOTICE, &mut |_, _| {}, &|| true).unwrap_err();
        assert_eq!(error.code(), "cancelled");
        let path = work.path.clone();
        drop(work);
        assert!(!path.exists(), "the work folder goes with its download");
        let work = prepare(&install).unwrap();
        fail_at(Some("write"));
        let error = download(&work, &good, check::NOTICE, &mut |_, _| {}, &|| false).unwrap_err();
        fail_at(None);
        assert_eq!(error.code(), "disk_full");
    }

    #[cfg(unix)]
    #[test]
    fn a_folder_without_write_permission_changes_nothing() {
        use std::os::unix::fs::PermissionsExt;
        let root = tempfile::tempdir().unwrap();
        let install = shell_install(root.path());
        std::fs::set_permissions(&install.dir, std::fs::Permissions::from_mode(0o555)).unwrap();
        // Root (a container's test user) writes anyway; there is nothing to check then.
        if tempfile::tempfile_in(&install.dir).is_err() {
            let error = prepare(&install).err().unwrap();
            assert_eq!(error.code(), "not_writable");
        }
        std::fs::set_permissions(&install.dir, std::fs::Permissions::from_mode(0o755)).unwrap();
        fail_at(Some("prepare"));
        assert!(prepare(&install).is_err());
        fail_at(None);
    }

    #[test]
    fn one_update_at_a_time_per_home() {
        let home = tempfile::tempdir().unwrap();
        let held = exclusive(home.path()).unwrap();
        assert!(matches!(exclusive(home.path()), Err(Error::Busy)));
        drop(held);
        assert!(exclusive(home.path()).is_ok());
    }

    #[test]
    fn sizes_and_digests_read_as_the_messages_show_them() {
        assert_eq!(megabytes(31_200_000), "31.2 MB");
        assert_eq!(megabytes(32_000_000), "32 MB");
        assert_eq!(megabytes(1_500_000), "1.5 MB");
        assert_eq!(
            short_digest(&format!("3f9a{}c2", "0".repeat(58))),
            "3f9a…c2"
        );
        assert!(space_needed(30_000_000) >= 120_000_000);
    }
}
