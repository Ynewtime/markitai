//! The file-system facts that output ownership, receipts, recovery state,
//! provider-batch evidence and history depend on, behind one interface.
//!
//! * **Identity.** A [`FileId`] names one file on one volume for as long as it
//!   exists: `st_dev`/`st_ino` on Unix; the 64-bit volume serial number and the
//!   128-bit file ID of `FILE_ID_INFO` on Windows (falling back to the 32-bit
//!   serial and 64-bit index where a file system has no `FileIdInfo`).
//! * **Links.** [`Status::links`] is the hard-link count (`st_nlink`,
//!   `FILE_STANDARD_INFO.NumberOfLinks`).
//! * **Privacy.** On Unix a private entry grants nothing to group or others
//!   (`mode & 0o077 == 0`). Windows has no mode bits and new entries inherit the
//!   parent's ACL, so a private entry there is one whose owner SID is this
//!   process's user, or the default owner its token assigns to new objects (an
//!   elevated administrator's files belong to `BUILTIN\Administrators`).
//! * **No-follow opens.** [`open_no_follow`] refuses a final symbolic link (and
//!   on Windows a junction): `O_NOFOLLOW | O_NONBLOCK` on Unix,
//!   `FILE_FLAG_OPEN_REPARSE_POINT` plus a check of the opened handle on Windows,
//!   which has no FIFOs to block on.
//! * **Durability.** [`sync_directory`] persists a directory's entries with
//!   `fsync` on Unix. Windows cannot flush a directory; it is an explicit no-op
//!   there, and [`sync_renamed`] instead flushes a file after the rename that
//!   named it, which commits the NTFS log records of that rename. Both are
//!   always called together, so each platform does exactly one of them.
//! * **Publication.** [`persist`] and [`persist_noclobber`] retry a rename that
//!   another process (an antivirus or indexing scanner) briefly blocks on
//!   Windows; everywhere else they are the plain `tempfile` operations.
//! * **Spelling.** [`canonicalize`] returns the final spelling of an existing
//!   path; on Windows without the `\\?\` prefix whenever the plain spelling
//!   names the same file.
use std::fs::{self, DirBuilder, File, Metadata, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};

#[cfg(unix)]
mod unix;
#[cfg(unix)]
use unix as imp;
#[cfg(windows)]
mod windows;
#[cfg(windows)]
use windows as imp;

/// One file on one volume, stable while the file exists.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FileId {
    /// The device (Unix) or volume serial number (Windows).
    pub volume: u64,
    /// The inode (Unix) or file ID (Windows) on that volume.
    pub file: u128,
}

/// One observation of a directory entry or an open file.
#[derive(Clone, Debug)]
pub struct Status {
    metadata: Metadata,
    id: FileId,
    links: u64,
    changed: (i64, i64),
    private: bool,
    owner: Owner,
}

/// Who owns an entry: a Unix user ID, or a Windows owner SID when its security
/// descriptor could be read.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Owner {
    #[cfg(unix)]
    User(u32),
    #[cfg_attr(not(windows), allow(dead_code))] // Only Windows reads SIDs.
    Sid(Vec<u8>),
    #[cfg_attr(not(windows), allow(dead_code))] // Only Windows can fail to read one.
    Unknown,
}

impl Status {
    /// The ordinary metadata of this observation (a link is not followed when
    /// the observation came from [`status`]).
    pub fn metadata(&self) -> &Metadata {
        &self.metadata
    }
    pub fn id(&self) -> FileId {
        self.id
    }
    /// The number of hard links naming this file.
    pub fn links(&self) -> u64 {
        self.links
    }
    /// When the file's metadata last changed (`st_ctime` with nanoseconds on
    /// Unix, `FILE_BASIC_INFO.ChangeTime` on Windows): a rewrite that restores
    /// the modification time still changes this.
    pub fn changed(&self) -> (i64, i64) {
        self.changed
    }
    /// See the module documentation for each platform's definition.
    pub fn private(&self) -> bool {
        self.private
    }
    /// Whether both entries have the same, known owner.
    pub fn same_owner(&self, other: &Status) -> bool {
        self.owner != Owner::Unknown && self.owner == other.owner
    }
    /// Whether this process's user owns the entry (the effective user ID on
    /// Unix; on Windows the same test as [`Status::private`]).
    pub fn owned_by_current_user(&self) -> bool {
        imp::owned_by_current_user(self)
    }
}

/// The entry at `path` itself: a final symbolic link or junction is observed,
/// not followed (`lstat`).
pub fn status(path: &Path) -> io::Result<Status> {
    imp::status(path)
}

/// The file an open handle refers to (`fstat`).
pub fn file_status(file: &File) -> io::Result<Status> {
    imp::file_status(file)
}

/// The file `path` finally names, every link followed (`stat`).
pub fn followed_status(path: &Path) -> io::Result<Status> {
    imp::followed_status(path)
}

/// Whether ordinary metadata shows an entry owned by the Unix superuser. The
/// symlink policy trusts such links above a path (`/var`, `/tmp` on macOS);
/// Windows has no equivalent system link owner, so nothing qualifies there.
pub fn root_owned(metadata: &Metadata) -> bool {
    imp::root_owned(metadata)
}

/// Open `path` with `options`, refusing a final symbolic link (or junction) and
/// never blocking on a FIFO. On Windows a reparse point that is not a link (a
/// cloud-file placeholder, for example) is reopened normally, and the reopened
/// handle must be the same file.
pub fn open_no_follow(options: &OpenOptions, path: &Path) -> io::Result<File> {
    imp::open_no_follow(options, path)
}

/// Open an existing file for reading without blocking on a FIFO; a final link
/// is followed only when `follow` is set.
pub fn open_read(path: &Path, follow: bool) -> io::Result<File> {
    let mut options = OpenOptions::new();
    options.read(true);
    if follow {
        imp::open_nonblocking(&options, path)
    } else {
        open_no_follow(&options, path)
    }
}

/// Open the directory at `path` itself, without following a final link, to
/// observe its identity or to synchronize it. Anything else is refused.
pub fn open_directory(path: &Path) -> io::Result<File> {
    imp::open_directory(path)
}

/// Restrict a file this process creates to its owner: mode `0600` on Unix.
/// Windows files inherit their parent directory's ACL.
pub fn private_file(options: &mut OpenOptions) -> &mut OpenOptions {
    imp::private_file(options)
}

/// A builder whose directories are restricted to their owner: mode `0700` on
/// Unix. Windows directories inherit their parent's ACL.
pub fn private_directory() -> DirBuilder {
    imp::private_directory()
}

/// Make the entries of an existing directory durable. Windows has no directory
/// flush; there a name becomes durable through [`sync_renamed`] or the next
/// flush on its volume, and this only checks that the directory exists.
pub fn sync_directory(path: &Path) -> io::Result<()> {
    imp::sync_directory(path)
}

/// After a rename made `file` visible under its new name: on Windows flush the
/// file, which commits the log records of the rename. On Unix this is a no-op,
/// because the caller synchronizes the parent directory instead.
pub fn sync_renamed(file: &File) -> io::Result<()> {
    imp::sync_renamed(file)
}

/// [`sync_renamed`] for a file named by its path (for example the file inside
/// a directory that was renamed as a whole).
pub fn sync_renamed_path(path: &Path) -> io::Result<()> {
    imp::sync_renamed_path(path)
}

/// Durably flush an existing regular file named by its path. Windows requires
/// write access to flush; nothing is written.
pub fn sync_file(path: &Path) -> io::Result<()> {
    imp::sync_file(path)
}

/// Open an existing regular file for a later durable flush (write access on
/// Windows; a read-only descriptor suffices on Unix).
pub fn open_for_sync(path: &Path) -> io::Result<File> {
    imp::open_for_sync(path)
}

/// Replace `path` with a staged temporary file and return its handle.
pub fn persist(
    temporary: tempfile::NamedTempFile,
    path: &Path,
) -> Result<File, tempfile::PersistError> {
    retry_persist(temporary, |temporary| temporary.persist(path))
}

/// Publish a staged temporary file at `path` only if nothing exists there.
pub fn persist_noclobber(
    temporary: tempfile::NamedTempFile,
    path: &Path,
) -> Result<File, tempfile::PersistError> {
    retry_persist(temporary, |temporary| temporary.persist_noclobber(path))
}

/// `std::fs::rename` with the same bounded retry as [`persist`].
pub fn rename(from: &Path, to: &Path) -> io::Result<()> {
    let mut attempt = 0;
    loop {
        let error = match fs::rename(from, to) {
            Ok(()) => return Ok(()),
            Err(error) => error,
        };
        let delay = retry_after(&error, attempt).ok_or(error)?;
        std::thread::sleep(delay);
        attempt += 1;
    }
}

fn retry_persist(
    mut temporary: tempfile::NamedTempFile,
    mut operation: impl FnMut(tempfile::NamedTempFile) -> Result<File, tempfile::PersistError>,
) -> Result<File, tempfile::PersistError> {
    let mut attempt = 0;
    loop {
        match operation(temporary) {
            Err(error) => match retry_after(&error.error, attempt) {
                Some(delay) => {
                    temporary = error.file;
                    std::thread::sleep(delay);
                    attempt += 1;
                }
                None => return Err(error),
            },
            result => return result,
        }
    }
}

/// Attempts a blocked rename gets in total, and the delay before each retry.
/// The reference writer used the same bound: five attempts, 50 ms more each.
const RENAME_ATTEMPTS: u32 = 5;
const RENAME_BACKOFF: std::time::Duration = std::time::Duration::from_millis(50);

/// The delay before retrying `attempt` (counted from zero) after `error`, or
/// `None` when the error is final.
fn retry_after(error: &io::Error, attempt: u32) -> Option<std::time::Duration> {
    retry_delay(cfg!(windows), error.raw_os_error(), attempt)
}

fn retry_delay(windows: bool, code: Option<i32>, attempt: u32) -> Option<std::time::Duration> {
    (windows && attempt + 1 < RENAME_ATTEMPTS && code.is_some_and(transient))
        .then(|| RENAME_BACKOFF * (attempt + 1))
}

/// Windows errors another process's short-lived open causes: access denied
/// (an open without delete sharing), a sharing violation and a lock violation.
/// An existing destination (`ERROR_ALREADY_EXISTS`, `ERROR_FILE_EXISTS`) is
/// never transient.
fn transient(code: i32) -> bool {
    matches!(code, 5 | 32 | 33)
}

/// The final spelling of an existing path: every link resolved and, on
/// Windows, each name in its stored case with long names for short ones.
pub fn canonicalize(path: &Path) -> io::Result<PathBuf> {
    imp::canonicalize(path)
}

/// The Python-compatible non-strict resolution of `path` on Windows (as
/// `os.path.realpath`): the absolute, lexically normalized path whose longest
/// existing prefix takes its final spelling; the missing rest is appended.
#[cfg(windows)]
pub fn resolve(path: &Path) -> io::Result<PathBuf> {
    windows::resolve(path)
}

/// The plain spelling of a verbatim (`\\?\`) path, when one names the same
/// file: a drive or UNC path whose every component is a valid ordinary name
/// (no reserved device name, no trailing dot or space, no character Windows
/// rejects). Length is not a reason to keep the prefix: the standard library
/// adds it again for long paths, and a parent and its child must keep one
/// spelling for prefix comparisons to hold.
#[cfg(any(windows, test))]
fn plain_spelling(verbatim: &str) -> Option<String> {
    let (prefix, rest) = if let Some(rest) = verbatim.strip_prefix(r"\\?\UNC\") {
        (r"\\".to_owned(), rest)
    } else {
        let rest = verbatim.strip_prefix(r"\\?\")?;
        let mut characters = rest.chars();
        let drive = characters.next().filter(char::is_ascii_alphabetic)?;
        let rest = characters.as_str().strip_prefix(':')?;
        let rest = rest.strip_prefix('\\').unwrap_or(rest);
        (format!("{drive}:\\"), rest)
    };
    let components: Vec<&str> = if rest.is_empty() {
        Vec::new()
    } else {
        rest.split('\\').collect()
    };
    if prefix == r"\\" && components.len() < 2 {
        return None;
    }
    if !components.iter().all(|name| ordinary_name(name)) {
        return None;
    }
    Some(format!("{prefix}{}", components.join("\\")))
}

/// A path component that Win32 path normalization leaves unchanged.
#[cfg(any(windows, test))]
fn ordinary_name(name: &str) -> bool {
    if name.is_empty()
        || name == "."
        || name == ".."
        || name.ends_with(['.', ' '])
        || name.chars().any(|c| {
            (c as u32) < 0x20 || matches!(c, '<' | '>' | ':' | '"' | '/' | '|' | '?' | '*')
        })
    {
        return false;
    }
    let base = name.split('.').next().unwrap_or(name).trim_end_matches(' ');
    let upper = base.to_ascii_uppercase();
    let device = matches!(
        upper.as_str(),
        "CON" | "PRN" | "AUX" | "NUL" | "CONIN$" | "CONOUT$"
    ) || ["COM", "LPT"].iter().any(|family| {
        upper.strip_prefix(family).is_some_and(|number| {
            matches!(
                number,
                "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | "¹" | "²" | "³"
            )
        })
    });
    !device
}

#[cfg(not(any(unix, windows)))]
mod imp {
    //! Platforms without native file identity: every protected operation
    //! reports that it is unsupported instead of weakening its guarantee.
    use super::*;
    fn unsupported() -> io::Error {
        io::Error::new(
            io::ErrorKind::Unsupported,
            "native file identity is not supported on this platform",
        )
    }
    pub(super) fn status(_: &Path) -> io::Result<Status> {
        Err(unsupported())
    }
    pub(super) fn file_status(_: &File) -> io::Result<Status> {
        Err(unsupported())
    }
    pub(super) fn followed_status(_: &Path) -> io::Result<Status> {
        Err(unsupported())
    }
    pub(super) fn owned_by_current_user(_: &Status) -> bool {
        false
    }
    pub(super) fn root_owned(_: &Metadata) -> bool {
        false
    }
    pub(super) fn open_no_follow(_: &OpenOptions, _: &Path) -> io::Result<File> {
        Err(unsupported())
    }
    pub(super) fn open_nonblocking(options: &OpenOptions, path: &Path) -> io::Result<File> {
        options.open(path)
    }
    pub(super) fn open_directory(_: &Path) -> io::Result<File> {
        Err(unsupported())
    }
    pub(super) fn private_file(options: &mut OpenOptions) -> &mut OpenOptions {
        options
    }
    pub(super) fn private_directory() -> DirBuilder {
        DirBuilder::new()
    }
    pub(super) fn sync_directory(_: &Path) -> io::Result<()> {
        Err(unsupported())
    }
    pub(super) fn sync_renamed(_: &File) -> io::Result<()> {
        Err(unsupported())
    }
    pub(super) fn sync_renamed_path(_: &Path) -> io::Result<()> {
        Err(unsupported())
    }
    pub(super) fn sync_file(path: &Path) -> io::Result<()> {
        File::open(path)?.sync_all()
    }
    pub(super) fn open_for_sync(path: &Path) -> io::Result<File> {
        File::open(path)
    }
    pub(super) fn canonicalize(path: &Path) -> io::Result<PathBuf> {
        fs::canonicalize(path)
    }
}

#[cfg(test)]
mod tests;
