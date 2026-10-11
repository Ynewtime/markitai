//! Private directories that external runtime trees read.
//!
//! A subscription request writes its prompt, settings and images to a private
//! temporary directory, starts the runtime there and removes the directory
//! when the request ends. A host that ends on a signal runs no destructors,
//! so each directory is also registered here, and [`super::terminate_all`]
//! removes, after killing the trees, the files written through
//! [`Workspace::file`], the directories named by [`Workspace::directory`]
//! when they are empty, and the directory itself. On Unix that reads
//! preallocated static bytes and calls only open, unlinkat, rmdir and close,
//! all async-signal-safe; a Windows console control handler runs on an
//! ordinary thread and removes each directory whole.
//!
//! Anything else a killed runtime wrote keeps its directory, as does a host
//! killed outright. Directory names carry their owner's process id, and the
//! first workspace a later process creates sweeps those whose owner has
//! ended and that are older than any runtime request can run.

use std::cell::UnsafeCell;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use std::time::Duration;

/// Runtimes whose workspaces are swept: `markitai-<runtime>-…`.
const RUNTIMES: [&str; 3] = ["chatgpt", "claude", "copilot"];
/// Every runtime request ends within an hour; a workspace this much older
/// belongs to no running request.
const STALE: Duration = Duration::from_secs(2 * 3600);
const ENTRIES: usize = 64;
const BYTES: usize = 4096;
const FREE: u8 = 0;
const BUSY: u8 = 1;
const READY: u8 = 2;
const TAKEN: u8 = 3;

/// One registered directory: its path, then each name to remove as a kind
/// byte (`f` file, `d` directory) and the name, all NUL-terminated.
struct Entry {
    state: AtomicU8,
    len: AtomicUsize,
    bytes: UnsafeCell<[u8; BYTES]>,
}
// SAFETY: `bytes` is written only by the owner while the entry is BUSY, and
// read only by the one caller that moved it from READY to TAKEN, after which
// it is never written again.
unsafe impl Sync for Entry {}

static REGISTRY: [Entry; ENTRIES] = [const {
    Entry {
        state: AtomicU8::new(FREE),
        len: AtomicUsize::new(0),
        bytes: UnsafeCell::new([0; BYTES]),
    }
}; ENTRIES];

fn encode(path: &Path) -> &[u8] {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        path.as_os_str().as_bytes()
    }
    #[cfg(not(unix))]
    {
        path.as_os_str().as_encoded_bytes()
    }
}

/// Reserve an entry naming `dir`; `None` when every entry is in use or the
/// path does not fit, which leaves that directory to the sweep.
fn register(dir: &Path) -> Option<usize> {
    let path = encode(dir);
    if path.contains(&0) || path.len() >= BYTES {
        return None;
    }
    let index = REGISTRY.iter().position(|entry| {
        entry
            .state
            .compare_exchange(FREE, BUSY, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    })?;
    let entry = &REGISTRY[index];
    // SAFETY: this owner holds BUSY.
    let bytes = unsafe { &mut *entry.bytes.get() };
    bytes[..path.len()].copy_from_slice(path);
    bytes[path.len()] = 0;
    entry.len.store(path.len() + 1, Ordering::Relaxed);
    entry.state.store(READY, Ordering::Release);
    Some(index)
}

/// Add a name to remove; skipped once a terminating host has taken the entry
/// or when it does not fit.
fn record(index: usize, kind: u8, name: &str) {
    let entry = &REGISTRY[index];
    if entry
        .state
        .compare_exchange(READY, BUSY, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return;
    }
    let len = entry.len.load(Ordering::Relaxed);
    let name = name.as_bytes();
    if !name.is_empty() && !name.contains(&0) && len + name.len() + 2 <= BYTES {
        // SAFETY: this owner holds BUSY.
        let bytes = unsafe { &mut *entry.bytes.get() };
        bytes[len] = kind;
        bytes[len + 1..len + 1 + name.len()].copy_from_slice(name);
        bytes[len + 1 + name.len()] = 0;
        entry.len.store(len + name.len() + 2, Ordering::Relaxed);
    }
    entry.state.store(READY, Ordering::Release);
}

fn release(index: usize) {
    // A terminating host may hold it as TAKEN; it is then never reused.
    let _ =
        REGISTRY[index]
            .state
            .compare_exchange(READY, FREE, Ordering::AcqRel, Ordering::Acquire);
}

/// Remove every registered workspace. Async-signal-safe on Unix.
pub(super) fn remove_all() {
    #[cfg(windows)]
    let deadline = std::time::Instant::now() + Duration::from_secs(1);
    for entry in &REGISTRY {
        if entry
            .state
            .compare_exchange(READY, TAKEN, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            continue;
        }
        let len = entry.len.load(Ordering::Relaxed);
        // SAFETY: a TAKEN entry is never written again.
        let bytes = unsafe { &(&*entry.bytes.get())[..len] };
        #[cfg(unix)]
        remove(bytes);
        #[cfg(windows)]
        remove(bytes, deadline);
        #[cfg(not(any(unix, windows)))]
        let _ = bytes;
    }
}

#[cfg(unix)]
fn remove(bytes: &[u8]) {
    let Some(end) = bytes.iter().position(|byte| *byte == 0) else {
        return;
    };
    let directory = bytes.as_ptr().cast::<libc::c_char>();
    // SAFETY: each pointer names a NUL-terminated string inside `bytes`.
    unsafe {
        let fd = libc::open(
            directory,
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
        );
        if fd >= 0 {
            // Files first, so the directories they were in can be emptied.
            for pass in *b"fd" {
                let mut at = end + 1;
                while at < bytes.len() {
                    let Some(length) = bytes[at + 1..].iter().position(|byte| *byte == 0) else {
                        break;
                    };
                    if bytes[at] == pass {
                        let flags = if pass == b'd' { libc::AT_REMOVEDIR } else { 0 };
                        libc::unlinkat(fd, bytes[at + 1..].as_ptr().cast(), flags);
                    }
                    at += length + 2;
                }
            }
            libc::close(fd);
        }
        libc::rmdir(directory);
    }
}

/// Terminated runtimes close their files during rundown, so a removal that
/// meets one still open is retried until `deadline`.
#[cfg(windows)]
fn remove(bytes: &[u8], deadline: std::time::Instant) {
    let Some(end) = bytes.iter().position(|byte| *byte == 0) else {
        return;
    };
    // SAFETY: the bytes came from `as_encoded_bytes` in this process.
    let path = Path::new(unsafe { std::ffi::OsStr::from_encoded_bytes_unchecked(&bytes[..end]) });
    loop {
        match std::fs::remove_dir_all(path) {
            Err(error)
                if error.kind() != std::io::ErrorKind::NotFound
                    && std::time::Instant::now() < deadline =>
            {
                std::thread::sleep(Duration::from_millis(25));
            }
            _ => return,
        }
    }
}

/// A runtime request's private directory, `markitai-<runtime>-<pid>-<random>`
/// in the temporary directory, owner-only on Unix. Dropping it removes the
/// directory and everything in it.
pub(crate) struct Workspace {
    directory: Option<tempfile::TempDir>,
    entry: Option<usize>,
}

impl Workspace {
    pub(crate) fn create(runtime: &str) -> std::io::Result<Self> {
        debug_assert!(RUNTIMES.contains(&runtime));
        let root = std::env::temp_dir();
        static SWEPT: std::sync::Once = std::sync::Once::new();
        SWEPT.call_once(|| sweep(&root, std::process::id(), STALE));
        let prefix = format!("markitai-{runtime}-{}-", std::process::id());
        let mut builder = tempfile::Builder::new();
        builder.prefix(&prefix);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            builder.permissions(std::fs::Permissions::from_mode(0o700));
        }
        let directory = builder.tempdir_in(&root)?;
        let entry = register(directory.path());
        Ok(Self {
            directory: Some(directory),
            entry,
        })
    }

    pub(crate) fn path(&self) -> &Path {
        self.directory
            .as_ref()
            .expect("a workspace keeps its directory until dropped")
            .path()
    }

    /// Create the private file `name` with `bytes`, never replacing one; it
    /// is owner-only on Unix and removed even when the host ends on a signal.
    pub(crate) fn file(&self, name: &str, bytes: &[u8]) -> std::io::Result<PathBuf> {
        use std::io::Write;
        if let Some(index) = self.entry {
            record(index, b'f', name);
        }
        let path = self.path().join(name);
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        options.open(&path)?.write_all(bytes)?;
        Ok(path)
    }

    /// A directory the runtime may create for its own state. It goes with the
    /// workspace; a host ending on a signal removes it only when it is empty.
    pub(crate) fn directory(&self, name: &str) -> PathBuf {
        if let Some(index) = self.entry {
            record(index, b'd', name);
        }
        self.path().join(name)
    }
}

impl Drop for Workspace {
    fn drop(&mut self) {
        drop(self.directory.take());
        if let Some(index) = self.entry {
            release(index);
        }
    }
}

/// Remove the workspaces in `root` that a host which ran no cleanup left: a
/// directory of this user, older than `stale`, whose owner is not running
/// (older builds named no owner).
fn sweep(root: &Path, own: u32, stale: Duration) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(rest) = name.to_str().and_then(|name| {
            let name = name.strip_prefix("markitai-")?;
            RUNTIMES
                .iter()
                .find_map(|runtime| name.strip_prefix(runtime)?.strip_prefix('-'))
        }) else {
            continue;
        };
        // `<pid>-<random>`; tempfile's random suffix has no hyphen.
        let owner = rest
            .split_once('-')
            .and_then(|(pid, _)| pid.parse::<u32>().ok());
        if owner.is_some_and(|pid| pid == own || running(pid)) {
            continue;
        }
        let path = entry.path();
        let Ok(metadata) = std::fs::symlink_metadata(&path) else {
            continue;
        };
        if !metadata.is_dir() {
            continue;
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            // SAFETY: geteuid has no preconditions.
            if metadata.uid() != unsafe { libc::geteuid() } {
                continue;
            }
        }
        if metadata
            .modified()
            .ok()
            .and_then(|modified| modified.elapsed().ok())
            .is_none_or(|age| age < stale)
        {
            continue;
        }
        let _ = std::fs::remove_dir_all(&path);
    }
}

/// Whether process `pid` may still be running; anything unclear counts as
/// running, so its workspace is kept.
fn running(pid: u32) -> bool {
    #[cfg(unix)]
    {
        let Ok(pid) = libc::pid_t::try_from(pid) else {
            return true;
        };
        if pid <= 0 {
            return true;
        }
        // SAFETY: signal 0 only checks that the process exists.
        let exists = unsafe { libc::kill(pid, 0) } == 0;
        exists || std::io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH)
    }
    #[cfg(windows)]
    {
        use windows_sys::Win32::Foundation::{
            CloseHandle, ERROR_INVALID_PARAMETER, FALSE, GetLastError, WAIT_TIMEOUT,
        };
        use windows_sys::Win32::System::Threading::{
            OpenProcess, PROCESS_SYNCHRONIZE, WaitForSingleObject,
        };
        // SAFETY: the handle is checked and closed here.
        unsafe {
            let process = OpenProcess(PROCESS_SYNCHRONIZE, FALSE, pid);
            if process.is_null() {
                return GetLastError() != ERROR_INVALID_PARAMETER;
            }
            let running = WaitForSingleObject(process, 0) == WAIT_TIMEOUT;
            CloseHandle(process);
            running
        }
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = pid;
        true
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    #[test]
    fn a_sweep_removes_only_stale_workspaces_of_ended_owners() {
        let root = tempfile::tempdir().unwrap();
        let mut ended = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "no-such-test", "--quiet"])
            .stdout(std::process::Stdio::null())
            .spawn()
            .unwrap();
        ended.wait().unwrap();
        let ended = ended.id();
        let own = std::process::id();
        let names = [
            format!("markitai-chatgpt-{ended}-AbC123"),
            format!("markitai-copilot-{ended}-ZZ9zz9"),
            "markitai-claude-Legacy".into(),
            format!("markitai-chatgpt-{own}-Mine12"),
            format!("markitai-browser-{ended}-Other1"),
        ];
        for name in &names {
            std::fs::create_dir(root.path().join(name)).unwrap();
            std::fs::write(root.path().join(name).join("system.txt"), "prompt").unwrap();
        }
        // A file and a link with a workspace's name are not workspaces.
        std::fs::write(
            root.path().join(format!("markitai-claude-{ended}-File12")),
            "",
        )
        .unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("kept"), "").unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(
            outside.path(),
            root.path().join(format!("markitai-claude-{ended}-Link12")),
        )
        .unwrap();
        // Nothing is old enough yet.
        sweep(root.path(), own, Duration::from_secs(3600));
        for name in &names {
            assert!(root.path().join(name).exists(), "{name}");
        }
        sweep(root.path(), own, Duration::ZERO);
        let left: std::collections::BTreeSet<String> = std::fs::read_dir(root.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect();
        let expected: std::collections::BTreeSet<String> = [
            format!("markitai-chatgpt-{own}-Mine12"),
            format!("markitai-browser-{ended}-Other1"),
            format!("markitai-claude-{ended}-File12"),
        ]
        .into_iter()
        .chain(cfg!(unix).then(|| format!("markitai-claude-{ended}-Link12")))
        .collect();
        assert_eq!(left, expected);
        assert!(outside.path().join("kept").exists());
    }

    #[test]
    fn a_workspace_is_private_named_for_its_owner_and_removed_when_dropped() {
        let workspace = Workspace::create("chatgpt").unwrap();
        let path = workspace.path().to_owned();
        let name = path.file_name().unwrap().to_str().unwrap();
        assert!(
            name.starts_with(&format!("markitai-chatgpt-{}-", std::process::id())),
            "{name}"
        );
        let file = workspace.file("system.txt", b"prompt").unwrap();
        assert_eq!(std::fs::read(&file).unwrap(), b"prompt");
        assert!(workspace.file("system.txt", b"again").is_err());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = |path: &Path| std::fs::metadata(path).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode(&path), 0o700);
            assert_eq!(mode(&file), 0o600);
        }
        std::fs::create_dir(workspace.directory("state")).unwrap();
        std::fs::write(path.join("state/runtime.db"), "").unwrap();
        drop(workspace);
        assert!(!path.exists());
    }

    /// The termination path of a host: run only in an isolated process,
    /// because it removes every registered workspace of the process.
    pub(in crate::process_groups) fn terminate_all_removes_registered_workspaces() {
        let clean = Workspace::create("claude").unwrap();
        clean.file("system.txt", b"prompt").unwrap();
        clean.file("settings.json", b"{}").unwrap();
        std::fs::create_dir(clean.directory("state")).unwrap();
        let clean_path = clean.path().to_owned();
        // A runtime wrote into its log directory: that directory and the
        // workspace stay for the sweep, everything markitai wrote goes.
        let busy = Workspace::create("chatgpt").unwrap();
        busy.file("models.json", b"{}").unwrap();
        std::fs::create_dir(busy.directory("log")).unwrap();
        std::fs::write(busy.path().join("log/runtime.log"), "").unwrap();
        let busy_path = busy.path().to_owned();
        crate::process_groups::terminate_all();
        assert!(!clean_path.exists());
        #[cfg(unix)]
        {
            assert!(busy_path.join("log/runtime.log").exists());
            assert!(!busy_path.join("models.json").exists());
        }
        #[cfg(windows)]
        assert!(!busy_path.exists());
        // Dropping after termination is harmless.
        drop(clean);
        drop(busy);
        assert!(!busy_path.exists());
    }
}
