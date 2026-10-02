//! Process trees that this library starts for external runtimes.
//!
//! Chromium, LibreOffice and the official subscription runtimes each run as
//! the root of a tree that can be stopped as a whole: a new process group on
//! Unix, a Job Object on Windows. A terminal interrupt reaches only the
//! foreground process group, and a console control event only the processes
//! attached to that console; Windows runtimes start in a hidden console of
//! their own, so neither reaches them. A host about to terminate calls
//! [`terminate_all`] first. On Unix it only loads atomics and calls kill(2),
//! both async-signal-safe, so a signal handler may call it; a Windows console
//! control handler runs on an ordinary thread. Caller-held browser sessions
//! are never registered.
//!
//! A Windows job is created with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`, so its
//! tree also ends when this process ends for any other reason, even one that
//! runs no cleanup at all.
//!
//! This module also finds runtime programs the way a command shell would
//! ([`find_program`]), which on Windows means trying the PATHEXT extensions.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus};
use std::sync::atomic::{AtomicIsize, Ordering};

const SLOTS: usize = 128;
const RESERVED: isize = -1;
/// Each entry is free (0), reserved, or the tree's address: the leader's
/// process id, which is its process group id (Unix), or the job handle
/// (Windows).
static TREES: [AtomicIsize; SLOTS] = [const { AtomicIsize::new(0) }; SLOTS];

/// One registry entry, reserved before spawning so an unregistered tree
/// never runs. Dropping it releases the entry; on Windows that also ends any
/// process still in the tree and closes the job.
pub(crate) struct Slot(usize);

impl Slot {
    pub(crate) fn reserve() -> Option<Self> {
        TREES
            .iter()
            .position(|slot| {
                slot.compare_exchange(0, RESERVED, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
            })
            .map(Self)
    }

    /// Start `command` as the root of a new tree addressed by this entry.
    /// Its own process-group or creation-flag settings are replaced.
    pub(crate) fn spawn(&self, command: &mut Command) -> std::io::Result<Child> {
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            let child = command.process_group(0).spawn()?;
            self.publish(&child);
            Ok(child)
        }
        #[cfg(windows)]
        {
            windows::spawn(self, command)
        }
        #[cfg(not(any(unix, windows)))]
        {
            let _ = command;
            Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "external runtime process trees are not supported on this platform",
            ))
        }
    }

    /// Address the new group led by `child`, which was started with its own
    /// process group. The leader's id cannot be reused until it is reaped, so
    /// [`Slot::retire`] must precede every reap.
    #[cfg(unix)]
    pub(crate) fn publish(&self, child: &Child) {
        if let Ok(pid) = isize::try_from(child.id())
            && pid > 0
        {
            TREES[self.0].store(pid, Ordering::Release);
        }
    }

    /// Stop addressing the group, keeping the entry reserved. A Windows job
    /// handle cannot come to name another tree, so it stays addressed until
    /// the entry is dropped.
    pub(crate) fn retire(&self) {
        #[cfg(not(windows))]
        TREES[self.0].store(RESERVED, Ordering::Release);
    }

    /// Whether the leader has exited, without reaping it.
    pub(crate) fn exited(&self, child: &Child) -> bool {
        leader_exited(child)
    }

    /// Nonblocking reap that retires the entry first. A leader that has not
    /// exited stays registered.
    pub(crate) fn try_reap(&self, child: &mut Child) -> std::io::Result<Option<ExitStatus>> {
        if !leader_exited(child) {
            return Ok(None);
        }
        self.retire();
        child.try_wait()
    }

    /// Ask the tree to stop: SIGTERM to the Unix group. Windows has no such
    /// request for these runtimes; [`Slot::kill_tree`] follows either way.
    pub(crate) fn request_stop(&self, child: &Child) {
        #[cfg(unix)]
        signal_group(child, libc::SIGTERM);
        #[cfg(not(unix))]
        let _ = child;
    }

    /// Kill every process in the tree now. On Unix the group is addressed by
    /// the leader's id, which callers keep valid by not reaping it first. On
    /// Windows the job is terminated and this waits, briefly, until its
    /// processes are gone, so their files are closed before callers remove
    /// a private workspace.
    pub(crate) fn kill_tree(&self, child: &Child) {
        #[cfg(unix)]
        signal_group(child, libc::SIGKILL);
        #[cfg(windows)]
        {
            let _ = child;
            let job = TREES[self.0].load(Ordering::Acquire);
            if job > 0 {
                windows::kill(job);
            }
        }
        #[cfg(not(any(unix, windows)))]
        let _ = child;
    }
}

impl Drop for Slot {
    fn drop(&mut self) {
        let _address = TREES[self.0].swap(0, Ordering::AcqRel);
        #[cfg(windows)]
        if _address > 0 {
            windows::kill(_address);
            windows::close(_address);
        }
    }
}

/// Signal the group whose id is the leader's id; never group zero.
#[cfg(unix)]
fn signal_group(child: &Child, signal: libc::c_int) {
    if let Ok(pid) = libc::pid_t::try_from(child.id())
        && pid > 0
    {
        unsafe {
            libc::kill(-pid, signal);
        }
    }
}

/// Whether the leader has exited, leaving it unreaped (Unix `WNOWAIT`; a
/// Windows process handle stays valid until the `Child` is dropped).
fn leader_exited(child: &Child) -> bool {
    #[cfg(unix)]
    {
        let pid: libc::id_t = child.id();
        loop {
            // A report sets si_signo to SIGCHLD; no waitable child leaves zeros.
            let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
            let flags = libc::WEXITED | libc::WNOHANG | libc::WNOWAIT;
            if unsafe { libc::waitid(libc::P_PID, pid, &mut info, flags) } == 0 {
                return info.si_signo == libc::SIGCHLD;
            }
            if std::io::Error::last_os_error().kind() != std::io::ErrorKind::Interrupted {
                // Already reaped (ECHILD) or unknown: nothing is left to address.
                return true;
            }
        }
    }
    #[cfg(windows)]
    {
        windows::exited(child)
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = child;
        false
    }
}

/// Immediately kill every registered tree. Async-signal-safe on Unix. On
/// Windows each job is taken from its entry before it is terminated, so a
/// concurrent owner never closes a handle in use here; the host is about to
/// end, which closes them.
pub fn terminate_all() {
    for slot in &TREES {
        let address = slot.load(Ordering::Acquire);
        if address <= 0 {
            continue;
        }
        #[cfg(unix)]
        if let Ok(pid) = libc::pid_t::try_from(address) {
            unsafe {
                libc::kill(-pid, libc::SIGKILL);
            }
        }
        #[cfg(windows)]
        if slot
            .compare_exchange(address, RESERVED, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            windows::terminate(address);
        }
    }
}

/// File extensions that process creation can start: programs directly, and
/// command scripts through the command processor, whose argument quoting the
/// standard library performs.
#[cfg(any(windows, test))]
const LAUNCHABLE: [&str; 4] = [".com", ".exe", ".bat", ".cmd"];

#[cfg(any(windows, test))]
fn launchable_extension(name: &str) -> bool {
    Path::new(name)
        .extension()
        .and_then(OsStr::to_str)
        .is_some_and(|extension| {
            LAUNCHABLE
                .iter()
                .any(|known| known[1..].eq_ignore_ascii_case(extension))
        })
}

/// The launchable extensions PATHEXT lists, in its order; the Windows
/// default when it lists none.
#[cfg(any(windows, test))]
fn search_extensions(pathext: Option<&OsStr>) -> Vec<String> {
    let mut extensions: Vec<String> = Vec::new();
    for entry in pathext
        .and_then(OsStr::to_str)
        .unwrap_or_default()
        .split(';')
    {
        let entry = entry.trim().to_ascii_lowercase();
        if LAUNCHABLE.contains(&entry.as_str()) && !extensions.contains(&entry) {
            extensions.push(entry);
        }
    }
    if extensions.is_empty() {
        extensions = LAUNCHABLE
            .iter()
            .map(|extension| (*extension).into())
            .collect();
    }
    extensions
}

/// File names to try for `name` in one directory, as the Windows command
/// processor does: a name with a launchable extension as given, any other
/// with each PATHEXT extension appended. An extension-less file, such as the
/// shell script npm installs beside its `.cmd` shim, is never chosen.
#[cfg(any(windows, test))]
fn windows_candidates(name: &str, pathext: Option<&OsStr>) -> Vec<String> {
    if launchable_extension(name) {
        return vec![name.into()];
    }
    search_extensions(pathext)
        .into_iter()
        .map(|extension| format!("{name}{extension}"))
        .collect()
}

/// Whether `path` is a file this platform can start: executable by someone
/// on Unix, a program or command script by extension on Windows.
pub(crate) fn launchable(path: &Path) -> bool {
    let Ok(metadata) = std::fs::metadata(path) else {
        return false;
    };
    if !metadata.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        metadata.permissions().mode() & 0o111 != 0
    }
    #[cfg(windows)]
    {
        path.file_name()
            .and_then(OsStr::to_str)
            .is_some_and(launchable_extension)
    }
    #[cfg(not(any(unix, windows)))]
    {
        true
    }
}

/// The first of `names` found in the directories of `path`, in PATH order and
/// then in the order given. Empty and relative entries are skipped: they
/// would select a program relative to the working directory, such as a
/// document's folder. On Windows each name is tried with the launchable
/// PATHEXT extensions (`pathext`, normally the variable's value).
pub(crate) fn find_program(
    names: &[&str],
    path: Option<&OsStr>,
    pathext: Option<&OsStr>,
) -> Option<PathBuf> {
    #[cfg(not(windows))]
    let _ = pathext;
    for directory in std::env::split_paths(path?) {
        if directory.as_os_str().is_empty() || !directory.is_absolute() {
            continue;
        }
        for name in names {
            #[cfg(windows)]
            let files = windows_candidates(name, pathext);
            #[cfg(not(windows))]
            let files = [*name];
            for file in files {
                let candidate = directory.join(file);
                if launchable(&candidate) {
                    return Some(candidate);
                }
            }
        }
    }
    None
}

/// The program a configured runtime path names. On Windows a path without a
/// launchable extension names the first PATHEXT variant that exists, as the
/// command processor would run it; otherwise the path is returned unchanged
/// and the caller's checks decide.
pub(crate) fn configured_program(path: &Path, pathext: Option<&OsStr>) -> PathBuf {
    #[cfg(windows)]
    if !path
        .file_name()
        .and_then(OsStr::to_str)
        .is_some_and(launchable_extension)
        && let Some(name) = path.file_name().and_then(OsStr::to_str)
    {
        for file in windows_candidates(name, pathext) {
            let candidate = path.with_file_name(file);
            if launchable(&candidate) {
                return candidate;
            }
        }
    }
    #[cfg(not(windows))]
    let _ = pathext;
    path.to_path_buf()
}

/// `path` without the `\\?\` prefix that Windows canonicalization adds, when
/// the shorter spelling names the same file: a drive or UNC path within the
/// classic length limit. The command processor cannot run a script by a
/// prefixed path, and the prefix only obscures paths shown to people.
pub(crate) fn plain(path: PathBuf) -> PathBuf {
    #[cfg(windows)]
    if let Some(text) = path.to_str()
        && let Some(shorter) = plain_spelling(text)
    {
        return PathBuf::from(shorter);
    }
    path
}

#[cfg(any(windows, test))]
fn plain_spelling(text: &str) -> Option<String> {
    const LIMIT: usize = 260;
    let shorter = if let Some(rest) = text.strip_prefix(r"\\?\UNC\") {
        format!(r"\\{rest}")
    } else {
        let rest = text.strip_prefix(r"\\?\")?;
        let bytes = rest.as_bytes();
        if bytes.len() < 3 || !bytes[0].is_ascii_alphabetic() || &bytes[1..3] != br":\" {
            return None;
        }
        rest.to_owned()
    };
    // Verbatim components may hold what the ordinary syntax reinterprets.
    let ordinary = shorter.len() < LIMIT
        && !shorter.contains('/')
        && !shorter[2..]
            .split('\\')
            .any(|part| part == "." || part == ".." || part.ends_with('.') || part.ends_with(' '));
    ordinary.then_some(shorter)
}

#[cfg(windows)]
mod windows {
    use super::{RESERVED, Slot, TREES};
    use std::os::windows::io::AsRawHandle;
    use std::os::windows::process::CommandExt;
    use std::process::{Child, Command};
    use std::sync::atomic::Ordering;
    use std::time::{Duration, Instant};
    use windows_sys::Win32::Foundation::{
        CloseHandle, FALSE, HANDLE, INVALID_HANDLE_VALUE, WAIT_OBJECT_0,
    };
    use windows_sys::Win32::System::Diagnostics::ToolHelp::{
        CreateToolhelp32Snapshot, TH32CS_SNAPTHREAD, THREADENTRY32, Thread32First, Thread32Next,
    };
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
        JOBOBJECT_BASIC_ACCOUNTING_INFORMATION, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JobObjectBasicAccountingInformation, JobObjectExtendedLimitInformation,
        QueryInformationJobObject, SetInformationJobObject, TerminateJobObject,
    };
    use windows_sys::Win32::System::Threading::{
        CREATE_NEW_PROCESS_GROUP, CREATE_NO_WINDOW, CREATE_SUSPENDED, OpenThread, ResumeThread,
        THREAD_SUSPEND_RESUME, WaitForSingleObject,
    };

    // Declared here so that creating an unnamed job with default security
    // needs none of windows-sys's security types. The handle is not
    // inheritable, so no runtime receives it.
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn CreateJobObjectW(attributes: *const core::ffi::c_void, name: *const u16) -> HANDLE;
    }

    /// Exit code of a process ended through its job.
    const KILLED: u32 = 1;

    struct Owned(HANDLE);
    impl Drop for Owned {
        fn drop(&mut self) {
            unsafe {
                CloseHandle(self.0);
            }
        }
    }

    fn job() -> std::io::Result<Owned> {
        let job = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
        if job.is_null() {
            return Err(std::io::Error::last_os_error());
        }
        let job = Owned(job);
        let mut limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        let set = unsafe {
            SetInformationJobObject(
                job.0,
                JobObjectExtendedLimitInformation,
                (&raw const limits).cast(),
                size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            )
        };
        if set == 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(job)
    }

    /// Start the process suspended, in a new process group and a hidden
    /// console of its own, place it in a new job, publish the job, and only
    /// then let it run. A launcher such as `soffice.exe` therefore cannot
    /// start a child outside the job before it is assigned.
    pub(super) fn spawn(slot: &Slot, command: &mut Command) -> std::io::Result<Child> {
        let job = job()?;
        command.creation_flags(CREATE_SUSPENDED | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW);
        let mut child = command.spawn()?;
        if unsafe { AssignProcessToJobObject(job.0, child.as_raw_handle()) } == 0 {
            // A process that cannot be contained never runs.
            let error = std::io::Error::last_os_error();
            let _ = child.kill();
            let _ = child.wait();
            return Err(error);
        }
        let handle = job.0 as isize;
        std::mem::forget(job);
        let published =
            TREES[slot.0].compare_exchange(RESERVED, handle, Ordering::AcqRel, Ordering::Acquire);
        debug_assert!(published.is_ok(), "one tree per registry entry");
        if published.is_err() {
            terminate(handle);
            close(handle);
            let _ = child.wait();
            return Err(std::io::Error::other(
                "external runtime registry entry is already in use",
            ));
        }
        if let Err(error) = resume(child.id()) {
            // The entry's owner closes the job when it drops the entry.
            kill(handle);
            let _ = child.wait();
            return Err(error);
        }
        Ok(child)
    }

    /// Resume every thread of a process created suspended: its main thread.
    /// The standard library closes the thread handle that process creation
    /// returns, so the thread is found through a snapshot.
    fn resume(pid: u32) -> std::io::Result<()> {
        let snapshot = unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0) };
        if snapshot == INVALID_HANDLE_VALUE {
            return Err(std::io::Error::last_os_error());
        }
        let snapshot = Owned(snapshot);
        let size = size_of::<THREADENTRY32>() as u32;
        let mut entry = THREADENTRY32 {
            dwSize: size,
            ..Default::default()
        };
        let mut resumed = false;
        let mut more = unsafe { Thread32First(snapshot.0, &mut entry) } != 0;
        while more {
            if entry.th32OwnerProcessID == pid {
                let thread =
                    unsafe { OpenThread(THREAD_SUSPEND_RESUME, FALSE, entry.th32ThreadID) };
                if thread.is_null() {
                    return Err(std::io::Error::last_os_error());
                }
                let thread = Owned(thread);
                if unsafe { ResumeThread(thread.0) } == u32::MAX {
                    return Err(std::io::Error::last_os_error());
                }
                resumed = true;
            }
            entry.dwSize = size;
            more = unsafe { Thread32Next(snapshot.0, &mut entry) } != 0;
        }
        if resumed {
            Ok(())
        } else {
            Err(std::io::Error::other(
                "the new runtime process has no thread to resume",
            ))
        }
    }

    pub(super) fn exited(child: &Child) -> bool {
        unsafe { WaitForSingleObject(child.as_raw_handle(), 0) == WAIT_OBJECT_0 }
    }

    pub(super) fn terminate(job: isize) {
        unsafe {
            TerminateJobObject(job as HANDLE, KILLED);
        }
    }

    /// Terminate the job and wait, at most a few seconds, until none of its
    /// processes is still running.
    pub(super) fn kill(job: isize) {
        terminate(job);
        let deadline = Instant::now() + Duration::from_secs(5);
        while active(job) > 0 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    fn active(job: isize) -> u32 {
        let mut info = JOBOBJECT_BASIC_ACCOUNTING_INFORMATION::default();
        let queried = unsafe {
            QueryInformationJobObject(
                job as HANDLE,
                JobObjectBasicAccountingInformation,
                (&raw mut info).cast(),
                size_of::<JOBOBJECT_BASIC_ACCOUNTING_INFORMATION>() as u32,
                std::ptr::null_mut(),
            )
        };
        if queried == 0 {
            0
        } else {
            info.ActiveProcesses
        }
    }

    pub(super) fn close(job: isize) {
        unsafe {
            CloseHandle(job as HANDLE);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    /// Run the named test alone in a child of this test binary, for tests
    /// whose process-wide effects must not reach concurrently running tests.
    /// Returns false in the child, which then performs the test body.
    fn isolated(name: &str) -> bool {
        if std::env::var_os("MARKITAI_PROCESS_GROUP_CHILD").is_some() {
            return false;
        }
        let output = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", name, "--test-threads", "1", "--nocapture"])
            .env("MARKITAI_PROCESS_GROUP_CHILD", "1")
            .output()
            .unwrap();
        let text = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success(),
            "{text}{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(text.contains("1 passed"), "{text}");
        true
    }

    /// A long sleeper started by re-running this binary. A role given in the
    /// environment selects one of the helper "tests" below, which do nothing
    /// in an ordinary run.
    fn helper(role: &str) -> Command {
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args([
                "--exact",
                &format!("process_groups::tests::{role}"),
                "--test-threads",
                "1",
                "--nocapture",
            ])
            .env("MARKITAI_PROCESS_GROUP_ROLE", role)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());
        command
    }

    fn role(name: &str) -> bool {
        std::env::var("MARKITAI_PROCESS_GROUP_ROLE").as_deref() == Ok(name)
    }

    #[test]
    fn helper_sleeper() {
        if role("helper_sleeper") {
            std::thread::sleep(Duration::from_secs(60));
        }
    }

    /// Starts a sleeper of its own, writes its process id, then waits on it.
    #[test]
    fn helper_parent() {
        if !role("helper_parent") {
            return;
        }
        let mut grandchild = helper("helper_sleeper").spawn().unwrap();
        let path = std::env::var_os("MARKITAI_PROCESS_GROUP_PID_FILE").unwrap();
        let staging = PathBuf::from(&path).with_extension("partial");
        std::fs::write(&staging, grandchild.id().to_string()).unwrap();
        std::fs::rename(staging, path).unwrap();
        let _ = grandchild.wait();
    }

    fn alive(pid: u32) -> bool {
        #[cfg(unix)]
        {
            // A killed grandchild is reparented and may linger as a zombie.
            let output = Command::new("ps")
                .args(["-p", &pid.to_string(), "-o", "stat="])
                .output()
                .unwrap();
            let state = String::from_utf8_lossy(&output.stdout);
            !state.trim().is_empty() && !state.trim().starts_with('Z')
        }
        #[cfg(windows)]
        {
            use windows_sys::Win32::Foundation::{CloseHandle, FALSE, WAIT_TIMEOUT};
            use windows_sys::Win32::System::Threading::{
                OpenProcess, PROCESS_SYNCHRONIZE, WaitForSingleObject,
            };
            let process = unsafe { OpenProcess(PROCESS_SYNCHRONIZE, FALSE, pid) };
            if process.is_null() {
                return false;
            }
            let running = unsafe { WaitForSingleObject(process, 0) } == WAIT_TIMEOUT;
            unsafe { CloseHandle(process) };
            running
        }
    }

    fn wait_gone(pid: u32) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while alive(pid) {
            assert!(Instant::now() < deadline, "process {pid} survived");
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    fn reap(slot: &Slot, child: &mut Child) -> ExitStatus {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(status) = slot.try_reap(child).unwrap() {
                return status;
            }
            assert!(Instant::now() < deadline, "leader did not exit");
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    #[test]
    fn exited_leader_is_reaped_only_after_it_exits_and_its_tree_is_killed_as_one() {
        let slot = Slot::reserve().unwrap();
        let mut child = slot.spawn(&mut helper("helper_sleeper")).unwrap();
        assert!(TREES[slot.0].load(Ordering::Acquire) > 0);
        assert!(!slot.exited(&child));
        assert!(slot.try_reap(&mut child).unwrap().is_none());
        assert!(TREES[slot.0].load(Ordering::Acquire) > 0);
        // Only this test's own tree is killed; never the global registry.
        slot.kill_tree(&child);
        let status = reap(&slot, &mut child);
        assert!(!status.success());
        #[cfg(unix)]
        assert_eq!(TREES[slot.0].load(Ordering::Acquire), RESERVED);
        let pid = child.id();
        drop(slot);
        drop(child);
        wait_gone(pid);
    }

    #[test]
    fn terminate_all_kills_a_registered_tree_in_an_isolated_process() {
        // terminate_all is process-wide; run it only in a dedicated child test
        // process so concurrently running tests keep their own runtimes.
        if isolated(
            "process_groups::tests::terminate_all_kills_a_registered_tree_in_an_isolated_process",
        ) {
            return;
        }
        let directory = tempfile::tempdir().unwrap();
        let pid_file = directory.path().join("grandchild.pid");
        let slot = Slot::reserve().unwrap();
        let mut child = slot
            .spawn(helper("helper_parent").env("MARKITAI_PROCESS_GROUP_PID_FILE", &pid_file))
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(30);
        let grandchild: u32 = loop {
            if let Ok(text) = std::fs::read_to_string(&pid_file) {
                break text.trim().parse().unwrap();
            }
            assert!(Instant::now() < deadline, "the tree never started");
            std::thread::sleep(Duration::from_millis(10));
        };
        assert!(alive(grandchild));
        terminate_all();
        reap(&slot, &mut child);
        wait_gone(grandchild);
    }

    #[cfg(windows)]
    #[test]
    fn a_dropped_entry_ends_what_is_left_of_its_tree() {
        let directory = tempfile::tempdir().unwrap();
        let pid_file = directory.path().join("grandchild.pid");
        let slot = Slot::reserve().unwrap();
        let mut child = slot
            .spawn(helper("helper_parent").env("MARKITAI_PROCESS_GROUP_PID_FILE", &pid_file))
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(30);
        let grandchild: u32 = loop {
            if let Ok(text) = std::fs::read_to_string(&pid_file) {
                break text.trim().parse().unwrap();
            }
            assert!(Instant::now() < deadline, "the tree never started");
            std::thread::sleep(Duration::from_millis(10));
        };
        // The leader alone exits; its sleeper stays in the job.
        let _ = child.kill();
        let _ = child.wait();
        assert!(alive(grandchild));
        drop(slot);
        assert!(!alive(grandchild));
    }

    #[test]
    fn registry_capacity_is_bounded_and_released() {
        if isolated("process_groups::tests::registry_capacity_is_bounded_and_released") {
            return;
        }
        let slots: Vec<_> = std::iter::from_fn(Slot::reserve).take(SLOTS + 1).collect();
        assert_eq!(slots.len(), SLOTS);
        assert!(Slot::reserve().is_none());
        drop(slots);
        assert!(Slot::reserve().is_some());
    }

    fn executable(path: &Path) {
        std::fs::write(path, b"").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700)).unwrap();
        }
    }

    #[test]
    fn search_follows_path_order_and_skips_relative_entries_and_unlaunchable_files() {
        let root = tempfile::tempdir().unwrap();
        let (first, second) = (root.path().join("first"), root.path().join("second"));
        std::fs::create_dir_all(&first).unwrap();
        std::fs::create_dir_all(&second).unwrap();
        let file = |name: &str| {
            if cfg!(windows) {
                format!("{name}.exe")
            } else {
                name.into()
            }
        };
        // Present but not startable: a directory, and (Unix) a file without
        // execute permission.
        std::fs::create_dir(first.join(file("tool"))).unwrap();
        #[cfg(unix)]
        std::fs::write(first.join("other"), b"").unwrap();
        executable(&second.join(file("tool")));
        executable(&second.join(file("other")));
        let path = std::env::join_paths([PathBuf::from("relative"), first.clone(), second.clone()])
            .unwrap();
        assert_eq!(
            find_program(&["tool"], Some(&path), None),
            Some(second.join(file("tool")))
        );
        assert_eq!(
            find_program(&["missing", "other"], Some(&path), None),
            Some(second.join(file("other")))
        );
        assert_eq!(find_program(&["tool"], None, None), None);
        // A directory named relative to the working directory is never
        // searched, even when it holds the program.
        if let Some(relative) = relative_to_working_directory(&second) {
            assert!(relative.is_relative());
            let path = std::env::join_paths([relative]).unwrap();
            assert_eq!(find_program(&["tool"], Some(&path), None), None);
        }
    }

    /// `path` spelled relative to the working directory, when both share a
    /// root (not across Windows drives).
    fn relative_to_working_directory(path: &Path) -> Option<PathBuf> {
        let working = std::env::current_dir().ok()?;
        let base: Vec<_> = working.components().collect();
        let target: Vec<_> = path.components().collect();
        let common = base
            .iter()
            .zip(&target)
            .take_while(|(left, right)| left == right)
            .count();
        if common == 0 {
            return None;
        }
        let mut relative: PathBuf = std::iter::repeat_n("..", base.len() - common).collect();
        relative.extend(&target[common..]);
        Some(relative)
    }

    #[test]
    fn windows_names_try_pathext_launchers_in_order_and_never_an_extensionless_script() {
        let pathext = OsStr::new(".COM;.EXE;.BAT;.CMD;.VBS;.JS;.WS;.MSC;.PS1");
        assert_eq!(
            windows_candidates("claude", Some(pathext)),
            ["claude.com", "claude.exe", "claude.bat", "claude.cmd"]
        );
        assert_eq!(
            windows_candidates("codex", Some(OsStr::new(" .cmd ; .EXE;;.cmd"))),
            ["codex.cmd", "codex.exe"]
        );
        // Unset, empty or useless PATHEXT falls back to the system default.
        for value in [None, Some(OsStr::new("")), Some(OsStr::new(".PS1;.JS"))] {
            assert_eq!(
                windows_candidates("copilot", value),
                ["copilot.com", "copilot.exe", "copilot.bat", "copilot.cmd"]
            );
        }
        assert_eq!(
            windows_candidates("soffice.exe", Some(pathext)),
            ["soffice.exe"]
        );
        assert_eq!(windows_candidates("Setup.CMD", None), ["Setup.CMD"]);
        // A dotted name whose suffix is not a program extension still gets one.
        assert_eq!(
            windows_candidates("chrome.beta", Some(OsStr::new(".EXE"))),
            ["chrome.beta.exe"]
        );
        assert!(launchable_extension("run.Bat"));
        assert!(!launchable_extension("claude"));
        assert!(!launchable_extension("claude.ps1"));
    }

    #[test]
    fn verbatim_prefix_is_dropped_only_when_the_plain_spelling_names_the_same_file() {
        assert_eq!(
            plain_spelling(r"\\?\C:\Users\me\AppData\Roaming\npm\claude.cmd").as_deref(),
            Some(r"C:\Users\me\AppData\Roaming\npm\claude.cmd")
        );
        assert_eq!(
            plain_spelling(r"\\?\UNC\server\share\tools\codex.exe").as_deref(),
            Some(r"\\server\share\tools\codex.exe")
        );
        for kept in [
            r"C:\already\plain.exe",
            r"\\?\Volume{0b1c}\tool.exe",
            r"\\?\C:\trailing dot.\tool.exe",
            r"\\?\C:\trailing space \tool.exe",
            r"\\?\C:\a\..\tool.exe",
        ] {
            assert_eq!(plain_spelling(kept), None, "{kept}");
        }
        let long = format!(r"\\?\C:\{}\tool.exe", "d".repeat(300));
        assert_eq!(plain_spelling(&long), None);
        assert_eq!(
            plain(PathBuf::from("/usr/local/bin/claude")),
            PathBuf::from("/usr/local/bin/claude")
        );
    }
}
