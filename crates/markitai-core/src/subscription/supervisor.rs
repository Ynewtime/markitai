//! Bounded official-runtime processes: per-runtime admission, the stderr
//! drain every runtime uses, and the line-oriented process the Claude and
//! Codex (ChatGPT) runtimes share.

use super::FailureKind;
use crate::subscription::line::read_line;
use std::collections::HashMap;
use std::ffi::OsString;
use std::io::{BufReader, Read, Write};
use std::path::Path;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
    mpsc::{self, Receiver, SyncSender},
};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

pub(super) const TICK: Duration = Duration::from_millis(20);
/// Processes one runtime may run at once. Each runtime counts its own.
const MAX_PROCESSES: usize = 8;
/// Stderr beyond this many bytes fails the request.
const STDERR_LIMIT: usize = 1024 * 1024;

/// A supervisor failure; `?` turns it into the calling runtime's own failure.
pub(super) struct ProcessFailure {
    pub(super) kind: FailureKind,
    pub(super) message: &'static str,
}
fn fail(kind: FailureKind, message: &'static str) -> ProcessFailure {
    ProcessFailure { kind, message }
}

/// The running processes of one runtime.
pub(super) struct Admission(AtomicUsize);
impl Admission {
    pub(super) const fn new() -> Self {
        Self(AtomicUsize::new(0))
    }
    /// Wait for a free process slot; `check` reports cancellation or an
    /// expired deadline while waiting.
    pub(super) fn acquire<E>(
        &'static self,
        check: impl Fn() -> Result<(), E>,
    ) -> Result<Permit, E> {
        loop {
            check()?;
            // Rust 1.99 renames this `try_update`; the minimum supported Rust predates the new name.
            #[allow(deprecated)]
            let admitted = self
                .0
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                    (n < MAX_PROCESSES).then_some(n + 1)
                });
            if admitted.is_ok() {
                return Ok(Permit(self));
            }
            std::thread::sleep(TICK);
        }
    }
}
pub(super) struct Permit(&'static Admission);
impl Drop for Permit {
    fn drop(&mut self) {
        self.0.0.fetch_sub(1, Ordering::AcqRel);
    }
}

/// A runtime's stderr, drained without being forwarded, since diagnostics may
/// contain credentials. Only the leading part a caller asked for is kept.
pub(super) struct Stderr {
    keep: usize,
    kept: Mutex<Vec<u8>>,
    overflow: AtomicBool,
    done: AtomicBool,
}
impl Stderr {
    pub(super) fn new(keep: usize) -> Arc<Self> {
        Arc::new(Self {
            keep,
            kept: Mutex::new(Vec::new()),
            overflow: AtomicBool::new(false),
            done: AtomicBool::new(false),
        })
    }
    /// Read until end of stream, a read error or more than [`STDERR_LIMIT`]
    /// bytes, keeping exactly the first `keep` bytes.
    pub(super) fn drain(&self, mut reader: impl Read) {
        let mut buffer = [0; 8192];
        let mut total = 0usize;
        loop {
            let n = match reader.read(&mut buffer) {
                Ok(0) => break,
                Ok(n) => n,
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(_) => break,
            };
            total = total.saturating_add(n);
            if total > STDERR_LIMIT {
                self.overflow.store(true, Ordering::Release);
                break;
            }
            let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
            let room = self.keep.saturating_sub(kept.len());
            kept.extend_from_slice(&buffer[..n.min(room)]);
        }
        self.done.store(true, Ordering::Release);
    }
    pub(super) fn overflowed(&self) -> bool {
        self.overflow.load(Ordering::Acquire)
    }
    fn done(&self) -> bool {
        self.done.load(Ordering::Acquire)
    }
    fn kept(&self) -> Vec<u8> {
        self.kept.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

/// The fixed failure messages of one runtime.
pub(super) struct Messages {
    pub(super) unsupported: &'static str,
    pub(super) cancelled: &'static str,
    pub(super) timeout: &'static str,
    pub(super) malformed: &'static str,
    pub(super) limit: &'static str,
    pub(super) transport: &'static str,
}

/// A runtime that writes one JSON line at a time to stdout.
pub(super) struct Runtime {
    /// Workspace and thread name prefix.
    pub(super) name: &'static str,
    /// Variables set after the retained environment.
    pub(super) environment: &'static [(&'static str, &'static str)],
    /// Leading stderr bytes kept for [`Process::stderr`].
    pub(super) stderr_kept: usize,
    pub(super) messages: Messages,
    pub(super) admission: Admission,
}
impl Runtime {
    fn check(&self, deadline: Instant, cancel: Option<&AtomicBool>) -> Result<(), ProcessFailure> {
        if cancel.is_some_and(|c| c.load(Ordering::Acquire)) {
            return Err(fail(FailureKind::Cancelled, self.messages.cancelled));
        }
        if Instant::now() >= deadline {
            return Err(fail(FailureKind::Timeout, self.messages.timeout));
        }
        Ok(())
    }
    pub(super) fn deadline(&self, timeout: Duration) -> Result<Instant, ProcessFailure> {
        if timeout.is_zero() || timeout > Duration::from_secs(3600) {
            return Err(self.limit());
        }
        Instant::now()
            .checked_add(timeout)
            .ok_or_else(|| self.limit())
    }
    pub(super) fn workspace(&self) -> Result<tempfile::TempDir, ProcessFailure> {
        let prefix = format!("markitai-{}-", self.name);
        let mut builder = tempfile::Builder::new();
        builder.prefix(&prefix);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            builder.permissions(std::fs::Permissions::from_mode(0o700));
        }
        builder.tempdir().map_err(|_| self.transport())
    }
    pub(super) fn spawn(
        &'static self,
        executable: &Path,
        environment: &HashMap<String, String>,
        args: &[OsString],
        workspace: tempfile::TempDir,
        deadline: Instant,
        cancel: Option<&AtomicBool>,
    ) -> Result<Process, ProcessFailure> {
        if !cfg!(any(unix, windows)) {
            return Err(fail(FailureKind::Unsupported, self.messages.unsupported));
        }
        let permit = self.admission.acquire(|| self.check(deadline, cancel))?;
        let group = crate::process_groups::Slot::reserve().ok_or_else(|| self.limit())?;
        let mut command = Command::new(executable);
        command
            .args(args)
            .env_clear()
            .envs(environment)
            .envs(self.environment.iter().copied())
            .current_dir(workspace.path())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let child = group.spawn(&mut command).map_err(|_| self.transport())?;
        let drain = Stderr::new(self.stderr_kept);
        let mut process = Process {
            runtime: self,
            child,
            incoming: None,
            outgoing: None,
            workers: Vec::new(),
            stderr: drain.clone(),
            deadline,
            events: 0,
            _workspace: workspace,
            _permit: permit,
            group,
        };
        let stdout = process
            .child
            .stdout
            .take()
            .ok_or_else(|| self.transport())?;
        let mut stdin = process.child.stdin.take().ok_or_else(|| self.transport())?;
        let stderr = process
            .child
            .stderr
            .take()
            .ok_or_else(|| self.transport())?;
        let (tx, rx) = mpsc::sync_channel(2);
        process.incoming = Some(rx);
        process.workers.push(
            std::thread::Builder::new()
                .name(format!("{}-read", self.name))
                .spawn(move || {
                    let mut reader = BufReader::new(stdout);
                    let mut total = 0;
                    loop {
                        let line = read_line(&mut reader, &mut total);
                        let end = !matches!(&line, Ok(Some(_)));
                        if tx.send(line).is_err() || end {
                            break;
                        }
                    }
                })
                .map_err(|_| self.transport())?,
        );
        let (tx, rx) = mpsc::sync_channel::<Option<Vec<u8>>>(1);
        process.outgoing = Some(tx);
        process.workers.push(
            std::thread::Builder::new()
                .name(format!("{}-write", self.name))
                .spawn(move || {
                    while let Ok(Some(bytes)) = rx.recv() {
                        if stdin.write_all(&bytes).is_err() || stdin.flush().is_err() {
                            break;
                        }
                    }
                })
                .map_err(|_| self.transport())?,
        );
        process.workers.push(
            std::thread::Builder::new()
                .name(format!("{}-stderr", self.name))
                .spawn(move || drain.drain(stderr))
                .map_err(|_| self.transport())?,
        );
        Ok(process)
    }
    fn limit(&self) -> ProcessFailure {
        fail(FailureKind::ResourceLimit, self.messages.limit)
    }
    fn transport(&self) -> ProcessFailure {
        fail(FailureKind::Transport, self.messages.transport)
    }
}

pub(super) struct Process {
    runtime: &'static Runtime,
    child: Child,
    incoming: Option<Receiver<Result<Option<Vec<u8>>, FailureKind>>>,
    outgoing: Option<SyncSender<Option<Vec<u8>>>>,
    workers: Vec<JoinHandle<()>>,
    stderr: Arc<Stderr>,
    deadline: Instant,
    events: usize,
    _workspace: tempfile::TempDir,
    _permit: Permit,
    group: crate::process_groups::Slot,
}
impl Process {
    fn check(&self, cancel: Option<&AtomicBool>) -> Result<(), ProcessFailure> {
        self.runtime.check(self.deadline, cancel)?;
        if self.stderr.overflowed() {
            return Err(self.runtime.limit());
        }
        Ok(())
    }
    pub(super) fn send(
        &self,
        bytes: Option<Vec<u8>>,
        cancel: Option<&AtomicBool>,
    ) -> Result<(), ProcessFailure> {
        let mut pending = bytes;
        loop {
            self.check(cancel)?;
            match self
                .outgoing
                .as_ref()
                .ok_or_else(|| self.runtime.transport())?
                .try_send(pending)
            {
                Ok(()) => return Ok(()),
                Err(mpsc::TrySendError::Full(value)) => pending = value,
                Err(mpsc::TrySendError::Disconnected(_)) => return Err(self.runtime.transport()),
            }
            std::thread::sleep(TICK);
        }
    }
    pub(super) fn next(
        &mut self,
        cancel: Option<&AtomicBool>,
    ) -> Result<Option<Vec<u8>>, ProcessFailure> {
        loop {
            self.check(cancel)?;
            match self
                .incoming
                .as_ref()
                .ok_or_else(|| self.runtime.transport())?
                .recv_timeout(TICK)
            {
                Ok(Ok(line)) => {
                    if line.is_none() {
                        // EOF on stdout does not establish that stderr stayed
                        // within its bound. Retain the original overall deadline
                        // while the independent drain reaches its final state.
                        while !self.stderr.done() {
                            self.check(cancel)?;
                            std::thread::sleep(TICK);
                        }
                    }
                    self.check(cancel)?;
                    self.events += 1;
                    if self.events > 4096 {
                        return Err(self.runtime.limit());
                    }
                    return Ok(line);
                }
                Ok(Err(kind)) => {
                    return Err(fail(kind, self.runtime.messages.malformed));
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    return Err(self.runtime.transport());
                }
            }
        }
    }
    /// The kept leading stderr bytes, complete once stdout has ended.
    pub(super) fn stderr(&self) -> Vec<u8> {
        self.stderr.kept()
    }
    pub(super) fn finish(
        &mut self,
        cancel: Option<&AtomicBool>,
    ) -> Result<ExitStatus, ProcessFailure> {
        loop {
            self.check(cancel)?;
            if let Some(status) = self
                .group
                .try_reap(&mut self.child)
                .map_err(|_| self.runtime.transport())?
            {
                return Ok(status);
            }
            std::thread::sleep(TICK);
        }
    }
}
impl Drop for Process {
    fn drop(&mut self) {
        // The child leads a private tree; descendants end with it.
        self.group.kill_tree(&self.child);
        self.group.retire();
        let _ = self.child.kill();
        let _ = self.child.wait();
        self.outgoing.take();
        self.incoming.take();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A reader returning chunks of the given sizes from a counting pattern.
    struct Chunks(Vec<usize>, u8);
    impl Read for Chunks {
        fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
            let Some(first) = self.0.first_mut() else {
                return Ok(0);
            };
            let n = (*first).min(buffer.len());
            *first -= n;
            if *first == 0 {
                self.0.remove(0);
            }
            for byte in &mut buffer[..n] {
                *byte = self.1;
                self.1 = self.1.wrapping_add(1);
            }
            Ok(n)
        }
    }

    #[test]
    fn stderr_keeps_one_contiguous_prefix_and_flags_overflow() {
        // A chunk crossing the kept size used to be dropped while later
        // smaller chunks were still appended, leaving a gap.
        let stderr = Stderr::new(64 * 1024);
        stderr.drain(Chunks(vec![60 * 1024, 8 * 1024, 10], 0));
        assert!(stderr.done() && !stderr.overflowed());
        assert_eq!(
            stderr.kept(),
            (0..64 * 1024).map(|i| i as u8).collect::<Vec<_>>()
        );
        let stderr = Stderr::new(16);
        stderr.drain(Chunks(vec![STDERR_LIMIT], 0));
        assert!(stderr.done() && !stderr.overflowed());
        assert_eq!(stderr.kept(), (0..16).collect::<Vec<u8>>());
        let stderr = Stderr::new(0);
        stderr.drain(Chunks(vec![STDERR_LIMIT, 1], 0));
        assert!(stderr.done() && stderr.overflowed());
        assert!(stderr.kept().is_empty());
    }

    #[test]
    fn admission_admits_eight_processes_per_runtime() {
        static FIRST: Admission = Admission::new();
        static SECOND: Admission = Admission::new();
        let ready = || Ok::<_, FailureKind>(());
        let held: Vec<_> = (0..MAX_PROCESSES)
            .map(|_| FIRST.acquire(ready).unwrap())
            .collect();
        let deadline = Instant::now() + Duration::from_millis(100);
        let full = FIRST.acquire(|| {
            if Instant::now() >= deadline {
                Err(FailureKind::Timeout)
            } else {
                Ok(())
            }
        });
        assert_eq!(full.err(), Some(FailureKind::Timeout));
        // Another runtime's processes do not count against this one.
        drop(SECOND.acquire(ready).unwrap());
        drop(held);
        drop(FIRST.acquire(ready).unwrap());
        assert_eq!(FIRST.0.load(Ordering::Acquire), 0);
    }
}
