use super::{Config, Failure, FailureKind};
use std::ffi::OsString;
use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
    mpsc::{self, Receiver, SyncSender},
};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

const TICK: Duration = Duration::from_millis(20);
const LINE_LIMIT: usize = 16 * 1024 * 1024;
const STREAM_LIMIT: usize = 64 * 1024 * 1024;
const STDERR_LIMIT: usize = 1024 * 1024;
static ACTIVE: AtomicUsize = AtomicUsize::new(0);
struct Permit;
impl Drop for Permit {
    fn drop(&mut self) {
        ACTIVE.fetch_sub(1, Ordering::AcqRel);
    }
}
impl Permit {
    fn acquire(deadline: Instant, cancel: Option<&AtomicBool>) -> Result<Self, Failure> {
        loop {
            check(deadline, cancel)?;
            if ACTIVE
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                    (n < 8).then_some(n + 1)
                })
                .is_ok()
            {
                return Ok(Self);
            }
            std::thread::sleep(TICK);
        }
    }
}
pub(super) fn check(deadline: Instant, cancel: Option<&AtomicBool>) -> Result<(), Failure> {
    if cancel.is_some_and(|c| c.load(Ordering::Acquire)) {
        return Err(Failure::new(
            FailureKind::Cancelled,
            "ChatGPT request was cancelled",
        ));
    }
    if Instant::now() >= deadline {
        return Err(Failure::new(
            FailureKind::Timeout,
            "ChatGPT runtime deadline exceeded; completion is unknown",
        ));
    }
    Ok(())
}
pub(super) fn deadline(timeout: Duration) -> Result<Instant, Failure> {
    if timeout.is_zero() || timeout > Duration::from_secs(3600) {
        return Err(limit());
    }
    Instant::now().checked_add(timeout).ok_or_else(limit)
}
pub(super) fn workspace() -> Result<tempfile::TempDir, Failure> {
    let mut builder = tempfile::Builder::new();
    builder.prefix("markitai-chatgpt-");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        builder.permissions(std::fs::Permissions::from_mode(0o700));
    }
    builder.tempdir().map_err(|_| transport())
}
pub(super) struct Process {
    child: Child,
    incoming: Option<Receiver<Result<Option<Vec<u8>>, FailureKind>>>,
    outgoing: Option<SyncSender<Option<Vec<u8>>>>,
    workers: Vec<JoinHandle<()>>,
    overflow: Arc<AtomicBool>,
    stderr_done: Arc<AtomicBool>,
    stderr: Arc<Mutex<Vec<u8>>>,
    deadline: Instant,
    events: usize,
    _workspace: tempfile::TempDir,
    _permit: Permit,
    group: crate::process_groups::Slot,
}
impl Process {
    pub(super) fn spawn(
        config: &Config,
        args: &[OsString],
        workspace: tempfile::TempDir,
        deadline: Instant,
        cancel: Option<&AtomicBool>,
    ) -> Result<Self, Failure> {
        if !cfg!(unix) {
            return Err(Failure::new(
                FailureKind::Unsupported,
                "Bounded ChatGPT process-tree cleanup is unavailable on this platform",
            ));
        }
        let permit = Permit::acquire(deadline, cancel)?;
        let group = crate::process_groups::Slot::reserve().ok_or_else(limit)?;
        let mut command = Command::new(&config.executable);
        command
            .args(args)
            .env_clear()
            .envs(&config.environment)
            .current_dir(workspace.path())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            command.process_group(0);
        }
        let child = command.spawn().map_err(|_| transport())?;
        group.publish(&child);
        let overflow = Arc::new(AtomicBool::new(false));
        let stderr_done = Arc::new(AtomicBool::new(false));
        let stderr_bytes = Arc::new(Mutex::new(Vec::new()));
        let mut process = Self {
            child,
            incoming: None,
            outgoing: None,
            workers: Vec::new(),
            overflow: overflow.clone(),
            stderr_done: stderr_done.clone(),
            stderr: stderr_bytes.clone(),
            deadline,
            events: 0,
            _workspace: workspace,
            _permit: permit,
            group,
        };
        let stdout = process.child.stdout.take().ok_or_else(transport)?;
        let mut stdin = process.child.stdin.take().ok_or_else(transport)?;
        let mut stderr = process.child.stderr.take().ok_or_else(transport)?;
        let (tx, rx) = mpsc::sync_channel(2);
        process.incoming = Some(rx);
        process.workers.push(
            std::thread::Builder::new()
                .name("chatgpt-read".into())
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
                .map_err(|_| transport())?,
        );
        let (tx, rx) = mpsc::sync_channel::<Option<Vec<u8>>>(1);
        process.outgoing = Some(tx);
        process.workers.push(
            std::thread::Builder::new()
                .name("chatgpt-write".into())
                .spawn(move || {
                    while let Ok(Some(bytes)) = rx.recv() {
                        if stdin.write_all(&bytes).is_err() || stdin.flush().is_err() {
                            break;
                        }
                    }
                })
                .map_err(|_| transport())?,
        );
        process.workers.push(
            std::thread::Builder::new()
                .name("chatgpt-stderr".into())
                .spawn(move || {
                    let mut buffer = [0; 8192];
                    let mut total = 0usize;
                    loop {
                        let n = match stderr.read(&mut buffer) {
                            Ok(0) => break,
                            Ok(n) => n,
                            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {
                                continue;
                            }
                            Err(_) => break,
                        };
                        total = total.saturating_add(n);
                        if total > STDERR_LIMIT {
                            overflow.store(true, Ordering::Release);
                            break;
                        }
                        let mut saved = stderr_bytes.lock().unwrap_or_else(|e| e.into_inner());
                        if saved.len().saturating_add(n) <= 64 * 1024 {
                            saved.extend_from_slice(&buffer[..n]);
                        }
                    }
                    stderr_done.store(true, Ordering::Release);
                })
                .map_err(|_| transport())?,
        );
        Ok(process)
    }
    fn check(&self, cancel: Option<&AtomicBool>) -> Result<(), Failure> {
        check(self.deadline, cancel)?;
        if self.overflow.load(Ordering::Acquire) {
            return Err(limit());
        }
        Ok(())
    }
    pub(super) fn send(
        &self,
        bytes: Option<Vec<u8>>,
        cancel: Option<&AtomicBool>,
    ) -> Result<(), Failure> {
        let mut pending = bytes;
        loop {
            self.check(cancel)?;
            match self
                .outgoing
                .as_ref()
                .ok_or_else(transport)?
                .try_send(pending)
            {
                Ok(()) => return Ok(()),
                Err(mpsc::TrySendError::Full(value)) => pending = value,
                Err(mpsc::TrySendError::Disconnected(_)) => return Err(transport()),
            }
            std::thread::sleep(TICK);
        }
    }
    pub(super) fn next(&mut self, cancel: Option<&AtomicBool>) -> Result<Option<Vec<u8>>, Failure> {
        loop {
            self.check(cancel)?;
            match self
                .incoming
                .as_ref()
                .ok_or_else(transport)?
                .recv_timeout(TICK)
            {
                Ok(Ok(line)) => {
                    if line.is_none() {
                        // EOF on stdout does not establish that stderr stayed
                        // within its bound. Retain the original overall deadline
                        // while the independent drain reaches its final state.
                        while !self.stderr_done.load(Ordering::Acquire) {
                            self.check(cancel)?;
                            std::thread::sleep(TICK);
                        }
                    }
                    self.check(cancel)?;
                    self.events += 1;
                    if self.events > 4096 {
                        return Err(limit());
                    }
                    return Ok(line);
                }
                Ok(Err(kind)) => {
                    return Err(Failure::new(
                        kind,
                        "ChatGPT output is malformed or exceeds its limits",
                    ));
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => return Err(transport()),
            }
        }
    }
    pub(super) fn stderr(&self) -> Vec<u8> {
        self.stderr
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }
    pub(super) fn finish(&mut self, cancel: Option<&AtomicBool>) -> Result<ExitStatus, Failure> {
        loop {
            self.check(cancel)?;
            if let Some(status) = self
                .group
                .try_reap(&mut self.child)
                .map_err(|_| transport())?
            {
                return Ok(status);
            }
            std::thread::sleep(TICK);
        }
    }
}
impl Drop for Process {
    fn drop(&mut self) {
        #[cfg(unix)]
        {
            if let Ok(pid) = i32::try_from(self.child.id()) {
                // The child starts a private process group; never address group zero.
                if pid > 0 {
                    unsafe {
                        libc::kill(-pid, libc::SIGKILL);
                    }
                }
            }
        }
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
fn read_line(reader: &mut impl BufRead, total: &mut usize) -> Result<Option<Vec<u8>>, FailureKind> {
    let mut line = Vec::new();
    loop {
        let bytes = match reader.fill_buf() {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => return Err(FailureKind::Transport),
        };
        if bytes.is_empty() {
            return if line.is_empty() {
                Ok(None)
            } else {
                Ok(Some(line))
            };
        }
        let count = bytes
            .iter()
            .position(|b| *b == b'\n')
            .map_or(bytes.len(), |i| i + 1);
        if line.len().saturating_add(count) > LINE_LIMIT
            || total.saturating_add(count) > STREAM_LIMIT
        {
            return Err(FailureKind::ResourceLimit);
        }
        let ended = bytes[count - 1] == b'\n';
        line.extend_from_slice(&bytes[..count]);
        *total += count;
        reader.consume(count);
        if ended {
            return Ok(Some(line));
        }
    }
}
fn limit() -> Failure {
    Failure::new(
        FailureKind::ResourceLimit,
        "ChatGPT runtime resource limit exceeded",
    )
}
fn transport() -> Failure {
    Failure::new(
        FailureKind::Transport,
        "ChatGPT runtime process communication failed",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    struct InterruptedChunks {
        bytes: std::io::Cursor<Vec<u8>>,
        interrupt: bool,
    }
    impl Read for InterruptedChunks {
        fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
            self.interrupt = !self.interrupt;
            if self.interrupt {
                return Err(std::io::ErrorKind::Interrupted.into());
            }
            let size = output.len().min(2);
            self.bytes.read(&mut output[..size])
        }
    }

    #[test]
    fn interrupted_pipe_reads_preserve_frames_counts_and_stream_limits() {
        let bytes = b"{\"x\":1}\n{}".to_vec();
        let mut reader = BufReader::new(InterruptedChunks {
            bytes: std::io::Cursor::new(bytes.clone()),
            interrupt: false,
        });
        let mut total = 0;
        assert_eq!(
            read_line(&mut reader, &mut total).unwrap().unwrap(),
            b"{\"x\":1}\n"
        );
        assert_eq!(read_line(&mut reader, &mut total).unwrap().unwrap(), b"{}");
        assert!(read_line(&mut reader, &mut total).unwrap().is_none());
        assert_eq!(total, bytes.len());

        let mut reader = BufReader::new(InterruptedChunks {
            bytes: std::io::Cursor::new(b"{}\n".to_vec()),
            interrupt: false,
        });
        let mut total = STREAM_LIMIT - 2;
        assert_eq!(
            read_line(&mut reader, &mut total),
            Err(FailureKind::ResourceLimit)
        );
    }
}
