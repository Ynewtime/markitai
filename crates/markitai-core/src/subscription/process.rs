use super::supervisor::{Admission, Permit, Stderr, TICK};
use super::{CopilotConfig, Failure, FailureKind};
use serde_json::{Value, json};
use std::io::{BufReader, Read, Write};
use std::process::{Child, Command, Stdio};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
    mpsc::{self, Receiver, SyncSender},
};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

pub(super) const MAX_FRAME: usize = 100 * 1024 * 1024;
const MAX_STREAM: usize = 128 * 1024 * 1024;
const MAX_EVENTS: usize = 4096;
const HEADER_LIMIT: usize = 8192;

type FrameResult = Result<Value, FailureKind>;
static ADMISSION: Admission = Admission::new();

pub(super) struct Process {
    child: Child,
    _permit: Permit,
    group: crate::process_groups::Slot,
    incoming: Option<Receiver<FrameResult>>,
    outgoing: Option<SyncSender<Vec<u8>>>,
    workers: Vec<JoinHandle<()>>,
    stderr: Arc<Stderr>,
    pub(super) workspace: tempfile::TempDir,
    deadline: Instant,
    sequence: u64,
    events: usize,
}
impl Process {
    pub(super) fn spawn(
        config: &CopilotConfig,
        timeout: Duration,
        cancel: Option<&AtomicBool>,
    ) -> Result<Self, Failure> {
        if !cfg!(any(unix, windows)) {
            return Err(Failure::new(
                FailureKind::Unsupported,
                "Bounded Copilot process-tree cleanup is not available on this platform",
            ));
        }
        if timeout.is_zero() || timeout > Duration::from_secs(3600) {
            return Err(Failure::new(
                FailureKind::ResourceLimit,
                "Copilot timeout must be positive and at most one hour",
            ));
        }
        let deadline = Instant::now().checked_add(timeout).ok_or_else(limit)?;
        let permit = ADMISSION.acquire(|| {
            if cancel.is_some_and(|flag| flag.load(Ordering::Acquire)) {
                return Err(Failure::new(
                    FailureKind::Cancelled,
                    "Copilot request was cancelled",
                ));
            }
            if Instant::now() >= deadline {
                return Err(Failure::new(
                    FailureKind::Timeout,
                    "Copilot process admission deadline exceeded",
                ));
            }
            Ok(())
        })?;
        let group = crate::process_groups::Slot::reserve().ok_or_else(limit)?;
        let mut directory = tempfile::Builder::new();
        directory.prefix("markitai-copilot-");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            directory.permissions(std::fs::Permissions::from_mode(0o700));
        }
        let workspace = directory.tempdir().map_err(|_| {
            Failure::new(
                FailureKind::Transport,
                "Cannot create private Copilot workspace",
            )
        })?;
        let mut command = Command::new(&config.executable);
        command
            .args([
                "--headless",
                "--stdio",
                "--no-auto-update",
                "--log-level",
                "error",
            ])
            .env_clear()
            .envs(&config.environment)
            .env("COPILOT_AUTO_UPDATE", "false")
            .current_dir(workspace.path())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if let Some(token) = &config.token {
            command
                .args([
                    "--auth-token-env",
                    "MARKITAI_COPILOT_TOKEN",
                    "--no-auto-login",
                ])
                .env("MARKITAI_COPILOT_TOKEN", token);
        }
        let child = group.spawn(&mut command).map_err(|_| {
            Failure::new(
                FailureKind::Transport,
                "Cannot start the official Copilot runtime",
            )
        })?;
        let drain = Stderr::new(0);
        let mut process = Self {
            child,
            _permit: permit,
            group,
            incoming: None,
            outgoing: None,
            workers: Vec::new(),
            stderr: drain.clone(),
            workspace,
            deadline,
            sequence: 0,
            events: 0,
        };
        let stdout = process.child.stdout.take().ok_or_else(protocol)?;
        let mut stdin = process.child.stdin.take().ok_or_else(protocol)?;
        let stderr = process.child.stderr.take().ok_or_else(protocol)?;
        let (incoming_tx, incoming_rx) = mpsc::sync_channel(2);
        let (outgoing_tx, outgoing_rx) = mpsc::sync_channel::<Vec<u8>>(1);
        process.incoming = Some(incoming_rx);
        process.outgoing = Some(outgoing_tx);
        let reader = std::thread::Builder::new()
            .name("copilot-read".into())
            .spawn(move || {
                let mut input = BufReader::new(stdout);
                let mut total = 0usize;
                loop {
                    let message = read_frame(&mut input, &mut total);
                    let failed = message.is_err();
                    if incoming_tx.send(message).is_err() || failed {
                        break;
                    }
                }
            })
            .map_err(|_| transport())?;
        process.workers.push(reader);
        let writer = std::thread::Builder::new()
            .name("copilot-write".into())
            .spawn(move || {
                while let Ok(bytes) = outgoing_rx.recv() {
                    if write!(stdin, "Content-Length: {}\r\n\r\n", bytes.len()).is_err()
                        || stdin.write_all(&bytes).is_err()
                        || stdin.flush().is_err()
                    {
                        break;
                    }
                }
            })
            .map_err(|_| transport())?;
        process.workers.push(writer);
        let errors = std::thread::Builder::new()
            .name("copilot-stderr".into())
            .spawn(move || drain.drain(stderr))
            .map_err(|_| transport())?;
        process.workers.push(errors);
        Ok(process)
    }
    fn check(&self, cancel: Option<&AtomicBool>) -> Result<(), Failure> {
        if cancel.is_some_and(|flag| flag.load(Ordering::Acquire)) {
            return Err(Failure::new(
                FailureKind::Cancelled,
                "Copilot request was cancelled",
            ));
        }
        if Instant::now() >= self.deadline {
            return Err(Failure::new(
                FailureKind::Timeout,
                "Copilot request deadline exceeded; completion is unknown",
            ));
        }
        if self.stderr.overflowed() {
            return Err(limit());
        }
        Ok(())
    }
    fn send(&self, value: &Value, cancel: Option<&AtomicBool>) -> Result<(), Failure> {
        let mut bytes = serde_json::to_vec(value).map_err(|_| protocol())?;
        if bytes.len() > MAX_FRAME {
            return Err(limit());
        }
        loop {
            self.check(cancel)?;
            match self
                .outgoing
                .as_ref()
                .ok_or_else(transport)?
                .try_send(bytes)
            {
                Ok(()) => return Ok(()),
                Err(mpsc::TrySendError::Disconnected(_)) => return Err(transport()),
                Err(mpsc::TrySendError::Full(returned)) => bytes = returned,
            }
            std::thread::sleep(TICK);
        }
    }
    pub(super) fn next(&mut self, cancel: Option<&AtomicBool>) -> Result<Value, Failure> {
        loop {
            self.check(cancel)?;
            match self
                .incoming
                .as_ref()
                .ok_or_else(transport)?
                .recv_timeout(TICK)
            {
                Ok(Ok(message)) => {
                    self.events += 1;
                    if self.events > MAX_EVENTS {
                        return Err(limit());
                    }
                    if !message.is_object()
                        || message.get("jsonrpc").and_then(Value::as_str) != Some("2.0")
                    {
                        return Err(protocol());
                    }
                    if message.get("method").is_some() && message.get("id").is_some() {
                        // Every callback is denied. No automatic tool or permission approval.
                        self.send(&json!({"jsonrpc":"2.0","id":message["id"],"error":{"code":-32601,"message":"Markitai does not permit runtime callbacks"}}), cancel)?;
                        return Err(Failure::new(
                            FailureKind::Permission,
                            "Copilot requested an unsupported tool or permission",
                        ));
                    }
                    return Ok(message);
                }
                Ok(Err(kind)) => {
                    // A runtime whose stderr the drain closed may end its
                    // stdout too; the overflow is the cause to report.
                    if self.stderr.overflowed() {
                        return Err(limit());
                    }
                    return Err(Failure::new(
                        kind,
                        "Copilot protocol stream ended or exceeded its limits",
                    ));
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => return Err(transport()),
                Err(mpsc::RecvTimeoutError::Timeout) => {}
            }
        }
    }
    pub(super) fn call(
        &mut self,
        method: &str,
        params: Value,
        cancel: Option<&AtomicBool>,
        on_event: &mut impl FnMut(&Value) -> Result<(), Failure>,
    ) -> Result<Value, Failure> {
        self.sequence = self.sequence.checked_add(1).ok_or_else(limit)?;
        let id = self.sequence;
        self.send(
            &json!({"jsonrpc":"2.0","id":id,"method":method,"params":params}),
            cancel,
        )?;
        loop {
            let message = self.next(cancel)?;
            if message.get("method").is_some() {
                on_event(&message)?;
                continue;
            }
            if message.get("id").and_then(Value::as_u64) != Some(id) {
                return Err(protocol());
            }
            if message.get("error").is_some() {
                return Err(Failure::new(
                    FailureKind::Protocol,
                    "Copilot rejected the protocol request",
                ));
            }
            return message.get("result").cloned().ok_or_else(protocol);
        }
    }
}
impl Drop for Process {
    fn drop(&mut self) {
        // The child leads its own tree, so descendants holding its pipes
        // cannot keep our reader threads alive after cancellation.
        self.group.kill_tree(&self.child);
        self.group.retire();
        let _ = self.child.kill();
        let _ = self.child.wait();
        self.incoming.take();
        self.outgoing.take();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}
pub(super) fn read_frame(input: &mut impl Read, total: &mut usize) -> FrameResult {
    let mut header = Vec::with_capacity(128);
    while !header.ends_with(b"\r\n\r\n") {
        if header.len() == HEADER_LIMIT {
            return Err(FailureKind::ResourceLimit);
        }
        let mut byte = [0];
        input
            .read_exact(&mut byte)
            .map_err(|_| FailureKind::Transport)?;
        header.push(byte[0]);
    }
    let text = std::str::from_utf8(&header).map_err(|_| FailureKind::Protocol)?;
    let mut length = None;
    for line in text[..text.len() - 4].split("\r\n") {
        let (name, value) = line.split_once(':').ok_or(FailureKind::Protocol)?;
        if name.eq_ignore_ascii_case("Content-Length") {
            if length.is_some()
                || value.trim().is_empty()
                || !value.trim().bytes().all(|b| b.is_ascii_digit())
            {
                return Err(FailureKind::Protocol);
            }
            length = Some(
                value
                    .trim()
                    .parse::<usize>()
                    .map_err(|_| FailureKind::ResourceLimit)?,
            );
        } else if !name.eq_ignore_ascii_case("Content-Type") {
            return Err(FailureKind::Protocol);
        }
    }
    let length = length
        .filter(|length| *length > 0 && *length <= MAX_FRAME)
        .ok_or(FailureKind::ResourceLimit)?;
    *total = total
        .checked_add(header.len())
        .and_then(|value| value.checked_add(length))
        .filter(|total| *total <= MAX_STREAM)
        .ok_or(FailureKind::ResourceLimit)?;
    let mut body = vec![0; length];
    input
        .read_exact(&mut body)
        .map_err(|_| FailureKind::Transport)?;
    serde_json::from_slice(&body).map_err(|_| FailureKind::Protocol)
}
pub(super) fn protocol() -> Failure {
    Failure::new(
        FailureKind::Protocol,
        "Copilot runtime returned an invalid or unsupported response",
    )
}
pub(super) fn limit() -> Failure {
    Failure::new(
        FailureKind::ResourceLimit,
        "Copilot request or response exceeds its bounded resource limits",
    )
}
fn transport() -> Failure {
    Failure::new(
        FailureKind::Transport,
        "Copilot runtime transport failed; completion is unknown",
    )
}
