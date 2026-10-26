//! Starting the installed program again in this server's place.
//!
//! Unix: `exec` in place, so the process keeps its PID, terminal and
//! foreground job (systemd and launchd see the same main process). The
//! listening socket stays open across the exec and is handed over
//! (`MARKITAI_SERVE_LISTEN_FD`), so connections queue instead of failing;
//! the access token goes through an inherited pipe
//! (`MARKITAI_SERVE_TOKEN_FD`), never the environment.
//!
//! Windows has no exec: the listener is closed, the new program is started
//! with the same arguments on the same console (it binds again, retrying for
//! up to five seconds), the token is written to its standard input, and this
//! process waits up to 30 seconds for it to report ready in the control
//! file. It then stays as an idle parent and exits with the child's code.
//! A child that does not get ready is ended, the files are put back, and
//! this process serves again itself. On Unix the new version watches itself
//! for as long ([`Watchdog`]): not ready in 30 seconds, it puts the old
//! version back and execs it.
use crate::update::control::{self, Pending};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::time::Duration;

#[cfg(unix)]
const LISTEN_FD: &str = "MARKITAI_SERVE_LISTEN_FD";
#[cfg(unix)]
const TOKEN_FD: &str = "MARKITAI_SERVE_TOKEN_FD";
#[cfg(windows)]
const TOKEN_STDIN: &str = "MARKITAI_SERVE_TOKEN_STDIN";
const RESTARTED_FROM: &str = "MARKITAI_SERVE_RESTARTED_FROM";
/// How long a new version may take to get ready before it is undone.
const READY: Duration = Duration::from_secs(30);
const NOT_READY: &str = "it did not get ready within 30 seconds";
/// How long a restarted Windows server retries binding the port its
/// parent just closed.
#[cfg(windows)]
const REBIND: Duration = Duration::from_secs(5);

/// What a restarted server receives from the one before it.
#[derive(Default)]
pub(super) struct Handover {
    pub(super) listener: Option<std::net::TcpListener>,
    pub(super) token: Option<String>,
    /// The version the previous server ran.
    pub(super) from: Option<String>,
    /// This process serves again after its restart failed (Windows: the
    /// child it started may have only just closed the port).
    pub(super) again: bool,
}

impl Handover {
    /// Take what the environment hands over. The inherited descriptors are
    /// marked close-on-exec again, so programs this server starts do not
    /// inherit them.
    pub(super) fn take() -> Self {
        let from = std::env::var(RESTARTED_FROM)
            .ok()
            .filter(|value| !value.is_empty());
        let mut handover = Self {
            from,
            ..Self::default()
        };
        if handover.from.is_none() {
            return handover;
        }
        #[cfg(unix)]
        {
            use std::os::fd::FromRawFd;
            let fd = |name: &str| {
                std::env::var(name)
                    .ok()
                    .and_then(|value| value.parse::<i32>().ok())
                    .filter(|fd| *fd > 2 && valid_fd(*fd))
            };
            if let Some(listen) = fd(LISTEN_FD) {
                close_on_exec(listen, true);
                // SAFETY: the previous server left this descriptor open for
                // this process alone; it is owned from here on.
                handover.listener = Some(unsafe { std::net::TcpListener::from_raw_fd(listen) });
            }
            if let Some(token) = fd(TOKEN_FD) {
                close_on_exec(token, true);
                // SAFETY: as above, the read end of the token pipe.
                let mut file = unsafe { std::fs::File::from_raw_fd(token) };
                let mut text = String::new();
                let _ = std::io::Read::read_to_string(
                    &mut std::io::Read::take(&mut file, 4096),
                    &mut text,
                );
                handover.token = Some(text.trim().to_owned()).filter(|token| !token.is_empty());
            }
        }
        #[cfg(windows)]
        if std::env::var_os(TOKEN_STDIN).is_some() {
            let mut line = String::new();
            let _ = std::io::BufRead::read_line(&mut std::io::stdin().lock(), &mut line);
            handover.token = Some(line.trim().to_owned()).filter(|token| !token.is_empty());
        }
        handover
    }

    /// What this process keeps for itself when its restart failed: the
    /// token, so open tabs stay signed in, and on Unix the listening socket,
    /// so the port never closes.
    pub(super) fn again(kept: Kept, token: Option<String>) -> Self {
        #[cfg(unix)]
        let listener = kept.listener.map(std::net::TcpListener::from);
        #[cfg(not(unix))]
        let listener = {
            let _ = kept;
            None
        };
        Self {
            listener,
            token,
            from: None,
            again: true,
        }
    }
}

#[cfg(unix)]
fn valid_fd(fd: i32) -> bool {
    // SAFETY: F_GETFD only reads the descriptor's flags.
    unsafe { libc::fcntl(fd, libc::F_GETFD) != -1 }
}

#[cfg(unix)]
fn close_on_exec(fd: i32, on: bool) {
    // SAFETY: F_SETFD changes only this descriptor's close-on-exec flag.
    unsafe {
        libc::fcntl(fd, libc::F_SETFD, if on { libc::FD_CLOEXEC } else { 0 });
    }
}

/// Bind as the server does, retrying for a moment after a Windows restart,
/// whose parent has only just closed the port.
pub(super) async fn bind(
    host: &str,
    port: u16,
    restarted: bool,
) -> std::io::Result<tokio::net::TcpListener> {
    #[cfg(windows)]
    if restarted {
        let deadline = std::time::Instant::now() + REBIND;
        loop {
            match tokio::net::TcpListener::bind((host, port)).await {
                Err(error)
                    if error.kind() == std::io::ErrorKind::AddrInUse
                        && std::time::Instant::now() < deadline =>
                {
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
                result => return result,
            }
        }
    }
    let _ = restarted;
    tokio::net::TcpListener::bind((host, port)).await
}

/// What the server keeps so it can be started again: a copy of the listening
/// socket (Unix).
#[derive(Default)]
pub(super) struct Kept {
    #[cfg(unix)]
    pub(super) listener: Option<std::os::fd::OwnedFd>,
}

impl Kept {
    pub(super) fn of(listener: &tokio::net::TcpListener) -> Self {
        #[cfg(unix)]
        {
            use std::os::fd::AsFd;
            Self {
                listener: listener.as_fd().try_clone_to_owned().ok(),
            }
        }
        #[cfg(not(unix))]
        {
            let _ = listener;
            Self {}
        }
    }
}

/// How a relaunch ended, when it returns at all.
pub(super) enum Relaunched {
    /// Windows: the child served and exited with this code.
    #[cfg_attr(not(windows), allow(dead_code))]
    Exited(i32),
    /// The new program could not take over; serve again here.
    Failed(String),
}

/// Start `exe` with this process's arguments in its place. Unix: does not
/// return unless the exec fails. `handover` passes the socket and token to
/// a program that understands them (one newer than 2.1.0).
pub(super) fn relaunch(
    exe: &Path,
    from: &str,
    token: Option<&str>,
    kept: &Kept,
    handover: bool,
    home: &Path,
) -> Relaunched {
    relaunch_with(exe, from, token, kept, handover, home, &[])
}

/// [`relaunch`] with environment changes (the tests' debug-build variables).
fn relaunch_with(
    exe: &Path,
    from: &str,
    token: Option<&str>,
    kept: &Kept,
    handover: bool,
    home: &Path,
    environment: &[(&str, Option<&str>)],
) -> Relaunched {
    let arguments: Vec<OsString> = std::env::args_os().collect();
    let mut command = std::process::Command::new(exe);
    command.args(arguments.iter().skip(1));
    command.env(RESTARTED_FROM, from);
    // The update marker, if any, is the restarted server's.
    match Pending::load(home) {
        Some(pending) => command.env(control::UPDATE_ID, pending.id),
        None => command.env_remove(control::UPDATE_ID),
    };
    for (name, value) in environment {
        match value {
            Some(value) => command.env(name, value),
            None => command.env_remove(name),
        };
    }
    #[cfg(unix)]
    {
        use std::os::fd::AsRawFd;
        use std::os::unix::process::CommandExt;
        if let Some(name) = arguments.first() {
            command.arg0(name);
        }
        command.env_remove(LISTEN_FD).env_remove(TOKEN_FD);
        let mut _token_pipe = None;
        if handover {
            if let Some(listener) = &kept.listener {
                close_on_exec(listener.as_raw_fd(), false);
                command.env(LISTEN_FD, listener.as_raw_fd().to_string());
            }
            if let Some(token) = token
                && let Some(read) = token_pipe(token)
            {
                close_on_exec(read.as_raw_fd(), false);
                command.env(TOKEN_FD, read.as_raw_fd().to_string());
                _token_pipe = Some(read);
            }
        } else {
            // Without handover the new program binds the port itself (the
            // kept copy closes on exec).
            command.env_remove(RESTARTED_FROM);
        }
        let error = command.exec();
        // Serving again here: the kept socket is this process's alone again.
        if let Some(listener) = &kept.listener {
            close_on_exec(listener.as_raw_fd(), true);
        }
        Relaunched::Failed(error.to_string())
    }
    #[cfg(windows)]
    {
        // The listener closed with the server; the child binds again.
        let _ = kept;
        command.env_remove(TOKEN_STDIN);
        if !handover {
            command.env_remove(RESTARTED_FROM);
        }
        let token = token.filter(|_| handover);
        if token.is_some() {
            command.env(TOKEN_STDIN, "1");
            command.stdin(std::process::Stdio::piped());
        }
        let mut child = match command.spawn() {
            Ok(child) => child,
            Err(error) => return Relaunched::Failed(error.to_string()),
        };
        if let (Some(token), Some(mut stdin)) = (token, child.stdin.take()) {
            let _ = std::io::Write::write_all(&mut stdin, format!("{token}\n").as_bytes());
        }
        let id = child.id();
        let deadline = std::time::Instant::now() + ready_wait();
        loop {
            if control::Control::read(home).is_some_and(|found| found.pid == id) {
                break;
            }
            match child.try_wait() {
                Ok(Some(status)) => {
                    return Relaunched::Failed(format!(
                        "it exited with status {}",
                        status.code().unwrap_or(-1)
                    ));
                }
                Err(error) => return Relaunched::Failed(error.to_string()),
                Ok(None) if std::time::Instant::now() >= deadline => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Relaunched::Failed(NOT_READY.into());
                }
                Ok(None) => std::thread::sleep(Duration::from_millis(100)),
            }
        }
        // The child shares this console: Ctrl-C is its to handle (it drains
        // and exits), and this idle parent then exits with its code.
        let _delegated = crate::signals::Delegated::install().ok();
        let code = child
            .wait()
            .ok()
            .and_then(|status| status.code())
            .unwrap_or(1);
        Relaunched::Exited(code)
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (command, token, kept, handover, home, from);
        Relaunched::Failed("restarting is not supported on this system".into())
    }
}

/// [`READY`]; a debug build takes `MARKITAI_TEST_UPDATE_READY_MS` (the tests).
fn ready_wait() -> Duration {
    std::env::var("MARKITAI_TEST_UPDATE_READY_MS")
        .ok()
        .filter(|_| cfg!(debug_assertions))
        .and_then(|value| value.parse().ok())
        .map_or(READY, Duration::from_millis)
}

#[derive(Debug, PartialEq, Eq)]
enum Watch {
    Waiting,
    Done,
    Fired,
}

/// Unix: a new version's first start that is not ready in [`READY`] puts
/// the old version back and execs it, with the handed token and socket.
pub(super) struct Watchdog(Arc<(Mutex<Watch>, Condvar)>);

impl Watchdog {
    /// Watch this start of `pending`'s new version (Unix; the Windows
    /// parent watches its child itself).
    pub(super) fn start(
        pending: Option<&Pending>,
        home: &Path,
        handover: Option<&Handover>,
    ) -> Option<Self> {
        if !cfg!(unix) {
            return None;
        }
        let pending = pending?.clone();
        let home: PathBuf = home.to_owned();
        let token = handover.and_then(|handover| handover.token.clone());
        #[cfg(unix)]
        let kept = Kept {
            listener: handover
                .and_then(|handover| handover.listener.as_ref())
                .and_then(|listener| std::os::fd::AsFd::as_fd(listener).try_clone_to_owned().ok()),
        };
        #[cfg(not(unix))]
        let kept = Kept::default();
        let shared = Arc::new((Mutex::new(Watch::Waiting), Condvar::new()));
        let watched = shared.clone();
        let wait = ready_wait();
        std::thread::Builder::new()
            .name("markitai-ready-watch".into())
            .spawn(move || {
                let (lock, changed) = &*watched;
                let state = lock.lock().unwrap_or_else(PoisonError::into_inner);
                let (mut state, _) = changed
                    .wait_timeout_while(state, wait, |state| *state == Watch::Waiting)
                    .unwrap_or_else(PoisonError::into_inner);
                if *state != Watch::Waiting {
                    return;
                }
                // Held from here on: the server can no longer commit.
                *state = Watch::Fired;
                let code =
                    roll_back_and_relaunch(&pending, &home, NOT_READY, token.as_deref(), kept);
                std::process::exit(code);
            })
            .ok()?;
        Some(Self(shared))
    }

    /// The start got ready or failed: whether it is still this process's to
    /// commit or undo (false once the watchdog has undone it).
    pub(super) fn finish(&self) -> bool {
        let (lock, changed) = &*self.0;
        let mut state = lock.lock().unwrap_or_else(PoisonError::into_inner);
        if *state == Watch::Fired {
            return false;
        }
        *state = Watch::Done;
        changed.notify_all();
        true
    }
}

/// A pipe holding `token`, its write end closed; the read end is returned.
#[cfg(unix)]
fn token_pipe(token: &str) -> Option<std::os::fd::OwnedFd> {
    use std::os::fd::FromRawFd;
    let mut fds = [0; 2];
    // SAFETY: `fds` has room for the two descriptors pipe(2) returns.
    if unsafe { libc::pipe(fds.as_mut_ptr()) } != 0 {
        return None;
    }
    // SAFETY: both descriptors were just created and are owned here.
    let (read, write) = unsafe {
        (
            std::os::fd::OwnedFd::from_raw_fd(fds[0]),
            std::os::fd::OwnedFd::from_raw_fd(fds[1]),
        )
    };
    close_on_exec(fds[0], true);
    close_on_exec(fds[1], true);
    let mut write = std::fs::File::from(write);
    // A token is far smaller than a pipe's buffer: this never blocks.
    std::io::Write::write_all(&mut write, token.as_bytes()).ok()?;
    drop(write);
    Some(read)
}

/// The previous update's new version failed before it got ready: put the
/// old version back, then start it in this process's place, with the token
/// when it takes one.
pub(super) fn roll_back_and_relaunch(
    pending: &Pending,
    home: &Path,
    reason: &str,
    token: Option<&str>,
    kept: Kept,
) -> i32 {
    let lang = crate::app::i18n::lang();
    let (to, from) = (&pending.to, &pending.from);
    crate::app::logging::shaped(
        crate::app::logging::Shape::item(crate::term::Mark::Error),
        &format!("Error: {to} didn't start: {reason}"),
        (lang == crate::app::i18n::Lang::Zh)
            .then(|| format!("{to} 无法启动：{reason}"))
            .as_deref(),
    );
    if let Err(error) = pending.roll_back(home, reason) {
        eprintln!("Error: Serve: cannot put {from} back: {error}");
        return 1;
    }
    crate::app::logging::shaped(
        crate::app::logging::Shape::item(crate::term::Mark::Action),
        &format!("Back on {from}"),
        (lang == crate::app::i18n::Lang::Zh)
            .then(|| format!("已回到 {from}"))
            .as_deref(),
    );
    // A Windows child tells its parent, which serves again itself.
    if cfg!(windows) && std::env::var_os(RESTARTED_FROM).is_some() {
        return 75;
    }
    // A debug build posing as the new version (the tests) runs as the old.
    let posed: &[(&str, Option<&str>)] = if cfg!(debug_assertions) {
        &[
            ("MARKITAI_TEST_UPDATE_FAIL_START", None),
            ("MARKITAI_TEST_UPDATE_HANG_START", None),
            ("MARKITAI_TEST_UPDATE_AS", Some(from.as_str())),
        ]
    } else {
        &[]
    };
    #[cfg(unix)]
    let has_listener = kept.listener.is_some();
    #[cfg(not(unix))]
    let has_listener = false;
    let handover = understands_handover(Some(from)) && (token.is_some() || has_listener);
    match relaunch_with(&pending.exe, to, token, &kept, handover, home, posed) {
        Relaunched::Exited(code) => code,
        Relaunched::Failed(error) => {
            eprintln!("Error: Serve: cannot start {from} again: {error}");
            1
        }
    }
}

/// Whether `version` can take a handed-over socket and token.
pub(super) fn understands_handover(version: Option<&str>) -> bool {
    version
        .and_then(crate::update::version::Version::parse)
        .is_some_and(|version| control::supports_handover(&version))
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    fn the_token_pipe_carries_the_token_and_closes_on_exec() {
        use std::os::fd::AsRawFd;
        let read = token_pipe("t0k3n").unwrap();
        // SAFETY: F_GETFD only reads the flags.
        let flags = unsafe { libc::fcntl(read.as_raw_fd(), libc::F_GETFD) };
        assert_eq!(flags & libc::FD_CLOEXEC, libc::FD_CLOEXEC);
        let mut text = String::new();
        std::io::Read::read_to_string(&mut std::fs::File::from(read), &mut text).unwrap();
        assert_eq!(text, "t0k3n");
    }

    #[test]
    fn only_newer_versions_take_a_handover() {
        assert!(!understands_handover(Some("2.1.0")));
        assert!(understands_handover(Some("2.2.0")));
        assert!(!understands_handover(None));
    }
}
