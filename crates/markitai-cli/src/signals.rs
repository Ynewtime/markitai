//! Interrupts and termination requests.
//!
//! Unix signal handlers only set lock-free atomics and call async-signal-safe
//! functions; the coordinator performs I/O. Windows console control events
//! arrive on a thread of their own and follow the same policy: Ctrl-C and
//! Ctrl-Break are the interrupt (recorded as SIGINT, exit status 130), and
//! closing the console window, logging off or shutting down is termination
//! (SIGTERM, 143). Windows gives a closing console only a few seconds, so
//! termination always cleans up and exits at once, as SIGHUP does on Unix.
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};

static INTERRUPTED: AtomicI32 = AtomicI32::new(0);

/// Whether a terminal status line (`app::progress`) is on screen. A handler
/// that is about to end the process erases it first, so the shell prompt does
/// not start behind a half-written status.
pub(crate) static STATUS_LINE: AtomicBool = AtomicBool::new(false);

fn erase_status_line() {
    if STATUS_LINE.swap(false, Ordering::Relaxed) {
        const ERASE: &[u8] = b"\r\x1b[K";
        #[cfg(unix)]
        // SAFETY: write(2) is async-signal-safe and reads only this constant.
        unsafe {
            libc::write(libc::STDERR_FILENO, ERASE.as_ptr().cast(), ERASE.len());
        }
        // A console control handler is an ordinary thread.
        #[cfg(not(unix))]
        {
            use std::io::Write;
            let _ = std::io::stderr().write_all(ERASE);
        }
    }
}

pub(crate) fn interrupted() -> Option<i32> {
    match INTERRUPTED.load(Ordering::Relaxed) {
        0 => None,
        signal => Some(signal),
    }
}

#[cfg(unix)]
extern "C" fn interrupt(signal: libc::c_int) {
    if INTERRUPTED.swap(signal, Ordering::Relaxed) != 0 {
        // A second explicit interrupt forces termination without running
        // cleanup. External runtimes live in their own process groups, beyond
        // the terminal's reach, so kill them first (async-signal-safe).
        markitai_core::terminate_child_process_groups();
        erase_status_line();
        unsafe { libc::_exit(128 + signal) }
    }
}

/// Kill external runtime process groups, then terminate by the same signal
/// with its default action, as if no handler had been installed.
#[cfg(unix)]
extern "C" fn fatal(signal: libc::c_int) {
    markitai_core::terminate_child_process_groups();
    erase_status_line();
    unsafe {
        let mut action: libc::sigaction = std::mem::zeroed();
        action.sa_sigaction = libc::SIG_DFL;
        libc::sigemptyset(&mut action.sa_mask);
        libc::sigaction(signal, &action, std::ptr::null_mut());
        // Delivered with the default action once this handler returns.
        libc::raise(signal);
    }
}

/// Terminating signals that a command handles itself; the rest get cleanup.
#[derive(Clone, Copy)]
pub(crate) enum Owned {
    /// Conversion and other commands: SIGINT, SIGTERM and SIGHUP.
    None,
    /// MCP drains on SIGINT asynchronously.
    Interrupt,
    /// Serve drains on SIGINT and SIGTERM asynchronously.
    InterruptAndTerminate,
}

/// Process-lifetime cleanup for signals without a command-specific policy.
/// They keep their default terminate-on-signal behavior; only the external
/// runtime groups this process started no longer outlive it. Signals that an
/// asynchronous runtime owns are excluded, because its registration chains to
/// a previously installed handler.
///
/// On Windows the asynchronous runtime owns Ctrl-C alone; Ctrl-Break and the
/// termination events still clean up here and exit with 130 or 143. A Ctrl-C
/// that a parent disabled for this process (as a new process group starts)
/// stays disabled, like an inherited ignored SIGINT on Unix.
pub(crate) fn install_fatal_cleanup(owned: Owned) {
    #[cfg(windows)]
    {
        console::RUNTIME_OWNS_CTRL_C.store(!matches!(owned, Owned::None), Ordering::Relaxed);
        // As on Unix, a host that cannot register keeps the default behavior.
        let _ = console::register();
    }
    #[cfg(not(any(unix, windows)))]
    let _ = owned;
    #[cfg(unix)]
    for signal in [libc::SIGINT, libc::SIGTERM, libc::SIGHUP] {
        let skip = match owned {
            Owned::None => false,
            Owned::Interrupt => signal == libc::SIGINT,
            Owned::InterruptAndTerminate => signal != libc::SIGHUP,
        };
        if skip {
            continue;
        }
        let mut previous: libc::sigaction = unsafe { std::mem::zeroed() };
        if unsafe { libc::sigaction(signal, std::ptr::null(), &mut previous) } != 0
            || previous.sa_sigaction == libc::SIG_IGN
        {
            // An inherited ignore (for example nohup) is preserved.
            continue;
        }
        let mut action: libc::sigaction = unsafe { std::mem::zeroed() };
        action.sa_sigaction = fatal as *const () as libc::sighandler_t;
        action.sa_flags = libc::SA_RESTART;
        unsafe {
            libc::sigemptyset(&mut action.sa_mask);
            libc::sigaction(signal, &action, std::ptr::null_mut());
        }
    }
}

/// Controlled interruption: the first interrupt is recorded for the
/// coordinator to drain, a second one kills external runtimes and exits.
pub(crate) struct Guard {
    #[cfg(unix)]
    previous: [(libc::c_int, libc::sigaction); 2],
}

impl Guard {
    pub(crate) fn install() -> std::io::Result<Self> {
        INTERRUPTED.store(0, Ordering::Relaxed);
        #[cfg(unix)]
        {
            // sigaction's all-zero representation is valid before sigemptyset;
            // only the C signal handler pointer is installed, with no Rust state.
            let mut action: libc::sigaction = unsafe { std::mem::zeroed() };
            action.sa_sigaction = interrupt as *const () as libc::sighandler_t;
            action.sa_flags = libc::SA_RESTART;
            unsafe {
                libc::sigemptyset(&mut action.sa_mask);
            }
            let mut previous = [
                (libc::SIGINT, unsafe { std::mem::zeroed() }),
                (libc::SIGTERM, unsafe { std::mem::zeroed() }),
            ];
            for index in 0..previous.len() {
                let (signal, old) = &mut previous[index];
                if unsafe { libc::sigaction(*signal, &action, old) } != 0 {
                    let error = std::io::Error::last_os_error();
                    for (signal, old) in &previous[..index] {
                        unsafe {
                            libc::sigaction(*signal, old, std::ptr::null_mut());
                        }
                    }
                    return Err(error);
                }
            }
            Ok(Self { previous })
        }
        #[cfg(windows)]
        {
            console::register()?;
            console::GUARDED.store(true, Ordering::Relaxed);
            Ok(Self {})
        }
        #[cfg(not(any(unix, windows)))]
        {
            Err(std::io::Error::other(
                "Controlled recovery interruption is not supported on this platform",
            ))
        }
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        #[cfg(unix)]
        for (signal, old) in &self.previous {
            unsafe {
                libc::sigaction(*signal, old, std::ptr::null_mut());
            }
        }
        #[cfg(windows)]
        console::GUARDED.store(false, Ordering::Relaxed);
    }
}

/// While held, an interactive child that shares this console receives
/// Ctrl-C and Ctrl-Break itself and this process ignores them, as a Unix
/// process that replaced itself with the child would never see them.
#[cfg(windows)]
pub(crate) struct Delegated(());

#[cfg(windows)]
impl Delegated {
    pub(crate) fn install() -> std::io::Result<Self> {
        console::register()?;
        console::DELEGATED.store(true, Ordering::Relaxed);
        Ok(Self(()))
    }
}

#[cfg(windows)]
impl Drop for Delegated {
    fn drop(&mut self) {
        console::DELEGATED.store(false, Ordering::Relaxed);
    }
}

/// A console control event, as the policy below distinguishes them.
#[cfg(any(windows, test))]
#[derive(Clone, Copy)]
enum Event {
    /// Ctrl-C (`ctrl_c`) or Ctrl-Break.
    Interrupt {
        ctrl_c: bool,
    },
    /// The console window closes, the user logs off or the system shuts down.
    Termination,
    Other,
}

/// What this process does with an event.
#[cfg(any(windows, test))]
#[derive(Debug, PartialEq, Eq)]
enum Response {
    /// Consumed without ending the process.
    Handled,
    /// Left to the next handler: an asynchronous runtime's, or the default.
    Passed,
    /// Kill the external runtime trees and exit with this status at once.
    Exit(i32),
}

/// Who owns the console's interrupts at the moment.
#[cfg(any(windows, test))]
#[derive(Clone, Copy, Default)]
struct Owners {
    /// A [`Guard`] drains on the first interrupt.
    guarded: bool,
    /// An asynchronous runtime handles Ctrl-C.
    runtime_owns_ctrl_c: bool,
    /// An interactive child that shares the console receives them itself.
    delegated: bool,
}

/// The Windows policy. Events are recorded under the Unix signal numbers, so
/// every platform reports the same exit status, 128 plus the signal.
#[cfg(any(windows, test))]
fn respond(event: Event, owners: Owners, recorded: &AtomicI32) -> Response {
    const SIGINT: i32 = 2;
    const SIGTERM: i32 = 15;
    match event {
        Event::Other => Response::Passed,
        // Windows allows a closing console only a few seconds.
        Event::Termination => Response::Exit(128 + SIGTERM),
        Event::Interrupt { .. } if owners.delegated => Response::Handled,
        Event::Interrupt { .. } if owners.guarded => {
            if recorded.swap(SIGINT, Ordering::Relaxed) == 0 {
                // The coordinator drains; a second interrupt exits.
                Response::Handled
            } else {
                Response::Exit(128 + SIGINT)
            }
        }
        Event::Interrupt { ctrl_c: true } if owners.runtime_owns_ctrl_c => Response::Passed,
        Event::Interrupt { .. } => Response::Exit(128 + SIGINT),
    }
}

#[cfg(windows)]
mod console {
    use super::{Event, INTERRUPTED, Owners, Response};
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicBool, Ordering};
    use windows_sys::Win32::Foundation::{FALSE, TRUE};
    use windows_sys::Win32::System::Console::{
        CTRL_BREAK_EVENT, CTRL_C_EVENT, CTRL_CLOSE_EVENT, CTRL_LOGOFF_EVENT, CTRL_SHUTDOWN_EVENT,
        SetConsoleCtrlHandler,
    };
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, TerminateProcess};
    use windows_sys::core::BOOL;

    /// A [`super::Guard`] is installed.
    pub(super) static GUARDED: AtomicBool = AtomicBool::new(false);
    /// An asynchronous runtime handles Ctrl-C itself; its handler, registered
    /// later, is called before this one and claims the events it receives.
    pub(super) static RUNTIME_OWNS_CTRL_C: AtomicBool = AtomicBool::new(false);
    /// A [`super::Delegated`] child owns the console's interrupts.
    pub(super) static DELEGATED: AtomicBool = AtomicBool::new(false);

    /// Register the handler once for the life of the process.
    pub(super) fn register() -> std::io::Result<()> {
        static REGISTERED: OnceLock<Result<(), i32>> = OnceLock::new();
        let result = REGISTERED.get_or_init(|| {
            if unsafe { SetConsoleCtrlHandler(Some(handler), TRUE) } != 0 {
                Ok(())
            } else {
                Err(std::io::Error::last_os_error().raw_os_error().unwrap_or(0))
            }
        });
        (*result).map_err(std::io::Error::from_raw_os_error)
    }

    extern "system" fn handler(event: u32) -> BOOL {
        let event = match event {
            CTRL_C_EVENT => Event::Interrupt { ctrl_c: true },
            CTRL_BREAK_EVENT => Event::Interrupt { ctrl_c: false },
            CTRL_CLOSE_EVENT | CTRL_LOGOFF_EVENT | CTRL_SHUTDOWN_EVENT => Event::Termination,
            _ => Event::Other,
        };
        let owners = Owners {
            guarded: GUARDED.load(Ordering::Relaxed),
            runtime_owns_ctrl_c: RUNTIME_OWNS_CTRL_C.load(Ordering::Relaxed),
            delegated: DELEGATED.load(Ordering::Relaxed),
        };
        match super::respond(event, owners, &INTERRUPTED) {
            Response::Handled => TRUE,
            Response::Passed => FALSE,
            Response::Exit(code) => exit_now(code),
        }
    }

    /// Kill the external runtime trees, erase the status line and end the
    /// process at once, running no other cleanup (Unix `_exit`).
    fn exit_now(code: i32) -> ! {
        markitai_core::terminate_child_process_groups();
        super::erase_status_line();
        unsafe {
            TerminateProcess(GetCurrentProcess(), code as u32);
        }
        std::process::exit(code)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CTRL_C: Event = Event::Interrupt { ctrl_c: true };
    const CTRL_BREAK: Event = Event::Interrupt { ctrl_c: false };

    #[test]
    fn windows_events_drain_once_in_a_batch_and_otherwise_clean_up_with_unix_statuses() {
        let none = AtomicI32::new(0);
        // A conversion: both interrupts clean up and exit 130; closing the
        // console 143; anything else is left to the default handler.
        let conversion = Owners::default();
        assert_eq!(respond(CTRL_C, conversion, &none), Response::Exit(130));
        assert_eq!(respond(CTRL_BREAK, conversion, &none), Response::Exit(130));
        assert_eq!(
            respond(Event::Termination, conversion, &none),
            Response::Exit(143)
        );
        assert_eq!(respond(Event::Other, conversion, &none), Response::Passed);
        assert_eq!(none.load(Ordering::Relaxed), 0);

        // A batch records the first interrupt for the coordinator and exits
        // at the second, whichever key; termination never waits.
        let batch = Owners {
            guarded: true,
            ..Owners::default()
        };
        let recorded = AtomicI32::new(0);
        assert_eq!(respond(CTRL_BREAK, batch, &recorded), Response::Handled);
        assert_eq!(recorded.load(Ordering::Relaxed), 2);
        assert_eq!(respond(CTRL_C, batch, &recorded), Response::Exit(130));
        assert_eq!(
            respond(Event::Termination, batch, &AtomicI32::new(0)),
            Response::Exit(143)
        );

        // Serve and MCP: their runtime owns Ctrl-C alone.
        let server = Owners {
            runtime_owns_ctrl_c: true,
            ..Owners::default()
        };
        assert_eq!(respond(CTRL_C, server, &none), Response::Passed);
        assert_eq!(respond(CTRL_BREAK, server, &none), Response::Exit(130));
        assert_eq!(
            respond(Event::Termination, server, &none),
            Response::Exit(143)
        );

        // During a delegated login the child handles interrupts; this
        // process neither records nor exits, but still ends with the console.
        let login = Owners {
            delegated: true,
            guarded: true,
            runtime_owns_ctrl_c: false,
        };
        let untouched = AtomicI32::new(0);
        for event in [CTRL_C, CTRL_BREAK] {
            assert_eq!(respond(event, login, &untouched), Response::Handled);
        }
        assert_eq!(untouched.load(Ordering::Relaxed), 0);
        assert_eq!(
            respond(Event::Termination, login, &untouched),
            Response::Exit(143)
        );
    }
}
