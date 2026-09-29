//! Signal handlers only set lock-free atomics; the coordinator performs I/O.
use std::sync::atomic::{AtomicI32, Ordering};

static INTERRUPTED: AtomicI32 = AtomicI32::new(0);

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
        unsafe { libc::_exit(128 + signal) }
    }
}

/// Kill external runtime process groups, then terminate by the same signal
/// with its default action, as if no handler had been installed.
#[cfg(unix)]
extern "C" fn fatal(signal: libc::c_int) {
    markitai_core::terminate_child_process_groups();
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
pub(crate) fn install_fatal_cleanup(owned: Owned) {
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
        #[cfg(not(unix))]
        {
            Err(std::io::Error::other(
                "Controlled recovery interruption is not supported on this platform yet",
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
    }
}
