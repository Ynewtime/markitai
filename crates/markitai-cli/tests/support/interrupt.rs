//! Interrupting a CLI started by a test, as a terminal would.
//!
//! Unix sends SIGINT to the process. Windows sends Ctrl-Break to the process
//! group that the CLI leads, which reaches it alone: Ctrl-C can only be sent
//! to every process on the console, the test runner included.
#![allow(dead_code)]

use std::process::{Child, Command};

/// Prepare `command` so that [`interrupt`] can reach the started process
/// alone: on Windows it leads a new process group on this console.
pub fn interruptible(command: &mut Command) -> &mut Command {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        windows::ensure_console();
        command.creation_flags(windows_sys::Win32::System::Threading::CREATE_NEW_PROCESS_GROUP);
    }
    command
}

/// Deliver the terminal interrupt to `child`, started through [`interruptible`].
pub fn interrupt(child: &Child) {
    #[cfg(unix)]
    assert_eq!(
        unsafe { libc::kill(child.id() as libc::pid_t, libc::SIGINT) },
        0,
        "{}",
        std::io::Error::last_os_error()
    );
    #[cfg(windows)]
    windows::interrupt(child.id());
}

/// Whether `pid` still names a process that has not been removed.
pub fn alive(pid: u32) -> bool {
    #[cfg(unix)]
    {
        unsafe { libc::kill(pid as libc::pid_t, 0) == 0 }
    }
    #[cfg(windows)]
    {
        windows::alive(pid)
    }
}

/// End `pid` regardless of what it is doing, after a failed assertion.
pub fn kill(pid: u32) {
    #[cfg(unix)]
    unsafe {
        libc::kill(pid as libc::pid_t, libc::SIGKILL);
    }
    #[cfg(windows)]
    windows::kill(pid);
}

/// Process table rows for `pids`, for a failure message.
pub fn process_states(pids: &[u32]) -> String {
    #[cfg(unix)]
    let output = {
        let list = pids
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(",");
        Command::new("ps")
            .args(["-o", "pid,ppid,pgid,stat,etime,command", "-p", &list])
            .output()
    };
    #[cfg(windows)]
    let output = {
        let mut command = Command::new("tasklist");
        command.args(["/V", "/FO", "CSV"]);
        for pid in pids {
            command.args(["/FI", &format!("PID eq {pid}")]);
        }
        command.output()
    };
    match output {
        Ok(output) => String::from_utf8_lossy(&output.stdout).into_owned(),
        Err(error) => format!("process listing unavailable: {error}"),
    }
}

#[cfg(windows)]
mod windows {
    use std::sync::Once;
    use windows_sys::Win32::Foundation::{CloseHandle, FALSE, WAIT_TIMEOUT};
    use windows_sys::Win32::System::Console::{
        AllocConsole, CTRL_BREAK_EVENT, GenerateConsoleCtrlEvent, GetConsoleProcessList,
        GetStdHandle, STD_ERROR_HANDLE, STD_INPUT_HANDLE, STD_OUTPUT_HANDLE, SetStdHandle,
    };
    use windows_sys::Win32::System::Threading::{
        OpenProcess, PROCESS_SYNCHRONIZE, PROCESS_TERMINATE, TerminateProcess, WaitForSingleObject,
    };

    /// Console control events only reach processes on the sender's console.
    /// A test runner started without one gets a console of its own, which the
    /// CLI then shares; its redirected standard handles are kept.
    pub(super) fn ensure_console() {
        static ONCE: Once = Once::new();
        ONCE.call_once(|| unsafe {
            let mut attached = [0u32; 1];
            if GetConsoleProcessList(attached.as_mut_ptr(), 1) != 0 {
                return;
            }
            let saved = [STD_INPUT_HANDLE, STD_OUTPUT_HANDLE, STD_ERROR_HANDLE]
                .map(|which| (which, GetStdHandle(which)));
            assert_ne!(
                AllocConsole(),
                0,
                "no console for control events: {}",
                std::io::Error::last_os_error()
            );
            for (which, handle) in saved {
                SetStdHandle(which, handle);
            }
        });
    }

    pub(super) fn interrupt(group: u32) {
        let sent = unsafe { GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, group) };
        assert_ne!(
            sent,
            0,
            "Ctrl-Break could not be sent: {}",
            std::io::Error::last_os_error()
        );
    }

    pub(super) fn alive(pid: u32) -> bool {
        let process = unsafe { OpenProcess(PROCESS_SYNCHRONIZE, FALSE, pid) };
        if process.is_null() {
            return false;
        }
        let running = unsafe { WaitForSingleObject(process, 0) } == WAIT_TIMEOUT;
        unsafe { CloseHandle(process) };
        running
    }

    pub(super) fn kill(pid: u32) {
        let process = unsafe { OpenProcess(PROCESS_TERMINATE, FALSE, pid) };
        if !process.is_null() {
            unsafe {
                TerminateProcess(process, 1);
                CloseHandle(process);
            }
        }
    }
}
