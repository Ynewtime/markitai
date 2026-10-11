//! Starting the CLI so that its piped marks do not depend on the test runner.
//!
//! On Windows the marks in a pipe or a file follow the output code page of
//! the console the CLI is attached to (the Unicode ones only at 65001), and a
//! test runner reached over SSH or started by hand has whatever console page
//! that machine uses. A CLI started without a console has no page to follow
//! and writes the Unicode marks, as on macOS and Linux, so the tests that read
//! them start it that way. Tests of the code page itself start it in a console
//! of their own instead.
#![allow(dead_code)]

use std::process::Command;

/// Start `command` without a console on Windows; unchanged elsewhere.
pub fn detached(command: &mut Command) -> &mut Command {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(windows_sys::Win32::System::Threading::DETACHED_PROCESS);
    }
    command
}
