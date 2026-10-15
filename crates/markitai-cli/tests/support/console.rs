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

/// Pin the terminal language of a started CLI to English.
///
/// Without a language environment setting the CLI follows the display language
/// of the system on Windows, so a test that reads its sentences would pass on
/// an English machine and fail on a Chinese one. Every spawned CLI calls this
/// right after clearing the environment; a test that wants Chinese sets
/// `MARKITAI_LANG` itself afterwards, which replaces this value.
pub fn english(command: &mut Command) -> &mut Command {
    command.env("MARKITAI_LANG", "en")
}

/// As [`english`], except when the test passes a language variable of its own
/// (`MARKITAI_LANG`, `LC_ALL`, `LC_MESSAGES`, `LANG`) to check how the CLI
/// reads it: pinning would then hide what the test looks at.
pub fn english_unless_chosen<'a>(
    command: &'a mut Command,
    envs: &[(&str, &str)],
) -> &'a mut Command {
    let chosen = envs
        .iter()
        .any(|(name, _)| matches!(*name, "MARKITAI_LANG" | "LC_ALL" | "LC_MESSAGES" | "LANG"));
    if chosen { command } else { english(command) }
}
