//! The command-line program. The `markitai` and `mkai` executables run it,
//! and so do the Python wheel's `markitai`, `mkai` and `markitai-mcp` entry
//! points, through the wheel's native extension, each in a process of its
//! own.

// Standard error goes through the CLI's logging sink (redaction, the file log,
// the terminal style); declared first so that every module prints that way.
macro_rules! eprintln {
    ($($argument:tt)*) => { $crate::app::logging::diagnostic(format_args!($($argument)*)) };
}
mod app;
mod diagnostics;
#[cfg(all(test, target_os = "linux"))]
mod file_lock_test;
mod history;
mod job_settings;
mod mcp;
mod output_claims;
mod pricing;
mod report;
mod report_store;
mod run_state;
mod server;
mod signals;
mod sort;
mod task;
mod term;
mod update;

/// Runs the command line `arguments`, the program name first (which selects
/// the MCP launcher when it is `markitai-mcp`), and returns the exit status
/// for the caller to exit with at once.
pub fn run(arguments: Vec<std::ffi::OsString>) -> i32 {
    limit_malloc_arenas();
    let code = app::run(arguments);
    signals::yield_to_handler();
    code
}

/// [`run`] for the Python wheel's entry points, which also say how the
/// wheel was installed (`prefix`, `executable`, `installer`, `user`), so
/// `markitai update` can name the command that updates it.
pub fn run_python(
    arguments: Vec<std::ffi::OsString>,
    install: std::collections::HashMap<String, String>,
) -> i32 {
    if let Some(install) = update::method::PythonInstall::from_map(&install) {
        update::method::set_python_install(install);
    }
    run(arguments)
}

/// glibc gives each thread that allocates a heap of its own, up to eight per
/// processor, and keeps what a thread frees in that heap. Local OCR reads a
/// scan on many threads that each free large buffers, so the freed memory
/// piled up: a 40-page scan peaked at 1.3 GB instead of 0.7 GB with four
/// heaps, which convert as fast. `MALLOC_ARENA_MAX` still decides when set.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn limit_malloc_arenas() {
    if std::env::var_os("MALLOC_ARENA_MAX").is_none() {
        // SAFETY: mallopt only sets an allocator parameter, and no other
        // thread of the program exists yet to allocate while it changes.
        unsafe {
            libc::mallopt(libc::M_ARENA_MAX, 4);
        }
    }
}

#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
fn limit_malloc_arenas() {}
