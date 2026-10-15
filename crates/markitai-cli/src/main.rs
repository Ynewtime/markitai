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
fn main() {
    limit_malloc_arenas();
    let code = app::run();
    signals::yield_to_handler();
    std::process::exit(code);
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
        // thread exists yet to allocate while it changes.
        unsafe {
            libc::mallopt(libc::M_ARENA_MAX, 4);
        }
    }
}

#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
fn limit_malloc_arenas() {}
