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
    std::process::exit(app::run());
}
