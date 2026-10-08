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
fn main() {
    std::process::exit(app::run());
}
