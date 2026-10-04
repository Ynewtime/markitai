mod app;
mod diagnostics;
#[cfg(all(test, target_os = "linux"))]
mod file_lock_test;
mod history;
mod mcp;
#[cfg_attr(not(unix), allow(dead_code, unused_imports))]
mod output_claims;
mod pricing;
#[cfg_attr(not(unix), allow(dead_code))]
mod report;
mod report_store;
#[cfg_attr(not(unix), allow(dead_code, unused_imports))]
mod run_state;
mod server;
mod signals;
mod sort;
mod task;
fn main() {
    std::process::exit(app::run());
}
