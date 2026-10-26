//! Every integration test file is a module of this one test binary: linking
//! one binary instead of one per file is most of a cold test build's time.
//! Cargo.toml turns automatic test discovery off, so a new `tests/*.rs` file
//! must be listed here; `every_test_file_is_a_module` fails until it is.

// Support code the test modules share, declared once (`use crate::console;`).
#[path = "support/console.rs"]
mod console;
#[path = "../../markitai-core/src/subscription/fake_runtime.rs"]
mod fake_runtime;
#[path = "support/interrupt.rs"]
mod interrupt;

mod cache;
mod chatgpt_auth;
mod cli_help;
mod cli_messages;
mod cli_polish;
mod cli_ux;
mod cli_zh;
mod contracts;
mod doctor;
mod document_json;
mod empty_home;
mod fetch_cache;
mod guided;
mod help_language;
mod history;
mod interactive_config;
mod interrupted_batch;
mod llm_concurrency;
mod llm_keys;
mod logging;
mod mcp;
mod mcp_launcher;
mod numbers_packages;
mod output_preflight;
mod profile_assets;
mod provider_batch;
mod python_batch;
mod recovery;
mod remote_fetch;
mod reports;
mod runtime_interrupt;
mod serve;
mod serve_keyless;
mod serve_providers;
mod serve_settings;
mod serve_startup;
mod serve_terminal_usage;
mod serve_web;
mod stdout_closed;
mod stdout_images;
mod subscription_auth;
mod terminal_language;
mod terminal_usage;
mod update;
mod zh_output;

#[test]
fn every_test_file_is_a_module() {
    let tests = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let listed = include_str!("integration.rs");
    let mut missing = Vec::new();
    for entry in std::fs::read_dir(&tests).unwrap() {
        let path = entry.unwrap().path();
        let stem = path.file_stem().unwrap().to_str().unwrap().to_owned();
        if path.extension().is_some_and(|extension| extension == "rs")
            && stem != "integration"
            && !listed.contains(&format!("\nmod {stem};\n"))
        {
            missing.push(stem);
        }
    }
    assert!(
        missing.is_empty(),
        "add `mod NAME;` to tests/integration.rs for: {missing:?}"
    );
}
