use serde_json::{Value, json};
use std::path::Path;
use std::process::{Command, Output};

fn invoke(root: &Path, args: &[&str]) -> Output {
    invoke_env(root, args, &[])
}
fn invoke_env(root: &Path, args: &[&str], overrides: &[(&str, &str)]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_markitai"))
        .current_dir(root)
        .env("MARKITAI_HOME", root.join("home"))
        .env_remove("MARKITAI_CONFIG")
        .env_remove("MODEL")
        .env_remove("MARKITAI_PURE")
        .env_remove("MARKITAI_RECORD_HISTORY")
        .env("NO_PROXY", "127.0.0.1,localhost")
        .envs(overrides.iter().copied())
        .args(args)
        .output()
        .expect("CLI starts")
}
fn fixture() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("note.txt"), "# Contract\n\nHello 世界\n").unwrap();
    dir
}
#[test]
fn single_stdout_is_markdown_and_explicit_file_target_is_exact() {
    let dir = fixture();
    let output = invoke(dir.path(), &["note.txt", "--pure"]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        "# Contract\n\nHello 世界\n"
    );
    let output = invoke(dir.path(), &["note.txt", "-o", "chosen.md", "--json"]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let body: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(body["version"], "1.0");
    assert_eq!(body["ok"], true);
    assert!(dir.path().join("chosen.md").is_file());
    assert_eq!(body["items"][0]["status"], "completed");
}
#[test]
fn pure_stdout_keeps_existing_yaml_and_okf_still_emits_its_schema() {
    let dir = fixture();
    let text = "---\ntitle: Original\ncustom: 42\n---\n\n# Heading\nBody  \n";
    std::fs::write(dir.path().join("existing.md"), text).unwrap();
    let output = invoke(dir.path(), &["existing.md", "--pure"]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(String::from_utf8(output.stdout).unwrap(), text);
    let output = invoke(dir.path(), &["existing.md", "--pure", "--profile", "okf"]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let content = String::from_utf8(output.stdout).unwrap();
    assert!(content.contains("type: Document\n"));
    assert!(content.contains("custom: 42\n"));
    assert!(content.ends_with("# Heading\nBody  \n"));
}
#[test]
fn single_runtime_error_retains_json_and_usage_error_has_none() {
    let dir = fixture();
    let output = invoke(dir.path(), &["missing.txt", "-o", "out", "--json"]);
    assert_eq!(output.status.code(), Some(1));
    let body: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(body["ok"], false);
    assert_eq!(body["totals"]["failed"], 1);
    let output = invoke(dir.path(), &["note.txt", "--json"]);
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
}

/// A valid 1x1 RGBA PNG (every chunk checksum correct).
const TINY_PNG: &[u8] = &[
    137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 1, 0, 0, 0, 1, 8, 6, 0,
    0, 0, 31, 21, 196, 137, 0, 0, 0, 11, 73, 68, 65, 84, 120, 156, 99, 96, 0, 2, 0, 0, 5, 0, 1,
    122, 94, 171, 63, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130,
];

#[test]
fn image_only_cli_items_skip_without_writing_empty_documents() {
    let dir = fixture();
    // A real image with no extractor requested is skipped, not converted.
    std::fs::write(dir.path().join("scan.png"), TINY_PNG).unwrap();
    let stdout = invoke(dir.path(), &["scan.png"]);
    assert!(stdout.status.success());
    assert!(stdout.stdout.is_empty());
    let output = invoke(dir.path(), &["scan.png", "-o", "out", "--json"]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let body: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(body["items"][0]["status"], "skipped");
    assert_eq!(body["items"][0]["skip_reason"], "image_only");
    assert_eq!(body["totals"]["skipped"], 1);
    assert!(!dir.path().join("out/scan.png.md").exists());
    // A skip writes nothing at all: no output directory, no ownership files.
    assert!(!dir.path().join("out").exists());
    let missing = invoke(dir.path(), &["missing.jpg"]);
    assert_eq!(missing.status.code(), Some(1));
}

#[test]
fn a_file_that_is_not_an_image_is_an_error_and_not_a_skip() {
    let dir = fixture();
    std::fs::write(dir.path().join("empty.png"), b"").unwrap();
    std::fs::write(dir.path().join("fake.jpg"), b"not an image at all").unwrap();
    std::fs::write(dir.path().join("cut.png"), &TINY_PNG[..TINY_PNG.len() - 20]).unwrap();
    for (name, reason) in [
        ("empty.png", "File is empty (0 bytes)"),
        ("fake.jpg", "File is not a valid image: "),
        (
            "cut.png",
            "File is not a valid image: the file appears to be damaged or truncated",
        ),
    ] {
        let output = invoke(dir.path(), &[name, "-o", "out"]);
        assert_eq!(output.status.code(), Some(1), "{name}");
        let message = String::from_utf8_lossy(&output.stderr);
        assert!(
            message.starts_with(&format!("Error: {reason}")),
            "{name}: {message}"
        );
        assert!(!message.contains("Skipped"), "{message}");
    }
    // Nothing was claimed or created for any of them.
    assert!(!dir.path().join("out").exists());
    // In a batch they fail while a genuine image is skipped, and the summary
    // names both without internal keys.
    std::fs::create_dir(dir.path().join("pics")).unwrap();
    for name in ["empty.png", "fake.jpg"] {
        std::fs::rename(dir.path().join(name), dir.path().join("pics").join(name)).unwrap();
    }
    std::fs::write(dir.path().join("pics/real.png"), TINY_PNG).unwrap();
    let output = invoke(dir.path(), &["pics", "-o", "batch-out"]);
    assert_eq!(output.status.code(), Some(10));
    let message = String::from_utf8_lossy(&output.stderr);
    assert!(
        message.contains("Error: empty.png: File is empty (0 bytes)"),
        "{message}"
    );
    assert!(
        message.contains("Skipped 1 item (images with no text): real.png."),
        "{message}"
    );
    assert!(!message.contains("image_only"), "{message}");
    // Neither the skip nor the early failures left ownership metadata behind.
    let ownership = dir.path().join("batch-out/.markitai/ownership");
    assert!(!ownership.exists(), "{ownership:?}");
}

#[test]
fn paired_boolean_flags_use_the_last_occurrence() {
    let dir = fixture();
    let output = invoke(
        dir.path(),
        &[
            "note.txt",
            "--pure",
            "--no-pure",
            "--llm",
            "--no-llm",
            "--compress",
            "--no-compress",
        ],
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stdout.starts_with(b"---\n"));
    let output = invoke(
        dir.path(),
        &[
            "note.txt",
            "--no-pure",
            "--pure",
            "--pure",
            "--no-compress",
            "--compress",
        ],
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stdout.starts_with(b"# Contract\n"));
}
#[test]
fn batch_keeps_relative_directories_and_partial_failure_exit() {
    let dir = fixture();
    std::fs::create_dir_all(dir.path().join("input/nested")).unwrap();
    std::fs::write(dir.path().join("input/nested/good.txt"), "Good\n").unwrap();
    std::fs::write(dir.path().join("input/bad.ipynb"), "{bad").unwrap();
    let output = invoke(dir.path(), &["input", "-o", "out", "--json", "-j", "2"]);
    assert_eq!(
        output.status.code(),
        Some(10),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let body: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(body["totals"]["completed"], 1);
    assert_eq!(body["totals"]["failed"], 1);
    assert!(dir.path().join("out/nested/good.txt.md").is_file());
}
#[test]
fn config_overrides_are_transient_and_secret_values_hidden() {
    let dir = fixture();
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"llm":{"model_list":[{"model_name":"default","litellm_params":{"api_key":"do-not-print","model":"test"}}]}}"#,
    )
    .unwrap();
    let output = invoke(
        dir.path(),
        &[
            "-c",
            "config.json",
            "--config-json",
            r#"{"llm":{"enabled":true}}"#,
            "config",
            "list",
        ],
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(!text.contains("do-not-print"));
    let body: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(body["llm"]["enabled"], true);
    let persisted = std::fs::read_to_string(dir.path().join("config.json")).unwrap();
    assert!(!persisted.contains("enabled"));
}
#[test]
fn unsupported_options_fail_instead_of_ignoring_requests() {
    let dir = fixture();
    let output = invoke(dir.path(), &["note.txt", "--resume"]);
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("not implemented"));
    assert!(output.stdout.is_empty());
}
#[test]
fn dry_run_does_not_create_outputs() {
    let dir = fixture();
    let output = invoke(dir.path(), &["note.txt", "-o", "preview", "--dry-run"]);
    assert!(output.status.success());
    assert!(!dir.path().join("preview").exists());
}

#[test]
fn url_batch_overwrite_preserves_both_custom_named_results() {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    let dir = fixture();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    listener.set_nonblocking(true).unwrap();
    let server = std::thread::spawn(move || {
        for _ in 0..2 {
            let started = std::time::Instant::now();
            let (mut stream, _) = loop {
                match listener.accept() {
                    Ok(connection) => break connection,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        assert!(
                            started.elapsed() < std::time::Duration::from_secs(15),
                            "CLI did not connect to fixture server"
                        );
                        std::thread::sleep(std::time::Duration::from_millis(10));
                    }
                    Err(error) => panic!("Fixture accept failed: {error}"),
                }
            };
            stream.set_nonblocking(false).unwrap();
            stream
                .set_read_timeout(Some(std::time::Duration::from_secs(10)))
                .unwrap();
            let mut request_reader = bounded_fixture_io::Reader::new(
                &stream,
                std::time::Instant::now() + std::time::Duration::from_secs(10),
            );
            let mut buffer = [0u8; 8192];
            let count = request_reader.read(&mut buffer).unwrap();
            let request = String::from_utf8_lossy(&buffer[..count]);
            let body = if request.contains("?id=first") {
                "First result\n"
            } else {
                "Second result\n"
            };
            write!(stream,"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",body.len(),body).unwrap();
        }
    });
    std::fs::write(
        dir.path().join("sources.urls"),
        format!("http://{address}/page?id=first shared\nhttp://{address}/page?id=second shared\n"),
    )
    .unwrap();
    let output = invoke(
        dir.path(),
        &[
            "sources.urls",
            "-o",
            "out",
            "--json",
            "--config-json",
            r#"{"output":{"on_conflict":"overwrite"}}"#,
        ],
    );
    server.join().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let body: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(body["totals"]["completed"], 2);
    assert_ne!(body["items"][0]["output"], body["items"][1]["output"]);
    let first = std::fs::read_to_string(dir.path().join("out/shared.md")).unwrap();
    let second = std::fs::read_to_string(dir.path().join("out/shared.v2.md")).unwrap();
    assert!(first.contains("First result"));
    assert!(second.contains("Second result"));
}

#[test]
fn config_set_redacts_nested_credentials_when_echoing_whole_section() {
    let dir = fixture();
    let output = invoke(
        dir.path(),
        &[
            "-c",
            "new.json",
            "config",
            "set",
            "llm",
            r#"{"enabled":false,"model_list":[{"model_name":"default","litellm_params":{"api_key":"nested-secret","model":"test"}}]}"#,
        ],
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!String::from_utf8_lossy(&output.stdout).contains("nested-secret"));
    assert!(
        std::fs::read_to_string(dir.path().join("new.json"))
            .unwrap()
            .contains("nested-secret")
    );
}

#[test]
fn directory_filters_on_url_lists_are_usage_errors() {
    let dir = fixture();
    std::fs::write(dir.path().join("urls.urls"), "# no network\n").unwrap();
    let output = invoke(dir.path(), &["urls.urls", "-o", "out", "--max-depth", "1"]);
    assert_eq!(output.status.code(), Some(2));
    assert!(!dir.path().join("out").exists());
}

#[test]
fn stdout_ignores_history_and_explicit_opt_out_preserves_markdown() {
    let dir = fixture();
    let output = invoke_env(
        dir.path(),
        &["note.txt"],
        &[("MARKITAI_RECORD_HISTORY", "1")],
    );
    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).contains("# Contract"));
    assert!(!dir.path().join("home/serve/jobs").exists());
    let output = invoke_env(
        dir.path(),
        &["note.txt", "--no-record-history", "--pure"],
        &[("MARKITAI_RECORD_HISTORY", "1")],
    );
    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).starts_with("# Contract"));
    assert!(!dir.path().join("home/serve/jobs").exists());
}

#[test]
fn config_set_uses_declared_types_and_rejects_unknown_leaf_without_writes() {
    let dir = fixture();
    let output = invoke(
        dir.path(),
        &["-c", "config.json", "config", "set", "output.dir", "true"],
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let original = std::fs::read_to_string(dir.path().join("config.json")).unwrap();
    let value: Value = serde_json::from_str(&original).unwrap();
    assert_eq!(value["output"]["dir"], "true");
    let output = invoke(
        dir.path(),
        &["-c", "config.json", "config", "set", "image.qualty", "70"],
    );
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(
        std::fs::read_to_string(dir.path().join("config.json")).unwrap(),
        original
    );
}

#[test]
fn config_set_rejects_transient_overrides_before_creating_file() {
    let dir = fixture();
    let output = invoke(
        dir.path(),
        &[
            "-c",
            "new.json",
            "--config-json",
            "{}",
            "config",
            "set",
            "llm.enabled",
            "true",
        ],
    );
    assert_eq!(output.status.code(), Some(2));
    assert!(!dir.path().join("new.json").exists());
}

#[test]
fn config_display_hides_url_credentials_and_header_values_but_keeps_environment_references() {
    let dir = fixture();
    let cfg = json!({"llm":{"model_list":[{"model_name":"default","litellm_params":{"model":"test","api_key":"env:CONFIG_TEST_KEY","api_base":"https://user:fake-pass@example.com:443/private?token=fake-token#hidden","max_tokens":777}}]},"fetch":{"playwright":{"extra_http_headers":{"X-Access":"env:INLINE_SECRET"}}}});
    std::fs::write(dir.path().join("markitai.json"), cfg.to_string()).unwrap();
    let output = invoke(dir.path(), &["config", "list", "--format", "json"]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let displayed: Value = serde_json::from_slice(&output.stdout).unwrap();
    let model = &displayed["llm"]["model_list"][0]["litellm_params"];
    assert_eq!(model["api_base"], "https://example.com:443");
    assert_eq!(model["api_key"], "env:CONFIG_TEST_KEY");
    assert_eq!(model["max_tokens"], 777);
    assert_eq!(
        displayed["fetch"]["playwright"]["extra_http_headers"]["X-Access"],
        "[REDACTED]"
    );
    let output = invoke(
        dir.path(),
        &[
            "config",
            "get",
            "fetch.playwright.extra_http_headers.X-Access",
        ],
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), "[REDACTED]");
    assert_eq!(
        serde_json::from_slice::<Value>(&std::fs::read(dir.path().join("markitai.json")).unwrap())
            .unwrap(),
        cfg
    );
}

#[cfg(unix)]
#[test]
fn config_set_preserves_symlink_and_unknown_file_keys() {
    let dir = fixture();
    let target = dir.path().join("real.json");
    let link = dir.path().join("link.json");
    std::fs::write(
        &target,
        r#"{"llm":{"enabled":false},"extension":{"keep":"yes"}}"#,
    )
    .unwrap();
    std::os::unix::fs::symlink("real.json", &link).unwrap();
    let output = invoke(
        dir.path(),
        &["-c", "link.json", "config", "set", "llm.enabled", "true"],
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(link.is_symlink());
    let saved: Value = serde_json::from_slice(&std::fs::read(target).unwrap()).unwrap();
    assert_eq!(
        saved,
        json!({"llm":{"enabled":true},"extension":{"keep":"yes"}})
    );
}

#[test]
fn config_validate_missing_positional_file_is_usage_error() {
    let dir = fixture();
    let output = invoke(dir.path(), &["config", "validate", "absent.json"]);
    assert_eq!(output.status.code(), Some(2));
    assert!(!dir.path().join("absent.json").exists());
    assert!(!String::from_utf8_lossy(&output.stdout).contains("valid"));
}

#[test]
fn config_display_omits_model_nulls_but_direct_null_and_dictionary_null_remain() {
    let dir = fixture();
    let cfg = json!({"llm":{"model_list":[{"model_name":"default","litellm_params":{"model":"test"}}],"router_settings":{"fallbacks":[{"custom":null}]}}});
    std::fs::write(dir.path().join("markitai.json"), cfg.to_string()).unwrap();
    for args in [vec!["config", "list"], vec!["config", "get", "llm"]] {
        let output = invoke(dir.path(), &args);
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let value: Value = serde_json::from_slice(&output.stdout).unwrap();
        let llm = if args.len() == 2 {
            &value["llm"]
        } else {
            &value
        };
        assert!(llm["model_list"][0].get("model_info").is_none());
        assert!(
            llm["model_list"][0]["litellm_params"]
                .get("api_key")
                .is_none()
        );
        assert_eq!(
            llm["router_settings"]["fallbacks"][0].get("custom"),
            Some(&Value::Null)
        );
    }
    let output = invoke(
        dir.path(),
        &["config", "get", "llm.model_list[0].litellm_params.api_key"],
    );
    assert!(output.status.success());
    assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), "null");
}

#[cfg(test)]
mod bounded_fixture_io {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/support/bounded_read.rs"
    ));
}
