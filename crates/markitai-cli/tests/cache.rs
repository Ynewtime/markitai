use serde_json::{Value, json};
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::Path;
use std::process::{Command, Output, Stdio};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::time::{Duration, Instant};

struct ModelServer {
    base: String,
    calls: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl ModelServer {
    fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}/v1", listener.local_addr().unwrap());
        listener.set_nonblocking(true).unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let (counter, stopped) = (calls.clone(), stop.clone());
        let thread = std::thread::spawn(move || {
            while !stopped.load(Ordering::SeqCst) {
                let (mut stream, _) = match listener.accept() {
                    Ok(connection) => connection,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(2));
                        continue;
                    }
                    Err(error) => panic!("mock accept: {error}"),
                };
                stream.set_nonblocking(false).unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(3)))
                    .unwrap();
                let mut request = Vec::new();
                let deadline = Instant::now() + Duration::from_secs(3);
                loop {
                    let remaining = deadline.saturating_duration_since(Instant::now());
                    assert!(!remaining.is_zero(), "mock request deadline exceeded");
                    stream.set_read_timeout(Some(remaining)).unwrap();
                    let mut chunk = [0u8; 4096];
                    let count = match stream.read(&mut chunk) {
                        Ok(count) => count,
                        Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                        Err(error) => panic!("mock read: {error}"),
                    };
                    assert!(count > 0, "request ended early");
                    request.extend_from_slice(&chunk[..count]);
                    if let Some(split) = request.windows(4).position(|s| s == b"\r\n\r\n") {
                        let header = String::from_utf8_lossy(&request[..split]);
                        let length = header
                            .lines()
                            .find_map(|line| {
                                line.to_ascii_lowercase()
                                    .strip_prefix("content-length:")
                                    .and_then(|value| value.trim().parse::<usize>().ok())
                            })
                            .unwrap();
                        if request.len() >= split + 4 + length {
                            break;
                        }
                    }
                    assert!(request.len() < 1024 * 1024);
                }
                let split = request.windows(4).position(|s| s == b"\r\n\r\n").unwrap();
                let payload: Value = serde_json::from_slice(&request[split + 4..]).unwrap();
                let generation = counter.fetch_add(1, Ordering::SeqCst) + 1;
                let body = json!({"choices":[{"message":{"content":model_content(&payload,&format!("# Enhanced\n\nGeneration {generation}. 世界"))},"finish_reason":"stop"}],"usage":{"prompt_tokens":9,"completion_tokens":5}}).to_string();
                write!(stream,"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
            }
        });
        Self {
            base,
            calls,
            stop,
            thread: Some(thread),
        }
    }
    fn count(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl Drop for ModelServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        let result = self.thread.take().unwrap().join();
        if !std::thread::panicking() {
            result.expect("mock server failed");
        }
    }
}

fn invoke(root: &Path, args: &[&str]) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_markitai"));
    cmd.env_clear();
    for name in ["PATH", "SYSTEMROOT", "TMPDIR", "TEMP", "TMP"] {
        if let Some(value) = std::env::var_os(name) {
            cmd.env(name, value);
        }
    }
    cmd.current_dir(root)
        .env("MARKITAI_HOME", root.join("home"))
        .env("NO_PROXY", "127.0.0.1,localhost")
        .stdin(Stdio::null())
        .args(args)
        .output()
        .unwrap()
}

fn setup(root: &Path, server: &ModelServer) -> Value {
    std::fs::write(root.join("source.md"), "# Source\n\nUnchanged content.\n").unwrap();
    let cfg = json!({"llm":{"enabled":true,"on_failure":"fail","router_settings":{"num_retries":0},"model_list":[{"model_name":"mock","litellm_params":{"model":"openai/cache-test","api_base":server.base,"api_key":"local-test"}}]},"image":{"alt_enabled":false,"desc_enabled":false},"cache":{"enabled":true},"output":{"on_conflict":"overwrite"}});
    save(root, &cfg);
    cfg
}

fn save(root: &Path, cfg: &Value) {
    std::fs::write(root.join("markitai.json"), cfg.to_string()).unwrap();
}

fn json_output(output: Output) -> Value {
    assert!(
        output.status.success(),
        "stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

fn convert(root: &Path, source: &str, flags: &[&str]) -> Value {
    let mut args = vec![source, "-o", "out", "--json"];
    args.extend_from_slice(flags);
    let value = json_output(invoke(root, &args));
    assert_eq!(value["items"][0]["status"], "completed");
    value["items"][0].clone()
}

#[test]
fn persistent_hit_survives_process_rename_and_missing_credentials() {
    let root = tempfile::tempdir().unwrap();
    let server = ModelServer::start();
    let mut cfg = setup(root.path(), &server);
    let first = convert(root.path(), "source.md", &[]);
    assert_eq!(first["llm_cache_hit"], false);
    assert_eq!(server.count(), 1);
    assert!(
        first["llm_usage"]
            .as_object()
            .is_some_and(|v| !v.is_empty())
    );
    assert!(root.path().join("home/cache.db").is_file());
    std::fs::copy(
        root.path().join("source.md"),
        root.path().join("renamed.md"),
    )
    .unwrap();
    cfg["llm"]["model_list"][0]["litellm_params"]["api_key"] =
        json!("os.environ/MISSING_CACHE_TEST_KEY");
    cfg["llm"]["model_list"][0]["litellm_params"]["api_base"] = json!("http://127.0.0.1:9/v1");
    save(root.path(), &cfg);
    let hit = convert(root.path(), "renamed.md", &[]);
    assert_eq!(hit["llm_cache_hit"], true);
    assert_eq!(hit["cache_hit"], true);
    assert_eq!(hit["fetch_cache_hit"], false);
    assert_eq!(hit["cost_usd"], 0.0);
    assert_eq!(hit["llm_usage"], json!({}));
    assert_eq!(server.count(), 1);
    let rendered = std::fs::read_to_string(root.path().join("out/renamed.md.llm.md")).unwrap();
    assert!(rendered.contains("source: renamed.md"));
    assert!(rendered.contains("Generation 1."));
}

#[test]
fn bypass_refresh_patterns_last_flag_and_pure_keep_their_contracts() {
    let root = tempfile::tempdir().unwrap();
    let server = ModelServer::start();
    let mut cfg = setup(root.path(), &server);
    convert(root.path(), "source.md", &[]);
    let refreshed = convert(root.path(), "source.md", &["--no-cache"]);
    assert_eq!(refreshed["llm_cache_hit"], false);
    assert_eq!(server.count(), 2);
    assert_eq!(
        convert(root.path(), "source.md", &["--no-cache", "--cache"])["llm_cache_hit"],
        true
    );
    assert_eq!(server.count(), 2);
    convert(root.path(), "source.md", &["--cache", "--no-cache"]);
    assert_eq!(server.count(), 3);
    assert_eq!(
        convert(
            root.path(),
            "source.md",
            &["--no-cache-for", "other.md, **/*.md"]
        )["llm_cache_hit"],
        false
    );
    assert_eq!(server.count(), 4);
    assert_eq!(
        convert(root.path(), "source.md", &[])["llm_cache_hit"],
        true
    );
    let rendered = std::fs::read_to_string(root.path().join("out/source.md.llm.md")).unwrap();
    assert!(rendered.contains("Generation 4."));
    for _ in 0..2 {
        assert_eq!(
            convert(root.path(), "source.md", &["--pure"])["llm_cache_hit"],
            false
        );
    }
    assert_eq!(server.count(), 6);
    cfg["cache"] = json!({"enabled":false,"global_dir":root.path().join("disabled")});
    save(root.path(), &cfg);
    for _ in 0..2 {
        assert_eq!(
            convert(root.path(), "source.md", &["--cache"])["llm_cache_hit"],
            false
        );
    }
    assert_eq!(server.count(), 8);
    assert!(!root.path().join("disabled").exists());
    cfg["cache"] = json!({"enabled":true,"no_cache_patterns":["source.md"]});
    save(root.path(), &cfg);
    assert_eq!(
        convert(root.path(), "source.md", &["--no-cache-for", ""])["llm_cache_hit"],
        false
    );
    assert_eq!(server.count(), 9);
    assert_eq!(
        convert(root.path(), "source.md", &["--no-cache-for", " , "])["llm_cache_hit"],
        true
    );
    assert_eq!(server.count(), 9);
}

#[test]
fn stats_clear_and_incomplete_store_preflight_preserve_entries() {
    let root = tempfile::tempdir().unwrap();
    let server = ModelServer::start();
    setup(root.path(), &server);
    let empty = json_output(invoke(root.path(), &["cache", "stats", "--json"]));
    assert_eq!(
        empty,
        json!({"cache":null,"enabled":true,"fetch_cache":null})
    );
    assert!(!root.path().join("home/cache.db").exists());
    convert(root.path(), "source.md", &[]);
    let stats = json_output(invoke(
        root.path(),
        &["cache", "stats", "--json", "-v", "--limit", "1"],
    ));
    assert_eq!(stats["cache"]["count"], 1);
    assert_eq!(stats["cache"]["entries"].as_array().unwrap().len(), 1);
    assert!(stats["cache"]["size_bytes"].as_u64().unwrap() > 0);
    let abort = invoke(root.path(), &["cache", "clear"]);
    assert!(abort.status.success());
    assert!(String::from_utf8_lossy(&abort.stdout).contains("Aborted"));
    assert_eq!(
        convert(root.path(), "source.md", &[])["llm_cache_hit"],
        true
    );
    std::fs::write(root.path().join("home/fetch_cache.db"), "unmanaged fixture").unwrap();
    assert!(
        !invoke(root.path(), &["cache", "clear", "-y"])
            .status
            .success()
    );
    assert_eq!(
        convert(root.path(), "source.md", &[])["llm_cache_hit"],
        true
    );
    let stats_error = invoke(root.path(), &["cache", "stats", "--json"]);
    assert_eq!(stats_error.status.code(), Some(1));
    let stats_error: Value = serde_json::from_slice(&stats_error.stdout).unwrap();
    assert!(stats_error["fetch_cache"].get("error").is_some());
    std::fs::remove_file(root.path().join("home/fetch_cache.db")).unwrap();
    assert!(
        invoke(root.path(), &["cache", "clear", "-y"])
            .status
            .success()
    );
    let cleared = json_output(invoke(root.path(), &["cache", "stats", "--json"]));
    assert_eq!(cleared["cache"]["count"], 0);
    assert_eq!(
        convert(root.path(), "source.md", &[])["llm_cache_hit"],
        false
    );
    assert_eq!(server.count(), 2);
    std::fs::write(root.path().join("home/cache.db"), "corrupt fixture").unwrap();
    let corrupt_stats = invoke(root.path(), &["cache", "stats", "--json"]);
    assert_eq!(corrupt_stats.status.code(), Some(1));
    let corrupt_stats: Value = serde_json::from_slice(&corrupt_stats.stdout).unwrap();
    assert!(corrupt_stats["cache"].get("error").is_some());
    let uncached = convert(root.path(), "source.md", &[]);
    assert_eq!(uncached["llm_cache_hit"], false);
    assert!(
        uncached["warnings"]
            .as_array()
            .unwrap()
            .iter()
            .any(|warning| warning
                .as_str()
                .is_some_and(|text| text.contains("cache is unavailable")))
    );
    assert_eq!(server.count(), 3);
    std::fs::create_dir(root.path().join("batch")).unwrap();
    std::fs::write(
        root.path().join("batch/input.md"),
        "# Batch\n\nUncached body.\n",
    )
    .unwrap();
    let batch = invoke(root.path(), &["batch", "-o", "batch-out"]);
    assert!(batch.status.success());
    assert!(String::from_utf8_lossy(&batch.stderr).contains("cache is unavailable"));
    let quiet = invoke(root.path(), &["batch", "-o", "batch-out", "--quiet"]);
    assert!(quiet.status.success());
    assert!(quiet.stderr.is_empty());
    assert_eq!(server.count(), 5);
}

// Document cleanup is structured; pure vision and connection probes remain text.
fn model_content(request: &Value, markdown: &str) -> String {
    let messages = request["messages"].as_array().unwrap();
    if !messages.iter().any(|message| {
        message["role"] == "system"
            && message["content"]
                .as_str()
                .is_some_and(|text| text.contains("MARKITAI_DOCUMENT_JSON_V1"))
    }) {
        return markdown.to_owned();
    }
    // Echo protected source spans once and in order; test-specific output still
    // identifies the model invocation/generation used by the existing assertions.
    let source = messages
        .iter()
        .filter(|message| message["role"] == "user")
        .filter_map(|message| message["content"].as_str())
        .collect::<Vec<_>>()
        .join("\n");
    json!({"cleaned_markdown":format!("{markdown}\n\n{source}"),"frontmatter":{"description":"Local test document","tags":["fixture"]}}).to_string()
}

fn invoke_spa(root: &Path, args: &[&str]) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_markitai"));
    cmd.env_clear();
    for name in [
        "HOME",
        "PATH",
        "USERPROFILE",
        "SYSTEMROOT",
        "WINDIR",
        "TMPDIR",
        "LANG",
        "TZ",
    ] {
        if let Some(value) = std::env::var_os(name) {
            cmd.env(name, value);
        }
    }
    // LANG passes through, so pin the terminal language these assertions read.
    cmd.current_dir(root)
        .env("MARKITAI_HOME", root.join("home"))
        .env("MARKITAI_LANG", "en")
        .env("NO_PROXY", "127.0.0.1,localhost")
        .stdin(Stdio::null())
        .args(["--config", "markitai.json"])
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn learned_domain_commands_keep_empty_json_and_clear_contracts_without_creating_a_database() {
    let dir = tempfile::tempdir().unwrap();
    save(
        dir.path(),
        &json!({"cache":{"enabled":false},"log":{"dir":null},
        "prompts":{"dir":dir.path().join("prompts")}}),
    );
    assert_eq!(
        json_output(invoke_spa(dir.path(), &["cache", "spa-domains", "--json"])),
        json!([])
    );
    assert_eq!(
        json_output(invoke_spa(
            dir.path(),
            &["cache", "spa-domains", "--clear", "--json"]
        )),
        json!({"cleared":0})
    );
    let cleared = invoke_spa(
        dir.path(),
        &["cache", "clear", "--yes", "--include-spa-domains"],
    );
    assert!(
        cleared.status.success(),
        "{}",
        String::from_utf8_lossy(&cleared.stderr)
    );
    assert!(String::from_utf8_lossy(&cleared.stdout).contains("Cleared 0 learned SPA domains"));
    assert!(!dir.path().join("home/learned_spa_domains.db").exists());
}

#[test]
fn combined_clear_preflights_learned_domains_before_removing_cached_model_results() {
    let dir = tempfile::tempdir().unwrap();
    let server = ModelServer::start();
    let mut cfg = setup(dir.path(), &server);
    cfg["prompts"] = json!({"dir":dir.path().join("prompts")});
    cfg["log"] = json!({"dir":null});
    save(dir.path(), &cfg);
    let first = json_output(invoke_spa(
        dir.path(),
        &["source.md", "-o", "out", "--json"],
    ));
    assert_eq!(first["items"][0]["status"], "completed");
    assert_eq!(server.count(), 1);
    let cache = dir.path().join("home/cache.db");
    let before = std::fs::read(&cache).unwrap();
    let domains = dir.path().join("home/learned_spa_domains.db");
    let invalid = b"PRIVATE_DOMAIN_FIXTURE_CONTENT: not a SQLite database";
    std::fs::write(&domains, invalid).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&domains, std::fs::Permissions::from_mode(0o600)).unwrap();
    }
    for args in [
        vec!["cache", "spa-domains", "--json"],
        vec!["cache", "spa-domains", "--clear", "--json"],
        vec!["cache", "clear", "--yes", "--include-spa-domains"],
    ] {
        let output = invoke_spa(dir.path(), &args);
        assert!(!output.status.success());
        assert!(
            !String::from_utf8_lossy(&output.stderr).contains("PRIVATE_DOMAIN_FIXTURE_CONTENT")
        );
        assert_eq!(std::fs::read(&domains).unwrap(), invalid);
        assert_eq!(std::fs::read(&cache).unwrap(), before);
    }
    let cleared = invoke_spa(dir.path(), &["cache", "clear", "--yes"]);
    assert!(
        cleared.status.success(),
        "{}",
        String::from_utf8_lossy(&cleared.stderr)
    );
    let stats = json_output(invoke_spa(dir.path(), &["cache", "stats", "--json"]));
    assert_eq!(stats["cache"]["count"], 0);
    assert_eq!(std::fs::read(&domains).unwrap(), invalid);
}
