use serde_json::{Value, json};
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::Path;
use std::process::{Command, Output, Stdio};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::time::Duration;

struct Server {
    base: String,
    requests: Arc<Mutex<Vec<String>>>,
    generation: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Server {
    fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let generation = Arc::new(AtomicUsize::new(1));
        let stop = Arc::new(AtomicBool::new(false));
        let (log, version, stopped) = (requests.clone(), generation.clone(), stop.clone());
        let thread = std::thread::spawn(move || {
            while !stopped.load(Ordering::SeqCst) {
                let (mut stream, _) = match listener.accept() {
                    Ok(connection) => connection,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(2));
                        continue;
                    }
                    Err(error) => panic!("mock accept failed: {error}"),
                };
                stream.set_nonblocking(false).unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let mut request_reader = bounded_fixture_io::Reader::new(
                    &stream,
                    std::time::Instant::now() + Duration::from_secs(5),
                );
                let mut bytes = Vec::new();
                loop {
                    let mut chunk = [0u8; 4096];
                    let n = request_reader.read(&mut chunk).unwrap();
                    assert!(n > 0, "request ended before complete headers/body");
                    bytes.extend_from_slice(&chunk[..n]);
                    if let Some(split) = bytes.windows(4).position(|s| s == b"\r\n\r\n") {
                        let headers = String::from_utf8_lossy(&bytes[..split]);
                        let length = headers
                            .lines()
                            .find_map(|line| {
                                line.to_ascii_lowercase()
                                    .strip_prefix("content-length:")
                                    .and_then(|n| n.trim().parse::<usize>().ok())
                            })
                            .unwrap_or(0);
                        if bytes.len() >= split + 4 + length {
                            break;
                        }
                    }
                    assert!(bytes.len() < 1024 * 1024);
                }
                let request = String::from_utf8(bytes).unwrap();
                log.lock().unwrap().push(request.clone());
                let generation = version.load(Ordering::SeqCst);
                let etag = format!("\"v{generation}\"");
                let conditional = request.starts_with("GET /conditional");
                let is_hit = conditional
                    && request.lines().any(|line| {
                        line.split_once(':').is_some_and(|(name, value)| {
                            name.eq_ignore_ascii_case("if-none-match") && value.trim() == etag
                        })
                    });
                let (status, mime, body) = if request.starts_with("POST ") {
                    ("200 OK", "application/json", json!({"choices":[{"message":{"content":model_content(&serde_json::from_str(request.split_once("\r\n\r\n").unwrap().1).unwrap(),"# Enhanced\n\nLocal model answer.")},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":3}}).to_string())
                } else if is_hit {
                    ("304 Not Modified", "text/html", String::new())
                } else {
                    (
                        "200 OK",
                        "text/html",
                        format!(
                            "<html><title>Cached page</title><article><h1>Cached page</h1><p>Generation {generation}. 世界</p></article></html>"
                        ),
                    )
                };
                let validators = if conditional {
                    format!("ETag: {etag}\r\nLast-Modified: Mon, 28 Sep 2026 00:00:00 GMT\r\n")
                } else {
                    String::new()
                };
                write!(stream, "HTTP/1.1 {status}\r\nContent-Type: {mime}\r\nContent-Length: {}\r\n{validators}Connection: close\r\n\r\n{body}", body.len()).unwrap();
            }
        });
        Self {
            base,
            requests,
            generation,
            stop,
            thread: Some(thread),
        }
    }
    fn count(&self) -> usize {
        self.requests.lock().unwrap().len()
    }
    fn advance(&self) {
        self.generation.fetch_add(1, Ordering::SeqCst);
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        let result = self.thread.take().unwrap().join();
        if !std::thread::panicking() {
            result.expect("mock HTTP server failed");
        }
    }
}

fn invoke(root: &Path, args: &[&str]) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_markitai"));
    command.env_clear();
    for name in ["PATH", "SYSTEMROOT", "TMPDIR", "TEMP", "TMP"] {
        if let Some(value) = std::env::var_os(name) {
            command.env(name, value);
        }
    }
    command
        .env("MARKITAI_HOME", root.join("home"))
        .env("NO_PROXY", "127.0.0.1,localhost")
        .current_dir(root)
        .stdin(Stdio::null())
        .args(args)
        .output()
        .unwrap()
}
fn success(output: Output) -> Value {
    assert!(
        output.status.success(),
        "stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}
fn save(root: &Path, cfg: &Value) {
    std::fs::write(root.join("markitai.json"), cfg.to_string()).unwrap();
}
fn setup(root: &Path) -> Value {
    let cfg = json!({"llm":{"enabled":false},"cache":{"enabled":true},"output":{"on_conflict":"overwrite"}});
    save(root, &cfg);
    cfg
}
fn convert(root: &Path, source: &str, flags: &[&str]) -> Value {
    let mut args = vec![source, "-o", "out", "--json"];
    args.extend_from_slice(flags);
    let value = success(invoke(root, &args));
    assert_eq!(value["items"][0]["status"], "completed");
    value["items"][0].clone()
}
fn fetched(item: &Value, expected: bool) {
    assert_eq!(item["fetch_cache_hit"], expected);
    assert_eq!(item["cache_hit"], false);
    assert_eq!(item["llm_cache_hit"], false);
    assert_eq!(item["fetch_strategy"], "static");
    assert_eq!(item["cost_usd"], 0.0);
}

#[test]
fn cross_process_reuse_bypass_scope_and_batch_keep_independent_hit_fields() {
    let root = tempfile::tempdir().unwrap();
    let server = Server::start();
    let url = format!("{}/page", server.base);
    let mut cfg = setup(root.path());
    fetched(&convert(root.path(), &url, &[]), false);
    fetched(&convert(root.path(), &url, &[]), true);
    assert_eq!(server.count(), 1);
    assert!(root.path().join("home/fetch_cache.db").is_file());
    server.advance();
    fetched(
        &convert(root.path(), &url, &["--no-cache", "--cache"]),
        true,
    );
    fetched(
        &convert(root.path(), &url, &["--cache", "--no-cache"]),
        false,
    );
    assert_eq!(server.count(), 2);
    let hit = convert(root.path(), &url, &[]);
    fetched(&hit, true);
    assert!(
        std::fs::read_to_string(root.path().join(hit["output"].as_str().unwrap()))
            .unwrap()
            .contains("Generation 2.")
    );
    fetched(
        &convert(root.path(), &url, &["--no-cache-for", "127.0.0.1"]),
        false,
    );
    assert_eq!(server.count(), 3);
    cfg["fetch"] = json!({"strategy":"static"});
    save(root.path(), &cfg);
    fetched(&convert(root.path(), &url, &[]), true);
    fetched(&convert(root.path(), &url, &["-s", "static"]), false);
    fetched(&convert(root.path(), &url, &["-s", "static"]), true);
    fetched(&convert(root.path(), &url, &["-s", "auto"]), true);
    assert_eq!(server.count(), 4);
    cfg["cache"]["fetch_ttl_seconds"] = json!(0);
    save(root.path(), &cfg);
    fetched(&convert(root.path(), &url, &[]), false);
    fetched(&convert(root.path(), &url, &[]), false);
    assert_eq!(server.count(), 6);
    cfg["cache"]["fetch_ttl_seconds"] = json!(86400);
    save(root.path(), &cfg);
    std::fs::write(root.path().join("batch.urls"), format!("{url} named\n")).unwrap();
    let batch = success(invoke(
        root.path(),
        &["batch.urls", "-o", "batch-out", "--json"],
    ));
    fetched(&batch["items"][0], true);
    assert_eq!(server.count(), 6);
    cfg["cache"] = json!({"enabled":false,"global_dir":root.path().join("disabled")});
    save(root.path(), &cfg);
    fetched(&convert(root.path(), &url, &["--cache"]), false);
    fetched(&convert(root.path(), &url, &["--cache"]), false);
    assert_eq!(server.count(), 8);
    assert!(!root.path().join("disabled").exists());
    cfg["cache"] = json!({"enabled":true});
    save(root.path(), &cfg);
    std::fs::write(
        root.path().join("home/fetch_cache.db"),
        "damaged fetch store",
    )
    .unwrap();
    let uncached = convert(root.path(), &url, &[]);
    fetched(&uncached, false);
    assert!(
        uncached["warnings"]
            .as_array()
            .unwrap()
            .iter()
            .any(|warning| warning
                .as_str()
                .is_some_and(|text| text.contains("fetch cache is unavailable")))
    );
    let plain = invoke(root.path(), &[&url, "-o", "out"]);
    assert!(plain.status.success());
    assert!(String::from_utf8_lossy(&plain.stderr).contains("fetch cache is unavailable"));
    let quiet = invoke(root.path(), &[&url, "-o", "out", "--quiet"]);
    assert!(quiet.status.success());
    assert!(quiet.stderr.is_empty());
    assert_eq!(server.count(), 11);
    let stats_error = invoke(root.path(), &["cache", "stats", "--json"]);
    assert_eq!(stats_error.status.code(), Some(1));
    let stats: Value = serde_json::from_slice(&stats_error.stdout).unwrap();
    assert!(stats["fetch_cache"]["error"].is_string());
}

#[test]
fn validators_stats_and_combined_clear_work_across_processes() {
    let root = tempfile::tempdir().unwrap();
    let server = Server::start();
    let url = format!("{}/conditional", server.base);
    let mut cfg = setup(root.path());
    cfg["cache"]["fetch_ttl_seconds"] = json!(0);
    save(root.path(), &cfg);
    fetched(&convert(root.path(), &url, &[]), false);
    fetched(&convert(root.path(), &url, &[]), true);
    assert_eq!(server.count(), 2);
    let conditional = server.requests.lock().unwrap()[1].to_ascii_lowercase();
    assert!(conditional.contains("if-none-match: \"v1\""));
    assert!(conditional.contains("if-modified-since: mon, 28 sep 2026 00:00:00 gmt"));
    assert!(conditional.contains("accept: text/markdown, text/html;q=0.9, */*;q=0.5"));
    server.advance();
    fetched(&convert(root.path(), &url, &[]), false);
    let updated = convert(root.path(), &url, &[]);
    fetched(&updated, true);
    assert_eq!(server.count(), 4);
    assert!(
        std::fs::read_to_string(root.path().join(updated["output"].as_str().unwrap()))
            .unwrap()
            .contains("Generation 2.")
    );
    cfg["llm"] = json!({"enabled":true,"on_failure":"fail","router_settings":{"num_retries":0},"model_list":[{"model_name":"mock","litellm_params":{"model":"openai/cache-test","api_base":format!("{}/v1",server.base),"api_key":"local-test"}}]});
    save(root.path(), &cfg);
    std::fs::write(root.path().join("source.md"), "# Source\n\nLocal input.\n").unwrap();
    assert_eq!(
        convert(root.path(), "source.md", &[])["llm_cache_hit"],
        false
    );
    let stats = success(invoke(root.path(), &["cache", "stats", "--json", "-v"]));
    assert_eq!(stats["cache"]["count"], 1);
    assert_eq!(stats["fetch_cache"]["count"], 1);
    assert!(stats["fetch_cache"]["size_bytes"].as_u64().unwrap() > 0);
    let plain_stats = invoke(root.path(), &["cache", "stats"]);
    assert!(plain_stats.status.success());
    assert!(String::from_utf8_lossy(&plain_stats.stdout).contains("URL fetch cache: 1 entry ("));
    // No answer on stdin: nothing is cleared and the run fails.
    let abort = invoke(root.path(), &["cache", "clear"]);
    assert_eq!(abort.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&abort.stdout).contains("Aborted"));
    let stats = success(invoke(root.path(), &["cache", "stats", "--json"]));
    assert_eq!(stats["cache"]["count"], 1);
    assert_eq!(stats["fetch_cache"]["count"], 1);
    let cleared = invoke(
        root.path(),
        &["cache", "clear", "-y", "--include-spa-domains"],
    );
    assert!(cleared.status.success());
    assert!(String::from_utf8_lossy(&cleared.stdout).contains("Cleared 2 cache entries"));
    assert!(String::from_utf8_lossy(&cleared.stdout).contains("Cleared 0 learned SPA domains"));
    let stats = success(invoke(root.path(), &["cache", "stats", "--json"]));
    assert_eq!(stats["cache"]["count"], 0);
    assert_eq!(stats["fetch_cache"]["count"], 0);
    std::fs::write(root.path().join("home/cache.db"), "damaged LLM store").unwrap();
    fetched(&convert(root.path(), &url, &["--no-llm"]), false);
    assert!(
        !invoke(root.path(), &["cache", "clear", "-y"])
            .status
            .success()
    );
    let stats_error = invoke(root.path(), &["cache", "stats", "--json"]);
    assert_eq!(stats_error.status.code(), Some(1));
    let stats: Value = serde_json::from_slice(&stats_error.stdout).unwrap();
    assert!(stats["cache"]["error"].is_string());
    assert_eq!(
        stats["fetch_cache"]["count"], 1,
        "failed preflight must retain the fetch store"
    );
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

#[cfg(test)]
mod bounded_fixture_io {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/support/bounded_read.rs"
    ));
}
