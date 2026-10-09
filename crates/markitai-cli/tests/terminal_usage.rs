#![cfg(unix)]

use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

// Real CLI processes use their own configuration, cache, history and output
// directories. HOME is inherited unchanged; provider environment is not.
fn run(root: &Path, source: &str, resume: bool, expected_exit: i32) -> Value {
    let stdout = tempfile::NamedTempFile::new_in(root).unwrap();
    let stderr = tempfile::NamedTempFile::new_in(root).unwrap();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_markitai"));
    cmd.env_clear();
    for key in [
        "HOME",
        "PATH",
        "SYSTEMROOT",
        "WINDIR",
        "LANG",
        "LC_ALL",
        "TZ",
    ] {
        if let Some(value) = std::env::var_os(key) {
            cmd.env(key, value);
        }
    }
    cmd.current_dir(root)
        .env("MARKITAI_HOME", root.join("home"))
        .env("TMPDIR", root.join("tmp"))
        .env("NO_PROXY", "127.0.0.1,localhost")
        .args(["--config", "markitai.json", source, "-o", "out", "--json"])
        .stdin(Stdio::null())
        .stdout(stdout.reopen().unwrap())
        .stderr(stderr.reopen().unwrap());
    if resume {
        cmd.arg("--resume");
    }
    let mut child = cmd.spawn().unwrap();
    let deadline = Instant::now() + Duration::from_secs(35);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!(
                "CLI timed out: {}",
                std::fs::read_to_string(stderr.path()).unwrap()
            );
        }
        std::thread::sleep(Duration::from_millis(5));
    };
    let out = std::fs::read(stdout.path()).unwrap();
    let err = std::fs::read_to_string(stderr.path()).unwrap();
    assert_eq!(
        status.code(),
        Some(expected_exit),
        "stdout={} stderr={err}",
        String::from_utf8_lossy(&out)
    );
    let envelope: Value = serde_json::from_slice(&out).unwrap();
    assert_eq!(envelope["version"], "1.0");
    assert!(!err.contains("Recorded in history:"));
    assert!(!err.contains("loopback-only"));
    envelope
}

struct Model {
    base: String,
    stop: Arc<AtomicBool>,
    recovered: Arc<AtomicBool>,
    requests: Arc<Mutex<Vec<Value>>>,
    thread: Option<JoinHandle<()>>,
}
impl Model {
    fn new() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let stop = Arc::new(AtomicBool::new(false));
        let recovered = Arc::new(AtomicBool::new(false));
        let requests = Arc::new(Mutex::new(Vec::new()));
        let stopping = stop.clone();
        let recovery = recovered.clone();
        let captured = requests.clone();
        let thread = std::thread::spawn(move || {
            while !stopping.load(Ordering::Acquire) {
                let mut stream = match listener.accept() {
                    Ok((stream, _)) => stream,
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(2));
                        continue;
                    }
                    Err(e) => panic!("loopback accept: {e}"),
                };
                stream.set_nonblocking(false).unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                stream
                    .set_write_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let deadline = Instant::now() + Duration::from_secs(5);
                let mut bytes = Vec::new();
                let mut buffer = [0; 8192];
                let end = loop {
                    let remaining = deadline.saturating_duration_since(Instant::now());
                    assert!(!remaining.is_zero(), "loopback request deadline exceeded");
                    stream.set_read_timeout(Some(remaining)).unwrap();
                    let n = match stream.read(&mut buffer) {
                        Ok(n) => n,
                        Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                        Err(error) => panic!("loopback read: {error}"),
                    };
                    assert!(n > 0);
                    bytes.extend_from_slice(&buffer[..n]);
                    assert!(bytes.len() < 2_000_000);
                    if let Some(i) = bytes.windows(4).position(|v| v == b"\r\n\r\n") {
                        break i + 4;
                    }
                };
                let header = String::from_utf8_lossy(&bytes[..end]);
                if header.starts_with("GET ") {
                    let bad = header.lines().next().unwrap().contains("/bad");
                    // Plain alphabetic sentinels survive HTML-to-Markdown escaping unchanged.
                    let marker = if bad {
                        "TERMINALBADPAID"
                    } else {
                        "TERMINALGOODDOCUMENT"
                    };
                    let body = format!(
                        "<!doctype html><title>Article</title><article><h1>Article</h1><p>{marker}: A complete authored document whose body must remain intact.</p></article>"
                    );
                    respond(&mut stream, 200, "text/html", body.as_bytes());
                    continue;
                }
                assert!(header.starts_with("POST "));
                let length = header
                    .lines()
                    .find_map(|line| {
                        line.split_once(':')
                            .filter(|(key, _)| key.eq_ignore_ascii_case("content-length"))
                            .map(|(_, value)| value.trim().parse::<usize>().unwrap())
                    })
                    .unwrap();
                assert!(length < 2_000_000);
                while bytes.len() < end + length {
                    let remaining = deadline.saturating_duration_since(Instant::now());
                    assert!(!remaining.is_zero(), "loopback request deadline exceeded");
                    stream.set_read_timeout(Some(remaining)).unwrap();
                    let n = match stream.read(&mut buffer) {
                        Ok(n) => n,
                        Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                        Err(error) => panic!("loopback read: {error}"),
                    };
                    assert!(n > 0);
                    bytes.extend_from_slice(&buffer[..n]);
                }
                let request: Value = serde_json::from_slice(&bytes[end..end + length]).unwrap();
                captured.lock().unwrap().push(request.clone());
                let text = request["messages"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|m| m["role"] == "user")
                    .unwrap()["content"]
                    .as_str()
                    .unwrap();
                let (status, response) = if text.contains("TERMINALBADINVALID") {
                    (200, reply("not a structured document"))
                } else if text.contains("TERMINALBAD") && !recovery.load(Ordering::Acquire) {
                    let mut response = json!({"error":{"code":"invalid_api_key","type":"invalid_api_key","message":"Incorrect API key provided: loopback-only"}});
                    if !text.contains("TERMINALBADUNKNOWN") {
                        let (input, output) = if text.contains("TERMINALBADZERO") {
                            (0, 0)
                        } else {
                            (11, 3)
                        };
                        response["usage"] =
                            json!({"prompt_tokens":input,"completion_tokens":output});
                    }
                    (401, response)
                } else {
                    assert!(
                        request["messages"][0]["content"]
                            .as_str()
                            .unwrap()
                            .contains("MARKITAI_DOCUMENT_JSON_V1")
                    );
                    (200, reply(&json!({"cleaned_markdown":text,"frontmatter":{"description":"An authored test document.","tags":["fixture"]}}).to_string()))
                };
                respond(
                    &mut stream,
                    status,
                    "application/json",
                    &serde_json::to_vec(&response).unwrap(),
                );
            }
        });
        Self {
            base,
            stop,
            recovered,
            requests,
            thread: Some(thread),
        }
    }
    fn count(&self) -> usize {
        self.requests.lock().unwrap().len()
    }
}
impl Drop for Model {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
fn respond(stream: &mut TcpStream, status: u16, mime: &str, body: &[u8]) {
    write!(stream, "HTTP/1.1 {status} Result\r\nContent-Type: {mime}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len()).unwrap();
    stream.write_all(body).unwrap();
}
fn reply(text: &str) -> Value {
    json!({"choices":[{"message":{"content":text},"finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":5}})
}
fn configure(root: &Path, model: &Model) {
    std::fs::create_dir(root.join("tmp")).unwrap();
    let cfg = json!({
        "prompts":{"dir":root.join("prompts")},"log":{"dir":null},
        "llm":{"enabled":true,"on_failure":"fail","keep_base":true,"concurrency":1,
            "router_settings":{"num_retries":0,"timeout":5},
            "model_list":[{"model_name":"default","litellm_params":{"model":"openai/terminal-fixture","api_key":"loopback-only","api_base":model.base}}]},
        "cache":{"enabled":false},"history":{"record":true},
        "fetch":{"strategy":"static","remote_consent":"never"},
        "ocr":{"enabled":false},"screenshot":{"enabled":false},
        "image":{"alt_enabled":false,"desc_enabled":false},
        "batch":{"concurrency":1,"url_concurrency":1},
        "output":{"report":true,"on_conflict":"overwrite"}
    });
    std::fs::write(root.join("markitai.json"), cfg.to_string()).unwrap();
}
fn input(root: &Path, name: &str, marker: &str) {
    let path = root.join(name);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, format!("# Original heading\n\n{marker}: A complete authored document whose body must remain intact.\n")).unwrap();
}
fn files(root: &Path, suffix: &str) -> Vec<PathBuf> {
    if !root.exists() {
        return vec![];
    }
    let mut paths: Vec<_> = std::fs::read_dir(root)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.file_name().unwrap().to_string_lossy().ends_with(suffix))
        .collect();
    paths.sort();
    paths
}
fn read(path: &Path) -> Value {
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}
fn report(root: &Path) -> Value {
    let paths = files(&root.join("out/.markitai/reports"), ".report.json");
    assert_eq!(paths.len(), 1, "report paths={paths:?}");
    read(&paths[0])
}
fn state(root: &Path) -> Value {
    let paths = files(&root.join("out/.markitai/states"), ".state.json");
    assert_eq!(paths.len(), 1, "state paths={paths:?}");
    read(&paths[0])
}
fn history(root: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    let path = root.join("home/serve/jobs");
    let mut jobs = BTreeMap::new();
    if !path.exists() {
        return jobs;
    }
    for entry in std::fs::read_dir(path).unwrap() {
        let path = entry.unwrap().path();
        if path.file_name().unwrap() == ".publish.lock" {
            assert!(path.is_file());
            continue;
        }
        let name = path.file_name().unwrap().to_str().unwrap();
        assert!(path.is_dir() && name.len() == 12 && name.bytes().all(|b| b.is_ascii_hexdigit()));
        let meta = path.join("meta.json");
        jobs.insert(meta.clone(), std::fs::read(meta).unwrap());
    }
    jobs
}
fn usage(diagnostics: &Value, status: &str, requests: u64, input: u64, output: u64) {
    let attempt = &diagnostics["last_attempt"];
    assert_eq!(attempt["operation"], "convert");
    assert_eq!(attempt["status"], status);
    if status == "error" {
        assert!(attempt["error"].is_string());
    } else {
        assert!(attempt["error"].is_null());
    }
    let value = &attempt["usage"];
    assert_eq!(value["requests"], requests);
    assert_eq!(value["input_tokens"], input);
    assert_eq!(value["output_tokens"], output);
    let records = value["by_model"].as_object().unwrap();
    assert_eq!(records.len(), 1);
    let model = records.values().next().unwrap();
    assert_eq!(model["requests"], requests);
    assert_eq!(model["input_tokens"], input);
    assert_eq!(model["output_tokens"], output);
    assert_eq!(value["cost_usd"], 0.0);
}
fn assert_failed(item: &Value, requests: u64, input: u64, output: u64) {
    assert_eq!(item["status"], "failed");
    assert!(item["error"].is_string());
    usage(&item["diagnostics"], "error", requests, input, output);
    assert_eq!(item["error"], item["diagnostics"]["last_attempt"]["error"]);
    assert_eq!(
        item["llm_usage"],
        item["diagnostics"]["last_attempt"]["usage"]["by_model"]
    );
    assert!(!item.to_string().contains("loopback-only"));
}

#[test]
fn single_failures_distinguish_paid_zero_unknown_and_all_invalid_attempts() {
    for (marker, expected) in [
        ("TERMINALBADPAID", Some((1, 11, 3))),
        ("TERMINALBADZERO", Some((1, 0, 0))),
        ("TERMINALBADUNKNOWN", None),
        ("TERMINALBADINVALID", Some((3, 21, 15))),
    ] {
        let root = tempfile::tempdir().unwrap();
        let model = Model::new();
        configure(root.path(), &model);
        input(root.path(), "note.md", marker);
        let result = run(root.path(), "note.md", false, 1);
        let item = &result["items"][0];
        assert_eq!(result["items"].as_array().unwrap().len(), 1);
        if let Some((requests, input, output)) = expected {
            assert_failed(item, requests, input, output);
            assert_eq!(model.count(), requests as usize);
        } else {
            assert_eq!(model.count(), 1);
            assert_eq!(item["status"], "failed");
            assert!(item["error"].is_string());
            assert!(item.get("diagnostics").is_none());
            assert_eq!(item["llm_usage"], json!({}));
        }
        assert!(files(&root.path().join("out/.markitai/reports"), ".report.json").is_empty());
        let jobs = history(root.path());
        assert_eq!(jobs.len(), 1);
        let meta: Value = serde_json::from_slice(jobs.values().next().unwrap()).unwrap();
        assert_eq!(meta["items"][0]["error"], item["error"]);
        assert_eq!(meta["items"][0].get("diagnostics"), item.get("diagnostics"));
    }
}

#[test]
fn early_failure_has_no_invented_measurement() {
    let root = tempfile::tempdir().unwrap();
    let model = Model::new();
    configure(root.path(), &model);
    std::fs::write(root.path().join("bad.ipynb"), "{not json").unwrap();
    let result = run(root.path(), "bad.ipynb", false, 1);
    assert_eq!(model.count(), 0);
    assert!(result["items"][0].get("diagnostics").is_none());
    assert_eq!(result["items"][0]["llm_usage"], json!({}));
    let jobs = history(root.path());
    assert_eq!(jobs.len(), 1);
    let meta: Value = serde_json::from_slice(jobs.values().next().unwrap()).unwrap();
    assert!(meta["items"][0].get("diagnostics").is_none());
}

#[test]
fn mixed_directory_reports_keep_success_totals_separate_from_terminal_observations() {
    let root = tempfile::tempdir().unwrap();
    let model = Model::new();
    configure(root.path(), &model);
    input(root.path(), "input/a.md", "TERMINALGOODDOCUMENT");
    input(root.path(), "input/b.md", "TERMINALBADPAID");
    let result = run(root.path(), "input", false, 10);
    assert_eq!(model.count(), 2);
    let items = result["items"].as_array().unwrap();
    assert_eq!(items.len(), 2);
    assert!(items[0]["source"].as_str().unwrap().ends_with("a.md"));
    assert!(items[1]["source"].as_str().unwrap().ends_with("b.md"));
    usage(&items[0]["diagnostics"], "done", 1, 7, 5);
    assert_failed(&items[1], 1, 11, 3);
    let report = report(root.path());
    assert_eq!(report["llm_usage"]["requests"], 1);
    assert_eq!(report["llm_usage"]["input_tokens"], 7);
    assert_eq!(report["llm_usage"]["output_tokens"], 5);
    assert_eq!(report["documents"]["b.md"]["llm_usage"], json!({}));
    assert_eq!(report["documents"]["b.md"]["cost_usd"], 0.0);
    assert_eq!(
        report["terminal_diagnostics"]["documents"]["a.md"],
        items[0]["diagnostics"]
    );
    assert_eq!(
        report["terminal_diagnostics"]["documents"]["b.md"],
        items[1]["diagnostics"]
    );
    let saved = state(root.path());
    assert_eq!(
        saved["documents"]["b.md"]["diagnostics"],
        items[1]["diagnostics"]
    );
    let jobs = history(root.path());
    assert_eq!(jobs.len(), 1);
    let meta: Value = serde_json::from_slice(jobs.values().next().unwrap()).unwrap();
    for (archived, actual) in meta["items"].as_array().unwrap().iter().zip(items) {
        assert_eq!(archived["diagnostics"], actual["diagnostics"]);
    }
}

#[test]
fn resumed_missing_source_clears_prior_paid_failure_without_rewriting_old_history() {
    let root = tempfile::tempdir().unwrap();
    let model = Model::new();
    configure(root.path(), &model);
    input(root.path(), "input/a.md", "TERMINALGOODDOCUMENT");
    input(root.path(), "input/b.md", "TERMINALBADPAID");
    run(root.path(), "input", false, 10);
    let previous = history(root.path());
    std::fs::remove_file(root.path().join("input/b.md")).unwrap();
    let result = run(root.path(), "input", true, 10);
    assert_eq!(model.count(), 2);
    assert_eq!(result["items"].as_array().unwrap().len(), 1);
    assert!(
        result["items"][0]["source"]
            .as_str()
            .unwrap()
            .ends_with("b.md")
    );
    assert!(result["items"][0].get("diagnostics").is_none());
    assert_eq!(result["items"][0]["llm_usage"], json!({}));
    let saved = state(root.path());
    assert!(saved["documents"]["b.md"].get("diagnostics").is_none());
    let report = report(root.path());
    usage(
        &report["terminal_diagnostics"]["documents"]["a.md"],
        "done",
        1,
        7,
        5,
    );
    assert!(
        report["terminal_diagnostics"]["documents"]
            .get("b.md")
            .is_none()
    );
    let jobs = history(root.path());
    assert_eq!(jobs.len(), 2);
    for (path, bytes) in &previous {
        assert_eq!(jobs.get(path), Some(bytes));
    }
    let (_, bytes) = jobs
        .iter()
        .find(|(path, _)| !previous.contains_key(*path))
        .unwrap();
    let meta: Value = serde_json::from_slice(bytes).unwrap();
    assert_eq!(meta["items"].as_array().unwrap().len(), 1);
    assert!(meta["items"][0].get("diagnostics").is_none());
}

#[test]
fn url_list_retry_replaces_paid_failure_and_compaction_preserves_latest_identity() {
    let root = tempfile::tempdir().unwrap();
    let model = Model::new();
    configure(root.path(), &model);
    let good = format!("{}/good", model.base);
    let bad = format!("{}/bad", model.base);
    let good_key = format!("{good} first-name");
    let bad_key = format!("{bad} retry-name.md");
    std::fs::write(
        root.path().join("pages.urls"),
        format!("{good_key}\n{bad_key}\n"),
    )
    .unwrap();
    let result = run(root.path(), "pages.urls", false, 10);
    assert_eq!(model.count(), 2);
    assert_eq!(result["items"][0]["source"], good);
    assert_eq!(result["items"][1]["source"], bad);
    assert_failed(&result["items"][1], 1, 11, 3);
    let initial = report(root.path());
    assert_eq!(initial["llm_usage"]["requests"], 1);
    assert_eq!(
        initial["terminal_diagnostics"]["urls"][&bad_key],
        result["items"][1]["diagnostics"]
    );
    let previous = history(root.path());
    model.recovered.store(true, Ordering::Release);
    let retried = run(root.path(), "pages.urls", true, 0);
    assert_eq!(model.count(), 3);
    assert_eq!(retried["items"].as_array().unwrap().len(), 1);
    assert_eq!(retried["items"][0]["source"], bad);
    usage(&retried["items"][0]["diagnostics"], "done", 1, 7, 5);
    let saved = state(root.path());
    assert_eq!(
        saved["urls"][&bad_key]["diagnostics"],
        retried["items"][0]["diagnostics"]
    );
    assert_eq!(saved["urls"][&bad_key]["status"], "completed");
    assert!(root.path().join("out/retry-name.llm.md").is_file());
    let jobs = history(root.path());
    assert_eq!(jobs.len(), 2);
    for (path, bytes) in previous {
        assert_eq!(jobs.get(&path), Some(&bytes));
    }
    let quiet = run(root.path(), "pages.urls", true, 0);
    assert_eq!(quiet["items"], json!([]));
    assert_eq!(model.count(), 3);
    assert_eq!(history(root.path()), jobs);
    let recovered = report(root.path());
    usage(
        &recovered["terminal_diagnostics"]["urls"][&good_key],
        "done",
        1,
        7,
        5,
    );
    usage(
        &recovered["terminal_diagnostics"]["urls"][&bad_key],
        "done",
        1,
        7,
        5,
    );
    assert_eq!(
        state(root.path())["urls"][&bad_key]["diagnostics"],
        retried["items"][0]["diagnostics"]
    );
}

#[test]
fn single_url_paid_failure_keeps_string_error_and_single_report_eligibility() {
    let root = tempfile::tempdir().unwrap();
    let model = Model::new();
    configure(root.path(), &model);
    let url = format!("{}/bad", model.base);
    let result = run(root.path(), &url, false, 1);
    assert_eq!(model.count(), 1);
    assert_eq!(result["items"][0]["source"], url);
    assert_failed(&result["items"][0], 1, 11, 3);
    assert!(files(&root.path().join("out/.markitai/reports"), ".report.json").is_empty());
    let jobs = history(root.path());
    assert_eq!(jobs.len(), 1);
    let meta: Value = serde_json::from_slice(jobs.values().next().unwrap()).unwrap();
    assert_eq!(meta["items"][0]["name"], url);
    assert_eq!(
        meta["items"][0]["diagnostics"],
        result["items"][0]["diagnostics"]
    );
}
