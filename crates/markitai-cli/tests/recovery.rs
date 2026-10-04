//! Durable batch state, ownership and `--resume`.
#![cfg(unix)]

use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::sync::{
    Arc, Condvar, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

#[path = "support/interrupt.rs"]
mod interrupt;

const WAIT: Duration = Duration::from_secs(20);
const INTERRUPTED: &str = "Interrupted: stopping new work and waiting for active conversions.";

// Processes receive only an isolated configuration/home. Gates are ordinary HTTP
// responses, so these tests exercise production dispatch without hidden flags.
fn configure(root: &Path, server: &Server, model: bool) -> Value {
    let mut cfg = json!({
        "llm":{"enabled":false}, "ocr":{"enabled":false},
        "screenshot":{"enabled":false},
        "image":{"alt_enabled":false,"desc_enabled":false},
        "cache":{"enabled":false}, "history":{"record":false},
        "log":{"dir":null}, "prompts":{"dir":root.join("private-prompts")},
        "batch":{"concurrency":1,"url_concurrency":1,"scan_max_depth":8,
                 "state_flush_interval_seconds":3600},
        "output":{"on_conflict":"rename","report":false}
    });
    if model {
        cfg["llm"] = json!({
            "enabled":true,"on_failure":"fail","keep_base":true,
            "router_settings":{"num_retries":0,"timeout":30},
            "model_list":[{"model_name":"recovery","litellm_params":{
                "model":"openai/recovery-fixture","api_base":format!("{}/v1",server.base),
                "api_key":"local-fixture-only"}}]
        });
    }
    save_config(root, &cfg);
    cfg
}
fn save_config(root: &Path, cfg: &Value) {
    std::fs::write(root.join("markitai.json"), cfg.to_string()).unwrap();
}
fn write(root: &Path, relative: &str, text: &str) {
    let path = root.join(relative);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
}
fn wait_until(description: &str, mut ready: impl FnMut() -> bool) {
    let deadline = Instant::now() + WAIT;
    while !ready() {
        assert!(
            Instant::now() < deadline,
            "timed out waiting for {description}"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[derive(Default)]
struct Captured {
    stderr: Vec<u8>,
}
struct Running {
    child: Option<Child>,
    stdout: Option<JoinHandle<Vec<u8>>>,
    stderr: Option<JoinHandle<()>>,
    captured: Arc<(Mutex<Captured>, Condvar)>,
}
impl Running {
    fn spawn(root: &Path, cwd: &Path, args: &[&str]) -> Self {
        let mut command = Command::new(env!("CARGO_BIN_EXE_markitai"));
        command.env_clear();
        for name in ["HOME", "PATH", "SYSTEMROOT", "TMPDIR", "TEMP", "TMP"] {
            if let Some(value) = std::env::var_os(name) {
                command.env(name, value);
            }
        }
        interrupt::interruptible(&mut command);
        let mut child = command
            .current_dir(cwd)
            .env("MARKITAI_HOME", root.join("home"))
            .env("NO_PROXY", "127.0.0.1,localhost")
            .arg("--config")
            .arg(root.join("markitai.json"))
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let mut stdout = child.stdout.take().unwrap();
        let stderr = child.stderr.take().unwrap();
        let captured = Arc::new((Mutex::new(Captured::default()), Condvar::new()));
        let shared = captured.clone();
        Self {
            child: Some(child),
            stdout: Some(std::thread::spawn(move || {
                let mut bytes = Vec::new();
                stdout.read_to_end(&mut bytes).unwrap();
                bytes
            })),
            stderr: Some(std::thread::spawn(move || {
                let mut reader = BufReader::new(stderr);
                let mut line = Vec::new();
                while reader.read_until(b'\n', &mut line).unwrap() != 0 {
                    shared.0.lock().unwrap().stderr.extend_from_slice(&line);
                    shared.1.notify_all();
                    line.clear();
                }
            })),
            captured,
        }
    }
    fn interrupt(&self) {
        interrupt::interrupt(self.child.as_ref().unwrap());
        let deadline = Instant::now() + WAIT;
        let mut captured = self.captured.0.lock().unwrap();
        while !String::from_utf8_lossy(&captured.stderr)
            .lines()
            .any(|line| line == INTERRUPTED)
        {
            let remaining = deadline.saturating_duration_since(Instant::now());
            assert!(
                !remaining.is_zero(),
                "no interruption acknowledgement: {}",
                String::from_utf8_lossy(&captured.stderr)
            );
            captured = self.captured.1.wait_timeout(captured, remaining).unwrap().0;
        }
    }
    fn kill(mut self) -> Output {
        self.child.as_mut().unwrap().kill().unwrap();
        self.finish()
    }
    fn finish(mut self) -> Output {
        let deadline = Instant::now() + WAIT;
        let status = loop {
            if let Some(status) = self.child.as_mut().unwrap().try_wait().unwrap() {
                break status;
            }
            assert!(
                Instant::now() < deadline,
                "CLI did not exit; stderr={}",
                String::from_utf8_lossy(&self.captured.0.lock().unwrap().stderr)
            );
            std::thread::sleep(Duration::from_millis(5));
        };
        let _ = self.child.take();
        let stdout = self.stdout.take().unwrap().join().unwrap();
        self.stderr.take().unwrap().join().unwrap();
        let stderr = self.captured.0.lock().unwrap().stderr.clone();
        Output {
            status,
            stdout,
            stderr,
        }
    }
}
impl Drop for Running {
    fn drop(&mut self) {
        if let Some(child) = self.child.as_mut() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}
fn invoke(root: &Path, args: &[&str]) -> Output {
    Running::spawn(root, root, args).finish()
}
fn envelope(output: Output, code: i32) -> Value {
    assert_eq!(
        output.status.code(),
        Some(code),
        "stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value = serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "not one JSON envelope: {error}; stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    });
    assert_eq!(value["version"], "1.0");
    assert!(value["items"].is_array());
    value
}

#[derive(Default)]
struct HttpState {
    counts: HashMap<(String, String), usize>,
    held: HashSet<(String, String)>,
    failing_models: HashSet<String>,
}
struct Server {
    base: String,
    state: Arc<(Mutex<HttpState>, Condvar)>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}
impl Server {
    fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let state = Arc::new((Mutex::new(HttpState::default()), Condvar::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let shared = state.clone();
        let stopping = stop.clone();
        let thread = std::thread::spawn(move || {
            let mut workers = Vec::new();
            while !stopping.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((stream, _)) => {
                        let shared = shared.clone();
                        let stopping = stopping.clone();
                        workers.push(std::thread::spawn(move || serve(stream, shared, stopping)));
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(2));
                    }
                    Err(error) => panic!("loopback accept failed: {error}"),
                }
            }
            for worker in workers {
                worker.join().unwrap();
            }
        });
        Self {
            base,
            state,
            stop,
            thread: Some(thread),
        }
    }
    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base)
    }
    fn hold(&self, method: &str, key: &str) {
        self.state
            .0
            .lock()
            .unwrap()
            .held
            .insert((method.into(), key.into()));
    }
    fn release(&self, method: &str, key: &str) {
        self.state
            .0
            .lock()
            .unwrap()
            .held
            .remove(&(method.into(), key.into()));
        self.state.1.notify_all();
    }
    fn fail_model(&self, marker: &str, fail: bool) {
        let mut state = self.state.0.lock().unwrap();
        if fail {
            state.failing_models.insert(marker.into());
        } else {
            state.failing_models.remove(marker);
        }
    }
    fn count(&self, method: &str, key: &str) -> usize {
        *self
            .state
            .0
            .lock()
            .unwrap()
            .counts
            .get(&(method.into(), key.into()))
            .unwrap_or(&0)
    }
    fn wait(&self, method: &str, key: &str, count: usize) {
        let deadline = Instant::now() + WAIT;
        let mut state = self.state.0.lock().unwrap();
        while state
            .counts
            .get(&(method.into(), key.into()))
            .copied()
            .unwrap_or(0)
            < count
        {
            let remaining = deadline.saturating_duration_since(Instant::now());
            assert!(
                !remaining.is_zero(),
                "no {method} {key}; counts={:?}",
                state.counts
            );
            state = self.state.1.wait_timeout(state, remaining).unwrap().0;
        }
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        {
            let mut state = self.state.0.lock().unwrap();
            self.stop.store(true, Ordering::SeqCst);
            state.held.clear();
            self.state.1.notify_all();
        }
        let joined = self.thread.take().unwrap().join();
        if !std::thread::panicking() {
            joined.unwrap();
        }
    }
}
fn serve(mut stream: TcpStream, shared: Arc<(Mutex<HttpState>, Condvar)>, stop: Arc<AtomicBool>) {
    // Accepted sockets inherit nonblocking mode on some macOS versions.
    stream.set_nonblocking(false).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    stream
        .set_write_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    let mut bytes = Vec::new();
    let request_deadline = Instant::now() + Duration::from_secs(5);
    let (split, length) = loop {
        let remaining = request_deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return;
        }
        stream.set_read_timeout(Some(remaining)).unwrap();
        let mut chunk = [0u8; 4096];
        match stream.read(&mut chunk) {
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Ok(0) | Err(_) => return,
            Ok(n) => bytes.extend_from_slice(&chunk[..n]),
        }
        assert!(bytes.len() < 1024 * 1024);
        if let Some(split) = bytes.windows(4).position(|value| value == b"\r\n\r\n") {
            let headers = String::from_utf8_lossy(&bytes[..split]);
            let length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().unwrap())
                })
                .unwrap_or(0);
            if bytes.len() >= split + 4 + length {
                break (split, length);
            }
        }
    };
    let request = String::from_utf8(bytes).unwrap();
    let mut first = request.lines().next().unwrap().split_whitespace();
    let method = first.next().unwrap().to_owned();
    let path = first.next().unwrap().to_owned();
    let body = &request[split + 4..split + 4 + length];
    let marker = if method == "POST" {
        let start = body
            .find("MARKERX")
            .expect("mock model received no document marker");
        body[start..]
            .chars()
            .take_while(char::is_ascii_alphanumeric)
            .collect::<String>()
    } else {
        format!(
            "MARKERX{}",
            path.trim_matches('/')
                .replace('-', "X")
                .to_ascii_uppercase()
        )
    };
    let key = if method == "POST" {
        marker.clone()
    } else {
        path.clone()
    };
    let mut state = shared.0.lock().unwrap();
    *state
        .counts
        .entry((method.clone(), key.clone()))
        .or_default() += 1;
    shared.1.notify_all();
    while state.held.contains(&(method.clone(), key.clone())) && !stop.load(Ordering::SeqCst) {
        state = shared.1.wait(state).unwrap();
    }
    if stop.load(Ordering::SeqCst) {
        return;
    }
    let fail = method == "POST" && state.failing_models.contains(&marker);
    drop(state);
    let (status, mime, body) = if fail {
        (
            "400 Bad Request",
            "application/json",
            json!({"error":{"message":"deterministic model failure"}}).to_string(),
        )
    } else if method == "POST" {
        ("200 OK", "application/json", json!({
            "choices":[{"message":{"content":model_content(&serde_json::from_str(request.split_once("\r\n\r\n").unwrap().1).unwrap(),&format!("# Enhanced\n\n{marker} response."))},"finish_reason":"stop"}],
            "usage":{"prompt_tokens":7,"completion_tokens":3}
        }).to_string())
    } else {
        (
            "200 OK",
            "text/html",
            format!(
                "<html><title>Recovery fixture</title><article><h1>Recovery fixture</h1><p>{marker} local page content.</p></article></html>"
            ),
        )
    };
    // A killed client is an expected scenario; it does not invalidate the gate.
    let _ = write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: {mime}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
}

fn state_path(output: &Path) -> PathBuf {
    let paths: Vec<_> = std::fs::read_dir(output.join(".markitai/states"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.file_name()
                .unwrap()
                .to_string_lossy()
                .ends_with(".state.json")
        })
        .collect();
    assert_eq!(paths.len(), 1, "state paths={paths:?}");
    paths[0].clone()
}
fn snapshot(output: &Path) -> Value {
    serde_json::from_slice(&std::fs::read(state_path(output)).unwrap()).unwrap()
}
fn journal(output: &Path) -> Vec<Value> {
    let path = state_path(output).with_extension("jsonl");
    match std::fs::read_to_string(path) {
        Ok(text) => text
            .lines()
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_str(line).unwrap())
            .collect(),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(error) => panic!("journal read failed: {error}"),
    }
}
// Observe the final published status without substituting a second recovery
// implementation. Tests here do not manufacture stale or malformed journals.
fn published_entry(output: &Path, kind: &str, key: &str) -> Value {
    let mut entry = snapshot(output)[if kind == "url" { "urls" } else { "documents" }][key].clone();
    for event in journal(output) {
        if event["type"] == kind && event["key"] == key {
            for (name, value) in event["data"].as_object().unwrap() {
                entry[name] = value.clone();
            }
        }
    }
    entry
}
fn markdown_files(output: &Path) -> Vec<PathBuf> {
    let mut paths: Vec<_> = std::fs::read_dir(output)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "md"))
        .collect();
    paths.sort();
    paths
}
fn no_versions(output: &Path) {
    assert!(
        markdown_files(output).iter().all(|path| !path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .contains(".v2")),
        "retry created another version: {:?}",
        markdown_files(output)
    );
}
fn no_reports(output: &Path) {
    let path = output.join(".markitai/reports");
    assert!(!path.exists() || std::fs::read_dir(path).unwrap().next().is_none());
}

#[test]
fn url_list_retries_only_failed_owned_target_without_report_or_versions() {
    for retry_policy in ["rename", "skip"] {
        let root = tempfile::tempdir().unwrap();
        let server = Server::start();
        let mut cfg = configure(root.path(), &server, true);
        server.fail_model("MARKERXBAD", true);
        let good = server.url("/good");
        let bad = server.url("/bad");
        write(
            root.path(),
            "sources.urls",
            &format!("{good} done\n{bad} keep.md\n"),
        );
        let first = envelope(
            invoke(root.path(), &["sources.urls", "-o", "out", "--json"]),
            10,
        );
        assert_eq!(first["ok"], false);
        let out = root.path().join("out");
        let key = format!("{bad} keep.md");
        let failed = published_entry(&out, "url", &key);
        assert_eq!(failed["status"], "failed");
        assert_eq!(
            Path::new(failed["target"].as_str().unwrap())
                .file_name()
                .unwrap(),
            "keep.md"
        );
        assert!(
            std::fs::read_to_string(out.join("keep.md"))
                .unwrap()
                .contains("MARKERXBAD")
        );
        let completed_before = std::fs::read(out.join("done.llm.md")).unwrap();
        no_reports(&out);
        server.fail_model("MARKERXBAD", false);
        cfg["output"]["on_conflict"] = json!(retry_policy);
        save_config(root.path(), &cfg);
        envelope(
            invoke(
                root.path(),
                &["sources.urls", "-o", "out", "--resume", "--json"],
            ),
            0,
        );
        assert_eq!(server.count("GET", "/good"), 1);
        assert_eq!(server.count("POST", "MARKERXGOOD"), 1);
        assert_eq!(server.count("GET", "/bad"), 2);
        assert_eq!(server.count("POST", "MARKERXBAD"), 2);
        assert_eq!(
            std::fs::read(out.join("done.llm.md")).unwrap(),
            completed_before
        );
        let complete = published_entry(&out, "url", &key);
        assert_eq!(complete["status"], "completed");
        assert_eq!(
            Path::new(complete["output"].as_str().unwrap())
                .file_name()
                .unwrap(),
            "keep.llm.md"
        );
        assert!(
            std::fs::read_to_string(out.join("keep.llm.md"))
                .unwrap()
                .contains("MARKERXBAD")
        );
        no_versions(&out);
        no_reports(&out);
    }
}

#[test]
fn mixed_directory_merges_new_work_from_another_cwd_without_reopening_completed_items() {
    let root = tempfile::tempdir().unwrap();
    let server = Server::start();
    configure(root.path(), &server, true);
    let old = server.url("/old-url");
    let new = server.url("/new-url");
    write(
        root.path(),
        "input/old.txt",
        "MARKERXOLDXFILE original file.",
    );
    write(
        root.path(),
        "input/sources.urls",
        &format!("{old} remote\n"),
    );
    envelope(invoke(root.path(), &["input", "-o", "out", "--json"]), 0);
    let out = root.path().join("out");
    std::fs::remove_file(out.join("old.txt.llm.md")).unwrap();
    write(
        root.path(),
        "input/old.txt",
        "MARKERXEDITED must not run again.",
    );
    write(root.path(), "input/new.txt", "MARKERXNEWXFILE new file.");
    write(
        root.path(),
        "input/sources.urls",
        &format!("{old} remote\n{new} new-remote\n"),
    );
    let cwd = root.path().join("elsewhere");
    std::fs::create_dir(&cwd).unwrap();
    envelope(
        Running::spawn(
            root.path(),
            &cwd,
            &["../input", "-o", "../out", "--resume", "--json"],
        )
        .finish(),
        0,
    );
    assert_eq!(server.count("POST", "MARKERXOLDXFILE"), 1);
    assert_eq!(server.count("POST", "MARKERXEDITED"), 0);
    assert_eq!(server.count("GET", "/old-url"), 1);
    assert_eq!(server.count("POST", "MARKERXNEWXFILE"), 1);
    assert_eq!(server.count("GET", "/new-url"), 1);
    assert!(!out.join("old.txt.llm.md").exists());
    let saved = snapshot(&out);
    assert_eq!(saved["documents"].as_object().unwrap().len(), 2);
    assert_eq!(saved["urls"].as_object().unwrap().len(), 2);
    assert_eq!(saved["documents"]["old.txt"]["status"], "completed");
    assert!(out.join("new.txt.llm.md").is_file());
    envelope(
        invoke(root.path(), &["input", "-o", "out", "--resume", "--json"]),
        0,
    );
    assert_eq!(server.count("POST", "MARKERXNEWXFILE"), 1);
    assert_eq!(server.count("POST", "MARKERXNEWXURL"), 1);
    no_versions(&out);
    no_reports(&out);
}

#[test]
fn merged_checkpoint_and_active_claim_precede_http_and_survive_process_kill() {
    let root = tempfile::tempdir().unwrap();
    let server = Server::start();
    configure(root.path(), &server, false);
    let done = server.url("/done");
    let hold = server.url("/hold");
    let queued = server.url("/queued");
    write(root.path(), "list.urls", &format!("{done} done\n"));
    envelope(
        invoke(root.path(), &["list.urls", "-o", "out", "--json"]),
        0,
    );
    write(
        root.path(),
        "list.urls",
        &format!("{done} done\n{hold} hold\n{queued} queued\n"),
    );
    server.hold("GET", "/hold");
    let running = Running::spawn(
        root.path(),
        root.path(),
        &["list.urls", "-o", "out", "--resume", "--json"],
    );
    server.wait("GET", "/hold", 1);
    let out = root.path().join("out");
    let saved = snapshot(&out);
    assert_eq!(
        saved["urls"].as_object().unwrap().len(),
        3,
        "new keys were not checkpointed before HTTP"
    );
    let active = published_entry(&out, "url", &format!("{hold} hold"));
    assert_eq!(active["status"], "in_progress");
    let target = Path::new(active["target"].as_str().unwrap());
    assert!(target.is_absolute());
    assert_eq!(target.file_name().unwrap(), "hold.md");
    assert_eq!(
        published_entry(&out, "url", &format!("{queued} queued"))["status"],
        "in_progress"
    );
    assert_eq!(server.count("GET", "/queued"), 0);
    assert!(!running.kill().status.success());
    server.release("GET", "/hold");
    envelope(
        invoke(
            root.path(),
            &["list.urls", "-o", "out", "--resume", "--json"],
        ),
        0,
    );
    assert_eq!(server.count("GET", "/done"), 1);
    assert_eq!(server.count("GET", "/hold"), 2);
    assert_eq!(server.count("GET", "/queued"), 1);
    assert_eq!(
        published_entry(&out, "url", &format!("{hold} hold"))["status"],
        "completed"
    );
    no_versions(&out);
}

#[test]
fn ctrl_c_flushes_completed_work_then_drains_active_work_without_success_report() {
    let root = tempfile::tempdir().unwrap();
    let server = Server::start();
    let mut cfg = configure(root.path(), &server, false);
    cfg["batch"]["url_concurrency"] = json!(2);
    cfg["output"]["report"] = json!(true);
    save_config(root.path(), &cfg);
    let fast = server.url("/fast");
    let slow = server.url("/slow");
    write(
        root.path(),
        "list.urls",
        &format!("{fast} fast\n{slow} slow\n"),
    );
    server.hold("GET", "/fast");
    server.hold("GET", "/slow");
    let running = Running::spawn(
        root.path(),
        root.path(),
        &["list.urls", "-o", "out", "--json"],
    );
    server.wait("GET", "/fast", 1);
    server.wait("GET", "/slow", 1);
    server.release("GET", "/fast");
    let out = root.path().join("out");
    wait_until("first published document", || out.join("fast.md").is_file());
    // Both claims were flushed before releasing fast. With no further job to
    // claim and a long interval, the completion has not been saved by a new claim.
    assert_ne!(
        published_entry(&out, "url", &format!("{fast} fast"))["status"],
        "completed"
    );
    running.interrupt();
    server.release("GET", "/slow");
    let interrupted = running.finish();
    assert_eq!(
        interrupted.status.code(),
        Some(130),
        "{}",
        String::from_utf8_lossy(&interrupted.stderr)
    );
    no_reports(&out);
    for (url, name) in [(&fast, "fast"), (&slow, "slow")] {
        assert_eq!(
            published_entry(&out, "url", &format!("{url} {name}"))["status"],
            "completed"
        );
    }
    envelope(
        invoke(
            root.path(),
            &["list.urls", "-o", "out", "--resume", "--json"],
        ),
        0,
    );
    assert_eq!(server.count("GET", "/fast"), 1);
    assert_eq!(server.count("GET", "/slow"), 1);
    no_versions(&out);
}

#[test]
fn ctrl_c_stops_dispatch_before_releasing_an_active_request_and_resume_runs_the_queue() {
    let root = tempfile::tempdir().unwrap();
    let server = Server::start();
    configure(root.path(), &server, false);
    let first = server.url("/first");
    let second = server.url("/second");
    let third = server.url("/third");
    write(
        root.path(),
        "list.urls",
        &format!("{first} first\n{second} second\n{third} third\n"),
    );
    server.hold("GET", "/first");
    let running = Running::spawn(
        root.path(),
        root.path(),
        &["list.urls", "-o", "out", "--json"],
    );
    server.wait("GET", "/first", 1);
    running.interrupt();
    server.release("GET", "/first");
    let interrupted = running.finish();
    assert_eq!(
        interrupted.status.code(),
        Some(130),
        "{}",
        String::from_utf8_lossy(&interrupted.stderr)
    );
    let out = root.path().join("out");
    for (url, name) in [(&second, "second"), (&third, "third")] {
        assert_eq!(
            published_entry(&out, "url", &format!("{url} {name}"))["status"],
            "pending"
        );
        assert_eq!(server.count("GET", &format!("/{name}")), 0);
    }
    envelope(
        invoke(
            root.path(),
            &["list.urls", "-o", "out", "--resume", "--json"],
        ),
        0,
    );
    for name in ["first", "second", "third"] {
        assert_eq!(server.count("GET", &format!("/{name}")), 1);
    }
    no_versions(&out);
}

#[test]
fn overlapping_output_roots_exclude_shared_members_but_allow_independent_processes() {
    let root = tempfile::tempdir().unwrap();
    let server = Server::start();
    let mut cfg = configure(root.path(), &server, false);
    cfg["output"]["on_conflict"] = json!("overwrite");
    save_config(root.path(), &cfg);
    let active = server.url("/active");
    let conflict = server.url("/conflict");
    let free = server.url("/free");
    write(
        root.path(),
        "input/sub/list.urls",
        &format!("{active} same\n"),
    );
    server.hold("GET", "/active");
    let running = Running::spawn(root.path(), root.path(), &["input", "-o", "out", "--json"]);
    server.wait("GET", "/active", 1);
    write(root.path(), "other.urls", &format!("{conflict} same.llm\n"));
    let denied = invoke(root.path(), &["other.urls", "-o", "out/sub", "--json"]);
    assert!(!denied.status.success(), "overlapping member was admitted");
    assert_eq!(server.count("GET", "/conflict"), 0);
    let single = invoke(
        root.path(),
        &[&conflict, "-o", "out/sub/same.llm.md", "--json"],
    );
    assert!(
        !single.status.success(),
        "single-item CLI ignored the member lease"
    );
    assert_eq!(server.count("GET", "/conflict"), 0);
    envelope(
        invoke(root.path(), &[&free, "-o", "out/sub/free.md", "--json"]),
        0,
    );
    assert_eq!(
        server.count("GET", "/free"),
        1,
        "unrelated member was serialized behind active conversion"
    );
    assert!(root.path().join("out/sub/free.md").is_file());
    server.release("GET", "/active");
    envelope(running.finish(), 0);
    assert!(
        std::fs::read_to_string(root.path().join("out/sub/same.md"))
            .unwrap()
            .contains("MARKERXACTIVE")
    );
    assert!(!root.path().join("out/sub/same.llm.md").exists());
    // The operating-system lease is released without deleting its lock inode.
    envelope(
        invoke(root.path(), &["other.urls", "-o", "out/sub", "--json"]),
        0,
    );
    assert_eq!(server.count("GET", "/conflict"), 1);
}

#[test]
fn complete_filename_families_do_not_overlap_under_any_conflict_policy() {
    for policy in ["rename", "overwrite", "skip"] {
        let root = tempfile::tempdir().unwrap();
        let server = Server::start();
        let mut cfg = configure(root.path(), &server, true);
        cfg["output"]["on_conflict"] = json!(policy);
        cfg["batch"]["url_concurrency"] = json!(2);
        save_config(root.path(), &cfg);
        let a = server.url("/a");
        let b = server.url("/b");
        let c = server.url("/c");
        write(
            root.path(),
            "list.urls",
            &format!("{a} x\n{b} x.llm\n{c} name\n{c} name.md\n{c} name\n"),
        );
        let stdout = envelope(
            invoke(root.path(), &["list.urls", "-o", "out", "--json"]),
            0,
        );
        let out = root.path().join("out");
        let saved = snapshot(&out);
        let mut produced = BTreeSet::new();
        assert_eq!(saved["urls"].as_object().unwrap().len(), 4);
        for (url, name, marker) in [
            (&a, "x", "MARKERXA"),
            (&b, "x.llm", "MARKERXB"),
            (&c, "name", "MARKERXC"),
            (&c, "name.md", "MARKERXC"),
        ] {
            let entry = &saved["urls"][format!("{url} {name}")];
            assert_eq!(entry["status"], "completed");
            let output = PathBuf::from(entry["output"].as_str().unwrap());
            assert!(produced.insert(output.clone()));
            assert!(std::fs::read_to_string(output).unwrap().contains(marker));
        }
        assert_eq!(stdout["items"].as_array().unwrap().len(), 4);
        assert_eq!(
            markdown_files(&out).len(),
            8,
            "a base/enhanced member was overwritten or skipped"
        );
        assert!(
            std::fs::read_to_string(out.join("x.llm.md"))
                .unwrap()
                .contains("MARKERXA")
        );
        assert_eq!(server.count("POST", "MARKERXA"), 1);
        assert_eq!(server.count("POST", "MARKERXB"), 1);
        assert_eq!(server.count("POST", "MARKERXC"), 2);
        assert_eq!(server.count("GET", "/c"), 2);
    }
}

#[test]
fn retries_preserve_edited_replaced_and_unrelated_sibling_files_before_network() {
    for mutation in ["edited", "identical_replacement", "unrelated_sibling"] {
        let root = tempfile::tempdir().unwrap();
        let server = Server::start();
        configure(root.path(), &server, true);
        server.fail_model("MARKERXOWNED", true);
        let url = server.url("/owned");
        write(root.path(), "list.urls", &format!("{url} own\n"));
        envelope(
            invoke(root.path(), &["list.urls", "-o", "out", "--json"]),
            10,
        );
        let out = root.path().join("out");
        let base = out.join("own.md");
        let old_bytes = std::fs::read(&base).unwrap();
        match mutation {
            "edited" => {
                std::fs::write(&base, "An external editor owns these changed bytes.").unwrap()
            }
            "identical_replacement" => {
                let replacement = out.join("external.tmp");
                std::fs::write(&replacement, &old_bytes).unwrap();
                std::fs::rename(replacement, &base).unwrap();
            }
            "unrelated_sibling" => {
                std::fs::write(out.join("own.llm.md"), "Unrelated enhanced sibling.").unwrap()
            }
            _ => unreachable!(),
        }
        let before: BTreeMap<_, _> = markdown_files(&out)
            .into_iter()
            .map(|path| {
                let bytes = std::fs::read(&path).unwrap();
                (path, bytes)
            })
            .collect();
        server.fail_model("MARKERXOWNED", false);
        let failed = envelope(
            invoke(
                root.path(),
                &["list.urls", "-o", "out", "--resume", "--json"],
            ),
            10,
        );
        assert_eq!(failed["items"][0]["status"], "failed");
        assert_eq!(
            server.count("GET", "/owned"),
            1,
            "unsafe retry reached network: {mutation}"
        );
        assert_eq!(server.count("POST", "MARKERXOWNED"), 1);
        let after: BTreeMap<_, _> = markdown_files(&out)
            .into_iter()
            .map(|path| {
                let bytes = std::fs::read(&path).unwrap();
                (path, bytes)
            })
            .collect();
        assert_eq!(after, before, "retry changed external output: {mutation}");
    }
}

#[test]
fn unavailable_state_storage_prevents_fetch_and_preserves_the_obstruction() {
    let root = tempfile::tempdir().unwrap();
    let server = Server::start();
    configure(root.path(), &server, false);
    let url = server.url("/never");
    write(root.path(), "list.urls", &format!("{url} never\n"));
    write(
        root.path(),
        "out/.markitai/states",
        "Existing regular file must survive.",
    );
    let failed = invoke(root.path(), &["list.urls", "-o", "out", "--json"]);
    assert!(!failed.status.success());
    assert_eq!(server.count("GET", "/never"), 0);
    assert_eq!(
        std::fs::read_to_string(root.path().join("out/.markitai/states")).unwrap(),
        "Existing regular file must survive."
    );
    assert!(markdown_files(&root.path().join("out")).is_empty());
}

#[test]
fn resume_dry_run_and_empty_inputs_do_not_create_persistence_shells() {
    let root = tempfile::tempdir().unwrap();
    let server = Server::start();
    configure(root.path(), &server, false);
    let url = server.url("/preview");
    write(root.path(), "input/note.txt", "MARKERXNOTE preview only.");
    write(root.path(), "list.urls", &format!("{url} preview\n"));
    for source in ["input", "list.urls"] {
        let dry = invoke(
            root.path(),
            &[source, "-o", "preview", "--resume", "--dry-run"],
        );
        assert!(
            dry.status.success(),
            "{}",
            String::from_utf8_lossy(&dry.stderr)
        );
        assert!(!root.path().join("preview").exists());
    }
    std::fs::create_dir(root.path().join("empty")).unwrap();
    let empty = envelope(
        invoke(
            root.path(),
            &["empty", "-o", "empty-out", "--resume", "--json"],
        ),
        0,
    );
    assert_eq!(empty["items"], json!([]));
    assert!(!root.path().join("empty-out").exists());
    write(root.path(), "empty.urls", "# no entries\n");
    let empty_list = invoke(
        root.path(),
        &["empty.urls", "-o", "list-out", "--resume", "--json"],
    );
    assert_eq!(empty_list.status.code(), Some(1));
    assert!(!root.path().join("list-out").exists());
    assert_eq!(server.count("GET", "/preview"), 0);
}

#[test]
fn explicit_filename_rename_preserves_the_original_and_routes_enhanced_output_to_the_claim() {
    for model in [false, true] {
        let root = tempfile::tempdir().unwrap();
        let server = Server::start();
        let mut cfg = configure(root.path(), &server, model);
        cfg["llm"]["keep_base"] = json!(false);
        save_config(root.path(), &cfg);
        write(root.path(), "note.txt", "MARKERXFIRST explicit output.");
        let first = envelope(
            invoke(root.path(), &["note.txt", "-o", "out/exact.md", "--json"]),
            0,
        );
        let original = root.path().join("out/exact.md");
        let before = std::fs::read(&original).unwrap();
        assert_eq!(
            Path::new(first["items"][0]["output"].as_str().unwrap())
                .file_name()
                .unwrap(),
            "exact.md"
        );
        write(root.path(), "note.txt", "MARKERXSECOND changed input.");
        let second = envelope(
            invoke(root.path(), &["note.txt", "-o", "out/exact.md", "--json"]),
            0,
        );
        let selected = PathBuf::from(second["items"][0]["output"].as_str().unwrap());
        let selected = if selected.is_absolute() {
            selected
        } else {
            root.path().join(selected)
        };
        assert_eq!(selected.file_name().unwrap(), "exact.v2.md");
        assert!(
            std::fs::read_to_string(&selected)
                .unwrap()
                .contains("MARKERXSECOND")
        );
        assert_eq!(std::fs::read(original).unwrap(), before);
        assert_eq!(markdown_files(&root.path().join("out")).len(), 2);
        assert!(!root.path().join("out/exact.v2.llm.md").exists());
        if model {
            assert_eq!(server.count("POST", "MARKERXFIRST"), 1);
            assert_eq!(server.count("POST", "MARKERXSECOND"), 1);
        }
    }
}

#[test]
fn completed_reservations_survive_missing_outputs_and_actual_filesystem_case_aliases() {
    for remove_output in [false, true] {
        let root = tempfile::tempdir().unwrap();
        let server = Server::start();
        let mut cfg = configure(root.path(), &server, false);
        cfg["output"]["on_conflict"] = json!("overwrite");
        save_config(root.path(), &cfg);
        let old = server.url("/old");
        let new = server.url("/new");
        write(root.path(), "list.urls", &format!("{old} Case\n"));
        envelope(
            invoke(root.path(), &["list.urls", "-o", "out", "--json"]),
            0,
        );
        let out = root.path().join("out");
        let original = out.join("Case.md");
        let original_bytes = std::fs::read(&original).unwrap();
        // Probe this exact output directory. A case-sensitive volume still
        // exercises the missing completed reservation using the exact spelling.
        let probe = out.join("MarkitaiCaseProbe");
        std::fs::write(&probe, "probe").unwrap();
        let candidate = if out.join("markitaicaseprobe").exists() {
            "case"
        } else {
            "Case"
        };
        std::fs::remove_file(probe).unwrap();
        if remove_output {
            std::fs::remove_file(&original).unwrap();
        }
        write(
            root.path(),
            "list.urls",
            &format!("{old} Case\n{new} {candidate}\n"),
        );
        envelope(
            invoke(
                root.path(),
                &["list.urls", "-o", "out", "--resume", "--json"],
            ),
            0,
        );
        let saved = snapshot(&out);
        let produced = PathBuf::from(
            saved["urls"][format!("{new} {candidate}")]["output"]
                .as_str()
                .unwrap(),
        );
        assert_eq!(
            produced.file_name().unwrap().to_str().unwrap(),
            format!("{candidate}.v2.md")
        );
        assert!(
            std::fs::read_to_string(&produced)
                .unwrap()
                .contains("MARKERXNEW")
        );
        assert_eq!(saved["urls"][format!("{old} Case")]["status"], "completed");
        if remove_output {
            assert!(
                !original.exists(),
                "completed missing output was stolen or recreated"
            );
        } else {
            assert_eq!(std::fs::read(original).unwrap(), original_bytes);
        }
        assert_eq!(server.count("GET", "/old"), 1);
        assert_eq!(server.count("GET", "/new"), 1);
    }
}

#[test]
fn a_completed_literal_llm_suffix_reserves_its_actual_family_not_a_guessed_enhancement() {
    let root = tempfile::tempdir().unwrap();
    let server = Server::start();
    let mut cfg = configure(root.path(), &server, false);
    cfg["output"]["on_conflict"] = json!("overwrite");
    save_config(root.path(), &cfg);
    let old = server.url("/old");
    let new = server.url("/new");
    write(root.path(), "list.urls", &format!("{old} x.llm\n"));
    envelope(
        invoke(root.path(), &["list.urls", "-o", "out", "--json"]),
        0,
    );
    let out = root.path().join("out");
    let original = std::fs::read(out.join("x.llm.md")).unwrap();
    write(
        root.path(),
        "list.urls",
        &format!("{old} x.llm\n{new} x.llm.llm\n"),
    );
    envelope(
        invoke(
            root.path(),
            &["list.urls", "-o", "out", "--resume", "--json"],
        ),
        0,
    );
    let saved = snapshot(&out);
    let produced = PathBuf::from(
        saved["urls"][format!("{new} x.llm.llm")]["output"]
            .as_str()
            .unwrap(),
    );
    assert_eq!(produced.file_name().unwrap(), "x.llm.llm.v2.md");
    assert!(
        std::fs::read_to_string(produced)
            .unwrap()
            .contains("MARKERXNEW")
    );
    assert!(!out.join("x.llm.llm.md").exists());
    assert_eq!(std::fs::read(out.join("x.llm.md")).unwrap(), original);
    assert_eq!(server.count("GET", "/old"), 1);
}

#[test]
fn a_failed_bare_url_adopts_a_named_key_and_reuses_its_owned_base() {
    let root = tempfile::tempdir().unwrap();
    let server = Server::start();
    let mut cfg = configure(root.path(), &server, true);
    let done = server.url("/adoption-done");
    let retry = server.url("/adoption-retry");
    server.fail_model("MARKERXADOPTIONXRETRY", true);
    write(root.path(), "list.urls", &format!("{done} done\n{retry}\n"));
    envelope(
        invoke(root.path(), &["list.urls", "-o", "out", "--json"]),
        10,
    );
    let out = root.path().join("out");
    let before = snapshot(&out);
    let prior = &before["urls"][&retry];
    assert_eq!(prior["status"], "failed");
    let target = PathBuf::from(prior["target"].as_str().unwrap());
    assert!(
        std::fs::read_to_string(&target)
            .unwrap()
            .contains("MARKERXADOPTIONXRETRY")
    );
    let done_base = std::fs::read(out.join("done.md")).unwrap();
    let done_enhanced = std::fs::read(out.join("done.llm.md")).unwrap();
    let generation = before["_markitai"]["generation"].clone();
    assert!(generation.is_string());

    // The adopted target remains authoritative even though the newly named entry
    // asks for a different stem and ordinary occupied outputs would be skipped.
    write(
        root.path(),
        "list.urls",
        &format!("{done} done\n{retry} renamed.md\n"),
    );
    cfg["output"]["on_conflict"] = json!("skip");
    cfg["output"]["report"] = json!(true);
    save_config(root.path(), &cfg);
    server.fail_model("MARKERXADOPTIONXRETRY", false);
    let resumed = envelope(
        invoke(
            root.path(),
            &["list.urls", "-o", "out", "--resume", "--json"],
        ),
        0,
    );
    assert_eq!(resumed["items"].as_array().unwrap().len(), 1);
    assert_eq!(resumed["items"][0]["status"], "completed");
    let key = format!("{retry} renamed.md");
    let after = snapshot(&out);
    assert!(after["urls"].get(&retry).is_none());
    assert_eq!(after["urls"].as_object().unwrap().len(), 2);
    assert_eq!(after["_markitai"]["generation"], generation);
    assert_eq!(after["urls"][&key]["status"], "completed");
    let stem = target
        .file_name()
        .unwrap()
        .to_str()
        .unwrap()
        .strip_suffix(".md")
        .unwrap();
    let enhanced = target.with_file_name(format!("{stem}.llm.md"));
    assert_eq!(
        Path::new(after["urls"][&key]["output"].as_str().unwrap()),
        enhanced
    );
    assert!(
        std::fs::read_to_string(&enhanced)
            .unwrap()
            .contains("MARKERXADOPTIONXRETRY")
    );
    assert_eq!(std::fs::read(out.join("done.md")).unwrap(), done_base);
    assert_eq!(
        std::fs::read(out.join("done.llm.md")).unwrap(),
        done_enhanced
    );
    assert!(!out.join("renamed.md").exists() && !out.join("renamed.llm.md").exists());
    assert_eq!(server.count("GET", "/adoption-done"), 1);
    assert_eq!(server.count("POST", "MARKERXADOPTIONXDONE"), 1);
    assert_eq!(server.count("GET", "/adoption-retry"), 2);
    assert_eq!(server.count("POST", "MARKERXADOPTIONXRETRY"), 2);
    no_versions(&out);

    let reports: Vec<_> = std::fs::read_dir(out.join(".markitai/reports"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect();
    assert_eq!(reports.len(), 1);
    let report: Value = serde_json::from_slice(&std::fs::read(&reports[0]).unwrap()).unwrap();
    assert_eq!(report["summary"]["completed_urls"], 2);
    let report_urls = &report["url_sources"]["unknown.urls"]["urls"];
    assert_eq!(report_urls[format!("{done} done")]["status"], "completed");
    assert_eq!(report_urls[&key]["status"], "completed");
    assert!(report_urls.get(&retry).is_none());

    envelope(
        invoke(
            root.path(),
            &["list.urls", "-o", "out", "--resume", "--json"],
        ),
        0,
    );
    assert_eq!(server.count("GET", "/adoption-done"), 1);
    assert_eq!(server.count("POST", "MARKERXADOPTIONXDONE"), 1);
    assert_eq!(server.count("GET", "/adoption-retry"), 2);
    assert_eq!(server.count("POST", "MARKERXADOPTIONXRETRY"), 2);
    assert_eq!(
        std::fs::read(out.join("done.llm.md")).unwrap(),
        done_enhanced
    );
    no_versions(&out);
}

#[test]
fn legacy_failed_targets_without_receipts_use_the_selected_ordinary_conflict_policy() {
    for policy in ["rename", "skip", "overwrite"] {
        let root = tempfile::tempdir().unwrap();
        let server = Server::start();
        let mut cfg = configure(root.path(), &server, true);
        let url = server.url("/legacy-target");
        let key = format!("{url} keep");
        write(root.path(), "list.urls", &format!("{url} keep\n"));
        server.fail_model("MARKERXLEGACYXTARGET", true);
        envelope(
            invoke(root.path(), &["list.urls", "-o", "out", "--json"]),
            10,
        );
        let out = root.path().join("out");
        let state_file = state_path(&out);
        let native = snapshot(&out);
        let base = out.join("keep.md");

        // Let the public CLI select the compatibility filename, then install an
        // authored minimal legacy checkpoint. No native fence or receipt remains.
        let mut urls = serde_json::Map::new();
        urls.insert(
            key.clone(),
            json!({
                "status":"failed", "url":url, "source_file":root.path().join("list.urls"),
                "target":base, "output":null, "error":"A previous legacy conversion failed",
            }),
        );
        let legacy = json!({
            "version":"1.0", "options":native["options"], "documents":{}, "urls":urls,
        });
        let legacy_bytes = serde_json::to_vec_pretty(&legacy).unwrap();
        std::fs::write(&state_file, &legacy_bytes).unwrap();
        assert!(journal(&out).is_empty());
        let legacy_journal = format!(
            "{{malformed old line\n{}\n",
            json!({
                "type":"url", "key":key, "data":{"status":"failed","error":"legacy journal failure"},
            })
        );
        let journal_file = state_file.with_extension("jsonl");
        std::fs::write(&journal_file, &legacy_journal).unwrap();
        let receipts = out.join(".markitai/ownership/records");
        assert!(receipts.is_dir());
        std::fs::remove_dir_all(&receipts).unwrap();
        let external = b"Legacy state does not prove that this existing document belongs to it.\n";
        std::fs::write(&base, external).unwrap();
        cfg["output"]["on_conflict"] = json!(policy);
        save_config(root.path(), &cfg);
        server.fail_model("MARKERXLEGACYXTARGET", false);

        let result = envelope(
            invoke(
                root.path(),
                &["list.urls", "-o", "out", "--resume", "--json"],
            ),
            0,
        );
        let saved = snapshot(&out);
        assert!(saved["_markitai"]["generation"].is_string());
        assert_eq!(saved["urls"][&key]["status"], "completed");
        let backups = || {
            std::fs::read_dir(out.join(".markitai/states"))
                .unwrap()
                .map(|entry| entry.unwrap().path())
                .filter(|path| {
                    path.is_dir()
                        && path
                            .file_name()
                            .unwrap()
                            .to_string_lossy()
                            .contains(".legacy.")
                })
                .collect::<Vec<_>>()
        };
        let preserved = backups();
        assert_eq!(preserved.len(), 1);
        assert_eq!(
            std::fs::read(preserved[0].join(state_file.file_name().unwrap())).unwrap(),
            legacy_bytes
        );
        assert_eq!(
            std::fs::read(preserved[0].join(journal_file.file_name().unwrap())).unwrap(),
            legacy_journal.as_bytes()
        );
        let manifest: Value =
            serde_json::from_slice(&std::fs::read(preserved[0].join("manifest.json")).unwrap())
                .unwrap();
        assert_eq!(manifest["kind"], "legacy-recovery-pair");
        assert_eq!(manifest["generation"], saved["_markitai"]["generation"]);
        match policy {
            "rename" => {
                assert_eq!(result["items"][0]["status"], "completed");
                assert_eq!(std::fs::read(&base).unwrap(), external);
                assert!(
                    std::fs::read_to_string(out.join("keep.v2.md"))
                        .unwrap()
                        .contains("MARKERXLEGACYXTARGET")
                );
                assert!(
                    std::fs::read_to_string(out.join("keep.v2.llm.md"))
                        .unwrap()
                        .contains("MARKERXLEGACYXTARGET")
                );
                assert_eq!(
                    Path::new(saved["urls"][&key]["output"].as_str().unwrap())
                        .file_name()
                        .unwrap(),
                    "keep.v2.llm.md"
                );
                assert_eq!(markdown_files(&out).len(), 3);
            }
            "skip" => {
                assert_eq!(result["items"][0]["status"], "skipped");
                assert_eq!(std::fs::read(&base).unwrap(), external);
                assert_eq!(markdown_files(&out), vec![base.clone()]);
                assert!(
                    !receipts.exists(),
                    "skipping a foreign file fabricated ownership evidence"
                );
            }
            "overwrite" => {
                assert_eq!(result["items"][0]["status"], "completed");
                assert_ne!(std::fs::read(&base).unwrap(), external);
                assert!(
                    std::fs::read_to_string(&base)
                        .unwrap()
                        .contains("MARKERXLEGACYXTARGET")
                );
                assert!(
                    std::fs::read_to_string(out.join("keep.llm.md"))
                        .unwrap()
                        .contains("MARKERXLEGACYXTARGET")
                );
                assert_eq!(
                    Path::new(saved["urls"][&key]["output"].as_str().unwrap())
                        .file_name()
                        .unwrap(),
                    "keep.llm.md"
                );
                assert_eq!(markdown_files(&out).len(), 2);
            }
            _ => unreachable!(),
        }
        let requests = if policy == "skip" { 1 } else { 2 };
        assert_eq!(
            server.count("GET", "/legacy-target"),
            requests,
            "policy={policy}"
        );
        assert_eq!(
            server.count("POST", "MARKERXLEGACYXTARGET"),
            requests,
            "policy={policy}"
        );
        no_reports(&out);
        envelope(
            invoke(
                root.path(),
                &["list.urls", "-o", "out", "--resume", "--json"],
            ),
            0,
        );
        assert_eq!(server.count("GET", "/legacy-target"), requests);
        assert_eq!(server.count("POST", "MARKERXLEGACYXTARGET"), requests);
        assert_eq!(
            backups(),
            preserved,
            "native resume created another legacy backup"
        );
    }
}

#[test]
fn moving_a_directory_url_list_preserves_pending_and_failed_output_provenance() {
    for previous in ["pending", "failed"] {
        let root = tempfile::tempdir().unwrap();
        let server = Server::start();
        configure(root.path(), &server, true);
        let url = server.url("/moved-item");
        let key = format!("{url} foo");
        // Directory discovery sorts URL displays, so the held request must sort first.
        let gate = server.url("/a-provenance-gate");
        let list = if previous == "pending" {
            format!("{gate} gate\n{url} foo\n")
        } else {
            format!("{url} foo\n")
        };
        write(root.path(), "input/a/list.urls", &list);
        let out = root.path().join("out");
        if previous == "pending" {
            server.hold("GET", "/a-provenance-gate");
            let running =
                Running::spawn(root.path(), root.path(), &["input", "-o", "out", "--json"]);
            server.wait("GET", "/a-provenance-gate", 1);
            assert_eq!(published_entry(&out, "url", &key)["status"], "in_progress");
            assert_eq!(server.count("GET", "/moved-item"), 0);
            assert!(!running.kill().status.success());
            server.release("GET", "/a-provenance-gate");
        } else {
            server.fail_model("MARKERXMOVEDXITEM", true);
            envelope(invoke(root.path(), &["input", "-o", "out", "--json"]), 10);
            assert_eq!(published_entry(&out, "url", &key)["status"], "failed");
            assert!(
                std::fs::read_to_string(out.join("a/foo.md"))
                    .unwrap()
                    .contains("MARKERXMOVEDXITEM")
            );
            server.fail_model("MARKERXMOVEDXITEM", false);
        }
        let original_source = published_entry(&out, "url", &key)["source_file"].clone();
        assert!(
            original_source
                .as_str()
                .unwrap()
                .ends_with("input/a/list.urls")
        );
        std::fs::create_dir_all(root.path().join("input/b")).unwrap();
        std::fs::rename(
            root.path().join("input/a/list.urls"),
            root.path().join("input/b/list.urls"),
        )
        .unwrap();
        envelope(
            invoke(root.path(), &["input", "-o", "out", "--resume", "--json"]),
            0,
        );
        let saved = snapshot(&out);
        assert_eq!(saved["urls"][&key]["status"], "completed");
        assert_eq!(saved["urls"][&key]["source_file"], original_source);
        let produced = PathBuf::from(saved["urls"][&key]["output"].as_str().unwrap());
        assert_eq!(produced.parent().unwrap().file_name().unwrap(), "a");
        assert_eq!(produced.file_name().unwrap(), "foo.llm.md");
        assert!(
            std::fs::read_to_string(&produced)
                .unwrap()
                .contains("MARKERXMOVEDXITEM")
        );
        assert!(!out.join("b/foo.md").exists() && !out.join("b/foo.llm.md").exists());
        let requests = if previous == "pending" { 1 } else { 2 };
        assert_eq!(server.count("GET", "/moved-item"), requests);
        assert_eq!(server.count("POST", "MARKERXMOVEDXITEM"), requests);
        no_versions(&out.join("a"));
        no_reports(&out);
    }
}

#[test]
fn a_completed_bare_url_can_adopt_a_named_key_without_rewriting_external_edits() {
    let root = tempfile::tempdir().unwrap();
    let server = Server::start();
    configure(root.path(), &server, true);
    let url = server.url("/completed-adoption");
    write(root.path(), "list.urls", &format!("{url}\n"));
    envelope(
        invoke(root.path(), &["list.urls", "-o", "out", "--json"]),
        0,
    );
    let out = root.path().join("out");
    let saved = snapshot(&out);
    let produced = PathBuf::from(saved["urls"][&url]["output"].as_str().unwrap());
    let edited = b"This completed output was subsequently edited by its user.\n";
    std::fs::write(&produced, edited).unwrap();
    let before: BTreeMap<_, _> = markdown_files(&out)
        .into_iter()
        .map(|path| {
            let bytes = std::fs::read(&path).unwrap();
            (path, bytes)
        })
        .collect();
    write(
        root.path(),
        "list.urls",
        &format!("{url} renamed-completed\n"),
    );
    let result = envelope(
        invoke(
            root.path(),
            &["list.urls", "-o", "out", "--resume", "--json"],
        ),
        0,
    );
    assert_eq!(result["items"], json!([]));
    let after = snapshot(&out);
    assert!(after["urls"].get(&url).is_none());
    assert_eq!(after["urls"].as_object().unwrap().len(), 1);
    let key = format!("{url} renamed-completed");
    assert_eq!(after["urls"][&key]["status"], "completed");
    assert_eq!(
        Path::new(after["urls"][&key]["output"].as_str().unwrap()),
        produced
    );
    assert_eq!(std::fs::read(&produced).unwrap(), edited);
    let actual: BTreeMap<_, _> = markdown_files(&out)
        .into_iter()
        .map(|path| {
            let bytes = std::fs::read(&path).unwrap();
            (path, bytes)
        })
        .collect();
    assert_eq!(actual, before);
    assert_eq!(server.count("GET", "/completed-adoption"), 1);
    assert_eq!(server.count("POST", "MARKERXCOMPLETEDXADOPTION"), 1);
    no_reports(&out);
}

#[test]
fn concurrent_distinct_urls_skipping_the_same_existing_member_make_no_requests() {
    let root = tempfile::tempdir().unwrap();
    let server = Server::start();
    let mut cfg = configure(root.path(), &server, false);
    cfg["output"]["on_conflict"] = json!("skip");
    cfg["batch"]["url_concurrency"] = json!(2);
    save_config(root.path(), &cfg);
    let first = server.url("/skip-one");
    let second = server.url("/skip-two");
    write(
        root.path(),
        "list.urls",
        &format!("{first} shared\n{second} shared\n"),
    );
    let existing = "An unrelated existing document must survive both skips.\n";
    write(root.path(), "out/shared.md", existing);
    let result = envelope(
        invoke(root.path(), &["list.urls", "-o", "out", "--json"]),
        0,
    );
    let items = result["items"].as_array().unwrap();
    assert_eq!(items.len(), 2);
    assert!(items.iter().all(|item| item["status"] == "skipped"));
    assert_eq!(result["totals"]["skipped"], 2);
    assert_eq!(server.count("GET", "/skip-one"), 0);
    assert_eq!(server.count("GET", "/skip-two"), 0);
    let out = root.path().join("out");
    assert_eq!(markdown_files(&out), vec![out.join("shared.md")]);
    assert_eq!(
        std::fs::read_to_string(out.join("shared.md")).unwrap(),
        existing
    );
    let saved = snapshot(&out);
    for url in [first, second] {
        assert_eq!(
            saved["urls"][format!("{url} shared")]["status"],
            "completed"
        );
    }
    no_reports(&out);
}

#[test]
fn largest_valid_flush_interval_does_not_panic_or_lose_final_state() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let server = Server::start();
    let mut cfg = configure(root, &server, false);
    cfg["batch"]["state_flush_interval_seconds"] = json!(u64::MAX);
    save_config(root, &cfg);
    write(root, "input/note.txt", "Completed with a large interval.");
    let value = envelope(invoke(root, &["input", "-o", "out/", "--json"]), 0);
    assert_eq!(value["totals"]["completed"], 1);
    assert_eq!(
        snapshot(&root.join("out"))["documents"]["note.txt"]["status"],
        "completed"
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

#[test]
fn legacy_directory_takeover_retains_completed_work_raw_pair_and_quiet_contract() {
    for quiet in [false, true] {
        let root = tempfile::tempdir().unwrap();
        let server = Server::start();
        configure(root.path(), &server, true);
        write(root.path(), "input/a.txt", "MARKERXLEGACYXDONE\n");
        write(root.path(), "input/b.txt", "MARKERXLEGACYXRETRY\n");
        server.fail_model("MARKERXLEGACYXRETRY", true);
        envelope(invoke(root.path(), &["input", "-o", "out", "--json"]), 10);
        let out = root.path().join("out");
        let state_file = state_path(&out);
        let mut legacy = snapshot(&out);
        legacy.as_object_mut().unwrap().remove("_markitai");
        legacy["future_unknown"] = json!({"text":"保留原文", "value":17});
        let done = PathBuf::from(legacy["documents"]["a.txt"]["output"].as_str().unwrap());
        let done_bytes = std::fs::read(&done).unwrap();
        let original = format!(" {}\n\n", serde_json::to_string_pretty(&legacy).unwrap());
        let journal_text = format!(
            "{}\n",
            json!({
                "type":"file", "key":"b.txt", "data":{"status":"failed","error":"previous attempt"}
            })
        );
        let journal_file = state_file.with_extension("jsonl");
        std::fs::write(&state_file, &original).unwrap();
        std::fs::write(&journal_file, &journal_text).unwrap();
        server.fail_model("MARKERXLEGACYXRETRY", false);
        let mut args = vec!["input", "-o", "out", "--resume", "--json"];
        if quiet {
            args.push("--quiet");
        }
        let output = invoke(root.path(), &args);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert_eq!(
            stderr.contains("Preserved original legacy recovery files"),
            !quiet,
            "{stderr}"
        );
        if quiet {
            assert!(stderr.is_empty(), "{stderr}");
        }
        let result = envelope(output, 0);
        assert_eq!(result["totals"]["failed"], 0);
        assert_eq!(std::fs::read(&done).unwrap(), done_bytes);
        assert_eq!(server.count("POST", "MARKERXLEGACYXDONE"), 1);
        assert_eq!(server.count("POST", "MARKERXLEGACYXRETRY"), 2);
        let backups: Vec<_> = std::fs::read_dir(out.join(".markitai/states"))
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| {
                path.is_dir()
                    && path
                        .file_name()
                        .unwrap()
                        .to_string_lossy()
                        .contains(".legacy.")
            })
            .collect();
        assert_eq!(backups.len(), 1);
        assert_eq!(
            std::fs::read(backups[0].join(state_file.file_name().unwrap())).unwrap(),
            original.as_bytes()
        );
        assert_eq!(
            std::fs::read(backups[0].join(journal_file.file_name().unwrap())).unwrap(),
            journal_text.as_bytes()
        );
        // The prior failed target still has no imported write authority.
        assert!(out.join("b.txt.md").exists());
        assert!(out.join("b.txt.v2.md").exists());
        assert!(out.join("b.txt.v2.llm.md").exists());
        envelope(invoke(root.path(), &args), 0);
        assert_eq!(server.count("POST", "MARKERXLEGACYXDONE"), 1);
        assert_eq!(server.count("POST", "MARKERXLEGACYXRETRY"), 2);
    }
}
