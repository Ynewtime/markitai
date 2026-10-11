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

const WAIT: Duration = Duration::from_secs(25);

// Only real subprocesses reach the history writer. HTTP response gates control
// completion without a production testing flag or access to the user's state.
fn configure(root: &Path) -> Value {
    let cfg = json!({
        "llm":{"enabled":false}, "ocr":{"enabled":false},
        "screenshot":{"enabled":false},
        "image":{"alt_enabled":false,"desc_enabled":false,"compress":false,
                 "filter":{"min_width":1,"min_height":1,"min_area":1}},
        "cache":{"enabled":false}, "history":{"record":false},
        "prompts":{"dir":root.join("private-prompts")}, "log":{"dir":null},
        "fetch":{"strategy":"static","remote_consent":"never"},
        "batch":{"concurrency":2,"url_concurrency":2,"scan_max_depth":8,
                 "state_flush_interval_seconds":3600},
        "output":{"on_conflict":"rename","report":false}
    });
    save(root, &cfg);
    cfg
}
fn save(root: &Path, cfg: &Value) {
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
        assert!(Instant::now() < deadline, "timed out: {description}");
        std::thread::sleep(Duration::from_millis(5));
    }
}

struct Running {
    child: Option<Child>,
    stdout: Option<JoinHandle<Vec<u8>>>,
    stderr: Option<JoinHandle<()>>,
    captured: Arc<Mutex<Vec<u8>>>,
}
impl Running {
    fn spawn(root: &Path, args: &[&str], env: &[(&str, &str)]) -> Self {
        let mut command = Command::new(env!("CARGO_BIN_EXE_markitai"));
        command.env_clear();
        for name in ["PATH", "SYSTEMROOT", "TMPDIR", "TEMP", "TMP"] {
            if let Some(value) = std::env::var_os(name) {
                command.env(name, value);
            }
        }
        let mut child = command
            .current_dir(root)
            .env("MARKITAI_HOME", root.join("home"))
            .env("NO_PROXY", "127.0.0.1,localhost")
            .envs(env.iter().copied())
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
        let captured = Arc::new(Mutex::new(Vec::new()));
        let shared = captured.clone();
        Self {
            child: Some(child),
            captured,
            stdout: Some(std::thread::spawn(move || {
                let mut bytes = Vec::new();
                stdout.read_to_end(&mut bytes).unwrap();
                bytes
            })),
            stderr: Some(std::thread::spawn(move || {
                let mut reader = BufReader::new(stderr);
                let mut line = Vec::new();
                while reader.read_until(b'\n', &mut line).unwrap() != 0 {
                    shared.lock().unwrap().extend_from_slice(&line);
                    line.clear();
                }
            })),
        }
    }
    #[cfg(unix)]
    fn interrupt(&self) {
        assert!(
            Command::new("/bin/kill")
                .args(["-INT", &self.child.as_ref().unwrap().id().to_string()])
                .status()
                .unwrap()
                .success()
        );
        wait_until("public interruption acknowledgement", || {
            String::from_utf8_lossy(&self.captured.lock().unwrap())
                .lines()
                .any(|line| {
                    line == "  ! Interrupted: stopping new work and waiting for active conversions."
                })
        });
    }
    fn finish(mut self) -> Output {
        let deadline = Instant::now() + WAIT;
        let status = loop {
            if let Some(status) = self.child.as_mut().unwrap().try_wait().unwrap() {
                break status;
            }
            assert!(
                Instant::now() < deadline,
                "CLI did not exit: {}",
                String::from_utf8_lossy(&self.captured.lock().unwrap())
            );
            std::thread::sleep(Duration::from_millis(5));
        };
        let _ = self.child.take();
        let stdout = self.stdout.take().unwrap().join().unwrap();
        self.stderr.take().unwrap().join().unwrap();
        let stderr = self.captured.lock().unwrap().clone();
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
    Running::spawn(root, args, &[]).finish()
}
fn status(output: &Output, code: i32) {
    assert_eq!(
        output.status.code(),
        Some(code),
        "stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
fn envelope(output: &Output, code: i32) -> Value {
    status(output, code);
    let value: Value = serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "not exactly one JSON envelope: {error}; {}",
            String::from_utf8_lossy(&output.stdout)
        )
    });
    assert_eq!(value["version"], "1.0");
    assert!(value["items"].is_array());
    assert!(!String::from_utf8_lossy(&output.stderr).contains("Recorded in history"));
    value
}

#[derive(Default)]
struct HttpState {
    counts: HashMap<String, usize>,
    held: HashSet<String>,
    failed: HashSet<String>,
    expired: Vec<String>,
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
                        let state = shared.clone();
                        let stop = stopping.clone();
                        workers.push(std::thread::spawn(move || serve(stream, state, stop)));
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(2))
                    }
                    Err(error) => panic!("loopback accept: {error}"),
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
    fn hold(&self, path: &str) {
        self.state.0.lock().unwrap().held.insert(path.into());
    }
    fn release(&self, path: &str) {
        self.state.0.lock().unwrap().held.remove(path);
        self.state.1.notify_all();
    }
    fn fail(&self, path: &str, fail: bool) {
        let mut state = self.state.0.lock().unwrap();
        if fail {
            state.failed.insert(path.into());
        } else {
            state.failed.remove(path);
        }
    }
    fn count(&self, path: &str) -> usize {
        *self.state.0.lock().unwrap().counts.get(path).unwrap_or(&0)
    }
    fn wait(&self, path: &str, count: usize) {
        wait_until(path, || self.count(path) >= count);
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        let expired = {
            let mut state = self
                .state
                .0
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            state.held.clear();
            std::mem::take(&mut state.expired)
        };
        self.state.1.notify_all();
        let result = self.thread.take().map(JoinHandle::join);
        if !std::thread::panicking() {
            if let Some(result) = result {
                result.expect("loopback worker failed");
            }
            assert!(expired.is_empty(), "HTTP gates expired: {expired:?}");
        }
    }
}
fn serve(mut stream: TcpStream, shared: Arc<(Mutex<HttpState>, Condvar)>, stop: Arc<AtomicBool>) {
    stream.set_nonblocking(false).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    stream
        .set_write_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    let mut bytes = Vec::new();
    let request_deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let remaining = request_deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return;
        }
        stream.set_read_timeout(Some(remaining)).unwrap();
        let mut chunk = [0; 4096];
        match stream.read(&mut chunk) {
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Ok(0) | Err(_) => return,
            Ok(n) => bytes.extend_from_slice(&chunk[..n]),
        }
        assert!(bytes.len() < 1024 * 1024);
        if let Some(split) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
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
                break;
            }
        }
    }
    let request = String::from_utf8(bytes).unwrap();
    let mut words = request.lines().next().unwrap().split_whitespace();
    let method = words.next().unwrap();
    let path = words.next().unwrap();
    let mut state = shared.0.lock().unwrap_or_else(|error| error.into_inner());
    *state.counts.entry(path.into()).or_default() += 1;
    let deadline = Instant::now() + WAIT;
    while state.held.contains(path) && !stop.load(Ordering::SeqCst) {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            // Close the response without poisoning shared state. The owning
            // test observes the failure; Drop reports expiry only if no other
            // assertion is already unwinding.
            state.expired.push(path.to_owned());
            return;
        }
        state = shared
            .1
            .wait_timeout(state, remaining)
            .unwrap_or_else(|error| error.into_inner())
            .0;
    }
    if stop.load(Ordering::SeqCst) {
        return;
    }
    let failed = state.failed.contains(path);
    drop(state);
    let (status, mime, body) = if failed {
        (
            "404 Not Found",
            "text/plain",
            "Deliberate local failure".into(),
        )
    } else if method == "POST" {
        ("200 OK", "application/json", json!({"choices":[{"message":{"content":model_content(&serde_json::from_str(request.split_once("\r\n\r\n").unwrap().1).unwrap(),"# Enhanced history\n\nLocal model response.")},"finish_reason":"stop"}],"usage":{"prompt_tokens":9,"completion_tokens":5}}).to_string())
    } else {
        (
            "200 OK",
            "text/html",
            format!(
                "<html><title>History fixture</title><article><h1>History fixture</h1><p>Local content for {path} 世界.</p></article></html>"
            ),
        )
    };
    let _ = write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: {mime}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
}

fn jobs(root: &Path) -> Vec<PathBuf> {
    let dir = root.join("home/serve/jobs");
    if !dir.exists() {
        return Vec::new();
    }
    let mut paths: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|path| {
            if path.file_name().unwrap() == ".publish.lock" {
                assert!(path.is_file() && !path.is_symlink());
                false
            } else {
                true
            }
        })
        .collect();
    paths.sort();
    for path in &paths {
        let name = path.file_name().unwrap().to_str().unwrap();
        assert!(
            path.is_dir()
                && name.len() == 12
                && name
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
            "unexpected history residue: {}",
            path.display()
        );
    }
    paths
}
fn files(root: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    let mut result = BTreeMap::new();
    fn visit(base: &Path, dir: &Path, result: &mut BTreeMap<PathBuf, Vec<u8>>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            assert!(
                !path.is_symlink(),
                "history must contain independent regular copies"
            );
            if path.is_dir() {
                visit(base, &path, result);
            } else {
                result.insert(
                    path.strip_prefix(base).unwrap().into(),
                    std::fs::read(&path).unwrap(),
                );
            }
        }
    }
    visit(root, root, &mut result);
    result
}
fn keys(value: &Value, expected: &[&str]) {
    assert_eq!(
        value
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>(),
        expected.iter().copied().collect::<BTreeSet<_>>()
    );
}
fn stamp(value: &Value) -> chrono::DateTime<chrono::FixedOffset> {
    let text = value.as_str().unwrap();
    assert_eq!(
        text.split_once('.')
            .unwrap()
            .1
            .chars()
            .take_while(char::is_ascii_digit)
            .count(),
        3
    );
    chrono::DateTime::parse_from_rfc3339(text).unwrap()
}
fn read_job(path: &Path) -> Value {
    let meta: Value =
        serde_json::from_slice(&std::fs::read(path.join("meta.json")).unwrap()).unwrap();
    keys(
        &meta,
        &[
            "job_id",
            "created_at",
            "finished_at",
            "status",
            "options",
            "dir_size_bytes",
            "items",
        ],
    );
    assert_eq!(
        meta["job_id"].as_str().unwrap(),
        path.file_name().unwrap().to_str().unwrap()
    );
    assert_eq!(meta["status"], "done");
    assert!(stamp(&meta["created_at"]) <= stamp(&meta["finished_at"]));
    keys(&meta["options"], &["preset", "llm", "ocr", "origin"]);
    assert_eq!(meta["options"]["origin"], "cli");
    assert!(meta["options"]["preset"].is_null() || meta["options"]["preset"].is_string());
    assert!(meta["options"]["llm"].is_boolean() && meta["options"]["ocr"].is_boolean());
    let copied = files(&path.join("out"));
    assert_eq!(
        meta["dir_size_bytes"].as_u64(),
        Some(copied.values().map(|v| v.len() as u64).sum())
    );
    for (index, item) in meta["items"].as_array().unwrap().iter().enumerate() {
        let mut item_keys = vec![
            "item_id",
            "name",
            "kind",
            "status",
            "error",
            "output",
            "output_name",
            "duration_ms",
            "finished_at",
            "cost_usd",
            "llm_enhanced",
            "operation",
            "skipped",
            "skip_reason",
            "retryable",
            "warnings",
        ];
        if let Some(diagnostics) = item.get("diagnostics") {
            item_keys.push("diagnostics");
            keys(diagnostics, &["last_attempt"]);
            let attempt = &diagnostics["last_attempt"];
            keys(attempt, &["operation", "status", "error", "usage"]);
            assert_eq!(attempt["operation"], "convert");
            assert_eq!(attempt["status"], item["status"]);
            assert_eq!(attempt["error"], item["error"]);
            let usage = &attempt["usage"];
            keys(
                usage,
                &[
                    "requests",
                    "input_tokens",
                    "output_tokens",
                    "cost_usd",
                    "by_model",
                ],
            );
            assert!(
                usage["requests"].as_u64().unwrap() > 0
                    || usage["input_tokens"].as_u64().unwrap() > 0
                    || usage["output_tokens"].as_u64().unwrap() > 0
                    || !usage["by_model"].as_object().unwrap().is_empty()
            );
        }
        if let Some(pricing) = item.get("pricing") {
            item_keys.push("pricing");
            keys(
                pricing,
                &[
                    "priced_requests",
                    "unpriced_requests",
                    "cost_status",
                    "pricing_snapshots",
                ],
            );
            assert_eq!(pricing["cost_status"], "unknown");
            assert_eq!(pricing["priced_requests"], 0);
            assert_eq!(
                pricing["unpriced_requests"],
                item["diagnostics"]["last_attempt"]["usage"]["requests"]
            );
            assert_eq!(pricing["pricing_snapshots"], json!([]));
        }
        keys(item, &item_keys);
        assert_eq!(item["item_id"], format!("i{}", index + 1));
        assert!(item["name"].is_string());
        assert!(matches!(item["kind"].as_str(), Some("file" | "url")));
        assert!(matches!(item["status"].as_str(), Some("done" | "error")));
        assert_eq!(item["finished_at"], meta["finished_at"]);
        assert_eq!(item["operation"], "convert");
        assert_eq!(item["retryable"], item["kind"] == "url");
        assert!(item["skipped"].is_boolean() && item["llm_enhanced"].is_boolean());
        assert!(item["duration_ms"].is_null() || item["duration_ms"].as_u64().is_some());
        let cost = item["cost_usd"].as_f64().unwrap();
        assert!(cost.is_finite() && cost >= 0.0);
        for key in ["error", "skip_reason", "output", "output_name"] {
            assert!(item[key].is_null() || item[key].is_string());
        }
        let warnings = item["warnings"].as_array().unwrap();
        assert!(warnings.iter().all(Value::is_string));
        assert_eq!(
            warnings
                .iter()
                .map(|v| v.as_str().unwrap())
                .collect::<BTreeSet<_>>()
                .len(),
            warnings.len()
        );
        if let Some(output) = item["output"].as_str() {
            assert_eq!(
                Path::new(output).file_name().unwrap().to_str().unwrap(),
                output
            );
            assert!(copied.contains_key(Path::new(output)));
        } else {
            assert!(item["output_name"].is_null());
        }
    }
    meta
}
fn only_job(root: &Path) -> (PathBuf, Value) {
    let paths = jobs(root);
    assert_eq!(paths.len(), 1, "jobs={paths:?}");
    let meta = read_job(&paths[0]);
    (paths[0].clone(), meta)
}
fn observed_completion(root: &Path, output: &str) -> bool {
    fn has(value: &Value, suffix: &str) -> bool {
        value["status"] == "completed"
            && value["output"]
                .as_str()
                .is_some_and(|s| s.ends_with(suffix))
            || value
                .as_object()
                .is_some_and(|m| m.values().any(|v| has(v, suffix)))
    }
    let dir = root.join("out/.markitai/states");
    let Ok(entries) = std::fs::read_dir(dir) else {
        return false;
    };
    entries
        .flatten()
        .filter_map(|e| std::fs::read_to_string(e.path()).ok())
        .any(|raw| {
            serde_json::from_str::<Value>(&raw).is_ok_and(|v| has(&v, output))
                || raw
                    .lines()
                    .filter_map(|line| serde_json::from_str::<Value>(line).ok())
                    .any(|v| has(&v, output))
        })
}

#[test]
fn history_selection_obeys_flags_environment_configuration_and_default_off() {
    let cases: &[(bool, Option<&str>, &[&str], bool)] = &[
        (false, None, &[], false),
        (true, None, &[], true),
        (false, Some("  YeS  "), &[], true),
        (true, Some("off"), &[], false),
        (true, Some("FALSE"), &[], false),
        (true, Some("no"), &[], false),
        (true, Some("0"), &[], false),
        (true, Some("unexpected"), &[], false),
        (true, Some("   "), &[], true),
        (false, Some("0"), &["--record-history"], true),
        (true, Some("1"), &["--no-record-history"], false),
        (
            false,
            None,
            &["--no-record-history", "--record-history"],
            true,
        ),
        (
            true,
            None,
            &["--record-history", "--no-record-history"],
            false,
        ),
    ];
    for &(configured, environment, flags, enabled) in cases {
        let root = tempfile::tempdir().unwrap();
        let mut cfg = configure(root.path());
        cfg["history"]["record"] = json!(configured);
        save(root.path(), &cfg);
        write(root.path(), "note.txt", "History selection fixture.\n");
        let mut args = vec!["note.txt", "-o", "out", "--json"];
        args.extend_from_slice(flags);
        let env: Vec<_> = environment
            .map(|s| ("MARKITAI_RECORD_HISTORY", s))
            .into_iter()
            .collect();
        envelope(&Running::spawn(root.path(), &args, &env).finish(), 0);
        assert_eq!(
            jobs(root.path()).len(),
            usize::from(enabled),
            "config={configured}, env={environment:?}, flags={flags:?}"
        );
        if enabled {
            only_job(root.path());
        } else {
            assert!(!root.path().join("home/serve").exists());
        }
    }
}

#[test]
fn all_four_modes_publish_one_job_with_mode_appropriate_names_and_real_outputs() {
    let server = Server::start();
    for mode in 0..4 {
        let root = tempfile::tempdir().unwrap();
        configure(root.path());
        write(root.path(), "doc.txt", "Single local file.\n");
        write(root.path(), "input/nested/doc.txt", "Nested local file.\n");
        write(
            root.path(),
            "list.urls",
            &format!("{} chosen\n", server.url("/named")),
        );
        let inputs = [
            root.path().join("doc.txt").to_string_lossy().into_owned(),
            "input".into(),
            server.url("/single"),
            "list.urls".into(),
        ];
        let cli = envelope(
            &invoke(
                root.path(),
                &[&inputs[mode], "-o", "out", "--record-history", "--json"],
            ),
            0,
        );
        let (job, meta) = only_job(root.path());
        let item = &meta["items"][0];
        assert_eq!(meta["items"].as_array().unwrap().len(), 1);
        let expected = [
            "doc.txt".into(),
            "nested/doc.txt".into(),
            server.url("/single"),
            server.url("/named"),
        ];
        assert_eq!(item["name"], expected[mode]);
        assert_eq!(item["kind"], if mode < 2 { "file" } else { "url" });
        assert_eq!(item["status"], "done");
        assert_eq!(item["skipped"], false);
        assert!(item["error"].is_null() && item["skip_reason"].is_null());
        assert_eq!(item["output_name"], item["output"]);
        assert_eq!(item["llm_enhanced"], false);
        let original = PathBuf::from(cli["items"][0]["output"].as_str().unwrap());
        let original = if original.is_absolute() {
            original
        } else {
            root.path().join(original)
        };
        assert_eq!(
            std::fs::read(job.join("out").join(item["output"].as_str().unwrap())).unwrap(),
            std::fs::read(original).unwrap()
        );
        assert_eq!(
            meta["options"],
            json!({"preset":null,"llm":false,"ocr":false,"origin":"cli"})
        );
        assert!(!job.join("uploads").exists());
    }
}

#[test]
fn human_confirmation_quiet_and_json_keep_output_channels_separate() {
    for (flags, confirmation) in [
        (&[][..], true),
        (&["--quiet"][..], false),
        (&["--json"][..], false),
    ] {
        let root = tempfile::tempdir().unwrap();
        configure(root.path());
        write(root.path(), "doc.txt", "Visible body.\n");
        let mut args = vec![
            "doc.txt",
            "-o",
            "out",
            "--record-history",
            "--preset",
            "minimal",
        ];
        args.extend_from_slice(flags);
        let output = invoke(root.path(), &args);
        status(&output, 0);
        if flags.contains(&"--json") {
            envelope(&output, 0);
        }
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert_eq!(stderr.contains("Recorded in history:"), confirmation);
        assert!(!stderr.contains("markitai serve"));
        let (path, meta) = only_job(root.path());
        if confirmation {
            assert!(stderr.contains(path.to_str().unwrap()));
        }
        assert_eq!(meta["options"]["preset"], "minimal");
    }
}

#[test]
fn enhanced_url_history_copies_only_the_final_member_and_normalizes_output_name() {
    let server = Server::start();
    let root = tempfile::tempdir().unwrap();
    let mut cfg = configure(root.path());
    cfg["llm"] = json!({"enabled":true,"keep_base":true,"on_failure":"fail",
        "router_settings":{"num_retries":0,"timeout":20},
        "model_list":[{"model_name":"history","litellm_params":{"model":"openai/history-fixture",
            "api_base":format!("{}/v1",server.base),"api_key":"local-fixture-only"}}]});
    save(root.path(), &cfg);
    let cli = envelope(
        &invoke(
            root.path(),
            &[
                &server.url("/enhanced"),
                "-o",
                "out/chosen.md",
                "--record-history",
                "--json",
            ],
        ),
        0,
    );
    let (job, meta) = only_job(root.path());
    let item = &meta["items"][0];
    assert_eq!(meta["options"]["llm"], true);
    assert_eq!(item["output"], "chosen.llm.md");
    assert_eq!(item["output_name"], "chosen.md");
    assert_eq!(item["llm_enhanced"], true);
    assert_eq!(item["cost_usd"], cli["items"][0]["cost_usd"]);
    assert!(root.path().join("out/chosen.md").is_file());
    assert!(
        std::fs::read_to_string(job.join("out/chosen.llm.md"))
            .unwrap()
            .contains("Local model response.")
    );
    assert!(!job.join("out/chosen.md").exists());
    assert_eq!(server.count("/v1/chat/completions"), 1);
}

#[test]
fn directory_failures_and_skips_remain_distinct_in_terminal_history() {
    let root = tempfile::tempdir().unwrap();
    let mut cfg = configure(root.path());
    cfg["output"]["on_conflict"] = json!("skip");
    save(root.path(), &cfg);
    write(root.path(), "input/a.txt", "Will be skipped.\n");
    write(root.path(), "input/b.ipynb", "{invalid notebook");
    std::fs::write(
        root.path().join("input/c.png"),
        [
            0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48,
            0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00,
            0x00, 0x1f, 0x15, 0xc4, 0x89, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x44, 0x41, 0x54, 0x78,
            0x9c, 0x63, 0xf8, 0xcf, 0xc0, 0xf0, 0x1f, 0x00, 0x05, 0x00, 0x01, 0xff, 0x89, 0x99,
            0x3d, 0x1d, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
        ],
    )
    .unwrap();
    write(root.path(), "input/d.txt", "New completed file.\n");
    write(
        root.path(),
        "out/a.txt.md",
        "Existing downloadable output.\n",
    );
    let cli = envelope(
        &invoke(
            root.path(),
            &["input", "-o", "out", "--record-history", "--json"],
        ),
        10,
    );
    assert_eq!(cli["items"].as_array().unwrap().len(), 4);
    let (job, meta) = only_job(root.path());
    let items = meta["items"].as_array().unwrap();
    assert_eq!(
        items
            .iter()
            .map(|x| x["name"].as_str().unwrap())
            .collect::<Vec<_>>(),
        vec!["a.txt", "b.ipynb", "c.png", "d.txt"]
    );
    assert_eq!(items[0]["status"], "done");
    assert_eq!(items[0]["skipped"], true);
    assert_eq!(items[0]["skip_reason"], "exists");
    assert_eq!(items[0]["output"], "a.txt.md");
    assert_eq!(
        std::fs::read(job.join("out/a.txt.md")).unwrap(),
        b"Existing downloadable output.\n"
    );
    assert_eq!(items[1]["status"], "error");
    assert!(!items[1]["error"].as_str().unwrap().is_empty());
    assert!(items[1]["output"].is_null());
    assert_eq!(items[1]["skipped"], false);
    assert_eq!(items[2]["status"], "done");
    assert_eq!(items[2]["skipped"], true);
    assert_eq!(items[2]["skip_reason"], "image_only");
    assert!(items[2]["output"].is_null());
    assert_eq!(items[3]["status"], "done");
    assert_eq!(items[3]["skipped"], false);
}

#[test]
fn single_conversion_failure_is_recorded_but_invalid_input_is_not() {
    for (name, content, recorded) in [
        ("broken.ipynb", "{not JSON", true),
        ("unsupported.xyz", "unknown", false),
        ("missing.txt", "", false),
    ] {
        let root = tempfile::tempdir().unwrap();
        configure(root.path());
        if name != "missing.txt" {
            write(root.path(), name, content);
        }
        envelope(
            &invoke(
                root.path(),
                &[name, "-o", "out", "--record-history", "--json"],
            ),
            1,
        );
        assert_eq!(jobs(root.path()).len(), usize::from(recorded));
        if recorded {
            let (_, meta) = only_job(root.path());
            assert_eq!(meta["items"][0]["status"], "error");
            assert!(meta["items"][0]["output"].is_null());
            assert_eq!(meta["dir_size_bytes"], 0);
        }
    }
}

#[test]
fn stdout_dry_run_and_empty_input_do_not_create_history() {
    let server = Server::start();
    for mode in 0..6 {
        let root = tempfile::tempdir().unwrap();
        configure(root.path());
        write(root.path(), "doc.txt", "Stdout body.\n");
        std::fs::create_dir(root.path().join("empty")).unwrap();
        write(root.path(), "empty.urls", "# no URLs\n");
        let url = server.url("/stdout");
        let args = match mode {
            0 => vec!["doc.txt", "--record-history"],
            1 => vec![url.as_str(), "--record-history"],
            2 => vec!["doc.txt", "-o", "out", "--record-history", "--dry-run"],
            3 => vec![
                "empty",
                "-o",
                "out",
                "--record-history",
                "--dry-run",
                "--resume",
            ],
            4 => vec!["empty", "-o", "out", "--record-history", "--json"],
            _ => vec!["empty.urls", "-o", "out", "--record-history", "--json"],
        };
        let result = invoke(root.path(), &args);
        status(&result, if mode == 5 { 1 } else { 0 });
        if [4, 5].contains(&mode) {
            envelope(&result, if mode == 5 { 1 } else { 0 });
        }
        if mode == 0 {
            assert!(String::from_utf8_lossy(&result.stdout).contains("Stdout body."));
        }
        if mode == 1 {
            assert!(String::from_utf8_lossy(&result.stdout).contains("Local content for /stdout"));
            assert_eq!(server.count("/stdout"), 1);
        }
        assert!(!root.path().join("home/serve").exists());
        assert!(!root.path().join("out").exists());
    }
}

#[test]
fn url_list_history_uses_completion_order_and_keeps_named_identity_without_duplicates() {
    let server = Server::start();
    server.hold("/slow");
    let root = tempfile::tempdir().unwrap();
    let mut cfg = configure(root.path());
    cfg["batch"]["state_flush_interval_seconds"] = json!(1);
    save(root.path(), &cfg);
    write(
        root.path(),
        "list.urls",
        &format!(
            "{} slow\n{} fast\n{} second\n{} fast\n",
            server.url("/slow"),
            server.url("/fast"),
            server.url("/fast"),
            server.url("/fast")
        ),
    );
    let running = Running::spawn(
        root.path(),
        &["list.urls", "-o", "out", "--record-history", "--json"],
        &[],
    );
    server.wait("/slow", 1);
    server.wait("/fast", 2);
    wait_until("coordinator observed the second named completion", || {
        observed_completion(root.path(), "second.md")
    });
    server.release("/slow");
    let cli = envelope(&running.finish(), 0);
    assert_eq!(cli["items"].as_array().unwrap().len(), 3);
    let (_, meta) = only_job(root.path());
    let items = meta["items"].as_array().unwrap();
    assert_eq!(
        items
            .iter()
            .map(|x| x["output"].as_str().unwrap())
            .collect::<Vec<_>>(),
        vec!["fast.md", "second.md", "slow.md"]
    );
    assert_eq!(items[0]["name"], server.url("/fast"));
    assert_eq!(items[1]["name"], server.url("/fast"));
    assert_eq!(items[2]["name"], server.url("/slow"));
    assert_eq!(cli["items"][0]["source"], server.url("/slow"));
}

#[test]
fn resume_records_only_newly_observed_items_and_no_job_for_an_all_complete_run() {
    for directory in [false, true] {
        let server = Server::start();
        server.fail("/retry", true);
        let root = tempfile::tempdir().unwrap();
        configure(root.path());
        let list = if directory {
            "input/list.urls"
        } else {
            "list.urls"
        };
        let input = if directory { "input" } else { "list.urls" };
        write(
            root.path(),
            list,
            &format!(
                "{} completed\n{} retry\n",
                server.url("/done"),
                server.url("/retry")
            ),
        );
        if directory {
            write(root.path(), "input/local.txt", "Completed local file.\n");
        }
        envelope(
            &invoke(
                root.path(),
                &[input, "-o", "out", "--record-history", "--json"],
            ),
            10,
        );
        let (first_job, _) = only_job(root.path());
        let first_snapshot = files(&first_job);
        server.fail("/retry", false);
        write(
            root.path(),
            list,
            &format!(
                "{} completed\n{} retry\n{} added\n",
                server.url("/done"),
                server.url("/retry"),
                server.url("/new")
            ),
        );
        let cli = envelope(
            &invoke(
                root.path(),
                &[input, "-o", "out", "--record-history", "--resume", "--json"],
            ),
            0,
        );
        assert_eq!(cli["items"].as_array().unwrap().len(), 2);
        let all = jobs(root.path());
        assert_eq!(all.len(), 2);
        let second = all.iter().find(|p| **p != first_job).unwrap();
        let meta = read_job(second);
        assert_eq!(
            meta["items"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v["name"].as_str().unwrap())
                .collect::<BTreeSet<_>>(),
            [server.url("/retry"), server.url("/new")]
                .iter()
                .map(String::as_str)
                .collect()
        );
        assert_eq!(server.count("/done"), 1);
        assert_eq!(server.count("/retry"), 2);
        assert_eq!(files(&first_job), first_snapshot);
        let empty = envelope(
            &invoke(
                root.path(),
                &[input, "-o", "out", "--record-history", "--resume", "--json"],
            ),
            0,
        );
        assert_eq!(empty["items"], json!([]));
        assert_eq!(jobs(root.path()), all);
    }
}

fn email_fixture() -> &'static str {
    concat!(
        "MIME-Version: 1.0\r\nSubject: History PNG\r\nContent-Type: multipart/mixed; boundary=history-png\r\n\r\n",
        "--history-png\r\nContent-Type: text/plain; charset=utf-8\r\n\r\nNative attachment.\r\n",
        "--history-png\r\nContent-Type: image/png; name=pixel.png\r\nContent-Disposition: attachment; filename=pixel.png\r\nContent-Transfer-Encoding: base64\r\n\r\n",
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg==\r\n--history-png--\r\n"
    )
}

#[test]
fn flattened_history_outputs_and_assets_are_independent_after_sources_and_outputs_disappear() {
    for profile in [None, Some("rag")] {
        let root = tempfile::tempdir().unwrap();
        let mut cfg = configure(root.path());
        cfg["output"]["report"] = json!(true);
        cfg["output"]["profile"] = json!(profile);
        save(root.path(), &cfg);
        write(root.path(), "input/a/same.eml", email_fixture());
        write(root.path(), "input/b/same.eml", email_fixture());
        envelope(
            &invoke(
                root.path(),
                &["input", "-o", "out", "--record-history", "--json"],
            ),
            0,
        );
        let (job, meta) = only_job(root.path());
        let items = meta["items"].as_array().unwrap();
        assert_eq!(
            items
                .iter()
                .map(|v| v["output"].as_str().unwrap())
                .collect::<Vec<_>>(),
            vec!["same.eml.md", "same.eml (2).md"]
        );
        let copied = files(&job);
        let asset_prefix = if profile.is_some() {
            "out/assets"
        } else {
            "out/.markitai/assets"
        };
        let assets: Vec<_> = copied
            .iter()
            .filter(|(p, _)| p.starts_with(asset_prefix))
            .collect();
        assert_eq!(
            assets.len(),
            1,
            "identical source assets should share one copy"
        );
        assert!(assets[0].1.starts_with(b"\x89PNG\r\n\x1a\n"));
        for item in items {
            let text =
                std::fs::read_to_string(job.join("out").join(item["output"].as_str().unwrap()))
                    .unwrap();
            assert!(text.contains(assets[0].0.strip_prefix("out").unwrap().to_str().unwrap()));
        }
        assert!(!copied.keys().any(|p| p.components().any(|c| matches!(
            c.as_os_str().to_str(),
            Some("states" | "reports" | "ownership" | "uploads")
        ))));
        for path in files(&root.path().join("out")).keys() {
            std::fs::write(root.path().join("out").join(path), b"mutated original").unwrap();
        }
        assert_eq!(
            files(&job),
            copied,
            "history must not share mutable inodes with live output"
        );
        std::fs::remove_dir_all(root.path().join("input")).unwrap();
        std::fs::remove_dir_all(root.path().join("out")).unwrap();
        assert_eq!(files(&job), copied);
        read_job(&job);
    }
}

#[test]
fn output_removed_before_history_publication_keeps_item_without_a_download() {
    let server = Server::start();
    server.hold("/held");
    let root = tempfile::tempdir().unwrap();
    let mut cfg = configure(root.path());
    cfg["batch"]["state_flush_interval_seconds"] = json!(1);
    save(root.path(), &cfg);
    write(
        root.path(),
        "input/a.txt",
        "Completed then externally removed.\n",
    );
    write(
        root.path(),
        "input/urls.urls",
        &format!("{} held\n", server.url("/held")),
    );
    let running = Running::spawn(
        root.path(),
        &["input", "-o", "out", "--record-history", "--json"],
        &[],
    );
    server.wait("/held", 1);
    wait_until("durably observed local completion", || {
        observed_completion(root.path(), "a.txt.md")
    });
    std::fs::remove_file(root.path().join("out/a.txt.md")).unwrap();
    server.release("/held");
    envelope(&running.finish(), 0);
    let (_, meta) = only_job(root.path());
    let items = meta["items"].as_array().unwrap();
    let missing = items.iter().find(|v| v["name"] == "a.txt").unwrap();
    assert_eq!(missing["status"], "done");
    assert!(missing["output"].is_null() && missing["output_name"].is_null());
    let url = items.iter().find(|v| v["kind"] == "url").unwrap();
    assert_eq!(url["output"], "held.md");
}

#[test]
fn history_write_failure_is_nonfatal_for_success_and_partial_conversion_failure() {
    for failed in [false, true] {
        let root = tempfile::tempdir().unwrap();
        configure(root.path());
        write(root.path(), "home/serve/jobs", "Keep this blocker.\n");
        write(root.path(), "input/good.txt", "Still converted.\n");
        if failed {
            write(root.path(), "input/bad.ipynb", "{broken");
        }
        let output = invoke(
            root.path(),
            &["input", "-o", "out", "--record-history", "--json"],
        );
        let cli = envelope(&output, if failed { 10 } else { 0 });
        assert!(
            cli["items"]
                .as_array()
                .unwrap()
                .iter()
                .any(|v| v["status"] == "completed")
        );
        assert!(
            String::from_utf8_lossy(&output.stderr)
                .to_lowercase()
                .contains("history")
        );
        assert!(root.path().join("out/good.txt.md").is_file());
        assert_eq!(
            std::fs::read(root.path().join("home/serve/jobs")).unwrap(),
            b"Keep this blocker.\n"
        );
        assert_eq!(
            std::fs::read_dir(root.path().join("home/serve"))
                .unwrap()
                .count(),
            1
        );
    }
}

#[cfg(unix)]
#[test]
fn controlled_sigint_drains_active_output_without_publishing_history_or_dispatching_the_queue() {
    let server = Server::start();
    server.hold("/active");
    let root = tempfile::tempdir().unwrap();
    let mut cfg = configure(root.path());
    cfg["batch"]["url_concurrency"] = json!(1);
    save(root.path(), &cfg);
    write(
        root.path(),
        "list.urls",
        &format!(
            "{} active\n{} queued\n",
            server.url("/active"),
            server.url("/queued")
        ),
    );
    let running = Running::spawn(
        root.path(),
        &["list.urls", "-o", "out", "--record-history", "--json"],
        &[],
    );
    server.wait("/active", 1);
    running.interrupt();
    server.release("/active");
    let cli = envelope(&running.finish(), 130);
    assert_eq!(cli["items"].as_array().unwrap().len(), 1);
    assert_eq!(cli["items"][0]["status"], "completed");
    assert_eq!(server.count("/queued"), 0);
    assert!(root.path().join("out/active.md").is_file());
    assert!(!root.path().join("home/serve").exists());
    envelope(
        &invoke(
            root.path(),
            &[
                "list.urls",
                "-o",
                "out",
                "--record-history",
                "--resume",
                "--json",
            ],
        ),
        0,
    );
    let (_, meta) = only_job(root.path());
    assert_eq!(meta["items"].as_array().unwrap().len(), 1);
    assert_eq!(meta["items"][0]["name"], server.url("/queued"));
    assert_eq!(server.count("/active"), 1);
}

#[test]
fn resumed_failed_file_removed_from_discovery_is_still_an_observed_history_failure() {
    let root = tempfile::tempdir().unwrap();
    configure(root.path());
    write(root.path(), "input/bad.ipynb", "{invalid notebook");
    let args = ["input", "-o", "out", "--record-history", "--json"];
    envelope(&invoke(root.path(), &args), 10);
    let (first, first_meta) = only_job(root.path());
    assert_eq!(first_meta["items"][0]["name"], "bad.ipynb");
    assert_eq!(first_meta["items"][0]["status"], "error");
    let preserved = files(&first);
    std::fs::remove_file(root.path().join("input/bad.ipynb")).unwrap();
    assert_eq!(
        std::fs::read_dir(root.path().join("input"))
            .unwrap()
            .count(),
        0
    );

    let output = invoke(
        root.path(),
        &[
            "input",
            "-o",
            "out",
            "--record-history",
            "--resume",
            "--json",
        ],
    );
    let cli = envelope(&output, 10);
    assert_eq!(cli["items"].as_array().unwrap().len(), 1);
    assert_eq!(cli["items"][0]["status"], "failed");
    let all = jobs(root.path());
    assert_eq!(
        all.len(),
        2,
        "the retained failed state item was attempted this run"
    );
    let second = all.iter().find(|path| **path != first).unwrap();
    let meta = read_job(second);
    assert_eq!(meta["items"].as_array().unwrap().len(), 1);
    let item = &meta["items"][0];
    assert_eq!(item["name"], "bad.ipynb");
    assert_eq!(item["status"], "error");
    assert_eq!(item["kind"], "file");
    assert_eq!(item["retryable"], false);
    assert!(!item["error"].as_str().unwrap().is_empty());
    assert!(item["output"].is_null() && item["output_name"].is_null());
    assert!(item["duration_ms"].as_u64().is_some());
    assert_eq!(meta["dir_size_bytes"], 0);
    assert_eq!(files(&first), preserved);
}

#[test]
fn skipped_media_outputs_relocate_each_roots_assets_into_an_independent_archive() {
    const LITERALS: &str = concat!(
        "\n`<img src=\".markitai/assets/poster.png\" srcset=\".markitai/assets/poster.png 2x\">`\n",
        "```html\n<video src='.markitai/assets/clip.mp4' poster='.markitai/assets/poster.png'></video>\n```\n",
        "<!-- <source srcset=\".markitai/assets/poster.png 1x, .markitai/assets/shared.png 2x\"> -->\n",
        "<pre><audio src='.markitai/assets/audio.ogg'></audio></pre>\n",
        "<script type=\"text/plain\">const example = '<track src=\".markitai/assets/captions.vtt\">';</script>\n",
    );
    fn document(suffix: &str) -> String {
        format!(
            concat!(
                "# Saved media output\n\n",
                "<img src=\".markitai/assets/poster{suffix}.png?size=1&amp;theme=dark#preview\" srcset=\".markitai/assets/poster{suffix}.png 1x, .markitai/assets/shared.png 2x, https://example.invalid/remote.png 3x\" alt=\".markitai/assets/poster.png\" data-src=\".markitai/assets/poster.png\">\n",
                "<picture>\n",
                "  <source media=\"(min-width: 900px)\" srcset=\"data:image/svg+xml,%3Csvg%3E,%3C/svg%3E 1x, .markitai/assets/poster{suffix}.png 2x\">\n",
                "  <source srcset='.markitai/assets/poster{suffix}.png 320w, .markitai/assets/shared.png 640w'>\n",
                "  <img src='.markitai/assets/%73hared.png' srcset='.markitai/assets/shared.png, .markitai/assets/poster{suffix}.png 2x'>\n",
                "</picture>\n",
                "<video src=\".markitai/assets/clip{suffix}.mp4#t=2\" poster='.markitai/assets/poster{suffix}.png'>\n",
                "  <source src=.markitai/assets/clip{suffix}.mp4 type=video/mp4>\n",
                "  <track src=\".markitai/assets/captions{suffix}.vtt?lang=en&amp;mode=cc#cue\" kind=captions>\n",
                "</video>\n",
                "<audio src='.markitai/assets/audio{suffix}.ogg'><source src=\".markitai/assets/audio{suffix}.ogg\"></audio>\n",
                "<style>.example {{ background-image: url(.markitai/assets/poster{suffix}.png); }}</style>\n",
                "<p data-src=\".markitai/assets/poster.png\" title=\".markitai/assets/clip.mp4\" style=\"background-image: url(.markitai/assets/poster{suffix}.png)\">Literal .markitai/assets/poster.png</p>\n",
                "{literals}",
            ),
            suffix = suffix,
            literals = LITERALS,
        )
    }

    let root = tempfile::tempdir().unwrap();
    let mut cfg = configure(root.path());
    cfg["output"]["on_conflict"] = json!("skip");
    save(root.path(), &cfg);
    let original = document("");
    let shared = b"One identical asset used by both saved outputs.\n";
    let mut expected_assets = BTreeMap::new();
    for (folder, suffix) in [("a", ""), ("b", "-2")] {
        write(
            root.path(),
            &format!("input/{folder}/page.txt"),
            "Conversion must skip this input and archive the pre-existing output.\n",
        );
        write(root.path(), &format!("out/{folder}/page.txt.md"), &original);
        for (stem, extension) in [
            ("poster", "png"),
            ("clip", "mp4"),
            ("audio", "ogg"),
            ("captions", "vtt"),
        ] {
            // These are opaque saved assets: the skip path must copy their
            // exact bytes without trying to decode media or reconvert inputs.
            let bytes = format!("Saved {stem} bytes belonging to root {folder}.\n");
            write(
                root.path(),
                &format!("out/{folder}/.markitai/assets/{stem}.{extension}"),
                &bytes,
            );
            expected_assets.insert(
                PathBuf::from(format!(".markitai/assets/{stem}{suffix}.{extension}")),
                bytes.into_bytes(),
            );
        }
        write(
            root.path(),
            &format!("out/{folder}/.markitai/assets/shared.png"),
            std::str::from_utf8(shared).unwrap(),
        );
    }
    expected_assets.insert(
        PathBuf::from(".markitai/assets/shared.png"),
        shared.to_vec(),
    );
    let original_outputs = files(&root.path().join("out"));
    let original_inputs = files(&root.path().join("input"));
    let cli = envelope(
        &invoke(
            root.path(),
            &["input", "-o", "out", "--record-history", "--json"],
        ),
        0,
    );
    assert_eq!(cli["items"].as_array().unwrap().len(), 2);
    assert!(
        cli["items"]
            .as_array()
            .unwrap()
            .iter()
            .all(|item| item["status"] == "skipped" && item["skip_reason"] == "exists")
    );
    assert_eq!(files(&root.path().join("input")), original_inputs);
    for (path, bytes) in &original_outputs {
        assert_eq!(
            std::fs::read(root.path().join("out").join(path)).unwrap(),
            *bytes
        );
    }

    let (job, meta) = only_job(root.path());
    let archived = job.join("out");
    let items = meta["items"].as_array().unwrap();
    assert_eq!(items.len(), 2);
    let mut expected_files = expected_assets.clone();
    for (index, (name, suffix)) in [("page.txt.md", ""), ("page.txt (2).md", "-2")]
        .into_iter()
        .enumerate()
    {
        assert_eq!(
            items[index]["name"],
            format!("{}/page.txt", ["a", "b"][index])
        );
        assert_eq!(items[index]["output"], name);
        assert_eq!(items[index]["status"], "done");
        assert_eq!(items[index]["skipped"], true);
        assert_eq!(items[index]["skip_reason"], "exists");
        let actual = std::fs::read_to_string(archived.join(name)).unwrap();
        assert_eq!(actual, document(suffix));
        assert!(actual.ends_with(LITERALS));
        expected_files.insert(PathBuf::from(name), actual.into_bytes());

        // Every active local URL spelling in the exact document above is
        // resolved here, including encoded identity and query/fragment suffixes.
        // The external/data candidates and protected examples are not local
        // archive dependencies and must retain their original spelling.
        for url in [
            format!(".markitai/assets/poster{suffix}.png?size=1&amp;theme=dark#preview"),
            format!(".markitai/assets/poster{suffix}.png"),
            ".markitai/assets/shared.png".into(),
            ".markitai/assets/%73hared.png".into(),
            format!(".markitai/assets/clip{suffix}.mp4#t=2"),
            format!(".markitai/assets/clip{suffix}.mp4"),
            format!(".markitai/assets/captions{suffix}.vtt?lang=en&amp;mode=cc#cue"),
            format!(".markitai/assets/audio{suffix}.ogg"),
        ] {
            let path = url.split(['?', '#']).next().unwrap().replace("%73", "s");
            let path = Path::new(&path);
            assert_eq!(
                std::fs::read(archived.join(path)).unwrap(),
                expected_assets[path]
            );
        }
    }
    assert_eq!(files(&archived), expected_files);
    assert_eq!(expected_assets.len(), 9, "shared asset must be stored once");
    let complete_job = files(&job);
    for path in original_outputs.keys() {
        std::fs::write(root.path().join("out").join(path), b"live output mutated").unwrap();
    }
    assert_eq!(files(&job), complete_job);
    std::fs::remove_dir_all(root.path().join("input")).unwrap();
    std::fs::remove_dir_all(root.path().join("out")).unwrap();
    assert_eq!(files(&job), complete_job);
    read_job(&job);
}

#[test]
fn skipped_css_outputs_relocate_resources_without_rewriting_literal_examples() {
    const TEMPLATE: &str = r##"---
example: |
  <style>.example { background: url(.markitai/assets/poster.png); }</style>
quoted: "<div style='background:url(.markitai/assets/poster.png)'>"
---
# Saved CSS output

<div style="background: url(&quot;@POSTER@?size=1&amp;theme=dark#preview&quot;); --example: 'url(.markitai/assets/poster.png)'">Styled</div>
<p STYLE='mask-image: u\72l("@ESCAPED_POSTER@?\76 =1\26 mode=dark#\70 review"); background: url("@ENTITY_POSTER@?v=2&amp;mode=light#thumb")'>Escapes and entities</p>
<i style=background-image:url(@POSTER@) data-style="url(.markitai/assets/poster.png)">Unquoted attribute</i>
<style>
@import "@THEME@?variant=night#sheet" screen;
@import url('@THEME@?version=2') print;
@font-face { font-family: Fixture; src: local("url(.markitai/assets/font.woff2)"), url('@FONT@?#face') format("woff2"); }
.plain { background: URL( @POSTER@?x=1&amp;y=2#raw ); }
.escaped { background: u\72l('@ESCAPED_POSTER@?\76 =1\26 mode=dark#\70 review'); }
.choices { background-image: image-set("@POSTER@?density=1#one" 1x, url('.markitai/assets/shared.png') 2x); }
.webkit { background-image: -webkit-image-set(url("@POSTER@") 1x, ".markitai/assets/shared.png" 2x); }
.remote { background-image: url(https://example.invalid/p.png), url("data:image/svg+xml,%3Csvg%3E,%3C/svg%3E"); }
.literal::before { content: "url(.markitai/assets/poster.png) image-set('.markitai/assets/shared.png' 2x) &quot;"; }
/* url(.markitai/assets/poster.png) @import '.markitai/assets/theme.css'; */
</style>

`<span style="background:url(.markitai/assets/poster.png)">` and `url(.markitai/assets/poster.png)`.
```css
@import ".markitai/assets/theme.css";
.example { background: url(.markitai/assets/poster.png); }
```
    <div style="background:url(.markitai/assets/poster.png)">Indented example</div>
<!-- <style>.example { background: url(.markitai/assets/poster.png); }</style> -->
<pre><style>.example { background: url(.markitai/assets/poster.png); }</style></pre>
<code><span style="background:url(.markitai/assets/poster.png)">Code</span></code>
<script type="text/plain">const example = '<style>.x { background: url(.markitai/assets/poster.png); }</style>';</script>
<p title="url(.markitai/assets/poster.png)" data-style="url(.markitai/assets/poster.png)">Literal .markitai/assets/poster.png</p>
"##;
    fn document(suffix: &str) -> String {
        let poster = format!(".markitai/assets/poster{suffix}.png");
        // Identity paths retain their original CSS/HTML spelling. A relocated
        // path is rendered from the archive name; its surrounding syntax and
        // escaped query/fragment remain independently observable in TEMPLATE.
        let escaped = if suffix.is_empty() {
            r".markitai/assets/p\6f ster.png"
        } else {
            &poster
        };
        let entity = if suffix.is_empty() {
            ".markitai/assets/p&#111;ster.png"
        } else {
            &poster
        };
        TEMPLATE
            .replace("@POSTER@", &poster)
            .replace("@ESCAPED_POSTER@", escaped)
            .replace("@ENTITY_POSTER@", entity)
            .replace("@THEME@", &format!(".markitai/assets/theme{suffix}.css"))
            .replace("@FONT@", &format!(".markitai/assets/font{suffix}.woff2"))
    }

    let root = tempfile::tempdir().unwrap();
    let mut cfg = configure(root.path());
    cfg["output"]["on_conflict"] = json!("skip");
    save(root.path(), &cfg);
    let original = document("");
    let shared = b"Identical saved image payload, reused by both CSS documents.\n";
    let mut expected_assets = BTreeMap::new();
    for (folder, suffix) in [("a", ""), ("b", "-2")] {
        write(
            root.path(),
            &format!("input/{folder}/page.txt"),
            "Archive the saved output without converting this input.\n",
        );
        write(root.path(), &format!("out/{folder}/page.txt.md"), &original);
        for (stem, extension) in [("poster", "png"), ("font", "woff2"), ("theme", "css")] {
            let bytes = if extension == "css" {
                // Imported stylesheets need no recursive resource copying.
                format!("/* Theme for {folder} */ body {{ color: blue; }}\n")
            } else {
                format!("Saved {stem} payload from root {folder}.\n")
            };
            write(
                root.path(),
                &format!("out/{folder}/.markitai/assets/{stem}.{extension}"),
                &bytes,
            );
            expected_assets.insert(
                PathBuf::from(format!(".markitai/assets/{stem}{suffix}.{extension}")),
                bytes.into_bytes(),
            );
        }
        write(
            root.path(),
            &format!("out/{folder}/.markitai/assets/shared.png"),
            std::str::from_utf8(shared).unwrap(),
        );
    }
    expected_assets.insert(
        PathBuf::from(".markitai/assets/shared.png"),
        shared.to_vec(),
    );
    let original_outputs = files(&root.path().join("out"));
    let original_inputs = files(&root.path().join("input"));
    let cli = envelope(
        &invoke(
            root.path(),
            &["input", "-o", "out", "--record-history", "--json"],
        ),
        0,
    );
    assert_eq!(cli["items"].as_array().unwrap().len(), 2);
    assert!(
        cli["items"]
            .as_array()
            .unwrap()
            .iter()
            .all(|item| item["status"] == "skipped" && item["skip_reason"] == "exists")
    );
    assert_eq!(files(&root.path().join("input")), original_inputs);
    for (path, bytes) in &original_outputs {
        assert_eq!(
            std::fs::read(root.path().join("out").join(path)).unwrap(),
            *bytes
        );
    }

    let (job, meta) = only_job(root.path());
    let archived = job.join("out");
    let items = meta["items"].as_array().unwrap();
    assert_eq!(items.len(), 2);
    let mut expected_files = expected_assets.clone();
    for (index, (name, suffix)) in [("page.txt.md", ""), ("page.txt (2).md", "-2")]
        .into_iter()
        .enumerate()
    {
        assert_eq!(
            items[index]["name"],
            format!("{}/page.txt", ["a", "b"][index])
        );
        assert_eq!(items[index]["output"], name);
        assert_eq!(items[index]["status"], "done");
        assert_eq!(items[index]["skipped"], true);
        assert_eq!(items[index]["skip_reason"], "exists");
        let actual = std::fs::read_to_string(archived.join(name)).unwrap();
        assert_eq!(actual, document(suffix));
        expected_files.insert(PathBuf::from(name), actual.into_bytes());

        // These are the resource URLs after CSS/HTML escape decoding. Exact
        // document comparison above protects every authored spelling and
        // literal example, while these checks resolve all local resources.
        for url in [
            format!(".markitai/assets/poster{suffix}.png?size=1&theme=dark#preview"),
            format!(".markitai/assets/poster{suffix}.png?v=1&mode=dark#preview"),
            format!(".markitai/assets/poster{suffix}.png?v=2&mode=light#thumb"),
            format!(".markitai/assets/poster{suffix}.png"),
            format!(".markitai/assets/theme{suffix}.css?variant=night#sheet"),
            format!(".markitai/assets/theme{suffix}.css?version=2"),
            format!(".markitai/assets/font{suffix}.woff2?#face"),
            format!(".markitai/assets/poster{suffix}.png?x=1&amp;y=2#raw"),
            format!(".markitai/assets/poster{suffix}.png?density=1#one"),
            ".markitai/assets/shared.png".into(),
        ] {
            let path = Path::new(url.split(['?', '#']).next().unwrap());
            assert_eq!(
                std::fs::read(archived.join(path)).unwrap(),
                expected_assets[path]
            );
        }
    }
    assert_eq!(expected_assets.len(), 7, "identical shared image is reused");
    assert_eq!(files(&archived), expected_files);
    let complete_job = files(&job);
    for path in original_outputs.keys() {
        std::fs::write(root.path().join("out").join(path), b"live output mutated").unwrap();
    }
    assert_eq!(files(&job), complete_job);
    std::fs::remove_dir_all(root.path().join("input")).unwrap();
    std::fs::remove_dir_all(root.path().join("out")).unwrap();
    assert_eq!(files(&job), complete_job);
    read_job(&job);
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
