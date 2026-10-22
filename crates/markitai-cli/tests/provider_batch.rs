//! `--llm-batch` against a loopback Anthropic Message Batches server: submit,
//! wait, collect, a pending handoff, resume and an uncertain creation proven
//! by reconciliation. No provider is contacted; keys are fixture strings.
#[path = "support/console.rs"]
mod console;

use serde_json::{Value, json};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;

const BATCH: &str = "msgbatch_01FixtureBatch000000000000";

#[derive(Clone, Debug)]
struct Seen {
    method: String,
    path: String,
    headers: String,
    body: Vec<u8>,
}

/// What the fake does next; tests switch it between runs.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// Accept the create, report the batch ended, serve results.
    Normal,
    /// Accept the create, then fail status reads (the CLI hands the job back).
    StatusDown,
    /// Read the create body, then drop the connection without an answer.
    CreateDisconnects,
}

#[derive(Default)]
struct State {
    seen: Vec<Seen>,
    /// The create body's requests, kept to answer their results.
    requests: Vec<Value>,
    created: bool,
}

struct Fake {
    base: String,
    state: Arc<Mutex<State>>,
    mode: Arc<Mutex<Mode>>,
    /// Requests whose custom ID ends with one of these fail as `errored`.
    errored: Arc<Mutex<Vec<String>>>,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

fn read_request(stream: &mut TcpStream) -> Option<Seen> {
    stream.set_nonblocking(false).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .unwrap();
    let mut reader = bounded_fixture_io::Reader::new(
        stream,
        std::time::Instant::now() + Duration::from_secs(10),
    );
    let mut bytes = Vec::new();
    loop {
        let mut chunk = [0; 8192];
        let count = reader.read(&mut chunk).ok()?;
        if count == 0 {
            return None;
        }
        bytes.extend_from_slice(&chunk[..count]);
        assert!(bytes.len() < 8 * 1024 * 1024);
        if let Some(split) = bytes.windows(4).position(|s| s == b"\r\n\r\n") {
            let headers = String::from_utf8_lossy(&bytes[..split]).into_owned();
            let length = headers
                .lines()
                .find_map(|line| {
                    line.to_ascii_lowercase()
                        .strip_prefix("content-length:")
                        .and_then(|value| value.trim().parse::<usize>().ok())
                })
                .unwrap_or(0);
            if bytes.len() >= split + 4 + length {
                let mut first = headers.lines().next()?.split_whitespace();
                return Some(Seen {
                    method: first.next()?.to_owned(),
                    path: first.next()?.to_owned(),
                    body: bytes[split + 4..split + 4 + length].to_vec(),
                    headers,
                });
            }
        }
    }
}

fn batch_object(state: &State, ended: bool) -> Value {
    let total = state.requests.len();
    let counts = if ended {
        json!({"processing":0,"succeeded":total,"errored":0,"canceled":0,"expired":0})
    } else {
        json!({"processing":total,"succeeded":0,"errored":0,"canceled":0,"expired":0})
    };
    json!({"id":BATCH,"type":"message_batch","processing_status":if ended {"ended"} else {"in_progress"},
        "request_counts":counts,"ended_at":if ended {json!("2026-10-11T08:20:00Z")} else {Value::Null},
        "created_at":"2026-10-11T08:00:00Z","expires_at":"2026-10-12T08:00:00Z","archived_at":null,
        "cancel_initiated_at":null,"results_url":if ended {json!(format!("https://api.anthropic.com/v1/messages/batches/{BATCH}/results"))} else {Value::Null}})
}

/// A forced-tool answer that keeps the protected source exactly, as a
/// compliant model would.
fn result_row(request: &Value, errored: bool) -> Value {
    let id = request["custom_id"].as_str().unwrap();
    if errored {
        return json!({"custom_id":id,"result":{"type":"errored","error":{"type":"error",
            "error":{"type":"overloaded_error","message":"Overloaded"}}}});
    }
    let params = &request["params"];
    let source = params["messages"][0]["content"].as_str().unwrap();
    let input = json!({"cleaned_markdown":source,"frontmatter":{"description":"Batch fixture","tags":["fixture"]}});
    json!({"custom_id":id,"result":{"type":"succeeded","message":{"id":"msg_fixture","type":"message",
        "role":"assistant","model":params["model"],"content":[{"type":"tool_use","id":"toolu_fixture",
        "name":params["tool_choice"]["name"],"input":input}],"stop_reason":"tool_use","stop_sequence":null,
        "usage":{"input_tokens":1000,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":100}}}})
}

impl Fake {
    fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let base = format!("http://{}/v1", listener.local_addr().unwrap());
        let state = Arc::new(Mutex::new(State::default()));
        let mode = Arc::new(Mutex::new(Mode::Normal));
        let errored = Arc::new(Mutex::new(Vec::<String>::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let (shared, current, failing, stopped) =
            (state.clone(), mode.clone(), errored.clone(), stop.clone());
        let thread = std::thread::spawn(move || {
            while !stopped.load(Ordering::SeqCst) {
                let mut stream = match listener.accept() {
                    Ok((stream, _)) => stream,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(2));
                        continue;
                    }
                    Err(error) => panic!("fixture accept: {error}"),
                };
                let Some(seen) = read_request(&mut stream) else {
                    continue;
                };
                let mode = *current.lock().unwrap();
                let mut state = shared.lock().unwrap();
                state.seen.push(seen.clone());
                let path = seen.path.split('?').next().unwrap().to_owned();
                let reply: Option<(u16, String)> = match (seen.method.as_str(), path.as_str()) {
                    ("POST", "/v1/messages/batches") => {
                        assert!(!state.created, "a second create was sent");
                        state.created = true;
                        let body: Value = serde_json::from_slice(&seen.body).unwrap();
                        state.requests = body["requests"].as_array().unwrap().clone();
                        (mode != Mode::CreateDisconnects)
                            .then(|| (200, batch_object(&state, false).to_string()))
                    }
                    ("GET", "/v1/messages/batches") => Some((
                        200,
                        json!({"data":[batch_object(&state, true)],"has_more":false,
                            "first_id":BATCH,"last_id":BATCH})
                        .to_string(),
                    )),
                    ("GET", p) if p == format!("/v1/messages/batches/{BATCH}") => {
                        if mode == Mode::StatusDown {
                            Some((503, r#"{"type":"error"}"#.into()))
                        } else {
                            Some((200, batch_object(&state, true).to_string()))
                        }
                    }
                    ("GET", p) if p == format!("/v1/messages/batches/{BATCH}/results") => {
                        let failing = failing.lock().unwrap();
                        let rows: String = state
                            .requests
                            .iter()
                            .rev()
                            .map(|request| {
                                let id = request["custom_id"].as_str().unwrap();
                                let errored =
                                    failing.iter().any(|tail| id.ends_with(tail.as_str()));
                                format!("{}\n", result_row(request, errored))
                            })
                            .collect();
                        Some((200, rows))
                    }
                    (method, path) => panic!("unexpected {method} {path}"),
                };
                drop(state);
                if let Some((status, body)) = reply {
                    let _ = write!(
                        stream,
                        "HTTP/1.1 {status} Fixture\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                }
            }
        });
        Self {
            base,
            state,
            mode,
            errored,
            stop,
            thread: Some(thread),
        }
    }
    fn set(&self, mode: Mode) {
        *self.mode.lock().unwrap() = mode;
    }
    fn seen(&self) -> Vec<Seen> {
        self.state.lock().unwrap().seen.clone()
    }
}
impl Drop for Fake {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        let _ = TcpStream::connect(
            self.base
                .trim_start_matches("http://")
                .trim_end_matches("/v1"),
        );
        if let Some(thread) = self.thread.take() {
            let result = thread.join();
            if !std::thread::panicking() {
                result.unwrap();
            }
        }
    }
}

struct Run {
    root: tempfile::TempDir,
}
impl Run {
    fn new(fake: &Fake, on_failure: &str) -> Self {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir(root.path().join("inputs")).unwrap();
        for (name, text) in [
            (
                "alpha.md",
                "# Alpha\n\nSee [the source](https://example.test/a).\n\n```rust\nlet value = 7;\n```\n",
            ),
            (
                "beta.md",
                "# Beta\n\nPlain prose about the second fixture.\n",
            ),
        ] {
            std::fs::write(root.path().join("inputs").join(name), text).unwrap();
        }
        let config = json!({"llm":{"enabled":true,"on_failure":on_failure,"model_list":[{"model_name":"default",
            "litellm_params":{"model":"anthropic/claude-haiku-5-5","api_key":"fixture-anthropic-key","api_base":fake.base}}]},
            "cache":{"enabled":false},"history":{"record":false}});
        std::fs::write(
            root.path().join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        Self { root }
    }
    fn path(&self, relative: &str) -> PathBuf {
        self.root.path().join(relative)
    }
    fn markitai(&self, args: &[&str]) -> Output {
        let mut command = Command::new(env!("CARGO_BIN_EXE_markitai"));
        console::english(command.env_clear());
        for key in ["PATH", "SYSTEMROOT", "TMPDIR", "TMP", "TEMP"] {
            if let Some(value) = std::env::var_os(key) {
                command.env(key, value);
            }
        }
        command
            .current_dir(self.root.path())
            .env("HOME", self.path("home"))
            .env("MARKITAI_HOME", self.path("markitai-home"))
            .env("NO_PROXY", "127.0.0.1,localhost")
            .env("NO_COLOR", "1")
            .stdin(Stdio::null())
            .args(args)
            .args(["-c", "config.json"])
            .output()
            .unwrap()
    }
    fn enhanced(&self) -> Vec<(String, String)> {
        let mut found = Vec::new();
        for entry in std::fs::read_dir(self.path("out")).unwrap() {
            let path = entry.unwrap().path();
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            if name.ends_with(".llm.md") {
                found.push((name, std::fs::read_to_string(&path).unwrap()));
            }
        }
        found.sort();
        found
    }
}

fn text(output: &Output) -> String {
    format!(
        "status={:?}\nstdout={}\nstderr={}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

fn check_enhanced(run: &Run) {
    let enhanced = run.enhanced();
    assert_eq!(enhanced.len(), 2, "{enhanced:?}");
    let alpha = &enhanced[0].1;
    assert!(alpha.contains("description: Batch fixture"), "{alpha}");
    assert!(alpha.contains("let value = 7;"), "{alpha}");
    assert!(alpha.contains("https://example.test/a"), "{alpha}");
    assert!(!alpha.contains("⟦MKTI"), "{alpha}");
}

#[test]
fn anthropic_batch_submits_inline_waits_and_collects() {
    let fake = Fake::start();
    let run = Run::new(&fake, "fallback");
    let output = run.markitai(&["inputs", "--llm", "--llm-batch", "-o", "out", "--json"]);
    assert_eq!(output.status.code(), Some(0), "{}", text(&output));
    check_enhanced(&run);
    let envelope: Value = serde_json::from_slice(&output.stdout).unwrap();
    let items = envelope["items"].as_array().unwrap();
    assert_eq!(items.len(), 2);
    for item in items {
        assert_eq!(item["status"], "completed", "{item}");
    }
    let seen = fake.seen();
    let create = seen.iter().find(|seen| seen.method == "POST").unwrap();
    let headers = create.headers.to_ascii_lowercase();
    assert!(
        headers.contains("x-api-key: fixture-anthropic-key"),
        "{headers}"
    );
    assert!(
        headers.contains("anthropic-version: 2023-06-01"),
        "{headers}"
    );
    assert!(!headers.contains("authorization:"), "{headers}");
    let body: Value = serde_json::from_slice(&create.body).unwrap();
    let requests = body["requests"].as_array().unwrap();
    assert_eq!(requests.len(), 2);
    for request in requests {
        let id = request["custom_id"].as_str().unwrap();
        assert!(id.len() <= 64 && id.contains('-'), "{id}");
        assert_eq!(request["params"]["model"], "claude-haiku-5-5");
        // Without a configured cap, the model's output limit.
        assert_eq!(request["params"]["max_tokens"], 128_000);
    }
    // One create, status and results reads; the results_url was not followed.
    assert_eq!(seen.iter().filter(|seen| seen.method == "POST").count(), 1);
    assert!(
        seen.iter()
            .all(|seen| seen.path.starts_with("/v1/messages/batches"))
    );
    // Observed usage is recorded per result. A loopback api_base has no
    // reviewed tariff, so its cost stays unknown rather than guessed (the
    // batch price itself is covered by the core plan tests).
    let saved = std::fs::read_dir(run.path("out/.markitai/provider-batches"))
        .unwrap()
        .filter_map(|entry| {
            let path = entry.unwrap().path().join("state.json");
            path.is_file()
                .then(|| std::fs::read_to_string(path).unwrap())
        })
        .collect::<Vec<_>>();
    assert_eq!(saved.len(), 1);
    let state: Value = serde_json::from_str(&saved[0]).unwrap();
    assert_eq!(state["phase"], "collected");
    assert_eq!(state["endpoint"]["provider"], "anthropic");
    assert_eq!(state["batch"]["id"], BATCH);
    let usage: Vec<_> = state["items"]
        .as_array()
        .unwrap()
        .iter()
        .map(|item| &item["result"]["usage"])
        .collect();
    assert!(
        usage.iter().all(|usage| usage["requests"] == 1
            && usage["input_tokens"] == 1000
            && usage["output_tokens"] == 100),
        "{usage:?}"
    );
    assert!(!saved[0].contains("fixture-anthropic-key"));
}

#[test]
fn a_pending_anthropic_batch_is_handed_back_and_collected_later_by_id_or_resume() {
    let fake = Fake::start();
    fake.set(Mode::StatusDown);
    let run = Run::new(&fake, "fallback");
    let output = run.markitai(&["inputs", "--llm", "--llm-batch", "-o", "out"]);
    assert_eq!(output.status.code(), Some(2), "{}", text(&output));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains(&format!("markitai --llm-batch-collect '{BATCH}' -o")),
        "{stderr}"
    );
    assert!(run.enhanced().is_empty());
    fake.set(Mode::Normal);
    let output = run.markitai(&["--llm-batch-collect", BATCH, "-o", "out"]);
    assert_eq!(output.status.code(), Some(0), "{}", text(&output));
    check_enhanced(&run);
    let again = run.markitai(&["--llm-batch-collect", BATCH, "-o", "out"]);
    assert_eq!(again.status.code(), Some(0), "{}", text(&again));
    assert!(String::from_utf8_lossy(&again.stderr).contains("already collected"));

    // The same handoff, finished by --resume instead of the ID.
    let fake = Fake::start();
    fake.set(Mode::StatusDown);
    let run = Run::new(&fake, "fallback");
    let output = run.markitai(&["inputs", "--llm", "--llm-batch", "-o", "out"]);
    assert_eq!(output.status.code(), Some(2), "{}", text(&output));
    fake.set(Mode::Normal);
    let output = run.markitai(&["--llm-batch", "--resume", "-o", "out"]);
    assert_eq!(output.status.code(), Some(0), "{}", text(&output));
    check_enhanced(&run);
    assert_eq!(
        fake.seen()
            .iter()
            .filter(|seen| seen.method == "POST")
            .count(),
        1
    );
}

#[test]
fn an_uncertain_anthropic_creation_is_proven_by_its_results_before_collection() {
    let fake = Fake::start();
    fake.set(Mode::CreateDisconnects);
    let run = Run::new(&fake, "fallback");
    let output = run.markitai(&["inputs", "--llm", "--llm-batch", "-o", "out"]);
    assert_ne!(output.status.code(), Some(0), "{}", text(&output));
    assert!(run.enhanced().is_empty());
    fake.set(Mode::Normal);
    let output = run.markitai(&["--llm-batch", "--resume", "-o", "out"]);
    assert_eq!(output.status.code(), Some(0), "{}", text(&output));
    check_enhanced(&run);
    let seen = fake.seen();
    // Never a second create: the listed batch was proven by a result's custom ID.
    assert_eq!(seen.iter().filter(|seen| seen.method == "POST").count(), 1);
    assert!(
        seen.iter()
            .any(|seen| seen.method == "GET" && seen.path.starts_with("/v1/messages/batches?"))
    );
}

#[test]
fn an_errored_request_keeps_the_base_under_fallback_and_fails_under_fail() {
    for (policy, code) in [("fallback", 0), ("fail", 10)] {
        let fake = Fake::start();
        // The second request (index 1) is errored by the provider.
        fake.errored.lock().unwrap().push("-1".into());
        let run = Run::new(&fake, policy);
        let output = run.markitai(&["inputs", "--llm", "--llm-batch", "-o", "out", "--json"]);
        assert_eq!(output.status.code(), Some(code), "{}", text(&output));
        let enhanced = run.enhanced();
        assert_eq!(enhanced.len(), 1, "{policy}: {enhanced:?}");
        let envelope: Value = serde_json::from_slice(&output.stdout).unwrap();
        let statuses: Vec<_> = envelope["items"]
            .as_array()
            .unwrap()
            .iter()
            .map(|item| item["status"].as_str().unwrap().to_owned())
            .collect();
        if policy == "fail" {
            assert!(statuses.contains(&"failed".to_owned()), "{statuses:?}");
        } else {
            assert!(
                statuses.iter().all(|status| status == "completed"),
                "{statuses:?}"
            );
        }
    }
}

#[cfg(test)]
mod bounded_fixture_io {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/support/bounded_read.rs"
    ));
}
