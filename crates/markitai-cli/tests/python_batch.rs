//! Batches Python markitai 1.3.0 left pending (real state directories in
//! `tests/fixtures/python-batch`) are collected after upgrading: by ID or by
//! `--llm-batch --resume`, from a loopback provider that answers the Python
//! requests. Python's files are never modified.
use crate::console;

use serde_json::{Value, json};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;

const OPENAI_BATCH: &str = "batch_stubpythonstate0001";
const ANTHROPIC_BATCH: &str = "msgbatch_01StubPythonState00000";

fn fixture(case: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/python-batch")
        .join(case)
}

fn copy_tree(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_tree(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), &target).unwrap();
        }
    }
}

fn digest_tree(root: &Path) -> Vec<(String, Vec<u8>)> {
    let mut files = Vec::new();
    let mut stack = vec![root.to_owned()];
    while let Some(directory) = stack.pop() {
        for entry in std::fs::read_dir(&directory).unwrap() {
            let entry = entry.unwrap();
            if entry.file_type().unwrap().is_dir() {
                stack.push(entry.path());
            } else {
                files.push((
                    entry
                        .path()
                        .strip_prefix(root)
                        .unwrap()
                        .display()
                        .to_string(),
                    std::fs::read(entry.path()).unwrap(),
                ));
            }
        }
    }
    files.sort();
    files
}

/// How the fake answers each Python request.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Answers {
    /// Keep the protected document text, as a compliant model does.
    Faithful,
    /// Drop the image placeholder of the deck.
    DropImage,
}

/// The document text a Python request sent: everything after its `---`.
fn protected_text(body: &Value) -> String {
    let user = body["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|message| message["role"] == "user")
        .unwrap();
    let text = user["content"].as_str().unwrap();
    text.rsplit_once("---\n\n").unwrap().1.trim().to_owned()
}

fn answer(body: &Value, answers: Answers) -> Value {
    let tool = body["tool_choice"]["function"]["name"]
        .as_str()
        .or_else(|| body["tool_choice"]["name"].as_str())
        .unwrap();
    if tool == "ImageAnalysisResult" {
        return json!({"caption":"Quarterly chart","description":"A framed chart titled Quarterly chart.","extracted_text":"Quarterly chart"});
    }
    let mut text = protected_text(body);
    if answers == Answers::DropImage {
        text = text.replace("__MARKITAI_IMG_0__", "");
    }
    json!({"cleaned_markdown":text,"frontmatter":{"description":"Imported from Python","tags":["python","batch"]}})
}

fn requests(case: &str, batch: &str) -> Vec<Value> {
    std::fs::read_to_string(
        fixture(case)
            .join(format!(".markitai/batch-{batch}"))
            .join("requests.jsonl"),
    )
    .unwrap()
    .lines()
    .map(|line| serde_json::from_str(line).unwrap())
    .collect()
}

struct Fake {
    base: String,
    seen: Arc<Mutex<Vec<String>>>,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

fn read_request(stream: &mut TcpStream) -> Option<(String, String)> {
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
        let mut chunk = [0; 4096];
        let count = reader.read(&mut chunk).ok()?;
        if count == 0 {
            return None;
        }
        bytes.extend_from_slice(&chunk[..count]);
        if let Some(split) = bytes.windows(4).position(|s| s == b"\r\n\r\n") {
            let headers = String::from_utf8_lossy(&bytes[..split]).into_owned();
            let mut first = headers.lines().next()?.split_whitespace();
            return Some((first.next()?.to_owned(), first.next()?.to_owned()));
        }
    }
}

impl Fake {
    fn start(case: &'static str, answers: Answers) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let address = listener.local_addr().unwrap();
        let base = format!("http://{address}/v1");
        let seen = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let (log, stopped) = (seen.clone(), stop.clone());
        let thread = std::thread::spawn(move || {
            let (batch, rows) = if case == "openai" {
                (OPENAI_BATCH, requests("openai", OPENAI_BATCH))
            } else {
                (ANTHROPIC_BATCH, requests("anthropic", ANTHROPIC_BATCH))
            };
            while !stopped.load(Ordering::SeqCst) {
                let mut stream = match listener.accept() {
                    Ok((stream, _)) => stream,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(2));
                        continue;
                    }
                    Err(error) => panic!("fixture accept: {error}"),
                };
                let Some((method, path)) = read_request(&mut stream) else {
                    continue;
                };
                log.lock().unwrap().push(format!("{method} {path}"));
                assert_eq!(method, "GET", "collection only reads: {path}");
                let total = rows.len();
                let body = if path == format!("/v1/batches/{batch}") {
                    json!({"id":batch,"object":"batch","endpoint":"/v1/chat/completions","input_file_id":"file-stub0001",
                        "status":"completed","output_file_id":"file-output0001","error_file_id":null,
                        "request_counts":{"total":total,"completed":total,"failed":0}})
                    .to_string()
                } else if path == "/v1/files/file-output0001/content" {
                    rows.iter()
                        .map(|row| {
                            let body = &row["body"];
                            let tool = body["tool_choice"]["function"]["name"].clone();
                            let arguments = answer(body, answers).to_string();
                            format!("{}\n", json!({"custom_id":row["custom_id"],"response":{"status_code":200,"request_id":"req_fixture",
                                "body":{"model":body["model"],"choices":[{"finish_reason":"tool_calls","message":{"role":"assistant",
                                "tool_calls":[{"id":"call_fixture","type":"function","function":{"name":tool,"arguments":arguments}}]}}],
                                "usage":{"prompt_tokens":900,"completion_tokens":90}}},"error":null}))
                        })
                        .collect()
                } else if path == format!("/v1/messages/batches/{batch}") {
                    json!({"id":batch,"type":"message_batch","processing_status":"ended",
                        "request_counts":{"processing":0,"succeeded":total,"errored":0,"canceled":0,"expired":0},
                        "ended_at":"2026-10-11T09:00:00Z","created_at":"2026-10-11T08:00:00Z","expires_at":"2026-10-12T08:00:00Z",
                        "archived_at":null,"cancel_initiated_at":null,"results_url":null})
                    .to_string()
                } else if path == format!("/v1/messages/batches/{batch}/results") {
                    rows.iter()
                        .map(|row| {
                            let params = &row["params"];
                            format!("{}\n", json!({"custom_id":row["custom_id"],"result":{"type":"succeeded","message":{
                                "id":"msg_fixture","type":"message","role":"assistant","model":params["model"],
                                "content":[{"type":"tool_use","id":"toolu_fixture","name":params["tool_choice"]["name"],"input":answer(params, answers)}],
                                "stop_reason":"tool_use","usage":{"input_tokens":900,"output_tokens":90}}}}))
                        })
                        .collect()
                } else {
                    panic!("unexpected {method} {path}");
                };
                let _ = write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
            }
        });
        Self {
            base,
            seen,
            stop,
            thread: Some(thread),
        }
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
    fn new(case: &str, model: &str, fake: &Fake, on_failure: &str) -> Self {
        let root = tempfile::tempdir().unwrap();
        copy_tree(&fixture(case), &root.path().join("out"));
        let config = json!({"llm":{"enabled":true,"on_failure":on_failure,"model_list":[{"model_name":"default",
            "litellm_params":{"model":model,"api_key":"fixture-key","api_base":fake.base}}]},
            "cache":{"enabled":false},"history":{"record":false}});
        std::fs::write(
            root.path().join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        Self { root }
    }
    fn out(&self) -> PathBuf {
        self.root.path().join("out")
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
            .env("HOME", self.root.path().join("home"))
            .env("MARKITAI_HOME", self.root.path().join("markitai-home"))
            .env("NO_PROXY", "127.0.0.1,localhost")
            .env("NO_COLOR", "1")
            .stdin(Stdio::null())
            .args(args)
            .args(["-c", "config.json"])
            .output()
            .unwrap()
    }
    fn read(&self, name: &str) -> String {
        std::fs::read_to_string(self.out().join(name)).unwrap()
    }
}

fn report(output: &Output) -> String {
    format!(
        "status={:?}\nstdout={}\nstderr={}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

#[test]
fn an_openai_python_batch_is_collected_by_id_with_its_image_alt_text() {
    let fake = Fake::start("openai", Answers::Faithful);
    let run = Run::new("openai", "openai/gpt-5.6-luna", &fake, "fallback");
    let state = run.out().join(format!(".markitai/batch-{OPENAI_BATCH}"));
    let before = digest_tree(&state);
    let bases = [run.read("deck.pptx.md"), run.read("notes.md.md")];
    let output = run.markitai(&["--llm-batch-collect", OPENAI_BATCH, "-o", "out"]);
    assert_eq!(output.status.code(), Some(0), "{}", report(&output));
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("Imported Python markitai batch"),
        "{}",
        report(&output)
    );
    let deck = run.read("deck.pptx.llm.md");
    assert!(
        deck.starts_with("---\ntitle: Batch import deck\nsource: deck.pptx\n"),
        "{deck}"
    );
    assert!(deck.contains("description: Imported from Python"), "{deck}");
    // The analyzed image gets its caption as alt text, in place.
    assert!(
        deck.contains("![Quarterly chart](.markitai/assets/deck.pptx.0001.jpg)"),
        "{deck}"
    );
    assert!(deck.contains("Revenue grew in every region."), "{deck}");
    assert!(!deck.contains("__MARKITAI_"), "{deck}");
    let notes = run.read("notes.md.llm.md");
    assert!(
        notes.contains("[the changelog](https://example.test/changelog)"),
        "{notes}"
    );
    assert!(notes.contains("print(\"unchanged code\")"), "{notes}");
    // Python's state and the base Markdown keep their bytes.
    assert_eq!(digest_tree(&state), before);
    assert_eq!([run.read("deck.pptx.md"), run.read("notes.md.md")], bases);
    // Only reads: the status, then the one results file.
    let seen = fake.seen.lock().unwrap().clone();
    assert!(seen.iter().all(|line| line.starts_with("GET ")), "{seen:?}");
    let again = run.markitai(&["--llm-batch-collect", OPENAI_BATCH, "-o", "out"]);
    assert_eq!(again.status.code(), Some(0), "{}", report(&again));
    assert!(String::from_utf8_lossy(&again.stderr).contains("already collected"));
}

#[test]
fn an_anthropic_python_batch_is_found_and_collected_by_resume() {
    let fake = Fake::start("anthropic", Answers::Faithful);
    let run = Run::new("anthropic", "anthropic/claude-haiku-5-5", &fake, "fallback");
    let output = run.markitai(&["--llm-batch", "--resume", "-o", "out", "--json"]);
    assert_eq!(output.status.code(), Some(0), "{}", report(&output));
    let envelope: Value = serde_json::from_slice(&output.stdout).unwrap();
    let items = envelope["items"].as_array().unwrap();
    assert_eq!(items.len(), 2, "{envelope}");
    assert!(
        items.iter().all(|item| item["status"] == "completed"),
        "{envelope}"
    );
    let deck = run.read("deck.pptx.llm.md");
    assert!(deck.contains("tags:\n- python\n- batch"), "{deck}");
    // Without --alt in the Python run, the alt text stays as converted.
    assert!(
        deck.contains("![chart.png](.markitai/assets/deck.pptx.0001.jpg)"),
        "{deck}"
    );
    // Nothing left to resume: the imported job is collected.
    let again = run.markitai(&["--llm-batch", "--resume", "-o", "out"]);
    assert_ne!(again.status.code(), Some(0), "{}", report(&again));
    assert!(String::from_utf8_lossy(&again.stderr).contains("No unfinished"));
}

#[test]
fn a_python_batch_needs_its_model_and_a_dropped_placeholder_keeps_the_base() {
    let fake = Fake::start("anthropic", Answers::Faithful);
    let run = Run::new("anthropic", "anthropic/claude-opus-5-5", &fake, "fallback");
    let output = run.markitai(&["--llm-batch-collect", ANTHROPIC_BATCH, "-o", "out"]);
    assert_eq!(output.status.code(), Some(1), "{}", report(&output));
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("anthropic/claude-haiku-5-5"),
        "{}",
        report(&output)
    );
    assert!(fake.seen.lock().unwrap().is_empty());

    for (policy, code) in [("fallback", 0), ("fail", 10)] {
        let fake = Fake::start("openai", Answers::DropImage);
        let run = Run::new("openai", "openai/gpt-5.6-luna", &fake, policy);
        let output = run.markitai(&["--llm-batch-collect", OPENAI_BATCH, "-o", "out"]);
        assert_eq!(
            output.status.code(),
            Some(code),
            "{policy}: {}",
            report(&output)
        );
        assert!(!run.out().join("deck.pptx.llm.md").exists(), "{policy}");
        assert!(run.out().join("notes.md.llm.md").exists(), "{policy}");
        if policy == "fallback" {
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                stderr.contains(
                    "deck.pptx: Provider Batch enhancement failed; base Markdown retained"
                ),
                "{stderr}"
            );
        }
        // Under `fail` the run report names why the deck has no enhancement.
        if policy == "fail" {
            let reports = run.out().join(".markitai/reports");
            let report_text: String = std::fs::read_dir(reports)
                .unwrap()
                .map(|entry| std::fs::read_to_string(entry.unwrap().path()).unwrap())
                .collect();
            assert!(
                report_text.contains("dropped 1 protected image, page or slide placeholder"),
                "{policy}: {report_text}"
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
