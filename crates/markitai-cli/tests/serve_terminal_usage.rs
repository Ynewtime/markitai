#![cfg(unix)]

use serde_json::{Value, json};
use std::{
    io::{BufRead, BufReader, Read, Write},
    net::{TcpListener, TcpStream},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

const WAIT: Duration = Duration::from_secs(30);
fn until(label: &str, mut condition: impl FnMut() -> bool) {
    let deadline = Instant::now() + WAIT;
    while !condition() {
        assert!(Instant::now() < deadline, "timed out: {label}");
        thread::sleep(Duration::from_millis(5));
    }
}

fn connect(
    port: u16,
    method: &str,
    path: &str,
    content_type: &str,
    body: &[u8],
) -> BufReader<TcpStream> {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
    stream.set_read_timeout(Some(WAIT)).unwrap();
    stream.set_write_timeout(Some(WAIT)).unwrap();
    write!(stream, "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n\r\n", body.len()).unwrap();
    stream.write_all(body).unwrap();
    BufReader::new(stream)
}
fn headers(reader: &mut impl BufRead) -> (u16, bool) {
    let mut line = String::new();
    reader.read_line(&mut line).unwrap();
    let status = line.split_whitespace().nth(1).unwrap().parse().unwrap();
    let mut size = line.len();
    let mut chunked = false;
    loop {
        line.clear();
        assert!(reader.read_line(&mut line).unwrap() > 0);
        size += line.len();
        assert!(size < 16384);
        if line == "\r\n" {
            break;
        }
        if let Some((name, value)) = line.split_once(':') {
            chunked |= name.eq_ignore_ascii_case("transfer-encoding") && value.trim() == "chunked";
        }
    }
    (status, chunked)
}
fn response_body(mut reader: impl Read, chunked: bool) -> Vec<u8> {
    let mut raw = Vec::new();
    reader
        .by_ref()
        .take(4 * 1024 * 1024)
        .read_to_end(&mut raw)
        .unwrap();
    assert!(raw.len() < 4 * 1024 * 1024);
    if !chunked {
        return raw;
    }
    let mut decoded = Vec::new();
    let mut at = 0;
    loop {
        let end = at
            + raw[at..]
                .windows(2)
                .position(|bytes| bytes == b"\r\n")
                .unwrap();
        let length = usize::from_str_radix(
            std::str::from_utf8(&raw[at..end])
                .unwrap()
                .split(';')
                .next()
                .unwrap(),
            16,
        )
        .unwrap();
        at = end + 2;
        if length == 0 {
            break;
        }
        decoded.extend_from_slice(&raw[at..at + length]);
        assert_eq!(&raw[at + length..at + length + 2], b"\r\n");
        at += length + 2;
    }
    decoded
}

struct Service {
    child: Option<Child>,
    reader: Option<JoinHandle<()>>,
    logs: Arc<Mutex<String>>,
    port: u16,
    root: PathBuf,
}
impl Service {
    fn start(root: &Path) -> Self {
        let mut command = Command::new(env!("CARGO_BIN_EXE_markitai"));
        command.env_clear();
        for name in ["PATH", "HOME", "TMPDIR", "TEMP", "TMP", "SYSTEMROOT"] {
            if let Some(value) = std::env::var_os(name) {
                command.env(name, value);
            }
        }
        let mut child = command
            .current_dir(root)
            .env("MARKITAI_HOME", root.join("home"))
            .env("NO_PROXY", "127.0.0.1,localhost")
            .args([
                "--config",
                "config.json",
                "serve",
                "--host",
                "127.0.0.1",
                "--port",
                "0",
                "--no-open",
                "--no-auth",
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let logs = Arc::new(Mutex::new(String::new()));
        let copy = logs.clone();
        let stderr = child.stderr.take().unwrap();
        let reader = thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                let mut log = copy.lock().unwrap();
                log.push_str(&line);
                log.push('\n');
            }
        });
        let mut service = Self {
            child: Some(child),
            reader: Some(reader),
            logs,
            port: 0,
            root: root.into(),
        };
        until("service startup", || {
            assert!(
                service
                    .child
                    .as_mut()
                    .unwrap()
                    .try_wait()
                    .unwrap()
                    .is_none(),
                "{}",
                service.logs.lock().unwrap()
            );
            let port = service.logs.lock().unwrap().lines().find_map(|line| {
                line.strip_prefix("Markitai server listening on http://127.0.0.1:")
                    .and_then(|value| value.parse().ok())
            });
            if let Some(port) = port {
                service.port = port;
                true
            } else {
                false
            }
        });
        service
    }
    fn request(
        &self,
        method: &str,
        path: &str,
        content_type: &str,
        body: &[u8],
        expected: u16,
    ) -> Value {
        let mut reader = connect(self.port, method, path, content_type, body);
        let (status, chunked) = headers(&mut reader);
        let raw = response_body(reader, chunked);
        assert_eq!(status, expected, "{}", String::from_utf8_lossy(&raw));
        serde_json::from_slice(&raw).unwrap()
    }
    fn get(&self, id: &str) -> Value {
        self.request(
            "GET",
            &format!("/api/jobs/{id}"),
            "application/json",
            &[],
            200,
        )
    }
    fn submit(&self, llm: bool) -> String {
        let body = format!(
            "--fixture\r\nContent-Disposition: form-data; name=\"files\"; filename=\"source.txt\"\r\nContent-Type: text/plain\r\n\r\nAuthored original body with a complete tail.\r\n--fixture\r\nContent-Disposition: form-data; name=\"options\"\r\n\r\n{}\r\n--fixture--\r\n",
            json!({"llm":llm})
        );
        self.request(
            "POST",
            "/api/jobs",
            "multipart/form-data; boundary=fixture",
            body.as_bytes(),
            201,
        )["job_id"]
            .as_str()
            .unwrap()
            .into()
    }
    fn retry(&self, id: &str, operation: &str) {
        let body = json!({"operation":operation,"options":{"llm":true}}).to_string();
        self.request(
            "POST",
            &format!("/api/jobs/{id}/items/i1/retry"),
            "application/json",
            body.as_bytes(),
            202,
        );
    }
    fn done(&self, id: &str) -> Value {
        let mut result = Value::Null;
        until("terminal job", || {
            result = self.get(id);
            result["status"] != "running"
        });
        assert_eq!(result["status"], "done", "{result}");
        result
    }
    fn events(&self, id: &str) -> JoinHandle<Vec<Value>> {
        let mut reader = connect(
            self.port,
            "GET",
            &format!("/api/jobs/{id}/events"),
            "application/json",
            &[],
        );
        let (status, chunked) = headers(&mut reader);
        assert_eq!(status, 200);
        thread::spawn(move || {
            String::from_utf8(response_body(reader, chunked))
                .unwrap()
                .lines()
                .filter_map(|line| line.strip_prefix("data:"))
                .map(|line| serde_json::from_str(line.trim()).unwrap())
                .collect()
        })
    }
    fn folder(&self, id: &str) -> PathBuf {
        self.root.join("home/serve/jobs").join(id)
    }
    fn stop(mut self) {
        let child = self.child.as_mut().unwrap();
        assert!(
            Command::new("/bin/kill")
                .args(["-INT", &child.id().to_string()])
                .status()
                .unwrap()
                .success()
        );
        until("service shutdown", || child.try_wait().unwrap().is_some());
        let status = child.wait().unwrap();
        self.child.take();
        self.reader.take().unwrap().join().unwrap();
        assert!(status.success(), "{status}: {}", self.logs.lock().unwrap());
    }
}
impl Drop for Service {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
    }
}

#[derive(Clone, Copy)]
struct Answer {
    status: u16,
    tokens: Option<(u64, u64)>,
}
struct Gate {
    answer: Answer,
    entered: usize,
    released: usize,
}
struct Model {
    port: u16,
    gate: Arc<(Mutex<Gate>, Condvar)>,
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
}
impl Model {
    fn start(answer: Answer) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        listener.set_nonblocking(true).unwrap();
        let gate = Arc::new((
            Mutex::new(Gate {
                answer,
                entered: 0,
                released: 0,
            }),
            Condvar::new(),
        ));
        let stop = Arc::new(AtomicBool::new(false));
        let (shared, stopping) = (gate.clone(), stop.clone());
        let worker = thread::spawn(move || {
            let mut workers = Vec::new();
            while !stopping.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        stream.set_nonblocking(false).unwrap();
                        stream.set_read_timeout(Some(WAIT)).unwrap();
                        stream.set_write_timeout(Some(WAIT)).unwrap();
                        let gate = shared.clone();
                        workers.push(thread::spawn(move || {
                            let mut reader = BufReader::new(stream.try_clone().unwrap());
                            let mut line = String::new();
                            reader.read_line(&mut line).unwrap();
                            assert_eq!(line.trim(), "POST /v1/chat/completions HTTP/1.1");
                            let mut length = 0;
                            let mut size = line.len();
                            loop {
                                line.clear(); assert!(reader.read_line(&mut line).unwrap() > 0);
                                size += line.len(); assert!(size < 16384);
                                if line == "\r\n" { break; }
                                if let Some((name, value)) = line.split_once(':')
                                    && name.eq_ignore_ascii_case("content-length") { length = value.trim().parse::<usize>().unwrap(); }
                            }
                            assert!(length > 0 && length < 1024 * 1024);
                            let mut body = vec![0;length]; reader.read_exact(&mut body).unwrap();
                            let request: Value = serde_json::from_slice(&body).unwrap();
                            assert_eq!(request["model"], "terminal-fixture");
                            let (lock, wake) = &*gate;
                            let mut state = lock.lock().unwrap();
                            let index = state.entered;
                            let answer = state.answer;
                            state.entered += 1; wake.notify_all();
                            let (state, timeout) = wake.wait_timeout_while(state, WAIT, |state| state.released <= index).unwrap();
                            assert!(!timeout.timed_out(), "model gate timed out");
                            drop(state);
                            let mut response = if answer.status == 200 {
                                assert!(request["messages"][0]["content"].as_str().unwrap().contains("MARKITAI_DOCUMENT_JSON_V1"));
                                let text = request["messages"].as_array().unwrap().iter().find(|message| message["role"] == "user").unwrap()["content"].as_str().unwrap();
                                let content = json!({"cleaned_markdown":text,"frontmatter":{"description":"Authored complete source","tags":["fixture"]}}).to_string();
                                json!({"model":"terminal-fixture","choices":[{"message":{"content":content},"finish_reason":"stop"}]})
                            } else {
                                json!({"error":{"message":"Incorrect API key provided: PRIVATE_LOOPBACK_KEY", "code":"invalid_api_key", "type":"invalid_api_key"}})
                            };
                            if let Some((input, output)) = answer.tokens { response["usage"] = json!({"prompt_tokens":input,"completion_tokens":output}); }
                            let encoded = response.to_string();
                            write!(stream, "HTTP/1.1 {} Fixture\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{encoded}", answer.status, encoded.len()).unwrap();
                        }));
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(5))
                    }
                    Err(error) => panic!("fixture accept: {error}"),
                }
            }
            for worker in workers {
                worker.join().unwrap();
            }
        });
        Self {
            port,
            gate,
            stop,
            worker: Some(worker),
        }
    }
    fn configure(&self, root: &Path) {
        std::fs::write(root.join("config.json"), json!({
            "llm":{"enabled":false,"keep_base":true,"on_failure":"fail","model_list":[{"model_name":"default","litellm_params":{"model":"openai/terminal-fixture","api_key":"PRIVATE_LOOPBACK_KEY","api_base":format!("http://127.0.0.1:{}/v1",self.port)}}],"router_settings":{"num_retries":0,"timeout":35}},
            "cache":{"enabled":false},"history":{"record":false},"ocr":{"enabled":false},"screenshot":{"enabled":false},"image":{"alt_enabled":false,"desc_enabled":false},"prompts":{"dir":root.join("prompts")},"log":{"dir":null},"fetch":{"strategy":"static","remote_consent":"never"},"output":{"report":false}
        }).to_string()).unwrap();
    }
    fn wait(&self, expected: usize) {
        until("model call", || {
            self.gate.0.lock().unwrap().entered == expected
        });
    }
    fn release(&self) {
        let (lock, wake) = &*self.gate;
        let mut state = lock.lock().unwrap();
        state.released = state.entered;
        wake.notify_all();
    }
    fn answer(&self, answer: Answer) {
        self.gate.0.lock().unwrap().answer = answer;
    }
    fn count(&self) -> usize {
        self.gate.0.lock().unwrap().entered
    }
}
impl Drop for Model {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        {
            let mut state = self.gate.0.lock().unwrap();
            state.released = usize::MAX;
            self.gate.1.notify_all();
        }
        if let Some(worker) = self.worker.take() {
            let result = worker.join();
            if !thread::panicking() {
                result.unwrap();
            }
        }
    }
}

fn observation(item: &Value, operation: &str, status: &str, input: u64, output: u64) -> Value {
    let value = item["diagnostics"].clone();
    let attempt = &value["last_attempt"];
    assert_eq!(attempt["operation"], operation);
    assert_eq!(attempt["status"], status);
    assert_eq!(attempt["usage"]["requests"], 1);
    assert_eq!(attempt["usage"]["input_tokens"], input);
    assert_eq!(attempt["usage"]["output_tokens"], output);
    assert_eq!(attempt["usage"]["by_model"].as_object().unwrap().len(), 1);
    assert_eq!(attempt["error"].is_null(), status == "done");
    assert!(!value.to_string().contains("PRIVATE_"));
    value
}
fn without_diagnostics(item: &Value) -> Value {
    let mut item = item.clone();
    item.as_object_mut().unwrap().remove("diagnostics");
    item
}
fn without_attempt_fields(item: &Value) -> Value {
    let mut item = without_diagnostics(item);
    item.as_object_mut().unwrap().remove("rerun_failure");
    item
}
fn retained_failure(item: &Value, operation: &str, code: &str) -> Value {
    let failure = item["rerun_failure"].clone();
    assert_eq!(failure["operation"], operation);
    assert_eq!(failure["error_code"], code);
    assert!(!failure["error"].as_str().unwrap().is_empty());
    chrono::DateTime::parse_from_rfc3339(failure["failed_at"].as_str().unwrap()).unwrap();
    assert!(!failure.to_string().contains("PRIVATE_"));
    failure
}
fn saved(folder: &Path) -> Value {
    serde_json::from_slice(&std::fs::read(folder.join("meta.json")).unwrap()).unwrap()
}

#[test]
fn paid_auth_failure_including_zero_tokens_survives_events_get_and_restart() {
    for (input, output) in [(0, 0), (13, 2)] {
        let model = Model::start(Answer {
            status: 401,
            tokens: Some((input, output)),
        });
        let root = tempfile::tempdir().unwrap();
        model.configure(root.path());
        let service = Service::start(root.path());
        let id = service.submit(true);
        model.wait(1);
        assert!(service.get(&id)["items"][0].get("diagnostics").is_none());
        let events = service.events(&id);
        model.release();
        let result = service.done(&id);
        let item = &result["items"][0];
        assert_eq!(item["status"], "error");
        assert!(item["error"].as_str().unwrap().contains("401"));
        let diagnostics = observation(item, "convert", "error", input, output);
        assert_eq!(diagnostics["last_attempt"]["error"], item["error"]);
        assert!(
            events
                .join()
                .unwrap()
                .iter()
                .any(|event| event["item_id"] == "i1" && event["diagnostics"] == diagnostics)
        );
        assert_eq!(saved(&service.folder(&id))["items"][0], {
            let mut expected = item.clone();
            expected["options"] = saved(&service.folder(&id))["items"][0]["options"].clone();
            expected
        });
        service.stop();
        let service = Service::start(root.path());
        assert_eq!(service.get(&id)["items"][0], *item);
        assert_eq!(model.count(), 1);
        service.stop();
    }
}

#[test]
fn post_core_publication_failure_retains_original_bytes_and_current_attempt_only() {
    let model = Model::start(Answer {
        status: 200,
        tokens: Some((7, 5)),
    });
    let root = tempfile::tempdir().unwrap();
    model.configure(root.path());
    let service = Service::start(root.path());
    let id = service.submit(false);
    let initial = service.done(&id)["items"][0].clone();
    assert!(initial.get("diagnostics").is_none());
    let folder = service.folder(&id);
    let original_path = folder.join("out").join(initial["output"].as_str().unwrap());
    let bytes = std::fs::read(&original_path).unwrap();
    let meta = saved(&folder);
    let blocked = folder.join("out").join(format!(
        "{}.llm.md",
        meta["native_bases"]["i1"].as_str().unwrap()
    ));
    std::fs::create_dir(&blocked).unwrap();
    service.retry(&id, "retry");
    model.wait(1);
    let events = service.events(&id);
    model.release();
    let failed = service.done(&id)["items"][0].clone();
    assert_eq!(without_attempt_fields(&failed), initial);
    let outcome = retained_failure(&failed, "retry", "internal_error");
    assert_eq!(outcome["error"], "internal server error");
    let diagnostics = observation(&failed, "retry", "error", 7, 5);
    assert_eq!(
        diagnostics["last_attempt"]["error"],
        "internal server error"
    );
    assert!(
        events
            .join()
            .unwrap()
            .iter()
            .any(|event| event["item_id"] == "i1"
                && event["diagnostics"] == diagnostics
                && event["rerun_failure"] == outcome)
    );
    assert_eq!(std::fs::read(&original_path).unwrap(), bytes);
    assert!(blocked.is_dir());
    service.stop();
    let service = Service::start(root.path());
    assert_eq!(service.get(&id)["items"][0], failed);
    assert_eq!(std::fs::read(&original_path).unwrap(), bytes);
    std::fs::remove_dir(&blocked).unwrap();

    model.answer(Answer {
        status: 200,
        tokens: Some((19, 3)),
    });
    service.retry(&id, "enhance");
    model.wait(2);
    assert!(service.get(&id)["items"][0].get("diagnostics").is_none());
    model.release();
    let enhanced = service.done(&id)["items"][0].clone();
    assert!(enhanced.get("rerun_failure").is_none());
    observation(&enhanced, "enhance", "done", 19, 3);
    let enhanced_path = folder
        .join("out")
        .join(enhanced["output"].as_str().unwrap());
    let enhanced_bytes = std::fs::read(&enhanced_path).unwrap();
    let enhanced_base_bytes = std::fs::read(&original_path).unwrap();

    model.answer(Answer {
        status: 401,
        tokens: None,
    });
    service.retry(&id, "enhance");
    model.wait(3);
    assert!(service.get(&id)["items"][0].get("diagnostics").is_none());
    model.release();
    let unknown = service.done(&id)["items"][0].clone();
    assert!(unknown.get("diagnostics").is_none());
    retained_failure(&unknown, "enhance", "conversion_error");
    assert_eq!(
        without_attempt_fields(&unknown),
        without_diagnostics(&enhanced)
    );
    assert_eq!(std::fs::read(&original_path).unwrap(), enhanced_base_bytes);
    assert_eq!(std::fs::read(&enhanced_path).unwrap(), enhanced_bytes);
    service.stop();
    let service = Service::start(root.path());
    assert_eq!(service.get(&id)["items"][0], unknown);
    assert_eq!(std::fs::read(&original_path).unwrap(), enhanced_base_bytes);
    assert_eq!(std::fs::read(&enhanced_path).unwrap(), enhanced_bytes);
    assert_eq!(model.count(), 3);
    service.stop();
}

#[test]
fn invalid_stored_diagnostics_do_not_erase_legacy_item_or_rewrite_history() {
    let model = Model::start(Answer {
        status: 401,
        tokens: None,
    });
    let root = tempfile::tempdir().unwrap();
    model.configure(root.path());
    let service = Service::start(root.path());
    let id = service.submit(false);
    let initial = service.done(&id)["items"][0].clone();
    let folder = service.folder(&id);
    service.stop();
    let mut metadata = saved(&folder);
    metadata["items"][0]["diagnostics"] = json!({"last_attempt":{"operation":"convert","status":"error","error":"PRIVATE_DIAGNOSTIC", "usage":{"requests":-1}}});
    let raw = serde_json::to_vec(&metadata).unwrap();
    std::fs::write(folder.join("meta.json"), &raw).unwrap();
    let service = Service::start(root.path());
    assert_eq!(service.get(&id)["items"][0], initial);
    assert_eq!(std::fs::read(folder.join("meta.json")).unwrap(), raw);
    let log = service.logs.lock().unwrap().clone();
    assert!(log.contains("ignored invalid stored attempt diagnostics"));
    assert!(!log.contains("PRIVATE_DIAGNOSTIC"));
    assert_eq!(model.count(), 0);
    service.stop();
    assert_eq!(std::fs::read(folder.join("meta.json")).unwrap(), raw);
}

#[test]
fn invalid_stored_rerun_outcomes_do_not_hide_output_or_rewrite_legacy_history() {
    let model = Model::start(Answer {
        status: 401,
        tokens: None,
    });
    let root = tempfile::tempdir().unwrap();
    model.configure(root.path());
    let service = Service::start(root.path());
    let id = service.submit(false);
    let initial = service.done(&id)["items"][0].clone();
    let folder = service.folder(&id);
    service.stop();
    let original = saved(&folder);
    let valid = json!({"operation":"enhance", "error_code":"enhancement_failed",
        "error":"PRIVATE_OUTCOME", "failed_at":"2026-10-03T00:00:00Z"});
    let mut cases = Vec::new();
    for (field, value) in [
        ("operation", json!("convert")),
        ("error_code", json!("bad code")),
        ("error", json!("")),
        ("error", json!("x".repeat(4097))),
        ("failed_at", json!("invalid date")),
        ("unexpected", json!(true)),
    ] {
        let mut invalid = valid.clone();
        invalid[field] = value;
        cases.push(invalid);
    }
    cases.push(json!({"operation":"enhance"}));
    for invalid in cases {
        let mut metadata = original.clone();
        metadata["items"][0]["rerun_failure"] = invalid;
        let raw = serde_json::to_vec(&metadata).unwrap();
        std::fs::write(folder.join("meta.json"), &raw).unwrap();
        let service = Service::start(root.path());
        assert_eq!(service.get(&id)["items"][0], initial);
        assert_eq!(std::fs::read(folder.join("meta.json")).unwrap(), raw);
        let log = service.logs.lock().unwrap().clone();
        assert!(log.contains("ignored invalid stored rerun failure"));
        assert!(!log.contains("PRIVATE_OUTCOME"));
        service.stop();
        assert_eq!(std::fs::read(folder.join("meta.json")).unwrap(), raw);
    }
    assert_eq!(model.count(), 0);
}
