use markitai_core::{
    ConversionOutput, ConvertContext, ConvertOptions, LlmRuntime, convert_with_context,
};
use serde_json::{Value, json};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{
    Arc, Condvar, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::thread;
use std::time::{Duration, Instant};

// Each public conversion gets a private configuration environment. HOME stays
// inherited, while no ambient provider credentials enter the child process.
fn isolated(name: &str) -> bool {
    if std::env::var("MARKITAI_METRIC_ROUTING_TEST").as_deref() == Ok(name) {
        return false;
    }
    let dir = tempfile::tempdir().unwrap();
    let stdout = dir.path().join("stdout");
    let stderr = dir.path().join("stderr");
    let mut command = Command::new(std::env::current_exe().unwrap());
    command
        .args(["--exact", name, "--nocapture"])
        .env_clear()
        .env("MARKITAI_METRIC_ROUTING_TEST", name)
        .env("MARKITAI_HOME", dir.path().join("state"))
        .current_dir(dir.path())
        .stdout(Stdio::from(std::fs::File::create(&stdout).unwrap()))
        .stderr(Stdio::from(std::fs::File::create(&stderr).unwrap()));
    for name in [
        "HOME",
        "PATH",
        "USERPROFILE",
        "SYSTEMROOT",
        "WINDIR",
        "LANG",
        "LC_ALL",
        "TZ",
    ] {
        if let Some(value) = std::env::var_os(name) {
            command.env(name, value);
        }
    }
    let mut child = command.spawn().unwrap();
    let deadline = Instant::now() + Duration::from_secs(60);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!("isolated routing test timed out: {name}");
        }
        thread::sleep(Duration::from_millis(10));
    };
    let stdout = std::fs::read_to_string(stdout).unwrap();
    let stderr = std::fs::read_to_string(stderr).unwrap();
    assert!(status.success(), "{name}: {status}\n{stdout}\n{stderr}");
    assert!(stdout.contains("1 passed"), "{stdout}");
    true
}

#[derive(Clone)]
struct Request {
    headers: String,
    body: Value,
}
#[derive(Default)]
struct State {
    requests: Vec<Request>,
    released: bool,
    body_started: usize,
}
type Handler = dyn Fn(&Request, usize) -> (u16, Value) + Send + Sync;
struct Server {
    base: String,
    state: Arc<(Mutex<State>, Condvar)>,
    stop: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<()>>,
}
fn read(stream: &mut TcpStream) -> Request {
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    let mut request_reader =
        bounded_fixture_io::Reader::new(stream, std::time::Instant::now() + Duration::from_secs(5));
    stream
        .set_write_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    let mut bytes = Vec::new();
    let mut buffer = [0; 8192];
    let end = loop {
        let count = request_reader.read(&mut buffer).unwrap();
        assert!(count > 0);
        bytes.extend_from_slice(&buffer[..count]);
        assert!(bytes.len() < 2_000_000);
        if let Some(index) = bytes.windows(4).position(|value| value == b"\r\n\r\n") {
            break index + 4;
        }
    };
    let headers = String::from_utf8(bytes[..end].to_vec()).unwrap();
    let length = headers
        .lines()
        .find_map(|line| {
            line.split_once(':')
                .filter(|(name, _)| name.eq_ignore_ascii_case("content-length"))
                .map(|(_, value)| value.trim().parse::<usize>().unwrap())
        })
        .unwrap();
    assert!(length < 2_000_000);
    while bytes.len() < end + length {
        let count = request_reader.read(&mut buffer).unwrap();
        assert!(count > 0);
        bytes.extend_from_slice(&buffer[..count]);
    }
    Request {
        headers,
        body: serde_json::from_slice(&bytes[end..end + length]).unwrap(),
    }
}
impl Server {
    fn new(
        held: bool,
        handler: impl Fn(&Request, usize) -> (u16, Value) + Send + Sync + 'static,
    ) -> Self {
        Self::start(held, false, handler)
    }
    fn stalled_body(
        handler: impl Fn(&Request, usize) -> (u16, Value) + Send + Sync + 'static,
    ) -> Self {
        Self::start(true, true, handler)
    }
    fn start(
        held: bool,
        stall_body: bool,
        handler: impl Fn(&Request, usize) -> (u16, Value) + Send + Sync + 'static,
    ) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let base = format!("http://{}/v1", listener.local_addr().unwrap());
        let state = Arc::new((
            Mutex::new(State {
                released: !held,
                ..Default::default()
            }),
            Condvar::new(),
        ));
        let stop = Arc::new(AtomicBool::new(false));
        let stopping = stop.clone();
        let captured = state.clone();
        let handler: Arc<Handler> = Arc::new(handler);
        let worker = thread::spawn(move || {
            let mut workers = Vec::new();
            while !stopping.load(Ordering::Acquire) {
                let (mut stream, _) = match listener.accept() {
                    Ok(value) => value,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2));
                        continue;
                    }
                    Err(error) => panic!("mock accept: {error}"),
                };
                let captured = captured.clone();
                let handler = handler.clone();
                workers.push(thread::spawn(move || {
                    stream.set_nonblocking(false).unwrap();
                    let request = read(&mut stream);
                    let (lock, changed) = &*captured;
                    let mut state = lock.lock().unwrap();
                    let index = state.requests.len();
                    state.requests.push(request.clone()); changed.notify_all();
                    let stalled = if stall_body {
                        let (status, body) = handler(&request, index);
                        let bytes = serde_json::to_vec(&body).unwrap();
                        assert!(bytes.len() > 1);
                        write!(stream, "HTTP/1.1 {status} Result\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", bytes.len()).unwrap();
                        stream.write_all(&bytes[..1]).unwrap();
                        stream.flush().unwrap();
                        state.body_started += 1;
                        changed.notify_all();
                        Some(bytes)
                    } else {
                        None
                    };
                    let deadline = Instant::now() + Duration::from_secs(10);
                    while !state.released {
                        let remaining = deadline.saturating_duration_since(Instant::now());
                        if remaining.is_zero() { break; }
                        state = changed.wait_timeout(state, remaining).unwrap().0;
                    }
                    let released = state.released; drop(state);
                    if let Some(bytes) = stalled {
                        if released {
                            let _ = stream.write_all(&bytes[1..]);
                        }
                        return;
                    }
                    let (status, body) = if released { handler(&request, index) } else { (500, json!({"error":"authored gate timeout"})) };
                    let bytes = serde_json::to_vec(&body).unwrap();
                    let _ = write!(stream, "HTTP/1.1 {status} Result\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", bytes.len());
                    let _ = stream.write_all(&bytes);
                }));
            }
            for worker in workers {
                let _ = worker.join();
            }
        });
        Self {
            base,
            state,
            stop,
            worker: Some(worker),
        }
    }
    fn wait(&self, count: usize) -> Vec<Request> {
        let (lock, changed) = &*self.state;
        let mut state = lock.lock().unwrap();
        let deadline = Instant::now() + Duration::from_secs(8);
        while state.requests.len() < count {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                drop(state);
                self.release();
                panic!("expected {count} real HTTP requests");
            }
            state = changed.wait_timeout(state, remaining).unwrap().0;
        }
        state.requests.clone()
    }
    fn requests(&self) -> Vec<Request> {
        self.state.0.lock().unwrap().requests.clone()
    }
    fn body_started(&self) -> usize {
        self.state.0.lock().unwrap().body_started
    }
    fn release(&self) {
        let mut state = self
            .state
            .0
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        state.released = true;
        self.state.1.notify_all();
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        self.release();
        self.stop.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}
fn success(request: &Request, _: usize) -> (u16, Value) {
    let content = &request.body["messages"][1]["content"];
    let user = content
        .as_str()
        .unwrap_or_else(|| content[0]["text"].as_str().unwrap());
    let text = if request.body["messages"][0]["content"]
        .as_str()
        .unwrap()
        .contains("MARKITAI_DOCUMENT_JSON_V1")
    {
        json!({"cleaned_markdown":user,"frontmatter":{"description":"Complete routing fixture","tags":["routing"]}}).to_string()
    } else {
        user.to_owned()
    };
    (
        200,
        json!({"choices":[{"message":{"content":text},"finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":5}}),
    )
}
fn deployment(base: &str, key: &str, id: &str, weight: u64) -> Value {
    json!({"model_name":"default","litellm_params":{"model":"openai/routing-fixture","api_base":base,"api_key":key,"weight":weight},"model_info":{"id":id}})
}
fn config(root: &Path, models: Vec<Value>) -> Value {
    json!({"cache":{"enabled":false,"global_dir":root.join("cache")},"prompts":{"dir":root.join("prompts")},"llm":{"enabled":true,"pure":true,"on_failure":"fail","model_list":models,"router_settings":{"routing_strategy":"least-busy","num_retries":0,"timeout":15}}})
}
fn source(root: &Path, name: &str) -> PathBuf {
    let path = root.join(format!("{name}.md"));
    std::fs::write(
        &path,
        format!("# {name}\n\nComplete source text for {name}.\n"),
    )
    .unwrap();
    path
}
fn call(
    source: &Path,
    cfg: Value,
    runtime: &LlmRuntime,
) -> markitai_core::Result<ConversionOutput> {
    convert_with_context(
        source.to_str().unwrap(),
        ConvertOptions {
            config: Some(cfg),
            ..Default::default()
        },
        ConvertContext {
            llm_runtime: Some(runtime),
            ..Default::default()
        },
    )
}

fn with_total(request: &Request, index: usize, total: u64) -> (u16, Value) {
    let (status, mut body) = success(request, index);
    body["usage"] = json!({"prompt_tokens":total-2,"completion_tokens":2,"total_tokens":total});
    (status, body)
}
// Selection evidence must not straddle a real wall-clock bucket boundary.
// This only waits for a fresh test window; it does not assert a speed result.
fn usage_window() {
    let second = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
        % 60;
    if second >= 30 {
        thread::sleep(Duration::from_secs(60 - second));
    }
}
fn metric_config(root: &Path, models: Vec<Value>, strategy: &str) -> Value {
    let mut cfg = config(root, models);
    cfg["llm"]["router_settings"]["routing_strategy"] = json!(strategy);
    cfg
}

#[test]
fn concurrent_successful_usage_updates_are_not_lost_or_charged_twice() {
    if isolated("concurrent_successful_usage_updates_are_not_lost_or_charged_twice") {
        return;
    }
    usage_window();
    let dir = tempfile::tempdir().unwrap();
    let first = Server::new(true, |request, index| with_total(request, index, 5));
    let second = Server::new(false, |request, index| with_total(request, index, 8));
    let cfg = metric_config(
        dir.path(),
        vec![
            deployment(&first.base, "key-a", "a", 999),
            deployment(&second.base, "key-b", "b", 1),
        ],
        "usage-based-routing",
    );
    let path = source(dir.path(), "complete");
    let runtime = LlmRuntime::new(2).unwrap();
    thread::scope(|scope| {
        let one = scope.spawn(|| call(&path, cfg.clone(), &runtime));
        first.wait(1);
        let two = scope.spawn(|| call(&path, cfg.clone(), &runtime));
        first.wait(2);
        // Usage scores count completed responses, not active reservations.
        first.release();
        let one = one.join().unwrap().unwrap();
        let two = two.join().unwrap().unwrap();
        assert_eq!(one.usage.requests + two.usage.requests, 2);
        assert_eq!(one.usage.input_tokens + two.usage.input_tokens, 6);
    });
    for _ in 0..2 {
        call(&path, cfg.clone(), &runtime).unwrap();
    }
    assert_eq!(first.requests().len(), 2);
    assert_eq!(second.requests().len(), 2); // B=8 beats A=10; a lost A update would choose A.
    call(&path, cfg, &runtime).unwrap();
    assert_eq!(first.requests().len(), 3);
    assert_eq!(second.requests().len(), 2);
}

#[test]
fn latency_timeout_penalty_changes_the_next_real_request_without_timing_assertions() {
    if isolated("latency_timeout_penalty_changes_the_next_real_request_without_timing_assertions") {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let first = Server::new(true, success);
    let second = Server::new(true, success);
    let mut cfg = metric_config(
        dir.path(),
        vec![
            deployment(&first.base, "key-a", "a", 1),
            deployment(&second.base, "key-b", "b", 1),
        ],
        "latency-based-routing",
    );
    cfg["llm"]["router_settings"]["num_retries"] = json!(1);
    cfg["llm"]["router_settings"]["timeout"] = json!(1);
    let path = source(dir.path(), "timeout");
    let runtime = LlmRuntime::new(2).unwrap();
    thread::scope(|scope| {
        let worker = scope.spawn(|| call(&path, cfg.clone(), &runtime));
        let deadline = Instant::now() + Duration::from_secs(5);
        let initial = loop {
            if !first.requests().is_empty() {
                break 0;
            }
            if !second.requests().is_empty() {
                break 1;
            }
            assert!(Instant::now() < deadline, "no first HTTP attempt");
            thread::sleep(Duration::from_millis(2));
        };
        let (held, alternate) = if initial == 0 {
            (&first, &second)
        } else {
            (&second, &first)
        };
        alternate.release();
        // The provider may have billed a timed-out request, so it is not sent again.
        assert!(worker.join().unwrap().is_err());
        assert_eq!(alternate.requests().len(), 0);
        let next = call(&path, cfg.clone(), &runtime).unwrap();
        assert_eq!(next.usage.requests, 1);
        assert!(
            next.llm_markdown
                .unwrap()
                .contains("Complete source text for timeout.")
        );
        // Without the timeout penalty the held deployment, never measured,
        // would beat the alternate's positive latency.
        call(&path, cfg.clone(), &runtime).unwrap();
        assert_eq!(held.requests().len(), 1);
        assert_eq!(alternate.requests().len(), 2);
        held.release();
    });
}

#[test]
fn latency_body_timeout_after_flushed_headers_penalizes_the_stalled_deployment() {
    if isolated("latency_body_timeout_after_flushed_headers_penalizes_the_stalled_deployment") {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let first = Server::stalled_body(success);
    let second = Server::stalled_body(success);
    let mut cfg = metric_config(
        dir.path(),
        vec![
            deployment(&first.base, "key-a", "a", 1),
            deployment(&second.base, "key-b", "b", 1),
        ],
        "latency-based-routing",
    );
    cfg["llm"]["router_settings"]["num_retries"] = json!(1);
    cfg["llm"]["router_settings"]["timeout"] = json!(1);
    let path = source(dir.path(), "body-timeout");
    let runtime = LlmRuntime::new(1).unwrap();
    thread::scope(|scope| {
        let worker = scope.spawn(|| call(&path, cfg.clone(), &runtime));
        let deadline = Instant::now() + Duration::from_secs(5);
        let initial = loop {
            if first.body_started() > 0 {
                break 0;
            }
            if second.body_started() > 0 {
                break 1;
            }
            assert!(
                Instant::now() < deadline,
                "no flushed HTTP response headers"
            );
            thread::sleep(Duration::from_millis(2));
        };
        let (stalled, alternate) = if initial == 0 {
            (&first, &second)
        } else {
            (&second, &first)
        };
        alternate.release();
        // A successful status means the provider billed it: no second request.
        assert!(worker.join().unwrap().is_err());
        assert_eq!(alternate.requests().len(), 0);
        let next = call(&path, cfg.clone(), &runtime).unwrap();
        assert_eq!(next.usage.requests, 1);
        assert_eq!(next.usage.input_tokens, 7);
        assert_eq!(next.usage.output_tokens, 5);
        assert!(
            next.llm_markdown
                .unwrap()
                .contains("Complete source text for body-timeout.")
        );
        // Without the body-timeout penalty the stalled deployment remains cold
        // and is selected again ahead of the alternate's positive latency.
        call(&path, cfg.clone(), &runtime).unwrap();
        assert_eq!(stalled.requests().len(), 1);
        assert_eq!(stalled.body_started(), 1);
        assert_eq!(alternate.requests().len(), 2);
        assert_eq!(alternate.body_started(), 2);
        stalled.release();
    });
}

#[test]
fn paid_invalid_typed_content_scores_once_but_paid_http_failure_is_not_success() {
    if isolated("paid_invalid_typed_content_scores_once_but_paid_http_failure_is_not_success") {
        return;
    }
    usage_window();
    let dir = tempfile::tempdir().unwrap();
    let first = Server::new(false, |request, index| {
        let (status, mut body) = with_total(request, index, 6);
        if index == 0 {
            body["choices"][0]["message"]["content"] = json!("invalid structured content");
        }
        (status, body)
    });
    let second = Server::new(false, |request, index| with_total(request, index, 8));
    let mut cfg = metric_config(
        dir.path(),
        vec![
            deployment(&first.base, "key-a", "a", 1),
            deployment(&second.base, "key-b", "b", 1),
        ],
        "usage-based-routing",
    );
    cfg["llm"]["pure"] = json!(false);
    let runtime = LlmRuntime::new(1).unwrap();
    let path = source(dir.path(), "typed");
    let result = call(&path, cfg.clone(), &runtime).unwrap();
    assert_eq!(result.usage.requests, 2);
    assert_eq!(result.usage.input_tokens, 10);
    assert_eq!(result.usage.output_tokens, 4);
    assert_eq!(first.requests().len(), 1);
    assert_eq!(second.requests().len(), 1);
    call(&path, cfg, &runtime).unwrap();
    assert_eq!(first.requests().len(), 2);
    assert_eq!(second.requests().len(), 1);

    let paid = Server::new(false, |_, _| {
        (
            503,
            json!({"error":{"message":"authored HTTP failure"},"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}),
        )
    });
    let unused = Server::new(false, success);
    let cfg = metric_config(
        dir.path(),
        vec![
            deployment(&paid.base, "key-a", "a", 1),
            deployment(&unused.base, "key-b", "b", 1),
        ],
        "usage-based-routing",
    );
    let runtime = LlmRuntime::new(1).unwrap();
    for _ in 0..2 {
        let failure = markitai_core::convert_with_context_detailed(
            path.to_str().unwrap(),
            ConvertOptions {
                config: Some(cfg.clone()),
                ..Default::default()
            },
            ConvertContext {
                llm_runtime: Some(&runtime),
                ..Default::default()
            },
        )
        .unwrap_err();
        assert_eq!(failure.usage.requests, 1);
        assert_eq!(failure.usage.input_tokens, 3);
        assert_eq!(failure.usage.output_tokens, 2);
    }
    assert_eq!(paid.requests().len(), 2);
    assert!(unused.requests().is_empty());
}

#[test]
fn metric_identity_and_independent_runtimes_do_not_mix_credentials_or_results() {
    if isolated("metric_identity_and_independent_runtimes_do_not_mix_credentials_or_results") {
        return;
    }
    usage_window();
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(false, |request, index| with_total(request, index, 10));
    let cfg = metric_config(
        dir.path(),
        vec![
            deployment(&server.base, "key-a", "same-id", 1),
            deployment(&server.base, "key-b", "same-id", 1),
        ],
        "usage-based-routing",
    );
    let runtime = LlmRuntime::new(1).unwrap();
    let path = source(dir.path(), "identity");
    call(&path, cfg.clone(), &runtime).unwrap();
    call(&path, cfg.clone(), &runtime).unwrap();
    call(&path, cfg.clone(), &LlmRuntime::new(1).unwrap()).unwrap();
    let requests = server.requests();
    for (request, key) in requests.iter().zip(["key-a", "key-b", "key-a"]) {
        assert!(
            request
                .headers
                .to_ascii_lowercase()
                .contains(&format!("authorization: bearer {key}"))
        );
    }
    assert_eq!(requests.len(), 3);
}

#[test]
fn cached_typed_answer_has_no_new_metric_sample_or_paid_usage() {
    if isolated("cached_typed_answer_has_no_new_metric_sample_or_paid_usage") {
        return;
    }
    for strategy in ["usage-based-routing", "latency-based-routing"] {
        let dir = tempfile::tempdir().unwrap();
        let server = Server::new(false, success);
        let mut cfg = metric_config(
            dir.path(),
            vec![deployment(&server.base, "key-a", "a", 1)],
            strategy,
        );
        cfg["llm"]["pure"] = json!(false);
        cfg["cache"]["enabled"] = json!(true);
        let runtime = LlmRuntime::new(2).unwrap();
        let path = source(dir.path(), "cached");
        let first = call(&path, cfg.clone(), &runtime).unwrap();
        let second = call(&path, cfg, &runtime).unwrap();
        assert_eq!(first.llm_markdown, second.llm_markdown);
        assert_eq!(server.requests().len(), 1);
        assert_eq!(first.usage.requests, 1);
        assert_eq!(second.usage.requests, 0);
        assert!(second.llm_cache_hit());
    }
}

#[test]
fn explicit_http_timeout_scores_a_penalty_and_preserves_paid_error_usage() {
    if isolated("explicit_http_timeout_scores_a_penalty_and_preserves_paid_error_usage") {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let timeout = Server::new(false, |_, _| {
        (
            408,
            json!({"error":{"message":"authored upstream timeout"},"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}),
        )
    });
    let good = Server::new(false, success);
    let cfg = metric_config(
        dir.path(),
        vec![
            deployment(&timeout.base, "key-a", "a", 1),
            deployment(&good.base, "key-b", "b", 1),
        ],
        "latency-based-routing",
    );
    let runtime = LlmRuntime::new(1).unwrap();
    let path = source(dir.path(), "http-timeout");
    let mut failures = 0;
    for _ in 0..3 {
        match markitai_core::convert_with_context_detailed(
            path.to_str().unwrap(),
            ConvertOptions {
                config: Some(cfg.clone()),
                ..Default::default()
            },
            ConvertContext {
                llm_runtime: Some(&runtime),
                ..Default::default()
            },
        ) {
            Ok(output) => assert_eq!(output.usage.requests, 1),
            Err(failure) => {
                failures += 1;
                assert!(failure.error.to_string().contains("HTTP 408"));
                assert_eq!(failure.usage.requests, 1);
                assert_eq!(failure.usage.input_tokens, 3);
                assert_eq!(failure.usage.output_tokens, 2);
            }
        }
    }
    assert_eq!(failures, 1);
    assert_eq!(timeout.requests().len(), 1);
    assert_eq!(good.requests().len(), 2);
}

#[cfg(test)]
mod bounded_fixture_io {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/support/bounded_read.rs"
    ));
}
