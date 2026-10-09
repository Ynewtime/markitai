//! Real stdio calls retain terminal work without changing error or job identity.
use super::*;

struct Model {
    base: String,
    requests: Arc<Mutex<Vec<Value>>>,
    release: Arc<(Mutex<bool>, Condvar)>,
    stop: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}

impl Model {
    fn new(handler: impl Fn(&Value, usize) -> (u16, Value, bool) + Send + Sync + 'static) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let release = Arc::new((Mutex::new(false), Condvar::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let (captured, released, stopped) = (requests.clone(), release.clone(), stop.clone());
        let handler = Arc::new(handler);
        let thread = thread::spawn(move || {
            let mut workers = Vec::new();
            while !stopped.load(Ordering::SeqCst) {
                let (mut stream, _) = match listener.accept() {
                    Ok(connection) => connection,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2));
                        continue;
                    }
                    Err(error) => panic!("mock accept failed: {error}"),
                };
                let (captured, released, handler) =
                    (captured.clone(), released.clone(), handler.clone());
                workers.push(thread::spawn(move || {
                    stream.set_nonblocking(false).unwrap();
                    stream.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
                    let mut bytes = Vec::new();
                    let mut buffer = [0; 8192];
                    let request_deadline = Instant::now() + Duration::from_secs(10);
                    let header_end = loop {
                        let remaining = request_deadline.saturating_duration_since(Instant::now());
                        assert!(!remaining.is_zero(), "mock request deadline exceeded");
                        stream.set_read_timeout(Some(remaining)).unwrap();
                        let read = match stream.read(&mut buffer) {
                            Ok(count) => count,
                            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                            Err(error) => panic!("mock read: {error}"),
                        };
                        assert!(read > 0, "incomplete request headers");
                        bytes.extend_from_slice(&buffer[..read]);
                        assert!(bytes.len() < 2_000_000);
                        if let Some(end) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
                            break end + 4;
                        }
                    };
                    let header = std::str::from_utf8(&bytes[..header_end]).unwrap();
                    let length = header.lines().find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length").then(|| value.trim().parse::<usize>().unwrap())
                    }).unwrap();
                    assert!(length < 2_000_000);
                    while bytes.len() < header_end + length {
                        let remaining = request_deadline.saturating_duration_since(Instant::now());
                        assert!(!remaining.is_zero(), "mock request deadline exceeded");
                        stream.set_read_timeout(Some(remaining)).unwrap();
                        let read = match stream.read(&mut buffer) {
                            Ok(count) => count,
                            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                            Err(error) => panic!("mock read: {error}"),
                        };
                        assert!(read > 0, "incomplete request body");
                        bytes.extend_from_slice(&buffer[..read]);
                    }
                    let request: Value = serde_json::from_slice(&bytes[header_end..header_end + length]).unwrap();
                    let index = {
                        let mut requests = captured.lock().unwrap();
                        let index = requests.len();
                        requests.push(request.clone());
                        index
                    };
                    let (status, response, hold) = handler(&request, index);
                    if hold {
                        let (lock, changed) = &*released;
                        let (_held, timeout) = changed.wait_timeout_while(lock.lock().unwrap(), Duration::from_secs(10), |released| !*released).unwrap();
                        assert!(!timeout.timed_out(), "held model was not released");
                    }
                    let body = serde_json::to_vec(&response).unwrap();
                    let header = format!("HTTP/1.1 {status} Response\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len());
                    // A deliberately disconnected MCP caller can close its HTTP
                    // worker while the test is cleaning up after a failure.
                    let _ = stream.write_all(header.as_bytes()).and_then(|_| stream.write_all(&body));
                }));
            }
            for worker in workers {
                worker.join().unwrap();
            }
        });
        Self {
            base,
            requests,
            release,
            stop,
            thread: Some(thread),
        }
    }

    fn count(&self) -> usize {
        self.requests.lock().unwrap().len()
    }
    fn wait(&self, count: usize) {
        let deadline = Instant::now() + Duration::from_secs(8);
        while self.count() < count {
            assert!(Instant::now() < deadline, "expected {count} model requests");
            thread::sleep(Duration::from_millis(5));
        }
    }
    fn release(&self) {
        *self.release.0.lock().unwrap() = true;
        self.release.1.notify_all();
    }
    fn configure(&self, client: &Client, share: bool) {
        let mut cfg = Client::defaults();
        cfg["prompts"] = json!({"dir":client.root().join("prompts")});
        cfg["llm"] = json!({"enabled":true,"on_failure":"fail","keep_base":true,"concurrency":4,
        "router_settings":{"timeout":10,"num_retries":0},
        "model_list":[{"model_name":"default","litellm_params":{
            "model":"openai/mcp-fixture","api_key":"private-mcp-fixture-key","api_base":self.base
        }}]});
        cfg["cache"]["enabled"] = json!(share);
        let cache = client.root().join("cache-unavailable");
        if !cache.exists() {
            std::fs::write(&cache, b"persistent cache deliberately unavailable").unwrap();
        }
        cfg["cache"]["global_dir"] = json!(cache);
        client.config(cfg);
    }
}
impl Drop for Model {
    fn drop(&mut self) {
        self.release();
        self.stop.store(true, Ordering::SeqCst);
        if let Some(worker) = self.thread.take() {
            worker.join().unwrap();
        }
    }
}

fn user(request: &Value) -> &str {
    request["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|message| message["role"] == "user")
        .unwrap()["content"]
        .as_str()
        .unwrap()
}
fn success(request: &Value) -> Value {
    let content = json!({"cleaned_markdown":user(request),"frontmatter":{
        "description":"Authored MCP usage fixture","tags":["fixture"]
    }})
    .to_string();
    json!({"model":"mcp-fixture","choices":[{"message":{"content":content},"finish_reason":"stop"}],
        "usage":{"prompt_tokens":7,"completion_tokens":5}})
}
fn denied(input: u64, output: u64, kind: &str) -> Value {
    json!({"error":{"message":"Incorrect API key provided: private-mcp-fixture-key","code":kind,"type":kind},
        "usage":{"prompt_tokens":input,"completion_tokens":output}})
}
fn attempt(value: &Value, status: &str, input: u64, output: u64) {
    let attempt = &value["diagnostics"]["last_attempt"];
    assert_eq!(attempt["operation"], "convert", "{value}");
    assert_eq!(attempt["status"], status, "{value}");
    if status == "done" {
        assert_eq!(attempt["error"], Value::Null);
    } else {
        assert_eq!(attempt["error"], value["error"]);
    }
    let usage = &attempt["usage"];
    assert_eq!(usage["requests"], 1, "{value}");
    assert_eq!(usage["input_tokens"], input);
    assert_eq!(usage["output_tokens"], output);
    let models = usage["by_model"].as_object().unwrap();
    assert_eq!(
        models
            .values()
            .map(|m| m["requests"].as_u64().unwrap())
            .sum::<u64>(),
        1
    );
    assert_eq!(
        models
            .values()
            .map(|m| m["input_tokens"].as_u64().unwrap())
            .sum::<u64>(),
        input
    );
    assert_eq!(
        models
            .values()
            .map(|m| m["output_tokens"].as_u64().unwrap())
            .sum::<u64>(),
        output
    );
    // The provider's refusal echoes the key; its message is shown without it.
    assert!(!value.to_string().contains("private-mcp-fixture-key"));
}

#[test]
fn both_protocols_keep_paid_failures_known_zero_success_and_unknown_distinct() {
    for modern in [false, true] {
        let model = Model::new(|request, _| {
            if user(request).contains("AUTH ZERO") {
                (401, denied(0, 0, "invalid_api_key"), false)
            } else if user(request).contains("QUOTA PAID") {
                (429, denied(13, 2, "insufficient_quota"), false)
            } else {
                (200, success(request), false)
            }
        });
        let mut client = Client::start(modern);
        model.configure(&client, false);
        let zero = client.file("zero.md", "# AUTH ZERO\n\nAuthored content.\n");
        let paid = client.file("paid.md", "# QUOTA PAID\n\nAuthored content.\n");
        let good = client.file("good.md", "# SUCCESS\n\nAuthored content.\n");
        for (source, input, output, code) in [(zero, 0, 0, 401), (paid, 13, 2, 429)] {
            let result = client.tool("convert_document", json!({"path":source}));
            assert_eq!(result["isError"], true, "{result}");
            let value = &result["structuredContent"];
            let error = value["error"].as_str().unwrap();
            assert!(error.contains(&format!("HTTP {code}")));
            assert_eq!(
                result["content"][0]["text"],
                format!("Error executing tool convert_document: {error}")
            );
            attempt(value, "error", input, output);
        }
        let result = client.success("convert_document", json!({"path":good}));
        assert_eq!(result.as_object().unwrap().len(), 13);
        assert_eq!(result["pricing"]["cost_status"], "unknown");
        assert_eq!(result["pricing"]["priced_requests"], 0);
        assert_eq!(result["pricing"]["unpriced_requests"], 1);
        assert_eq!(result["pricing"]["pricing_snapshots"], json!([]));
        attempt(&result, "done", 7, 5);
        let early = client.tool("convert_document", json!({"path":"relative.md"}));
        assert_eq!(early["isError"], true);
        assert!(early.get("structuredContent").is_none());
        let missing = client.tool(
            "convert_document",
            json!({"path":client.root().join("absent.md")}),
        );
        assert_eq!(missing["isError"], true);
        assert!(missing.get("structuredContent").is_none());
        assert_eq!(model.count(), 3);
        let definitions = client.request("tools/list", json!({}));
        for tool in definitions["result"]["tools"].as_array().unwrap() {
            let schema = &tool["outputSchema"];
            assert_eq!(schema["type"], "object");
            let branches = schema["anyOf"].as_array().unwrap();
            assert_eq!(branches.len(), 2);
            assert!(
                branches[1]["required"]
                    .as_array()
                    .unwrap()
                    .contains(&json!("diagnostics"))
            );
            assert!(branches[1]["properties"]["diagnostics"]["properties"]["last_attempt"]["properties"]["usage"].is_object());
        }
        client.stop();
    }
}

#[test]
fn batch_keeps_out_of_order_observations_attached_to_their_input_slots() {
    let model = Model::new(|request, _| {
        if user(request).contains("QUOTA PAID") {
            (429, denied(13, 2, "insufficient_quota"), false)
        } else {
            (200, success(request), user(request).contains("SLOW FIRST"))
        }
    });
    let mut client = Client::start(true);
    model.configure(&client, false);
    let slow = client.file("slow.md", "# SLOW FIRST\n\nAuthored content.\n");
    let paid = client.file("paid.md", "# QUOTA PAID\n\nAuthored content.\n");
    let missing = client.root().join("missing.md");
    let good = client.file("good.md", "# SUCCESS\n\nOther content.\n");
    let sources = json!([slow, paid, missing, good]);
    let ack = client.success("batch_convert", json!({"sources":sources,"concurrency":4}));
    let id = ack["job_id"].as_str().unwrap();
    let deadline = Instant::now() + Duration::from_secs(8);
    loop {
        let status = client.status(id);
        if status["done"] == 3 {
            let slots = status["results"].as_array().unwrap();
            assert_eq!(slots[0]["source"], json!(paid));
            attempt(&slots[0], "error", 13, 2);
            assert_eq!(slots[1]["source"], json!(missing));
            assert!(slots[1].get("diagnostics").is_none());
            attempt(&slots[2], "done", 7, 5);
            break;
        }
        assert!(Instant::now() < deadline, "unfinished batch: {status}");
        thread::sleep(Duration::from_millis(10));
    }
    model.release();
    let status = client.completed(id);
    assert_eq!(status["done"], 4);
    assert_eq!(status["failed"], 2);
    for (index, slot) in status["results"].as_array().unwrap().iter().enumerate() {
        assert_eq!(slot["source"], sources[index]);
    }
    attempt(&status["results"][0], "done", 7, 5);
    attempt(&status["results"][1], "error", 13, 2);
    assert!(status["results"][2].get("diagnostics").is_none());
    attempt(&status["results"][3], "done", 7, 5);
    assert_eq!(model.count(), 3);
    client.stop();
}

#[test]
fn shared_batch_charges_only_the_actual_http_owner() {
    let model = Model::new(|request, _| (200, success(request), true));
    let mut client = Client::start(false);
    model.configure(&client, true);
    let source = client.file(
        "shared.md",
        "# SHARED\n\nAll four callers retain this paragraph.\n",
    );
    let ack = client.success(
        "batch_convert",
        json!({"sources":[source,source,source,source],"concurrency":4}),
    );
    model.wait(1);
    // Let callers attach while the real response is held; this is not timing data.
    thread::sleep(Duration::from_millis(200));
    model.release();
    let status = client.completed(ack["job_id"].as_str().unwrap());
    assert_eq!(status["failed"], 0, "{status}");
    let slots = status["results"].as_array().unwrap();
    assert_eq!(slots.len(), 4);
    assert_eq!(
        slots
            .iter()
            .filter(|slot| slot.get("diagnostics").is_some())
            .count(),
        1
    );
    for slot in slots {
        if slot.get("diagnostics").is_some() {
            attempt(slot, "done", 7, 5);
        }
        assert!(
            std::fs::read_to_string(slot["markdown_file"].as_str().unwrap())
                .unwrap()
                .contains("All four callers retain this paragraph.")
        );
    }
    assert_eq!(model.count(), 1);
    client.stop();
}

#[test]
fn failed_shared_owner_and_later_waiter_keep_separate_attempts() {
    let model = Model::new(|request, index| {
        if index == 0 {
            (401, denied(9, 1, "invalid_api_key"), true)
        } else {
            (200, success(request), false)
        }
    });
    let mut client = Client::start(true);
    model.configure(&client, true);
    let source = client.file(
        "shared.md",
        "# SHARED RETRY\n\nIndependent attempt scopes.\n",
    );
    let ack = client.success(
        "batch_convert",
        json!({"sources":[source,source],"concurrency":2}),
    );
    model.wait(1);
    thread::sleep(Duration::from_millis(200));
    model.release();
    let status = client.completed(ack["job_id"].as_str().unwrap());
    assert_eq!(status["done"], 2);
    assert_eq!(status["failed"], 1, "{status}");
    for slot in status["results"].as_array().unwrap() {
        if slot["status"] == "error" {
            attempt(slot, "error", 9, 1);
        } else {
            attempt(slot, "done", 7, 5);
        }
    }
    assert_eq!(model.count(), 2);
    client.stop();
}

#[test]
fn eof_drains_the_admitted_response_without_starting_queued_batch_items() {
    let model = Model::new(|request, _| (200, success(request), true));
    let mut client = Client::start(false);
    model.configure(&client, false);
    let source = client.file("held.md", "# HELD\n\nDrain this admitted conversion.\n");
    let directory = client.root().join("drained");
    let ack = client.success(
        "batch_convert",
        json!({"sources":[source,source,source],"output_dir":directory,"concurrency":1}),
    );
    model.wait(1);
    client.input.take();
    // The server stops dispatching as soon as it reads the end of its input;
    // the margin covers a heavily loaded machine (gate r84 started the queued
    // item when dispatch waited for the whole session to end).
    thread::sleep(Duration::from_millis(1000));
    assert!(
        client.child.try_wait().unwrap().is_none(),
        "server exited before draining the model"
    );
    model.release();
    client.stop();
    let batch = directory.join(format!("batch-{}", ack["job_id"].as_str().unwrap()));
    assert!(batch.join("0001/held.md.llm.md").is_file());
    assert!(!batch.join("0002").exists());
    assert!(!batch.join("0003").exists());
    assert_eq!(model.count(), 1);
}
