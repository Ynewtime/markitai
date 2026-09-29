use serde_json::{Value, json};
use std::{
    io::{BufRead, BufReader, Read, Write},
    net::TcpListener,
    path::{Path, PathBuf},
    process::{Child, ChildStdin, Command, Stdio},
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};

#[path = "mcp/usage.rs"]
mod usage;

struct Client {
    child: Child,
    input: Option<ChildStdin>,
    messages: mpsc::Receiver<Value>,
    stdout: Option<thread::JoinHandle<()>>,
    stderr: Option<thread::JoinHandle<String>>,
    directory: tempfile::TempDir,
    config: PathBuf,
    next_id: u64,
    modern: bool,
}

impl Client {
    fn start(modern: bool) -> Self {
        Self::start_with_launcher(modern, false, true)
    }

    fn start_with_launcher(modern: bool, alias: bool, explicit_config: bool) -> Self {
        let directory = tempfile::tempdir().unwrap();
        let config = directory.path().join("config.json");
        std::fs::create_dir(directory.path().join("home")).unwrap();
        std::fs::create_dir(directory.path().join("tmp")).unwrap();
        std::fs::write(&config, Self::defaults().to_string()).unwrap();
        let executable = if alias {
            #[cfg(unix)]
            {
                let path = directory.path().join("markitai-mcp");
                std::os::unix::fs::symlink(env!("CARGO_BIN_EXE_markitai"), &path).unwrap();
                path
            }
            #[cfg(not(unix))]
            {
                panic!("this fixture requires Unix symlinks");
            }
        } else {
            PathBuf::from(env!("CARGO_BIN_EXE_markitai"))
        };
        let mut command = Command::new(executable);
        command
            .env_clear()
            .env("MARKITAI_HOME", directory.path().join("home"))
            .env("TMPDIR", directory.path().join("tmp"))
            .env("TMP", directory.path().join("tmp"))
            .env("TEMP", directory.path().join("tmp"))
            .env("NO_PROXY", "127.0.0.1,localhost")
            .current_dir(directory.path())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if explicit_config {
            command.arg("--config").arg(&config);
        }
        if !alias {
            command.arg("mcp");
        }
        for name in ["HOME", "USERPROFILE", "SYSTEMROOT", "WINDIR", "PATH"] {
            if let Some(value) = std::env::var_os(name) {
                command.env(name, value);
            }
        }
        let mut child = command.spawn().unwrap();
        let input = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        let mut stderr = child.stderr.take().unwrap();
        let (send, messages) = mpsc::channel();
        let stdout = thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let line = line.unwrap();
                let value: Value = serde_json::from_str(&line)
                    .unwrap_or_else(|_| panic!("non-JSON stdout: {line}"));
                assert_eq!(value["jsonrpc"], "2.0");
                if send.send(value).is_err() {
                    break;
                }
            }
        });
        let stderr = thread::spawn(move || {
            let mut text = String::new();
            stderr.read_to_string(&mut text).unwrap();
            text
        });
        let mut client = Self {
            child,
            input: Some(input),
            messages,
            stdout: Some(stdout),
            stderr: Some(stderr),
            directory,
            config,
            next_id: 1,
            modern,
        };
        if modern {
            let result = client.request("server/discover", json!({}));
            assert!(result.get("error").is_none(), "{result}");
            assert!(
                result["result"]["supportedVersions"]
                    .as_array()
                    .unwrap()
                    .contains(&json!("2026-07-28"))
            );
        } else {
            let result = client.request("initialize", json!({"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"isolated-test","version":"1"}}));
            assert_eq!(result["result"]["protocolVersion"], "2025-11-25");
            assert_eq!(result["result"]["serverInfo"]["name"], "markitai");
            client.send(json!({"jsonrpc":"2.0","method":"notifications/initialized"}));
        }
        client
    }

    fn defaults() -> Value {
        json!({"llm":{"enabled":false},"ocr":{"enabled":false},"screenshot":{"enabled":false},
            "image":{"alt_enabled":false,"desc_enabled":false},"cache":{"enabled":false},
            "history":{"record":false},"fetch":{"strategy":"static"},"output":{"on_conflict":"rename"}})
    }

    fn root(&self) -> &Path {
        self.directory.path()
    }
    fn file(&self, name: &str, body: &str) -> PathBuf {
        let path = self.root().join(name);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, body).unwrap();
        path
    }
    fn config(&self, value: Value) {
        std::fs::write(&self.config, value.to_string()).unwrap();
    }
    fn send(&mut self, value: Value) {
        writeln!(self.input.as_mut().unwrap(), "{value}").unwrap();
        self.input.as_mut().unwrap().flush().unwrap();
    }
    fn request(&mut self, method: &str, mut params: Value) -> Value {
        if self.modern {
            params["_meta"] = json!({"io.modelcontextprotocol/protocolVersion":"2026-07-28",
                "io.modelcontextprotocol/clientInfo":{"name":"isolated-test","version":"1"},
                "io.modelcontextprotocol/clientCapabilities":{}});
        }
        let id = self.next_id;
        self.next_id += 1;
        self.send(json!({"jsonrpc":"2.0","id":id,"method":method,"params":params}));
        loop {
            let message = self
                .messages
                .recv_timeout(Duration::from_secs(12))
                .unwrap_or_else(|error| {
                    panic!(
                        "MCP response {id} missing: {error}; process={:?}",
                        self.child.try_wait()
                    )
                });
            if message.get("id") == Some(&json!(id)) {
                return message;
            }
            assert!(
                message.get("id").is_none(),
                "unexpected response: {message}"
            );
        }
    }
    fn tool(&mut self, name: &str, arguments: Value) -> Value {
        let response = self.request("tools/call", json!({"name":name,"arguments":arguments}));
        assert!(response.get("error").is_none(), "{response}");
        response["result"].clone()
    }
    fn success(&mut self, name: &str, arguments: Value) -> Value {
        let result = self.tool(name, arguments);
        assert_eq!(result["isError"], false, "{result}");
        let value = result["structuredContent"].clone();
        let text: Value =
            serde_json::from_str(result["content"][0]["text"].as_str().unwrap()).unwrap();
        assert_eq!(text, value);
        value
    }
    fn status(&mut self, job_id: &str) -> Value {
        self.success("job_status", json!({"job_id":job_id}))
    }
    fn completed(&mut self, job_id: &str) -> Value {
        let deadline = Instant::now() + Duration::from_secs(12);
        loop {
            let status = self.status(job_id);
            if status["status"] == "completed" {
                return status;
            }
            assert!(Instant::now() < deadline, "job did not complete: {status}");
            thread::sleep(Duration::from_millis(10));
        }
    }
    fn stop(&mut self) {
        self.input.take();
        let deadline = Instant::now() + Duration::from_secs(12);
        let status = loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                break status;
            }
            assert!(Instant::now() < deadline, "MCP did not exit after EOF");
            thread::sleep(Duration::from_millis(10));
        };
        self.stdout.take().unwrap().join().unwrap();
        let stderr = self.stderr.take().unwrap().join().unwrap();
        assert!(status.success(), "MCP failed: {status}; {stderr}");
    }
}

#[cfg(unix)]
#[test]
fn mcp_alias_accepts_bare_launch_and_global_configuration_without_stdout_help() {
    for (modern, explicit_config) in [(false, false), (true, true)] {
        let mut client = Client::start_with_launcher(modern, true, explicit_config);
        let list = client.request("tools/list", json!({}));
        assert_eq!(list["result"]["tools"].as_array().unwrap().len(), 4);
        if explicit_config {
            let source = client.file("alias.md", "# Alias\n\nOriginal content.\n");
            let result = client.success("convert_document", json!({"path":source}));
            assert!(
                result["markdown"]
                    .as_str()
                    .unwrap()
                    .contains("Original content.")
            );
            assert!(Path::new(result["markdown_file"].as_str().unwrap()).is_file());
        }
        client.stop();
    }
}
impl Drop for Client {
    fn drop(&mut self) {
        self.input.take();
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(task) = self.stdout.take() {
            let _ = task.join();
        }
        if let Some(task) = self.stderr.take() {
            let _ = task.join();
        }
    }
}

#[test]
fn both_protocol_eras_list_the_four_typed_tools_and_continue_after_errors() {
    for modern in [false, true] {
        let mut client = Client::start(modern);
        let list = client.request("tools/list", json!({}));
        // 2026-07-28 clients reject a listing without its required cache
        // directives; the earlier era keeps the original result shape.
        if modern {
            assert_eq!(list["result"]["ttlMs"], 0, "{list}");
            assert_eq!(list["result"]["cacheScope"], "private", "{list}");
        } else {
            assert!(list["result"].get("ttlMs").is_none(), "{list}");
            assert!(list["result"].get("cacheScope").is_none(), "{list}");
        }
        let tools = list["result"]["tools"].as_array().unwrap();
        let names: Vec<_> = tools
            .iter()
            .map(|tool| tool["name"].as_str().unwrap())
            .collect();
        assert_eq!(
            names,
            [
                "convert_document",
                "convert_url",
                "batch_convert",
                "job_status"
            ]
        );
        for tool in tools {
            assert_eq!(tool["inputSchema"]["type"], "object");
            assert_eq!(tool["outputSchema"]["type"], "object");
            if tool["name"] != "job_status" {
                assert!(tool["description"].as_str().unwrap().contains("MODEL"));
                assert_eq!(
                    tool["inputSchema"]["properties"]["llm"]["default"],
                    Value::Null
                );
                assert!(
                    tool["inputSchema"]["properties"]["llm"]["anyOf"]
                        .as_array()
                        .unwrap()
                        .contains(&json!({"type":"null"}))
                );
            }
        }
        assert!(
            client
                .request("authored/unknown", json!({}))
                .get("error")
                .is_some()
        );
        // The maintained SDK ignores unparsable JSON and stays usable.
        writeln!(client.input.as_mut().unwrap(), "{{broken").unwrap();
        let missing = client.tool("job_status", json!({"job_id":"not-created"}));
        assert_eq!(missing["isError"], true);
        assert!(
            missing["content"][0]["text"]
                .as_str()
                .unwrap()
                .contains("Unknown job id")
        );
        let ping = client.request("ping", json!({}));
        if modern {
            // The 2026-07-28 protocol removed ping; the SDK rejects it.
            assert!(ping.get("error").is_some(), "{ping}");
        } else {
            assert!(ping.get("result").is_some(), "{ping}");
        }
        let recovered = client.request("tools/list", json!({}));
        assert_eq!(recovered["result"]["tools"].as_array().unwrap().len(), 4);
        client.stop();
    }
}

#[test]
fn single_document_writes_full_files_and_unicode_preview_survives_server_exit() {
    let mut client = Client::start(false);
    let source = client.file("文档.md", "# 文档\n\n原文内容。\n");
    let first = client.success("convert_document", json!({"path":source,"llm":null}));
    assert_eq!(first.as_object().unwrap().len(), 11);
    assert!(first["markdown"].as_str().unwrap().contains("原文内容"));
    assert_eq!(first["truncated"], false);
    assert_eq!(first["cost_usd"], 0.0);
    assert_eq!(first["skip_reason"], Value::Null);
    let first_path = PathBuf::from(first["markdown_file"].as_str().unwrap());
    assert!(first_path.is_file());
    assert!(first_path.starts_with(first["output_dir"].as_str().unwrap()));
    let large = client.file("large.md", &"🦀".repeat(40_001));
    let output = client.root().join("large-out");
    let preview = client.success(
        "convert_document",
        json!({"path":large,"output_dir":output,"llm":false}),
    );
    assert_eq!(preview["truncated"], true);
    assert_eq!(preview["markdown"].as_str().unwrap().chars().count(), 2_000);
    let full_path = PathBuf::from(preview["markdown_file"].as_str().unwrap());
    assert!(
        std::fs::read_to_string(&full_path)
            .unwrap()
            .contains(&"🦀".repeat(40_001))
    );
    client.stop();
    assert!(first_path.is_file());
    assert!(full_path.is_file());
}

#[test]
fn validation_and_expected_conversion_errors_leave_service_usable() {
    let mut client = Client::start(false);
    let file = client.file("input.md", "# Fine\n");
    let missing = client.root().join("missing.pdf");
    for (name, args, expected) in [
        (
            "convert_document",
            json!({"path":"relative.md"}),
            "absolute",
        ),
        (
            "convert_document",
            json!({"path":file,"output_dir":"relative"}),
            "absolute",
        ),
        (
            "convert_document",
            json!({"path":client.root()}),
            "directory",
        ),
        ("convert_document", json!({"path":missing}), "not found"),
        (
            "convert_document",
            json!({"path":file,"profile":"invalid"}),
            "profile",
        ),
        (
            "convert_url",
            json!({"url":"file:///not-a-url"}),
            "convert_document",
        ),
        ("batch_convert", json!({"sources":[]}), "at least one"),
        (
            "batch_convert",
            json!({"sources":[file,"relative.md"]}),
            "absolute",
        ),
    ] {
        let result = client.tool(name, args);
        assert_eq!(result["isError"], true, "{result}");
        assert!(
            result["content"][0]["text"]
                .as_str()
                .unwrap()
                .contains(expected),
            "{result}"
        );
    }
    let png = client.root().join("image.png");
    // A valid 1x1 RGBA image, created from a fixed authored fixture encoding.
    std::fs::write(
        &png,
        [
            137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 1, 0, 0, 0, 1,
            8, 6, 0, 0, 0, 31, 21, 196, 137, 0, 0, 0, 11, 73, 68, 65, 84, 120, 156, 99, 96, 0, 2,
            0, 0, 5, 0, 1, 165, 246, 69, 64, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130,
        ],
    )
    .unwrap();
    let image = client.tool(
        "convert_document",
        json!({"path":png,"llm":false,"ocr":false}),
    );
    assert_eq!(image["isError"], true);
    assert!(
        image["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("LLM")
    );
    assert!(client.success("convert_document", json!({"path":file}))["markdown_file"].is_string());
    client.stop();
}

#[test]
fn config_reload_explicit_false_and_output_conflict_are_honored() {
    let mut client = Client::start(true);
    let file = client.file("input.md", "# Base\n");
    let directory = client.root().join("out");
    let first = client.success(
        "convert_document",
        json!({"path":file,"output_dir":directory}),
    );
    let base_path = PathBuf::from(first["markdown_file"].as_str().unwrap());
    let base_bytes = std::fs::read(&base_path).unwrap();
    let mut cfg = Client::defaults();
    cfg["llm"]["enabled"] = json!(true);
    cfg["output"]["on_conflict"] = json!("skip");
    client.config(cfg);
    let missing_model = client.tool(
        "convert_document",
        json!({"path":file,"output_dir":client.root().join("model-out")}),
    );
    assert_eq!(missing_model["isError"], true, "{missing_model}");
    let text = missing_model["content"][0]["text"].as_str().unwrap();
    assert!(
        text.contains("MODEL") && text.contains("mcpServers"),
        "{text}"
    );
    let skipped = client.success(
        "convert_document",
        json!({"path":file,"output_dir":directory,"llm":false}),
    );
    assert_eq!(skipped["skip_reason"], "exists");
    assert_eq!(skipped["markdown_file"], first["markdown_file"]);
    assert_eq!(skipped["markdown"], first["markdown"]);
    assert_eq!(std::fs::read(&base_path).unwrap(), base_bytes);
    let enhanced_path = base_path.with_extension("llm.md");
    let enhanced_bytes = b"---\ntitle: Existing enhanced\n---\n\n# Saved enhancement\n";
    std::fs::write(&enhanced_path, enhanced_bytes).unwrap();
    let enhanced = client.success(
        "convert_document",
        json!({"path":file,"output_dir":directory,"llm":false}),
    );
    assert_eq!(enhanced["skip_reason"], "exists");
    assert_eq!(enhanced["markdown_file"], enhanced_path.to_str().unwrap());
    assert_eq!(enhanced["markdown"], "# Saved enhancement\n");
    assert_eq!(std::fs::read(&enhanced_path).unwrap(), enhanced_bytes);
    assert_eq!(std::fs::read(&base_path).unwrap(), base_bytes);
    client.stop();
}

#[test]
fn batch_files_isolate_duplicate_names_and_keep_error_slots() {
    let mut client = Client::start(false);
    let a = client.file("a/same.csv", "name,value\na,first\n");
    let b = client.file("b/same.csv", "name,value\nb,second\n");
    let missing = client.root().join("missing.txt");
    let directory = client.root().join("batch");
    let ack = client.success(
        "batch_convert",
        json!({"sources":[a,b,missing,a],"output_dir":directory,"concurrency":0}),
    );
    assert_eq!(ack["status"], "running");
    let id = ack["job_id"].as_str().unwrap();
    assert_eq!(id.len(), 8);
    let status = client.completed(id);
    assert_eq!(status["done"], 4);
    assert_eq!(status["failed"], 1);
    assert_eq!(status["results"][2]["status"], "error");
    assert_eq!(status["results"][2].as_object().unwrap().len(), 3);
    for (index, expected) in [(0, "first"), (1, "second"), (3, "first")] {
        let result = &status["results"][index];
        assert_eq!(result.as_object().unwrap().len(), 5);
        let path = Path::new(result["markdown_file"].as_str().unwrap());
        assert!(path.starts_with(directory.join(format!("batch-{id}/{:04}", index + 1))));
        assert!(std::fs::read_to_string(path).unwrap().contains(expected));
    }
    client.stop();
}

#[test]
fn batch_expands_home_paths_but_retains_the_callers_source_label() {
    let mut client = Client::start(false);
    let source = client.file("home-source.md", "# Isolated home document\n");
    let Some(home) = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")) else {
        return;
    };
    let home = PathBuf::from(home);
    // A relative spelling from the real home reaches only our private fixture.
    // No test input or state is written into the actual user home.
    let Some(common) = home.ancestors().find(|parent| source.starts_with(parent)) else {
        return; // Different Windows volumes cannot have a home-relative spelling.
    };
    let mut relative = PathBuf::from("~");
    for _ in home.strip_prefix(common).unwrap().components() {
        relative.push("..");
    }
    relative.push(source.strip_prefix(common).unwrap());
    let label = relative.to_string_lossy().into_owned();
    let ack = client.success("batch_convert", json!({"sources":[label],"concurrency":1}));
    let status = client.completed(ack["job_id"].as_str().unwrap());
    assert_eq!(status["failed"], 0, "{status}");
    assert_eq!(status["results"][0]["source"], label);
    let path = status["results"][0]["markdown_file"].as_str().unwrap();
    assert!(
        std::fs::read_to_string(path)
            .unwrap()
            .contains("Isolated home document")
    );
    client.stop();
}

struct Pages {
    base: String,
    release: Arc<(Mutex<bool>, Condvar)>,
    stop: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}
impl Pages {
    fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let release = Arc::new((Mutex::new(false), Condvar::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let (released, stopped) = (release.clone(), stop.clone());
        let thread = thread::spawn(move || {
            let mut workers = Vec::new();
            while !stopped.load(Ordering::SeqCst) {
                let (mut stream, _) = match listener.accept() {
                    Ok(stream) => stream,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2));
                        continue;
                    }
                    Err(error) => panic!("accept failed: {error}"),
                };
                let release = released.clone();
                workers.push(thread::spawn(move || {
                    stream.set_nonblocking(false).unwrap();
                    stream.set_read_timeout(Some(Duration::from_secs(8))).unwrap();
                    let mut request_reader = bounded_fixture_io::Reader::new(&stream, std::time::Instant::now() + Duration::from_secs(8));
                    let mut request = [0;8192];
                    let read = request_reader.read(&mut request).unwrap();
                    let request = String::from_utf8_lossy(&request[..read]);
                    if request.starts_with("GET /slow ") {
                        let (lock, changed) = &*release;
                        let (_released, timeout) = changed.wait_timeout_while(lock.lock().unwrap(), Duration::from_secs(8), |released| !*released).unwrap();
                        assert!(!timeout.timed_out(), "slow fixture was not released");
                    }
                    let body = "<html><body><h1>Fixture page</h1><p>Isolated loopback article.</p></body></html>";
                    write!(stream,"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",body.len()).unwrap();
                }));
            }
            for worker in workers {
                worker.join().unwrap();
            }
        });
        Self {
            base,
            release,
            stop,
            thread: Some(thread),
        }
    }
    fn release(&self) {
        *self.release.0.lock().unwrap() = true;
        self.release.1.notify_all();
    }
}
impl Drop for Pages {
    fn drop(&mut self) {
        self.release();
        self.stop.store(true, Ordering::SeqCst);
        if let Some(task) = self.thread.take() {
            let _ = task.join();
        }
    }
}

#[test]
fn url_tool_and_background_polling_preserve_source_order() {
    let pages = Pages::start();
    let mut client = Client::start(true);
    let fast = format!("{}/fast", pages.base);
    let slow = format!("{}/slow", pages.base);
    let single = client.success("convert_url", json!({"url":fast}));
    assert!(
        single["markdown"]
            .as_str()
            .unwrap()
            .contains("Fixture page")
    );
    let missing = client.root().join("absent.md");
    let ack = client.success(
        "batch_convert",
        json!({"sources":[slow,fast,missing],"concurrency":2}),
    );
    let id = ack["job_id"].as_str().unwrap();
    let deadline = Instant::now() + Duration::from_secs(6);
    loop {
        let status = client.status(id);
        if status["done"] == 2 {
            assert_eq!(status["status"], "running");
            assert_eq!(status["results"][0]["source"], fast);
            assert_eq!(status["results"][1]["status"], "error");
            break;
        }
        assert!(
            Instant::now() < deadline,
            "blocked status or serialized batch: {status}"
        );
        thread::sleep(Duration::from_millis(10));
    }
    pages.release();
    let status = client.completed(id);
    assert_eq!(status["done"], 3);
    assert_eq!(status["failed"], 1);
    assert_eq!(status["results"][0]["source"], slow);
    assert_eq!(status["results"][1]["source"], fast);
    client.stop();
}

#[cfg(test)]
mod bounded_fixture_io {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/support/bounded_read.rs"
    ));
}
