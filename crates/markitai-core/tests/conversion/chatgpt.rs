#![cfg(unix)]
use markitai_core::{
    ConvertContext, ConvertOptions, LlmRuntime, convert_detailed, convert_json,
    convert_with_context_detailed,
};
use serde_json::{Value, json};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

fn isolated(name: &str) -> bool {
    let exact = format!("chatgpt::{name}");
    if std::env::var("MARKITAI_CODEX_TEST").as_deref() == Ok(&exact) {
        return false;
    }
    let root = tempfile::tempdir().unwrap();
    let out = root.path().join("stdout");
    let err = root.path().join("stderr");
    let mut command = Command::new(std::env::current_exe().unwrap());
    command
        .args(["--exact", &exact, "--nocapture"])
        .env_clear()
        .env("MARKITAI_CODEX_TEST", &exact)
        .env("MARKITAI_CODEX_TEST_ROOT", root.path())
        .env("MARKITAI_HOME", root.path().join("state"))
        // This test binary plays Codex once `mode` installs a scenario there.
        .env("CODEX_HOME", root.path().join("codex"))
        .env("CODEX_CLI_PATH", super::fake_runtime::program())
        .env("OPENAI_API_KEY", "must-not-enter-runtime")
        .current_dir(root.path())
        .stdout(Stdio::from(std::fs::File::create(&out).unwrap()))
        .stderr(Stdio::from(std::fs::File::create(&err).unwrap()));
    for key in ["HOME", "PATH", "LANG", "LC_ALL", "TMPDIR"] {
        if let Some(value) = std::env::var_os(key) {
            command.env(key, value);
        }
    }
    let mut child = command.spawn().unwrap();
    let start = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if start.elapsed() > Duration::from_secs(60) {
            let _ = child.kill();
            let _ = child.wait();
            panic!("private Codex conversion test timed out");
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    let out = std::fs::read_to_string(out).unwrap();
    let err = std::fs::read_to_string(err).unwrap();
    assert!(status.success(), "{status}\n{out}\n{err}");
    assert!(out.contains("1 passed"), "{out}");
    true
}
fn root() -> PathBuf {
    std::env::var_os("MARKITAI_CODEX_TEST_ROOT").unwrap().into()
}
/// The fake Codex home, where it reads its scenario and records its calls.
fn codex() -> PathBuf {
    root().join("codex")
}
fn mode(name: &str) {
    super::fake_runtime::install(&codex(), &json!({"name":name}));
}
fn cfg() -> Value {
    json!({"log":{"dir":null},"history":{"record":false},"prompts":{"dir":root().join("prompts")},"cache":{"enabled":true,"global_dir":root().join("cache")},"image":{"compress":false,"alt_enabled":false,"desc_enabled":false},"llm":{"enabled":true,"keep_base":true,"on_failure":"fail","router_settings":{"num_retries":2,"timeout":5},"model_list":[{"model_name":"default","litellm_params":{"model":"chatgpt/gpt-5.5"},"model_info":{"supports_vision":true}}]}})
}
fn opts(config: Value) -> ConvertOptions {
    ConvertOptions {
        config: Some(config),
        ..Default::default()
    }
}
fn source() -> PathBuf {
    let path = root().join("source.md");
    std::fs::write(&path,"# Authored title\n\n完整段落🚀。\n\n```rust\nlet text = \"literal\";\n```\n\n[kept](https://example.test/path)\n\nTHE END\n").unwrap();
    path
}
fn records(name: &str) -> Vec<Value> {
    std::fs::read_to_string(codex().join(name))
        .unwrap_or_default()
        .lines()
        .map(|v| serde_json::from_str(v).unwrap())
        .collect()
}
fn unknown(usage: &markitai_core::ConversionUsage) {
    assert_eq!(
        (usage.requests, usage.input_tokens, usage.output_tokens),
        (0, 11, 5)
    );
    assert!(!usage.cost_complete());
    assert_eq!(usage.cost_usd, 0.0);
    let row = &usage.by_model["chatgpt/gpt-5.5"];
    assert_eq!(row["incomplete_request_observations"], 1);
    assert_eq!(row["cost_status"], "unknown");
    assert!(row.get("pricing_snapshot").is_none());
}
#[test]
fn text_pure_and_repeated_context_calls_do_not_cache_account_state() {
    if isolated("text_pure_and_repeated_context_calls_do_not_cache_account_state") {
        return;
    }
    mode("echo");
    let path = source();
    let runtime = LlmRuntime::new(2).unwrap();
    for _ in 0..2 {
        let result = convert_with_context_detailed(
            path.to_str().unwrap(),
            opts(cfg()),
            ConvertContext {
                llm_runtime: Some(&runtime),
                ..Default::default()
            },
        )
        .unwrap();
        let text = result.llm_markdown.unwrap();
        assert!(text.contains("完整段落🚀"));
        assert!(text.contains("let text = \"literal\";"));
        assert!(text.contains("[kept](https://example.test/path)"));
        assert!(text.contains("THE END"));
        assert_eq!(result.frontmatter["description"], "Authored Codex fixture.");
        assert_eq!(result.frontmatter["title"], "Authored title");
        unknown(&result.usage);
        assert_ne!(result.skip_reason.as_deref(), Some("llm_cache"));
    }
    let mut config = cfg();
    config["llm"]["pure"] = json!(true);
    let result = convert_detailed(path.to_str().unwrap(), opts(config)).unwrap();
    assert!(result.llm_markdown.unwrap().contains("THE END"));
    unknown(&result.usage);
    let calls = records("requests.jsonl");
    assert_eq!(calls.len(), 3);
    assert!(
        calls[0]["system"]
            .as_str()
            .unwrap()
            .contains("MARKITAI_DOCUMENT_JSON_V1")
    );
    assert!(
        !calls[2]["system"]
            .as_str()
            .unwrap()
            .contains("MARKITAI_DOCUMENT_JSON_V1")
    );
}
#[test]
fn terminal_failures_preserve_observed_totals_but_never_expose_runtime_diagnostics() {
    if isolated("terminal_failures_preserve_observed_totals_but_never_expose_runtime_diagnostics") {
        return;
    }
    let path = source();
    mode("nonzero");
    let failure = convert_detailed(path.to_str().unwrap(), opts(cfg())).unwrap_err();
    unknown(&failure.usage);
    assert_eq!(records("requests.jsonl").len(), 1);
    mode("after-terminal");
    let value: Value = serde_json::from_str(&convert_json(
        &json!({"source":path,"options":{"config":cfg()}}).to_string(),
    ))
    .unwrap();
    assert_eq!(value["ok"], false);
    assert_eq!(value["error"]["usage"]["input_tokens"], 11);
    assert_eq!(value["error"]["usage"]["requests"], 0);
    assert!(!value.to_string().contains("sk-fake"));
    assert_eq!(records("requests.jsonl").len(), 2);
    mode("auth-api");
    let failure = convert_detailed(path.to_str().unwrap(), opts(cfg())).unwrap_err();
    assert!(failure.usage.by_model.is_empty());
    assert_eq!(records("requests.jsonl").len(), 2);
    assert!(!failure.error.to_string().contains("sk-fake"));
}
#[test]
fn positive_dollar_budget_prevents_any_runtime_launch() {
    if isolated("positive_dollar_budget_prevents_any_runtime_launch") {
        return;
    }
    mode("echo");
    let path = source();
    let mut config = cfg();
    config["llm"]["max_cost_per_document_usd"] = json!(1.0);
    let failure = convert_detailed(path.to_str().unwrap(), opts(config)).unwrap_err();
    assert!(failure.error.to_string().contains("verified tariff"));
    assert!(failure.usage.by_model.is_empty());
    assert!(records("calls.jsonl").is_empty());
}
#[test]
fn png_reaches_the_official_adapter_boundary_with_original_byte_identity() {
    use sha2::{Digest, Sha256};
    if isolated("png_reaches_the_official_adapter_boundary_with_original_byte_identity") {
        return;
    }
    mode("echo");
    let path = root().join("pixels.png");
    image::RgbImage::from_pixel(4, 3, image::Rgb([9, 42, 123]))
        .save(&path)
        .unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let result = convert_detailed(path.to_str().unwrap(), opts(cfg())).unwrap();
    unknown(&result.usage);
    let calls = records("requests.jsonl");
    assert_eq!(calls.len(), 1);
    assert_eq!(
        calls[0]["image_hashes"],
        json!([markitai_core::hex(Sha256::digest(bytes))])
    );
}

struct RejectedHttp {
    base: String,
    count: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}
impl RejectedHttp {
    fn new() -> Self {
        use std::io::{Read, Write};
        use std::sync::{
            Arc,
            atomic::{AtomicBool, AtomicUsize, Ordering},
        };
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let count = Arc::new(AtomicUsize::new(0));
        let counter = count.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let stopping = stop.clone();
        let thread = std::thread::spawn(move || {
            while !stopping.load(Ordering::Acquire) {
                let (mut stream, _) = match listener.accept() {
                    Ok(v) => v,
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(2));
                        continue;
                    }
                    Err(e) => panic!("{e}"),
                };
                stream.set_nonblocking(false).unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(3)))
                    .unwrap();
                stream
                    .set_write_timeout(Some(Duration::from_secs(3)))
                    .unwrap();
                let deadline = Instant::now() + Duration::from_secs(5);
                let mut bytes = Vec::new();
                let mut buffer = [0; 4096];
                let end = loop {
                    assert!(Instant::now() < deadline);
                    let n = match stream.read(&mut buffer) {
                        Ok(n) => n,
                        Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                        Err(e) => panic!("{e}"),
                    };
                    assert!(n > 0);
                    bytes.extend_from_slice(&buffer[..n]);
                    assert!(bytes.len() < 1024 * 1024);
                    if let Some(i) = bytes.windows(4).position(|s| s == b"\r\n\r\n") {
                        break i + 4;
                    }
                };
                let head = String::from_utf8_lossy(&bytes[..end]);
                assert!(head.starts_with("POST "));
                let length = head
                    .lines()
                    .find_map(|line| {
                        line.split_once(':')
                            .filter(|(k, _)| k.eq_ignore_ascii_case("content-length"))
                            .map(|(_, v)| v.trim().parse::<usize>().unwrap())
                    })
                    .unwrap();
                assert!(length < 1024 * 1024);
                while bytes.len() < end + length {
                    assert!(Instant::now() < deadline);
                    let n = match stream.read(&mut buffer) {
                        Ok(n) => n,
                        Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                        Err(e) => panic!("{e}"),
                    };
                    assert!(n > 0);
                    bytes.extend_from_slice(&buffer[..n]);
                }
                let body: Value = serde_json::from_slice(&bytes[end..end + length]).unwrap();
                assert_eq!(body["model"], "fixture-primary");
                let i = counter.fetch_add(1, Ordering::AcqRel);
                std::fs::write(root().join(format!("http-{i}.json")),json!({"request":body,"previous_codex_requests":records("requests.jsonl").len()}).to_string()).unwrap();
                let response=json!({"error":{"message":"authored primary authentication rejection"},"usage":{"prompt_tokens":2,"completion_tokens":1}}).to_string();
                write!(stream,"HTTP/1.1 401 Unauthorized\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",response.len(),response).unwrap();
            }
        });
        Self {
            base,
            count,
            stop,
            thread: Some(thread),
        }
    }
    fn config(&self) -> Value {
        let mut value = cfg();
        value["llm"]["concurrency"] = json!(1);
        value["llm"]["router_settings"]["num_retries"] = json!(0);
        value["llm"]["router_settings"]["fallbacks"] = json!([{"default":["subscription"]}]);
        value["llm"]["model_list"] = json!([{"model_name":"default","litellm_params":{"model":"openai/fixture-primary","api_key":"loopback-only","api_base":self.base}},{"model_name":"subscription","litellm_params":{"model":"chatgpt/gpt-5.5"}}]);
        value
    }
    fn count(&self) -> usize {
        self.count.load(std::sync::atomic::Ordering::Acquire)
    }
}
impl Drop for RejectedHttp {
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Release);
        if let Some(thread) = self.thread.take() {
            let result = thread.join();
            if !std::thread::panicking() {
                result.unwrap();
            }
        }
    }
}
#[test]
fn configured_http_to_subscription_fallback_preserves_order_and_disables_pool_cache() {
    if isolated("configured_http_to_subscription_fallback_preserves_order_and_disables_pool_cache")
    {
        return;
    }
    mode("echo");
    let path = source();
    let server = RejectedHttp::new();
    let runtime = LlmRuntime::new(1).unwrap();
    for i in 0..2 {
        let result = convert_with_context_detailed(
            path.to_str().unwrap(),
            opts(server.config()),
            ConvertContext {
                llm_runtime: Some(&runtime),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(result.llm_markdown.unwrap().contains("THE END"));
        assert_eq!(
            (
                result.usage.requests,
                result.usage.input_tokens,
                result.usage.output_tokens
            ),
            (1, 13, 6)
        );
        assert_eq!(
            result.usage.by_model["chatgpt/gpt-5.5"]["incomplete_request_observations"],
            1
        );
        assert!(!result.usage.cost_complete());
        assert_ne!(result.skip_reason.as_deref(), Some("llm_cache"));
        assert_eq!(server.count(), i + 1);
        assert_eq!(records("requests.jsonl").len(), i + 1);
        let event: Value =
            serde_json::from_slice(&std::fs::read(root().join(format!("http-{i}.json"))).unwrap())
                .unwrap();
        assert_eq!(event["previous_codex_requests"], i);
    }
}
#[test]
fn shared_chunk_request_budget_stops_subscription_spawn_after_prior_paid_work() {
    if isolated("shared_chunk_request_budget_stops_subscription_spawn_after_prior_paid_work") {
        return;
    }
    mode("echo");
    let path = root().join("long.md");
    std::fs::write(
        &path,
        format!(
            "# Long document\n\n{}\nEND OF DOCUMENT\n",
            "Complete authored paragraph with Unicode 文本.\n\n".repeat(900)
        ),
    )
    .unwrap();
    let server = RejectedHttp::new();
    let mut config = server.config();
    config["llm"]["max_requests_per_document"] = json!(3);
    let out = root().join("out");
    let options = ConvertOptions {
        output_dir: Some(out.clone()),
        ..opts(config)
    };
    let failure = convert_detailed(path.to_str().unwrap(), options).unwrap_err();
    assert!(
        failure.error.to_string().to_lowercase().contains("budget"),
        "{}",
        failure.error
    );
    assert_eq!(server.count(), 2);
    assert_eq!(records("requests.jsonl").len(), 1);
    assert_eq!(records("calls.jsonl").len(), 3);
    assert_eq!(
        (
            failure.usage.requests,
            failure.usage.input_tokens,
            failure.usage.output_tokens
        ),
        (2, 15, 7)
    );
    assert_eq!(
        failure.usage.by_model["chatgpt/gpt-5.5"]["incomplete_request_observations"],
        1
    );
    assert!(out.join("long.md.md").is_file());
    assert!(!out.join("long.md.llm.md").exists());
}
