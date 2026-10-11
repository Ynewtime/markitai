use super::*;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::thread;
use std::time::{Duration, Instant};

fn isolated(name: &str) -> bool {
    let exact = format!("document_processing::{name}");
    if std::env::var("MARKITAI_DOCUMENT_TEST").as_deref() == Ok(&exact) {
        return false;
    }
    let dir = tempfile::tempdir().unwrap();
    let state = dir.path().join("state");
    std::fs::create_dir(&state).unwrap();
    let out = dir.path().join("stdout");
    let err = dir.path().join("stderr");
    let mut cmd = Command::new(std::env::current_exe().unwrap());
    cmd.args(["--exact", &exact, "--nocapture"])
        .env_clear()
        .env("MARKITAI_DOCUMENT_TEST", &exact)
        .env("MARKITAI_HOME", state)
        .current_dir(dir.path())
        .stdout(Stdio::from(std::fs::File::create(&out).unwrap()))
        .stderr(Stdio::from(std::fs::File::create(&err).unwrap()));
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
            cmd.env(name, value);
        }
    }
    let mut child = cmd.spawn().unwrap();
    let deadline = Instant::now() + Duration::from_secs(90);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if Instant::now() > deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!("isolated test timed out: {exact}");
        }
        thread::sleep(Duration::from_millis(10));
    };
    let stdout = std::fs::read_to_string(out).unwrap();
    let stderr = std::fs::read_to_string(err).unwrap();
    assert!(status.success(), "{exact}: {status}\n{stdout}\n{stderr}");
    assert!(
        stdout.contains("1 passed"),
        "test selector did not run: {stdout}"
    );
    true
}

type Handler = dyn Fn(&Value, usize) -> (u16, Value) + Send + Sync;
struct Server {
    base: String,
    requests: Arc<Mutex<Vec<Value>>>,
    peak: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<()>>,
}
impl Server {
    fn new(handler: impl Fn(&Value, usize) -> (u16, Value) + Send + Sync + 'static) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let stop = Arc::new(AtomicBool::new(false));
        let stopping = stop.clone();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let captured = requests.clone();
        let active = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let observed = peak.clone();
        let handler: Arc<Handler> = Arc::new(handler);
        let worker = thread::spawn(move || {
            let mut workers = Vec::new();
            while !stopping.load(Ordering::Acquire) {
                let (mut stream, _) = match listener.accept() {
                    Ok(value) => value,
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2));
                        continue;
                    }
                    Err(e) => panic!("mock accept: {e}"),
                };
                let handler = handler.clone();
                let captured = captured.clone();
                let active = active.clone();
                let observed = observed.clone();
                workers.push(thread::spawn(move || {
                    stream.set_nonblocking(false).unwrap();
                    stream.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
                    let mut request_reader = bounded_fixture_io::Reader::new(&stream, std::time::Instant::now() + Duration::from_secs(10));
                    stream.set_write_timeout(Some(Duration::from_secs(10))).unwrap();
                    let mut bytes=Vec::new();let mut buffer=[0;8192];
                    let head_end=loop {let count=request_reader.read(&mut buffer).unwrap();assert!(count>0);bytes.extend_from_slice(&buffer[..count]);if let Some(at)=bytes.windows(4).position(|v|v==b"\r\n\r\n"){break at+4;}assert!(bytes.len()<1_000_000);};
                    let head=String::from_utf8_lossy(&bytes[..head_end]).to_string();
                    if head.starts_with("GET ") {
                        let body="<!doctype html><title>URL article</title><article><h1>URL article</h1><p>A complete original URL document used by the typed persistent cache.</p></article>";
                        write!(stream,"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",body.len()).unwrap();return;
                    }
                    let length=head.lines().find_map(|line|line.split_once(':').filter(|(name,_)|name.eq_ignore_ascii_case("content-length")).map(|(_,n)|n.trim().parse::<usize>().unwrap())).unwrap();
                    assert!(length<2_000_000);
                    while bytes.len()<head_end+length {let n=request_reader.read(&mut buffer).unwrap();assert!(n>0);bytes.extend_from_slice(&buffer[..n]);}
                    let request:Value=serde_json::from_slice(&bytes[head_end..head_end+length]).unwrap();
                    let index={let mut values=captured.lock().unwrap();let index=values.len();values.push(request.clone());index};
                    let now=active.fetch_add(1,Ordering::SeqCst)+1;observed.fetch_max(now,Ordering::SeqCst);
                    let (status,response)=handler(&request,index);
                    let body=serde_json::to_vec(&response).unwrap();
                    write!(stream,"HTTP/1.1 {status} Result\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",body.len()).unwrap();stream.write_all(&body).unwrap();
                    active.fetch_sub(1,Ordering::SeqCst);
                }));
            }
            for worker in workers {
                worker.join().unwrap();
            }
        });
        Self {
            base,
            requests,
            peak,
            stop,
            worker: Some(worker),
        }
    }
    fn count(&self) -> usize {
        self.requests.lock().unwrap().len()
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            worker.join().unwrap();
        }
    }
}
fn content(request: &Value) -> &str {
    request["messages"][1]["content"].as_str().unwrap()
}
fn reply(text: &str) -> Value {
    json!({"model":"fixture","choices":[{"message":{"content":text},"finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":5}})
}
fn typed(body: &str, description: &str) -> Value {
    reply(&json!({"cleaned_markdown":body,"frontmatter":{"description":description,"tags":["'two words'","lang:rust"],"title":"Forbidden replacement","source":"wrong"}}).to_string())
}
fn cfg(server: &Server, root: &Path) -> Value {
    json!({"cache":{"enabled":false,"global_dir":root.join("cache")},"fetch":{"strategy":"static"},"llm":{"enabled":true,"on_failure":"fail","concurrency":3,"router_settings":{"timeout":10,"num_retries":0},"model_list":[{"model_name":"default","litellm_params":{"model":"openai/fixture","api_key":"local-fixture","api_base":server.base}}]}})
}
fn options_cfg(config: Value) -> ConvertOptions {
    ConvertOptions {
        config: Some(config),
        ..Default::default()
    }
}
fn long_source() -> String {
    format!(
        "# Original\n\nFIRST {}\n\nMIDDLE {}\n\nTAIL {}\n",
        "甲乙丙丁".repeat(6_000),
        "子丑寅卯".repeat(6_000),
        "辰巳午未".repeat(6_000)
    )
}

#[test]
fn structured_metadata_keeps_source_title_and_base_metadata() {
    if isolated("structured_metadata_keeps_source_title_and_base_metadata") {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("source.md");
    let original =
        "---\ncustom: retained source field\n---\n\n# Original title\n\nFaithful body.\n";
    std::fs::write(&path, original).unwrap();
    let server = Server::new(|request, _| {
        assert!(
            request["messages"][0]["content"]
                .as_str()
                .unwrap()
                .contains("MARKITAI_DOCUMENT_JSON_V1")
        );
        (200, typed(content(request), "  A\n useful\t description  "))
    });
    let mut config = cfg(&server, dir.path());
    config["llm"]["keep_base"] = json!(true);
    let mut options = options_cfg(config);
    options.output_dir = Some(dir.path().join("out"));
    let output = convert(path.to_str().unwrap(), options).unwrap();
    assert_eq!(output.frontmatter["title"], "Original title");
    assert_eq!(output.frontmatter["description"], "A useful description");
    assert_eq!(
        output.frontmatter["tags"],
        json!(["two-words", "lang-rust"])
    );
    assert!(
        output
            .markdown
            .contains("---\ncustom: retained source field\n---")
    );
    assert!(
        output
            .llm_markdown
            .as_deref()
            .unwrap()
            .contains("---\ncustom: retained source field\n---")
    );
    let base = std::fs::read_to_string(output.output_path.unwrap()).unwrap();
    let enhanced = std::fs::read_to_string(output.llm_output_path.unwrap()).unwrap();
    assert!(!base.contains("A useful description"));
    assert!(enhanced.contains("A useful description"));
    assert!(base.contains("---\ncustom: retained source field\n---"));
    assert!(enhanced.contains("---\ncustom: retained source field\n---"));
    assert!(!enhanced.contains("Forbidden replacement"));
    assert!(!enhanced.contains("cleaned_markdown"));
    assert_eq!(output.usage.requests, 1);
}

#[test]
fn long_unicode_document_runs_parallel_merges_in_order_and_restores_literals() {
    if isolated("long_unicode_document_runs_parallel_merges_in_order_and_restores_literals") {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("long.md");
    let literal = "````rust\nlet x = `__MARKITAI_IMAGE_1__`;  \n\n\n````\n";
    let source = format!(
        "{}\n{literal}\n[link](https://example.test/a?q=1)\n<!-- Page 7 -->\n",
        long_source()
    );
    std::fs::write(&path, &source).unwrap();
    let server = Server::new(|request, _| {
        let text = content(request);
        assert!(text.chars().count() <= 32_000);
        assert!(!text.contains("let x ="));
        thread::sleep(Duration::from_millis(if text.contains("FIRST") {
            150
        } else {
            30
        }));
        (
            200,
            typed(
                text,
                if text.contains("FIRST") {
                    "First metadata"
                } else {
                    "Later metadata"
                },
            ),
        )
    });
    let mut config = cfg(&server, dir.path());
    config["cache"]["enabled"] = json!(true);
    let output = convert(path.to_str().unwrap(), options_cfg(config.clone())).unwrap();
    assert!(
        !output
            .warnings
            .iter()
            .any(|warning| warning.contains("cache"))
    );
    let body = output.llm_markdown.unwrap();
    assert!(body.find("FIRST").unwrap() < body.find("MIDDLE").unwrap());
    assert!(body.find("MIDDLE").unwrap() < body.find("TAIL").unwrap());
    assert!(body.contains(literal));
    assert!(body.contains("[link](https://example.test/a?q=1)"));
    assert!(body.contains("<!-- Page 7 -->"));
    assert_eq!(output.frontmatter["description"], "First metadata");
    assert!(server.peak.load(Ordering::SeqCst) >= 2);
    assert_eq!(server.count(), 3);
    let usage = output.usage;
    assert_eq!(usage.requests, 3);
    assert_eq!(usage.input_tokens, 21);
    assert_eq!(usage.output_tokens, 15);
    let repeated = convert(path.to_str().unwrap(), options_cfg(config)).unwrap();
    assert!(repeated.llm_cache_hit());
    assert_eq!(repeated.usage.requests, 0);
    assert_eq!(repeated.llm_markdown.as_deref(), Some(body.as_str()));
    assert_eq!(server.count(), 3);
}

#[test]
fn url_text_cache_reuses_typed_body_and_metadata_without_credentials() {
    if isolated("url_text_cache_reuses_typed_body_and_metadata_without_credentials") {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let server = Server::new(|request, _| (200, typed(content(request), "URL metadata")));
    let url = format!("{}/article", server.base);
    let mut config = cfg(&server, dir.path());
    config["cache"]["enabled"] = json!(true);
    let first = convert(&url, options_cfg(config.clone())).unwrap();
    assert_eq!(server.count(), 1);
    config["llm"]["model_list"][0]["litellm_params"]["api_key"] =
        json!("env:ABSENT_TYPED_CACHE_KEY");
    let second = convert(&url, options_cfg(config)).unwrap();
    assert_eq!(server.count(), 1);
    assert!(second.llm_cache_hit());
    assert_eq!(second.llm_markdown, first.llm_markdown);
    assert_eq!(second.frontmatter["description"], "URL metadata");
    assert_eq!(second.usage.requests, 0);
}

#[test]
fn invalid_structured_reply_spends_budget_once_and_preserves_base_only() {
    if isolated("invalid_structured_reply_spends_budget_once_and_preserves_base_only") {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("source.md");
    std::fs::write(&path, "# Source\n\nBody.\n").unwrap();
    let server = Server::new(|_, _| (200, reply("Unstructured answer")));
    let mut config = cfg(&server, dir.path());
    config["llm"]["max_requests_per_document"] = json!(1);
    let out = dir.path().join("out");
    let mut options = options_cfg(config);
    options.output_dir = Some(out.clone());
    let error = convert(path.to_str().unwrap(), options).unwrap_err();
    assert!(error.to_string().contains("structured JSON"));
    assert_eq!(server.count(), 1);
    assert!(out.join("source.md.md").is_file());
    assert!(!out.join("source.md.llm.md").exists());
}

#[test]
fn failed_chunk_never_publishes_partial_and_retry_reuses_completed_chunks() {
    if isolated("failed_chunk_never_publishes_partial_and_retry_reuses_completed_chunks") {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("long.md");
    std::fs::write(&path, long_source()).unwrap();
    let failing = Arc::new(AtomicBool::new(true));
    let control = failing.clone();
    let server = Server::new(move |request, _| {
        let text = content(request);
        (
            200,
            if control.load(Ordering::Acquire) && text.contains("TAIL") {
                reply("not JSON")
            } else {
                typed(text, "Complete document")
            },
        )
    });
    let mut config = cfg(&server, dir.path());
    config["cache"]["enabled"] = json!(true);
    config["llm"]["max_requests_per_document"] = json!(6);
    let out = dir.path().join("out");
    let mut options = options_cfg(config.clone());
    options.output_dir = Some(out.clone());
    assert!(convert(path.to_str().unwrap(), options).is_err());
    // The failing chunk is asked twice: its second rejection repeats the first.
    assert_eq!(server.count(), 4);
    assert!(!out.join("long.md.llm.md").exists());
    failing.store(false, Ordering::Release);
    config["llm"]["max_requests_per_document"] = json!(2);
    let output = convert(path.to_str().unwrap(), options_cfg(config)).unwrap();
    assert_eq!(server.count(), 5);
    assert!(!output.llm_cache_hit());
    assert_eq!(output.usage.requests, 1);
    assert!(output.llm_markdown.unwrap().contains("TAIL"));
}

#[test]
fn multi_chunk_admission_rejects_before_any_http_request() {
    if isolated("multi_chunk_admission_rejects_before_any_http_request") {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("long.md");
    std::fs::write(&path, long_source()).unwrap();
    let server = Server::new(|request, _| (200, typed(content(request), "Unused")));
    let mut config = cfg(&server, dir.path());
    config["llm"]["max_requests_per_document"] = json!(3);
    let error = convert(path.to_str().unwrap(), options_cfg(config)).unwrap_err();
    assert!(error.to_string().contains("no chunks were sent"));
    assert_eq!(server.count(), 0);
}

#[test]
fn custom_prompt_still_requires_typed_result_and_retries_paid_invalid_answer() {
    if isolated("custom_prompt_still_requires_typed_result_and_retries_paid_invalid_answer") {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("source.md");
    std::fs::write(&path, "# Source\n\nBody.\n").unwrap();
    let prompt = dir.path().join("prompt.txt");
    std::fs::write(
        &prompt,
        "User-owned instructions. {metadata_section} Preserve {content}",
    )
    .unwrap();
    let server = Server::new(|request, index| {
        assert!(
            request["messages"][0]["content"]
                .as_str()
                .unwrap()
                .contains("User-owned instructions.")
        );
        (
            200,
            if index == 0 {
                reply(
                    r#"{"cleaned_markdown":"Body","frontmatter":{"description":null,"tags":["x"]}}"#,
                )
            } else {
                typed(content(request), "Accepted")
            },
        )
    });
    let mut config = cfg(&server, dir.path());
    config["prompts"] = json!({"document_process_system":prompt});
    let output = convert(path.to_str().unwrap(), options_cfg(config)).unwrap();
    assert_eq!(server.count(), 2);
    let usage = output.usage;
    assert_eq!(usage.requests, 2);
    assert_eq!(usage.input_tokens, 14);
    assert_eq!(usage.output_tokens, 10);
}

#[test]
fn protected_marker_loss_fails_instead_of_silently_losing_a_link() {
    if isolated("protected_marker_loss_fails_instead_of_silently_losing_a_link") {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("source.md");
    std::fs::write(&path, "# Source\n\n[Important](https://example.test/)\n").unwrap();
    let server = Server::new(|_, _| (200, typed("# Source\n\nLink was removed", "Rejected")));
    let mut config = cfg(&server, dir.path());
    config["llm"]["max_requests_per_document"] = json!(1);
    let error = convert(path.to_str().unwrap(), options_cfg(config)).unwrap_err();
    assert!(error.to_string().contains("protected document marker"));
    assert_eq!(server.count(), 1);
}

#[test]
fn parallel_validation_retries_share_one_budget_and_fallback_usage() {
    if isolated("parallel_validation_retries_share_one_budget_and_fallback_usage") {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("long.md");
    std::fs::write(&path, long_source()).unwrap();
    let server = Server::new(|_, _| {
        thread::sleep(Duration::from_millis(20));
        (200, reply("not structured"))
    });
    let mut config = cfg(&server, dir.path());
    config["llm"]["max_requests_per_document"] = json!(4);
    config["llm"]["on_failure"] = json!("fallback");
    let output = convert(path.to_str().unwrap(), options_cfg(config)).unwrap();
    assert_eq!(server.count(), 4);
    assert!(output.llm_markdown.is_none());
    assert!(output.markdown.contains("TAIL"));
    assert_eq!(output.usage.requests, 4);
    assert_eq!(output.usage.input_tokens, 28);
    assert!(!output.warnings.is_empty());
}

#[test]
fn pure_long_document_keeps_one_raw_call_without_typed_metadata() {
    if isolated("pure_long_document_keeps_one_raw_call_without_typed_metadata") {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("long.md");
    std::fs::write(&path, long_source()).unwrap();
    let server = Server::new(|request, _| {
        assert!(
            !request["messages"][0]["content"]
                .as_str()
                .unwrap()
                .contains("MARKITAI_DOCUMENT_JSON_V1")
        );
        let text = content(request);
        assert!(text.contains("FIRST") && text.contains("TAIL"));
        assert!(text.chars().count() > 64_000);
        (200, reply("# Pure reply\n\nUnmodified output.\n"))
    });
    let mut config = cfg(&server, dir.path());
    config["llm"]["pure"] = json!(true);
    config["cache"]["enabled"] = json!(true);
    let output = convert(path.to_str().unwrap(), options_cfg(config)).unwrap();
    assert_eq!(server.count(), 1);
    assert_eq!(
        output.llm_markdown.as_deref(),
        Some("# Pure reply\n\nUnmodified output.\n")
    );
    assert!(output.frontmatter.get("description").is_none());
}

#[cfg(test)]
mod bounded_fixture_io {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/support/bounded_read.rs"
    ));
}
