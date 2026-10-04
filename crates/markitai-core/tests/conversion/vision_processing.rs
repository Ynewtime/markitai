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
    let exact = format!("vision_processing::{name}");
    if std::env::var("MARKITAI_VISION_TEST").as_deref() == Ok(&exact) {
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
        .env("MARKITAI_VISION_TEST", &exact)
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
pub(super) struct Server {
    pub(super) base: String,
    pub(super) requests: Arc<Mutex<Vec<Value>>>,
    peak: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<()>>,
}
impl Server {
    pub(super) fn new(
        handler: impl Fn(&Value, usize) -> (u16, Value) + Send + Sync + 'static,
    ) -> Self {
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
    pub(super) fn count(&self) -> usize {
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
    request["messages"][1]["content"][0]["text"]
        .as_str()
        .unwrap()
}
pub(super) fn reply(text: &str) -> Value {
    json!({"model":"fixture","choices":[{"message":{"content":text},"finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":5}})
}
pub(super) fn typed(body: &str, description: &str) -> Value {
    reply(&json!({"cleaned_markdown":body,"frontmatter":{"description":description,"tags":["'two words'","lang:rust"],"title":"Forbidden replacement","source":"wrong"}}).to_string())
}
pub(super) fn cfg(server: &Server, root: &Path) -> Value {
    // Page-order assertions read the optional page comments.
    json!({"output":{"page_markers":true},"prompts":{"dir":root.join("prompts")},"ocr":{"enabled":false},"cache":{"enabled":false,"global_dir":root.join("cache")},"fetch":{"strategy":"static"},"image":{"compress":false,"filter":{"min_width":0,"min_height":0,"min_area":0},"alt_enabled":false,"desc_enabled":false},"llm":{"enabled":true,"on_failure":"fallback","keep_base":true,"concurrency":3,"router_settings":{"timeout":10,"num_retries":0},"model_list":[{"model_name":"default","litellm_params":{"model":"openai/fixture","api_key":"local-fixture","api_base":server.base}}]}})
}
fn options_cfg(config: Value) -> ConvertOptions {
    ConvertOptions {
        config: Some(config),
        ..Default::default()
    }
}

fn frames(request: &Value) -> Vec<u8> {
    use base64::Engine;
    request["messages"][1]["content"]
        .as_array()
        .unwrap()
        .iter()
        .skip(1)
        .map(|block| {
            let uri = block["image_url"]["url"].as_str().unwrap();
            let (mime, encoded) = uri.split_once(',').unwrap();
            assert_eq!(mime, "data:image/png;base64");
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(encoded)
                .unwrap();
            let image = image::load_from_memory(&bytes).unwrap().to_rgb8();
            assert_eq!(image.dimensions(), (32, 32));
            image.get_pixel(16, 16).0[0]
        })
        .collect()
}
fn fixture(root: &Path, pages: usize, changed: bool) -> std::path::PathBuf {
    let path = root.join("pages.tiff");
    let mut data = std::io::Cursor::new(Vec::new());
    {
        let mut encoder = tiff::encoder::TiffEncoder::new(&mut data).unwrap();
        for n in 1..=pages {
            let color = if changed && n == 11 { 111 } else { n as u8 };
            let pixels = image::RgbImage::from_pixel(32, 32, image::Rgb([color, 100, 200]));
            encoder
                .write_image::<tiff::encoder::colortype::RGB8>(32, 32, pixels.as_raw())
                .unwrap();
        }
    }
    std::fs::write(&path, data.into_inner()).unwrap();
    path
}
fn success(request: &Value) -> Value {
    let frames = frames(request);
    let body = format!(
        "{}\n\nTranscription of frames {}.",
        content(request),
        frames
            .iter()
            .map(u8::to_string)
            .collect::<Vec<_>>()
            .join(",")
    );
    if request["messages"][0]["content"]
        .as_str()
        .unwrap()
        .contains("MARKITAI_VISION_JSON_V1")
    {
        typed(&body, "Visual document description")
    } else {
        assert!(
            request["messages"][0]["content"]
                .as_str()
                .unwrap()
                .contains("MARKITAI_VISION_CLEAN_V1")
        );
        reply(&body)
    }
}

#[test]
fn twenty_one_pages_are_complete_and_merged_in_order_with_first_metadata() {
    if isolated("twenty_one_pages_are_complete_and_merged_in_order_with_first_metadata") {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let source = fixture(dir.path(), 21, false);
    let both = Arc::new(std::sync::Barrier::new(2));
    let waiting = both.clone();
    let server = Server::new(move |request, _| {
        let numbers = frames(request);
        if numbers[0] > 10 {
            waiting.wait();
        }
        (200, success(request))
    });
    let mut options = options_cfg(cfg(&server, dir.path()));
    options.output_dir = Some(dir.path().join("out"));
    let output = convert(source.to_str().unwrap(), options).unwrap();
    assert_eq!(server.count(), 3);
    assert_eq!(server.peak.load(Ordering::SeqCst), 2);
    assert_eq!(output.usage.requests, 3);
    assert_eq!(output.usage.input_tokens, 21);
    assert_eq!(output.usage.output_tokens, 15);
    assert_eq!(
        output.frontmatter["description"],
        "Visual document description"
    );
    let enhanced = output.llm_markdown.unwrap();
    let mut prior = 0;
    for n in 1..=21 {
        let marker = format!("<!-- Page number: {n} -->");
        assert_eq!(enhanced.matches(&marker).count(), 1, "page {n}: {enhanced}");
        let at = enhanced.find(&marker).unwrap();
        assert!(n == 1 || at > prior);
        prior = at;
    }
    assert!(enhanced.contains("[Original TIFF]"));
    assert!(enhanced.contains("Transcription of frames 1,2,3,4,5,6,7,8,9,10."));
    assert!(enhanced.contains("Transcription of frames 11,12,13,14,15,16,17,18,19,20."));
    assert!(enhanced.contains("Transcription of frames 21."));
    let saved = std::fs::read_to_string(output.output_path.unwrap()).unwrap();
    assert!(!saved.contains("Visual document description"));
    assert!(!enhanced.contains("Forbidden replacement"));
    let requests = server.requests.lock().unwrap();
    assert_eq!(frames(&requests[0]), (1..=10).collect::<Vec<u8>>());
    let mut sizes = requests.iter().map(|r| frames(r).len()).collect::<Vec<_>>();
    sizes.sort();
    assert_eq!(sizes, vec![1, 10, 10]);
}

#[test]
fn eleven_pages_cache_all_pixels_and_recompute_only_changed_batch() {
    if isolated("eleven_pages_cache_all_pixels_and_recompute_only_changed_batch") {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let source = fixture(dir.path(), 11, false);
    let server = Server::new(|request, _| (200, success(request)));
    let mut config = cfg(&server, dir.path());
    config["cache"]["enabled"] = json!(true);
    let convert_once = || convert(source.to_str().unwrap(), options_cfg(config.clone())).unwrap();
    let first = convert_once();
    assert_eq!(first.usage.requests, 2);
    assert!(!first.llm_cache_hit());
    let second = convert_once();
    assert_eq!(second.usage.requests, 0);
    assert!(second.llm_cache_hit());
    assert_eq!(first.llm_markdown, second.llm_markdown);
    assert_eq!(server.count(), 2);
    fixture(dir.path(), 11, true);
    let third = convert_once();
    assert_eq!(third.usage.requests, 1);
    assert!(!third.llm_cache_hit());
    assert_eq!(server.count(), 3);
    assert_eq!(
        frames(server.requests.lock().unwrap().last().unwrap()),
        vec![111]
    );
    assert!(
        third
            .llm_markdown
            .unwrap()
            .contains("Transcription of frames 111.")
    );
}

#[test]
fn first_auth_failure_stops_later_batches_and_keeps_paid_usage_and_base() {
    if isolated("first_auth_failure_stops_later_batches_and_keeps_paid_usage_and_base") {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let source = fixture(dir.path(), 21, false);
    let server = Server::new(|_, _| {
        (
            401,
            json!({"error":{"message":"Denied"},"usage":{"prompt_tokens":3,"completion_tokens":1}}),
        )
    });
    let mut options = options_cfg(cfg(&server, dir.path()));
    options.output_dir = Some(dir.path().join("out"));
    let output = convert(source.to_str().unwrap(), options).unwrap();
    assert!(
        output
            .warnings
            .iter()
            .any(|w| w.contains("LLM enhancement failed"))
    );
    assert!(output.llm_markdown.is_none());
    assert!(output.llm_output_path.is_none());
    assert!(output.output_path.unwrap().is_file());
    assert_eq!(output.usage.requests, 1);
    assert_eq!(output.usage.input_tokens, 3);
    assert_eq!(server.count(), 1);
}

#[test]
fn first_validation_or_transport_failure_never_dispatches_later_frames() {
    if isolated("first_validation_or_transport_failure_never_dispatches_later_frames") {
        return;
    }
    for invalid_json in [true, false] {
        let dir = tempfile::tempdir().unwrap();
        let source = fixture(dir.path(), 21, false);
        let server = Server::new(move |request, _| {
            assert_eq!(frames(request), (1..=10).collect::<Vec<u8>>());
            assert!(
                request["messages"][0]["content"]
                    .as_str()
                    .unwrap()
                    .contains("MARKITAI_VISION_JSON_V1")
            );
            if invalid_json {
                (200, reply("This is not the required JSON document answer."))
            } else {
                (
                    500,
                    json!({"error":{"message":"Fixture unavailable"},"usage":{"prompt_tokens":2,"completion_tokens":1}}),
                )
            }
        });
        let mut options = options_cfg(cfg(&server, dir.path()));
        options.output_dir = Some(dir.path().join("out"));
        let output = convert(source.to_str().unwrap(), options).unwrap();
        assert!(
            output
                .warnings
                .iter()
                .any(|warning| warning.contains("LLM enhancement failed"))
        );
        assert!(output.llm_markdown.is_none());
        assert!(output.llm_output_path.is_none());
        let base = std::fs::read_to_string(output.output_path.as_ref().unwrap()).unwrap();
        assert_eq!(base.matches("<!-- Page number:").count(), 21);
        assert!(base.contains("<!-- Page number: 21 -->"));
        let attempts = if invalid_json { 3 } else { 1 };
        assert_eq!(server.count(), attempts);
        assert_eq!(output.usage.requests, attempts as u64);
        assert_eq!(output.usage.input_tokens, if invalid_json { 21 } else { 2 });
        assert_eq!(
            output.usage.output_tokens,
            if invalid_json { 15 } else { 1 }
        );
        assert!(!output.llm_cache_hit());
    }
}

#[test]
fn later_failure_never_publishes_partial_and_retry_reuses_successful_batch() {
    if isolated("later_failure_never_publishes_partial_and_retry_reuses_successful_batch") {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let source = fixture(dir.path(), 11, false);
    let server = Server::new(|request, index| {
        if index == 1 {
            (
                500,
                json!({"error":{"message":"Temporary"},"usage":{"prompt_tokens":2,"completion_tokens":1}}),
            )
        } else {
            (200, success(request))
        }
    });
    let mut config = cfg(&server, dir.path());
    config["cache"]["enabled"] = json!(true);
    let options = || {
        let mut options = options_cfg(config.clone());
        options.output_dir = Some(dir.path().join("out"));
        options
    };
    let failed = convert(source.to_str().unwrap(), options()).unwrap();
    assert!(
        failed
            .warnings
            .iter()
            .any(|w| w.contains("LLM enhancement failed"))
    );
    assert!(failed.llm_output_path.is_none());
    assert!(failed.llm_markdown.is_none());
    assert!(failed.output_path.unwrap().is_file());
    assert_eq!(failed.usage.requests, 2);
    assert_eq!(failed.usage.input_tokens, 9);
    assert_eq!(server.count(), 2);
    let retried = convert(source.to_str().unwrap(), options()).unwrap();
    assert!(
        !retried
            .warnings
            .iter()
            .any(|w| w.contains("LLM enhancement failed"))
    );
    assert_eq!(retried.usage.requests, 1);
    assert_eq!(server.count(), 3);
    assert_eq!(
        frames(server.requests.lock().unwrap().last().unwrap()),
        vec![11]
    );
    assert_eq!(
        retried
            .llm_markdown
            .unwrap()
            .matches("<!-- Page number:")
            .count(),
        11
    );
}

#[test]
fn insufficient_budget_rejects_known_batches_before_any_request() {
    if isolated("insufficient_budget_rejects_known_batches_before_any_request") {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let source = fixture(dir.path(), 21, false);
    let server = Server::new(|_, _| panic!("budget preflight must send no requests"));
    let mut config = cfg(&server, dir.path());
    config["llm"]["max_requests_per_document"] = json!(2);
    let mut options = options_cfg(config);
    options.output_dir = Some(dir.path().join("out"));
    let output = convert(source.to_str().unwrap(), options).unwrap();
    assert!(
        output
            .warnings
            .iter()
            .any(|w| w.contains("LLM enhancement failed"))
    );
    assert_eq!(output.usage.requests, 0);
    assert_eq!(server.count(), 0);
    assert!(output.llm_markdown.is_none());
}

#[test]
fn later_fatal_response_cancels_queued_batches_before_admission() {
    if isolated("later_fatal_response_cancels_queued_batches_before_admission") {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let source = fixture(dir.path(), 41, false);
    let server = Server::new(|request, index| {
        if index == 0 {
            (200, success(request))
        } else {
            (
                402,
                json!({"error":{"message":"Billing disabled"},"usage":{"prompt_tokens":2,"completion_tokens":1}}),
            )
        }
    });
    let mut config = cfg(&server, dir.path());
    config["llm"]["concurrency"] = json!(1);
    let mut options = options_cfg(config);
    options.output_dir = Some(dir.path().join("out"));
    let output = convert(source.to_str().unwrap(), options).unwrap();
    assert_eq!(server.count(), 2);
    assert_eq!(output.usage.requests, 2);
    assert!(output.llm_markdown.is_none());
    assert!(
        output
            .warnings
            .iter()
            .any(|w| w.contains("LLM enhancement failed"))
    );
}

#[cfg(test)]
mod bounded_fixture_io {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/support/bounded_read.rs"
    ));
}
