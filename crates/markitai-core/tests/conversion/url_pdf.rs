use super::*;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::thread;
use std::time::{Duration, Instant};

const PDF: &[u8] = include_bytes!("../../src/pdf_raster/fixtures/mixed-native-scanned-blank.pdf");

// A fresh test process isolates dotenv lookup and provider environment without
// changing HOME or mutating process-global environment during parallel tests.
fn isolated(name: &str) -> bool {
    let exact = format!("url_pdf::{name}");
    if std::env::var("MARKITAI_URL_PDF_TEST").as_deref() == Ok(exact.as_str()) {
        return false;
    }
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("state");
    let tmp = dir.path().join("tmp");
    std::fs::create_dir(&home).unwrap();
    std::fs::create_dir(&tmp).unwrap();
    let stdout = dir.path().join("stdout");
    let stderr = dir.path().join("stderr");
    let mut command = Command::new(std::env::current_exe().unwrap());
    command
        .args(["--exact", &exact, "--nocapture"])
        .env_clear()
        .env("MARKITAI_URL_PDF_TEST", &exact)
        .env("MARKITAI_HOME", &home)
        .env("TMPDIR", &tmp)
        .current_dir(dir.path())
        .stdout(Stdio::from(std::fs::File::create(&stdout).unwrap()))
        .stderr(Stdio::from(std::fs::File::create(&stderr).unwrap()));
    for key in [
        "PATH",
        "HOME",
        "USERPROFILE",
        "SYSTEMROOT",
        "WINDIR",
        "LANG",
        "LC_ALL",
        "TZ",
    ] {
        if let Some(value) = std::env::var_os(key) {
            command.env(key, value);
        }
    }
    let mut child = command.spawn().unwrap();
    let deadline = Instant::now() + Duration::from_secs(120);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if Instant::now() >= deadline {
            child.kill().unwrap();
            let _ = child.wait();
            panic!("isolated URL PDF test timed out: {exact}");
        }
        thread::sleep(Duration::from_millis(10));
    };
    let output = std::fs::read_to_string(stdout).unwrap();
    let error = std::fs::read_to_string(stderr).unwrap();
    assert!(status.success(), "{exact}: {status}\n{output}\n{error}");
    assert!(
        output.contains("1 passed"),
        "test selector did not execute {exact}: {output}"
    );
    true
}

#[derive(Clone)]
struct Reply {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}
impl Reply {
    fn data(mime: &str, body: &[u8]) -> Self {
        Self {
            status: 200,
            headers: vec![("Content-Type".into(), mime.into())],
            body: body.to_vec(),
        }
    }
    fn pdf() -> Self {
        Self::data("application/pdf", PDF)
    }
    fn header(mut self, name: &str, value: &str) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }
}

#[derive(Clone, Debug)]
struct Request {
    head: String,
    body: Vec<u8>,
}
struct Server {
    base: String,
    requests: Arc<Mutex<Vec<Request>>>,
    stop: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<()>>,
}
impl Server {
    fn new(replies: Vec<Reply>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&requests);
        let stop = Arc::new(AtomicBool::new(false));
        let stopping = Arc::clone(&stop);
        let worker = thread::spawn(move || {
            let mut replies = VecDeque::from(replies);
            while !stopping.load(Ordering::Acquire) {
                let (mut stream, _) = match listener.accept() {
                    Ok(connection) => connection,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2));
                        continue;
                    }
                    Err(error) => panic!("mock accept failed: {error}"),
                };
                // macOS may inherit the listener's nonblocking mode.
                stream.set_nonblocking(false).unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let mut request_reader = bounded_fixture_io::Reader::new(
                    &stream,
                    std::time::Instant::now() + Duration::from_secs(5),
                );
                stream
                    .set_write_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let mut bytes = Vec::new();
                let request = loop {
                    let mut chunk = [0; 8192];
                    let count = request_reader.read(&mut chunk).unwrap();
                    assert!(count > 0, "truncated mock request");
                    bytes.extend_from_slice(&chunk[..count]);
                    assert!(
                        bytes.len() <= 32 * 1024 * 1024,
                        "unexpectedly large mock request"
                    );
                    if let Some(end) = bytes.windows(4).position(|s| s == b"\r\n\r\n") {
                        let head = String::from_utf8(bytes[..end].to_vec()).unwrap();
                        let length = head
                            .lines()
                            .find_map(|line| {
                                let (key, value) = line.split_once(':')?;
                                key.eq_ignore_ascii_case("content-length")
                                    .then(|| value.trim().parse::<usize>().unwrap())
                            })
                            .unwrap_or(0);
                        if bytes.len() >= end + 4 + length {
                            break Request {
                                head,
                                body: bytes[end + 4..end + 4 + length].to_vec(),
                            };
                        }
                    }
                };
                let mut reply = replies.pop_front().unwrap_or(Reply {
                    status: 500,
                    headers: Vec::new(),
                    body: b"Unexpected duplicate request".to_vec(),
                });
                if reply.status == 200 && request.head.starts_with("POST /v1/chat/completions ") {
                    let input: Value = serde_json::from_slice(&request.body).unwrap();
                    let mut payload: Value = serde_json::from_slice(&reply.body).unwrap();
                    if let Some(content) = payload.pointer_mut("/choices/0/message/content")
                        && let Some(markdown) = content.as_str()
                    {
                        *content = json!(super::mock_model_content(&input, markdown));
                    }
                    reply.body = serde_json::to_vec(&payload).unwrap();
                }
                captured.lock().unwrap().push(request);
                write!(
                    stream,
                    "HTTP/1.1 {} Mock\r\nContent-Length: {}\r\nConnection: close\r\n",
                    reply.status,
                    reply.body.len()
                )
                .unwrap();
                for (key, value) in reply.headers {
                    write!(stream, "{key}: {value}\r\n").unwrap();
                }
                stream.write_all(b"\r\n").unwrap();
                stream.write_all(&reply.body).unwrap();
            }
        });
        Self {
            base,
            requests,
            stop,
            worker: Some(worker),
        }
    }
    fn requests(&self) -> Vec<Request> {
        self.requests.lock().unwrap().clone()
    }
    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base)
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            let result = worker.join();
            if !thread::panicking() {
                result.unwrap();
            }
        }
    }
}

fn cfg() -> Value {
    // Page-order assertions read the optional page comments.
    json!({
        "output":{"page_markers":true},
        "cache":{"enabled":false,"global_dir":std::env::current_dir().unwrap().join("cache")},
        "history":{"record":false},"fetch":{"strategy":"static","no_remote":true},
        "llm":{"enabled":false},"ocr":{"enabled":false},
        "image":{"compress":false,"format":"png","alt_enabled":false,"desc_enabled":false}
    })
}
fn run(
    source: &str,
    config: Value,
    output: Option<PathBuf>,
) -> markitai_core::Result<markitai_core::ConversionOutput> {
    convert(
        source,
        ConvertOptions {
            config: Some(config),
            output_dir: output,
            ..Default::default()
        },
    )
}
#[cfg(target_os = "macos")]
fn model(config: &mut Value, server: &Server) {
    config["llm"] = json!({"enabled":true,"keep_base":true,"router_settings":{"num_retries":0,"timeout":5},"model_list":[{"model_name":"local-vision","litellm_params":{"model":"openai/mock","api_base":server.url("/v1"),"api_key":"isolated-test-key"},"model_info":{"supports_vision":true}}]});
}
#[cfg(target_os = "macos")]
fn model_server() -> Server {
    Server::new(vec![Reply::data("application/json", br##"{"choices":[{"message":{"content":"# Verified pages\n\nAll three pages reviewed."}}],"usage":{"prompt_tokens":8,"completion_tokens":7}}"##)])
}
#[cfg(target_os = "macos")]
fn image_payload(request: &Request) -> (Value, Vec<Vec<u8>>) {
    use base64::Engine;
    assert!(request.head.starts_with("POST /v1/chat/completions "));
    let body: Value = serde_json::from_slice(&request.body).unwrap();
    let images = body["messages"][1]["content"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|block| block["image_url"]["url"].as_str())
        .map(|url| {
            let encoded = url
                .strip_prefix("data:image/png;base64,")
                .expect("actual PNG MIME");
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(encoded)
                .unwrap();
            assert_eq!(
                image::guess_format(&bytes).unwrap(),
                image::ImageFormat::Png
            );
            let image = image::load_from_memory(&bytes).unwrap();
            assert_eq!((image.width(), image.height()), (1275, 1650));
            bytes
        })
        .collect();
    (body, images)
}

#[test]
fn url_pdf_redirect_preserves_original_identity_and_final_metadata() {
    if isolated("url_pdf_redirect_preserves_original_identity_and_final_metadata") {
        return;
    }
    let mut redirect = Reply::data("text/plain", b"");
    redirect.status = 302;
    redirect = redirect.header("Location", "/download?token=final-secret&view=full");
    let server = Server::new(vec![redirect, Reply::pdf()]);
    let source = server.url("/original?token=source-secret");
    let mut config = cfg();
    config["screenshot"] = json!({"screenshot_only":true,"enabled":false});
    let result = run(&source, config, Some(PathBuf::from("output"))).unwrap();
    assert_eq!(result.source, source);
    assert_eq!(
        result.frontmatter["source"],
        server.url("/original?token=REDACTED")
    );
    assert_eq!(
        result.frontmatter["source_url"],
        server.url("/download?token=REDACTED&view=full")
    );
    assert_eq!(result.fetch_strategy(), Some("static"));
    assert!(!result.fetch_cache_hit());
    assert!(result.markdown.contains("NATIVE PAGE ONE"));
    assert_eq!(result.markdown.matches("<!-- Page number:").count(), 3);
    let filename = format!(
        "{}_original.md",
        server
            .base
            .strip_prefix("http://")
            .unwrap()
            .replace(['.', ':'], "_")
    );
    assert_eq!(
        result.output_path,
        Some(PathBuf::from("output").join(filename))
    );
    let metadata = serde_json::to_string(&result.frontmatter).unwrap();
    assert!(!metadata.contains("source-secret") && !metadata.contains("final-secret"));
    assert!(!metadata.contains("markitai-fetch-") && !metadata.contains("%PDF-"));
    let requests = server.requests();
    assert_eq!(requests.len(), 2);
    assert!(
        requests[0]
            .head
            .starts_with("GET /original?token=source-secret ")
    );
    assert!(
        requests[1]
            .head
            .starts_with("GET /download?token=final-secret&view=full ")
    );
    assert!(requests.iter().all(|request| request.body.is_empty()));
}

#[test]
#[cfg(target_os = "macos")]
fn url_pdf_auto_extensionless_capture_retains_body_and_published_names() {
    if isolated("url_pdf_auto_extensionless_capture_retains_body_and_published_names") {
        return;
    }
    let server = Server::new(vec![Reply::pdf(), Reply::pdf()]);
    let mut config = cfg();
    config["fetch"]["strategy"] = json!("auto");
    config["screenshot"] = json!({"enabled":true,"screenshot_only":true});
    config["output"]["filename"] = json!("Report.md");
    let output = PathBuf::from("output");
    let directory = output.join(".markitai/screenshots");
    std::fs::create_dir_all(&directory).unwrap();
    let previous = directory.join("Report.page0001.png");
    std::fs::write(&previous, b"previous capture").unwrap();
    let result = run(
        &server.url("/download"),
        config.clone(),
        Some(output.clone()),
    )
    .unwrap();
    assert!(result.markdown.contains("NATIVE PAGE ONE"));
    assert_eq!(result.output_path, Some(output.join("Report.md")));
    assert_eq!(result.screenshots.len(), 3);
    assert!(result.screenshots[0].ends_with("Report.page0001.v2.png"));
    for (index, shot) in result.screenshots.iter().enumerate() {
        assert_eq!(
            image::guess_format(&std::fs::read(shot).unwrap()).unwrap(),
            image::ImageFormat::Png
        );
        let reference = format!(
            "<!-- ![Page {}](.markitai/screenshots/{}) -->",
            index + 1,
            shot.file_name().unwrap().to_str().unwrap()
        );
        assert!(
            result.markdown.contains(&reference),
            "{reference}: {}",
            result.markdown
        );
    }
    assert_eq!(std::fs::read(previous).unwrap(), b"previous capture");
    assert!(
        !result.assets.is_empty(),
        "PDF embedded pictures must survive capture-only URL identity"
    );
    assert!(
        std::fs::read_to_string(result.output_path.as_ref().unwrap())
            .unwrap()
            .ends_with(&result.markdown)
    );
    assert_eq!(result.fetch_strategy(), Some("static"));
    let memory = run(&server.url("/download"), config, None).unwrap();
    assert!(memory.markdown.contains("NATIVE PAGE ONE"));
    assert_eq!(memory.markdown.matches("<!-- ![Page ").count(), 3);
    assert!(memory.screenshots.is_empty() && memory.output_path.is_none());
    assert_eq!(
        server.requests().len(),
        2,
        "one GET per conversion; PDF viewer must not be navigated"
    );
}

#[test]
fn url_pdf_only_flag_does_not_imply_capture_or_reject_memory() {
    if isolated("url_pdf_only_flag_does_not_imply_capture_or_reject_memory") {
        return;
    }
    let server = Server::new(vec![Reply::pdf(), Reply::pdf()]);
    let source = server.url("/document.pdf");
    let baseline = run(&source, cfg(), None).unwrap();
    let mut config = cfg();
    config["screenshot"] = json!({"screenshot_only":true,"enabled":false});
    let result = run(&source, config, None).unwrap();
    assert_eq!(result.markdown, baseline.markdown);
    assert!(result.markdown.contains("NATIVE PAGE ONE"));
    assert!(result.screenshots.is_empty() && result.output_path.is_none());
    assert_eq!(server.requests().len(), 2);
}

#[test]
fn url_pdf_malformed_body_fails_once_before_model_or_publication() {
    if isolated("url_pdf_malformed_body_fails_once_before_model_or_publication") {
        return;
    }
    let server = Server::new(vec![Reply::data(
        "application/pdf",
        b"%PDF-1.7\ninvalid objects\n%%EOF",
    )]);
    let unused_model = Server::new(Vec::new());
    let mut config = cfg();
    config["ocr"] = json!({"enabled":true});
    config["llm"] = json!({"enabled":true,"router_settings":{"num_retries":0,"timeout":2},"model_list":[{"model_name":"local","litellm_params":{"model":"openai/mock","api_base":unused_model.url("/v1"),"api_key":"isolated"},"model_info":{"supports_vision":true}}]});
    let error = run(
        &server.url("/opaque"),
        config,
        Some(PathBuf::from("output")),
    )
    .unwrap_err();
    assert_eq!(error.code(), "conversion_error", "{error}");
    assert_eq!(server.requests().len(), 1);
    assert!(unused_model.requests().is_empty());
    assert!(!Path::new("output").exists());
}

#[test]
#[cfg(target_os = "macos")]
fn url_pdf_revalidation_replaces_cached_html_without_replaying_or_redownloading() {
    if isolated("url_pdf_revalidation_replaces_cached_html_without_replaying_or_redownloading") {
        return;
    }
    let html = b"<article><h1>Old representation</h1><p>This HTML must not survive the new PDF.</p></article>";
    let server = Server::new(vec![
        Reply::data("text/html", html).header("ETag", "\"old-html\""),
        Reply::pdf(),
        Reply::pdf(),
    ]);
    let source = server.url("/changing");
    let mut config = cfg();
    config["cache"]["enabled"] = json!(true);
    let old = run(&source, config.clone(), None).unwrap();
    assert!(old.markdown.contains("Old representation"));
    config["screenshot"] = json!({"enabled":true});
    for output in ["first", "second"] {
        let result = run(&source, config.clone(), Some(PathBuf::from(output))).unwrap();
        assert!(result.markdown.contains("NATIVE PAGE ONE"));
        assert!(!result.markdown.contains("Old representation"));
        assert_eq!(result.screenshots.len(), 3);
        assert!(!result.fetch_cache_hit());
    }
    let requests = server.requests();
    assert_eq!(requests.len(), 3);
    assert!(
        !requests[0]
            .head
            .to_ascii_lowercase()
            .contains("if-none-match:")
    );
    assert!(
        requests[1]
            .head
            .to_ascii_lowercase()
            .contains("if-none-match: \"old-html\"")
    );
    assert!(
        !requests[2]
            .head
            .to_ascii_lowercase()
            .contains("if-none-match:")
    );
}

#[test]
#[cfg(target_os = "macos")]
fn url_pdf_vlm_ocr_sends_three_ordered_pages_in_memory() {
    if isolated("url_pdf_vlm_ocr_sends_three_ordered_pages_in_memory") {
        return;
    }
    let server = Server::new(vec![Reply::pdf()]);
    let llm = model_server();
    let mut config = cfg();
    model(&mut config, &llm);
    config["ocr"] = json!({"enabled":true});
    let result = run(&server.url("/content"), config, None).unwrap();
    let requests = llm.requests();
    assert_eq!(requests.len(), 1);
    let (body, images) = image_payload(&requests[0]);
    assert_eq!(images.len(), 3);
    assert_ne!(images[0], images[1]);
    assert_ne!(images[1], images[2]);
    // The authored first page starts near the top, the scan starts below it,
    // and the third page is blank. This detects actual image reordering without
    // relying on filenames or on OCR recognizing this fixture exactly.
    let first_ink_rows: Vec<_> = images
        .iter()
        .map(|bytes| {
            image::load_from_memory(bytes)
                .unwrap()
                .to_rgb8()
                .enumerate_pixels()
                .filter(|(_, _, pixel)| pixel.0.iter().any(|channel| *channel < 230))
                .map(|(_, y, _)| y)
                .min()
        })
        .collect();
    assert!(first_ink_rows[0].is_some_and(|y| y < 200));
    assert!(first_ink_rows[1].is_some_and(|y| (250..400).contains(&y)));
    assert_eq!(first_ink_rows[2], None);
    assert!(
        body["messages"][1]["content"]
            .as_array()
            .unwrap()
            .iter()
            .any(|block| block["text"]
                .as_str()
                .is_some_and(|text| text.contains("NATIVE PAGE ONE")))
    );
    assert!(result.output_path.is_none() && result.screenshots.is_empty());
    let enhanced = result.llm_markdown.unwrap();
    assert!(enhanced.contains("All three pages reviewed."));
    assert_eq!(enhanced.matches("<!-- ![Page ").count(), 3);
    assert_eq!(result.usage.requests, 1);
    assert_eq!(server.requests().len(), 1);
}

#[test]
#[cfg(target_os = "macos")]
fn url_pdf_pure_keeps_url_text_precedence_even_with_only() {
    if isolated("url_pdf_pure_keeps_url_text_precedence_even_with_only") {
        return;
    }
    let server = Server::new(vec![Reply::pdf(), Reply::pdf()]);
    for only in [false, true] {
        let llm = model_server();
        let mut config = cfg();
        model(&mut config, &llm);
        config["llm"]["pure"] = json!(true);
        config["llm"]["max_vision_pages_per_document"] = json!(1);
        config["screenshot"] = json!({"enabled":true,"screenshot_only":only});
        let output = PathBuf::from(format!("output-{only}"));
        let result = run(&server.url("/document.pdf"), config, Some(output)).unwrap();
        let requests = llm.requests();
        assert_eq!(requests.len(), 1);
        let body: Value = serde_json::from_slice(&requests[0].body).unwrap();
        let text = body["messages"][1]["content"]
            .as_str()
            .expect("URL pure must remain text-only even when PDF captures exist");
        assert!(text.contains("NATIVE PAGE ONE"));
        assert!(result.markdown.contains("NATIVE PAGE ONE"));
        assert!(result.output_path.as_ref().unwrap().is_file());
        assert_eq!(result.screenshots.len(), 3);
        assert!(result.screenshots.iter().all(|path| path.is_file()));
        assert!(
            !result
                .llm_markdown
                .unwrap()
                .contains("<!-- Page images for reference -->")
        );
    }
    assert_eq!(server.requests().len(), 2);
}

#[test]
#[cfg(target_os = "macos")]
fn url_pdf_rag_profile_keeps_page_captures_and_relocates_embedded_assets() {
    if isolated("url_pdf_rag_profile_keeps_page_captures_and_relocates_embedded_assets") {
        return;
    }
    let server = Server::new(vec![Reply::pdf()]);
    let mut config = cfg();
    config["screenshot"] = json!({"enabled":true,"screenshot_only":true});
    config["output"]["profile"] = json!("rag");
    let output = PathBuf::from("output");
    let result = run(&server.url("/report"), config, Some(output.clone())).unwrap();
    assert!(result.markdown.contains("NATIVE PAGE ONE"));
    assert_eq!(result.markdown.matches("<!-- page:").count(), 3);
    assert_eq!(result.screenshots.len(), 3);
    assert_eq!(result.markdown.matches("<!-- ![Page ").count(), 3);
    assert!(!result.assets.is_empty());
    for asset in &result.assets {
        assert!(asset.starts_with(output.join("assets")));
        assert!(asset.is_file());
        assert!(result.markdown.contains(&format!(
            "assets/{}",
            asset.file_name().unwrap().to_str().unwrap()
        )));
    }
    for screenshot in &result.screenshots {
        assert!(screenshot.is_file());
    }
    assert!(result.output_path.unwrap().is_file());
    assert_eq!(server.requests().len(), 1);
}

#[test]
fn url_html_only_memory_rejects_after_one_bounded_static_response() {
    if isolated("url_html_only_memory_rejects_after_one_bounded_static_response") {
        return;
    }
    for strategy in ["auto", "static"] {
        let server = Server::new(vec![Reply::data("text/html", b"<article><h1>HTML page</h1><p>The response is a webpage, not a downloaded PDF.</p></article>")]);
        let mut config = cfg();
        config["fetch"]["strategy"] = json!(strategy);
        config["screenshot"] = json!({"enabled":true,"screenshot_only":true});
        let error = run(&server.url("/opaque"), config, None).unwrap_err();
        assert_eq!(error.code(), "invalid_input", "{error}");
        assert!(error.to_string().contains("output_dir"));
        assert_eq!(
            server.requests().len(),
            1,
            "must classify once without launching a browser"
        );
    }
}

#[test]
#[cfg(target_os = "macos")]
fn url_pdf_visual_model_failure_requires_useful_memory_or_retained_disk_output() {
    if isolated("url_pdf_visual_model_failure_requires_useful_memory_or_retained_disk_output") {
        return;
    }
    use lopdf::{Object, Stream, dictionary};
    let mut document = lopdf::Document::with_version("1.7");
    let pages = document.new_object_id();
    let content = document.add_object(Stream::new(
        dictionary! {},
        b"1 0 0 rg 0 0 100 100 re f".to_vec(),
    ));
    let page = document.add_object(dictionary! {
        "Type" => "Page", "Parent" => pages, "Contents" => content,
        "Resources" => dictionary! {},
        "MediaBox" => vec![0.into(), 0.into(), 100.into(), Object::Integer(100)]
    });
    document.objects.insert(
        pages,
        dictionary! {
            "Type" => "Pages", "Kids" => vec![Object::Reference(page)], "Count" => 1
        }
        .into(),
    );
    let catalog = document.add_object(dictionary! { "Type" => "Catalog", "Pages" => pages });
    document.trailer.set("Root", catalog);
    let mut bytes = Vec::new();
    document.save_to(&mut bytes).unwrap();
    for (directory, policy) in [
        (None, "fallback"),
        (Some("retained"), "fallback"),
        (Some("strict"), "fail"),
    ] {
        let server = Server::new(vec![Reply::data("application/pdf", &bytes)]);
        let mut rejected = Reply::data(
            "application/json",
            br#"{"error":{"message":"local model unavailable"}}"#,
        );
        rejected.status = 503;
        let llm = Server::new(vec![rejected]);
        let mut config = cfg();
        model(&mut config, &llm);
        config["llm"]["on_failure"] = json!(policy);
        config["screenshot"] = json!({"enabled":true});
        let result = run(&server.url("/visual"), config, directory.map(PathBuf::from));
        if directory.is_none() || policy == "fail" {
            let error = result.unwrap_err();
            assert_eq!(error.code(), "conversion_error", "{error}");
        } else {
            let result = result.unwrap();
            assert!(result.llm_markdown.is_none());
            assert_eq!(result.screenshots.len(), 1);
            assert!(result.output_path.unwrap().is_file());
            assert!(
                result
                    .warnings
                    .iter()
                    .any(|warning| warning.contains("LLM enhancement failed"))
            );
        }
        if let Some(directory) = directory {
            let root = Path::new(directory);
            let shot = root.join(".markitai/screenshots/visual.page0001.png");
            let pixels = image::open(&shot).unwrap().to_rgb8();
            assert_eq!(
                pixels.get_pixel(pixels.width() / 2, pixels.height() / 2).0,
                [255, 0, 0]
            );
            let markdown = std::fs::read_to_string(root.join("visual.md")).unwrap();
            assert!(
                markdown.contains("<!-- ![Page 1](.markitai/screenshots/visual.page0001.png) -->")
            );
        }
        assert_eq!(
            server.requests().len(),
            1,
            "model failure must not refetch PDF"
        );
        let requests = llm.requests();
        assert_eq!(requests.len(), 1);
        let request: Value = serde_json::from_slice(&requests[0].body).unwrap();
        assert_eq!(
            request["messages"][1]["content"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|block| block["type"] == "image_url")
                .count(),
            1
        );
    }
}

#[cfg(test)]
mod bounded_fixture_io {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/support/bounded_read.rs"
    ));
}
