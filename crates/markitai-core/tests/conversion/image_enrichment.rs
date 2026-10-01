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

// A fresh test process isolates dotenv lookup and provider environment without
// changing HOME or mutating process-global environment during parallel tests.
fn isolated(name: &str) -> bool {
    let exact = format!("image_enrichment::{name}");
    if std::env::var("MARKITAI_IMAGE_ENRICHMENT_TEST").as_deref() == Ok(exact.as_str()) {
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
        .env("MARKITAI_IMAGE_ENRICHMENT_TEST", &exact)
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
            panic!("isolated image enrichment test timed out: {exact}");
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
    echo: bool,
}
impl Reply {
    fn data(mime: &str, body: &[u8]) -> Self {
        Self {
            status: 200,
            headers: vec![("Content-Type".into(), mime.into())],
            body: body.to_vec(),
            echo: false,
        }
    }
    fn model(text: &str) -> Self {
        Self::data("application/json", &json!({"model":"mock", "choices":[{"message":{"content":text},"finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":5}}).to_string().into_bytes())
    }
    fn echo() -> Self {
        let mut reply = Self::model("");
        reply.echo = true;
        reply
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
                let parsed = serde_json::from_slice::<Value>(&request.body).ok();
                let structured = parsed
                    .as_ref()
                    .and_then(|v| v.pointer("/messages/0/content"))
                    .and_then(Value::as_str)
                    .is_some_and(|v| v.contains("MARKITAI_DOCUMENT_JSON_V1"));
                let echo = parsed.and_then(|v| {
                    v.pointer("/messages/1/content")
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                });
                captured.lock().unwrap().push(request);
                let reply = replies.pop_front().unwrap_or(Reply {
                    status: 500,
                    headers: Vec::new(),
                    body: b"Unexpected duplicate request".to_vec(),
                    echo: false,
                });
                let reply = if reply.echo {
                    let text = echo.expect("echo requires a text-model request");
                    Reply::model(&if structured {
                        json!({"cleaned_markdown":text,"frontmatter":{"description":"Image document fixture","tags":["fixture"]}}).to_string()
                    } else {
                        text
                    })
                } else {
                    reply
                };
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

fn cfg(server: &Server) -> Value {
    json!({
        "cache":{"enabled":false,"global_dir":std::env::current_dir().unwrap().join("cache")},
        "history":{"record":false},"fetch":{"strategy":"static","no_remote":true},
        "ocr":{"enabled":false},"screenshot":{"enabled":false},
        "prompts":{"dir":std::env::current_dir().unwrap().join("private-prompts")},
        "llm":{"enabled":true,"keep_base":true,"on_failure":"fallback","max_requests_per_document":50,
            "router_settings":{"num_retries":0,"timeout":3},
            "model_list":[{"model_name":"local","litellm_params":{"model":"openai/mock","api_base":server.url("/v1"),"api_key":"isolated"},"model_info":{"supports_vision":true}}]},
        "image":{"compress":false,"format":"png","alt_enabled":true,"desc_enabled":true}
    })
}
fn png() -> Vec<u8> {
    let mut bytes = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(image::RgbImage::from_fn(100, 100, |x, y| {
        image::Rgb([x as u8, y as u8, 71])
    }))
    .write_to(&mut bytes, image::ImageFormat::Png)
    .unwrap();
    bytes.into_inner()
}
fn analysis() -> Reply {
    Reply::model(&json!({"caption":"A [safe] chart", "description":"A chart with two ordered axes.","extracted_text":"Literal ``` fence\n42"}).to_string())
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
fn body(result: &markitai_core::ConversionOutput) -> &str {
    result.llm_markdown.as_deref().expect("enhanced Markdown")
}
fn requests(server: &Server) -> Vec<Value> {
    server
        .requests()
        .iter()
        .map(|request| {
            assert!(request.head.starts_with("POST /v1/chat/completions "));
            serde_json::from_slice(&request.body).unwrap()
        })
        .collect()
}
fn vision_bytes(request: &Value) -> Vec<Vec<u8>> {
    use base64::Engine;
    request["messages"][1]["content"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|block| block.pointer("/image_url/url").and_then(Value::as_str))
        .map(|url| {
            assert!(url.starts_with("data:image/png;base64,"));
            base64::engine::general_purpose::STANDARD
                .decode(url.split_once(',').unwrap().1)
                .unwrap()
        })
        .collect()
}

#[test]
fn standalone_analysis_publishes_rich_markdown_real_assets_and_upserts_sidecar() {
    if isolated("standalone_analysis_publishes_rich_markdown_real_assets_and_upserts_sidecar") {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let bytes = png();
    let path = dir.path().join("chart.png");
    std::fs::write(&path, &bytes).unwrap();
    let server = Server::new(vec![analysis(), analysis()]);
    let output = dir.path().join("out");
    let first = run(path.to_str().unwrap(), cfg(&server), Some(output.clone())).unwrap();
    assert_eq!(body(&first).matches("# chart\n").count(), 1);
    assert!(body(&first).contains("![A \\[safe\\] chart]"));
    assert!(body(&first).contains("````\nLiteral ``` fence\n42\n````"));
    assert_eq!(first.images.len(), 1);
    assert_eq!(first.usage.requests, 1);
    let entry = &first.images[0];
    let asset = Path::new(entry["asset"].as_str().unwrap());
    assert!(asset.is_absolute());
    assert_eq!(std::fs::read(asset).unwrap(), bytes);
    for key in ["alt", "desc", "text", "created"] {
        assert!(entry[key].is_string(), "{entry}");
    }
    assert_eq!(entry["llm_usage"]["mock"]["requests"], 1);
    let sidecar = asset.parent().unwrap().join("images.json");
    let before: Value = serde_json::from_slice(&std::fs::read(&sidecar).unwrap()).unwrap();
    assert_eq!(before["version"], "1.0");
    assert_eq!(before["images"].as_array().unwrap().len(), 1);
    assert_eq!(before["images"][0]["path"], entry["asset"]);
    assert!(before["images"][0].get("llm_usage").is_none());
    let other = dir.path().join("other.png");
    std::fs::write(&other, &bytes).unwrap();
    let second = run(other.to_str().unwrap(), cfg(&server), Some(output)).unwrap();
    assert_eq!(second.images[0]["asset"], entry["asset"]);
    let after: Value = serde_json::from_slice(&std::fs::read(sidecar).unwrap()).unwrap();
    assert_eq!(after["created"], before["created"]);
    assert_eq!(after["images"].as_array().unwrap().len(), 1);
    assert_eq!(after["images"][0]["source"], other.to_str().unwrap());
    let calls = requests(&server);
    assert_eq!(calls.len(), 2);
    assert_eq!(vision_bytes(&calls[0]), vec![bytes]);
}

#[cfg(unix)]
#[test]
fn stdout_store_links_enhanced_markdown_and_image_records_to_saved_files() {
    if isolated("stdout_store_links_enhanced_markdown_and_image_records_to_saved_files") {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let bytes = png();
    std::fs::write(dir.path().join("a.png"), &bytes).unwrap();
    let source = dir.path().join("report.md");
    std::fs::write(&source, "# Report\n\n![one](a.png)\n\n`![code](a.png)`\n").unwrap();
    let server = Server::new(vec![Reply::echo(), analysis()]);
    let store = dir.path().join("store");
    let output = markitai_core::convert_with_context(
        source.to_str().unwrap(),
        ConvertOptions {
            config: Some(cfg(&server)),
            ..Default::default()
        },
        markitai_core::ConvertContext {
            stdout_assets: Some(&store),
            ..Default::default()
        },
    )
    .unwrap();
    // Links name the canonical store path.
    let files: Vec<_> = std::fs::read_dir(store.join("blobs"))
        .unwrap()
        .map(|entry| std::fs::canonicalize(entry.unwrap().path()).unwrap())
        .collect();
    assert_eq!(files.len(), 1, "{files:?}");
    assert_eq!(std::fs::read(&files[0]).unwrap(), bytes);
    let uri = format!("file://{}", files[0].display());
    assert!(
        body(&output).contains(&format!("![A \\[safe\\] chart]({uri})")),
        "{}",
        body(&output)
    );
    assert!(output.markdown.contains(&format!("![one]({uri})")));
    assert!(body(&output).contains("`![code](a.png)`"));
    assert!(!body(&output).contains(".markitai/"));
    assert_eq!(output.images.len(), 1);
    assert_eq!(output.images[0]["asset"], files[0].to_str().unwrap());
    // Descriptions are not published without an output directory.
    let entries: Vec<_> = std::fs::read_dir(&store)
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect();
    assert_eq!(entries, vec![std::ffi::OsString::from("blobs")]);
    assert_eq!(requests(&server).len(), 2);
}

#[test]
fn embedded_local_duplicates_change_real_alts_and_preserve_literals_and_custom_prompt_data() {
    if isolated(
        "embedded_local_duplicates_change_real_alts_and_preserve_literals_and_custom_prompt_data",
    ) {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let bytes = png();
    for name in ["a.png", "b.png"] {
        std::fs::write(dir.path().join(name), &bytes).unwrap();
    }
    let source = dir.path().join("report.md");
    std::fs::write(&source,"# Report\n\nuntrusted {source} {content}\n\n![one](a.png) and ![two](b.png)\n\n`![code](a.png)`\n\n<!-- ![comment](b.png) -->\n").unwrap();
    let system = dir.path().join("system.txt");
    let user = dir.path().join("user.txt");
    std::fs::write(&system, "CUSTOM_IMAGE_SYSTEM").unwrap();
    std::fs::write(&user, "CUSTOM_IMAGE_USER {document_context} END").unwrap();
    let server = Server::new(vec![Reply::echo(), analysis()]);
    let mut config = cfg(&server);
    config["prompts"]["image_analysis_system"] = json!(system);
    config["prompts"]["image_analysis_user"] = json!(user);
    let output = run(
        source.to_str().unwrap(),
        config,
        Some(dir.path().join("out")),
    )
    .unwrap();
    assert_eq!(output.images.len(), 1);
    assert_eq!(output.usage.requests, 2);
    assert_eq!(body(&output).matches("![A \\[safe\\] chart]").count(), 2);
    assert!(body(&output).contains("`![code](a.png)`"));
    assert!(body(&output).contains("<!-- ![comment](b.png) -->"));
    assert!(!output.markdown.contains("A \\[safe\\] chart"));
    let calls = requests(&server);
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[1]["messages"][0]["content"], "CUSTOM_IMAGE_SYSTEM");
    let text = calls[1]["messages"][1]["content"][0]["text"]
        .as_str()
        .unwrap();
    assert!(text.contains("CUSTOM_IMAGE_USER"));
    assert!(text.contains("untrusted {source} {content}"));
    assert_eq!(vision_bytes(&calls[1]), vec![bytes]);
}

#[test]
fn document_request_limit_prevents_image_calls_without_losing_author_alt_or_base() {
    if isolated("document_request_limit_prevents_image_calls_without_losing_author_alt_or_base") {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("a.png"), png()).unwrap();
    let source = dir.path().join("report.md");
    std::fs::write(&source, "# Budget\n\n![Author alt](a.png)\n").unwrap();
    let server = Server::new(vec![Reply::echo()]);
    let mut config = cfg(&server);
    config["llm"]["max_requests_per_document"] = json!(1);
    let output = run(
        source.to_str().unwrap(),
        config,
        Some(dir.path().join("out")),
    )
    .unwrap();
    assert!(output.images.is_empty());
    assert_eq!(output.usage.requests, 1);
    assert!(body(&output).contains("![Author alt]"));
    assert_eq!(output.assets.len(), 1);
    assert!(
        output
            .warnings
            .iter()
            .any(|warning| warning.contains("budget exhausted"))
    );
    assert_eq!(requests(&server).len(), 1);
}

#[test]
fn structured_failure_uses_custom_caption_and_description_with_all_paid_usage() {
    if isolated("structured_failure_uses_custom_caption_and_description_with_all_paid_usage") {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("chart.png");
    std::fs::write(&source, png()).unwrap();
    let server = Server::new(vec![
        Reply::model("not JSON"),
        Reply::model("still not JSON"),
        Reply::model("finally not JSON"),
        Reply::model("Fallback caption"),
        Reply::model("Fallback details"),
    ]);
    let mut config = cfg(&server);
    for (name, text) in [
        ("image_caption_user", "CAPTION_USER {document_context}"),
        (
            "image_description_user",
            "DESCRIPTION_USER {document_context}",
        ),
    ] {
        let path = dir.path().join(name);
        std::fs::write(&path, text).unwrap();
        config["prompts"][name] = json!(path);
    }
    let output = run(source.to_str().unwrap(), config, None).unwrap();
    assert_eq!(output.usage.requests, 5);
    assert_eq!(output.usage.input_tokens, 35);
    assert_eq!(output.usage.output_tokens, 25);
    assert_eq!(output.images[0]["llm_usage"]["mock"]["requests"], 5);
    assert!(body(&output).contains("Fallback caption"));
    assert!(body(&output).contains("Fallback details"));
    let calls = requests(&server);
    assert_eq!(calls.len(), 5);
    for (index, marker) in [(3, "CAPTION_USER"), (4, "DESCRIPTION_USER")] {
        assert!(
            calls[index]["messages"][1]["content"][0]["text"]
                .as_str()
                .unwrap()
                .starts_with(marker)
        );
    }
}

#[test]
fn pure_local_document_skips_enrichment_but_pure_image_returns_structured_plain_content() {
    if isolated(
        "pure_local_document_skips_enrichment_but_pure_image_returns_structured_plain_content",
    ) {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let image = dir.path().join("chart.png");
    std::fs::write(&image, png()).unwrap();
    let source = dir.path().join("doc.md");
    std::fs::write(&source, "![Original](chart.png)\n").unwrap();
    let server = Server::new(vec![Reply::echo(), analysis()]);
    let mut config = cfg(&server);
    config["llm"]["pure"] = json!(true);
    let text = run(source.to_str().unwrap(), config.clone(), None).unwrap();
    assert_eq!(body(&text), "![Original](chart.png)\n");
    assert!(text.images.is_empty());
    assert_eq!(text.usage.requests, 1);
    let picture = run(image.to_str().unwrap(), config, None).unwrap();
    assert_eq!(
        body(&picture),
        "# chart\n\nA chart with two ordered axes.\n\nLiteral ``` fence\n42\n"
    );
    assert!(picture.images.is_empty());
    assert_eq!(picture.usage.requests, 1);
    assert!(!body(&picture).contains("!["));
    assert_eq!(requests(&server).len(), 2);
}

#[test]
fn url_images_resolve_redirects_and_complete_query_targets_including_pure() {
    if isolated("url_images_resolve_redirects_and_complete_query_targets_including_pure") {
        return;
    }
    for pure in [false, true] {
        let bytes = png();
        let source=Server::new(vec![
            Reply::data("text/html",b"<html><head><title>Images</title></head><body><article><h1>Images</h1><p>A sufficiently clear article with useful figures and author captions.</p><img alt='one' src='/figure?variant=one'><img alt='two' src='/figure?variant=two'></article></body></html>"),
            Reply {status:302,headers:vec![("Location".into(),"/actual.png?raw=1".into())],body:Vec::new(),echo:false},
            Reply::data("image/png",&bytes),Reply::data("image/png",&bytes),
        ]);
        let model = Server::new(vec![Reply::echo(), analysis()]);
        let mut config = cfg(&model);
        config["llm"]["pure"] = json!(pure);
        let output = run(&source.url("/article"), config, None).unwrap();
        assert_eq!(output.images.len(), 1, "{output:#?}");
        assert_eq!(output.usage.requests, 2);
        assert_eq!(body(&output).matches("![A \\[safe\\] chart]").count(), 2);
        assert!(!body(&output).contains("variant="));
        let gets = source.requests();
        assert_eq!(gets.len(), 4);
        assert!(gets[1].head.starts_with("GET /figure?variant=one "));
        assert!(gets[2].head.starts_with("GET /actual.png?raw=1 "));
        assert!(gets[3].head.starts_with("GET /figure?variant=two "));
        assert_eq!(vision_bytes(&requests(&model)[1]), vec![bytes]);
    }
}

#[test]
fn embedded_epub_binary_asset_is_analyzed_without_network_localization() {
    if isolated("embedded_epub_binary_asset_is_analyzed_without_network_localization") {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("book.epub");
    let bytes = png();
    let mut archive = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    for (name, body) in [
        ("META-INF/container.xml", b"<container><rootfiles><rootfile full-path='book/package.opf'/></rootfiles></container>".as_slice()),
        ("book/package.opf", b"<package><metadata><title>Embedded image</title></metadata><manifest><item id='chapter' href='chapter.xhtml' media-type='application/xhtml+xml'/><item id='figure' href='chart.png' media-type='image/png'/></manifest><spine><itemref idref='chapter'/></spine></package>".as_slice()),
        ("book/chapter.xhtml", b"<html xmlns='http://www.w3.org/1999/xhtml'><body><p>Embedded figure.</p><p><img src='chart.png' alt='Author caption'/></p></body></html>".as_slice()),
        ("book/chart.png", bytes.as_slice()),
    ] {
        archive.start_file(name, zip::write::SimpleFileOptions::default()).unwrap();
        archive.write_all(body).unwrap();
    }
    std::fs::write(&source, archive.finish().unwrap().into_inner()).unwrap();
    let server = Server::new(vec![Reply::echo(), analysis()]);
    let output = run(
        source.to_str().unwrap(),
        cfg(&server),
        Some(dir.path().join("out")),
    )
    .unwrap();
    assert_eq!(output.images.len(), 1, "{output:#?}");
    assert!(body(&output).contains("![A \\[safe\\] chart]"));
    assert_eq!(
        std::fs::read(output.images[0]["asset"].as_str().unwrap()).unwrap(),
        bytes
    );
    assert_eq!(requests(&server).len(), 2);
}

#[test]
fn alt_only_does_not_create_description_sidecar_and_no_model_keeps_error_kind() {
    if isolated("alt_only_does_not_create_description_sidecar_and_no_model_keeps_error_kind") {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("chart.png");
    std::fs::write(&source, png()).unwrap();
    let server = Server::new(vec![analysis()]);
    let mut config = cfg(&server);
    config["image"]["desc_enabled"] = json!(false);
    let output = run(
        source.to_str().unwrap(),
        config.clone(),
        Some(dir.path().join("out")),
    )
    .unwrap();
    assert_eq!(output.images.len(), 1);
    assert!(
        !Path::new(output.images[0]["asset"].as_str().unwrap())
            .parent()
            .unwrap()
            .join("images.json")
            .exists()
    );
    config["llm"]["model_list"] = json!([]);
    let error = run(source.to_str().unwrap(), config, None).unwrap_err();
    assert!(matches!(error, Error::NoModelConfigured), "{error:?}");
    assert_eq!(requests(&server).len(), 1);
}

#[test]
fn standalone_multiframe_tiff_sends_every_preview_once_and_keeps_original_download() {
    if isolated("standalone_multiframe_tiff_sends_every_preview_once_and_keeps_original_download") {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("pages.tiff");
    let mut bytes = std::io::Cursor::new(Vec::new());
    {
        let mut encoder = tiff::encoder::TiffEncoder::new(&mut bytes).unwrap();
        for color in [[200, 20, 30], [20, 200, 30]] {
            let pixels: Vec<u8> = (0..100 * 100).flat_map(|_| color).collect();
            encoder
                .write_image::<tiff::encoder::colortype::RGB8>(100, 100, &pixels)
                .unwrap();
        }
    }
    std::fs::write(&source, bytes.into_inner()).unwrap();
    let server=Server::new(vec![Reply::model(&json!({"caption":"Two pages", "description":"First red page, then green page.","extracted_text":"Page one\nPage two"}).to_string())]);
    let mut config = cfg(&server);
    config["llm"]["max_vision_pages_per_document"] = json!(2);
    let output = run(source.to_str().unwrap(), config, None).unwrap();
    assert_eq!(output.usage.requests, 1);
    assert_eq!(output.images.len(), 2);
    assert!(body(&output).contains("[Original TIFF]"));
    for number in [1, 2] {
        assert!(body(&output).contains(&format!("<!-- Page number: {number} -->")));
    }
    assert!(body(&output).contains("First red page, then green page."));
    assert!(body(&output).contains("Page one\nPage two"));
    assert_eq!(body(&output).matches("# pages\n").count(), 1);
    assert_eq!(output.images[1]["llm_usage"], json!({}));
    let calls = requests(&server);
    assert_eq!(calls.len(), 1);
    let frames = vision_bytes(&calls[0]);
    assert_eq!(frames.len(), 2);
    assert_eq!(
        image::load_from_memory(&frames[0])
            .unwrap()
            .to_rgb8()
            .get_pixel(0, 0)
            .0,
        [200, 20, 30]
    );
    assert_eq!(
        image::load_from_memory(&frames[1])
            .unwrap()
            .to_rgb8()
            .get_pixel(0, 0)
            .0,
        [20, 200, 30]
    );
}

fn mime_part(headers: &str, content: &[u8]) -> String {
    use base64::Engine;
    format!(
        "{headers}\r\nContent-Transfer-Encoding: base64\r\n\r\n{}\r\n",
        base64::engine::general_purpose::STANDARD.encode(content)
    )
}
fn multipart_email(kind: &str, members: &[String]) -> Vec<u8> {
    let mut message = format!(
        "MIME-Version: 1.0\r\nSubject: Related chart\r\nContent-Type: multipart/{kind}; boundary=mail-boundary\r\n\r\n"
    );
    for member in members {
        message.push_str("--mail-boundary\r\n");
        message.push_str(member);
    }
    message.push_str("--mail-boundary--\r\n");
    message.into_bytes()
}

#[test]
fn real_eml_cid_references_reach_alt_description_and_published_metadata_in_both_profiles() {
    if isolated(
        "real_eml_cid_references_reach_alt_description_and_published_metadata_in_both_profiles",
    ) {
        return;
    }
    for profile in [None, Some("rag")] {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("related.eml");
        let bytes = png();
        let html = b"<p>Inline chart and literal cid:logo@example.test.</p><p><img src='CID:%3Clogo%40example.test%3E' alt='Original one'> and <img src='cid:LOGO@example.test' alt='Original two'></p><pre><code>&lt;img src=\"cid:logo@example.test\"&gt;</code></pre><p><img src='cid:ordinary' alt='Not an image MIME part'></p>";
        std::fs::write(&source, multipart_email("related", &[
            mime_part("Content-Type: text/html; charset=utf-8", html),
            mime_part("Content-Type: image/png\r\nContent-ID: <logo@example.test>\r\nContent-Disposition: inline; filename=\"../../chart [x].png\"", &bytes),
            mime_part("Content-Type: application/octet-stream\r\nContent-ID: <ordinary>\r\nContent-Disposition: attachment; filename=\"not-a-picture.png\"", b"NON_IMAGE_ATTACHMENT"),
        ])).unwrap();
        let server = Server::new(vec![Reply::echo(), analysis()]);
        let mut config = cfg(&server);
        if let Some(profile) = profile {
            config["output"] = json!({"profile":profile});
        }
        let output = run(
            source.to_str().unwrap(),
            config,
            Some(dir.path().join("out")),
        )
        .unwrap();
        assert_eq!(output.images.len(), 1, "{output:#?}");
        // Both cid references and the reference-style attachment listing
        // name the same analyzed asset.
        assert_eq!(
            body(&output).matches("![A \\[safe\\] chart]").count(),
            3,
            "{}",
            body(&output)
        );
        assert!(body(&output).contains("literal cid:logo@example.test"));
        assert!(body(&output).contains("<img src=\"cid:logo@example.test\">"));
        assert!(body(&output).contains("cid:ordinary"));
        assert!(
            output
                .warnings
                .iter()
                .any(|warning| warning.contains("Content-ID") && warning.contains("ordinary"))
        );
        assert!(output.markdown.contains("![Original one]"));
        assert!(output.markdown.contains("## Attachments"));
        assert!(output.markdown.contains("- [not-a-picture.png]("));
        let asset = Path::new(output.images[0]["asset"].as_str().unwrap());
        assert!(asset.is_absolute());
        assert_eq!(std::fs::read(asset).unwrap(), bytes);
        assert_eq!(
            asset.parent().unwrap(),
            if profile.is_some() {
                dir.path().join("out/assets")
            } else {
                dir.path().join("out/.markitai/assets")
            }
        );
        let index: Value = serde_json::from_slice(
            &std::fs::read(asset.parent().unwrap().join("images.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(index["images"].as_array().unwrap().len(), 1);
        assert_eq!(index["images"][0]["path"], output.images[0]["asset"]);
        assert_eq!(index["images"][0]["source"], source.to_str().unwrap());
        assert_eq!(requests(&server).len(), 2);
        assert_eq!(vision_bytes(&requests(&server)[1]), vec![bytes]);
        assert!(
            output
                .assets
                .iter()
                .any(|path| std::fs::read(path).unwrap() == b"NON_IMAGE_ATTACHMENT")
        );
        assert!(!dir.path().join("chart [x].png").exists());
    }
}

#[test]
fn ambiguous_eml_content_ids_never_choose_an_arbitrary_image() {
    if isolated("ambiguous_eml_content_ids_never_choose_an_arbitrary_image") {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("ambiguous.eml");
    #[cfg(unix)]
    std::fs::write(dir.path().join("cid:missing"), png()).unwrap();
    std::fs::write(&source, multipart_email("related", &[
        mime_part("Content-Type: text/html; charset=utf-8", b"<p>Keep this body.</p><img src='cid:duplicate' alt='Ambiguous author alt'><img src='cid:missing' alt='Missing author alt'>"),
        mime_part("Content-Type: image/png\r\nContent-ID: <duplicate>\r\nContent-Disposition: inline; filename=first.png", &png()),
        mime_part("Content-Type: image/png\r\nContent-ID: <duplicate>\r\nContent-Disposition: inline; filename=second.png", &png()),
    ])).unwrap();
    let server = Server::new(vec![Reply::echo(), analysis()]);
    let output = run(
        source.to_str().unwrap(),
        cfg(&server),
        Some(dir.path().join("out")),
    )
    .unwrap();
    // Neither ambiguous reference binds. The listed attachments, identical
    // bytes, are one asset analyzed once, as the reference analyzes attachment
    // images.
    assert_eq!(output.images.len(), 1, "{output:#?}");
    assert!(body(&output).contains("![Ambiguous author alt](cid:duplicate)"));
    assert!(body(&output).contains("![Missing author alt](cid:missing)"));
    assert_eq!(
        output
            .warnings
            .iter()
            .filter(|warning| warning.contains("EML image reference"))
            .count(),
        2
    );
    assert_eq!(requests(&server).len(), 2);
    assert_eq!(output.usage.requests, 2);
    assert_eq!(vision_bytes(&requests(&server)[1]), vec![png()]);
    assert_eq!(
        body(&output)
            .split_once("## Attachments")
            .unwrap()
            .1
            .matches("![A \\[safe\\] chart](.markitai/assets/")
            .count(),
        2
    );
}

#[test]
fn cached_main_document_with_new_image_analysis_is_not_a_full_cache_hit() {
    if isolated("cached_main_document_with_new_image_analysis_is_not_a_full_cache_hit") {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let bytes = png();
    std::fs::write(dir.path().join("figure.png"), &bytes).unwrap();
    let source = dir.path().join("report.md");
    std::fs::write(
        &source,
        "# Cached report\n\nOriginal introduction.\n\n![Author figure](figure.png)\n\nOriginal conclusion.\n",
    )
    .unwrap();
    let server = Server::new(vec![Reply::echo(), analysis(), analysis()]);
    let mut config = cfg(&server);
    config["cache"]["enabled"] = json!(true);
    config["cache"]["global_dir"] = json!(dir.path().join("private-cache"));
    let first = run(source.to_str().unwrap(), config.clone(), None).unwrap();
    assert_eq!(first.usage.requests, 2);
    assert!(!first.llm_cache_hit());
    let second = run(source.to_str().unwrap(), config, None).unwrap();
    assert_eq!(second.usage.requests, 1);
    assert_eq!(second.usage.input_tokens, 7);
    assert_eq!(second.usage.output_tokens, 5);
    assert!(!second.llm_cache_hit());
    assert_eq!(second.images.len(), 1);
    assert_eq!(body(&second), body(&first));
    assert!(body(&second).contains("Original introduction."));
    assert!(body(&second).contains("Original conclusion."));
    assert!(body(&second).contains("![A \\[safe\\] chart]"));
    assert_eq!(second.frontmatter["description"], "Image document fixture");
    let calls = requests(&server);
    assert_eq!(calls.len(), 3);
    assert!(
        calls[0]["messages"][0]["content"]
            .as_str()
            .unwrap()
            .contains("MARKITAI_DOCUMENT_JSON_V1")
    );
    assert_eq!(vision_bytes(&calls[1]), vec![bytes.clone()]);
    assert_eq!(vision_bytes(&calls[2]), vec![bytes]);
}

#[cfg(test)]
mod bounded_fixture_io {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/support/bounded_read.rs"
    ));
}
