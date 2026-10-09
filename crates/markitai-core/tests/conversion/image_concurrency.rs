//! A document's image analysis runs beside its enhancement, within
//! `llm.concurrency`.
use super::*;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::thread;
use std::time::{Duration, Instant};

/// Answers each connection on its own thread and holds every answer until
/// `hold` requests are in flight together (or a second passed), recording
/// the largest number in flight.
struct Model {
    base: String,
    requests: Arc<Mutex<Vec<Value>>>,
    peak: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<()>>,
}

impl Model {
    fn new(hold: usize, answer: fn(&Value) -> String) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let base = format!("http://{}/v1", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let peak = Arc::new(AtomicUsize::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let (captured, highest, stopping) = (requests.clone(), peak.clone(), stop.clone());
        let worker = thread::spawn(move || {
            let active = Arc::new(AtomicUsize::new(0));
            let mut connections = Vec::new();
            while !stopping.load(Ordering::Acquire) {
                let mut stream = match listener.accept() {
                    Ok((stream, _)) => stream,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2));
                        continue;
                    }
                    Err(error) => panic!("mock accept failed: {error}"),
                };
                let (captured, highest, active) =
                    (captured.clone(), highest.clone(), active.clone());
                connections.push(thread::spawn(move || {
                    stream.set_nonblocking(false).unwrap();
                    let request = read(&stream);
                    let now = active.fetch_add(1, Ordering::AcqRel) + 1;
                    highest.fetch_max(now, Ordering::AcqRel);
                    let deadline = Instant::now() + Duration::from_secs(1);
                    while active.load(Ordering::Acquire) < hold && Instant::now() < deadline {
                        thread::sleep(Duration::from_millis(2));
                    }
                    let content = answer(&request);
                    captured.lock().unwrap().push(request);
                    // A client holds its permit until it has read the answer,
                    // so leaving before the answer never overstates the peak.
                    active.fetch_sub(1, Ordering::AcqRel);
                    let body = json!({"model":"mock","choices":[{"message":{"content":content},"finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":5}}).to_string();
                    write!(stream, "HTTP/1.1 200 Mock\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
                }));
            }
            for connection in connections {
                connection.join().unwrap();
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
    fn requests(&self) -> Vec<Value> {
        self.requests.lock().unwrap().clone()
    }
}
impl Drop for Model {
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

fn read(stream: &std::net::TcpStream) -> Value {
    let mut reader =
        bounded_fixture_io::Reader::new(stream, Instant::now() + Duration::from_secs(5));
    let mut bytes = Vec::new();
    loop {
        let mut chunk = [0; 8192];
        let count = reader.read(&mut chunk).unwrap();
        assert!(count > 0, "truncated mock request");
        bytes.extend_from_slice(&chunk[..count]);
        if let Some(end) = bytes.windows(4).position(|s| s == b"\r\n\r\n") {
            let head = String::from_utf8_lossy(&bytes[..end]).to_string();
            let length = head
                .lines()
                .find_map(|line| {
                    let (key, value) = line.split_once(':')?;
                    key.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().unwrap())
                })
                .unwrap_or(0);
            if bytes.len() >= end + 4 + length {
                return serde_json::from_slice(&bytes[end + 4..end + 4 + length]).unwrap();
            }
        }
    }
}

fn is_image(request: &Value) -> bool {
    request
        .pointer("/messages/1/content")
        .is_some_and(Value::is_array)
}

/// The document text comes back as its own cleanup; images get an analysis.
fn faithful(request: &Value) -> String {
    if is_image(request) {
        json!({"caption":"A chart","description":"Two axes.","extracted_text":""}).to_string()
    } else {
        let text = request["messages"][1]["content"].as_str().unwrap();
        json!({"cleaned_markdown":text,"frontmatter":{"description":"A report","tags":["report"]}})
            .to_string()
    }
}

fn png(shade: u8) -> Vec<u8> {
    let mut bytes = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(image::RgbImage::from_fn(100, 100, |x, y| {
        image::Rgb([x as u8, y as u8, shade])
    }))
    .write_to(&mut bytes, image::ImageFormat::Png)
    .unwrap();
    bytes.into_inner()
}

/// A Markdown report referencing two distinct images.
fn report(dir: &std::path::Path) -> String {
    std::fs::write(dir.join("a.png"), png(10)).unwrap();
    std::fs::write(dir.join("b.png"), png(200)).unwrap();
    let source = dir.join("report.md");
    std::fs::write(
        &source,
        "# Report\n\nQuarterly figures.\n\n![First](a.png)\n\n![Second](b.png)\n",
    )
    .unwrap();
    source.to_str().unwrap().to_owned()
}

fn config(model: &Model, concurrency: u64, cap: u64) -> Value {
    json!({
        "cache":{"enabled":false},"history":{"record":false},
        "ocr":{"enabled":false},"screenshot":{"enabled":false},
        "prompts":{"dir":"/nonexistent/markitai-test-prompts"},
        "llm":{"enabled":true,"on_failure":"fallback","concurrency":concurrency,
            "max_requests_per_document":cap,
            "router_settings":{"num_retries":0,"timeout":5},
            "model_list":[{"model_name":"default","litellm_params":{"model":"openai/mock","api_base":model.base,"api_key":"isolated"},"model_info":{"supports_vision":true}}]},
        "image":{"compress":false,"format":"png","alt_enabled":true,"desc_enabled":true}
    })
}

fn run(source: &str, config: Value) -> markitai_core::ConversionOutput {
    convert(
        source,
        ConvertOptions {
            config: Some(config),
            ..Default::default()
        },
    )
    .unwrap()
}

#[test]
fn images_are_analysed_beside_enhancement_within_the_concurrency_setting() {
    let dir = tempfile::tempdir().unwrap();
    let source = report(dir.path());
    // One text request and two image requests run together.
    let model = Model::new(3, faithful);
    let output = run(&source, config(&model, 3, 3));
    assert_eq!(model.peak.load(Ordering::Acquire), 3);
    assert_eq!(output.usage.requests, 3);
    assert_eq!(output.images.len(), 2);
    let enhanced = output.llm_markdown.as_deref().unwrap();
    assert_eq!(enhanced.matches("![A chart]").count(), 2, "{enhanced}");
    assert!(
        !output
            .warnings
            .iter()
            .any(|warning| warning.contains("failed") || warning.contains("sent again")),
        "{:?}",
        output.warnings
    );
    // The image prompt reads the base document, not the enhanced answer.
    let image = model.requests().into_iter().find(is_image).unwrap();
    assert!(
        image["messages"][1]["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("Quarterly figures."),
    );
    drop(model);

    // `llm.concurrency` still bounds them.
    let model = Model::new(2, faithful);
    let output = run(&source, config(&model, 2, 3));
    assert_eq!(model.peak.load(Ordering::Acquire), 2);
    assert_eq!(output.usage.requests, 3);
    assert_eq!(output.images.len(), 2);
}
