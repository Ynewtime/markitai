use crate::console;

use serde_json::{Value, json};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Command, Stdio};
use std::sync::{
    Arc, Condvar, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::time::Duration;

struct ModelServer {
    base: String,
    calls: Arc<AtomicUsize>,
    fetches: Arc<AtomicUsize>,
    peak: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

fn read_request(stream: &mut TcpStream) -> Vec<u8> {
    stream.set_nonblocking(false).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .unwrap();
    let mut request_reader = bounded_fixture_io::Reader::new(
        stream,
        std::time::Instant::now() + Duration::from_secs(10),
    );
    let mut bytes = Vec::new();
    loop {
        let mut chunk = [0; 4096];
        let count = request_reader.read(&mut chunk).unwrap();
        assert_ne!(count, 0, "incomplete mock request");
        bytes.extend_from_slice(&chunk[..count]);
        assert!(bytes.len() < 1024 * 1024);
        if let Some(split) = bytes.windows(4).position(|s| s == b"\r\n\r\n") {
            let headers = String::from_utf8_lossy(&bytes[..split]);
            let length = headers
                .lines()
                .find_map(|line| {
                    line.to_ascii_lowercase()
                        .strip_prefix("content-length:")
                        .and_then(|value| value.trim().parse::<usize>().ok())
                })
                .unwrap_or(0);
            if bytes.len() >= split + 4 + length {
                return bytes;
            }
        }
    }
}

impl ModelServer {
    fn start(expected_peak: usize) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let calls = Arc::new(AtomicUsize::new(0));
        let fetches = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let (count, pages, maximum, stopped) =
            (calls.clone(), fetches.clone(), peak.clone(), stop.clone());
        let active = Arc::new(AtomicUsize::new(0));
        let rendezvous = Arc::new((Mutex::new(0), Condvar::new()));
        let thread = std::thread::spawn(move || {
            let mut workers = Vec::new();
            while !stopped.load(Ordering::SeqCst) {
                let (mut stream, _) = match listener.accept() {
                    Ok(connection) => connection,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(2));
                        continue;
                    }
                    Err(error) => panic!("mock accept: {error}"),
                };
                let (count, pages, maximum, active, rendezvous) = (
                    count.clone(),
                    pages.clone(),
                    maximum.clone(),
                    active.clone(),
                    rendezvous.clone(),
                );
                workers.push(std::thread::spawn(move || {
                    let request = read_request(&mut stream);
                    if request.starts_with(b"GET ") {
                        pages.fetch_add(1, Ordering::SeqCst);
                        let body = "<html><body><h1>Source</h1><p>Readable content.</p></body></html>";
                        write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
                        return;
                    }
                    assert!(request.starts_with(b"POST /v1/chat/completions "));
                    let running = active.fetch_add(1, Ordering::SeqCst) + 1;
                    maximum.fetch_max(running, Ordering::SeqCst);
                    count.fetch_add(1, Ordering::SeqCst);
                    {
                        // A serial implementation must fail the peak=2 check even
                        // on a busy host; the first request waits for its peer.
                        let (lock, changed) = &*rendezvous;
                        let mut seen = lock.lock().unwrap();
                        *seen += 1;
                        changed.notify_all();
                        let _ = changed
                            .wait_timeout_while(seen, Duration::from_secs(3), |seen| {
                                *seen < expected_peak
                            })
                            .unwrap();
                    }
                    std::thread::sleep(Duration::from_millis(75));
                    let split=request.windows(4).position(|s|s==b"\r\n\r\n").unwrap();
                    let payload:Value=serde_json::from_slice(&request[split+4..]).unwrap();
                    let body = json!({"choices":[{"message":{"content":model_content(&payload,"# Enhanced\n\nConcurrent 世界")},"finish_reason":"stop"}],"usage":{"prompt_tokens":9,"completion_tokens":5}}).to_string();
                    // End the observed HTTP work before releasing the full response
                    // to the client, so a correctly reused slot cannot look active.
                    active.fetch_sub(1, Ordering::SeqCst);
                    write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
                }));
            }
            for worker in workers {
                worker.join().unwrap();
            }
        });
        Self {
            base,
            calls,
            fetches,
            peak,
            stop,
            thread: Some(thread),
        }
    }
}

impl Drop for ModelServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        let result = self.thread.take().unwrap().join();
        if !std::thread::panicking() {
            result.unwrap();
        }
    }
}

fn check_batch(urls: bool) {
    for limit in [1, 2] {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let server = ModelServer::start(limit);
        let cfg = json!({
            "llm":{"enabled":true,"concurrency":3,"on_failure":"fail","router_settings":{"num_retries":0},
                "model_list":[{"model_name":"mock","litellm_params":{"model":"openai/concurrency-test","api_base":format!("{}/v1",server.base),"api_key":"local-test"}}]},
            "image":{"alt_enabled":false,"desc_enabled":false},
            "cache":{"enabled":false},"fetch":{"strategy":"static"},
            "output":{"on_conflict":"overwrite"}
        });
        std::fs::write(root.join("markitai.json"), cfg.to_string()).unwrap();
        let source = if urls {
            let sources = (0..6)
                .map(|i| format!("{}/doc{i}", server.base))
                .collect::<Vec<_>>()
                .join("\n");
            std::fs::write(root.join("input.urls"), sources).unwrap();
            "input.urls"
        } else {
            std::fs::create_dir(root.join("inputs")).unwrap();
            for i in 0..6 {
                std::fs::write(
                    root.join(format!("inputs/doc{i}.md")),
                    format!("# Source {i}\n\nContent {i}.\n"),
                )
                .unwrap();
            }
            "inputs"
        };
        let mut command = Command::new(env!("CARGO_BIN_EXE_markitai"));
        console::english(command.env_clear());
        for key in ["PATH", "SYSTEMROOT", "TMPDIR", "TMP", "TEMP"] {
            if let Some(value) = std::env::var_os(key) {
                command.env(key, value);
            }
        }
        let output = command
            .current_dir(root)
            .env("MARKITAI_HOME", root.join("home"))
            .env("NO_PROXY", "127.0.0.1,localhost")
            .stdin(Stdio::null())
            .args([
                source,
                "-o",
                "out",
                "--json",
                "--batch-concurrency",
                "4",
                "--url-concurrency",
                "4",
                "--llm-concurrency",
                &limit.to_string(),
            ])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let envelope: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(envelope["version"], "1.0");
        assert_eq!(envelope["ok"], true);
        let items = envelope["items"].as_array().unwrap();
        assert_eq!(items.len(), 6);
        for item in items {
            assert_eq!(item["status"], "completed");
            assert_eq!(item["llm_cache_hit"], false);
            assert!(!item["llm_usage"].as_object().unwrap().is_empty());
        }
        assert_eq!(server.calls.load(Ordering::SeqCst), 6);
        assert_eq!(
            server.fetches.load(Ordering::SeqCst),
            if urls { 6 } else { 0 }
        );
        assert_eq!(server.peak.load(Ordering::SeqCst), limit);
    }
}

#[test]
fn directory_workers_share_the_cli_llm_limit() {
    check_batch(false);
}

#[test]
fn url_list_workers_share_the_cli_llm_limit() {
    check_batch(true);
}

// Document cleanup is structured; pure vision and connection probes remain text.
fn model_content(request: &Value, markdown: &str) -> String {
    let messages = request["messages"].as_array().unwrap();
    if !messages.iter().any(|message| {
        message["role"] == "system"
            && message["content"]
                .as_str()
                .is_some_and(|text| text.contains("MARKITAI_DOCUMENT_JSON_V1"))
    }) {
        return markdown.to_owned();
    }
    // Echo protected source spans once and in order; test-specific output still
    // identifies the model invocation/generation used by the existing assertions.
    let source = messages
        .iter()
        .filter(|message| message["role"] == "user")
        .filter_map(|message| message["content"].as_str())
        .collect::<Vec<_>>()
        .join("\n");
    json!({"cleaned_markdown":format!("{markdown}\n\n{source}"),"frontmatter":{"description":"Local test document","tags":["fixture"]}}).to_string()
}

#[cfg(test)]
mod bounded_fixture_io {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/support/bounded_read.rs"
    ));
}
