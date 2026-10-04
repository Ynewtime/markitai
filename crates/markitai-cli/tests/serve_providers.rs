#![cfg(unix)]

use serde_json::{Value, json};
use std::{
    collections::HashMap,
    io::{BufRead, BufReader, Read, Write},
    net::{TcpListener, TcpStream},
    path::Path,
    process::{Child, Command, Stdio},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

struct Server {
    child: Child,
    port: u16,
    logs: Arc<Mutex<String>>,
    reader: Option<thread::JoinHandle<()>>,
}
impl Server {
    fn start(root: &Path) -> Self {
        let mut command = Command::new(env!("CARGO_BIN_EXE_markitai"));
        command.env_clear();
        for key in ["PATH", "HOME", "TMPDIR", "SYSTEMROOT"] {
            if let Some(value) = std::env::var_os(key) {
                command.env(key, value);
            }
        }
        let mut child = command
            .current_dir(root)
            .env("MARKITAI_HOME", root.join("home"))
            .env("NO_PROXY", "127.0.0.1,localhost")
            .args([
                "--config",
                "config.json",
                "serve",
                "--host",
                "127.0.0.1",
                "--port",
                "0",
                "--no-open",
                "--no-auth",
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let logs = Arc::new(Mutex::new(String::new()));
        let captured = logs.clone();
        let stderr = child.stderr.take().unwrap();
        let reader = thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                let mut logs = captured.lock().unwrap();
                logs.push_str(&line);
                logs.push('\n');
            }
        });
        let deadline = Instant::now() + Duration::from_secs(20);
        let port = loop {
            let current = logs.lock().unwrap().clone();
            if let Some(port) = current.lines().find_map(|line| {
                line.strip_prefix("Markitai server listening on http://127.0.0.1:")
                    .and_then(|port| port.parse().ok())
            }) {
                break port;
            }
            if let Some(exit) = child.try_wait().unwrap() {
                panic!("service {exit}: {current}");
            }
            assert!(Instant::now() < deadline, "service startup: {current}");
            thread::sleep(Duration::from_millis(10));
        };
        Self {
            child,
            port,
            logs,
            reader: Some(reader),
        }
    }
    fn request(
        &self,
        method: &str,
        path: &str,
        body: Option<&Value>,
    ) -> (u16, HashMap<String, String>, Value) {
        let text = body.map(ToString::to_string).unwrap_or_default();
        let mut stream = TcpStream::connect(("127.0.0.1", self.port)).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(20)))
            .unwrap();
        write!(stream,"{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{text}",self.port,text.len()).unwrap();
        let mut raw = String::new();
        stream.read_to_string(&mut raw).unwrap();
        let (head, body) = raw.split_once("\r\n\r\n").unwrap();
        let status = head.split_whitespace().nth(1).unwrap().parse().unwrap();
        let headers = head
            .lines()
            .skip(1)
            .filter_map(|line| {
                line.split_once(':')
                    .map(|(key, value)| (key.to_ascii_lowercase(), value.trim().to_string()))
            })
            .collect();
        (
            status,
            headers,
            serde_json::from_str(body).unwrap_or_else(|_| panic!("JSON response: {raw}")),
        )
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        let _ = Command::new("/bin/kill")
            .args(["-TERM", &self.child.id().to_string()])
            .status();
        let deadline = Instant::now() + Duration::from_secs(3);
        loop {
            if self.child.try_wait().ok().flatten().is_some() {
                break;
            }
            if Instant::now() > deadline {
                let _ = self.child.kill();
                let _ = self.child.wait();
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
    }
}
struct Provider {
    base: String,
    seen: Arc<Mutex<Vec<String>>>,
    stop: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<()>>,
}
impl Provider {
    fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let base = format!("http://{}/v1", listener.local_addr().unwrap());
        let seen = Arc::new(Mutex::new(Vec::new()));
        let captured = seen.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let finish = stop.clone();
        let worker = thread::spawn(move || {
            while !finish.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        stream.set_nonblocking(false).unwrap();
                        stream
                            .set_read_timeout(Some(Duration::from_secs(3)))
                            .unwrap();
                        let mut request_reader = bounded_fixture_io::Reader::new(
                            &stream,
                            std::time::Instant::now() + Duration::from_secs(3),
                        );
                        let mut bytes = Vec::new();
                        let mut buf = [0; 4096];
                        loop {
                            let n = request_reader.read(&mut buf).unwrap();
                            assert!(n > 0);
                            bytes.extend_from_slice(&buf[..n]);
                            if let Some(end) =
                                bytes.windows(4).position(|value| value == b"\r\n\r\n")
                            {
                                let header = String::from_utf8_lossy(&bytes[..end]);
                                let length = header
                                    .lines()
                                    .find_map(|line| {
                                        line.to_ascii_lowercase()
                                            .strip_prefix("content-length:")
                                            .and_then(|n| n.trim().parse::<usize>().ok())
                                    })
                                    .unwrap_or(0);
                                if bytes.len() >= end + 4 + length {
                                    break;
                                }
                            }
                        }
                        let request = String::from_utf8(bytes).unwrap();
                        let failed = request.contains("rejected-test-model");
                        let get = request.starts_with("GET ");
                        captured.lock().unwrap().push(request);
                        let (status, body) = if get {
                            (
                                200,
                                r#"{"data":[{"id":"specific-test-model"},{"id":"another-model"}]}"#,
                            )
                        } else if failed {
                            (
                                401,
                                r#"{"error":"private-provider-key http://secret-base/path?token=secret"}"#,
                            )
                        } else {
                            (200, r#"{"choices":[{"message":{"content":"CONNECTED"}}]}"#)
                        };
                        let _ = write!(
                            stream,
                            "HTTP/1.1 {status} response\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                            body.len()
                        );
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2))
                    }
                    Err(error) => panic!("provider: {error}"),
                }
            }
        });
        Self {
            base,
            seen,
            stop,
            worker: Some(worker),
        }
    }
}
impl Drop for Provider {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(worker) = self.worker.take() {
            let result = worker.join();
            if !thread::panicking() {
                result.unwrap();
            }
        }
    }
}

#[test]
fn actual_service_discovers_cached_models_and_probes_stored_or_draft_without_mutation() {
    let root = tempfile::tempdir().unwrap();
    let backend = Provider::start();
    let config = json!({"llm":{"enabled":false,"model_list":[{"model_name":"default","litellm_params":{"model":"openai/specific-test-model","api_key":"private-provider-key","api_base":backend.base}}]},"ocr":{"enabled":false},"cache":{"enabled":false},"history":{"record":false},"log":{"dir":null},"prompts":{"dir":root.path().join("prompts")}});
    let original = serde_json::to_vec(&config).unwrap();
    std::fs::write(root.path().join("config.json"), &original).unwrap();
    let server = Server::start(root.path());
    let (status, headers, settings) = server.request("GET", "/api/settings/llm", None);
    assert_eq!(status, 200);
    assert_eq!(headers["cache-control"], "no-store");
    let id = settings["deployments"][0]["deployment_id"]
        .as_str()
        .unwrap();
    let discovery = json!({"provider":"openai","deployment_id":id});
    for cached in [false, true] {
        let (status, headers, result) = server.request(
            "POST",
            "/api/settings/llm/model-discovery",
            Some(&discovery),
        );
        assert_eq!(status, 200, "{result}");
        assert_eq!(headers["cache-control"], "no-store");
        assert_eq!(result["cached"], cached);
        assert_eq!(result["models"].as_array().unwrap().len(), 2);
    }
    assert_eq!(backend.seen.lock().unwrap().len(), 1);
    let (status, headers, result) = server.request(
        "POST",
        "/api/settings/llm/test",
        Some(&json!({"deployment_id":id})),
    );
    assert_eq!(status, 200, "{result}");
    assert_eq!(headers["cache-control"], "no-store");
    assert_eq!(result["ok"], true);
    let requests = backend.seen.lock().unwrap().clone();
    assert_eq!(requests.len(), 2);
    assert!(
        requests
            .iter()
            .all(|request| request.contains("authorization: Bearer private-provider-key"))
    );
    let body: Value = serde_json::from_str(requests[1].split_once("\r\n\r\n").unwrap().1).unwrap();
    assert_eq!(body["model"], "specific-test-model");
    assert_eq!(body["max_tokens"], 16);
    assert_eq!(
        body["messages"],
        json!([{"role":"user","content":"Reply with exactly OK."}])
    );
    let (_,_,failed)=server.request("POST","/api/settings/llm/test",Some(&json!({"model":"openai/rejected-test-model","api_key":"private-provider-key","api_base":backend.base})));
    assert_eq!(failed["ok"], false);
    assert!(!failed.to_string().contains("private-provider-key"));
    assert!(!failed.to_string().contains("secret-base"));
    assert_eq!(backend.seen.lock().unwrap().len(), 3);
    let (status, headers, _) = server.request(
        "POST",
        "/api/settings/llm/test",
        Some(&json!({"deployment_id":id,"model":"openai/other"})),
    );
    assert_eq!(status, 422);
    assert_eq!(headers["cache-control"], "no-store");
    assert_eq!(backend.seen.lock().unwrap().len(), 3);
    let (status, headers, detected) = server.request("GET", "/api/settings/llm/detected", None);
    assert_eq!(status, 200);
    assert_eq!(headers["cache-control"], "no-store");
    assert_eq!(detected, json!([]));
    assert_eq!(
        std::fs::read(root.path().join("config.json")).unwrap(),
        original
    );
    assert!(!server.logs.lock().unwrap().contains("private-provider-key"));
}

#[test]
fn actual_service_probes_and_discovers_with_user_dotenv_credentials() {
    // Conversion reads `MARKITAI_HOME/.env`; the settings Test and discovery
    // buttons must resolve the same keys instead of sending none (HTTP 401).
    let root = tempfile::tempdir().unwrap();
    let backend = Provider::start();
    std::fs::create_dir_all(root.path().join("home")).unwrap();
    std::fs::write(
        root.path().join("home/.env"),
        "DEEPSEEK_API_KEY=dotenv-fallback-key\nDOTENV_REFERENCE_KEY=dotenv-reference-key\n",
    )
    .unwrap();
    let config = json!({"llm":{"enabled":false,"model_list":[
        {"model_name":"default","litellm_params":{"model":"deepseek/specific-test-model","api_base":backend.base}},
        {"model_name":"default","litellm_params":{"model":"openai/specific-test-model","api_key":"env:DOTENV_REFERENCE_KEY","api_base":backend.base}}
    ]},"ocr":{"enabled":false},"cache":{"enabled":false},"history":{"record":false},"log":{"dir":null},"prompts":{"dir":root.path().join("prompts")}});
    std::fs::write(root.path().join("config.json"), config.to_string()).unwrap();
    let server = Server::start(root.path());
    let (status, _, settings) = server.request("GET", "/api/settings/llm", None);
    assert_eq!(status, 200, "{settings}");
    let ids: Vec<String> = settings["deployments"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| entry["deployment_id"].as_str().unwrap().to_owned())
        .collect();
    assert_eq!(ids.len(), 2);
    for (index, (id, key)) in ids
        .iter()
        .zip(["dotenv-fallback-key", "dotenv-reference-key"])
        .enumerate()
    {
        let (status, _, result) = server.request(
            "POST",
            "/api/settings/llm/test",
            Some(&json!({"deployment_id":id})),
        );
        assert_eq!(status, 200, "{result}");
        assert_eq!(result["ok"], true, "{result}");
        let requests = backend.seen.lock().unwrap().clone();
        assert_eq!(requests.len(), index + 1);
        assert!(
            requests[index].contains(&format!("authorization: Bearer {key}")),
            "probe {index} sent no dotenv key"
        );
    }
    let (status, _, result) = server.request(
        "POST",
        "/api/settings/llm/model-discovery",
        Some(&json!({"provider":"openai","deployment_id":ids[1]})),
    );
    assert_eq!(status, 200, "{result}");
    assert_eq!(result["models"].as_array().unwrap().len(), 2, "{result}");
    let requests = backend.seen.lock().unwrap().clone();
    assert!(requests[2].starts_with("GET "));
    assert!(requests[2].contains("authorization: Bearer dotenv-reference-key"));
    assert!(!server.logs.lock().unwrap().contains("dotenv-"));
}

#[cfg(test)]
mod bounded_fixture_io {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/support/bounded_read.rs"
    ));
}
