//! Actual service static routing and browser bootstrap contract. Browser rendering
//! and interactive behavior have separate real-browser acceptance.
#![cfg(unix)]
use serde_json::json;
use std::{
    collections::HashMap,
    io::{BufRead, BufReader, Read, Write},
    net::TcpStream,
    path::Path,
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
    thread::JoinHandle,
    time::{Duration, Instant},
};

struct Server {
    child: Child,
    reader: Option<JoinHandle<()>>,
    port: u16,
}
impl Server {
    fn start(root: &Path) -> Self {
        std::fs::write(root.join("config.json"),json!({"llm":{"enabled":false},"cache":{"enabled":false},"history":{"record":false},"log":{"dir":null},"prompts":{"dir":root.join("absent-prompts")}}).to_string()).unwrap();
        let mut command = Command::new(env!("CARGO_BIN_EXE_markitai"));
        command.env_clear();
        for key in ["HOME", "PATH", "TMPDIR", "SYSTEMROOT"] {
            if let Some(value) = std::env::var_os(key) {
                command.env(key, value);
            }
        }
        let mut child = command
            .current_dir(root)
            .env("MARKITAI_HOME", root.join("home"))
            .args([
                "-c",
                "config.json",
                "serve",
                "--host",
                "127.0.0.1",
                "--port",
                "0",
                "--no-open",
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let output = Arc::new(Mutex::new(String::new()));
        let captured = output.clone();
        let stderr = child.stderr.take().unwrap();
        let reader = std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                let mut text = captured.lock().unwrap_or_else(|e| e.into_inner());
                text.push_str(&line);
                text.push('\n');
            }
        });
        let mut server = Self {
            child,
            reader: Some(reader),
            port: 0,
        };
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            let text = output.lock().unwrap_or_else(|e| e.into_inner()).clone();
            if let Some(port) = text.lines().find_map(|line| {
                line.strip_prefix("Markitai server listening on http://127.0.0.1:")
                    .and_then(|value| value.parse().ok())
            }) {
                server.port = port;
                return server;
            }
            assert!(
                server.child.try_wait().unwrap().is_none(),
                "server exited: {text}"
            );
            assert!(
                Instant::now() < deadline,
                "server startup timed out: {text}"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }
    fn request(
        &self,
        method: &str,
        path: &str,
        host: Option<&str>,
    ) -> (u16, HashMap<String, String>, Vec<u8>) {
        let mut socket = TcpStream::connect(("127.0.0.1", self.port)).unwrap();
        socket
            .set_read_timeout(Some(Duration::from_secs(10)))
            .unwrap();
        socket
            .set_write_timeout(Some(Duration::from_secs(10)))
            .unwrap();
        write!(socket,"{method} {path} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\nContent-Length: 0\r\n\r\n",host.map(str::to_owned).unwrap_or_else(||format!("127.0.0.1:{}",self.port))).unwrap();
        let mut bytes = Vec::new();
        socket.read_to_end(&mut bytes).unwrap();
        let split = bytes
            .windows(4)
            .position(|part| part == b"\r\n\r\n")
            .unwrap();
        let header = String::from_utf8_lossy(&bytes[..split]);
        let mut lines = header.lines();
        let status = lines
            .next()
            .unwrap()
            .split_whitespace()
            .nth(1)
            .unwrap()
            .parse()
            .unwrap();
        let headers = lines
            .map(|line| {
                let (key, value) = line.split_once(':').unwrap();
                (key.to_ascii_lowercase(), value.trim().to_owned())
            })
            .collect();
        (status, headers, bytes[split + 4..].to_vec())
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
    }
}

#[test]
fn embedded_interface_and_fixed_offline_assets_are_actual_http_resources() {
    let directory = tempfile::tempdir().unwrap();
    let server = Server::start(directory.path());
    let (status, headers, html) = server.request("GET", "/", None);
    assert_eq!(status, 200);
    assert_eq!(headers["content-type"], "text/html; charset=utf-8");
    assert_eq!(headers["cache-control"], "no-cache");
    assert_eq!(headers["referrer-policy"], "no-referrer");
    assert_eq!(headers["x-content-type-options"], "nosniff");
    let policy = &headers["content-security-policy"];
    for expected in [
        "script-src 'self'",
        "connect-src 'self'",
        "frame-src 'none'",
        "object-src 'none'",
        "frame-ancestors 'none'",
    ] {
        assert!(policy.contains(expected), "{policy}");
    }
    assert!(!policy.contains("unsafe-inline") && !policy.contains("unsafe-eval"));
    assert_eq!(html, include_bytes!("../src/server/web/index.html"));
    for (path, expected) in [
        (
            "/ui/app.js",
            include_bytes!("../src/server/web/app.js").as_slice(),
        ),
        (
            "/ui/api.js",
            include_bytes!("../src/server/web/api.js").as_slice(),
        ),
        (
            "/ui/settings.js",
            include_bytes!("../src/server/web/settings.js").as_slice(),
        ),
        (
            "/ui/workspace.js",
            include_bytes!("../src/server/web/workspace.js").as_slice(),
        ),
        (
            "/ui/result-tools.js",
            include_bytes!("../src/server/web/result-tools.js").as_slice(),
        ),
        (
            "/ui/preview.js",
            include_bytes!("../src/server/web/preview.js").as_slice(),
        ),
        (
            "/ui/i18n.js",
            include_bytes!("../src/server/web/i18n.js").as_slice(),
        ),
        (
            "/ui/boot.js",
            include_bytes!("../src/server/web/boot.js").as_slice(),
        ),
        (
            "/ui/marked.js",
            include_bytes!("../../../vendor/web/marked.js").as_slice(),
        ),
        (
            "/ui/purify.js",
            include_bytes!("../../../vendor/web/purify.js").as_slice(),
        ),
    ] {
        let (status, headers, body) = server.request("GET", path, None);
        assert_eq!(status, 200, "{path}");
        assert_eq!(body, expected, "{path}");
        assert_eq!(headers["content-type"], "text/javascript; charset=utf-8");
        assert_eq!(headers["cache-control"], "no-cache");
        let (status, head_headers, body) = server.request("HEAD", path, None);
        assert_eq!(status, 200);
        assert!(body.is_empty());
        assert_eq!(head_headers["content-type"], headers["content-type"]);
    }
    let (status, headers, body) = server.request("GET", "/ui/style.css", None);
    assert_eq!(status, 200);
    assert_eq!(headers["content-type"], "text/css; charset=utf-8");
    assert_eq!(body, include_bytes!("../src/server/web/style.css"));
    let (status, headers, body) = server.request("GET", "/ui/icon.svg", None);
    assert_eq!(status, 200);
    assert_eq!(headers["content-type"], "image/svg+xml");
    assert_eq!(headers["x-content-type-options"], "nosniff");
    assert!(headers["content-security-policy"].contains("default-src 'none'"));
    assert_eq!(body, include_bytes!("../src/server/web/icon.svg"));
}

#[test]
fn ui_does_not_turn_unknown_api_or_working_directory_files_into_spa_routes() {
    let directory = tempfile::tempdir().unwrap();
    std::fs::write(directory.path().join("private.txt"), "not-a-web-resource").unwrap();
    let server = Server::start(directory.path());
    for path in [
        "/api/not-real",
        "/ui/not-real.js",
        "/ui/../../private.txt",
        "/private.txt",
        "/ui/index.html",
    ] {
        let (status, headers, body) = server.request("GET", path, None);
        assert_eq!(status, 404, "{path}");
        assert!(headers["content-type"].starts_with("application/json"));
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["code"], "not_found");
        assert!(!String::from_utf8_lossy(&body).contains("not-a-web-resource"));
    }
    let (status, _, _) = server.request("GET", "/", Some("unexpected.invalid"));
    assert_eq!(status, 400);
    let (status, headers, _) = server.request("POST", "/ui/app.js", None);
    assert_eq!(status, 405);
    assert!(headers["content-type"].starts_with("application/json"));
}
