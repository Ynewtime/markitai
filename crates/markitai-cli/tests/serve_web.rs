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
            .env("HOME", root.join("home"))
            .env("MARKITAI_HOME", root.join("home"))
            .env("MARKITAI_SERVE_TOKEN", "authored-serve-web-test-token")
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
        self.send(method, path, host, &[])
    }
    fn send(
        &self,
        method: &str,
        path: &str,
        host: Option<&str>,
        extra: &[(&str, &str)],
    ) -> (u16, HashMap<String, String>, Vec<u8>) {
        let mut socket = TcpStream::connect(("127.0.0.1", self.port)).unwrap();
        socket
            .set_read_timeout(Some(Duration::from_secs(10)))
            .unwrap();
        socket
            .set_write_timeout(Some(Duration::from_secs(10)))
            .unwrap();
        let host = host
            .map(str::to_owned)
            .unwrap_or_else(|| format!("127.0.0.1:{}", self.port));
        let mut head = format!(
            "{method} {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\nContent-Length: 0\r\n"
        );
        if path.starts_with("/api/")
            && !extra
                .iter()
                .any(|(name, _)| name.eq_ignore_ascii_case("authorization"))
        {
            head.push_str("Authorization: Bearer authored-serve-web-test-token\r\n");
        }
        for (name, value) in extra {
            head.push_str(&format!("{name}: {value}\r\n"));
        }
        head.push_str("\r\n");
        socket.write_all(head.as_bytes()).unwrap();
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

const CSP: &str = "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' blob:; font-src 'self'; frame-src 'none'; object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'";

fn security_headers(headers: &HashMap<String, String>, path: &str) {
    assert_eq!(headers["cache-control"], "no-cache", "{path}");
    assert_eq!(headers["x-content-type-options"], "nosniff", "{path}");
    assert_eq!(headers["referrer-policy"], "no-referrer", "{path}");
    assert_eq!(headers["content-security-policy"], CSP, "{path}");
    assert!(headers["etag"].starts_with("W/\""), "{path}");
}

#[test]
fn the_workbench_shell_is_served_at_home_and_workspace_addresses() {
    let directory = tempfile::tempdir().unwrap();
    let server = Server::start(directory.path());
    let index = include_bytes!("../src/server/web/dist/index.html").as_slice();
    for path in ["/", "/jobs"] {
        let (status, headers, html) = server.request("GET", path, None);
        assert_eq!(status, 200, "{path}");
        assert_eq!(headers["content-type"], "text/html; charset=utf-8");
        security_headers(&headers, path);
        assert_eq!(html, index, "{path}");
    }
    let page = String::from_utf8_lossy(index);
    assert!(page.contains(r#"<script type="module" src="/ui/app.js"></script>"#));
    assert!(page.contains(r#"<script src="/ui/boot.js"></script>"#));
    assert!(!page.contains("style="));
}

#[test]
fn the_openapi_document_and_the_download_ticket_route_are_served() {
    let directory = tempfile::tempdir().unwrap();
    let server = Server::start(directory.path());
    assert_eq!(
        server
            .send("GET", "/api/openapi.json", None, &[("Authorization", "")])
            .0,
        401,
        "loopback API access still requires the token"
    );
    let (status, headers, body) = server.request("GET", "/api/openapi.json", None);
    assert_eq!(status, 200);
    assert_eq!(headers["content-type"], "application/json");
    let document: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(document["openapi"], "3.1.0");
    assert_eq!(document["info"]["version"], env!("CARGO_PKG_VERSION"));
    for path in [
        "/api/jobs",
        "/api/download-tickets",
        "/api/settings/llm/model-discovery",
    ] {
        assert!(document["paths"][path].is_object(), "{path}");
    }
    // Mounted and validating: an empty body names no download.
    let (status, _, body) = server.send("POST", "/api/download-tickets", None, &[]);
    assert_eq!(status, 422);
    let refusal: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(refusal["reason"], "invalid_ticket_path");
}

#[test]
fn built_assets_are_sent_compressed_or_plain_and_revalidate() {
    let directory = tempfile::tempdir().unwrap();
    let server = Server::start(directory.path());
    let js = "text/javascript; charset=utf-8";
    let compressed: [(&str, &[u8], &[u8], &str); 4] = [
        (
            "/ui/app.js",
            include_bytes!("../src/server/web/dist/app.js.gz"),
            include_bytes!("../src/server/web/dist/app.js"),
            js,
        ),
        (
            "/ui/app.css",
            include_bytes!("../src/server/web/dist/app.css.gz"),
            include_bytes!("../src/server/web/dist/app.css"),
            "text/css; charset=utf-8",
        ),
        (
            "/ui/marked.js",
            include_bytes!("../src/server/web/dist/marked.js.gz"),
            include_bytes!("../../../vendor/web/marked.js"),
            js,
        ),
        (
            "/ui/purify.js",
            include_bytes!("../src/server/web/dist/purify.js.gz"),
            include_bytes!("../../../vendor/web/purify.js"),
            js,
        ),
    ];
    for (path, gz, plain, content_type) in compressed {
        let (status, headers, body) =
            server.send("GET", path, None, &[("Accept-Encoding", "br, gzip")]);
        assert_eq!(status, 200, "{path}");
        assert_eq!(headers["content-encoding"], "gzip", "{path}");
        assert_eq!(headers["vary"], "accept-encoding", "{path}");
        assert_eq!(headers["content-type"], content_type, "{path}");
        security_headers(&headers, path);
        assert_eq!(body, gz, "{path}");
        let etag = headers["etag"].clone();
        let (status, headers, body) = server.request("GET", path, None);
        assert_eq!(status, 200, "{path}");
        assert!(!headers.contains_key("content-encoding"), "{path}");
        assert_eq!(headers["etag"], etag, "{path}");
        assert_eq!(body, plain, "{path}");
        let (status, _, body) = server.send("GET", path, None, &[("Accept-Encoding", "gzip;q=0")]);
        assert_eq!((status, body.as_slice()), (200, plain), "{path}");
        let (status, headers, body) = server.send("GET", path, None, &[("If-None-Match", &etag)]);
        assert_eq!(status, 304, "{path}");
        assert!(body.is_empty(), "{path}");
        assert_eq!(headers["etag"], etag, "{path}");
        let (status, headers, body) = server.request("HEAD", path, None);
        assert_eq!(status, 200, "{path}");
        assert!(body.is_empty(), "{path}");
        assert_eq!(headers["content-type"], content_type, "{path}");
    }
    let plain: [(&str, &[u8], &str); 3] = [
        (
            "/ui/boot.js",
            include_bytes!("../src/server/web/dist/boot.js"),
            js,
        ),
        (
            "/ui/logo.svg",
            include_bytes!("../src/server/web/dist/logo.svg"),
            "image/svg+xml",
        ),
        (
            "/ui/inter-latin-wght.woff2",
            include_bytes!("../../../vendor/web/inter-latin-wght-normal.woff2"),
            "font/woff2",
        ),
    ];
    for (path, expected, content_type) in plain {
        let (status, headers, body) =
            server.send("GET", path, None, &[("Accept-Encoding", "gzip")]);
        assert_eq!(status, 200, "{path}");
        assert!(!headers.contains_key("content-encoding"), "{path}");
        assert_eq!(headers["content-type"], content_type, "{path}");
        security_headers(&headers, path);
        assert_eq!(body, expected, "{path}");
        let (status, _, _) = server.send("GET", path, None, &[("If-None-Match", "\"stale\", *")]);
        assert_eq!(status, 304, "{path}");
    }
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
        "/ui/app.js.gz",
        "/ui/style.css",
        "/jobs/123",
        "/favicon.ico",
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
