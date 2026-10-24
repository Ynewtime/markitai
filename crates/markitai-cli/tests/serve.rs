#![cfg(unix)]
#[path = "support/console.rs"]
mod console;

use serde_json::{Value, json};
use std::{
    collections::HashMap,
    io::{BufRead, BufReader, Read, Write},
    net::{TcpListener, TcpStream},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};
const WAIT: Duration = Duration::from_secs(30);
fn until(label: &str, mut predicate: impl FnMut() -> bool) {
    let deadline = Instant::now() + WAIT;
    while !predicate() {
        assert!(Instant::now() < deadline, "timed out: {label}");
        std::thread::sleep(Duration::from_millis(5));
    }
}
/// A saved job's time `hours` before now, written with an offset as the CLI
/// writes it. History expires seven days after a job finished.
fn hours_ago(hours: i64) -> String {
    let offset = chrono::FixedOffset::east_opt(8 * 3600).unwrap();
    (chrono::Utc::now() - chrono::Duration::hours(hours))
        .with_timezone(&offset)
        .to_rfc3339()
}
fn configure(root: &Path) {
    std::fs::write(root.join("config.json"),json!({"llm":{"enabled":false},"ocr":{"enabled":false},"screenshot":{"enabled":false},"image":{"alt_enabled":false,"desc_enabled":false,"compress":false,"filter":{"min_width":1,"min_height":1,"min_area":1}},"cache":{"enabled":false},"history":{"record":false},"prompts":{"dir":root.join("prompts")},"log":{"dir":null},"fetch":{"strategy":"static","remote_consent":"never"},"batch":{"concurrency":2,"url_concurrency":1},"output":{"on_conflict":"rename","report":false}}).to_string()).unwrap();
}
struct Server {
    child: Option<Child>,
    stderr: Option<JoinHandle<()>>,
    captured: Arc<Mutex<String>>,
    port: u16,
    root: PathBuf,
}
impl Server {
    fn start(root: &Path) -> Self {
        Self::start_with(root, &[])
    }
    /// With `MARKITAI_SERVE_TOKEN` among `variables` the server requires that
    /// token; otherwise it runs with `--no-auth`.
    fn start_with(root: &Path, variables: &[(&str, &std::ffi::OsStr)]) -> Self {
        if !root.join("config.json").exists() {
            configure(root);
        }
        let mut command = Command::new(env!("CARGO_BIN_EXE_markitai"));
        console::english(command.env_clear());
        for key in ["PATH", "HOME", "TMPDIR", "TEMP", "TMP", "SYSTEMROOT"] {
            if let Some(value) = std::env::var_os(key) {
                command.env(key, value);
            }
        }
        command.envs(variables.iter().copied());
        let mut child = command
            .current_dir(root)
            .env("HOME", root)
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
            ])
            .args(
                (!variables
                    .iter()
                    .any(|(name, _)| *name == "MARKITAI_SERVE_TOKEN"))
                .then_some("--no-auth"),
            )
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let captured = Arc::new(Mutex::new(String::new()));
        let copy = captured.clone();
        let reader = child.stderr.take().unwrap();
        let stderr = std::thread::spawn(move || {
            for line in BufReader::new(reader).lines().map_while(Result::ok) {
                let mut text = copy.lock().unwrap();
                text.push_str(&line);
                text.push('\n');
            }
        });
        let mut server = Self {
            child: Some(child),
            stderr: Some(stderr),
            captured,
            port: 0,
            root: root.into(),
        };
        until("server startup", || {
            if let Some(status) = server.child.as_mut().unwrap().try_wait().unwrap() {
                panic!(
                    "server exited {status}: {}",
                    server.captured.lock().unwrap()
                );
            }
            let text = server.captured.lock().unwrap();
            let found = text.lines().find_map(|line| {
                line.split_once("Markitai server listening on http://127.0.0.1:")
                    .and_then(|(_, s)| s.parse().ok())
            });
            if let Some(port) = found {
                server.port = port;
                true
            } else {
                false
            }
        });
        server
    }
    fn request(&self, method: &str, path: &str, headers: &[(&str, &str)], body: &[u8]) -> Reply {
        request(self.port, method, path, headers, body)
    }
    fn json(&self, path: &str) -> Value {
        let reply = self.request("GET", path, &[], &[]);
        assert_eq!(reply.status, 200, "{}", reply.text());
        reply.json()
    }
    fn submit(&self, files: &[(&str, &[u8])], urls: Value, options: Value) -> Value {
        let (content_type, body) = multipart(files, urls, options);
        let reply = self.request(
            "POST",
            "/api/jobs",
            &[("Content-Type", &content_type)],
            &body,
        );
        assert_eq!(reply.status, 201, "{}", reply.text());
        reply.json()
    }
    fn done(&self, id: &str) -> Value {
        let mut value = Value::Null;
        until("job completion", || {
            value = self.json(&format!("/api/jobs/{id}"));
            value["status"] != "running"
        });
        value
    }
    fn signal(&self) {
        assert!(
            Command::new("/bin/kill")
                .args(["-INT", &self.child.as_ref().unwrap().id().to_string()])
                .status()
                .unwrap()
                .success()
        );
        until("shutdown acknowledgement", || {
            self.captured
                .lock()
                .unwrap()
                .contains("Server stopping: draining active conversions and saving history.")
        });
    }
    fn finish(mut self, success: bool) {
        let mut child = self.child.take().unwrap();
        until("server exit", || child.try_wait().unwrap().is_some());
        let status = child.wait().unwrap();
        self.stderr.take().unwrap().join().unwrap();
        assert_eq!(
            status.success(),
            success,
            "status {status}: {}",
            self.captured.lock().unwrap()
        );
    }
    fn stop(self) {
        self.signal();
        self.finish(true);
    }
    fn jobdir(&self, id: &str) -> PathBuf {
        self.root.join("home/serve/jobs").join(id)
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        if let Some(thread) = self.stderr.take() {
            let _ = thread.join();
        }
    }
}
struct Reply {
    status: u16,
    headers: HashMap<String, String>,
    body: Vec<u8>,
}
impl Reply {
    fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }
    fn json(&self) -> Value {
        serde_json::from_slice(&self.body).unwrap()
    }
}
fn request(port: u16, method: &str, path: &str, headers: &[(&str, &str)], body: &[u8]) -> Reply {
    request_with_ready(port, method, path, headers, body, None)
}
fn request_with_ready(
    port: u16,
    method: &str,
    path: &str,
    headers: &[(&str, &str)],
    body: &[u8],
    ready: Option<std::sync::mpsc::Sender<()>>,
) -> Reply {
    let mut socket = TcpStream::connect(("127.0.0.1", port)).unwrap();
    socket.set_read_timeout(Some(WAIT)).unwrap();
    socket.set_write_timeout(Some(WAIT)).unwrap();
    let host = if headers
        .iter()
        .any(|(name, _)| name.eq_ignore_ascii_case("host"))
    {
        String::new()
    } else {
        format!("Host: 127.0.0.1:{port}\r\n")
    };
    write!(
        socket,
        "{method} {path} HTTP/1.1\r\n{host}Connection: close\r\nContent-Length: {}\r\n",
        body.len()
    )
    .unwrap();
    for (name, value) in headers {
        write!(socket, "{name}: {value}\r\n").unwrap();
    }
    socket.write_all(b"\r\n").unwrap();
    // A server may answer before it reads the body (a refusal, a size limit)
    // and close; the answer is still there to read.
    if let Err(error) = socket.write_all(body) {
        assert!(closed_early(&error), "{error}");
    }
    let mut raw = Vec::new();
    if let Some(ready) = ready {
        // Shutdown may close the listener before a newly spawned subscriber
        // connects. Wait for the service's response, not thread scheduling.
        while !raw.ends_with(b"\r\n\r\n") {
            assert!(raw.len() < 64 * 1024, "response headers too large");
            let mut byte = [0];
            socket.read_exact(&mut byte).unwrap();
            raw.extend_from_slice(&byte);
        }
        let status = String::from_utf8_lossy(&raw)
            .split_whitespace()
            .nth(1)
            .unwrap()
            .parse::<u16>()
            .unwrap();
        assert_eq!(status, 200, "event subscription rejected");
        ready.send(()).unwrap();
    }
    // The connection closes with `Connection: close`. When the server closed
    // while body bytes it never read were still arriving, its system resets
    // the connection; the reply already received stays readable before the
    // reset is reported, so a complete reply is kept.
    if let Err(error) = socket.read_to_end(&mut raw) {
        assert!(
            closed_early(&error) && reply_complete(&raw),
            "{error} after {} bytes",
            raw.len()
        );
    }
    let split = raw.windows(4).position(|v| v == b"\r\n\r\n").unwrap();
    let head = String::from_utf8_lossy(&raw[..split]);
    let mut lines = head.lines();
    let status = lines
        .next()
        .unwrap()
        .split_whitespace()
        .nth(1)
        .unwrap()
        .parse()
        .unwrap();
    let headers = lines
        .filter_map(|line| {
            line.split_once(':')
                .map(|(k, v)| (k.to_ascii_lowercase(), v.trim().to_owned()))
        })
        .collect::<HashMap<_, _>>();
    let mut body = raw[split + 4..].to_vec();
    if headers
        .get("transfer-encoding")
        .is_some_and(|v| v == "chunked")
    {
        let mut decoded = Vec::new();
        let mut cursor = 0;
        loop {
            let end = body[cursor..]
                .windows(2)
                .position(|v| v == b"\r\n")
                .unwrap()
                + cursor;
            let size = usize::from_str_radix(
                std::str::from_utf8(&body[cursor..end])
                    .unwrap()
                    .split(';')
                    .next()
                    .unwrap(),
                16,
            )
            .unwrap();
            cursor = end + 2;
            if size == 0 {
                break;
            }
            decoded.extend_from_slice(&body[cursor..cursor + size]);
            cursor += size + 2;
        }
        body = decoded;
    }
    Reply {
        status,
        headers,
        body,
    }
}
fn closed_early(error: &std::io::Error) -> bool {
    matches!(
        error.kind(),
        std::io::ErrorKind::ConnectionReset
            | std::io::ErrorKind::BrokenPipe
            | std::io::ErrorKind::ConnectionAborted
    )
}
/// Whether `raw` holds a whole HTTP reply: its headers and the body their
/// Content-Length or chunked encoding promises.
fn reply_complete(raw: &[u8]) -> bool {
    let Some(split) = raw.windows(4).position(|v| v == b"\r\n\r\n") else {
        return false;
    };
    let head = String::from_utf8_lossy(&raw[..split]).to_ascii_lowercase();
    let body = &raw[split + 4..];
    let header = |name: &str| {
        head.lines()
            .skip(1)
            .find_map(|line| line.strip_prefix(name).map(|value| value.trim().to_owned()))
    };
    if let Some(length) = header("content-length:") {
        return length
            .parse::<usize>()
            .is_ok_and(|length| body.len() >= length);
    }
    if header("transfer-encoding:").is_some_and(|value| value == "chunked") {
        return body.starts_with(b"0\r\n\r\n") || body.windows(7).any(|v| v == b"\r\n0\r\n\r\n");
    }
    false
}
fn multipart(files: &[(&str, &[u8])], urls: Value, options: Value) -> (String, Vec<u8>) {
    let boundary = "markitai-independent-serve-test";
    let mut body = Vec::new();
    for (name, bytes) in files {
        write!(body,"--{boundary}\r\nContent-Disposition: form-data; name=\"files\"; filename=\"{name}\"\r\nContent-Type: application/octet-stream\r\n\r\n").unwrap();
        body.extend_from_slice(bytes);
        body.extend_from_slice(b"\r\n");
    }
    for (name, value) in [("urls", urls), ("options", options)] {
        write!(
            body,
            "--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
        )
        .unwrap();
    }
    write!(body, "--{boundary}--\r\n").unwrap();
    (format!("multipart/form-data; boundary={boundary}"), body)
}
fn zip_contents(bytes: &[u8]) -> HashMap<String, Vec<u8>> {
    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(bytes)).unwrap();
    (0..archive.len())
        .map(|index| {
            let mut file = archive.by_index(index).unwrap();
            assert!(file.enclosed_name().is_some());
            let name = file.name().unwrap().into_owned();
            let mut bytes = Vec::new();
            file.read_to_end(&mut bytes).unwrap();
            (name, bytes)
        })
        .collect()
}

/// A server that refuses a request before reading its body and closes is
/// reset by its system once the unread bytes arrive; the client still gets the
/// reply. (macOS runners met this on refusals of JSON bodies.)
#[test]
fn the_test_client_keeps_a_reply_sent_before_an_unread_body() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let server = std::thread::spawn(move || {
        let (mut socket, _) = listener.accept().unwrap();
        let mut head = Vec::new();
        while !head.ends_with(b"\r\n\r\n") {
            let mut byte = [0];
            socket.read_exact(&mut byte).unwrap();
            head.push(byte[0]);
        }
        let body = br#"{"reason":"request_too_large"}"#;
        write!(
            socket,
            "HTTP/1.1 413 Payload Too Large\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        )
        .unwrap();
        socket.write_all(body).unwrap();
        // Let the unread body pile up, then close with it unread: a reset.
        std::thread::sleep(Duration::from_millis(200));
    });
    let reply = request(
        port,
        "POST",
        "/api/access-tokens",
        &[],
        &vec![b'x'; 8 << 20],
    );
    server.join().unwrap();
    assert_eq!(reply.status, 413);
    assert_eq!(reply.json()["reason"], "request_too_large");
    assert!(reply_complete(
        b"HTTP/1.1 204 No Content\r\nContent-Length: 0\r\n\r\n"
    ));
    assert!(reply_complete(
        b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n2\r\nok\r\n0\r\n\r\n"
    ));
    assert!(!reply_complete(
        b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nok"
    ));
    assert!(!reply_complete(b"HTTP/1.1 200 OK\r\n\r\nno length"));
}

/// An uploaded `.urls` list is a batch of URLs, not a document: its entries
/// become URL items, a named entry keeps its output name, and a comment, a blank
/// line and a line that is not a URL are skipped the way the CLI skips them.
#[test]
fn an_uploaded_urls_list_becomes_url_items_with_the_names_it_asks_for() {
    let temp = tempfile::tempdir().unwrap();
    let server = Server::start(temp.path());
    let list = b"# sources\r\n\r\nhttps://example.test/one\r\nhttps://example.test/two  Report\r\nnot a url\r\n";
    let created = server.submit(&[("links.urls", list)], json!([]), json!({}));
    let items = created["items"].as_array().unwrap();
    assert_eq!(items.len(), 2, "{created}");
    assert_eq!(items[0]["kind"], "url");
    assert_eq!(items[0]["name"], "https://example.test/one");
    assert_eq!(items[1]["name"], "https://example.test/two");
    // The reserved output name is what the job reports and the ledger shows: an
    // entry without a name is named from its URL, and a named one keeps that name
    // instead of the URL's last segment (`two`).
    let snapshot = server.json(&format!(
        "/api/jobs/{}",
        created["job_id"].as_str().unwrap()
    ));
    let items = snapshot["items"].as_array().unwrap();
    assert_eq!(items[0]["output_name"], "one.md", "{snapshot}");
    assert_eq!(items[1]["output_name"], "Report.md", "{snapshot}");
    let _ = server.request(
        "DELETE",
        &format!("/api/jobs/{}", created["job_id"].as_str().unwrap()),
        &[],
        &[],
    );
}

/// A JSON list carries names too, and a list that holds nothing usable is
/// refused with the file named rather than converting nothing quietly.
#[test]
fn a_urls_list_refuses_an_empty_result_and_a_json_list_keeps_its_names() {
    let temp = tempfile::tempdir().unwrap();
    let server = Server::start(temp.path());
    let created = server.submit(
        &[(
            "links.urls",
            br#"[{"url": "https://example.test/a", "output_name": "A"}]"#,
        )],
        json!([]),
        json!({}),
    );
    let items = created["items"].as_array().unwrap();
    assert_eq!(items.len(), 1, "{created}");
    let snapshot = server.json(&format!(
        "/api/jobs/{}",
        created["job_id"].as_str().unwrap()
    ));
    assert_eq!(snapshot["items"][0]["output_name"], "A.md", "{snapshot}");

    for (name, body) in [
        ("empty.urls", b"# only a comment\n".as_slice()),
        ("junk.urls", b"mailto:someone@example.test\n".as_slice()),
        ("broken.urls", b"[\"https://example.test/a\"".as_slice()),
    ] {
        let (content_type, body) = multipart(&[(name, body)], json!([]), json!({}));
        let reply = server.request(
            "POST",
            "/api/jobs",
            &[("Content-Type", &content_type)],
            &body,
        );
        assert_eq!(reply.status, 422, "{}", reply.text());
        assert!(reply.text().contains(name), "{}", reply.text());
    }
}

#[test]
fn an_unknown_page_is_the_workbench_with_404_and_an_unknown_resource_is_json() {
    let temp = tempfile::tempdir().unwrap();
    let server = Server::start(temp.path());
    let index = server.request("GET", "/", &[("Accept", "text/html")], b"");
    for path in ["/does-not-exist", "/jobs/abc", "/settings/nope"] {
        let page = server.request(
            "GET",
            path,
            &[
                ("Accept", "text/html,application/xhtml+xml"),
                ("If-None-Match", "*"),
            ],
            b"",
        );
        assert_eq!(page.status, 404, "{path}");
        assert!(
            page.headers["content-type"].starts_with("text/html"),
            "{path}"
        );
        assert!(
            page.headers.contains_key("content-security-policy"),
            "{path}"
        );
        assert_eq!(page.body, index.body, "{path}");
    }
    for (path, accept) in [
        ("/does-not-exist", "application/json"),
        ("/does-not-exist", "*/*"),
        ("/ui/nope.js", "text/html"),
        ("/api/nope", "text/html"),
    ] {
        let reply = server.request("GET", path, &[("Accept", accept)], b"");
        assert_eq!(reply.status, 404, "{path} {accept}");
        assert_eq!(reply.json()["reason"], "route_not_found", "{path} {accept}");
    }
    let post = server.request("POST", "/does-not-exist", &[("Accept", "text/html")], b"");
    assert_ne!(post.status, 200);
    assert!(
        !post
            .headers
            .get("content-type")
            .is_some_and(|value| value.starts_with("text/html"))
    );
}

#[test]
fn capabilities_report_the_browser_the_server_itself_finds() {
    // The server's own environment decides, not the caller's: a test process
    // with a Playwright cache in its home finds a browser that the server,
    // whose home is a temporary directory, does not. Each answer is pinned.
    let temp = tempfile::tempdir().unwrap();
    let browser = temp.path().join(if cfg!(windows) {
        "chrome.exe"
    } else {
        "chrome"
    });
    std::fs::write(&browser, b"").unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&browser, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    for (executable, expected) in [(browser, true), (temp.path().join("absent-browser"), false)] {
        let root = tempfile::tempdir().unwrap();
        let server = Server::start_with(
            root.path(),
            &[("MARKITAI_BROWSER_EXECUTABLE", executable.as_os_str())],
        );
        let caps = server.json("/api/capabilities");
        assert_eq!(caps["extras"]["browser"], expected, "{executable:?}");
    }
}

#[test]
fn multipart_jobs_preserve_member_identity_sse_results_and_restart_history() {
    let temp = tempfile::tempdir().unwrap();
    let server = Server::start(temp.path());
    let caps = server.json("/api/capabilities");
    assert_eq!(caps["limits"]["max_job_items"], 1000);
    let origin = Origin::start();
    let created = server.submit(
        &[
            ("notes.txt", b"first original"),
            ("notes.llm.txt", b"second original"),
            ("../NOTES.TXT", b"third original"),
        ],
        json!([origin.url("/notes.txt.llm")]),
        json!({"llm":false}),
    );
    let id = created["job_id"].as_str().unwrap().to_owned();
    assert_eq!(created["items"].as_array().unwrap().len(), 4);
    let snapshot = server.done(&id);
    assert_eq!(snapshot["status"], "done");
    assert_eq!(snapshot["done"], 4);
    assert_eq!(snapshot["failed"], 0);
    let mut outputs = std::collections::HashSet::new();
    for (index, expected) in ["first original", "second original", "third original"]
        .iter()
        .enumerate()
    {
        let item = &snapshot["items"][index];
        assert!(item["duration_ms"].is_u64());
        let output = item["output"].as_str().unwrap();
        assert!(outputs.insert(output.to_owned()));
        let result = server.json(&format!("/api/jobs/{id}/items/i{}/result", index + 1));
        assert_eq!(result["variant"], "base");
        assert!(result["markdown"].as_str().unwrap().contains(expected));
        let reply = server.request(
            "GET",
            &format!("/api/jobs/{id}/files/{}", encode_path(output)),
            &[],
            &[],
        );
        assert_eq!(reply.status, 200);
        assert_eq!(reply.text(), result["markdown"].as_str().unwrap());
    }
    assert!(outputs.contains("notes.txt.md"));
    let url_output = snapshot["items"][3]["output"].as_str().unwrap();
    assert_ne!(
        url_output, "notes.txt.llm.md",
        "URL must not occupy the first item's enhanced member"
    );
    assert!(
        server.json(&format!("/api/jobs/{id}/items/i4/result"))["markdown"]
            .as_str()
            .unwrap()
            .contains("Origin response")
    );
    let events = server.request("GET", &format!("/api/jobs/{id}/events"), &[], &[]);
    assert_eq!(events.status, 200);
    assert!(events.headers["content-type"].starts_with("text/event-stream"));
    assert!(events.text().starts_with("event: snapshot\n"));
    assert!(events.text().contains("event: job\n"));
    let zip = server.request("GET", &format!("/api/jobs/{id}/archive"), &[], &[]);
    assert_eq!(zip.status, 200);
    assert_eq!(zip_contents(&zip.body).len(), 4);
    let saved: Value =
        serde_json::from_slice(&std::fs::read(server.jobdir(&id).join("meta.json")).unwrap())
            .unwrap();
    assert_eq!(saved["status"], "done");
    assert_eq!(saved["version"], 2);
    server.stop();
    let restarted = Server::start(temp.path());
    assert_eq!(restarted.json("/api/history")[0]["job_id"], id);
    assert_eq!(
        restarted.json(&format!("/api/jobs/{id}"))["items"],
        snapshot["items"]
    );
    assert_eq!(
        restarted
            .request("DELETE", &format!("/api/history/{id}"), &[], &[])
            .status,
        204
    );
    assert_eq!(restarted.json("/api/history"), json!([]));
    assert!(!restarted.jobdir(&id).exists());
    restarted.stop();
}
fn encode_path(path: &str) -> String {
    url::form_urlencoded::byte_serialize(path.as_bytes())
        .collect::<String>()
        .replace('+', "%20")
        .replace("%2F", "/")
}

#[test]
fn rejected_requests_roll_back_uploads_and_keep_error_envelopes() {
    let temp = tempfile::tempdir().unwrap();
    let server = Server::start(temp.path());
    for (urls, options) in [
        (json!(["file:///private"]), json!({})),
        (json!([" "]), json!({})),
        (json!([]), json!({"unknown":true})),
        (json!([]), json!({"profile":"invalid-profile"})),
    ] {
        let (content, body) = multipart(&[("uploaded.txt", b"rollback")], urls, options);
        let reply = server.request("POST", "/api/jobs", &[("Content-Type", &content)], &body);
        assert_eq!(reply.status, 422, "{}", reply.text());
        assert_eq!(reply.json()["code"], "invalid_request");
    }
    let reply = server.request(
        "POST",
        "/api/jobs",
        &[("Origin", "https://untrusted.invalid")],
        &[],
    );
    assert_eq!(reply.status, 403);
    let reply = server.request(
        "GET",
        "/api/capabilities",
        &[("Host", "untrusted.invalid")],
        &[],
    );
    assert_eq!(reply.status, 400);
    assert_eq!(server.request("GET", "/api/missing", &[], &[]).status, 404);
    assert_eq!(server.request("PUT", "/api/jobs", &[], &[]).status, 405);
    let entries = std::fs::read_dir(temp.path().join("home/serve/jobs"))
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .filter(|name| name != ".serve.lock")
        .collect::<Vec<_>>();
    assert!(
        entries.is_empty(),
        "rejected uploads must not leave stage directories: {entries:?}"
    );
    server.stop();
}

#[test]
fn successful_and_failed_items_keep_public_types_and_asset_archives_self_contained() {
    let temp = tempfile::tempdir().unwrap();
    let server = Server::start(temp.path());
    let eml = "From: sender@example.test\r\nTo: receiver@example.test\r\nSubject: Attached pixel\r\nMIME-Version: 1.0\r\nContent-Type: multipart/mixed; boundary=history-png\r\n\r\n--history-png\r\nContent-Type: text/plain; charset=utf-8\r\n\r\nArchive body.\r\n--history-png\r\nContent-Type: image/png; name=pixel.png\r\nContent-Disposition: attachment; filename=pixel.png\r\nContent-Transfer-Encoding: base64\r\n\r\niVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg==\r\n--history-png--\r\n";
    let created = server.submit(
        &[
            ("letter.eml", eml.as_bytes()),
            ("bad.ipynb", b"{not valid json}"),
        ],
        json!([]),
        json!({}),
    );
    let id = created["job_id"].as_str().unwrap();
    let done = server.done(id);
    assert_eq!(done["done"], 1);
    assert_eq!(done["failed"], 1);
    assert!(done["items"][1]["error"].is_string());
    assert!(done["items"][1]["output"].is_null());
    let result = server.json(&format!("/api/jobs/{id}/items/i1/result"));
    let artifacts = result["artifacts"].as_array().unwrap();
    assert_eq!(artifacts.len(), 2, "{result}");
    // The document JSON names the upload as the user did, not its stored path.
    let document = server.json(&format!("/api/jobs/{id}/items/i1/document"));
    assert_eq!(document["schema_version"], "1.0");
    assert_eq!(document["source"]["path"], "letter.eml");
    assert_eq!(document["source"]["size"], eml.len());
    assert!(
        document["images"][0]["path"]
            .as_str()
            .unwrap()
            .starts_with(".markitai/assets/"),
        "{document}"
    );
    let failed = server.request(
        "GET",
        &format!("/api/jobs/{id}/items/i2/document"),
        &[],
        &[],
    );
    assert_eq!(failed.status, 404);
    let mut expected = HashMap::new();
    for artifact in artifacts {
        let name = artifact["relpath"].as_str().unwrap();
        let reply = server.request(
            "GET",
            &format!("/api/jobs/{id}/files/{}", encode_path(name)),
            &[],
            &[],
        );
        assert_eq!(reply.status, 200);
        assert_eq!(reply.body.len() as u64, artifact["size"].as_u64().unwrap());
        expected.insert(name.to_owned(), reply.body);
    }
    assert!(
        expected
            .values()
            .any(|bytes| bytes.starts_with(b"\x89PNG\r\n\x1a\n"))
    );
    let port = server.port;
    let path = format!("/api/jobs/{id}/archive");
    let other = path.clone();
    let handle = std::thread::spawn(move || request(port, "GET", &other, &[], &[]));
    let first = server.request("GET", &path, &[], &[]);
    let second = handle.join().unwrap();
    assert_eq!(zip_contents(&first.body), expected);
    assert_eq!(zip_contents(&second.body), expected);
    std::fs::remove_dir_all(server.jobdir(id).join("uploads")).unwrap();
    assert_eq!(
        server.json(&format!("/api/jobs/{id}/items/i1/result")),
        result
    );
    let archive = server.request("GET", "/api/history/archive", &[], &[]);
    assert_eq!(zip_contents(&archive.body), expected);
    server.stop();
}

struct Origin {
    port: u16,
    stop: Arc<AtomicBool>,
    gate: Arc<(Mutex<bool>, Condvar)>,
    entered: Arc<AtomicUsize>,
    thread: Option<JoinHandle<()>>,
}
impl Origin {
    fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        listener.set_nonblocking(true).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let entered = Arc::new(AtomicUsize::new(0));
        let thread_stop = stop.clone();
        let thread_gate = gate.clone();
        let thread_entered = entered.clone();
        let thread = std::thread::spawn(move || {
            let mut workers = Vec::new();
            while !thread_stop.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        let gate = thread_gate.clone();
                        let entered = thread_entered.clone();
                        workers.push(std::thread::spawn(move||{
            stream.set_nonblocking(false).unwrap();
            stream.set_write_timeout(Some(WAIT)).unwrap();
            let deadline = Instant::now() + WAIT;
            let mut request = Vec::new();
            let mut byte = [0];
            loop {
                let remaining = deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() { return; }
                stream.set_read_timeout(Some(remaining)).unwrap();
                match stream.read(&mut byte) {
                    Ok(1) => request.push(byte[0]),
                    Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                    _ => return,
                }
                if request.ends_with(b"\r\n\r\n") { break; }
                assert!(request.len() < 65536, "loopback header exceeds limit");
            }
            let path=String::from_utf8_lossy(&request).split_whitespace().nth(1).unwrap_or("").to_owned();entered.fetch_add(1,Ordering::SeqCst);
            if path=="/hold"{let(lock,notify)=&*gate;let ready=lock.lock().unwrap_or_else(|e|e.into_inner());let(ready,_)=notify.wait_timeout_while(ready,WAIT,|ready|!*ready).unwrap_or_else(|e|e.into_inner());if !*ready{return;}}
            let body=format!("<html><title>Local page</title><article><h1>Local article</h1><p>Origin response for {path}.</p></article></html>");let response=format!("HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",body.len());let _=stream.write_all(response.as_bytes());
        }));
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(2))
                    }
                    Err(_) => break,
                }
            }
            for worker in workers {
                let _ = worker.join();
            }
        });
        Self {
            port,
            stop,
            gate,
            entered,
            thread: Some(thread),
        }
    }
    fn url(&self, path: &str) -> String {
        format!("http://127.0.0.1:{}{path}", self.port)
    }
    fn release(&self) {
        let (lock, notify) = &*self.gate;
        *lock.lock().unwrap_or_else(|e| e.into_inner()) = true;
        notify.notify_all();
    }
}
impl Drop for Origin {
    fn drop(&mut self) {
        self.release();
        self.stop.store(true, Ordering::SeqCst);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[test]
fn shutdown_drains_the_active_url_stops_queued_work_and_persists_both_outcomes() {
    let origin = Origin::start();
    let temp = tempfile::tempdir().unwrap();
    let server = Server::start(temp.path());
    let created = server.submit(
        &[],
        json!([origin.url("/hold"), origin.url("/queued")]),
        json!({"strategy":"static"}),
    );
    let id = created["job_id"].as_str().unwrap().to_owned();
    until("first URL reaches origin", || {
        origin.entered.load(Ordering::SeqCst) == 1
    });
    let snapshot = server.json(&format!("/api/jobs/{id}"));
    assert_eq!(snapshot["items"][0]["status"], "running");
    assert_eq!(snapshot["items"][1]["status"], "queued");
    assert_eq!(
        server
            .request("GET", &format!("/api/jobs/{id}/archive"), &[], &[])
            .status,
        409
    );
    assert_eq!(
        server
            .request("DELETE", &format!("/api/history/{id}"), &[], &[])
            .status,
        409
    );
    let path = server.jobdir(&id);
    server.signal();
    origin.release();
    server.finish(true);
    assert_eq!(
        origin.entered.load(Ordering::SeqCst),
        1,
        "queued URL must not dispatch after shutdown acknowledgement"
    );
    let saved: Value =
        serde_json::from_slice(&std::fs::read(path.join("meta.json")).unwrap()).unwrap();
    assert_eq!(saved["status"], "done");
    assert_eq!(saved["items"][0]["status"], "done");
    assert_eq!(saved["items"][1]["status"], "error");
    assert_eq!(saved["items"][1]["error"], "cancelled (server shutdown)");
    assert!(
        path.join("out")
            .join(saved["items"][0]["output"].as_str().unwrap())
            .is_file()
    );
    let restarted = Server::start(temp.path());
    assert_eq!(
        restarted.json(&format!("/api/jobs/{id}"))["items"][1]["status"],
        "error"
    );
    restarted.stop();
}

#[test]
fn stop_request_cancels_waiting_items_while_the_active_url_finishes() {
    let origin = Origin::start();
    let temp = tempfile::tempdir().unwrap();
    let server = Server::start(temp.path());
    let created = server.submit(
        &[],
        json!([
            origin.url("/hold"),
            origin.url("/waiting-a"),
            origin.url("/waiting-b")
        ]),
        json!({"strategy":"static"}),
    );
    let id = created["job_id"].as_str().unwrap().to_owned();
    until("first URL reaches origin", || {
        origin.entered.load(Ordering::SeqCst) == 1
    });
    let stop = |job: &str| server.request("POST", &format!("/api/jobs/{job}/cancel"), &[], &[]);
    let reply = stop(&id);
    assert_eq!(reply.status, 202, "{}", reply.text());
    assert_eq!(reply.json(), json!({"job_id":id,"stopping":2}));
    let snapshot = server.json(&format!("/api/jobs/{id}"));
    assert_eq!(snapshot["status"], "running");
    assert_eq!(snapshot["items"][0]["status"], "running");
    origin.release();
    let snapshot = server.done(&id);
    assert_eq!(snapshot["status"], "done");
    assert_eq!(
        (snapshot["done"].clone(), snapshot["failed"].clone()),
        (json!(1), json!(2))
    );
    assert_eq!(snapshot["items"][0]["status"], "done");
    for index in [1, 2] {
        assert_eq!(snapshot["items"][index]["status"], "error");
        assert_eq!(
            snapshot["items"][index]["error"],
            "cancelled (stopped by request)"
        );
    }
    assert_eq!(
        origin.entered.load(Ordering::SeqCst),
        1,
        "stopped URLs must not dispatch"
    );
    let reply = stop(&id);
    assert_eq!(
        (reply.status, reply.json()["detail"].clone()),
        (409, json!("job is not running"))
    );
    assert_eq!(stop("ffffffffffff").status, 404);
    // A stopped item remains an ordinary retryable failure.
    let reply = server.request(
        "POST",
        &format!("/api/jobs/{id}/items/i2/retry"),
        &[("Content-Type", "application/json")],
        b"",
    );
    assert_eq!(reply.status, 202, "{}", reply.text());
    let snapshot = server.done(&id);
    assert_eq!(snapshot["items"][1]["status"], "done");
    assert_eq!(
        snapshot["items"][2]["error"],
        "cancelled (stopped by request)"
    );
    let path = server.jobdir(&id);
    server.stop();
    let saved: Value =
        serde_json::from_slice(&std::fs::read(path.join("meta.json")).unwrap()).unwrap();
    assert_eq!(saved["items"][2]["error"], "cancelled (stopped by request)");
}

#[test]
fn stop_request_without_waiting_items_is_a_conflict() {
    let origin = Origin::start();
    let temp = tempfile::tempdir().unwrap();
    let server = Server::start(temp.path());
    let created = server.submit(
        &[],
        json!([origin.url("/hold")]),
        json!({"strategy":"static"}),
    );
    let id = created["job_id"].as_str().unwrap().to_owned();
    until("URL reaches origin", || {
        origin.entered.load(Ordering::SeqCst) == 1
    });
    let reply = server.request("POST", &format!("/api/jobs/{id}/cancel"), &[], &[]);
    assert_eq!(
        (reply.status, reply.json()["detail"].clone()),
        (409, json!("no queued items to stop"))
    );
    let reply = server.request(
        "POST",
        &format!("/api/jobs/{id}/cancel"),
        &[("Origin", "http://evil.test")],
        &[],
    );
    assert_eq!(reply.status, 403, "{}", reply.text());
    origin.release();
    assert_eq!(server.done(&id)["items"][0]["status"], "done");
    server.stop();
}

#[test]
fn terminal_persistence_failure_is_visible_and_cannot_claim_saved_history() {
    let origin = Origin::start();
    let temp = tempfile::tempdir().unwrap();
    let server = Server::start(temp.path());
    let created = server.submit(&[], json!([origin.url("/hold")]), json!({}));
    let id = created["job_id"].as_str().unwrap();
    until("held conversion", || {
        origin.entered.load(Ordering::SeqCst) == 1
    });
    let meta = server.jobdir(id).join("meta.json");
    std::fs::remove_file(&meta).unwrap();
    std::fs::create_dir(&meta).unwrap();
    origin.release();
    let done = server.done(id);
    assert_eq!(done["status"], "error");
    assert!(
        done["persistence_error"]
            .as_str()
            .unwrap()
            .contains("could not be persisted")
    );
    assert_eq!(done["items"][0]["status"], "done");
    assert!(
        server.json(&format!("/api/jobs/{id}/items/i1/result"))["markdown"]
            .as_str()
            .unwrap()
            .contains("Origin response")
    );
    assert_eq!(server.json("/api/history"), json!([]));
    let events = server.request("GET", &format!("/api/jobs/{id}/events"), &[], &[]);
    assert!(events.text().contains("persistence_error"));
    server.signal();
    server.finish(false);
}

#[test]
fn late_cli_history_is_imported_without_following_external_symlinks() {
    let temp = tempfile::tempdir().unwrap();
    let server = Server::start(temp.path());
    assert_eq!(server.json("/api/history"), json!([]));
    let id = "012345abcdef";
    let folder = server.jobdir(id);
    std::fs::create_dir_all(folder.join("out")).unwrap();
    std::fs::write(
        folder.join("out/notes.llm.md"),
        "Legacy uploaded notes.llm body.\n",
    )
    .unwrap();
    std::fs::write(folder.join("meta.json"),json!({"job_id":id,"created_at":hours_ago(2),"finished_at":hours_ago(1),"status":"done","options":{"origin":"cli"},"items":[{"item_id":"i1","name":"notes.llm","kind":"file","status":"done","error":null,"output":"notes.llm.md","output_name":null,"duration_ms":1000,"cost_usd":null,"llm_enhanced":false,"operation":"convert","skipped":false,"skip_reason":null,"retryable":false,"warnings":[]}]}).to_string()).unwrap();
    let history = server.json("/api/history");
    assert_eq!(history[0]["origin"], "cli");
    assert_eq!(history[0]["cost_usd"], Value::Null);
    assert_eq!(history[0]["retryable"], false);
    let result = server.json(&format!("/api/jobs/{id}/items/i1/result"));
    assert_eq!(result["variant"], "base");
    assert_eq!(result["markdown"], "Legacy uploaded notes.llm body.\n");
    std::fs::write(temp.path().join("secret.txt"), "outside secret").unwrap();
    std::os::unix::fs::symlink(
        temp.path().join("secret.txt"),
        folder.join("out/escape.txt"),
    )
    .unwrap();
    assert_eq!(
        server
            .request("GET", &format!("/api/jobs/{id}/files/escape.txt"), &[], &[])
            .status,
        404
    );
    assert_eq!(
        server
            .request(
                "GET",
                &format!("/api/jobs/{id}/files/%2E%2E/secret.txt"),
                &[],
                &[]
            )
            .status,
        404
    );
    server.stop();
}

/// A URL's userinfo and secret parameter values reach the page only: the
/// job's answers and its saved history name it without them. A retry in the
/// same process fetches the address as given; after a restart, which kept no
/// credentials, the retry asks for the URL again instead of fetching another.
#[test]
fn url_credentials_reach_the_page_but_no_answer_or_saved_history() {
    let origin = Origin::start();
    let temp = tempfile::tempdir().unwrap();
    let server = Server::start(temp.path());
    let given = format!(
        "http://alice:pw0rd@127.0.0.1:{}/secret-page?api_key=k1secret&v=2",
        origin.port
    );
    let shown = origin.url("/secret-page?api_key=REDACTED&v=2");
    let leaks = |text: &str| text.contains("pw0rd") || text.contains("k1secret");
    let created = server.submit(&[], json!([given]), json!({}));
    assert_eq!(created["items"][0]["name"], shown.as_str(), "{created}");
    let id = created["job_id"].as_str().unwrap().to_owned();
    let snapshot = server.done(&id);
    assert_eq!(snapshot["done"], 1, "{snapshot}");
    assert_eq!(snapshot["items"][0]["name"], shown.as_str());
    // The page itself was asked for with its key.
    let markdown = server.json(&format!("/api/jobs/{id}/items/i1/result"))["markdown"].clone();
    assert!(
        markdown.as_str().unwrap().contains("k1secret"),
        "{markdown}"
    );
    for path in ["/api/history", "/api/history?include=items"] {
        let history = server.request("GET", path, &[], &[]).text();
        assert!(!leaks(&history) && history.contains(&shown), "{history}");
    }
    let retry = |server: &Server| {
        server.request(
            "POST",
            &format!("/api/jobs/{id}/items/i1/retry"),
            &[("Content-Type", "application/json")],
            b"",
        )
    };
    let before = origin.entered.load(Ordering::SeqCst);
    let reply = retry(&server);
    assert_eq!(reply.status, 202, "{}", reply.text());
    until("retry", || {
        origin.entered.load(Ordering::SeqCst) > before
            && server.json(&format!("/api/jobs/{id}"))["items"][0]["status"] == "done"
    });
    let saved = std::fs::read_to_string(server.jobdir(&id).join("meta.json")).unwrap();
    assert!(!leaks(&saved), "{saved}");
    server.stop();
    let restarted = Server::start(temp.path());
    let snapshot = restarted.json(&format!("/api/jobs/{id}"));
    assert_eq!(snapshot["items"][0]["name"], shown.as_str());
    assert!(!leaks(&snapshot.to_string()));
    let reply = retry(&restarted);
    assert_eq!(
        (reply.status, reply.json()["reason"].clone()),
        (409, json!("no_source")),
        "{}",
        reply.text()
    );
    restarted.stop();
}

#[test]
fn urlencoded_url_submission_preserves_form_contract() {
    let origin = Origin::start();
    let temp = tempfile::tempdir().unwrap();
    let server = Server::start(temp.path());
    let urls = json!([origin.url("/encoded")]).to_string();
    let body = url::form_urlencoded::Serializer::new(String::new())
        .append_pair("urls", &urls)
        .append_pair("options", "")
        .finish();
    let reply = server.request(
        "POST",
        "/api/jobs",
        &[("Content-Type", "application/x-www-form-urlencoded")],
        body.as_bytes(),
    );
    assert_eq!(reply.status, 201, "{}", reply.text());
    let created = reply.json();
    let id = created["job_id"].as_str().unwrap();
    assert_eq!(server.done(id)["done"], 1);
    assert!(
        server.json(&format!("/api/jobs/{id}/items/i1/result"))["markdown"]
            .as_str()
            .unwrap()
            .contains("/encoded")
    );
    server.stop();
}

#[test]
fn legacy_visible_assets_and_multi_history_zip_names_follow_the_saved_contract() {
    let temp = tempfile::tempdir().unwrap();
    let server = Server::start(temp.path());
    // Creation order is deliberately opposite to the UUID lexical order.
    for (id, created, text) in [
        ("ffffffffffff", hours_ago(48), "earlier"),
        ("000000000000", hours_ago(24), "later"),
    ] {
        let folder = server.jobdir(id);
        std::fs::create_dir_all(folder.join("out/assets")).unwrap();
        std::fs::write(
            folder.join("out/a.txt.md"),
            format!("Base {text}. ![asset](assets/shared.png)\n"),
        )
        .unwrap();
        std::fs::write(
            folder.join("out/a.txt.llm.md"),
            format!("Enhanced {text}. ![[assets/shared.png]]\n"),
        )
        .unwrap();
        std::fs::write(
            folder.join("out/assets/shared.png"),
            format!("asset {text}"),
        )
        .unwrap();
        std::fs::write(folder.join("out/assets/unrelated.png"), "unrelated").unwrap();
        std::fs::write(folder.join("meta.json"),json!({"job_id":id,"created_at":created,"finished_at":created,"status":"done","version":2,"options":{"origin":"cli","profile":"obsidian"},"items":[{"item_id":"i1","name":"a.txt","kind":"file","status":"done","error":null,"output":"a.txt.llm.md","output_name":"a.txt.md","duration_ms":0,"cost_usd":null,"llm_enhanced":true,"operation":"convert","skipped":false,"skip_reason":null,"retryable":false,"warnings":[]}]}).to_string()).unwrap();
    }
    let history = server.json("/api/history");
    assert_eq!(history[0]["job_id"], "000000000000");
    let result = server.json("/api/jobs/ffffffffffff/items/i1/result");
    assert_eq!(result["variant"], "llm");
    let names = result["artifacts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v["relpath"].as_str().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(names, vec!["a.txt.md", "a.txt.llm.md", "assets/shared.png"]);
    let response = server.request("GET", "/api/history/archive", &[], &[]);
    assert_eq!(response.status, 200);
    assert!(response.headers["content-disposition"].contains("markitai-all.zip"));
    let files = zip_contents(&response.body);
    assert_eq!(files.len(), 8);
    assert_eq!(files["a.txt/assets/shared.png"], b"asset earlier");
    assert_eq!(files["a.txt (2)/assets/shared.png"], b"asset later");
    assert!(files.contains_key("a.txt/a.txt.llm.md"));
    assert!(!files.keys().any(|name| name.starts_with("a.txt.llm/")));
    server.stop();
}

#[path = "serve/rerun.rs"]
mod rerun;

#[path = "serve/legacy_history.rs"]
mod legacy_history;

#[path = "serve/gates.rs"]
mod gates;

#[path = "serve/workbench.rs"]
mod workbench;

#[path = "serve/capture.rs"]
mod capture;

#[path = "serve/assets.rs"]
mod assets;

#[path = "serve/token.rs"]
mod token;

#[path = "serve/access.rs"]
mod access;

#[path = "serve/user_models.rs"]
mod user_models;

#[path = "serve/convert_defaults.rs"]
mod convert_defaults;
