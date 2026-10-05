#![cfg(unix)]

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
        if !root.join("config.json").exists() {
            configure(root);
        }
        let mut command = Command::new(env!("CARGO_BIN_EXE_markitai"));
        command.env_clear();
        for key in ["PATH", "HOME", "TMPDIR", "TEMP", "TMP", "SYSTEMROOT"] {
            if let Some(value) = std::env::var_os(key) {
                command.env(key, value);
            }
        }
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
                "--no-auth",
            ])
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
                line.strip_prefix("Markitai server listening on http://127.0.0.1:")
                    .and_then(|s| s.parse().ok())
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
    socket.write_all(body).unwrap();
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
    socket.read_to_end(&mut raw).unwrap();
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
            let name = file.name().to_owned();
            let mut bytes = Vec::new();
            file.read_to_end(&mut bytes).unwrap();
            (name, bytes)
        })
        .collect()
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
fn multipart_jobs_preserve_member_identity_sse_results_and_restart_history() {
    let temp = tempfile::tempdir().unwrap();
    let server = Server::start(temp.path());
    let caps = server.json("/api/capabilities");
    assert_eq!(caps["limits"]["max_job_items"], 1000);
    assert_eq!(
        caps["extras"]["browser"],
        markitai_core::browser_available()
    );
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
    std::fs::write(folder.join("meta.json"),json!({"job_id":id,"created_at":"2026-01-01T00:00:00Z","finished_at":"2026-01-01T00:00:01Z","status":"done","options":{"origin":"cli"},"items":[{"item_id":"i1","name":"notes.llm","kind":"file","status":"done","error":null,"output":"notes.llm.md","output_name":null,"duration_ms":1000,"cost_usd":null,"llm_enhanced":false,"operation":"convert","skipped":false,"skip_reason":null,"retryable":false,"warnings":[]}]}).to_string()).unwrap();
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
        ("ffffffffffff", "2026-01-01T00:00:00Z", "earlier"),
        ("000000000000", "2026-01-02T00:00:00Z", "later"),
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
