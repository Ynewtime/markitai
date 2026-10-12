//! The shared workbench's additions to the job API: job labels, history with
//! items, result excerpts, the models and upload size of an item, and an
//! OpenAI-compatible endpoint saved from the settings.
use super::*;

/// A multipart job of plain fields (`label`, `options`, …) and named files.
fn form(files: &[(&str, &[u8])], fields: &[(&str, &str)]) -> (String, Vec<u8>) {
    let boundary = "markitai-workbench-test";
    let mut body = Vec::new();
    for (name, bytes) in files {
        write!(body, "--{boundary}\r\nContent-Disposition: form-data; name=\"files\"; filename=\"{name}\"\r\n\r\n").unwrap();
        body.extend_from_slice(bytes);
        body.extend_from_slice(b"\r\n");
    }
    for (name, value) in fields {
        write!(
            body,
            "--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
        )
        .unwrap();
    }
    write!(body, "--{boundary}--\r\n").unwrap();
    (format!("multipart/form-data; boundary={boundary}"), body)
}
fn post(server: &Server, files: &[(&str, &[u8])], fields: &[(&str, &str)]) -> Reply {
    let (content_type, body) = form(files, fields);
    server.request(
        "POST",
        "/api/jobs",
        &[("Content-Type", &content_type)],
        &body,
    )
}

#[test]
fn labels_are_trimmed_bounded_and_kept_with_the_job() {
    let temp = tempfile::tempdir().unwrap();
    let server = Server::start(temp.path());
    let reply = post(
        &server,
        &[("a.txt", b"labelled")],
        &[("label", "  Quarterly reports  ")],
    );
    assert_eq!(reply.status, 201, "{}", reply.text());
    let labelled = reply.json()["job_id"].as_str().unwrap().to_owned();
    assert_eq!(server.done(&labelled)["label"], "Quarterly reports");
    let unnamed = server.submit(&[("b.txt", b"plain")], json!([]), json!({}))["job_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let snapshot = server.done(&unnamed);
    assert!(snapshot["label"].is_null() && snapshot.get("label").is_some());
    // The urlencoded form takes a label too; a blank one is no label.
    let reply = server.request(
        "POST",
        "/api/jobs",
        &[("Content-Type", "application/x-www-form-urlencoded")],
        b"urls=%5B%5D&label=%20%20",
    );
    assert_eq!(
        (reply.status, reply.json()["reason"].clone()),
        (422, json!("empty_job"))
    );
    let exact = "é".repeat(120);
    for (label, status) in [
        (exact.as_str(), 201),
        (&*"é".repeat(121), 422),
        ("two\nlines", 422),
        ("tab\there", 422),
    ] {
        let reply = post(&server, &[("c.txt", b"bounded")], &[("label", label)]);
        assert_eq!(reply.status, status, "{label}: {}", reply.text());
        if status == 422 {
            assert_eq!(reply.json()["reason"], "invalid_label");
            assert_eq!(reply.json()["code"], "invalid_request");
        } else {
            let id = reply.json()["job_id"].as_str().unwrap().to_owned();
            assert_eq!(server.done(&id)["label"], exact.as_str());
            server.request("DELETE", &format!("/api/history/{id}"), &[], &[]);
        }
    }
    let history = server.json("/api/history");
    let label = |id: &str| {
        history
            .as_array()
            .unwrap()
            .iter()
            .find(|entry| entry["job_id"] == id)
            .unwrap()["label"]
            .clone()
    };
    assert_eq!(label(&labelled), "Quarterly reports");
    assert!(label(&unnamed).is_null());
    // A refused label left no job behind.
    assert_eq!(history.as_array().unwrap().len(), 2);
    server.stop();
    let restarted = Server::start(temp.path());
    assert_eq!(
        restarted.json(&format!("/api/jobs/{labelled}"))["label"],
        "Quarterly reports"
    );
    restarted.stop();
}

#[test]
fn history_with_items_carries_each_jobs_snapshot_rows() {
    let temp = tempfile::tempdir().unwrap();
    let server = Server::start(temp.path());
    let id = server.submit(
        &[("a.txt", b"first"), ("b.txt", b"second")],
        json!([]),
        json!({"profile":"rag"}),
    )["job_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let snapshot = server.done(&id);
    let plain = server.json("/api/history");
    assert!(plain[0].get("items").is_none() && plain[0].get("options").is_none());
    let full = server.json("/api/history?include=items");
    assert_eq!(full[0]["items"], snapshot["items"]);
    assert_eq!(full[0]["options"], snapshot["options"]);
    assert_eq!(full[0]["options"]["profile"], "rag");
    assert_eq!(full[0]["items"][0]["options"]["profile"], "rag");
    // The summary fields are the same with or without the items.
    let mut summary = full[0].clone();
    summary.as_object_mut().unwrap().remove("items");
    summary.as_object_mut().unwrap().remove("options");
    assert_eq!(summary, plain[0]);
    assert_eq!(server.json("/api/history?include=other"), plain);
    // The uploaded size is reported per file item.
    assert_eq!(snapshot["items"][0]["source_size"], 5);
    assert_eq!(snapshot["items"][1]["source_size"], 6);
    assert!(snapshot["items"][0].get("models").is_none());
    server.stop();
}

#[test]
fn a_result_excerpt_stops_at_a_line_and_reports_the_whole_size() {
    let temp = tempfile::tempdir().unwrap();
    let server = Server::start(temp.path());
    let mut document = String::new();
    for line in 0..400 {
        document.push_str(&format!("Line {line:03} of the long notes — ünïcode\n\n"));
    }
    let id = server.submit(&[("long.md", document.as_bytes())], json!([]), json!({}))["job_id"]
        .as_str()
        .unwrap()
        .to_owned();
    assert_eq!(server.done(&id)["items"][0]["status"], "done");
    let path = format!("/api/jobs/{id}/items/i1/result");
    let whole = server.json(&path);
    assert_eq!(whole["truncated"], false);
    let size = whole["markdown"].as_str().unwrap().len();
    assert_eq!(whole["markdown_bytes"], size);
    assert!(size > 16384);
    for limit in [1024, 4000, 16384] {
        let excerpt = server.json(&format!("{path}?max_bytes={limit}"));
        let text = excerpt["markdown"].as_str().unwrap();
        assert_eq!(excerpt["truncated"], true);
        assert_eq!(excerpt["markdown_bytes"], size);
        assert!(
            text.len() <= limit && text.len() >= limit * 3 / 4,
            "{limit}"
        );
        assert!(text.ends_with('\n'), "{limit}: {text:?}");
        assert!(whole["markdown"].as_str().unwrap().starts_with(text));
        assert_eq!(excerpt["artifacts"], whole["artifacts"]);
    }
    let all = server.json(&format!("{path}?max_bytes={}", size + 1));
    assert_eq!(
        (all["truncated"].clone(), all["markdown"].clone()),
        (json!(false), whole["markdown"].clone())
    );
    for bad in ["1023", "0", "-5", "big", "1e5", ""] {
        let reply = server.request("GET", &format!("{path}?max_bytes={bad}"), &[], &[]);
        assert_eq!(reply.status, 422, "{bad}");
        assert_eq!(reply.json()["reason"], "invalid_max_bytes", "{bad}");
    }
    server.stop();
}

/// An OpenAI-compatible server on loopback: records each chat request and
/// answers with the document it was sent, as the enhancement prompt asks.
struct Endpoint {
    port: u16,
    requests: Arc<Mutex<Vec<(String, String, Value)>>>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}
impl Endpoint {
    fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        listener.set_nonblocking(true).unwrap();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let (seen, stopping) = (requests.clone(), stop.clone());
        let thread = std::thread::spawn(move || {
            while !stopping.load(Ordering::SeqCst) {
                let mut stream = match listener.accept() {
                    Ok((stream, _)) => stream,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(2));
                        continue;
                    }
                    Err(_) => break,
                };
                stream.set_nonblocking(false).unwrap();
                stream.set_read_timeout(Some(WAIT)).unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut start = String::new();
                reader.read_line(&mut start).unwrap();
                let (mut length, mut authorization) = (0, String::new());
                loop {
                    let mut line = String::new();
                    reader.read_line(&mut line).unwrap();
                    if line == "\r\n" || line.is_empty() {
                        break;
                    }
                    if let Some((name, value)) = line.split_once(':') {
                        if name.eq_ignore_ascii_case("content-length") {
                            length = value.trim().parse().unwrap();
                        } else if name.eq_ignore_ascii_case("authorization") {
                            authorization = value.trim().to_owned();
                        }
                    }
                }
                let mut body = vec![0; length];
                reader.read_exact(&mut body).unwrap();
                let request: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
                let text = request["messages"]
                    .as_array()
                    .and_then(|messages| messages.iter().find(|m| m["role"] == "user"))
                    .and_then(|message| message["content"].as_str())
                    .unwrap_or_default()
                    .to_owned();
                seen.lock()
                    .unwrap()
                    .push((start.trim().to_owned(), authorization, request));
                let content = json!({"cleaned_markdown":text,"frontmatter":{"description":"Endpoint fixture","tags":["fixture"]}}).to_string();
                let answer = json!({"model":"fake-model","choices":[{"message":{"content":content},"finish_reason":"stop"}],
                    "usage":{"prompt_tokens":7,"completion_tokens":3}})
                .to_string();
                let _ = write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{answer}",
                    answer.len()
                );
            }
        });
        Self {
            port,
            requests,
            stop,
            thread: Some(thread),
        }
    }
}
impl Drop for Endpoint {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[test]
fn an_endpoint_saved_as_custom_converts_through_its_own_address() {
    let endpoint = Endpoint::start();
    let temp = tempfile::tempdir().unwrap();
    let server = Server::start(temp.path());
    let revision = server.json("/api/settings/llm")["revision"].clone();
    let base = format!("http://127.0.0.1:{}/v1", endpoint.port);
    let body = json!({"expected_revision":revision,"deployments":[{"provider":"custom",
        "model":"openai/fake-model","api_key":"k","api_base":base,"model_name":"default","weight":1}]});
    let reply = server.request(
        "POST",
        "/api/settings/llm/deployments/batch",
        &[("Content-Type", "application/json")],
        body.to_string().as_bytes(),
    );
    assert_eq!(reply.status, 200, "{}", reply.text());
    assert_eq!(reply.json()["routable"], true);
    let providers = server.json("/api/settings/llm/providers");
    let cards = providers["providers"].as_array().unwrap();
    assert!(
        cards
            .iter()
            .any(|card| card["kind"] == "configured" && card["provider"] == "custom")
    );
    for name in ["custom", "openai"] {
        assert!(
            cards
                .iter()
                .any(|card| card["kind"] == "common" && card["provider"] == name)
        );
    }
    const NOTES: &[u8] = b"# Notes\n\nA paragraph the endpoint returns unchanged.\n";
    let id = server.submit(&[("notes.md", NOTES)], json!([]), json!({"llm":true}))["job_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let snapshot = server.done(&id);
    let item = &snapshot["items"][0];
    assert_eq!(item["status"], "done", "{item}");
    assert_eq!(item["llm_enhanced"], true, "{item}");
    assert_eq!(item["models"], json!(["fake-model"]));
    let requests = endpoint.requests.lock().unwrap().clone();
    assert!(!requests.is_empty());
    for (start, authorization, request) in &requests {
        assert_eq!(start, "POST /v1/chat/completions HTTP/1.1");
        assert_eq!(authorization, "Bearer k");
        assert_eq!(request["model"], "fake-model");
    }
    // The models survive a restart, and so does an older history without them.
    server.stop();
    let restarted = Server::start(temp.path());
    assert_eq!(
        restarted.json(&format!("/api/jobs/{id}"))["items"][0]["models"],
        json!(["fake-model"])
    );
    restarted.stop();
    let meta = temp
        .path()
        .join("home/serve/jobs")
        .join(&id)
        .join("meta.json");
    let mut saved: Value = serde_json::from_slice(&std::fs::read(&meta).unwrap()).unwrap();
    for item in saved["items"].as_array_mut().unwrap() {
        let item = item.as_object_mut().unwrap();
        item.remove("models");
        item.remove("source_size");
    }
    std::fs::write(&meta, saved.to_string()).unwrap();
    let legacy = Server::start(temp.path());
    let item = &legacy.json(&format!("/api/jobs/{id}"))["items"][0];
    assert_eq!(item["models"], json!(["fake-model"]));
    assert_eq!(item["source_size"], NOTES.len());
    legacy.stop();
}

#[test]
fn history_is_kept_for_seven_days_after_a_job_finished() {
    let temp = tempfile::tempdir().unwrap();
    configure(temp.path());
    let jobs = temp.path().join("home/serve/jobs");
    let saved = |id: &str, status: &str, finished: Option<String>| {
        let folder = jobs.join(id);
        std::fs::create_dir_all(folder.join("out")).unwrap();
        std::fs::write(folder.join("out/a.md"), id).unwrap();
        let item = json!({"item_id":"i1","name":"a.txt","kind":"file","output":"a.md","output_name":"a.md",
            "status":if status == "running" { "queued" } else { "done" }});
        let meta = json!({"job_id":id,"created_at":hours_ago(30 * 24),"finished_at":finished,
            "status":status,"version":2,"options":{},"items":[item]});
        std::fs::write(folder.join("meta.json"), meta.to_string()).unwrap();
        folder
    };
    let expired = saved("00000000000a", "done", Some(hours_ago(8 * 24)));
    let kept = saved("00000000000b", "done", Some(hours_ago(6 * 24)));
    // Interrupted by a stop a month ago: it finishes when the server starts.
    let interrupted = saved("00000000000c", "running", None);
    let server = Server::start(temp.path());
    assert_eq!(
        server.json("/api/capabilities")["limits"]["history_retention_days"],
        7
    );
    assert!(!expired.exists());
    assert!(kept.exists() && interrupted.exists());
    let listed = |server: &Server| -> Vec<String> {
        let mut ids: Vec<String> = server
            .json("/api/history")
            .as_array()
            .unwrap()
            .iter()
            .map(|entry| entry["job_id"].as_str().unwrap().to_owned())
            .collect();
        ids.sort();
        ids
    };
    assert_eq!(listed(&server), ["00000000000b", "00000000000c"]);
    let reply = server.request("GET", "/api/jobs/00000000000a", &[], &[]);
    assert_eq!(
        (reply.status, reply.json()["reason"].clone()),
        (404, json!("job_not_found"))
    );
    // A job that is already past its retention when it appears is never
    // listed, served or archived, even before a sweep removes it.
    let late = saved("00000000000d", "done", Some(hours_ago(8 * 24)));
    assert_eq!(listed(&server), ["00000000000b", "00000000000c"]);
    for path in [
        "/api/jobs/00000000000d",
        "/api/jobs/00000000000d/archive",
        "/api/jobs/00000000000d/items/i1/result",
    ] {
        assert_eq!(server.request("GET", path, &[], &[]).status, 404, "{path}");
    }
    let archive = server.request("GET", "/api/history/archive", &[], &[]);
    assert_eq!(archive.status, 200);
    let contents = zip_contents(&archive.body);
    assert_eq!(contents.len(), 2);
    assert!(!contents.values().any(|bytes| bytes == b"00000000000d"));
    assert!(late.exists());
    server.stop();
    // The next start sweeps it.
    let restarted = Server::start(temp.path());
    assert!(!late.exists());
    assert_eq!(listed(&restarted), ["00000000000b", "00000000000c"]);
    restarted.stop();
}

const PDF: &[u8] =
    include_bytes!("../../../markitai-core/src/pdf_raster/fixtures/mixed-native-scanned-blank.pdf");
const PNG: &[u8] = include_bytes!("../../../markitai-core/src/ocr/fixtures/english.png");
const HEIC: &[u8] =
    include_bytes!("../../../markitai-core/src/images/fixtures/heif/quadrants-orientation6.heic");

/// Width and height of a PNG, from its header.
fn png_size(bytes: &[u8]) -> (u32, u32) {
    assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
    let number = |at: usize| u32::from_be_bytes(bytes[at..at + 4].try_into().unwrap());
    (number(16), number(20))
}
fn sandboxed(reply: &Reply) {
    assert_eq!(
        reply.headers["content-security-policy"],
        "sandbox; default-src 'none'"
    );
    assert_eq!(reply.headers["x-content-type-options"], "nosniff");
    assert!(reply.headers["content-disposition"].starts_with("attachment; filename*=UTF-8''"));
}

#[test]
fn an_uploaded_pdf_is_described_drawn_and_read_by_line() {
    let temp = tempfile::tempdir().unwrap();
    let server = Server::start(temp.path());
    let capabilities = server.json("/api/capabilities");
    assert_eq!(
        (
            capabilities["implementation"].clone(),
            capabilities["docs_url"].clone(),
            capabilities["source_url"].clone()
        ),
        (
            json!("rust"),
            json!("https://github.com/Ynewtime/markitai-rust/blob/main/docs/index.md"),
            json!("https://github.com/Ynewtime/markitai-rust")
        )
    );
    assert_eq!(
        capabilities["features"],
        json!({"cancel":true,"download_tickets":true,"url_lists":true,"history_items":true,
            "original":true,"result_excerpt":true,"job_labels":true,"item_assets":true,
            "item_archive":true,"token_regenerate":false})
    );
    let id = server.submit(&[("report.pdf", PDF)], json!([]), json!({}))["job_id"]
        .as_str()
        .unwrap()
        .to_owned();
    server.done(&id);
    let base = format!("/api/jobs/{id}/items/i1/original");
    let reply = server.request("GET", &base, &[], &[]);
    assert_eq!(reply.status, 200, "{}", reply.text());
    sandboxed(&reply);
    let descriptor = reply.json();
    assert_eq!(
        descriptor,
        json!({"kind":"pdf","name":"report.pdf","media_type":"application/pdf","size":PDF.len(),
            "download":format!("{base}/file"),"view":null,"page_count":3,"text_layer":true,"reason":null,
            "pages":[{"width":612.0,"height":792.0},{"width":612.0,"height":792.0},{"width":612.0,"height":792.0}]})
    );
    let file = server.request("GET", &format!("{base}/file"), &[], &[]);
    assert_eq!((file.status, file.body.as_slice()), (200, PDF));
    assert_eq!(file.headers["content-type"], "application/pdf");
    sandboxed(&file);
    let page = server.request("GET", &format!("{base}/pages/1?width=400"), &[], &[]);
    assert_eq!(page.status, 200, "{}", page.text());
    assert_eq!(page.headers["content-type"], "image/png");
    assert_eq!(page.headers["cache-control"], "private, max-age=86400");
    sandboxed(&page);
    assert_eq!(png_size(&page.body), (400, 518));
    let tag = page.headers["etag"].clone();
    let again = server.request(
        "GET",
        &format!("{base}/pages/1?width=400"),
        &[("If-None-Match", &tag)],
        &[],
    );
    assert_eq!(again.status, 304);
    // Widths are bounded; the default is 1200.
    let narrow = server.request("GET", &format!("{base}/pages/2?width=5"), &[], &[]);
    assert_eq!(png_size(&narrow.body).0, 160);
    let default = server.request("GET", &format!("{base}/pages/3"), &[], &[]);
    assert_eq!(png_size(&default.body).0, 1200);
    let text = server.json(&format!("{base}/pages/1/text"));
    assert_eq!(
        (
            text["page"].clone(),
            text["width"].clone(),
            text["height"].clone()
        ),
        (json!(1), json!(612.0), json!(792.0))
    );
    let lines = text["lines"].as_array().unwrap();
    let heading = lines
        .iter()
        .find(|line| line["text"].as_str().unwrap().contains("NATIVE PAGE ONE"))
        .unwrap_or_else(|| panic!("{text}"));
    let bbox: Vec<f64> = heading["bbox"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap())
        .collect();
    assert!(
        bbox[0] < bbox[2] && bbox[1] < bbox[3] && bbox[3] < 396.0,
        "{bbox:?}"
    );
    assert_eq!(
        server.json(&format!("{base}/pages/2/text"))["lines"],
        json!([])
    );
    for (path, status, reason) in [
        (format!("{base}/pages/4"), 404, "page_not_found"),
        (format!("{base}/pages/0"), 404, "page_not_found"),
        (format!("{base}/pages/4/text"), 404, "page_not_found"),
        (format!("{base}/pages/one"), 422, "invalid_page_request"),
        (
            format!("{base}/pages/1?width=wide"),
            422,
            "invalid_page_request",
        ),
        (
            format!("/api/jobs/{id}/items/i9/original"),
            404,
            "item_not_found",
        ),
        (
            format!("/api/jobs/{id}/items/%2E%2E/original/file"),
            404,
            "item_not_found",
        ),
        (
            format!("/api/jobs/{id}/items/..%2Fmeta.json/original/file"),
            404,
            "item_not_found",
        ),
        (
            "/api/jobs/000000000000/items/i1/original".into(),
            404,
            "job_not_found",
        ),
    ] {
        let reply = server.request("GET", &path, &[], &[]);
        assert_eq!(
            (reply.status, reply.json()["reason"].as_str()),
            (status, Some(reason)),
            "{path}"
        );
    }
    // The original's download may be opened through a ticket.
    let reply = server.request(
        "POST",
        "/api/download-tickets",
        &[("Content-Type", "application/json")],
        json!({"path":format!("{base}/file")})
            .to_string()
            .as_bytes(),
    );
    assert_eq!(reply.status, 201, "{}", reply.text());
    server.stop();
}

#[test]
fn images_html_text_and_unsupported_uploads_say_what_they_can_show() {
    let temp = tempfile::tempdir().unwrap();
    let server = Server::start(temp.path());
    let page = b"<html><body><h1>Kept</h1><script>alert(1)</script></body></html>";
    let id = server.submit(
        &[
            ("scan.png", PNG),
            ("photo.heic", HEIC),
            ("page.html", page),
            ("notes.md", b"# Notes\n"),
            (
                "drawing.svg",
                b"<svg xmlns=\"http://www.w3.org/2000/svg\"/>",
            ),
            ("data.xyz", b"opaque"),
        ],
        json!([]),
        json!({}),
    )["job_id"]
        .as_str()
        .unwrap()
        .to_owned();
    server.done(&id);
    let describe = |item: &str| server.json(&format!("/api/jobs/{id}/items/{item}/original"));
    let png = describe("i1");
    assert_eq!(
        (
            png["kind"].clone(),
            png["pages"].clone(),
            png["page_count"].clone()
        ),
        (
            json!("image"),
            json!([{"width":819.0,"height":301.0}]),
            json!(1)
        )
    );
    // A page no wider than asked is the stored image; a wider one is scaled.
    let path = |item: &str, rest: &str| format!("/api/jobs/{id}/items/{item}/original/{rest}");
    let stored = server.request("GET", &path("i1", "pages/1?width=1000"), &[], &[]);
    assert_eq!((stored.status, stored.body.as_slice()), (200, PNG));
    let scaled = server.request("GET", &path("i1", "pages/1?width=400"), &[], &[]);
    assert_eq!(png_size(&scaled.body), (400, 147));
    assert_eq!(
        server
            .request("GET", &path("i1", "pages/2"), &[], &[])
            .status,
        404
    );
    // HEIC is always transcoded, turned as it is shown.
    assert_eq!(
        describe("i2")["pages"],
        json!([{"width":80.0,"height":120.0}])
    );
    let heic = server.request("GET", &path("i2", "pages/1"), &[], &[]);
    assert_eq!(heic.headers["content-type"], "image/png");
    assert_eq!(png_size(&heic.body), (80, 120));
    // Markup is downloaded as plain text and never rendered.
    let html = describe("i3");
    assert_eq!(
        (
            html["kind"].clone(),
            html["media_type"].clone(),
            html["pages"].clone()
        ),
        (json!("html"), json!("text/html"), json!([]))
    );
    let file = server.request("GET", &path("i3", "file"), &[], &[]);
    assert_eq!(file.headers["content-type"], "text/plain; charset=utf-8");
    assert_eq!(file.body, page);
    sandboxed(&file);
    for (rest, reason) in [
        ("pages/1", "page_not_found"),
        ("pages/1/text", "original_unavailable"),
    ] {
        let reply = server.request("GET", &path("i3", rest), &[], &[]);
        assert_eq!(reply.json()["reason"], reason, "{rest}");
    }
    assert_eq!(describe("i4")["kind"], "text");
    let svg = server.request("GET", &path("i5", "file"), &[], &[]);
    assert_eq!(svg.headers["content-type"], "text/plain; charset=utf-8");
    for item in ["i5", "i6"] {
        let other = describe(item);
        assert_eq!(
            (other["kind"].clone(), other["reason"].clone()),
            (json!("none"), json!("unsupported_format")),
            "{item}"
        );
        assert!(
            other["download"]
                .as_str()
                .unwrap()
                .ends_with("/original/file")
        );
    }
    server.stop();
}

#[test]
fn a_url_item_keeps_its_page_and_a_cli_record_keeps_nothing() {
    let origin = Origin::start();
    let temp = tempfile::tempdir().unwrap();
    let server = Server::start(temp.path());
    let id = server.submit(
        &[],
        json!([origin.url("/first"), origin.url("/second")]),
        json!({"strategy":"static"}),
    )["job_id"]
        .as_str()
        .unwrap()
        .to_owned();
    server.done(&id);
    let original = server.json(&format!("/api/jobs/{id}/items/i1/original"));
    assert_eq!(
        (
            original["kind"].clone(),
            original["name"].clone(),
            original["media_type"].clone()
        ),
        (
            json!("html"),
            json!(origin.url("/first")),
            json!("text/html")
        )
    );
    let file = server.request(
        "GET",
        &format!("/api/jobs/{id}/items/i1/original/file"),
        &[],
        &[],
    );
    assert_eq!(file.headers["content-type"], "text/plain; charset=utf-8");
    assert!(
        file.text().contains("Origin response for /first."),
        "{}",
        file.text()
    );
    assert!(file.headers["content-disposition"].ends_with(".html"));
    let snapshots = server.jobdir(&id).join("snapshots");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(snapshots.join("i1.html"))
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(mode & 0o077, 0);
    }
    // A retry replaces the page with the one it converted.
    let reply = server.request(
        "POST",
        &format!("/api/jobs/{id}/items/i1/retry"),
        &[("Content-Type", "application/json")],
        b"{}",
    );
    assert_eq!(reply.status, 202, "{}", reply.text());
    server.done(&id);
    assert!(
        std::fs::read_to_string(snapshots.join("i1.html"))
            .unwrap()
            .contains("/first")
    );
    // Deleting the item deletes its page.
    let reply = server.request("DELETE", &format!("/api/jobs/{id}/items/i1"), &[], &[]);
    assert_eq!(reply.status, 204, "{}", reply.text());
    assert!(!snapshots.join("i1.html").exists());
    assert!(snapshots.join("i2.html").exists());
    // A job the CLI recorded keeps no original.
    let cli = "0123456789ab";
    let folder = server.jobdir(cli);
    std::fs::create_dir_all(folder.join("out")).unwrap();
    std::fs::write(folder.join("out/report.pdf.md"), "# Report\n").unwrap();
    let meta = json!({"job_id":cli,"created_at":hours_ago(1),"finished_at":hours_ago(1),"status":"done",
        "options":{"origin":"cli"},"items":[{"item_id":"i1","name":"report.pdf","kind":"file","status":"done",
        "output":"report.pdf.md","output_name":"report.pdf.md"}]});
    std::fs::write(folder.join("meta.json"), meta.to_string()).unwrap();
    server.json("/api/history");
    let none = server.json(&format!("/api/jobs/{cli}/items/i1/original"));
    assert_eq!(
        none,
        json!({"kind":"none","name":"report.pdf","media_type":null,"size":null,"download":null,
            "view":null,"pages":[],"page_count":0,"text_layer":false,"reason":"not_kept"})
    );
    for rest in ["file", "pages/1", "pages/1/text"] {
        let reply = server.request(
            "GET",
            &format!("/api/jobs/{cli}/items/i1/original/{rest}"),
            &[],
            &[],
        );
        assert_eq!(
            (reply.status, reply.json()["reason"].clone()),
            (404, json!("original_unavailable")),
            "{rest}"
        );
    }
    server.stop();
}

#[test]
fn saved_page_screenshots_stand_in_for_an_original_that_was_not_kept() {
    let temp = tempfile::tempdir().unwrap();
    configure(temp.path());
    let id = "00000000abcd";
    let folder = temp.path().join("home/serve/jobs").join(id);
    std::fs::create_dir_all(folder.join("out/.markitai/screenshots")).unwrap();
    std::fs::write(folder.join("out/a.md"), "# A\n").unwrap();
    for name in ["a.full.png", "a.full--2.png"] {
        std::fs::write(folder.join("out/.markitai/screenshots").join(name), PNG).unwrap();
    }
    let item = json!({"item_id":"i1","name":"https://example.test/a","kind":"url","status":"done",
        "output":"a.md","output_name":"a.md"});
    let meta = json!({"job_id":id,"created_at":hours_ago(1),"finished_at":hours_ago(1),"status":"done",
        "version":2,"options":{},"items":[item],"native_bases":{"i1":"a"},
        "native_assets":{"i1":[".markitai/screenshots/a.full.png",".markitai/screenshots/a.full--2.png"]}});
    std::fs::write(folder.join("meta.json"), meta.to_string()).unwrap();
    let server = Server::start(temp.path());
    let base = format!("/api/jobs/{id}/items/i1/original");
    let original = server.json(&base);
    assert_eq!(
        (
            original["kind"].clone(),
            original["page_count"].clone(),
            original["download"].clone()
        ),
        (json!("screenshots"), json!(2), Value::Null)
    );
    assert_eq!(original["pages"][1], json!({"width":819.0,"height":301.0}));
    let stored = server.request("GET", &format!("{base}/pages/2?width=1000"), &[], &[]);
    assert_eq!(stored.body, PNG);
    let scaled = server.request("GET", &format!("{base}/pages/1?width=400"), &[], &[]);
    assert_eq!(png_size(&scaled.body), (400, 147));
    assert_eq!(
        server
            .request("GET", &format!("{base}/pages/3"), &[], &[])
            .status,
        404
    );
    assert_eq!(
        server
            .request("GET", &format!("{base}/file"), &[], &[])
            .json()["reason"],
        "original_unavailable"
    );
    server.stop();
}

#[test]
fn an_item_archive_holds_its_result_files_and_its_kept_original() {
    let origin = Origin::start();
    let temp = tempfile::tempdir().unwrap();
    let server = Server::start(temp.path());
    let capabilities = server.json("/api/capabilities");
    assert_eq!(capabilities["features"]["item_archive"], true);
    assert_eq!(capabilities["features"]["token_regenerate"], false);
    let notes = b"# Notes\n\nKept text.\n";
    let id = server.submit(
        &[("notes.md", notes), ("data.xyz", b"opaque")],
        json!([origin.url("/first")]),
        json!({"strategy":"static"}),
    )["job_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let snapshot = server.done(&id);
    assert_eq!(snapshot["items"][0]["status"], "done", "{snapshot}");
    assert_eq!(snapshot["items"][1]["status"], "error", "{snapshot}");
    for (item, original) in [("i1", "original/notes.md"), ("i3", "")] {
        let result = server.json(&format!("/api/jobs/{id}/items/{item}/result"));
        let reply = server.request(
            "GET",
            &format!("/api/jobs/{id}/items/{item}/archive"),
            &[],
            &[],
        );
        assert_eq!(reply.status, 200, "{}", reply.text());
        assert_eq!(reply.headers["content-type"], "application/zip");
        assert_eq!(reply.headers["x-content-type-options"], "nosniff");
        let output = snapshot["items"]
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["item_id"] == item)
            .unwrap()["output"]
            .as_str()
            .unwrap()
            .to_owned();
        let base = output.strip_suffix(".md").unwrap();
        assert_eq!(
            reply.headers["content-disposition"],
            format!("attachment; filename*=UTF-8''{}.zip", encode_path(base))
        );
        let files = zip_contents(&reply.body);
        let file = server.request(
            "GET",
            &format!("/api/jobs/{id}/items/{item}/original/file"),
            &[],
            &[],
        );
        let sent = url::form_urlencoded::parse(
            format!(
                "name={}",
                file.headers["content-disposition"]
                    .strip_prefix("attachment; filename*=UTF-8''")
                    .unwrap()
            )
            .as_bytes(),
        )
        .next()
        .unwrap()
        .1
        .into_owned();
        let original = if original.is_empty() {
            format!("original/{sent}")
        } else {
            assert_eq!(format!("original/{sent}"), original);
            original.to_owned()
        };
        let mut expected: std::collections::BTreeSet<String> = result["artifacts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|artifact| artifact["relpath"].as_str().unwrap().to_owned())
            .collect();
        expected.insert(original.clone());
        assert_eq!(
            files
                .keys()
                .cloned()
                .collect::<std::collections::BTreeSet<_>>(),
            expected,
            "{item}"
        );
        assert_eq!(files[&original], file.body, "{item}");
        for artifact in result["artifacts"].as_array().unwrap() {
            let relpath = artifact["relpath"].as_str().unwrap();
            assert_eq!(
                files[relpath],
                std::fs::read(server.jobdir(&id).join("out").join(relpath)).unwrap()
            );
        }
    }
    let reason = |path: &str| {
        let reply = server.request("GET", path, &[], &[]);
        (reply.status, reply.json()["reason"].clone())
    };
    assert_eq!(
        reason(&format!("/api/jobs/{id}/items/i2/archive")),
        (404, json!("result_unavailable"))
    );
    assert_eq!(
        reason(&format!("/api/jobs/{id}/items/i9/archive")),
        (404, json!("item_not_found"))
    );
    assert_eq!(
        reason("/api/jobs/000000000000/items/i1/archive"),
        (404, json!("job_not_found"))
    );
    // A running item is busy.
    let held = server.submit(
        &[],
        json!([origin.url("/hold")]),
        json!({"strategy":"static"}),
    )["job_id"]
        .as_str()
        .unwrap()
        .to_owned();
    until("held URL reaches origin", || {
        origin.entered.load(Ordering::SeqCst) >= 2
    });
    assert_eq!(
        reason(&format!("/api/jobs/{held}/items/i1/archive")),
        (409, json!("item_busy"))
    );
    origin.release();
    server.done(&held);
    assert_eq!(
        server
            .request(
                "GET",
                &format!("/api/jobs/{held}/items/i1/archive"),
                &[],
                &[]
            )
            .status,
        200
    );
    server.stop();
}

#[test]
fn a_kept_html_page_is_framed_with_its_base_and_a_closed_policy() {
    let origin = Origin::start();
    let temp = tempfile::tempdir().unwrap();
    let server = Server::start(temp.path());
    let page = b"<!DOCTYPE html><html><HEAD><title>Kept</title></HEAD><body><h1>Kept</h1><header>h</header></body></html>";
    let id = server.submit(
        &[("page.html", page), ("notes.md", b"# Notes\n")],
        json!([origin.url("/first?a=1&b=\"2\"")]),
        json!({"strategy":"static"}),
    )["job_id"]
        .as_str()
        .unwrap()
        .to_owned();
    server.done(&id);
    let describe = |item: &str| server.json(&format!("/api/jobs/{id}/items/{item}/original"));
    let path = |item: &str| format!("/api/jobs/{id}/items/{item}/original/view");
    const META: &str = "<meta http-equiv=\"x-dns-prefetch-control\" content=\"off\">";
    let upload = describe("i1");
    assert_eq!(upload["view"], path("i1"));
    assert_eq!(
        (upload["pages"].clone(), upload["page_count"].clone()),
        (json!([]), json!(0))
    );
    let view = server.request("GET", &path("i1"), &[], &[]);
    assert_eq!(view.status, 200, "{}", view.text());
    for (name, value) in [
        ("content-type", "text/html; charset=utf-8"),
        (
            "content-security-policy",
            "sandbox allow-popups allow-popups-to-escape-sandbox; default-src 'none'; style-src 'unsafe-inline'; img-src data:; font-src data:; form-action 'none'; frame-ancestors 'self'",
        ),
        ("x-content-type-options", "nosniff"),
        ("referrer-policy", "no-referrer"),
        ("x-dns-prefetch-control", "off"),
        ("cache-control", "no-store"),
        ("content-disposition", "inline"),
    ] {
        assert_eq!(view.headers[name], value, "{name}");
    }
    assert!(!view.headers.contains_key("x-frame-options"));
    assert_eq!(
        view.text(),
        format!(
            "<!DOCTYPE html><html><HEAD>{META}<base target=\"_blank\"><title>Kept</title></HEAD><body><h1>Kept</h1><header>h</header></body></html>"
        )
    );
    // A URL item's page resolves its links against its own address.
    let url = describe("i3");
    assert_eq!(url["kind"], "html");
    assert_eq!(url["view"], path("i3"));
    let framed = server.request("GET", &path("i3"), &[], &[]).text();
    let address = url["name"]
        .as_str()
        .unwrap()
        .replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('\'', "&#39;");
    assert!(address.contains("/first?a=1&amp;b="), "{address}");
    let snapshot = std::fs::read_to_string(server.jobdir(&id).join("snapshots/i3.html")).unwrap();
    let inserted = format!("{META}<base href=\"{address}\" target=\"_blank\">");
    assert_eq!(framed.matches(&inserted).count(), 1, "{framed}");
    assert_eq!(framed.replacen(&inserted, "", 1), snapshot);
    // Anything but HTML has no view.
    assert_eq!(describe("i2")["view"], Value::Null);
    let reply = server.request("GET", &path("i2"), &[], &[]);
    assert_eq!(
        (reply.status, reply.json()["reason"].clone()),
        (404, json!("original_unavailable"))
    );
    server.stop();
}

#[test]
fn a_url_page_with_screenshots_offers_both_and_a_large_page_is_not_framed() {
    let temp = tempfile::tempdir().unwrap();
    configure(temp.path());
    let id = "00000000abce";
    let folder = temp.path().join("home/serve/jobs").join(id);
    std::fs::create_dir_all(folder.join("out/.markitai/screenshots")).unwrap();
    std::fs::create_dir_all(folder.join("snapshots")).unwrap();
    std::fs::write(folder.join("out/a.md"), "# A\n").unwrap();
    std::fs::write(folder.join("out/b.md"), "# B\n").unwrap();
    for name in ["a.full.png", "a.full--2.png"] {
        std::fs::write(folder.join("out/.markitai/screenshots").join(name), PNG).unwrap();
    }
    std::fs::write(folder.join("snapshots/i1.html"), "<p>A</p>").unwrap();
    let mut large = b"<html><head></head><body>".to_vec();
    large.resize(10 * 1024 * 1024 + 1, b'x');
    std::fs::write(folder.join("snapshots/i2.html"), &large).unwrap();
    let item = |id: &str, base: &str| {
        json!({"item_id":id,"name":format!("https://example.test/{base}"),"kind":"url","status":"done",
            "output":format!("{base}.md"),"output_name":format!("{base}.md")})
    };
    let meta = json!({"job_id":id,"created_at":hours_ago(1),"finished_at":hours_ago(1),"status":"done",
        "version":2,"options":{},"items":[item("i1","a"),item("i2","b")],"native_bases":{"i1":"a","i2":"b"},
        "native_assets":{"i1":[".markitai/screenshots/a.full.png",".markitai/screenshots/a.full--2.png"],"i2":[]}});
    std::fs::write(folder.join("meta.json"), meta.to_string()).unwrap();
    let server = Server::start(temp.path());
    let base = format!("/api/jobs/{id}/items/i1/original");
    let original = server.json(&base);
    assert_eq!(
        (
            original["kind"].clone(),
            original["page_count"].clone(),
            original["view"].clone()
        ),
        (json!("html"), json!(2), json!(format!("{base}/view")))
    );
    assert_eq!(original["pages"][1], json!({"width":819.0,"height":301.0}));
    assert!(
        original["download"]
            .as_str()
            .unwrap()
            .ends_with("/original/file")
    );
    let stored = server.request("GET", &format!("{base}/pages/2?width=1000"), &[], &[]);
    assert_eq!((stored.status, stored.body.as_slice()), (200, PNG));
    assert_eq!(
        server
            .request("GET", &format!("{base}/pages/3"), &[], &[])
            .status,
        404
    );
    let framed = server
        .request("GET", &format!("{base}/view"), &[], &[])
        .text();
    assert_eq!(
        framed,
        "<meta http-equiv=\"x-dns-prefetch-control\" content=\"off\"><base href=\"https://example.test/a\" target=\"_blank\"><p>A</p>"
    );
    // Without screenshots, a page has no pages; past 10 MiB it is not framed.
    let large = server.json(&format!("/api/jobs/{id}/items/i2/original"));
    assert_eq!(
        (large["kind"].clone(), large["page_count"].clone()),
        (json!("html"), json!(0))
    );
    let reply = server.request(
        "GET",
        &format!("/api/jobs/{id}/items/i2/original/view"),
        &[],
        &[],
    );
    assert_eq!(
        (reply.status, reply.json()["reason"].clone()),
        (413, json!("result_too_large"))
    );
    server.stop();
}
