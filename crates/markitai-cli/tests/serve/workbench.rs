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
