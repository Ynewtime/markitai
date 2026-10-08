use super::*;

fn retry(server: &Server, id: &str, item: &str, body: Option<Value>) -> Reply {
    let raw = body.map(|v| v.to_string()).unwrap_or_default();
    server.request(
        "POST",
        &format!("/api/jobs/{id}/items/{item}/retry"),
        &[("Content-Type", "application/json")],
        raw.as_bytes(),
    )
}
fn result(server: &Server, id: &str, item: &str) -> Value {
    server.json(&format!("/api/jobs/{id}/items/{item}/result"))
}

#[test]
fn retry_inherits_per_item_options_replaces_supplied_options_and_survives_restart() {
    let temp = tempfile::tempdir().unwrap();
    let server = Server::start(temp.path());
    let created = server.submit(
        &[
            ("a.txt", b"First original body."),
            ("b.txt", b"Second original body."),
        ],
        json!([]),
        json!({"profile":"obsidian","llm":false}),
    );
    let id = created["job_id"].as_str().unwrap().to_owned();
    server.done(&id);
    let upload = server.jobdir(&id).join("uploads/a.txt");
    std::fs::write(&upload, b"Revised source body.").unwrap();
    assert_eq!(
        retry(
            &server,
            &id,
            "i1",
            Some(json!({"options":{"profile":"rag"}}))
        )
        .status,
        202
    );
    let done = server.done(&id);
    assert_eq!(done["items"][0]["operation"], "retry");
    assert!(
        result(&server, &id, "i1")["markdown"]
            .as_str()
            .unwrap()
            .contains("Revised source body.")
    );
    assert_eq!(retry(&server, &id, "i2", None).status, 202);
    server.done(&id);
    let meta: Value =
        serde_json::from_slice(&std::fs::read(server.jobdir(&id).join("meta.json")).unwrap())
            .unwrap();
    assert_eq!(meta["items"][0]["options"]["profile"], "rag");
    assert_eq!(meta["items"][1]["options"]["profile"], "obsidian");
    assert_eq!(meta["items"][0]["options"]["llm"], Value::Null);
    let before = result(&server, &id, "i1");
    server.stop();
    let server = Server::start(temp.path());
    assert_eq!(result(&server, &id, "i1"), before);
    assert_eq!(retry(&server, &id, "i1", None).status, 202);
    server.done(&id);
    let meta: Value =
        serde_json::from_slice(&std::fs::read(server.jobdir(&id).join("meta.json")).unwrap())
            .unwrap();
    assert_eq!(meta["items"][0]["options"]["profile"], "rag");
    assert_eq!(
        std::fs::read_dir(server.jobdir(&id).join("out"))
            .unwrap()
            .filter_map(Result::ok)
            .filter(|e| e.path().extension().is_some_and(|v| v == "md"))
            .count(),
        2
    );
    server.stop();
}

#[test]
fn retry_starts_while_initial_sibling_runs_and_duplicate_retry_is_rejected() {
    let origin = Origin::start();
    let temp = tempfile::tempdir().unwrap();
    let server = Server::start(temp.path());
    let created = server.submit(
        &[("first.txt", b"Original body.")],
        json!([origin.url("/hold")]),
        json!({}),
    );
    let id = created["job_id"].as_str().unwrap();
    until("file done alongside held URL", || {
        server.json(&format!("/api/jobs/{id}"))["items"][0]["status"] == "done"
    });
    std::fs::write(
        server.jobdir(id).join("uploads/first.txt"),
        b"Changed during sibling conversion.",
    )
    .unwrap();
    assert_eq!(retry(&server, id, "i1", None).status, 202);
    until("retry finishes independently", || {
        let snap = server.json(&format!("/api/jobs/{id}"));
        snap["status"] == "running"
            && snap["items"][0]["status"] == "done"
            && snap["items"][0]["operation"] == "retry"
    });
    assert!(
        result(&server, id, "i1")["markdown"]
            .as_str()
            .unwrap()
            .contains("Changed during sibling conversion.")
    );
    assert_eq!(
        server
            .request("DELETE", &format!("/api/jobs/{id}/items/i1"), &[], &[])
            .status,
        409
    );
    assert_eq!(retry(&server, id, "i2", None).status, 409);
    origin.release();
    server.done(id);
    server.stop();
}

#[test]
fn retry_admission_errors_do_not_change_terminal_rows_or_artifacts() {
    let temp = tempfile::tempdir().unwrap();
    let server = Server::start(temp.path());
    let created = server.submit(&[("a.txt", b"Keep this result.")], json!([]), json!({}));
    let id = created["job_id"].as_str().unwrap();
    server.done(id);
    let previous = result(&server, id, "i1");
    for body in [
        json!({"operation":"wrong"}),
        json!({"bogus":true}),
        json!({"options":{"bogus":true}}),
    ] {
        assert_eq!(retry(&server, id, "i1", Some(body)).status, 422);
    }
    assert_eq!(
        retry(&server, id, "i1", Some(json!({"operation":"enhance"}))).status,
        409
    );
    assert_eq!(retry(&server, id, "missing", None).status, 404);
    std::fs::remove_file(server.jobdir(id).join("uploads/a.txt")).unwrap();
    assert_eq!(retry(&server, id, "i1", None).status, 404);
    assert_eq!(result(&server, id, "i1"), previous);
    server.stop();
}

#[test]
fn deleting_one_item_keeps_shared_assets_and_last_delete_removes_job() {
    let temp = tempfile::tempdir().unwrap();
    let server = Server::start(temp.path());
    let created = server.submit(
        &[
            ("a.md", b"![shared](assets/shared.png)\nFirst."),
            ("b.md", b"![shared](assets/shared.png)\nSecond."),
        ],
        json!([]),
        json!({}),
    );
    let id = created["job_id"].as_str().unwrap().to_owned();
    server.done(&id);
    server.stop();
    let folder = temp.path().join("home/serve/jobs").join(&id);
    let assets = folder.join("out/assets");
    std::fs::create_dir_all(&assets).unwrap();
    std::fs::write(assets.join("shared.png"), b"shared-bytes").unwrap();
    let meta = folder.join("meta.json");
    let mut data: Value = serde_json::from_slice(&std::fs::read(&meta).unwrap()).unwrap();
    std::fs::write(assets.join("only-a.png"), b"exclusive").unwrap();
    std::fs::write(assets.join("images.json"),json!({"created":"prior","images":[{"path":assets.join("shared.png"),"desc":"shared"},{"path":assets.join("only-a.png"),"desc":"removed"}]}).to_string()).unwrap();
    data["native_assets"] =
        json!({"i1":["assets/shared.png","assets/only-a.png"],"i2":["assets/shared.png"]});
    std::fs::write(&meta, data.to_string()).unwrap();
    // The reference caches a job archive that still holds the deleted item.
    std::fs::write(folder.join("archive.zip"), b"reference-archive").unwrap();
    let server = Server::start(temp.path());
    assert_eq!(
        server
            .request("DELETE", &format!("/api/jobs/{id}/items/i1"), &[], &[])
            .status,
        204
    );
    assert!(!folder.join("archive.zip").exists());
    assert_eq!(
        std::fs::read(assets.join("shared.png")).unwrap(),
        b"shared-bytes"
    );
    assert!(!assets.join("only-a.png").exists());
    let index: Value =
        serde_json::from_slice(&std::fs::read(assets.join("images.json")).unwrap()).unwrap();
    assert_eq!(index["images"].as_array().unwrap().len(), 1);
    assert_eq!(index["images"][0]["desc"], "shared");
    assert_eq!(index["created"], "prior");
    assert_eq!(
        server
            .request(
                "GET",
                &format!("/api/jobs/{id}/files/assets/.images.lock"),
                &[],
                &[]
            )
            .status,
        404
    );
    let archive = server.request("GET", &format!("/api/jobs/{id}/archive"), &[], &[]);
    assert_eq!(archive.status, 200);
    assert!(
        !zip_contents(&archive.body)
            .keys()
            .any(|name| name.ends_with(".images.lock"))
    );
    assert!(!folder.join("uploads/a.md").exists());
    assert!(!folder.join("out/a.md.md").exists());
    assert_eq!(
        server.json(&format!("/api/jobs/{id}"))["items"][0]["item_id"],
        "i2"
    );
    assert_eq!(
        server
            .request("DELETE", &format!("/api/jobs/{id}/items/i1"), &[], &[])
            .status,
        404
    );
    server.stop();
    let server = Server::start(temp.path());
    assert_eq!(server.json(&format!("/api/jobs/{id}"))["total"], 1);
    assert_eq!(
        server
            .request("DELETE", &format!("/api/jobs/{id}/items/i2"), &[], &[])
            .status,
        204
    );
    assert!(!folder.exists());
    assert_eq!(
        server
            .request("GET", &format!("/api/jobs/{id}"), &[], &[])
            .status,
        404
    );
    server.stop();
}

struct Model {
    port: u16,
    mode: Arc<AtomicUsize>,
    entered: Arc<AtomicUsize>,
    gate: Arc<(Mutex<bool>, Condvar)>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}
impl Model {
    fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        listener.set_nonblocking(true).unwrap();
        let mode = Arc::new(AtomicUsize::new(0));
        let entered = Arc::new(AtomicUsize::new(0));
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let (worker_mode, worker_entered, worker_gate, worker_stop) =
            (mode.clone(), entered.clone(), gate.clone(), stop.clone());
        let thread = std::thread::spawn(move || {
            let mut workers = Vec::new();
            while !worker_stop.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        let (mode, entered, gate) = (
                            worker_mode.clone(),
                            worker_entered.clone(),
                            worker_gate.clone(),
                        );
                        workers.push(std::thread::spawn(move||{
                        // Accepted sockets can inherit the listener's nonblocking mode on macOS.
                        stream.set_nonblocking(false).unwrap();
                        stream.set_read_timeout(Some(WAIT)).unwrap();stream.set_write_timeout(Some(WAIT)).unwrap();
                        let mut headers=Vec::new();let mut byte=[0];
                        while !headers.ends_with(b"\r\n\r\n") {
                            assert!(headers.len()<16384, "model request headers exceed fixture limit");
                            stream.read_exact(&mut byte).expect("read complete model request headers");
                            headers.push(byte[0]);
                        }
                        let text=String::from_utf8(headers).unwrap();let length=text.lines().find_map(|line|line.split_once(':').filter(|(name,_)|name.eq_ignore_ascii_case("content-length")).map(|(_,value)|value.trim().parse::<usize>().unwrap())).unwrap();assert!(length<2*1024*1024);
                        let mut body=vec![0;length];stream.read_exact(&mut body).unwrap();let request:Value=serde_json::from_slice(&body).unwrap();assert_eq!(request["model"],"fixture");
                        entered.fetch_add(1,Ordering::SeqCst);
                        let mode=mode.load(Ordering::SeqCst);
                        if mode==2{let(lock,notify)=&*gate;let ready=lock.lock().unwrap_or_else(|e|e.into_inner());let(ready,_)=notify.wait_timeout_while(ready,WAIT,|ready|!*ready).unwrap_or_else(|e|e.into_inner());if !*ready{return;}}
                        let (status,body)=if mode==1{("503 Service Unavailable",json!({"error":{"message":"fixture failure"}}))}else{("200 OK",json!({"choices":[{"message":{"content":model_content(&request,"# Enhanced fixture\n\nVerified model output.")},"finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":5}}))};
                        let body=body.to_string();let response=format!("HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",body.len());let _=stream.write_all(response.as_bytes());
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
            mode,
            entered,
            gate,
            stop,
            thread: Some(thread),
        }
    }
    fn configure(&self, root: &Path) {
        configure(root);
        let path = root.join("config.json");
        let mut cfg: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        cfg["llm"] = json!({"enabled":false,"on_failure":"fallback","keep_base":true,"router_settings":{"num_retries":0,"timeout":20},"model_list":[{"model_name":"fixture","litellm_params":{"model":"openai/fixture","api_key":"fixture-key","api_base":format!("http://127.0.0.1:{}/v1",self.port)}}]});
        std::fs::write(path, cfg.to_string()).unwrap();
    }
    fn hold(&self) {
        *self.gate.0.lock().unwrap_or_else(|e| e.into_inner()) = false;
        self.mode.store(2, Ordering::SeqCst);
    }
    fn release(&self) {
        *self.gate.0.lock().unwrap_or_else(|e| e.into_inner()) = true;
        self.gate.1.notify_all();
    }
}
impl Drop for Model {
    fn drop(&mut self) {
        self.release();
        self.stop.store(true, Ordering::SeqCst);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
fn enhance() -> Value {
    json!({"operation":"enhance","options":{"llm":true,"alt":false,"desc":false}})
}

#[test]
fn enhance_failure_restores_the_previous_pair_and_plain_retry_prunes_stale_variant() {
    let model = Model::start();
    let temp = tempfile::tempdir().unwrap();
    model.configure(temp.path());
    let server = Server::start(temp.path());
    let created = server.submit(
        &[("document.txt", b"Original conversion body.")],
        json!([]),
        json!({"llm":false}),
    );
    let id = created["job_id"].as_str().unwrap();
    server.done(id);
    assert_eq!(retry(&server, id, "i1", Some(enhance())).status, 202);
    let enhanced = server.done(id);
    assert_eq!(enhanced["items"][0]["llm_enhanced"], true);
    assert_eq!(enhanced["items"][0]["operation"], "enhance");
    let previous = result(&server, id, "i1");
    assert_eq!(previous["variant"], "llm");
    let out = server.jobdir(id).join("out");
    let base = std::fs::read(out.join("document.txt.md")).unwrap();
    let llm = std::fs::read(out.join("document.txt.llm.md")).unwrap();
    model.mode.store(1, Ordering::SeqCst);
    std::fs::write(
        server.jobdir(id).join("uploads/document.txt"),
        b"New source must not leak from a failed enhance.",
    )
    .unwrap();
    assert_eq!(retry(&server, id, "i1", Some(enhance())).status, 202);
    let failed = server.done(id);
    assert!(
        enhanced["items"][0]["diagnostics"]["last_attempt"]["usage"]["requests"]
            .as_u64()
            .is_some_and(|count| count > 0)
    );
    assert!(failed["items"][0].get("diagnostics").is_none());
    let outcome = failed["items"][0]["rerun_failure"].clone();
    assert_eq!(outcome["operation"], "enhance");
    assert_eq!(outcome["error_code"], "enhancement_failed");
    assert_eq!(
        outcome["error"],
        "LLM enhancement did not produce an enhanced result"
    );
    chrono::DateTime::parse_from_rfc3339(outcome["failed_at"].as_str().unwrap()).unwrap();
    let mut old_result = failed["items"][0].clone();
    old_result.as_object_mut().unwrap().remove("rerun_failure");
    let mut retained = enhanced["items"][0].clone();
    retained.as_object_mut().unwrap().remove("diagnostics");
    assert_eq!(old_result, retained);
    assert_eq!(result(&server, id, "i1"), previous);
    assert_eq!(std::fs::read(out.join("document.txt.md")).unwrap(), base);
    assert_eq!(std::fs::read(out.join("document.txt.llm.md")).unwrap(), llm);
    assert_eq!(model.entered.load(Ordering::SeqCst), 2);
    // The no-usage failure is durable even though the old successful output
    // still owns this row's status, timing and price.
    server.stop();
    let server = Server::start(temp.path());
    assert_eq!(
        server.json(&format!("/api/jobs/{id}"))["items"][0],
        failed["items"][0]
    );
    assert_eq!(result(&server, id, "i1"), previous);
    assert_eq!(
        retry(&server, id, "i1", Some(json!({"options":{"llm":false}}))).status,
        202
    );
    let plain = server.done(id);
    assert!(plain["items"][0].get("rerun_failure").is_none());
    assert_eq!(plain["items"][0]["llm_enhanced"], false);
    assert!(
        result(&server, id, "i1")["markdown"]
            .as_str()
            .unwrap()
            .contains("New source must not leak")
    );
    assert!(!out.join("document.txt.llm.md").exists());
    assert!(std::fs::read_dir(server.jobdir(id)).unwrap().all(|e| {
        !e.unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with(".retry-")
    }));
    server.stop();
}

#[test]
fn queued_duplicate_retry_is_rejected_and_shutdown_restores_queued_prior_result() {
    let model = Model::start();
    let temp = tempfile::tempdir().unwrap();
    model.configure(temp.path());
    let server = Server::start(temp.path());
    let created = server.submit(
        &[
            ("a.txt", b"First saved result."),
            ("b.txt", b"Second saved result."),
        ],
        json!([]),
        json!({}),
    );
    let id = created["job_id"].as_str().unwrap().to_owned();
    let original = server.done(&id);
    model.hold();
    assert_eq!(retry(&server, &id, "i1", Some(enhance())).status, 202);
    until("active enhancement", || {
        model.entered.load(Ordering::SeqCst) == 1
    });
    assert_eq!(retry(&server, &id, "i2", Some(enhance())).status, 202);
    assert_eq!(retry(&server, &id, "i2", Some(enhance())).status, 409);
    let events_port = server.port;
    let event_path = format!("/api/jobs/{id}/events");
    let (ready, subscribed) = std::sync::mpsc::channel();
    let events = std::thread::spawn(move || {
        request_with_ready(events_port, "GET", &event_path, &[], &[], Some(ready))
    });
    subscribed.recv_timeout(WAIT).unwrap();
    server.signal();
    model.release();
    server.finish(true);
    assert_eq!(model.entered.load(Ordering::SeqCst), 1);
    let events = events.join().unwrap();
    assert_eq!(events.status, 200);
    assert_eq!(
        events.headers.get("content-type").unwrap(),
        "text/event-stream"
    );
    let server = Server::start(temp.path());
    let saved = server.json(&format!("/api/jobs/{id}"));
    assert_eq!(saved["items"][0]["llm_enhanced"], true);
    let mut queued_result = saved["items"][1].clone();
    let failure = queued_result
        .as_object_mut()
        .unwrap()
        .remove("rerun_failure")
        .unwrap();
    assert_eq!(failure["operation"], "enhance");
    assert_eq!(failure["error_code"], "shutdown");
    assert_eq!(failure["error"], "cancelled (server shutdown)");
    assert_eq!(queued_result, original["items"][1]);
    server.stop();
}

#[test]
fn retry_metadata_failure_leaves_recovery_material_and_restart_restores_previous_bytes() {
    for operation in ["enhance", "retry"] {
        let model = Model::start();
        let temp = tempfile::tempdir().unwrap();
        model.configure(temp.path());
        let server = Server::start(temp.path());
        let created = server.submit(
            &[("a.txt", b"Durable original result.")],
            json!([]),
            json!({"profile":"obsidian","llm":false,"ocr":false}),
        );
        let id = created["job_id"].as_str().unwrap().to_owned();
        let initial = server.done(&id);
        let previous = result(&server, &id, "i1");
        let folder = server.jobdir(&id);
        let original = std::fs::read(folder.join("meta.json")).unwrap();
        let body = std::fs::read(folder.join("out/a.txt.md")).unwrap();
        model.hold();
        let mut attempt = enhance();
        attempt["operation"] = json!(operation);
        attempt["options"]["profile"] = json!("okf");
        assert_eq!(retry(&server, &id, "i1", Some(attempt)).status, 202);
        until("held enhanced output", || {
            model.entered.load(Ordering::SeqCst) == 1
        });
        std::fs::remove_file(folder.join("meta.json")).unwrap();
        std::fs::create_dir(folder.join("meta.json")).unwrap();
        model.release();
        let failed = server.done(&id);
        assert_eq!(failed["status"], "error");
        assert!(failed["persistence_error"].is_string());
        if operation == "retry" {
            assert_eq!(failed["items"][0]["options"]["profile"], "okf");
            assert!(failed["items"][0]["options"]["ocr"].is_null());
        }
        assert!(folder.join("out/a.txt.llm.md").exists());
        server.signal();
        server.finish(false);
        std::fs::remove_dir(folder.join("meta.json")).unwrap();
        std::fs::write(folder.join("meta.json"), original).unwrap();
        let server = Server::start(temp.path());
        let restored = server.json(&format!("/api/jobs/{id}"));
        assert_eq!(
            restored["items"][0]["options"],
            initial["items"][0]["options"]
        );
        assert_eq!(restored["items"][0]["options"]["ocr"], false);
        assert_eq!(result(&server, &id, "i1"), previous);
        assert_eq!(std::fs::read(folder.join("out/a.txt.md")).unwrap(), body);
        assert!(!folder.join("out/a.txt.llm.md").exists());
        server.stop();
    }
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

#[test]
fn failed_retry_restores_only_retained_outputs_options_and_preserves_siblings_after_restart() {
    let temp = tempfile::tempdir().unwrap();
    let server = Server::start(temp.path());
    let created = server.submit(
        &[
            ("saved.json", br#"{"body":"Retained source"}"#),
            ("sibling.txt", b"Sibling source"),
            ("failed.json", b"{invalid"),
        ],
        json!([]),
        json!({"profile":"obsidian","llm":false,"ocr":false,"backend":"native"}),
    );
    let id = created["job_id"].as_str().unwrap();
    let initial = server.done(id);
    assert_eq!(initial["items"][0]["status"], "done");
    assert_eq!(initial["items"][2]["status"], "error");
    let retained = result(&server, id, "i1");
    assert_eq!(
        retry(
            &server,
            id,
            "i2",
            Some(json!({"options":{"profile":"rag","llm":false,"ocr":false}}))
        )
        .status,
        202
    );
    let sibling = server.done(id)["items"][1].clone();
    std::fs::write(server.jobdir(id).join("uploads/saved.json"), b"{invalid").unwrap();
    assert_eq!(
        retry(
            &server,
            id,
            "i1",
            Some(json!({"options":{"profile":"okf","llm":false}}))
        )
        .status,
        202
    );
    let failed = server.done(id);
    assert_eq!(failed["items"][0]["rerun_failure"]["operation"], "retry");
    assert_eq!(
        failed["items"][0]["options"],
        initial["items"][0]["options"]
    );
    assert_eq!(failed["items"][0]["options"]["ocr"], false);
    assert_eq!(failed["items"][1], sibling);
    assert_eq!(result(&server, id, "i1"), retained);
    // With no successful prior output, failure retains the new repeat selection.
    assert_eq!(
        retry(
            &server,
            id,
            "i3",
            Some(json!({"options":{"profile":"okf","llm":false}}))
        )
        .status,
        202
    );
    let done = server.done(id);
    assert_eq!(done["items"][2]["status"], "error");
    assert_eq!(done["items"][2]["options"]["profile"], "okf");
    assert!(done["items"][2]["options"]["ocr"].is_null());
    assert_eq!(done["items"][0]["options"], initial["items"][0]["options"]);
    assert_eq!(done["items"][1], sibling);
    server.stop();
    let server = Server::start(temp.path());
    let restored = server.json(&format!("/api/jobs/{id}"));
    for index in 0..3 {
        assert_eq!(
            restored["items"][index]["options"],
            done["items"][index]["options"]
        );
    }
    assert_eq!(result(&server, id, "i1"), retained);
    server.stop();
}
