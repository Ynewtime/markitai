//! Requests the service refuses early, in its own words: model processing without
//! a model, malformed bodies and options, and bulk uploads.
use super::*;

fn post_job(server: &Server, files: &[(&str, &[u8])], options: Value) -> Reply {
    let (content, body) = multipart(files, json!([]), options);
    server.request("POST", "/api/jobs", &[("Content-Type", &content)], &body)
}
fn retry(server: &Server, id: &str, body: Value) -> Reply {
    server.request(
        "POST",
        &format!("/api/jobs/{id}/items/i1/retry"),
        &[("Content-Type", "application/json")],
        body.to_string().as_bytes(),
    )
}
fn stage_entries(temp: &Path) -> Vec<String> {
    std::fs::read_dir(temp.join("home/serve/jobs"))
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        // The lifetime and publication locks are permanent; all other entries are jobs or stages.
        .filter(|name| name != ".publish.lock" && name != ".serve.lock")
        .collect()
}

#[test]
fn a_request_for_model_processing_is_refused_before_a_job_exists_while_no_model_is_routable() {
    let temp = tempfile::tempdir().unwrap();
    let server = Server::start(temp.path());
    let capabilities = server.json("/api/capabilities");
    assert_eq!(capabilities["llm"]["routable"], false, "{capabilities}");
    for options in [
        json!({"llm":true}),
        json!({"preset":"standard"}),
        json!({"preset":"RICH"}),
        json!({"preset":"minimal","llm":true}),
    ] {
        let reply = post_job(
            &server,
            &[("a.txt", b"x"), ("b.txt", b"y")],
            options.clone(),
        );
        assert_eq!(reply.status, 422, "{options}: {}", reply.text());
        let value = reply.json();
        assert_eq!(value["reason"], "llm_unavailable", "{options}");
        assert_eq!(value["code"], "invalid_request");
        let detail = value["detail"].as_str().unwrap();
        assert!(
            detail.contains("model") && !detail.contains("line 1"),
            "{detail}"
        );
    }
    assert!(
        stage_entries(temp.path()).is_empty(),
        "one refusal, not one failing item per file: {:?}",
        stage_entries(temp.path())
    );
    // Everything that does not need a model is accepted and converts.
    for options in [
        json!({}),
        json!({"llm":false}),
        json!({"preset":"minimal"}),
        json!({"preset":"standard","llm":false}),
        json!({"alt":true,"desc":true}),
    ] {
        let created = server.submit(&[("ok.txt", b"plain text")], json!([]), options.clone());
        let done = server.done(created["job_id"].as_str().unwrap());
        assert_eq!(done["items"][0]["status"], "done", "{options}: {done}");
    }

    // A retry that names model processing is refused the same way and leaves the item alone.
    let created = server.submit(&[("again.txt", b"text")], json!([]), json!({}));
    let id = created["job_id"].as_str().unwrap();
    server.done(id);
    for body in [
        json!({"options":{"llm":true}}),
        json!({"operation":"retry","options":{"preset":"standard"}}),
    ] {
        let reply = retry(&server, id, body.clone());
        assert_eq!(reply.status, 422, "{body}: {}", reply.text());
        assert_eq!(reply.json()["reason"], "llm_unavailable");
    }
    let snapshot = server.json(&format!("/api/jobs/{id}"));
    assert_eq!(snapshot["items"][0]["status"], "done");
    assert_eq!(snapshot["items"][0]["operation"], "convert");
    // Opting out of the model is a valid retry; an explicit enhancement keeps its own refusal.
    assert_eq!(
        retry(&server, id, json!({"options":{"llm":false}})).status,
        202
    );
    server.done(id);
    let enhance = retry(
        &server,
        id,
        json!({"operation":"enhance","options":{"llm":true}}),
    );
    assert_eq!(
        (enhance.status, enhance.json()["reason"].clone()),
        (409, json!("llm_unavailable"))
    );
    server.stop();
}

#[test]
fn malformed_bodies_and_options_get_the_services_wording_and_a_stable_reason() {
    let temp = tempfile::tempdir().unwrap();
    let server = Server::start(temp.path());
    let leaks = |reply: &Reply| {
        let detail = reply.json()["detail"].as_str().unwrap().to_owned();
        for framework in ["boundary", "`", "line 1", "column", "expected one of"] {
            assert!(!detail.contains(framework), "{framework} leaked: {detail}");
        }
        detail
    };
    // Not a multipart body.
    for (content_type, body) in [
        ("application/json", r#"{"urls":["https://example.test"]}"#),
        ("multipart/form-data", "no boundary"),
    ] {
        let reply = server.request(
            "POST",
            "/api/jobs",
            &[("Content-Type", content_type)],
            body.as_bytes(),
        );
        assert_eq!(reply.status, 400, "{}", reply.text());
        assert_eq!(reply.json()["reason"], "invalid_multipart");
        assert!(leaks(&reply).contains("multipart/form-data"));
    }
    // No body at all is an empty job, not a protocol error.
    let reply = server.request("POST", "/api/jobs", &[], &[]);
    assert_eq!(
        (reply.status, reply.json()["reason"].clone()),
        (422, json!("empty_job"))
    );
    // Unknown and mistyped options are named.
    for (options, expected) in [
        (
            json!({"bogus":true}),
            "unknown option 'bogus'; supported options: preset",
        ),
        (json!({"llm":"yes"}), "option 'llm' must be true or false"),
        (
            json!({"profile":"pdf"}),
            "option 'profile' must be one of: rag, obsidian, okf",
        ),
        (
            json!({"strategy":"x"}),
            "option 'strategy' must be one of: auto",
        ),
    ] {
        let reply = post_job(&server, &[("a.txt", b"x")], options.clone());
        assert_eq!(reply.status, 422, "{options}: {}", reply.text());
        assert_eq!(reply.json()["reason"], "invalid_options");
        assert!(
            leaks(&reply).starts_with(expected),
            "{options}: {}",
            reply.text()
        );
    }
    // Retry bodies likewise.
    let created = server.submit(&[("a.txt", b"x")], json!([]), json!({}));
    let id = created["job_id"].as_str().unwrap();
    server.done(id);
    for (body, reason, expected) in [
        (
            json!({"bogus":1}),
            "invalid_retry_body",
            "unknown field 'bogus' in the retry body",
        ),
        (
            json!({"operation":"fly"}),
            "invalid_retry_body",
            "operation must be 'retry' or 'enhance'",
        ),
        (
            json!({"options":{"nope":1}}),
            "invalid_options",
            "unknown option 'nope'",
        ),
    ] {
        let reply = retry(&server, id, body.clone());
        assert_eq!(reply.status, 422, "{body}: {}", reply.text());
        assert_eq!(reply.json()["reason"], reason, "{body}");
        assert!(
            leaks(&reply).starts_with(expected),
            "{body}: {}",
            reply.text()
        );
    }
    assert!(
        stage_entries(temp.path()).iter().all(|name| name == id),
        "no stage directories were left"
    );
    server.stop();
}

#[test]
fn a_job_of_many_small_uploads_is_created_whole_and_keeps_every_upload_for_retry() {
    let temp = tempfile::tempdir().unwrap();
    let server = Server::start(temp.path());
    let names = (0..250)
        .map(|index| format!("note-{index:03}.txt"))
        .collect::<Vec<_>>();
    let files = names
        .iter()
        .map(|name| (name.as_str(), name.as_bytes()))
        .collect::<Vec<_>>();
    let started = Instant::now();
    let created = server.submit(&files, json!([]), json!({}));
    let creation = started.elapsed();
    let id = created["job_id"].as_str().unwrap();
    assert_eq!(created["items"].as_array().unwrap().len(), 250);
    let done = server.done(id);
    assert_eq!(
        (done["done"].as_u64(), done["failed"].as_u64()),
        (Some(250), Some(0))
    );
    // Every upload is retained, byte for byte, in the published job directory.
    for name in &names {
        assert_eq!(
            std::fs::read(server.jobdir(id).join("uploads").join(name)).unwrap(),
            name.as_bytes(),
            "{name}"
        );
    }
    assert!(stage_entries(temp.path()).iter().all(|entry| entry == id));
    // The retained upload is what a retry converts again.
    assert_eq!(retry(&server, id, json!({})).status, 202);
    println!("created 250 uploads in {creation:?}");
    server.stop();
}

#[test]
fn cloudflare_request_consent_is_fresh_trusted_and_persisted_only_as_disclosure() {
    let temp = tempfile::tempdir().unwrap();
    configure(temp.path());
    let config_path = temp.path().join("config.json");
    let mut cfg: Value = serde_json::from_slice(&std::fs::read(&config_path).unwrap()).unwrap();
    cfg["fetch"]["remote_consent"] = json!("ask");
    cfg["fetch"]["cloudflare"] = json!({"api_token":"synthetic-token","account_id":"synthetic-account","convert_enabled":true});
    std::fs::write(&config_path, cfg.to_string()).unwrap();
    let server = Server::start(temp.path());
    let caps = server.json("/api/capabilities");
    assert_eq!(caps["remote_services"]["cloudflare"]["available"], true);
    assert!(!caps.to_string().contains("synthetic-"));
    let forwarded = server.request(
        "GET",
        "/api/capabilities",
        &[("X-Forwarded-For", "192.0.2.1")],
        &[],
    );
    assert_eq!(
        forwarded.json()["remote_services"]["cloudflare"]["reason"],
        "client_not_trusted"
    );
    let selection = json!({"backend":"cloudflare"});
    let refused = post_job(&server, &[("no.txt", b"local")], selection.clone());
    assert_eq!(
        refused.json()["reason"],
        "remote_processing_confirmation_required"
    );
    assert!(stage_entries(temp.path()).is_empty());
    let confirmed = json!({"backend":"cloudflare","remote_processing":"cloudflare"});
    let (content, body) = multipart(&[("no.txt", b"local")], json!([]), confirmed.clone());
    let untrusted = server.request(
        "POST",
        "/api/jobs",
        &[("Content-Type", &content), ("X-Forwarded-For", "192.0.2.1")],
        &body,
    );
    assert_eq!(untrusted.status, 403);
    assert_eq!(untrusted.json()["reason"], "remote_processing_forbidden");
    assert!(stage_entries(temp.path()).is_empty());
    // TXT is outside FORMATS, so accepted selection is provably not execution.
    let created = server.submit(
        &[("local.txt", b"local text")],
        json!([]),
        confirmed.clone(),
    );
    let id = created["job_id"].as_str().unwrap();
    let done = server.done(id);
    assert_eq!(done["items"][0]["status"], "done");
    assert_eq!(
        done["items"][0]["remote_processing"]["execution"],
        "unknown"
    );
    assert!(done["options"]["remote_processing"].is_null());
    let before =
        std::fs::read(temp.path().join(format!("home/serve/jobs/{id}/meta.json"))).unwrap();
    for operation in ["retry", "enhance"] {
        let reply = retry(&server, id, json!({"operation":operation}));
        assert_eq!(reply.status, 422);
        assert_eq!(
            reply.json()["reason"],
            "remote_processing_confirmation_required"
        );
    }
    assert_eq!(
        std::fs::read(temp.path().join(format!("home/serve/jobs/{id}/meta.json"))).unwrap(),
        before
    );
    assert_eq!(retry(&server, id, json!({"options":confirmed})).status, 202);
    server.done(id);
    // A local retry does not erase the scope of earlier accepted attempts.
    assert_eq!(
        retry(&server, id, json!({"options":{"backend":"native"}})).status,
        202
    );
    let done = server.done(id);
    assert_eq!(done["items"][0]["remote_processing"]["requested"], true);
    server.stop();
    let restarted = Server::start(temp.path());
    let restored = restarted.json(&format!("/api/jobs/{id}"));
    assert_eq!(
        restored["items"][0]["remote_processing"]["external_charges"],
        "not_included"
    );
    assert_eq!(
        restarted.json("/api/history")[0]["remote_processing"]["requested"],
        true
    );
    restarted.stop();
}

#[test]
fn cloudflare_never_policy_is_refused_before_job_publication() {
    let temp = tempfile::tempdir().unwrap();
    let server = Server::start(temp.path());
    assert_eq!(
        server.json("/api/capabilities")["remote_services"]["cloudflare"]["reason"],
        "disabled_by_policy"
    );
    let refused = post_job(
        &server,
        &[("no.txt", b"local")],
        json!({"backend":"cloudflare","remote_processing":"cloudflare"}),
    );
    assert_eq!(refused.status, 422);
    assert_eq!(refused.json()["reason"], "remote_processing_disabled");
    assert!(stage_entries(temp.path()).is_empty());
    server.stop();
}

fn configure_cloudflare_test(root: &Path) {
    configure(root);
    let path = root.join("config.json");
    let mut cfg: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    cfg["fetch"]["remote_consent"] = json!("ask");
    cfg["fetch"]["cloudflare"] =
        json!({"api_token":"synthetic-token","account_id":"synthetic-account"});
    std::fs::write(path, cfg.to_string()).unwrap();
}

#[test]
fn cloudflare_item_options_survive_sibling_route_changes_restart_and_fresh_consent() {
    let temp = tempfile::tempdir().unwrap();
    configure_cloudflare_test(temp.path());
    let server = Server::start(temp.path());
    // Neither TXT input is a supported Cloudflare format: no remote calls.
    let created = server.submit(
        &[("a.txt", b"First"), ("b.txt", b"Second")],
        json!([]),
        json!({"backend":"native"}),
    );
    let id = created["job_id"].as_str().unwrap();
    let initial = server.done(id);
    assert_eq!(initial["items"][0]["options"]["backend"], "native");
    assert_eq!(
        retry(
            &server,
            id,
            json!({"options":{"backend":"cloudflare","remote_processing":"cloudflare"}})
        )
        .status,
        202
    );
    server.done(id);
    let reply = server.request(
        "POST",
        &format!("/api/jobs/{id}/items/i2/retry"),
        &[("Content-Type", "application/json")],
        br#"{"options":{"backend":"native"}}"#,
    );
    assert_eq!(reply.status, 202);
    let done = server.done(id);
    assert_eq!(done["options"]["backend"], "native");
    assert_eq!(done["items"][0]["options"]["backend"], "cloudflare");
    assert_eq!(done["items"][1]["options"]["backend"], "native");
    for item in done["items"].as_array().unwrap() {
        assert!(item["options"].get("remote_processing").is_none());
    }
    server.stop();
    let server = Server::start(temp.path());
    let restored = server.json(&format!("/api/jobs/{id}"));
    assert_eq!(restored["items"][0]["options"], done["items"][0]["options"]);
    assert_eq!(restored["items"][1]["options"], done["items"][1]["options"]);
    for operation in ["retry", "enhance"] {
        let response = retry(
            &server,
            id,
            json!({"operation":operation,"options":restored["items"][0]["options"]}),
        );
        assert_eq!(response.status, 422);
        assert_eq!(
            response.json()["reason"],
            "remote_processing_confirmation_required"
        );
    }
    let response = server.request(
        "POST",
        &format!("/api/jobs/{id}/items/i2/retry"),
        &[("Content-Type", "application/json")],
        json!({"options":restored["items"][1]["options"]})
            .to_string()
            .as_bytes(),
    );
    assert_eq!(response.status, 202);
    assert_eq!(
        server.done(id)["items"][0]["options"]["backend"],
        "cloudflare"
    );
    server.stop();
}

#[test]
fn cloudflare_item_options_are_present_on_live_sse_items_without_consent() {
    let origin = Origin::start();
    let temp = tempfile::tempdir().unwrap();
    configure_cloudflare_test(temp.path());
    let server = Server::start(temp.path());
    // File backend does not send URLs to Cloudflare; this holds only a loopback static URL.
    let created = server.submit(
        &[],
        json!([origin.url("/hold")]),
        json!({"strategy":"static","backend":"cloudflare","remote_processing":"cloudflare"}),
    );
    let id = created["job_id"].as_str().unwrap().to_owned();
    until("held loopback URL", || {
        origin.entered.load(Ordering::SeqCst) == 1
    });
    let port = server.port;
    let path = format!("/api/jobs/{id}/events");
    let (ready, subscribed) = std::sync::mpsc::channel();
    let events =
        std::thread::spawn(move || request_with_ready(port, "GET", &path, &[], &[], Some(ready)));
    subscribed.recv_timeout(WAIT).unwrap();
    origin.release();
    let reply = events.join().unwrap();
    assert_eq!(reply.status, 200);
    let mut snapshots = 0;
    let mut items = 0;
    for block in reply.text().split("\n\n") {
        let kind = block.lines().find_map(|line| {
            line.strip_prefix("event: ")
                .or_else(|| line.strip_prefix("event:"))
        });
        let payload = block.lines().find_map(|line| {
            line.strip_prefix("data: ")
                .or_else(|| line.strip_prefix("data:"))
        });
        let (Some(kind), Some(payload)) = (kind, payload) else {
            continue;
        };
        let value: Value = serde_json::from_str(payload).unwrap();
        let item = match kind.trim() {
            "snapshot" => {
                snapshots += 1;
                &value["items"][0]
            }
            "item" => {
                items += 1;
                &value
            }
            _ => continue,
        };
        assert_eq!(item["options"]["backend"], "cloudflare");
        assert_eq!(item["options"]["strategy"], "static");
        assert!(item["options"].get("remote_processing").is_none());
    }
    assert!(snapshots > 0, "initial snapshot missing");
    assert!(items > 0, "live item event missing");
    server.done(&id);
    server.stop();
}
