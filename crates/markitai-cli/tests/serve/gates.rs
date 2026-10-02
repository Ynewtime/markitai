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
        // The publication lock is permanent; everything else is a job or a leftover stage.
        .filter(|name| name != ".publish.lock")
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
