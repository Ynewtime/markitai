//! Access levels: the server token is the admin's; user tokens an admin
//! creates reach everything but the settings and the token routes, and see
//! only their own jobs. Also the batch deletion and the Edit config routes.
use super::*;

const ADMIN: &str = "admin-test-token-0001";

fn start(root: &Path) -> Server {
    // A user token's URL job reaches public addresses only; this debug build
    // lets the tests treat the loopback origin as public so a held job can
    // reach it. a_user_url_job_cannot_reach_this_machine covers the default.
    Server::start_with(
        root,
        &[
            ("MARKITAI_SERVE_TOKEN", std::ffi::OsStr::new(ADMIN)),
            (
                "MARKITAI_TEST_PUBLIC_NET_ALLOW",
                std::ffi::OsStr::new("127.0.0.1"),
            ),
        ],
    )
}
fn bearer(token: &str) -> String {
    format!("Bearer {token}")
}
fn call(server: &Server, method: &str, path: &str, token: &str, body: Option<Value>) -> Reply {
    let authorization = bearer(token);
    let mut headers = vec![("Authorization", authorization.as_str())];
    let body = body.map(|value| value.to_string()).unwrap_or_default();
    if !body.is_empty() {
        headers.push(("Content-Type", "application/json"));
    }
    server.request(method, path, &headers, body.as_bytes())
}
fn submit_as(server: &Server, token: &str, files: &[(&str, &[u8])], urls: Value) -> String {
    let (content_type, body) = multipart(files, urls, json!({"strategy":"static"}));
    let authorization = bearer(token);
    let reply = server.request(
        "POST",
        "/api/jobs",
        &[
            ("Content-Type", content_type.as_str()),
            ("Authorization", authorization.as_str()),
        ],
        &body,
    );
    assert_eq!(reply.status, 201, "{}", reply.text());
    reply.json()["job_id"].as_str().unwrap().to_owned()
}
fn done_as(server: &Server, token: &str, id: &str) -> Value {
    let mut value = Value::Null;
    until("job completion", || {
        let reply = call(server, "GET", &format!("/api/jobs/{id}"), token, None);
        assert_eq!(reply.status, 200, "{}", reply.text());
        value = reply.json();
        value["status"] != "running"
    });
    value
}
fn reason(reply: &Reply) -> (u16, Value) {
    (reply.status, reply.json()["reason"].clone())
}
fn sha256(text: &str) -> String {
    use sha2::Digest;
    sha2::Sha256::digest(text.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
fn history_ids(server: &Server, token: &str, query: &str) -> Vec<String> {
    let reply = call(server, "GET", &format!("/api/history{query}"), token, None);
    assert_eq!(reply.status, 200, "{}", reply.text());
    let mut ids: Vec<String> = reply
        .json()
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| entry["job_id"].as_str().unwrap().to_owned())
        .collect();
    ids.sort();
    ids
}
/// Opens a job's event stream; returns it once the first snapshot arrived.
fn subscribe(port: u16, id: &str, token: &str) -> TcpStream {
    let mut socket = TcpStream::connect(("127.0.0.1", port)).unwrap();
    socket.set_read_timeout(Some(WAIT)).unwrap();
    write!(
        socket,
        "GET /api/jobs/{id}/events HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nAuthorization: Bearer {token}\r\nConnection: close\r\n\r\n"
    )
    .unwrap();
    let mut seen = Vec::new();
    let mut block = [0; 1024];
    while !String::from_utf8_lossy(&seen).contains("event: snapshot") {
        let count = socket.read(&mut block).unwrap();
        assert!(count > 0, "{}", String::from_utf8_lossy(&seen));
        seen.extend_from_slice(&block[..count]);
    }
    assert!(seen.starts_with(b"HTTP/1.1 200"));
    socket
}

#[test]
fn a_user_token_is_shown_once_kept_as_a_digest_and_sees_only_its_own_jobs() {
    let origin = Origin::start();
    let temp = tempfile::tempdir().unwrap();
    let server = start(temp.path());
    let capabilities = call(&server, "GET", "/api/capabilities", ADMIN, None).json();
    assert_eq!(capabilities["access"], json!({"role":"admin"}));
    for feature in [
        "user_tokens",
        "config_edit",
        "batch_delete",
        "token_regenerate",
    ] {
        assert_eq!(capabilities["features"][feature], true, "{feature}");
    }

    // Creating: the label is checked, the token shown in this answer only.
    for (body, status, expected) in [
        (json!({"label":"  "}), 422, "invalid_label"),
        (json!({"label":"a".repeat(61)}), 422, "invalid_label"),
        (json!({"label":"tab\there"}), 422, "invalid_label"),
        (json!({"label":"x","extra":1}), 422, "invalid_json"),
        (json!({"name":"x"}), 422, "invalid_json"),
        (json!(["x"]), 422, "invalid_json"),
        (json!({"label":"x".repeat(2000)}), 413, "request_too_large"),
    ] {
        let reply = call(
            &server,
            "POST",
            "/api/access-tokens",
            ADMIN,
            Some(body.clone()),
        );
        assert_eq!(reason(&reply), (status, json!(expected)), "{body}");
    }
    let reply = call(
        &server,
        "POST",
        "/api/access-tokens",
        ADMIN,
        Some(json!({"label":"  Team laptop "})),
    );
    assert_eq!(reply.status, 201, "{}", reply.text());
    assert_eq!(reply.headers["cache-control"], "no-store");
    let created = reply.json();
    let user = created["token"].as_str().unwrap().to_owned();
    let user_id = created["id"].as_str().unwrap().to_owned();
    assert!(user.starts_with("mku_") && user.len() == 68, "{user}");
    assert!(
        user[4..]
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    );
    assert!(
        user_id.starts_with("u_") && user_id.len() == 14,
        "{user_id}"
    );
    assert_eq!(
        created,
        json!({"id":user_id,"label":"Team laptop","hint":&user[64..],
            "created_at":created["created_at"],"last_used_at":null,"token":user})
    );
    let listed = call(&server, "GET", "/api/access-tokens", ADMIN, None);
    assert_eq!(listed.headers["cache-control"], "no-store");
    assert!(!listed.text().contains(&user));
    assert_eq!(listed.json()["tokens"][0]["id"], json!(user_id));
    assert!(listed.json()["tokens"][0].get("token").is_none());

    // Stored as a digest only, owner-only.
    let file = temp.path().join("home/serve/access.json");
    let saved = std::fs::read_to_string(&file).unwrap();
    assert!(!saved.contains(&user) && !saved.contains(&user[4..]));
    assert!(saved.contains(&sha256(&user)));
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(&file).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600);
    }

    let capabilities = call(&server, "GET", "/api/capabilities", &user, None).json();
    assert_eq!(
        capabilities["access"],
        json!({"role":"user","label":"Team laptop"})
    );
    assert!(
        call(&server, "GET", "/api/access-tokens", ADMIN, None).json()["tokens"][0]["last_used_at"]
            .is_string()
    );

    // Every settings route, regeneration and the token routes are the admin's.
    for (method, path, body) in [
        ("GET", "/api/settings/llm", None),
        ("GET", "/api/settings/llm/providers", None),
        ("GET", "/api/settings/llm/detected", None),
        (
            "POST",
            "/api/settings/llm/models",
            Some(json!({"model_name":"x","model":"openai/x"})),
        ),
        (
            "POST",
            "/api/settings/llm/test",
            Some(json!({"model":"openai/x"})),
        ),
        (
            "POST",
            "/api/settings/llm/model-discovery",
            Some(json!({"provider":"openai"})),
        ),
        ("POST", "/api/settings/llm/config/open", None),
        ("GET", "/api/settings/config", None),
        (
            "PUT",
            "/api/settings/config",
            Some(json!({"expected_revision":"absent","text":"{}"})),
        ),
        ("POST", "/api/access-token/regenerate", None),
        ("GET", "/api/access-tokens", None),
        ("POST", "/api/access-tokens", Some(json!({"label":"mine"}))),
        ("DELETE", &format!("/api/access-tokens/{user_id}"), None),
    ] {
        let reply = call(&server, method, path, &user, body);
        assert_eq!(
            reason(&reply),
            (403, json!("settings_forbidden")),
            "{method} {path}"
        );
        assert_eq!(
            reply.headers["cache-control"], "no-store",
            "{method} {path}"
        );
    }

    // Jobs: the admin's and the user's; a user's URL job is allowed.
    let theirs = submit_as(&server, ADMIN, &[("admin-notes.txt", b"admin")], json!([]));
    let mine = submit_as(&server, &user, &[("user-notes.txt", b"user")], json!([]));
    let output = done_as(&server, ADMIN, &theirs)["items"][0]["output"]
        .as_str()
        .unwrap()
        .to_owned();
    done_as(&server, &user, &mine);
    let meta: Value =
        serde_json::from_slice(&std::fs::read(server.jobdir(&mine).join("meta.json")).unwrap())
            .unwrap();
    assert_eq!(meta["owner"], json!(user_id));
    let meta: Value =
        serde_json::from_slice(&std::fs::read(server.jobdir(&theirs).join("meta.json")).unwrap())
            .unwrap();
    assert!(meta.get("owner").is_none());
    let snapshot = call(&server, "GET", &format!("/api/jobs/{mine}"), &user, None);
    assert!(!snapshot.text().contains("owner") && !snapshot.text().contains(&user_id));
    let mut both = vec![theirs.clone(), mine.clone()];
    both.sort();
    assert_eq!(history_ids(&server, ADMIN, ""), both);
    assert_eq!(history_ids(&server, &user, ""), std::slice::from_ref(&mine));
    assert_eq!(
        history_ids(&server, &user, "?include=items"),
        std::slice::from_ref(&mine)
    );
    for path in [
        format!("/api/jobs/{theirs}"),
        format!("/api/jobs/{theirs}/events"),
        format!("/api/jobs/{theirs}/archive"),
        format!("/api/jobs/{theirs}/files/{output}"),
        format!("/api/jobs/{theirs}/items/i1/result"),
        format!("/api/jobs/{theirs}/items/i1/original"),
        format!("/api/jobs/{theirs}/items/i1/archive"),
    ] {
        let reply = call(&server, "GET", &path, &user, None);
        assert_eq!(reason(&reply), (404, json!("job_not_found")), "{path}");
        assert_eq!(
            call(&server, "GET", &path, ADMIN, None).status,
            200,
            "{path}"
        );
    }
    // Their item's block boxes are as hidden; the admin is told none are kept.
    let blocks = format!("/api/jobs/{theirs}/items/i1/original/blocks");
    assert_eq!(
        reason(&call(&server, "GET", &blocks, &user, None)),
        (404, json!("job_not_found"))
    );
    assert_eq!(
        reason(&call(&server, "GET", &blocks, ADMIN, None)),
        (404, json!("original_unavailable"))
    );
    for (method, path) in [
        ("POST", format!("/api/jobs/{theirs}/cancel")),
        ("POST", format!("/api/jobs/{theirs}/items/i1/retry")),
        ("DELETE", format!("/api/jobs/{theirs}/items/i1")),
        ("DELETE", format!("/api/history/{theirs}")),
    ] {
        let reply = call(&server, method, &path, &user, None);
        assert_eq!(
            reason(&reply),
            (404, json!("job_not_found")),
            "{method} {path}"
        );
    }
    let batch = call(
        &server,
        "POST",
        "/api/history/delete",
        &user,
        Some(json!({"jobs":[theirs],"items":[{"job_id":theirs,"item_id":"i1"}]})),
    );
    assert_eq!(batch.status, 200, "{}", batch.text());
    assert_eq!(
        batch.json(),
        json!({"deleted":0,"failed":2,"results":[
            {"job_id":theirs,"item_id":null,"ok":false,"reason":"job_not_found","detail":"job not found"},
            {"job_id":theirs,"item_id":"i1","ok":false,"reason":"job_not_found","detail":"job not found"}]})
    );
    assert!(server.jobdir(&theirs).exists());
    // The history archive holds the caller's jobs only.
    let archive = call(&server, "GET", "/api/history/archive", &user, None);
    assert_eq!(archive.status, 200);
    let names: Vec<String> = zip_contents(&archive.body).into_keys().collect();
    assert!(
        names.iter().all(|name| name.contains("user-notes")),
        "{names:?}"
    );
    assert!(!names.is_empty());
    let archive = call(&server, "GET", "/api/history/archive", ADMIN, None);
    let names: Vec<String> = zip_contents(&archive.body).into_keys().collect();
    assert!(
        names.iter().any(|name| name.contains("admin-notes")),
        "{names:?}"
    );
    // Tickets: only for the caller's own jobs, and they act as that caller.
    let ticket = |path: String| {
        call(
            &server,
            "POST",
            "/api/download-tickets",
            &user,
            Some(json!({"path":path})),
        )
    };
    assert_eq!(
        reason(&ticket(format!("/api/jobs/{theirs}/archive"))),
        (404, json!("job_not_found"))
    );
    let issued = ticket("/api/history/archive".into());
    assert_eq!(issued.status, 201, "{}", issued.text());
    let opened = server.request("GET", issued.json()["url"].as_str().unwrap(), &[], &[]);
    assert_eq!(opened.status, 200);
    let names: Vec<String> = zip_contents(&opened.body).into_keys().collect();
    assert!(
        names.iter().all(|name| name.contains("user-notes")),
        "{names:?}"
    );
    let issued = ticket(format!("/api/jobs/{mine}/archive"));
    assert_eq!(issued.status, 201, "{}", issued.text());
    let outstanding = issued.json()["url"].as_str().unwrap().to_owned();

    // A held URL job of the user's keeps its event stream open.
    let held = submit_as(&server, &user, &[], json!([origin.url("/hold")]));
    until("held URL reaches origin", || {
        origin.entered.load(Ordering::SeqCst) >= 1
    });
    let mut stream = subscribe(server.port, &held, &user);

    // Regenerating the admin token keeps user tokens.
    let regenerated = call(&server, "POST", "/api/access-token/regenerate", ADMIN, None);
    assert_eq!(regenerated.status, 200, "{}", regenerated.text());
    let admin = regenerated.json()["token"].as_str().unwrap().to_owned();
    let mut rest = Vec::new();
    stream.read_to_end(&mut rest).unwrap();
    assert_eq!(
        call(&server, "GET", "/api/capabilities", &user, None).status,
        200
    );
    let mut stream = subscribe(server.port, &held, &user);
    // Regeneration revoked every ticket; the user takes a new one.
    assert_eq!(
        reason(&server.request("GET", &outstanding, &[], &[])),
        (401, json!("ticket_invalid"))
    );
    let issued = ticket(format!("/api/jobs/{mine}/archive"));
    assert_eq!(issued.status, 201, "{}", issued.text());
    let outstanding = issued.json()["url"].as_str().unwrap().to_owned();

    // Revoking: the next request, its ticket and its open stream all stop.
    let revoked = call(
        &server,
        "DELETE",
        &format!("/api/access-tokens/{user_id}"),
        &admin,
        None,
    );
    assert_eq!(revoked.status, 204, "{}", revoked.text());
    assert_eq!(revoked.headers["cache-control"], "no-store");
    let mut rest = Vec::new();
    stream.read_to_end(&mut rest).unwrap();
    assert_eq!(
        reason(&call(&server, "GET", "/api/capabilities", &user, None)),
        (401, json!("token_required"))
    );
    assert_eq!(
        reason(&call(
            &server,
            "GET",
            &format!("/api/jobs/{held}/events"),
            &user,
            None
        )),
        (401, json!("token_required"))
    );
    assert_eq!(
        reason(&server.request("GET", &outstanding, &[], &[])),
        (401, json!("ticket_invalid"))
    );
    assert_eq!(
        reason(&call(
            &server,
            "DELETE",
            &format!("/api/access-tokens/{user_id}"),
            &admin,
            None
        )),
        (404, json!("token_not_found"))
    );
    // The revoked user's jobs stay, for admins.
    assert_eq!(history_ids(&server, &admin, ""), both);
    origin.release();
    done_as(&server, &admin, &held);
    let log = server.captured.lock().unwrap().clone();
    assert!(
        log.contains("Access token \"Team laptop\" created from the web workbench."),
        "{log}"
    );
    assert!(
        log.contains("Access token \"Team laptop\" revoked from the web workbench."),
        "{log}"
    );
    assert!(!log.contains(&user[4..]));
    server.stop();
}

#[test]
fn user_tokens_survive_a_restart_and_the_limit_and_revocation_hold() {
    let temp = tempfile::tempdir().unwrap();
    let server = start(temp.path());
    let created = call(
        &server,
        "POST",
        "/api/access-tokens",
        ADMIN,
        Some(json!({"label":"Desk"})),
    );
    assert_eq!(created.status, 201);
    let token = created.json()["token"].as_str().unwrap().to_owned();
    for index in 1..50 {
        let reply = call(
            &server,
            "POST",
            "/api/access-tokens",
            ADMIN,
            Some(json!({"label":format!("Token {index}")})),
        );
        assert_eq!(reply.status, 201, "{}", reply.text());
    }
    assert_eq!(
        reason(&call(
            &server,
            "POST",
            "/api/access-tokens",
            ADMIN,
            Some(json!({"label":"One more"}))
        )),
        (409, json!("token_limit"))
    );
    assert_eq!(
        call(&server, "GET", "/api/capabilities", &token, None).status,
        200
    );
    server.stop();

    let server = start(temp.path());
    let listed = call(&server, "GET", "/api/access-tokens", ADMIN, None).json();
    let tokens = listed["tokens"].as_array().unwrap();
    assert_eq!(tokens.len(), 50);
    // Oldest first; last use is not kept across a restart.
    assert_eq!(tokens[0]["label"], "Desk");
    assert!(tokens.iter().all(|entry| entry["last_used_at"].is_null()));
    let capabilities = call(&server, "GET", "/api/capabilities", &token, None);
    assert_eq!(
        capabilities.json()["access"],
        json!({"role":"user","label":"Desk"})
    );
    server.stop();
}

#[test]
fn without_a_token_there_are_no_user_tokens() {
    let temp = tempfile::tempdir().unwrap();
    let server = Server::start(temp.path());
    let capabilities = server.json("/api/capabilities");
    assert_eq!(capabilities["access"], json!({"role":"admin"}));
    assert_eq!(capabilities["features"]["user_tokens"], false);
    for (method, path, body) in [
        ("GET", "/api/access-tokens", &b""[..]),
        ("POST", "/api/access-tokens", &br#"{"label":"x"}"#[..]),
        ("DELETE", "/api/access-tokens/u_000000000000", &b""[..]),
    ] {
        let reply = server.request(method, path, &[("Content-Type", "application/json")], body);
        assert_eq!(
            reason(&reply),
            (409, json!("token_disabled")),
            "{method} {path}"
        );
        assert_eq!(reply.headers["cache-control"], "no-store");
        // A client that is not trusted is refused as the settings refuse it.
        let reply = server.request(method, path, &[("X-Forwarded-For", "203.0.113.9")], body);
        assert_eq!(
            reason(&reply),
            (403, json!("settings_forbidden")),
            "{method} {path}"
        );
    }
    let remote = server.request(
        "GET",
        "/api/capabilities",
        &[("X-Forwarded-For", "203.0.113.9")],
        &[],
    );
    assert_eq!(remote.json()["access"], json!({"role":"user"}));
    let reply = server.request(
        "GET",
        "/api/settings/config",
        &[("X-Forwarded-For", "203.0.113.9")],
        &[],
    );
    assert_eq!(reason(&reply), (403, json!("settings_forbidden")));
    assert!(!temp.path().join("home/serve/access.json").exists());
    server.stop();
}

#[test]
fn a_batch_deletion_reports_every_entry_in_order() {
    let origin = Origin::start();
    let temp = tempfile::tempdir().unwrap();
    let server = Server::start(temp.path());
    let two = server.submit(&[("a.txt", b"a"), ("b.txt", b"b")], json!([]), json!({}))["job_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let single = server.submit(&[("c.txt", b"c")], json!([]), json!({}))["job_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let whole = server.submit(&[("d.txt", b"d")], json!([]), json!({}))["job_id"]
        .as_str()
        .unwrap()
        .to_owned();
    for id in [&two, &single, &whole] {
        server.done(id);
    }
    let held = server.submit(
        &[],
        json!([origin.url("/hold")]),
        json!({"strategy":"static"}),
    )["job_id"]
        .as_str()
        .unwrap()
        .to_owned();
    until("held URL reaches origin", || {
        origin.entered.load(Ordering::SeqCst) >= 1
    });
    let send = |content_type: &str, body: &[u8]| {
        server.request(
            "POST",
            "/api/history/delete",
            &[("Content-Type", content_type)],
            body,
        )
    };
    let json_body = |value: Value| send("application/json", value.to_string().as_bytes());
    // Refusals before anything is deleted.
    let many: Vec<String> = (0..1001).map(|index| format!("{index:012x}")).collect();
    for (body, expected) in [
        (json!({}), "invalid_delete_request"),
        (json!({"jobs":[],"items":[]}), "invalid_delete_request"),
        (json!({"jobs":"x"}), "invalid_delete_request"),
        (json!({"jobs":[1]}), "invalid_delete_request"),
        (json!({"jobs":["x"],"other":[]}), "invalid_delete_request"),
        (json!({"items":[{"job_id":"x"}]}), "invalid_delete_request"),
        (
            json!({"items":[{"job_id":"x","item_id":"i1","extra":1}]}),
            "invalid_delete_request",
        ),
        (json!({"jobs":many}), "invalid_delete_request"),
        (json!([]), "invalid_delete_request"),
    ] {
        let reply = json_body(body.clone());
        assert_eq!(reason(&reply), (422, json!(expected)), "{body}");
        assert!(reply.json()["detail"].is_string());
    }
    assert_eq!(
        reason(&send("text/plain", b"{}")),
        (422, json!("json_required"))
    );
    assert_eq!(
        reason(&send("application/json", b"{oops")),
        (422, json!("invalid_json"))
    );
    let large = format!("{{\"jobs\":[\"{}\"]}}", "a".repeat(300 * 1024));
    assert_eq!(
        reason(&send("application/json", large.as_bytes())),
        (413, json!("request_too_large"))
    );
    assert!(server.jobdir(&two).exists() && server.jobdir(&whole).exists());

    let reply = json_body(json!({
        "items":[{"job_id":two,"item_id":"i1"},{"job_id":two,"item_id":"i9"},{"job_id":single,"item_id":"i1"}],
        "jobs":[whole,"000000000000",held],
    }));
    assert_eq!(reply.status, 200, "{}", reply.text());
    assert_eq!(reply.headers["cache-control"], "no-store");
    let answer = reply.json();
    let outcome: Vec<_> = answer["results"]
        .as_array()
        .unwrap()
        .iter()
        .map(|result| {
            (
                result["job_id"].as_str().unwrap().to_owned(),
                result["item_id"].clone(),
                result["ok"].clone(),
                result["reason"].clone(),
            )
        })
        .collect();
    assert_eq!(
        outcome,
        [
            (whole.clone(), Value::Null, json!(true), Value::Null),
            (
                "000000000000".into(),
                Value::Null,
                json!(false),
                json!("job_not_found")
            ),
            (
                held.clone(),
                Value::Null,
                json!(false),
                json!("job_running")
            ),
            (two.clone(), json!("i1"), json!(true), Value::Null),
            (
                two.clone(),
                json!("i9"),
                json!(false),
                json!("item_not_found")
            ),
            (single.clone(), json!("i1"), json!(true), Value::Null),
        ]
    );
    assert_eq!(
        (answer["deleted"].clone(), answer["failed"].clone()),
        (json!(3), json!(3))
    );
    assert_eq!(answer["results"][1]["detail"], "job not found");
    // The whole job and the single item's job are gone; one item remains.
    for id in [&whole, &single] {
        let reply = server.request("GET", &format!("/api/jobs/{id}"), &[], &[]);
        assert_eq!(reason(&reply), (404, json!("job_not_found")));
        assert!(!server.jobdir(id).exists());
    }
    let left = server.json(&format!("/api/jobs/{two}"));
    assert_eq!(left["items"].as_array().unwrap().len(), 1);
    assert_eq!(left["items"][0]["item_id"], "i2");
    origin.release();
    server.done(&held);
    server.stop();
}

#[test]
fn the_config_file_is_edited_over_http_with_secrets_masked() {
    let temp = tempfile::tempdir().unwrap();
    configure(temp.path());
    let mut config: Value =
        serde_json::from_slice(&std::fs::read(temp.path().join("config.json")).unwrap()).unwrap();
    config["llm"]["model_list"] = json!([{"model_name":"default","litellm_params":{
        "model":"openai/first","api_key":"sk-fixture-1234",
        "api_base":"https://user:pw@gateway.example/v1"}}]);
    config["fetch"]["playwright"] =
        json!({"extra_http_headers":{"X-Fixture":"fixture-header-secret"}});
    std::fs::write(temp.path().join("config.json"), config.to_string()).unwrap();
    let server = start(temp.path());
    let created = call(
        &server,
        "POST",
        "/api/access-tokens",
        ADMIN,
        Some(json!({"label":"Viewer"})),
    );
    let user = created.json()["token"].as_str().unwrap().to_owned();

    let reply = call(&server, "GET", "/api/settings/config", ADMIN, None);
    assert_eq!(reply.status, 200, "{}", reply.text());
    assert_eq!(reply.headers["cache-control"], "no-store");
    let view = reply.json();
    let text = view["text"].as_str().unwrap().to_owned();
    for secret in ["fixture-1234", "user:pw", "fixture-header-secret"] {
        assert!(
            !text.contains(secret) && !reply.text().contains(secret),
            "{secret}"
        );
    }
    assert_eq!(view["masked"], 3);
    assert_eq!(view["origin"], "explicit");
    assert_eq!(view["exists"], true);
    assert_eq!(view["read_only"], false);
    assert_eq!(view["problem"], Value::Null);
    let revision = view["revision"].as_str().unwrap().to_owned();
    let put =
        |token: &str, body: Value| call(&server, "PUT", "/api/settings/config", token, Some(body));
    assert_eq!(
        reason(&put(
            &user,
            json!({"expected_revision":revision,"text":text})
        )),
        (403, json!("settings_forbidden"))
    );
    // Shapes and sizes.
    for body in [
        json!({"text":text}),
        json!({"expected_revision":revision,"text":text,"x":1}),
        json!({"expected_revision":1,"text":text}),
        json!([revision, text]),
    ] {
        let reply = put(ADMIN, body);
        assert_eq!(reason(&reply), (422, json!("invalid_settings_request")));
        assert_eq!(reply.headers["cache-control"], "no-store");
    }
    let huge = json!({"expected_revision":revision,"text":" ".repeat(1024 * 1024 + 1)});
    assert_eq!(reason(&put(ADMIN, huge)), (413, json!("config_too_large")));
    // A stale revision names the current one.
    let stale = put(ADMIN, json!({"expected_revision":"sha256:0","text":text}));
    assert_eq!(stale.status, 409);
    assert_eq!(
        stale.json()["detail"],
        json!({"code":"stale_revision","current_revision":revision})
    );
    let broken = put(
        ADMIN,
        json!({"expected_revision":revision,"text":"{\n  \"llm\": {,\n}"}),
    );
    assert_eq!(reason(&broken), (422, json!("invalid_config")));
    assert!(
        broken.json()["detail"]
            .as_str()
            .unwrap()
            .starts_with("line 2, column 11: ")
    );

    // A new model applies at once, the masked secrets are kept.
    let mut edited: Value = serde_json::from_str(&text).unwrap();
    edited["llm"]["model_list"]
        .as_array_mut()
        .unwrap()
        .push(json!({"model_name":"default","litellm_params":{"model":"openai/second","api_key":"env:FIXTURE_KEY"}}));
    let reply = put(
        ADMIN,
        json!({"expected_revision":revision,"text":edited.to_string()}),
    );
    assert_eq!(reply.status, 200, "{}", reply.text());
    let answer = reply.json();
    assert_eq!(answer["restart_required"], false);
    assert_ne!(answer["revision"], json!(revision));
    let saved: Value =
        serde_json::from_slice(&std::fs::read(temp.path().join("config.json")).unwrap()).unwrap();
    assert_eq!(
        saved["llm"]["model_list"][0]["litellm_params"],
        config["llm"]["model_list"][0]["litellm_params"]
    );
    assert_eq!(saved["fetch"]["playwright"], config["fetch"]["playwright"]);
    let models: Vec<_> =
        call(&server, "GET", "/api/settings/llm", ADMIN, None).json()["deployments"]
            .as_array()
            .unwrap()
            .iter()
            .map(|entry| entry["model"].clone())
            .collect();
    assert_eq!(models, [json!("openai/first"), json!("openai/second")]);
    // Anything else waits for a restart.
    let mut edited: Value = serde_json::from_str(answer["text"].as_str().unwrap()).unwrap();
    edited["batch"]["concurrency"] = json!(3);
    let reply = put(
        ADMIN,
        json!({"expected_revision":answer["revision"],"text":edited.to_string()}),
    );
    assert_eq!(reply.status, 200, "{}", reply.text());
    assert_eq!(reply.json()["restart_required"], true);
    server.stop();
}

/// Windows writes `config.json` with CRLF. The server sends the file's own
/// text and saves exactly what it is sent, so a page that sends the file's
/// endings back (the workbench does) changes only the lines it edited.
#[test]
fn a_crlf_config_file_round_trips_byte_for_byte() {
    let temp = tempfile::tempdir().unwrap();
    configure(temp.path());
    let path = temp.path().join("config.json");
    let mut config: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    config["llm"]["model_list"] = json!([{"model_name":"default","litellm_params":{
        "model":"openai/first","api_key":"sk-fixture-1234"}}]);
    let original = serde_json::to_string_pretty(&config)
        .unwrap()
        .replace('\n', "\r\n")
        + "\r\n";
    std::fs::write(&path, &original).unwrap();
    let server = start(temp.path());
    let get = || call(&server, "GET", "/api/settings/config", ADMIN, None).json();
    let put = |revision: &Value, text: &str| {
        call(
            &server,
            "PUT",
            "/api/settings/config",
            ADMIN,
            Some(json!({"expected_revision":revision,"text":text})),
        )
    };

    // GET sends the CRLF text (a secret masked); sending it back unchanged
    // writes the same bytes.
    let view = get();
    let text = view["text"].as_str().unwrap().to_owned();
    assert_eq!(view["masked"], 1);
    assert!(!text.contains("fixture-1234"));
    assert_eq!(text.matches("\r\n").count(), text.matches('\n').count());
    assert!(text.ends_with("}\r\n"));
    let reply = put(&view["revision"], &text);
    assert_eq!(reply.status, 200, "{}", reply.text());
    assert_eq!(std::fs::read_to_string(&path).unwrap(), original);
    assert_eq!(reply.json()["revision"], view["revision"]);
    assert_eq!(reply.json()["restart_required"], false);

    // One changed value: only that line differs, every ending stays CRLF.
    let edited = text.replace("\"openai/first\"", "\"openai/second\"");
    assert_ne!(edited, text);
    let reply = put(&view["revision"], &edited);
    assert_eq!(reply.status, 200, "{}", reply.text());
    let saved = std::fs::read_to_string(&path).unwrap();
    assert_eq!(
        saved,
        original.replace("\"openai/first\"", "\"openai/second\"")
    );
    let (before, after): (Vec<_>, Vec<_>) = (
        original.split("\r\n").collect(),
        saved.split("\r\n").collect(),
    );
    assert_eq!(before.len(), after.len());
    let changed: Vec<_> = before
        .iter()
        .zip(&after)
        .filter(|(old, new)| old != new)
        .map(|(_, new)| *new)
        .collect();
    assert_eq!(changed, ["          \"model\": \"openai/second\""]);
    assert_eq!(saved.matches("\r\n").count(), saved.matches('\n').count());

    // Without a final line break, the file stays without one.
    let view = get();
    let bare = view["text"]
        .as_str()
        .unwrap()
        .trim_end_matches("\r\n")
        .to_owned();
    let reply = put(&view["revision"], &bare);
    assert_eq!(reply.status, 200, "{}", reply.text());
    let saved = std::fs::read_to_string(&path).unwrap();
    assert!(saved.ends_with('}') && saved.contains("sk-fixture-1234"));
    let view = get();
    let reply = put(&view["revision"], view["text"].as_str().unwrap());
    assert_eq!(reply.status, 200, "{}", reply.text());
    assert_eq!(std::fs::read_to_string(&path).unwrap(), saved);

    // A byte order mark makes the file unreadable as configuration for both
    // backends: the page gets no text to edit and the file is left alone.
    let marked = format!("\u{feff}{saved}");
    std::fs::write(&path, &marked).unwrap();
    let view = get();
    assert_eq!(view["text"], Value::Null);
    assert!(
        view["problem"]
            .as_str()
            .is_some_and(|problem| problem.starts_with("line 1, column 1")),
        "{view}"
    );
    assert_eq!(std::fs::read_to_string(&path).unwrap(), marked);
    server.stop();
}

#[test]
fn cloudflare_processing_billed_to_the_server_is_the_admins_alone() {
    let temp = tempfile::tempdir().unwrap();
    configure(temp.path());
    let path = temp.path().join("config.json");
    let mut cfg: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    cfg["fetch"]["remote_consent"] = json!("ask");
    cfg["fetch"]["cloudflare"] = json!({"api_token":"synthetic-token","account_id":"synthetic-account","convert_enabled":true});
    std::fs::write(&path, cfg.to_string()).unwrap();
    let server = start(temp.path());
    let created = call(
        &server,
        "POST",
        "/api/access-tokens",
        ADMIN,
        Some(json!({"label":"guest"})),
    );
    let user = created.json()["token"].as_str().unwrap().to_owned();
    let cloudflare = |token: &str| {
        call(&server, "GET", "/api/capabilities", token, None).json()["remote_services"]
            ["cloudflare"]
            .clone()
    };
    assert_eq!(cloudflare(ADMIN)["available"], true);
    assert_eq!(cloudflare(&user)["available"], false);
    assert_eq!(cloudflare(&user)["reason"], "client_not_trusted");
    let confirmed = json!({"backend":"cloudflare","remote_processing":"cloudflare"});
    let (content, body) = multipart(&[("no.txt", b"local")], json!([]), confirmed);
    let authorization = bearer(&user);
    let refused = server.request(
        "POST",
        "/api/jobs",
        &[
            ("Content-Type", content.as_str()),
            ("Authorization", authorization.as_str()),
        ],
        &body,
    );
    assert_eq!(refused.status, 403, "{}", refused.text());
    assert_eq!(refused.json()["reason"], "remote_processing_forbidden");
    server.stop();
}

#[test]
fn a_user_url_job_cannot_reach_this_machine() {
    // No MARKITAI_TEST_PUBLIC_NET_ALLOW: the loopback origin is private, as a
    // real one is. A user token's URL job is held to public addresses, so it
    // never reaches the server's own machine; the admin's unchanged.
    let origin = Origin::start();
    let temp = tempfile::tempdir().unwrap();
    let server = Server::start_with(
        temp.path(),
        &[("MARKITAI_SERVE_TOKEN", std::ffi::OsStr::new(ADMIN))],
    );
    let created = call(
        &server,
        "POST",
        "/api/access-tokens",
        ADMIN,
        Some(json!({"label": "Reader"})),
    );
    assert_eq!(created.status, 201, "{}", created.text());
    let user = created.json()["token"].as_str().unwrap().to_owned();

    // The user's URL item fails without reaching the origin.
    let mine = submit_as(&server, &user, &[], json!([origin.url("/page")]));
    let item = done_as(&server, &user, &mine)["items"][0].clone();
    assert_eq!(item["status"], "error", "{item}");
    assert_eq!(origin.entered.load(Ordering::SeqCst), 0);

    // The admin reaches the same loopback URL (not a user job).
    let theirs = submit_as(&server, ADMIN, &[], json!([origin.url("/page")]));
    let admin_item = done_as(&server, ADMIN, &theirs)["items"][0].clone();
    assert_eq!(admin_item["status"], "done", "{admin_item}");
    assert!(origin.entered.load(Ordering::SeqCst) >= 1);
}
