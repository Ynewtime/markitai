//! Replacing the access token from the web workbench: the old token, its
//! download tickets and its open event streams stop working at once.
use super::*;

const TOKEN: &str = "startup-test-token-0001";

/// Opens the job's event stream and returns it once the first snapshot
/// arrived.
fn subscribe(port: u16, id: &str, authorization: &str) -> TcpStream {
    let mut socket = TcpStream::connect(("127.0.0.1", port)).unwrap();
    socket.set_read_timeout(Some(WAIT)).unwrap();
    write!(
        socket,
        "GET /api/jobs/{id}/events HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nAuthorization: {authorization}\r\nConnection: close\r\n\r\n"
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
fn a_regenerated_token_replaces_the_old_one_its_tickets_and_its_event_streams() {
    let origin = Origin::start();
    let temp = tempfile::tempdir().unwrap();
    let server = Server::start_with(
        temp.path(),
        &[("MARKITAI_SERVE_TOKEN", std::ffi::OsStr::new(TOKEN))],
    );
    let old = format!("Bearer {TOKEN}");
    let get = |path: &str, authorization: &str| {
        server.request("GET", path, &[("Authorization", authorization)], &[])
    };
    let capabilities = get("/api/capabilities", &old).json();
    assert_eq!(capabilities["features"]["token_regenerate"], true);
    // A running job holds its event stream open.
    let (content_type, body) = multipart(
        &[],
        json!([origin.url("/hold")]),
        json!({"strategy":"static"}),
    );
    let created = server.request(
        "POST",
        "/api/jobs",
        &[("Content-Type", &content_type), ("Authorization", &old)],
        &body,
    );
    assert_eq!(created.status, 201, "{}", created.text());
    let id = created.json()["job_id"].as_str().unwrap().to_owned();
    until("held URL reaches origin", || {
        origin.entered.load(Ordering::SeqCst) == 1
    });
    let mut stream = subscribe(server.port, &id, &old);
    let ticket = server.request(
        "POST",
        "/api/download-tickets",
        &[
            ("Content-Type", "application/json"),
            ("Authorization", &old),
        ],
        json!({"path":format!("/api/jobs/{id}/items/i1/archive")})
            .to_string()
            .as_bytes(),
    );
    assert_eq!(ticket.status, 201, "{}", ticket.text());
    let ticket = ticket.json()["url"].as_str().unwrap().to_owned();
    let regenerate = |headers: &[(&str, &str)], body: &[u8]| {
        server.request("POST", "/api/access-token/regenerate", headers, body)
    };
    // Another site may not ask, nor may a body say anything.
    let refused = regenerate(
        &[("Authorization", &old), ("Origin", "https://evil.invalid")],
        b"",
    );
    assert_eq!(
        (refused.status, refused.json()["reason"].clone()),
        (403, json!("origin_not_allowed"))
    );
    let refused = regenerate(
        &[
            ("Authorization", &old),
            ("Content-Type", "application/json"),
        ],
        b"{\"x\":1}",
    );
    assert_eq!(refused.status, 422, "{}", refused.text());
    assert_eq!(regenerate(&[], b"").status, 401);
    let same_origin = format!("http://127.0.0.1:{}", server.port);
    let reply = regenerate(
        &[
            ("Authorization", &old),
            ("Origin", &same_origin),
            ("Content-Type", "application/json"),
        ],
        b"{}",
    );
    assert_eq!(reply.status, 200, "{}", reply.text());
    assert_eq!(reply.headers["cache-control"], "no-store");
    let answer = reply.json();
    let token = answer["token"].as_str().unwrap().to_owned();
    assert_eq!(token.len(), 64);
    assert!(token.bytes().all(|byte| byte.is_ascii_hexdigit()) && token != TOKEN);
    assert_eq!(
        answer,
        json!({"token":token,"network_url":null,"persistent":false,"startup_variable":"MARKITAI_SERVE_TOKEN"})
    );
    // The open stream ends, and its reconnect with the old token is refused.
    let mut rest = Vec::new();
    stream.read_to_end(&mut rest).unwrap();
    let ended = request(
        server.port,
        "GET",
        &format!("/api/jobs/{id}/events"),
        &[("Authorization", &old)],
        &[],
    );
    assert_eq!(
        (ended.status, ended.json()["reason"].clone()),
        (401, json!("token_required"))
    );
    assert_eq!(get("/api/capabilities", &old).status, 401);
    let new = format!("Bearer {token}");
    assert_eq!(get("/api/capabilities", &new).status, 200);
    let spent = server.request("GET", &ticket, &[], &[]);
    assert_eq!(
        (spent.status, spent.json()["reason"].clone()),
        (401, json!("ticket_invalid"))
    );
    // Without a body at all, again.
    let again = regenerate(&[("Authorization", &new)], b"");
    assert_eq!(again.status, 200, "{}", again.text());
    assert_eq!(get("/api/capabilities", &new).status, 401);
    let newest = format!("Bearer {}", again.json()["token"].as_str().unwrap());
    origin.release();
    until("held job finishes", || {
        get(&format!("/api/jobs/{id}"), &newest).json()["status"] != "running"
    });
    let log = server.captured.lock().unwrap().clone();
    assert_eq!(
        log.matches("Access token regenerated from the web workbench; other tabs and devices must sign in again.").count(),
        2,
        "{log}"
    );
    assert!(!log.contains(&token));
    server.stop();
}

#[test]
fn without_a_token_there_is_nothing_to_regenerate() {
    let temp = tempfile::tempdir().unwrap();
    let server = Server::start(temp.path());
    assert_eq!(
        server.json("/api/capabilities")["features"]["token_regenerate"],
        false
    );
    let reply = server.request("POST", "/api/access-token/regenerate", &[], b"");
    assert_eq!(
        (reply.status, reply.json()["reason"].clone()),
        (409, json!("token_disabled"))
    );
    assert_eq!(reply.headers["cache-control"], "no-store");
    // A client that is not trusted is refused as the settings refuse it.
    let reply = server.request(
        "POST",
        "/api/access-token/regenerate",
        &[("X-Forwarded-For", "203.0.113.9")],
        b"",
    );
    assert_eq!(
        (reply.status, reply.json()["reason"].clone()),
        (403, json!("settings_forbidden"))
    );
    server.stop();
}
