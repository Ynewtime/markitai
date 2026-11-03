//! `serve --supervised`: a server run for the app that started it, which
//! holds the token, updates the program and stops the server by closing its
//! standard input.
use super::*;
use std::ffi::OsStr;
use std::process::ChildStdin;

const TOKEN: &str = "supervised-test-token-0001";
/// The features that follow from the server issuing tokens of its own.
const TOKEN_FEATURES: [&str; 4] = [
    "token_regenerate",
    "user_tokens",
    "user_models",
    "user_config",
];

/// A supervised server whose standard input this test holds.
fn start(root: &Path, variables: &[(&str, &OsStr)]) -> (Server, ChildStdin) {
    let mut command = Server::command(root, variables);
    command.arg("--supervised").stdin(Stdio::piped());
    let mut server = Server::spawn(root, command);
    let stdin = server.child.as_mut().unwrap().stdin.take().unwrap();
    (server, stdin)
}
fn reason(reply: &Reply) -> (u16, Value) {
    (reply.status, reply.json()["reason"].clone())
}
fn get(server: &Server, path: &str, authorization: &str) -> Reply {
    server.request("GET", path, &[("Authorization", authorization)], &[])
}

#[test]
fn closing_standard_input_drains_as_a_signal_does() {
    let origin = Origin::start();
    let temp = tempfile::tempdir().unwrap();
    let (server, stdin) = start(temp.path(), &[]);
    let created = server.submit(
        &[],
        json!([origin.url("/hold"), origin.url("/queued")]),
        json!({"strategy":"static"}),
    );
    let id = created["job_id"].as_str().unwrap().to_owned();
    until("first URL reaches origin", || {
        origin.entered.load(Ordering::SeqCst) == 1
    });
    let path = server.jobdir(&id);
    drop(stdin);
    until("shutdown acknowledgement", || {
        server
            .captured
            .lock()
            .unwrap()
            .contains("Server stopping: draining active conversions and saving history.")
    });
    origin.release();
    server.finish(true);
    assert_eq!(
        origin.entered.load(Ordering::SeqCst),
        1,
        "queued URL must not dispatch after standard input closed"
    );
    let saved: Value =
        serde_json::from_slice(&std::fs::read(path.join("meta.json")).unwrap()).unwrap();
    assert_eq!(saved["items"][0]["status"], "done");
    assert_eq!(saved["items"][1]["status"], "error");
    assert_eq!(saved["items"][1]["error"], "cancelled (server shutdown)");
    assert!(!temp.path().join("home/serve/control").exists());
}

#[test]
fn the_app_that_starts_a_supervised_server_holds_its_token() {
    let temp = tempfile::tempdir().unwrap();
    let admin = format!("Bearer {TOKEN}");
    let variables = [("MARKITAI_SERVE_TOKEN", OsStr::new(TOKEN))];
    // An ordinary server of this home issues a user token.
    let server = Server::start_with(temp.path(), &variables);
    let capabilities = get(&server, "/api/capabilities", &admin).json();
    assert_eq!(capabilities["supervised"], false);
    for feature in TOKEN_FEATURES {
        assert_eq!(capabilities["features"][feature], true, "{feature}");
    }
    let created = server.request(
        "POST",
        "/api/access-tokens",
        &[
            ("Authorization", &admin),
            ("Content-Type", "application/json"),
        ],
        br#"{"label":"phone"}"#,
    );
    assert_eq!(created.status, 201, "{}", created.text());
    let user = format!("Bearer {}", created.json()["token"].as_str().unwrap());
    assert_eq!(get(&server, "/api/me/models", &user).status, 200);
    server.stop();

    let (server, stdin) = start(temp.path(), &variables);
    let capabilities = get(&server, "/api/capabilities", &admin).json();
    assert_eq!(capabilities["supervised"], true);
    for feature in TOKEN_FEATURES {
        assert_eq!(capabilities["features"][feature], false, "{feature}");
    }
    // Only the app's token opens it: the user token and its own models are shut out.
    for path in ["/api/capabilities", "/api/me/models", "/api/me/config"] {
        assert_eq!(
            reason(&get(&server, path, &user)),
            (401, json!("token_required")),
            "{path}"
        );
    }
    for (method, path, body) in [
        ("POST", "/api/access-token/regenerate", &b""[..]),
        ("GET", "/api/access-tokens", &b""[..]),
        ("POST", "/api/access-tokens", &br#"{"label":"x"}"#[..]),
        ("DELETE", "/api/access-tokens/u_000000000000", &b""[..]),
    ] {
        let reply = server.request(
            method,
            path,
            &[
                ("Authorization", &admin),
                ("Content-Type", "application/json"),
            ],
            body,
        );
        assert_eq!(
            reason(&reply),
            (409, json!("token_disabled")),
            "{method} {path}"
        );
    }
    assert_eq!(get(&server, "/api/capabilities", &admin).status, 200);
    drop(stdin);
    server.finish(true);
    // The home's user token is left for its own server.
    let kept = std::fs::read_to_string(temp.path().join("home/serve/access.json")).unwrap();
    assert!(kept.contains("\"phone\""), "{kept}");
}

#[test]
fn a_supervised_server_neither_checks_for_nor_offers_an_update() {
    // A release host that only notes that someone asked.
    let host = TcpListener::bind("127.0.0.1:0").unwrap();
    host.set_nonblocking(true).unwrap();
    let base = format!("http://{}", host.local_addr().unwrap());
    let admin = format!("Bearer {TOKEN}");
    // Posing as a release, a server checks at once and offers the web update.
    let variables = [
        ("MARKITAI_SERVE_TOKEN", OsStr::new(TOKEN)),
        ("MARKITAI_TEST_UPDATE_AS", OsStr::new("2.3.2")),
        ("MARKITAI_DOWNLOAD_URL", OsStr::new(&base)),
    ];
    let ordinary = tempfile::tempdir().unwrap();
    let server = Server::start_with(ordinary.path(), &variables);
    assert!(get(&server, "/api/capabilities", &admin).json()["update"].is_object());
    until("the ordinary server's check", || host.accept().is_ok());
    server.stop();
    assert!(ordinary.path().join("home/update-check.json").exists());

    let temp = tempfile::tempdir().unwrap();
    let (server, stdin) = start(temp.path(), &variables);
    let capabilities = get(&server, "/api/capabilities", &admin).json();
    assert_eq!(capabilities["supervised"], true);
    assert!(capabilities.get("update").is_none(), "{capabilities}");
    for (method, path) in [("GET", "/api/update"), ("POST", "/api/update/check")] {
        let reply = server.request(method, path, &[("Authorization", &admin)], &[]);
        assert_eq!(
            reason(&reply),
            (404, json!("route_not_found")),
            "{method} {path}"
        );
    }
    // `markitai update` finds the server, which refuses to restart.
    let control: Value =
        serde_json::from_slice(&std::fs::read(temp.path().join("home/serve/control")).unwrap())
            .unwrap();
    let secret = control["secret"].as_str().unwrap();
    let header = [("X-Markitai-Control", secret)];
    assert_eq!(
        server
            .request("GET", "/api/update/control", &header, &[])
            .status,
        200
    );
    assert_eq!(
        reason(&server.request("POST", "/api/update/restart", &header, &[])),
        (409, json!("update_unavailable"))
    );
    // The ordinary server asked within its startup; this one has had longer.
    std::thread::sleep(Duration::from_secs(1));
    assert!(
        host.accept().is_err(),
        "the supervised server asked for a release"
    );
    assert!(!temp.path().join("home/update-check.json").exists());
    drop(stdin);
    server.finish(true);
}
