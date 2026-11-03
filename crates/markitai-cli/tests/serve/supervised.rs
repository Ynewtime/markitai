//! `serve --supervised`: a server run for the app that started it, which
//! holds the token, updates the program and stops the server through its
//! standard input: a line `drain` or the input's end drains it as a signal
//! does, a line `now` stops it at once.
use super::*;
use crate::{fake_runtime, interrupt};
use std::ffi::OsStr;
use std::process::ChildStdin;

const TOKEN: &str = "supervised-test-token-0001";
const DRAINING: &str = "Server stopping: draining active conversions and saving history.";
/// The exit status after `now`: 128 + SIGHUP.
const STOPPED_NOW: i32 = 129;
/// The features that follow from the server issuing tokens of its own.
const TOKEN_FEATURES: [&str; 4] = [
    "token_regenerate",
    "user_tokens",
    "user_models",
    "user_config",
];

/// A supervised server with the app's token, whose standard input this test
/// holds; its standard output is piped too.
fn start(root: &Path, variables: &[(&str, &OsStr)]) -> (Server, ChildStdin) {
    let mut all = vec![("MARKITAI_SERVE_TOKEN", OsStr::new(TOKEN))];
    all.extend_from_slice(variables);
    let mut command = Server::command(root, &all);
    command
        .arg("--supervised")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped());
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
/// Submits a job as the app does; its id.
fn submit(server: &Server, files: &[(&str, &[u8])], urls: Value, options: Value) -> String {
    let (content_type, body) = multipart(files, urls, options);
    let admin = format!("Bearer {TOKEN}");
    let reply = server.request(
        "POST",
        "/api/jobs",
        &[("Content-Type", &content_type), ("Authorization", &admin)],
        &body,
    );
    assert_eq!(reply.status, 201, "{}", reply.text());
    reply.json()["job_id"].as_str().unwrap().to_owned()
}
fn ask(stdin: &mut ChildStdin, lines: &str) {
    stdin.write_all(lines.as_bytes()).unwrap();
    stdin.flush().unwrap();
}
/// Waits for the server to exit by itself: its exit code and its stderr. One
/// still running after the wait is killed as the server is dropped.
fn exit(mut server: Server) -> (Option<i32>, String) {
    until("server exit", || {
        server.child.as_mut().unwrap().try_wait().unwrap().is_some()
    });
    let status = server.child.take().unwrap().wait().unwrap();
    server.stderr.take().unwrap().join().unwrap();
    let text = server.captured.lock().unwrap().clone();
    (status.code(), text)
}

#[test]
fn closing_standard_input_drains_as_a_signal_does() {
    let origin = Origin::start();
    let temp = tempfile::tempdir().unwrap();
    let (server, mut stdin) = start(temp.path(), &[]);
    let id = submit(
        &server,
        &[],
        json!([origin.url("/hold"), origin.url("/queued")]),
        json!({"strategy":"static"}),
    );
    until("first URL reaches origin", || {
        origin.entered.load(Ordering::SeqCst) == 1
    });
    let path = server.jobdir(&id);
    // Lines that are not requests change nothing.
    ask(&mut stdin, "stop\nNOW\n now\nnow please\n");
    std::thread::sleep(Duration::from_millis(500));
    assert_eq!(
        get(&server, "/api/capabilities", &format!("Bearer {TOKEN}")).status,
        200
    );
    let said = server.captured.lock().unwrap().clone();
    assert!(!said.contains("Server stopping"), "{said}");
    drop(stdin);
    until("shutdown acknowledgement", || {
        server.captured.lock().unwrap().contains(DRAINING)
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

    let (mut server, stdin) = start(temp.path(), &variables);
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
    // A user's own models and configuration are a user token's, as on any server.
    for path in ["/api/me/models", "/api/me/config"] {
        assert_eq!(
            reason(&get(&server, path, &admin)),
            (403, json!("user_token_required")),
            "{path}"
        );
    }
    assert_eq!(get(&server, "/api/capabilities", &admin).status, 200);
    let mut stdout = server.child.as_mut().unwrap().stdout.take().unwrap();
    drop(stdin);
    let (code, stderr) = exit(server);
    assert_eq!(code, Some(0), "{stderr}");
    let mut printed = String::new();
    stdout.read_to_string(&mut printed).unwrap();
    // Nor does the server print the token, alone or in an address: the app
    // writes what it prints to a log.
    assert!(!printed.contains(TOKEN), "{printed}");
    assert!(!stderr.contains(TOKEN), "{stderr}");
    assert!(
        stderr.contains("The app that started this server stops it through standard input.")
            && !stderr.contains("Ctrl-C"),
        "{stderr}"
    );
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
    // `markitai update` finds the server, which says it is supervised and
    // refuses to restart.
    let control: Value =
        serde_json::from_slice(&std::fs::read(temp.path().join("home/serve/control")).unwrap())
            .unwrap();
    let secret = control["secret"].as_str().unwrap();
    let header = [("X-Markitai-Control", secret)];
    let status = server.request("GET", "/api/update/control", &header, &[]);
    assert_eq!(
        (status.status, &status.json()["supervised"]),
        (200, &json!(true)),
        "{}",
        status.text()
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

#[test]
fn a_line_now_stops_at_once_and_ends_the_runtime_a_conversion_started() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    // The conversion's model is the Codex stand-in, whose turn starts a
    // descendant of its own and hangs.
    let runtime = root.join("runtime");
    let tmp = root.join("tmp");
    fake_runtime::install(&runtime, &json!({"name":"sleep"}));
    std::fs::create_dir(&tmp).unwrap();
    std::fs::write(
        root.join("config.json"),
        json!({"log":{"dir":null},"history":{"record":false},"cache":{"enabled":false},
            "image":{"alt_enabled":false,"desc_enabled":false},
            "llm":{"enabled":true,"on_failure":"fail","router_settings":{"num_retries":0,"timeout":120},
                "model_list":[{"model_name":"default","litellm_params":{"model":"chatgpt/gpt-5.5"}}]}})
        .to_string(),
    )
    .unwrap();
    let program = fake_runtime::program();
    let (server, mut stdin) = start(
        root,
        &[
            ("CODEX_CLI_PATH", program.as_os_str()),
            ("CODEX_HOME", runtime.as_os_str()),
            ("TMPDIR", tmp.as_os_str()),
        ],
    );
    let id = submit(
        &server,
        &[("source.md", b"# Title\n\nBody.\n")],
        json!([]),
        json!({"llm":true}),
    );
    let mut grandchild = 0;
    until("the runtime's turn starts", || {
        std::fs::read_to_string(runtime.join("grandchild.pid"))
            .ok()
            .and_then(|text| text.trim().parse().ok())
            .is_some_and(|pid| {
                grandchild = pid;
                true
            })
    });
    ask(&mut stdin, "now\n");
    let (code, stderr) = exit(server);
    assert_eq!(code, Some(STOPPED_NOW), "{stderr}");
    assert!(
        stderr.contains("Server stopping now:") && !stderr.contains(DRAINING),
        "{stderr}"
    );
    // The runtime's tree and its private workspace end with the server.
    let deadline = Instant::now() + Duration::from_secs(10);
    while interrupt::alive(grandchild) {
        if Instant::now() > deadline {
            interrupt::kill(grandchild);
            panic!("the runtime's descendant outlived the server");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let left: Vec<_> = std::fs::read_dir(&tmp)
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .filter(|name| name.to_string_lossy().starts_with("markitai-"))
        .collect();
    assert!(
        left.is_empty(),
        "runtime workspaces outlived the server: {left:?}"
    );
    // The conversion did not finish; the next start says it was interrupted.
    let server = Server::start_with(root, &[("MARKITAI_SERVE_TOKEN", OsStr::new(TOKEN))]);
    let job = get(
        &server,
        &format!("/api/jobs/{id}"),
        &format!("Bearer {TOKEN}"),
    )
    .json();
    assert_eq!(job["items"][0]["status"], "error", "{job}");
    assert_eq!(job["items"][0]["error_code"], "interrupted", "{job}");
    server.stop();
}

#[test]
fn a_line_drain_drains_with_input_open_so_now_can_still_stop_at_once() {
    let origin = Origin::start();
    let temp = tempfile::tempdir().unwrap();
    let (server, mut stdin) = start(temp.path(), &[]);
    let id = submit(
        &server,
        &[],
        json!([origin.url("/hold"), origin.url("/queued")]),
        json!({"strategy":"static"}),
    );
    until("first URL reaches origin", || {
        origin.entered.load(Ordering::SeqCst) == 1
    });
    let path = server.jobdir(&id);
    // As a Windows app may end it.
    ask(&mut stdin, "drain\r\n");
    until("shutdown acknowledgement", || {
        server.captured.lock().unwrap().contains(DRAINING)
    });
    // The held conversion keeps the drain waiting; `now` ends it there.
    ask(&mut stdin, "now\n");
    let (code, stderr) = exit(server);
    assert_eq!(code, Some(STOPPED_NOW), "{stderr}");
    assert_eq!(
        origin.entered.load(Ordering::SeqCst),
        1,
        "queued URL must not dispatch after drain"
    );
    // Abandoned, not saved: the next start marks the job's conversions interrupted.
    let saved: Value =
        serde_json::from_slice(&std::fs::read(path.join("meta.json")).unwrap()).unwrap();
    assert_eq!(saved["status"], "running", "{saved}");
    assert_ne!(saved["items"][0]["status"], "done", "{saved}");
}

#[test]
fn standard_input_at_its_end_from_the_start_stops_the_server_once_it_listens() {
    // An app hands over a pipe it keeps open. With nothing to read the server
    // starts, then drains at once, as when the app closes its end.
    let temp = tempfile::tempdir().unwrap();
    let mut command = Server::command(temp.path(), &[("MARKITAI_SERVE_TOKEN", OsStr::new(TOKEN))]);
    command.arg("--supervised").stdin(Stdio::null());
    let (code, stderr) = Server::exited(command);
    assert_eq!(code, Some(0), "{stderr}");
    let listening = stderr.find("Markitai server listening on http://127.0.0.1:");
    let draining = stderr.find(DRAINING);
    assert!(
        listening.is_some() && draining.is_some() && listening < draining,
        "{stderr}"
    );
}

#[test]
fn a_supervised_server_needs_the_apps_token_and_refuses_no_auth() {
    let temp = tempfile::tempdir().unwrap();
    for set in [true, false] {
        let mut command =
            Server::command(temp.path(), &[("MARKITAI_SERVE_TOKEN", OsStr::new(" "))]);
        if !set {
            command.env_remove("MARKITAI_SERVE_TOKEN");
        }
        command.arg("--supervised");
        let (code, stderr) = Server::exited(command);
        assert_eq!(code, Some(1), "{stderr}");
        assert!(
            stderr.contains("MARKITAI_SERVE_TOKEN") && !stderr.contains("listening"),
            "{stderr}"
        );
    }
    let mut command = Server::command(temp.path(), &[("MARKITAI_SERVE_TOKEN", OsStr::new(TOKEN))]);
    command.args(["--supervised", "--no-auth"]);
    let (code, stderr) = Server::exited(command);
    assert_eq!(code, Some(2), "{stderr}");
    assert!(
        stderr.contains("'--supervised' cannot be used with '--no-auth'"),
        "{stderr}"
    );
}
