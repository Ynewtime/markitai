#![cfg(unix)]
use serde_json::{Value, json};
use std::{
    fs,
    io::{BufRead, BufReader, Read, Write},
    net::TcpStream,
    path::Path,
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
    thread::JoinHandle,
    time::{Duration, Instant},
};
const WAIT: Duration = Duration::from_secs(30);
struct Server {
    child: Child,
    port: u16,
    log: Arc<Mutex<String>>,
    reader: Option<JoinHandle<()>>,
}
impl Server {
    fn start(root: &Path, extra: &[&str]) -> Self {
        let mut command = Command::new(env!("CARGO_BIN_EXE_markitai"));
        command.env_clear();
        for key in [
            "PATH",
            "HOME",
            "USERPROFILE",
            "SYSTEMROOT",
            "TMPDIR",
            "TMP",
            "TEMP",
        ] {
            if let Some(value) = std::env::var_os(key) {
                command.env(key, value);
            }
        }
        let mut child = command
            .current_dir(root)
            .env("MARKITAI_HOME", root.join("home"))
            .env("NO_PROXY", "127.0.0.1,localhost")
            .args(["--config", "config.json"])
            .args(extra)
            .args([
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
        let log = Arc::new(Mutex::new(String::new()));
        let copy = log.clone();
        let stderr = child.stderr.take().unwrap();
        let reader = std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                let mut log = copy.lock().unwrap();
                log.push_str(&line);
                log.push('\n');
            }
        });
        let mut server = Self {
            child,
            port: 0,
            log,
            reader: Some(reader),
        };
        let deadline = Instant::now() + WAIT;
        server.port = loop {
            assert!(
                server.child.try_wait().unwrap().is_none(),
                "{}",
                server.log.lock().unwrap()
            );
            if let Some(port) = server.log.lock().unwrap().lines().find_map(|line| {
                line.strip_prefix("Markitai server listening on http://127.0.0.1:")
                    .and_then(|p| p.parse().ok())
            }) {
                break port;
            }
            assert!(Instant::now() < deadline, "service startup timeout");
            std::thread::sleep(Duration::from_millis(5));
        };
        server
    }
    fn request(&self, method: &str, path: &str, value: Option<Value>) -> (u16, Value) {
        request(self.port, method, path, value)
    }
    fn view(&self) -> Value {
        let (status, value) = self.request("GET", "/api/settings/llm", None);
        assert_eq!(status, 200, "{value}");
        value
    }
    fn stop(mut self) {
        unsafe {
            libc::kill(self.child.id() as i32, libc::SIGINT);
        };
        let deadline = Instant::now() + WAIT;
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                assert!(status.success(), "{}", self.log.lock().unwrap());
                break;
            }
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(5));
        }
        self.reader.take().unwrap().join().unwrap();
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
    }
}
fn request(port: u16, method: &str, path: &str, value: Option<Value>) -> (u16, Value) {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
    stream.set_read_timeout(Some(WAIT)).unwrap();
    stream.set_write_timeout(Some(WAIT)).unwrap();
    let body = value.map(|v| v.to_string()).unwrap_or_default();
    write!(stream,"{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",body.len()).unwrap();
    let mut raw = Vec::new();
    stream.take(4 * 1024 * 1024).read_to_end(&mut raw).unwrap();
    let split = raw.windows(4).position(|v| v == b"\r\n\r\n").unwrap();
    let headers = std::str::from_utf8(&raw[..split]).unwrap();
    let status = headers
        .lines()
        .next()
        .unwrap()
        .split_whitespace()
        .nth(1)
        .unwrap()
        .parse()
        .unwrap();
    if path.starts_with("/api/settings/llm") {
        assert!(
            headers
                .lines()
                .any(|line| line.eq_ignore_ascii_case("cache-control: no-store")),
            "{headers}"
        );
    }
    let body = &raw[split + 4..];
    let value = if body.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(body)
            .unwrap_or_else(|error| panic!("{error}: {}", String::from_utf8_lossy(body)))
    };
    (status, value)
}
fn configuration(root: &Path) -> Value {
    let value = json!({"unknown":{"keep":"界"},"log":{"dir":null},"prompts":{"dir":root.join("prompts")},"cache":{"enabled":false},"history":{"record":false},"llm":{"enabled":false,"model_list":[{"model_name":"legacy","litellm_params":{"model":"openai/initial","api_key":"env:AUTHOR_KEY"},"unknown_model":{"keep":1}}]}});
    fs::write(root.join("config.json"), value.to_string()).unwrap();
    value
}
fn saved(root: &Path) -> Value {
    serde_json::from_slice(&fs::read(root.join("config.json")).unwrap()).unwrap()
}
fn rev(view: &Value) -> String {
    view["revision"].as_str().unwrap().to_owned()
}
#[test]
fn a_subscription_model_is_routable_in_settings_as_in_conversion() {
    // The conversion side routes subscription runtimes without an API key;
    // the settings view must not report that no model can be routed.
    let dir = tempfile::tempdir().unwrap();
    let value = json!({"log":{"dir":null},"cache":{"enabled":false},"history":{"record":false},"llm":{"enabled":true,"model_list":[{"model_name":"default","litellm_params":{"model":"chatgpt/gpt-5.5"}}]}});
    fs::write(dir.path().join("config.json"), value.to_string()).unwrap();
    // The official Codex program is found, as conversion requires; it is
    // never run while the view is computed.
    let codex = dir.path().join("codex");
    fs::write(&codex, "").unwrap();
    fs::write(
        dir.path().join(".env"),
        format!("CODEX_CLI_PATH={}\n", codex.display()),
    )
    .unwrap();
    let server = Server::start(dir.path(), &[]);
    let view = server.view();
    assert_eq!(view["routable"], true, "{view}");
    server.stop();
}

#[test]
fn actual_http_settings_crud_preserves_ids_credentials_fields_and_restart() {
    let dir = tempfile::tempdir().unwrap();
    let initial = configuration(dir.path());
    let server = Server::start(dir.path(), &[]);
    let view = server.view();
    assert_eq!(view["config_origin"], "explicit");
    assert!(
        view["deployments"][0]["deployment_id"]
            .as_str()
            .unwrap()
            .starts_with("legacy-")
    );
    let(status,view)=server.request("POST","/api/settings/llm/deployments/batch",Some(json!({"expected_revision":rev(&view),"deployments":[{"model_name":"pool","model":"openai/one","api_key":"env:AUTHOR_NEW","api_base":"https://user:password@example.test/private?token=secret"},{"model_name":"pool","model":"openai/two","api_key":"env:AUTHOR_NEW","api_base":"https://user:password@example.test/private?token=secret"}]})));
    assert_eq!(status, 200, "{view}");
    assert_eq!(view["deployments"].as_array().unwrap().len(), 3);
    for entry in view["deployments"].as_array().unwrap() {
        assert!(entry["deployment_id"].as_str().unwrap().len() > 27);
    }
    assert_eq!(saved(dir.path())["unknown"], initial["unknown"]);
    assert_eq!(
        saved(dir.path())["llm"]["model_list"][0]["unknown_model"],
        json!({"keep":1})
    );
    let (status, error) = server.request(
        "PUT",
        "/api/settings/llm/models/pool",
        Some(json!({"model":"openai/no"})),
    );
    assert_eq!(status, 409);
    assert_eq!(error["detail"]["code"], "ambiguous_legacy_model_name");
    let (status, cards) = server.request("GET", "/api/settings/llm/providers", None);
    assert_eq!(status, 200);
    let card = cards["providers"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["kind"] == "configured" && p["model_count"] == 2)
        .unwrap();
    let provider = card["provider_id"].as_str().unwrap();
    assert_eq!(card["api_base"], "https://example.test");
    assert!(!cards.to_string().contains("AUTHOR_NEW"));
    assert!(!cards.to_string().contains("private"));
    let (status, credentials) = server.request(
        "GET",
        &format!("/api/settings/llm/providers/{provider}/credentials"),
        None,
    );
    assert_eq!(status, 200);
    assert_eq!(credentials["api_key"], "env:AUTHOR_NEW");
    assert_eq!(
        credentials["api_base"],
        "https://user:password@example.test/private?token=secret"
    );
    let (status, view) = server.request(
        "PATCH",
        &format!("/api/settings/llm/providers/{provider}"),
        Some(json!({"expected_revision":rev(&view),"api_base":null,"api_key":"env:REPLACED"})),
    );
    assert_eq!(status, 200, "{view}");
    let model_id = view["deployments"][1]["deployment_id"].as_str().unwrap();
    let (status, view) = server.request(
        "PATCH",
        &format!("/api/settings/llm/deployments/{model_id}"),
        Some(json!({"expected_revision":rev(&view),"weight":0})),
    );
    assert_eq!(status, 200);
    assert_eq!(view["deployments"][1]["weight"], 0);
    let (status, view) = server.request(
        "DELETE",
        &format!(
            "/api/settings/llm/providers/{provider}?expected_revision={}",
            rev(&view)
        ),
        None,
    );
    assert_eq!(status, 200, "{view}");
    assert_eq!(view["deployments"].as_array().unwrap().len(), 1);
    let before = view.clone();
    server.stop();
    let server = Server::start(dir.path(), &[]);
    assert_eq!(server.view(), before);
    assert_eq!(
        server
            .request("DELETE", "/api/settings/llm/models/legacy", None)
            .0,
        200
    );
    let raw = saved(dir.path());
    assert!(raw["llm"]["model_list"].as_array().unwrap().is_empty());
    assert_eq!(raw["llm"]["providers"].as_array().unwrap().len(), 1);
    server.stop();
}
#[test]
fn actual_concurrent_cas_and_invalid_batches_have_one_atomic_winner() {
    let dir = tempfile::tempdir().unwrap();
    configuration(dir.path());
    let server = Server::start(dir.path(), &[]);
    let revision = rev(&server.view());
    let old = fs::read(dir.path().join("config.json")).unwrap();
    let(status,_)=server.request("POST","/api/settings/llm/deployments/batch",Some(json!({"expected_revision":revision,"deployments":[{"model_name":"x","model":"openai/x"},{"model_name":"y","model":"openai/y","weight":-1}]})));
    assert_eq!(status, 422);
    assert_eq!(fs::read(dir.path().join("config.json")).unwrap(), old);
    let threads:Vec<_>=(0..2).map(|i|{let revision=revision.clone();let port=server.port;std::thread::spawn(move||request(port,"POST","/api/settings/llm/models",Some(json!({"model_name":format!("m{i}"),"model":"openai/added","expected_revision":revision}))))}).collect();
    let mut statuses = threads
        .into_iter()
        .map(|t| {
            let (status, body) = t.join().unwrap();
            if status == 409 {
                assert_eq!(body["detail"]["code"], "stale_revision");
                assert!(body["detail"]["current_revision"].is_string());
            }
            status
        })
        .collect::<Vec<_>>();
    statuses.sort_unstable();
    assert_eq!(statuses, vec![200, 409]);
    assert_eq!(
        saved(dir.path())["llm"]["model_list"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    let id = server.view()["deployments"][0]["deployment_id"]
        .as_str()
        .unwrap()
        .to_owned();
    assert_eq!(
        server
            .request(
                "PATCH",
                &format!("/api/settings/llm/deployments/{id}"),
                Some(json!({"api_key":null}))
            )
            .0,
        422
    );
    assert_eq!(
        server
            .request(
                "DELETE",
                &format!("/api/settings/llm/deployments/{id}"),
                None
            )
            .0,
        422
    );
    server.stop();
}
#[test]
fn configuration_failure_and_session_override_do_not_report_false_save_success() {
    let dir = tempfile::tempdir().unwrap();
    configuration(dir.path());
    let server = Server::start(dir.path(), &[]);
    let before = server.view();
    let path = dir.path().join("config.json");
    let bytes = fs::read(&path).unwrap();
    fs::remove_file(&path).unwrap();
    assert_eq!(
        server
            .request("POST", "/api/settings/llm/config/open", None)
            .0,
        404
    );
    fs::create_dir(&path).unwrap();
    assert_eq!(
        server
            .request(
                "POST",
                "/api/settings/llm/models",
                Some(json!({"model_name":"x","model":"openai/x"}))
            )
            .0,
        500
    );
    assert_eq!(server.view(), before);
    fs::remove_dir(&path).unwrap();
    fs::write(&path, &bytes).unwrap();
    server.stop();
    let server = Server::start(
        dir.path(),
        &[
            "--config-json",
            r#"{"llm":{"model_list":[{"model_name":"session","litellm_params":{"model":"openai/session"}}]}}"#,
        ],
    );
    assert_eq!(
        server
            .request(
                "POST",
                "/api/settings/llm/models",
                Some(json!({"model_name":"x","model":"openai/x"}))
            )
            .0,
        409
    );
    assert_eq!(fs::read(path).unwrap(), bytes);
    server.stop();
}
