#![cfg(unix)]
//! A connection saved through the settings API without a key never carries
//! the server's own key (a process variable or a dotenv file) to the address
//! it names: not through model discovery, the connection test or a
//! conversion. Local keyless servers keep working; the server's key still
//! goes to its own provider's endpoint from an environment card or a
//! hand-written entry; a keyed connection cannot be made keyless at another
//! host. Two loopback model servers stand for a caller's host and the
//! operator's endpoint.
#[path = "support/console.rs"]
mod console;

use serde_json::{Value, json};
use std::{
    io::{BufRead, BufReader, Read, Write},
    net::{TcpListener, TcpStream},
    path::Path,
    process::{Child, Command, Stdio},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

const WAIT: Duration = Duration::from_secs(30);
const NO_KEY: &str = "markitai-no-key";
/// Fake keys in the server's environment; none may reach the caller's host.
const ENVIRONMENT: [(&str, &str); 4] = [
    ("OPENAI_API_KEY", "sk-test-env-openai-0000"),
    ("GROQ_API_KEY", "gsk-test-env-groq-0000"),
    ("LM_STUDIO_API_KEY", "lm-test-env-lmstudio-0000"),
    ("ANTHROPIC_API_KEY", "sk-ant-test-env-0000"),
];
const DOTENV: (&str, &str) = ("MISTRAL_API_KEY", "dotenv-test-mistral-0000");

struct Seen {
    line: String,
    /// The credential headers, lowercased name and value as sent.
    credentials: Vec<String>,
}

/// Records each request's line and credential headers; answers model lists,
/// chat completions and Anthropic messages as a keyless server would.
struct ModelServer {
    base: String,
    seen: Arc<Mutex<Vec<Seen>>>,
    stop: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<()>>,
}
impl ModelServer {
    fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let seen = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let (record, finish) = (seen.clone(), stop.clone());
        let worker = thread::spawn(move || {
            while !finish.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((stream, _)) => answer(stream, &record),
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2))
                    }
                    Err(error) => panic!("model server: {error}"),
                }
            }
        });
        Self {
            base,
            seen,
            stop,
            worker: Some(worker),
        }
    }
    fn take(&self) -> Vec<Seen> {
        std::mem::take(&mut *self.seen.lock().unwrap())
    }
}
impl Drop for ModelServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(worker) = self.worker.take() {
            let result = worker.join();
            if !thread::panicking() {
                result.unwrap();
            }
        }
    }
}

fn text(content: &Value) -> String {
    match content {
        Value::String(text) => text.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|part| part["text"].as_str())
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

fn answer(stream: TcpStream, record: &Mutex<Vec<Seen>>) {
    stream.set_nonblocking(false).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut line = String::new();
    reader.read_line(&mut line).unwrap();
    let (mut length, mut credentials, mut header) = (0, Vec::new(), String::new());
    loop {
        header.clear();
        assert!(reader.read_line(&mut header).unwrap() > 0);
        if header == "\r\n" {
            break;
        }
        let (name, value) = header.split_once(':').unwrap();
        let name = name.trim().to_ascii_lowercase();
        if name == "content-length" {
            length = value.trim().parse().unwrap();
        }
        if matches!(
            name.as_str(),
            "authorization" | "x-api-key" | "api-key" | "x-goog-api-key"
        ) {
            credentials.push(format!("{name}: {}", value.trim()));
        }
    }
    assert!(length < 4 * 1024 * 1024);
    let mut body = vec![0; length];
    reader.read_exact(&mut body).unwrap();
    let line = line.trim().to_owned();
    let path = line.split_whitespace().nth(1).unwrap_or("").to_owned();
    record.lock().unwrap().push(Seen {
        line: line.clone(),
        credentials,
    });
    let path = path.split('?').next().unwrap().trim_end_matches('/');
    let reply = if line.starts_with("GET ") {
        if path.ends_with("/api/tags") {
            json!({"models":[{"name":"listed"}]})
        } else {
            json!({"data":[{"id":"listed"}]})
        }
    } else {
        let request: Value = serde_json::from_slice(&body).unwrap();
        let messages = request["messages"].as_array().cloned().unwrap_or_default();
        let system = text(&request["system"])
            + &messages
                .iter()
                .filter(|message| message["role"] == "system")
                .map(|message| text(&message["content"]))
                .collect::<String>();
        let user = messages
            .iter()
            .find(|message| message["role"] == "user")
            .map(|message| text(&message["content"]))
            .unwrap_or_default();
        let content = if system.contains("MARKITAI_DOCUMENT_JSON_V1") {
            json!({"cleaned_markdown":user,"frontmatter":{"description":"A keyless check","tags":["fixture"]}}).to_string()
        } else {
            "OK".into()
        };
        if path.ends_with("/messages") {
            json!({"id":"msg","type":"message","role":"assistant","model":request["model"],"content":[{"type":"text","text":content}],"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":10}})
        } else {
            json!({"id":"chat","object":"chat.completion","model":request["model"],"choices":[{"index":0,"message":{"role":"assistant","content":content},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":10,"total_tokens":20}})
        }
    }
    .to_string();
    let mut stream = stream;
    let _ = write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{reply}",
        reply.len()
    );
}

struct Server {
    child: Child,
    port: u16,
    log: Arc<Mutex<String>>,
    reader: Option<thread::JoinHandle<()>>,
}
impl Server {
    fn start(root: &Path, operator: &str) -> Self {
        std::fs::create_dir_all(root.join("home")).unwrap();
        std::fs::write(
            root.join("home/.env"),
            format!("{}={}\n", DOTENV.0, DOTENV.1),
        )
        .unwrap();
        let mut command = Command::new(env!("CARGO_BIN_EXE_markitai"));
        console::english(command.env_clear());
        for key in ["PATH", "TMPDIR", "SYSTEMROOT"] {
            if let Some(value) = std::env::var_os(key) {
                command.env(key, value);
            }
        }
        let mut child = command
            .current_dir(root)
            .env("HOME", root)
            .env("MARKITAI_HOME", root.join("home"))
            .env("NO_PROXY", "127.0.0.1,localhost")
            // The operator's own OpenAI endpoint, as an environment card uses it.
            .env("OPENAI_BASE_URL", format!("{operator}/v1"))
            .envs(ENVIRONMENT)
            .args(["--config", "config.json", "serve", "--host", "127.0.0.1"])
            .args(["--port", "0", "--no-open", "--no-auth"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let log = Arc::new(Mutex::new(String::new()));
        let copy = log.clone();
        let stderr = child.stderr.take().unwrap();
        let reader = thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                let mut log = copy.lock().unwrap();
                log.push_str(&line);
                log.push('\n');
            }
        });
        let deadline = Instant::now() + WAIT;
        let port = loop {
            assert!(
                child.try_wait().unwrap().is_none(),
                "{}",
                log.lock().unwrap()
            );
            if let Some(port) = log.lock().unwrap().lines().find_map(|line| {
                line.strip_prefix("Markitai server listening on http://127.0.0.1:")
                    .and_then(|port| port.parse().ok())
            }) {
                break port;
            }
            assert!(Instant::now() < deadline, "service startup timeout");
            thread::sleep(Duration::from_millis(5));
        };
        Self {
            child,
            port,
            log,
            reader: Some(reader),
        }
    }
    fn send(&self, method: &str, path: &str, content_type: &str, body: &[u8]) -> (u16, Value) {
        let mut stream = TcpStream::connect(("127.0.0.1", self.port)).unwrap();
        stream.set_read_timeout(Some(WAIT)).unwrap();
        write!(stream, "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{}\r\nConnection: close\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n\r\n", self.port, body.len()).unwrap();
        stream.write_all(body).unwrap();
        let mut raw = Vec::new();
        stream.take(4 * 1024 * 1024).read_to_end(&mut raw).unwrap();
        let split = raw.windows(4).position(|v| v == b"\r\n\r\n").unwrap();
        let head = String::from_utf8_lossy(&raw[..split]);
        let status = head.split_whitespace().nth(1).unwrap().parse().unwrap();
        let body = &raw[split + 4..];
        let value = serde_json::from_slice(body).unwrap_or(Value::Null);
        (status, value)
    }
    fn call(&self, method: &str, path: &str, body: Option<Value>) -> (u16, Value) {
        let body = body.map(|value| value.to_string()).unwrap_or_default();
        self.send(method, path, "application/json", body.as_bytes())
    }
    fn ok(&self, method: &str, path: &str, body: Option<Value>) -> Value {
        let (status, value) = self.call(method, path, body);
        assert_eq!(status, 200, "{method} {path}: {value}");
        value
    }
    fn revision(&self) -> String {
        self.ok("GET", "/api/settings/llm", None)["revision"]
            .as_str()
            .unwrap()
            .to_owned()
    }
    fn cards(&self) -> Vec<Value> {
        self.ok("GET", "/api/settings/llm/providers", None)["providers"]
            .as_array()
            .unwrap()
            .clone()
    }
    fn deployments(&self) -> Vec<Value> {
        self.ok("GET", "/api/settings/llm", None)["deployments"]
            .as_array()
            .unwrap()
            .clone()
    }
    /// Saves one deployment; its id and its saved connection's card.
    fn save(&self, deployment: Value) -> (String, Value) {
        let (models, cards) = (self.deployments(), self.cards());
        self.ok(
            "POST",
            "/api/settings/llm/deployments/batch",
            Some(json!({"expected_revision":self.revision(),"deployments":[deployment]})),
        );
        let id = self
            .deployments()
            .into_iter()
            .find(|entry| !models.contains(entry))
            .unwrap()["deployment_id"]
            .as_str()
            .unwrap()
            .to_owned();
        let card = self
            .cards()
            .into_iter()
            .find(|card| card["kind"] == "configured" && !cards.contains(card))
            .unwrap();
        (id, card)
    }
    /// Converts one small document with the model; whether the model answered.
    fn convert(&self) -> bool {
        let boundary = "keyless-check-boundary";
        let mut body = format!("--{boundary}\r\nContent-Disposition: form-data; name=\"files\"; filename=\"doc.md\"\r\nContent-Type: text/markdown\r\n\r\n# Keyless check\n\nA short synthetic document.\n\r\n").into_bytes();
        for (name, value) in [
            ("urls", "[]".to_owned()),
            (
                "options",
                json!({"preset":"minimal","llm":true,"no_cache":true}).to_string(),
            ),
        ] {
            body.extend(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n").bytes());
        }
        body.extend(format!("--{boundary}--\r\n").bytes());
        let (status, created) = self.send(
            "POST",
            "/api/jobs",
            &format!("multipart/form-data; boundary={boundary}"),
            &body,
        );
        assert_eq!(status, 201, "{created}");
        let id = created["job_id"].as_str().unwrap();
        let deadline = Instant::now() + WAIT;
        loop {
            let job = self.ok("GET", &format!("/api/jobs/{id}"), None);
            if job["status"] != "running" && job["status"] != "queued" {
                return job["items"][0]["llm_enhanced"] == true;
            }
            assert!(Instant::now() < deadline, "conversion timeout");
            thread::sleep(Duration::from_millis(20));
        }
    }
    fn remove(&self, card: &Value) {
        let id = card["provider_id"].as_str().unwrap();
        let path = format!(
            "/api/settings/llm/providers/{id}?expected_revision={}",
            self.revision()
        );
        self.ok("DELETE", &path, None);
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

fn configure(root: &Path, operator: &str, models: Value) {
    std::fs::write(
        root.join("config.json"),
        json!({"log":{"dir":null},"cache":{"enabled":false},"history":{"record":false},"ocr":{"enabled":false},
            "prompts":{"dir":root.join("prompts")},"llm":{"enabled":false,"model_list":models}})
        .to_string()
        .replace("OPERATOR", operator),
    )
    .unwrap();
}

fn secrets(seen: &[Seen]) -> Vec<String> {
    seen.iter()
        .flat_map(|seen| {
            seen.credentials
                .iter()
                .filter(|credential| {
                    ENVIRONMENT
                        .iter()
                        .chain([&DOTENV])
                        .any(|(_, secret)| credential.contains(secret))
                })
                .map(move |credential| format!("{} {credential}", seen.line))
        })
        .collect()
}

#[test]
fn a_keyless_connection_never_carries_the_servers_key_to_its_address() {
    let dir = tempfile::tempdir().unwrap();
    let (caller, operator) = (ModelServer::start(), ModelServer::start());
    // Written by hand: keyless at the operator's endpoint, so the
    // environment's key goes there.
    configure(
        dir.path(),
        &operator.base,
        json!([{"model_name":"default","litellm_params":{"model":"groq/hand","api_base":"OPERATOR/v1","weight":0},"model_info":{"id":"handwritten"}}]),
    );
    let server = Server::start(dir.path(), &operator.base);
    for (provider, model, suffix) in [
        ("openai", "openai/listed", "/v1"),
        ("groq", "groq/listed", "/v1"),
        ("lm_studio", "lm_studio/listed", "/v1"),
        ("custom", "openai/listed", "/v1"),
        ("anthropic", "anthropic/listed", ""),
        ("mistral", "mistral/listed", "/v1"),
    ] {
        let base = format!("{}{suffix}", caller.base);
        let (id, card) = server.save(
            json!({"model_name":"default","model":model,"provider":provider,"api_base":base}),
        );
        assert_eq!(card["api_key_configured"], false, "{provider}: {card}");
        let reference = card["provider_id"].as_str().unwrap();
        let credentials = server.ok(
            "GET",
            &format!("/api/settings/llm/providers/{reference}/credentials"),
            None,
        );
        assert_eq!(credentials["api_key"], Value::Null, "{provider}");
        assert_eq!(credentials["api_key_source"], Value::Null, "{provider}");
        let found = server.ok(
            "POST",
            "/api/settings/llm/model-discovery",
            Some(json!({"provider":card["provider"],"provider_id":reference,"refresh":true})),
        );
        assert_eq!(
            found["models"].as_array().unwrap().len(),
            1,
            "{provider}: {found}"
        );
        let tested = server.ok(
            "POST",
            "/api/settings/llm/test",
            Some(json!({"deployment_id":id})),
        );
        assert_eq!(tested["ok"], true, "{provider}: {tested}");
        let draft = server.ok(
            "POST",
            "/api/settings/llm/test",
            Some(json!({"model":model,"api_base":base})),
        );
        assert_eq!(draft["ok"], true, "{provider}: {draft}");
        // A keyless endpoint converts: local servers, an OpenAI-compatible
        // endpoint and any other address that checks no key.
        assert!(
            server.convert(),
            "{provider}: {}",
            server.log.lock().unwrap()
        );
        let seen = caller.take();
        assert!(seen.len() >= 4, "{provider}: {} requests", seen.len());
        assert_eq!(secrets(&seen), Vec::<String>::new(), "{provider}");
        for request in &seen {
            // Discovery sends no key; the rest send the placeholder.
            let expected: &[&str] = if request.line.starts_with("GET ") {
                &[]
            } else if provider == "anthropic" {
                &["x-api-key: markitai-no-key"]
            } else {
                &["authorization: Bearer markitai-no-key"]
            };
            assert_eq!(
                request.credentials, expected,
                "{provider}: {}",
                request.line
            );
        }
        server.remove(&card);
    }
    assert!(operator.take().is_empty());

    // The environment card: the server's own key at its own provider's
    // endpoint (here the operator's OPENAI_BASE_URL), never at another.
    let (status, refused) = server.call(
        "POST",
        "/api/settings/llm/deployments/batch",
        Some(json!({"expected_revision":server.revision(),"deployments":[
            {"model_name":"default","model":"openai/listed","credential_provider_id":"env:openai","api_base":format!("{}/v1",caller.base)}]})),
    );
    assert_eq!(
        (status, &refused["reason"]),
        (422, &json!("invalid_settings_request"))
    );
    server.ok(
        "POST",
        "/api/settings/llm/deployments/batch",
        Some(json!({"expected_revision":server.revision(),"deployments":[
            {"model_name":"default","model":"openai/listed","credential_provider_id":"env:openai"}]})),
    );
    let id = server
        .deployments()
        .into_iter()
        .find(|entry| entry["model"] == "openai/listed")
        .unwrap()["deployment_id"]
        .as_str()
        .unwrap()
        .to_owned();
    assert_eq!(
        server.ok(
            "POST",
            "/api/settings/llm/test",
            Some(json!({"deployment_id":id}))
        )["ok"],
        true
    );
    assert!(server.convert());
    let seen = operator.take();
    assert!(seen.len() >= 2);
    for request in &seen {
        assert_eq!(
            request.credentials,
            ["authorization: Bearer sk-test-env-openai-0000"],
            "{}",
            request.line
        );
    }
    // Moving it elsewhere without a key is refused; clearing and moving it
    // leaves it keyless there.
    let (status, _) = server.call(
        "PATCH",
        &format!("/api/settings/llm/deployments/{id}"),
        Some(
            json!({"expected_revision":server.revision(),"api_base":format!("{}/v1",caller.base)}),
        ),
    );
    assert_eq!(status, 422);
    server.ok(
        "PATCH",
        &format!("/api/settings/llm/deployments/{id}"),
        Some(json!({"expected_revision":server.revision(),"api_key":null,"api_base":format!("{}/v1",caller.base)})),
    );
    assert_eq!(
        server.ok(
            "POST",
            "/api/settings/llm/test",
            Some(json!({"deployment_id":id}))
        )["ok"],
        true
    );
    let seen = caller.take();
    assert_eq!(seen.len(), 1);
    assert_eq!(
        seen[0].credentials,
        ["authorization: Bearer markitai-no-key"]
    );
    let path = format!(
        "/api/settings/llm/deployments/{id}?expected_revision={}",
        server.revision()
    );
    server.ok("DELETE", &path, None);
    // The hand-written entry, after an unrelated edit through the API.
    server.ok(
        "PATCH",
        "/api/settings/llm/deployments/handwritten",
        Some(json!({"expected_revision":server.revision(),"weight":1})),
    );
    assert_eq!(
        server.ok(
            "POST",
            "/api/settings/llm/test",
            Some(json!({"deployment_id":"handwritten"}))
        )["ok"],
        true
    );
    let seen = operator.take();
    assert_eq!(seen.len(), 1);
    assert_eq!(
        seen[0].credentials,
        ["authorization: Bearer gsk-test-env-groq-0000"]
    );

    // A connection with a key cannot become keyless at another host in one
    // change, through the connection or through its deployment.
    let (id, card) = server.save(
        json!({"model_name":"keyed","model":"openai/listed","provider":"openai",
        "api_key":"sk-literal-test-key-0000","api_base":format!("{}/v1",operator.base)}),
    );
    let reference = card["provider_id"].as_str().unwrap();
    for path in [
        format!("/api/settings/llm/providers/{reference}"),
        format!("/api/settings/llm/deployments/{id}"),
    ] {
        let (status, error) = server.call(
            "PATCH",
            &path,
            Some(json!({"expected_revision":server.revision(),"api_key":null,"api_base":format!("{}/v1",caller.base)})),
        );
        assert_eq!(status, 422, "{path}: {error}");
        assert_eq!(error["reason"], "invalid_settings_request");
        assert_eq!(
            error["detail"],
            "a connection with a key cannot become one without a key at another address; add a new connection for that server instead"
        );
    }
    assert!(caller.take().is_empty());
    assert!(!server.log.lock().unwrap().contains("test-env"));
    assert!(!server.log.lock().unwrap().contains(NO_KEY));
}

#[test]
fn routability_and_cards_follow_what_a_connection_will_send() {
    let dir = tempfile::tempdir().unwrap();
    let operator = ModelServer::start();
    // Saved by the settings API without a key at the provider's own
    // endpoint: it sends no key, so the server's OPENAI_API_KEY does not make
    // it routable.
    configure(
        dir.path(),
        &operator.base,
        json!([{"model_name":"default","litellm_params":{"model":"openai/web","use_environment_credentials":false},"model_info":{"id":"web"}}]),
    );
    let server = Server::start(dir.path(), &operator.base);
    let view = server.ok("GET", "/api/settings/llm", None);
    assert_eq!(view["routable"], false, "{view}");
    drop(server);
    // Python markitai 1.2.1's keyless entry reads as no key, and routes.
    configure(
        dir.path(),
        &operator.base,
        json!([{"model_name":"default","litellm_params":{"model":"openai/python","api_key":NO_KEY,"api_base":"OPERATOR/v1"},"model_info":{"id":"python"}}]),
    );
    let server = Server::start(dir.path(), &operator.base);
    let view = server.ok("GET", "/api/settings/llm", None);
    assert_eq!(view["routable"], true, "{view}");
    assert_eq!(view["deployments"][0]["api_key_configured"], false);
    let card = server
        .cards()
        .into_iter()
        .find(|card| card["deployment_id"] == "python")
        .unwrap();
    assert_eq!(card["api_key_configured"], false);
    let credentials = server.ok(
        "GET",
        &format!(
            "/api/settings/llm/providers/{}/credentials",
            card["provider_id"].as_str().unwrap()
        ),
        None,
    );
    assert_eq!(credentials["api_key_masked"], Value::Null);
    assert!(server.convert());
    let seen = operator.take();
    assert_eq!(seen.len(), 1);
    assert_eq!(
        seen[0].credentials,
        ["authorization: Bearer markitai-no-key"]
    );
}

#[test]
fn cards_name_the_environment_key_a_connection_authenticates_with() {
    let dir = tempfile::tempdir().unwrap();
    let operator = ModelServer::start();
    configure(
        dir.path(),
        &operator.base,
        json!([
            // Hand-written without a key, at the provider's own endpoint.
            {"model_name":"default","litellm_params":{"model":"groq/own"},"model_info":{"id":"groq-own"}},
            // Hand-written without a key, at an address of the operator's own.
            {"model_name":"default","litellm_params":{"model":"anthropic/proxied","api_base":"OPERATOR/v1"},"model_info":{"id":"anthropic-proxied"}},
            // A reference, a literal key, the keyless placeholder, and no environment.
            {"model_name":"default","litellm_params":{"model":"deepseek/referenced","api_key":"env:DEEPSEEK_TEST_UNSET"},"model_info":{"id":"deepseek-referenced"}},
            {"model_name":"default","litellm_params":{"model":"xai/literal","api_key":"xai-literal-test"},"model_info":{"id":"xai-literal"}},
            {"model_name":"default","litellm_params":{"model":"cerebras/keyless","api_key":NO_KEY,"api_base":"OPERATOR/v1"},"model_info":{"id":"cerebras-keyless"}},
            {"model_name":"default","litellm_params":{"model":"mistral/literal-mode","use_environment_credentials":false},"model_info":{"id":"mistral-literal"}}
        ]),
    );
    let server = Server::start(dir.path(), &operator.base);
    let cards = server.cards();
    let card = |deployment: &str| {
        cards
            .iter()
            .find(|card| card["deployment_id"] == deployment)
            .unwrap_or_else(|| panic!("{deployment}: {cards:?}"))
            .clone()
    };
    assert_eq!(card("groq-own")["credential"], "env:GROQ_API_KEY");
    assert_eq!(card("groq-own")["api_key_configured"], false);
    assert_eq!(
        card("anthropic-proxied")["credential"],
        "env:ANTHROPIC_API_KEY"
    );
    assert_eq!(
        card("deepseek-referenced")["credential"],
        "env:DEEPSEEK_TEST_UNSET"
    );
    for deployment in ["xai-literal", "cerebras-keyless", "mistral-literal"] {
        assert!(card(deployment).get("credential").is_none(), "{deployment}");
    }
    // The hand-written groq entry at its own endpoint stands for the
    // environment card; the anthropic one elsewhere does not.
    let environment: Vec<_> = cards
        .iter()
        .filter(|card| card["kind"] == "environment")
        .map(|card| card["provider"].as_str().unwrap())
        .collect();
    assert!(!environment.contains(&"groq"), "{environment:?}");
    assert!(environment.contains(&"anthropic"), "{environment:?}");
    // Its credentials name the variable, never the key.
    let credentials = server.ok(
        "GET",
        &format!(
            "/api/settings/llm/providers/{}/credentials",
            card("groq-own")["provider_id"].as_str().unwrap()
        ),
        None,
    );
    assert_eq!(credentials["api_key_masked"], "env:GROQ_API_KEY");
    assert_eq!(credentials["api_key_source"], "environment");
    assert!(!credentials.to_string().contains("gsk-test"));
    // A hand-written entry discovers with the environment's key.
    let found = server.ok(
        "POST",
        "/api/settings/llm/model-discovery",
        Some(json!({"provider":"anthropic","deployment_id":"anthropic-proxied","refresh":true})),
    );
    assert_eq!(found["status"], "ok", "{found}");
    let seen = operator.take();
    assert_eq!(seen.len(), 1);
    assert!(
        seen[0].line.starts_with("GET /v1/models"),
        "{}",
        seen[0].line
    );
    assert_eq!(seen[0].credentials, ["x-api-key: sk-ant-test-env-0000"]);
}
