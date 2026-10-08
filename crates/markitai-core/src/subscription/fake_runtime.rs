//! Authored stand-ins for the official Copilot, Claude and Codex runtimes,
//! played by the test binary itself. They never contact a service, read a
//! real login or need an interpreter.
//!
//! An adapter chooses its runtime's arguments and clears its environment, so
//! neither can tell a re-executed test binary which runtime to play. A
//! fixture instead writes [`MARKER`] into the runtime's home directory
//! (`COPILOT_HOME`, `CLAUDE_CONFIG_DIR` or `CODEX_HOME`, which the adapters
//! keep) and points the adapter at [`program`]. A static initializer runs
//! before the test harness reads its arguments: when the marker is present
//! it plays the runtime whose home holds it and exits, so the harness never
//! starts. A missing marker costs one failed file open per variable.
//!
//! Integration tests include this file by path, so it uses only the standard
//! library, `serde_json` and `sha2`.
#![allow(dead_code)]

use serde_json::{Value, json};
use std::io::{BufRead, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// The scenario file a fixture writes into the runtime's home.
pub const MARKER: &str = "markitai-fake-runtime.json";
/// Set for a descendant that only sleeps for the given number of seconds.
const SLEEP: &str = "MARKITAI_FAKE_RUNTIME_SLEEP";

/// The program an adapter starts to reach a fake runtime.
pub fn program() -> PathBuf {
    std::env::current_exe().unwrap()
}

/// Make `home` the home of a fake runtime playing `scenario`. Rewriting it
/// changes the scenario for the next start.
pub fn install(home: &Path, scenario: &Value) {
    std::fs::create_dir_all(home).unwrap();
    write_json(&home.join(MARKER), scenario);
}

#[derive(Clone, Copy)]
enum Runtime {
    Copilot,
    Claude,
    Codex,
}

#[used]
#[cfg_attr(
    target_vendor = "apple",
    unsafe(link_section = "__DATA,__mod_init_func")
)]
#[cfg_attr(
    all(unix, not(target_vendor = "apple")),
    unsafe(link_section = ".init_array")
)]
#[cfg_attr(windows, unsafe(link_section = ".CRT$XCU"))]
static PLAY: extern "C" fn() = play;

extern "C" fn play() {
    if let Some(seconds) = std::env::var_os(SLEEP) {
        let seconds = seconds.to_str().and_then(|s| s.parse().ok()).unwrap_or(60);
        std::thread::sleep(Duration::from_secs(seconds));
        std::process::exit(0);
    }
    let Some((runtime, home, scenario)) = detect() else {
        return;
    };
    let code = std::panic::catch_unwind(|| match runtime {
        Runtime::Copilot => copilot(&home, &scenario),
        Runtime::Claude => claude(&home, &scenario),
        Runtime::Codex => codex(&home, &scenario),
    })
    // A failed fixture assertion has already printed its message.
    .unwrap_or(1);
    let _ = std::io::stdout().flush();
    std::process::exit(code);
}

fn detect() -> Option<(Runtime, PathBuf, Value)> {
    for (variable, runtime) in [
        ("COPILOT_HOME", Runtime::Copilot),
        ("CLAUDE_CONFIG_DIR", Runtime::Claude),
        ("CODEX_HOME", Runtime::Codex),
    ] {
        let Some(home) = std::env::var_os(variable) else {
            continue;
        };
        let home = PathBuf::from(home);
        let Ok(bytes) = std::fs::read(home.join(MARKER)) else {
            continue;
        };
        let Ok(scenario) = serde_json::from_slice(&bytes) else {
            eprintln!("malformed fake runtime scenario in {}", home.display());
            std::process::exit(2);
        };
        return Some((runtime, home, scenario));
    }
    None
}

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

fn arguments() -> Vec<String> {
    std::env::args().skip(1).collect()
}

fn after<'a>(args: &'a [String], flag: &str) -> &'a str {
    let index = args
        .iter()
        .position(|arg| arg == flag)
        .unwrap_or_else(|| panic!("{flag} is missing"));
    args.get(index + 1)
        .unwrap_or_else(|| panic!("{flag} has no value"))
}

fn working_directory() -> PathBuf {
    std::env::current_dir().unwrap()
}

fn same_file(left: &Path, right: &Path) -> bool {
    left.canonicalize().unwrap() == right.canonicalize().unwrap()
}

#[cfg(unix)]
fn permissions(path: &Path) -> u32 {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(path).unwrap().permissions().mode() & 0o777
}

/// Written whole under its final name, so a polling test never reads a part.
fn write_json(path: &Path, value: &Value) {
    write_text(path, &value.to_string());
}

fn write_text(path: &Path, text: &str) {
    let staging = path.with_extension("partial");
    std::fs::write(&staging, text).unwrap();
    std::fs::rename(staging, path).unwrap();
}

fn append(path: &Path, value: &Value) {
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .unwrap();
    writeln!(file, "{value}").unwrap();
}

fn emit(value: &Value) {
    let mut out = std::io::stdout().lock();
    writeln!(out, "{value}").unwrap();
    out.flush().unwrap();
}

fn stderr(text: &str) {
    let mut err = std::io::stderr().lock();
    let _ = err.write_all(text.as_bytes());
    let _ = err.flush();
}

/// Start a descendant that inherits this runtime's output pipes and sleeps,
/// as a tool subprocess would; returns its process id.
fn sleeper(detached_input: bool) -> u32 {
    let mut command = std::process::Command::new(program());
    command.env(SLEEP, "60");
    if detached_input {
        command.stdin(std::process::Stdio::null());
    }
    command.spawn().unwrap().id()
}

fn close_stdout() {
    #[cfg(unix)]
    {
        use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
        let _ = std::io::stdout().flush();
        drop(unsafe { OwnedFd::from_raw_fd(std::io::stdout().as_raw_fd()) });
    }
    #[cfg(windows)]
    {
        use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
        let _ = std::io::stdout().flush();
        drop(unsafe { OwnedHandle::from_raw_handle(std::io::stdout().as_raw_handle()) });
    }
}

fn sha256(bytes: &[u8]) -> String {
    use sha2::Digest;
    sha2::Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Strict standard base64 with padding.
fn base64(text: &str) -> Option<Vec<u8>> {
    fn value(byte: u8) -> Option<u32> {
        Some(match byte {
            b'A'..=b'Z' => byte - b'A',
            b'a'..=b'z' => byte - b'a' + 26,
            b'0'..=b'9' => byte - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return None,
        } as u32)
    }
    let bytes = text.as_bytes();
    if !bytes.len().is_multiple_of(4) {
        return None;
    }
    let mut output = Vec::with_capacity(bytes.len() / 4 * 3);
    for (index, chunk) in bytes.chunks(4).enumerate() {
        let last = index + 1 == bytes.len() / 4;
        let padding = chunk.iter().rev().take_while(|&&byte| byte == b'=').count();
        if padding > 2 || (padding > 0 && !last) {
            return None;
        }
        let mut word = 0;
        for &byte in &chunk[..4 - padding] {
            word = word << 6 | value(byte)?;
        }
        word <<= 6 * padding as u32;
        let decoded = [(word >> 16) as u8, (word >> 8) as u8, word as u8];
        output.extend_from_slice(&decoded[..3 - padding]);
    }
    Some(output)
}

/// GitHub Copilot's headless JSON-RPC runtime over Content-Length frames.
fn copilot(home: &Path, scenario: &Value) -> i32 {
    let args = arguments();
    let mode = scenario["mode"].as_str().unwrap_or("normal");
    for key in ["OPENAI_API_KEY", "COPILOT_PROVIDER_API_KEY"] {
        assert!(env(key).is_none(), "{key} reached the runtime");
    }
    assert_eq!(env("HOME").as_deref(), scenario["home"].as_str());
    if args == ["login"] {
        assert_eq!(
            env("COPILOT_CACHE_HOME").as_deref(),
            scenario["cache_home"].as_str()
        );
        write_json(
            &home.join("login.json"),
            &json!({"pid":std::process::id(),"args":args,"cache_home":env("COPILOT_CACHE_HOME")}),
        );
        return scenario["login_exit"].as_i64().unwrap_or(0) as i32;
    }
    if let Some(cache) = scenario["cache_home"].as_str() {
        assert_eq!(env("COPILOT_CACHE_HOME").as_deref(), Some(cache));
        std::fs::create_dir_all(cache).unwrap();
        std::fs::write(
            Path::new(cache).join("fixture-cache-access"),
            "private cache preserved",
        )
        .unwrap();
    }
    let cwd = working_directory();
    assert_ne!(
        Some(cwd.to_string_lossy().as_ref()),
        scenario["home"].as_str()
    );
    #[cfg(unix)]
    {
        assert_eq!(permissions(&cwd), 0o700);
        std::fs::write(home.join("cwd-mode"), format!("0o{:o}", permissions(&cwd))).unwrap();
    }
    for flag in ["--headless", "--stdio", "--no-auto-update"] {
        assert!(args.iter().any(|arg| arg == flag), "{flag} is missing");
    }
    std::fs::write(home.join("pid"), std::process::id().to_string()).unwrap();
    std::fs::write(home.join("cwd"), cwd.to_string_lossy().as_bytes()).unwrap();
    stderr(&"credential-must-never-appear-in-error\n".repeat(256));
    if mode == "stderr-flood" {
        // More than the 1 MiB stderr bound, then silence until killed.
        stderr(&"secret".repeat(200_000));
        std::thread::sleep(Duration::from_secs(60));
        return 0;
    }

    let send = |value: Value| {
        let data = serde_json::to_vec(&value).unwrap();
        let mut packet = format!("Content-Length: {}\r\n\r\n", data.len()).into_bytes();
        packet.extend(data);
        let mut out = std::io::stdout().lock();
        if mode == "split" {
            for piece in packet.chunks(3) {
                out.write_all(piece).unwrap();
                out.flush().unwrap();
            }
        } else {
            out.write_all(&packet).unwrap();
            out.flush().unwrap();
        }
    };
    let respond = |request: &Value, result: Value| {
        send(json!({"jsonrpc":"2.0","id":request["id"],"result":result}));
    };
    let event = |kind: &str, data: Value, id: Option<&str>, session: &str, agent: Option<&str>| {
        let mut item = json!({"id":id.unwrap_or(kind),"type":kind,"data":data,
            "parentId":null,"timestamp":"2026-09-29T00:00:00Z"});
        if let Some(agent) = agent {
            item["agentId"] = json!(agent);
        }
        send(
            json!({"jsonrpc":"2.0","method":"session.event","params":{"sessionId":session,"event":item}}),
        );
    };
    let usage = |tokens: u64, id: &str, agent: Option<&str>| {
        let mut payload = json!({"apiCallId":"api-1",
            "model":if mode == "wrong_model" { "wrong-model" } else { "fixture" },
            "inputTokens":tokens,"outputTokens":3,"cost":12.5,
            "finishReason":if mode == "length" { "length" } else { "stop" }});
        if mode == "byok" {
            payload["isByok"] = json!(true);
        }
        event(
            "assistant.usage",
            payload,
            Some(id),
            "session-fixture",
            agent,
        );
    };
    let session_event =
        |kind: &str, data: Value, id: Option<&str>| event(kind, data, id, "session-fixture", None);

    let mut input = std::io::stdin().lock();
    loop {
        let mut length = None;
        loop {
            let mut line = Vec::new();
            if input.read_until(b'\n', &mut line).unwrap() == 0 {
                return 0;
            }
            if line == b"\r\n" {
                break;
            }
            let line = String::from_utf8(line).unwrap();
            let (name, value) = line.split_once(':').unwrap();
            if name.eq_ignore_ascii_case("content-length") {
                length = Some(value.trim().parse::<usize>().unwrap());
            }
        }
        let mut body = vec![0; length.unwrap()];
        input.read_exact(&mut body).unwrap();
        let request: Value = serde_json::from_slice(&body).unwrap();
        append(&home.join("requests.jsonl"), &request);
        match request["method"].as_str() {
            Some("connect") => {
                let client = &request["params"]["clientInfo"];
                let known = [
                    "editorName",
                    "editorVersion",
                    "extensionName",
                    "extensionVersion",
                ];
                if client
                    .as_object()
                    .is_some_and(|fields| fields.keys().any(|key| !known.contains(&key.as_str())))
                {
                    send(
                        json!({"jsonrpc":"2.0","id":request["id"],"error":{"code":-32602,
                        "message":"Invalid connect request: unknown clientInfo field"}}),
                    );
                    continue;
                }
                assert_eq!(client["editorName"], "markitai");
                assert!(
                    client["editorVersion"]
                        .as_str()
                        .is_some_and(|version| !version.is_empty())
                );
                assert_eq!(request["params"]["supportedTaskKinds"], json!([]));
                respond(
                    &request,
                    json!({"protocolVersion":if mode == "wrong_protocol" { 2 } else { 3 }}),
                );
            }
            Some("status.get") => {
                respond(&request, json!({"version":"1.0.90-2","protocolVersion":3}));
            }
            Some("auth.getStatus") => respond(
                &request,
                json!({"isAuthenticated":mode != "unauth","login":"fixture-user",
                    "authType":"env","host":"https://github.com"}),
            ),
            Some("models.list") => respond(
                &request,
                json!({"models":[
                    {"id":"z","name":"Zed","policy":{"state":"enabled"},"capabilities":{"supports":{"vision":true}}},
                    {"id":"hidden","policy":{"state":"disabled"}},
                    {"id":"a","name":"Alpha"}]}),
            ),
            Some("session.create") => {
                let params = &request["params"];
                assert_eq!(params["availableTools"], json!([]));
                assert_eq!(params["tools"], json!([]));
                assert_eq!(params["mcpServers"], json!({}));
                for key in [
                    "enableConfigDiscovery",
                    "enableFileHooks",
                    "enableHostGitOperations",
                    "enableSessionStore",
                    "enableSkills",
                ] {
                    assert_eq!(params[key], json!(false), "{key}");
                }
                assert_eq!(
                    params["systemMessage"],
                    json!({"mode":"replace","content":"fixed system"})
                );
                assert!(same_file(
                    Path::new(params["workingDirectory"].as_str().unwrap()),
                    &cwd
                ));
                session_event(
                    "session.start",
                    json!({"context":{"cwd":cwd.to_string_lossy()}}),
                    None,
                );
                respond(&request, json!({"sessionId":"session-fixture"}));
            }
            Some("session.send") => {
                let params = &request["params"];
                assert_eq!(params["sessionId"], "session-fixture");
                for image in params["attachments"].as_array().unwrap() {
                    assert_eq!(image["type"], "blob");
                    assert_eq!(image["mimeType"], "image/png");
                    assert_eq!(
                        base64(image["data"].as_str().unwrap()).unwrap(),
                        b"\x89PNG\r\n\x1a\nfixture"
                    );
                }
                match mode {
                    "hang" => {
                        let pid = sleeper(false);
                        write_text(&home.join("descendant"), &pid.to_string());
                        std::thread::sleep(Duration::from_secs(60));
                    }
                    "oversized" => {
                        let mut out = std::io::stdout().lock();
                        out.write_all(b"Content-Length: 99999999999\r\n\r\n")
                            .unwrap();
                        out.flush().unwrap();
                        drop(out);
                        std::thread::sleep(Duration::from_secs(60));
                    }
                    "callback" => {
                        send(
                            json!({"jsonrpc":"2.0","id":"permission-id","method":"permission.request",
                            "params":{"secret":"credential-must-never-appear-in-error"}}),
                        );
                        std::thread::sleep(Duration::from_secs(60));
                    }
                    "permission" => {
                        session_event(
                            "permission.requested",
                            json!({"requestId":"blocked","permissionRequest":{"kind":"shell"}}),
                            None,
                        );
                        std::thread::sleep(Duration::from_secs(60));
                    }
                    _ => {}
                }
                if mode != "no_usage" {
                    usage(7, "u1", (mode == "subagent").then_some("unexpected-agent"));
                }
                if mode == "duplicate" {
                    usage(7, "u1", None);
                    usage(7, "u2", None);
                }
                if mode == "conflict" {
                    usage(999, "u2", None);
                }
                if mode == "paid_error" {
                    session_event(
                        "session.error",
                        json!({"errorType":"authentication","message":"credential-must-never-appear-in-error"}),
                        None,
                    );
                    std::thread::sleep(Duration::from_secs(60));
                }
                if mode == "eof" {
                    return 0;
                }
                event(
                    "assistant.message",
                    json!({"messageId":"unrelated","content":"foreign"}),
                    None,
                    "other-session",
                    None,
                );
                let text = params["prompt"].as_str().unwrap();
                if mode == "split" {
                    // Split by characters, never inside one.
                    let middle = text
                        .char_indices()
                        .nth(text.chars().count() / 2)
                        .map_or(text.len(), |(index, _)| index);
                    for index in [1, 0] {
                        let id = format!("part{index}");
                        let content = if index == 0 {
                            &text[..middle]
                        } else {
                            &text[middle..]
                        };
                        session_event(
                            "assistant.message",
                            json!({"messageId":id,"apiCallId":"api-1","chunkIndex":index,
                                "chunkCount":2,"content":content,"originatingMessageId":"message-1"}),
                            Some(&id),
                        );
                    }
                } else if mode != "empty" {
                    session_event(
                        "assistant.message",
                        json!({"messageId":"answer","content":text,"originatingMessageId":"message-1"}),
                        None,
                    );
                }
                session_event("session.idle", json!({"aborted":mode == "aborted"}), None);
                respond(&request, json!({"messageId":"message-1"}));
            }
            _ => send(json!({"jsonrpc":"2.0","id":request["id"],
                "error":{"code":-32601,"message":"unknown fixture operation"}})),
        }
    }
}

/// Claude Code: version and status probes, the login hand-off, and one
/// stream-JSON turn.
fn claude(home: &Path, scenario: &Value) -> i32 {
    let args = arguments();
    let mode = scenario["mode"].as_str().unwrap_or("ok");
    let record = home.join("calls.jsonl");
    let cwd = working_directory();
    #[cfg_attr(not(unix), allow(unused_mut))]
    let mut row = json!({"args":args,"cwd":cwd.to_string_lossy(),"pid":std::process::id(),"home":env("HOME")});
    #[cfg(unix)]
    {
        row["mode"] = json!(permissions(&cwd));
    }
    append(&record, &row);
    if let Some(expected) = scenario.get("home") {
        assert_eq!(env("HOME").as_deref(), expected.as_str());
    }
    for key in [
        "ANTHROPIC_API_KEY",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "OPENAI_API_KEY",
        "COPILOT_GITHUB_TOKEN",
    ] {
        assert!(env(key).is_none(), "{key} reached the runtime");
    }
    if args == ["auth", "login"] {
        write_json(
            &home.join("claude-login.json"),
            &json!({"pid":std::process::id(),"home":env("HOME")}),
        );
        return scenario["login_exit"].as_i64().unwrap_or(0) as i32;
    }
    #[cfg(unix)]
    assert_eq!(permissions(&cwd), 0o700);
    if args == ["--version"] {
        println!(
            "{}",
            if mode == "wrong-version" {
                "2.1.283 (Claude Code)"
            } else {
                "2.1.284 (Claude Code)"
            }
        );
        return 0;
    }
    if args == ["auth", "status"] {
        let signed_in = mode != "signed-out";
        let status = json!({"loggedIn":signed_in,
            "authMethod":if matches!(mode, "byok-status" | "byok") { "apiKey" } else { "claude.ai" },
            "apiProvider":"firstParty","email":"fixture@example.invalid","subscriptionType":"pro",
            "unexpected_token":"do-not-echo","token":"fixture-secret-never-exposed"});
        println!("{}", serde_json::to_string_pretty(&status).unwrap());
        return if signed_in { 0 } else { 1 };
    }
    for flag in [
        "--print",
        "--safe-mode",
        "--restricted",
        "--tools",
        "--disallowedTools",
        "--permission-prompts",
        "--strict-mcp-config",
        "--setting-sources=",
        "--no-session-persistence",
        "--no-chrome",
        "--disable-slash-commands",
    ] {
        assert!(args.iter().any(|arg| arg == flag), "{flag} is missing");
    }
    assert_eq!(after(&args, "--tools"), "");
    assert_eq!(after(&args, "--disallowedTools"), "*");
    assert_eq!(after(&args, "--permission-prompts"), "none");
    assert_eq!(after(&args, "--max-turns"), "1");
    for refused in ["--bare", "--dangerously-skip-permissions"] {
        assert!(!args.iter().any(|arg| arg == refused), "{refused}");
    }
    for flag in ["--system-prompt-file", "--settings", "--mcp-config"] {
        let path = Path::new(after(&args, flag));
        assert!(same_file(path.parent().unwrap(), &cwd), "{flag}");
        #[cfg(unix)]
        assert_eq!(permissions(path), 0o600, "{flag}");
    }
    let system = std::fs::read_to_string(after(&args, "--system-prompt-file")).unwrap();
    let mcp: Value =
        serde_json::from_slice(&std::fs::read(after(&args, "--mcp-config")).unwrap()).unwrap();
    assert_eq!(mcp, json!({"mcpServers":{}}));

    let mut input = std::io::stdin().lock();
    let mut line = String::new();
    input.read_line(&mut line).unwrap();
    let first: Value = serde_json::from_str(&line).unwrap();
    assert_eq!(
        first,
        json!({"type":"control_request","request_id":"markitai-initialize",
            "request":{"subtype":"initialize","hooks":null,"agents":{},"skills":[]}})
    );
    match mode {
        "callback-init" => {
            emit(
                &json!({"type":"control_request","request_id":"unexpected","request":{"subtype":"can_use_tool"}}),
            );
            std::thread::sleep(Duration::from_secs(30));
        }
        "hang-init" => std::thread::sleep(Duration::from_secs(30)),
        "stderr-flood" => {
            stderr(&"secret".repeat(200_000));
            std::thread::sleep(Duration::from_secs(30));
        }
        _ => {}
    }
    let account = json!({"apiProvider":if mode == "wrong-provider" { "bedrock" } else { "firstParty" },
        "subscriptionType":"pro","apiKeySource":"none","email":"fixture@example.invalid"});
    emit(
        &json!({"type":"control_response","response":{"subtype":"success",
        "request_id":"markitai-initialize","response":{"models":[{"value":"sonnet",
        "resolvedModel":"claude-fixture-1","displayName":"Fixture Claude",
        "description":"local only"}],"account":account}}}),
    );
    let mut line = String::new();
    if input.read_line(&mut line).unwrap() == 0 {
        return 0;
    }
    let user: Value = serde_json::from_str(&line).unwrap();
    assert_eq!(user["type"], "user");
    assert_eq!(user["message"]["role"], "user");
    assert!(user["parent_tool_use_id"].is_null());
    assert_eq!(user["session_id"], "");
    let content = user["message"]["content"].as_array().unwrap();
    assert_eq!(content[0]["type"], "text");
    let images: Vec<Value> = content[1..]
        .iter()
        .map(|part| {
            assert_eq!(part["type"], "image");
            assert_eq!(part["source"]["type"], "base64");
            let bytes = base64(part["source"]["data"].as_str().unwrap()).unwrap();
            json!({"mime":part["source"]["media_type"],"sha256":sha256(&bytes)})
        })
        .collect();
    let text = content[0]["text"].as_str().unwrap();
    append(
        &record,
        &json!({"request":{"system":system,"text":text,"images":images}}),
    );
    let mut rest = String::new();
    input.read_to_string(&mut rest).unwrap();
    assert_eq!(rest, "");

    let mut initial = json!({"type":"system","subtype":"init","session_id":"fixture-session",
        "claude_code_version":"2.1.284","apiKeySource":"none","tools":[],"mcp_servers":[],
        "agents":[],"skills":[],"plugins":[],"permissionMode":"dontAsk"});
    if mode.starts_with("catalog-") {
        initial["agents"] = json!(["claude", "Explore", "general-purpose", "Plan"]);
    }
    if mode == "catalog-custom" {
        initial["agents"][3] = json!("Unrequested custom agent");
    }
    if mode == "catalog-agent-tool" {
        initial["tools"] = json!(["Agent"]);
    }
    emit(&initial);
    match mode {
        "catalog-task" => emit(
            &json!({"type":"system","subtype":"task_started","session_id":"fixture-session","task_id":"forbidden"}),
        ),
        "catalog-user-message" => emit(
            &json!({"type":"user","session_id":"fixture-session","message":{"role":"user","content":"unrequested"}}),
        ),
        "catalog-callback" => emit(
            &json!({"type":"control_request","request_id":"forbidden","request":{"subtype":"can_use_tool"}}),
        ),
        _ => {}
    }
    emit(
        &json!({"type":"system","subtype":"session_state_changed","state":"running","session_id":"fixture-session"}),
    );
    let mut usage = json!({"input_tokens":11,"output_tokens":7,"cache_read_input_tokens":3,
        "cache_creation_input_tokens":2});
    let mut assistant = json!({"type":"assistant","session_id":"fixture-session",
        "parent_tool_use_id":null,"message":{"id":"msg-one","model":"claude-fixture-1",
        "content":[{"type":"text","text":"partial is not final"}],"usage":usage,
        "stop_reason":"end_turn"}});
    match mode {
        "tool" => {
            assistant["message"]["content"] = json!([{"type":"tool_use","id":"tool-one",
                "name":"Bash","input":{"command":"forbidden"}}]);
        }
        "catalog-subagent" => assistant["parent_tool_use_id"] = json!("forbidden"),
        "wrong-model" => assistant["message"]["model"] = json!("unrequested-model"),
        _ => {}
    }
    if !matches!(mode, "aggregate-only" | "unknown-usage") {
        emit(&assistant);
    }
    if mode == "duplicate" {
        emit(&assistant);
    }
    if mode == "conflict" {
        // The repeated message changes its count; the turn's totals follow.
        usage["output_tokens"] = json!(2);
        assistant["message"]["usage"] = usage.clone();
        emit(&assistant);
    }
    match mode {
        "paid-callback" => {
            emit(
                &json!({"type":"control_request","request_id":"bad","request":{"subtype":"can_use_tool"}}),
            );
            std::thread::sleep(Duration::from_secs(30));
        }
        "missing-terminal" => return 0,
        "hang-paid" => {
            sleeper(true);
            std::thread::sleep(Duration::from_secs(30));
        }
        "output-limit" => {
            let mut out = std::io::stdout().lock();
            out.write_all("x".repeat(16 * 1024 * 1024 + 32).as_bytes())
                .unwrap();
            out.write_all(b"\n").unwrap();
            out.flush().unwrap();
            drop(out);
            std::thread::sleep(Duration::from_secs(30));
        }
        "malformed-paid" => {
            let mut out = std::io::stdout().lock();
            out.write_all(b"{\"bad\":\n").unwrap();
            out.flush().unwrap();
            return 0;
        }
        _ => {}
    }
    let mut result = json!({"type":"result","subtype":"success","is_error":false,"result":text,
        "session_id":"fixture-session","stop_reason":"end_turn","terminal_reason":"completed",
        "num_turns":1,"result_index":0,"queued_turn_count":0,"permission_denials":[],
        "usage":usage,"modelUsage":{"claude-fixture-1":{"inputTokens":11,"outputTokens":7,
        "cacheReadInputTokens":3,"cacheCreationInputTokens":2,"costUSD":9999.99,
        "provider":"firstParty"}},"total_cost_usd":9999.99});
    match mode {
        "unknown-usage" => {
            let fields = result.as_object_mut().unwrap();
            fields.remove("usage");
            fields.remove("modelUsage");
        }
        "paid-auth-error" => {
            result["is_error"] = json!(true);
            result["api_error_status"] = json!(401);
            result["result"] = json!("DO NOT ECHO SECRET");
        }
        "aborted" => result["terminal_reason"] = json!("aborted_streaming"),
        "truncated" => result["stop_reason"] = json!("max_tokens"),
        "wrong-session" => result["session_id"] = json!("another-session"),
        _ => {}
    }
    emit(&result);
    if mode == "extra-result" {
        emit(&result);
    } else {
        emit(
            &json!({"type":"system","subtype":"session_state_changed","state":"idle","session_id":"fixture-session"}),
        );
    }
    if mode == "stderr-after-terminal" {
        close_stdout();
        stderr(&"secret".repeat(200_000));
        std::thread::sleep(Duration::from_secs(30));
    }
    if mode == "bad-exit" { 7 } else { 0 }
}

/// Codex: version and login probes, and one `exec --json` turn.
fn codex(home: &Path, scenario: &Value) -> i32 {
    let args = arguments();
    let name = scenario["name"].as_str().unwrap_or("ok");
    let cwd = working_directory();
    let mut names: Vec<String> = std::env::vars().map(|(key, _)| key).collect();
    names.sort();
    append(
        &home.join("calls.jsonl"),
        &json!({"args":args,"cwd":cwd.to_string_lossy(),"home":env("HOME"),"env_names":names}),
    );
    if args == ["--version"] {
        println!(
            "codex-cli {}",
            if name == "version" {
                "0.158.0"
            } else {
                "0.159.0"
            }
        );
        return 0;
    }
    if args == ["login", "status"] {
        return match name {
            "auth-none" => {
                stderr("Not logged in\n");
                1
            }
            "auth-api" => {
                stderr("Logged in using an API key - sk-fake-never-expose\n");
                0
            }
            _ => {
                stderr("Logged in using ChatGPT\n");
                0
            }
        };
    }
    assert!(args.iter().any(|arg| arg == "exec"));
    for flag in [
        "--ignore-user-config",
        "--ignore-rules",
        "--ephemeral",
        "--skip-git-repo-check",
        "--json",
    ] {
        assert!(args.iter().any(|arg| arg == flag), "{flag} is missing");
    }
    assert_eq!(after(&args, "--sandbox"), "read-only");
    assert_eq!(after(&args, "--model"), "gpt-5.5");
    for key in [
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "CODEX_API_KEY",
        "HTTP_PROXY",
        "HTTPS_PROXY",
    ] {
        assert!(env(key).is_none(), "{key} reached the runtime");
    }
    let mut settings = serde_json::Map::new();
    for (index, arg) in args.iter().enumerate() {
        if arg == "-c" {
            let (key, value) = args[index + 1].split_once('=').unwrap();
            settings.insert(key.into(), serde_json::from_str(value).unwrap());
        }
    }
    assert!(!settings.contains_key("forced_login_method"));
    assert_eq!(settings["model_provider"], "openai");
    assert_eq!(settings["project_doc_max_bytes"], 0);
    assert_eq!(settings["web_search"], "disabled");
    assert_eq!(settings["features.shell_tool"], false);
    let catalog_path = PathBuf::from(settings["model_catalog_json"].as_str().unwrap());
    let catalog: Value = serde_json::from_slice(&std::fs::read(&catalog_path).unwrap()).unwrap();
    let models = catalog["models"].as_array().unwrap();
    assert_eq!(models.len(), 1);
    assert_eq!(models[0]["slug"], "gpt-5.5");
    assert!(models[0]["apply_patch_tool_type"].is_null());
    assert!(models[0]["tool_mode"].is_null());
    assert_eq!(models[0]["experimental_supported_tools"], json!([]));
    let images: Vec<PathBuf> = args
        .iter()
        .enumerate()
        .filter(|(_, arg)| *arg == "--image")
        .map(|(index, _)| PathBuf::from(&args[index + 1]))
        .collect();
    let instructions = PathBuf::from(settings["model_instructions_file"].as_str().unwrap());
    let mut user = String::new();
    std::io::stdin().read_to_string(&mut user).unwrap();
    let system = std::fs::read_to_string(&instructions).unwrap();
    let hashes: Vec<String> = images
        .iter()
        .map(|path| sha256(&std::fs::read(path).unwrap()))
        .collect();
    #[cfg_attr(not(unix), allow(unused_mut))]
    let mut request = json!({"system":system,"user":user,"image_hashes":hashes,
        "workspace":cwd.to_string_lossy()});
    #[cfg(unix)]
    {
        request["workspace_mode"] = json!(permissions(&cwd));
        let files = [instructions.clone(), catalog_path.clone()];
        request["file_modes"] = files
            .iter()
            .chain(&images)
            .map(|path| json!(permissions(path)))
            .collect();
    }
    write_json(&home.join("request.json"), &request);
    append(&home.join("requests.jsonl"), &request);
    emit(&json!({"type":"thread.started","thread_id":"authored-thread"}));
    emit(
        &json!({"type":"item.completed","item":{"id":"warning","type":"error",
        "message":"`[features].codex_hooks` is deprecated. Use hooks instead."}}),
    );
    emit(&json!({"type":"turn.started"}));
    match name {
        "stderr" => stderr(&"x".repeat(1024 * 1024 + 1)),
        "sleep" => {
            let pid = sleeper(false);
            write_text(&home.join("grandchild.pid"), &pid.to_string());
            std::thread::sleep(Duration::from_secs(60));
        }
        "tool" => {
            emit(
                &json!({"type":"item.started","item":{"id":"tool","type":"command_execution",
                "command":"must never run"}}),
            );
            std::thread::sleep(Duration::from_secs(60));
        }
        "failed" => {
            emit(&json!({"type":"turn.failed","error":{"message":"sk-fake-never-expose"}}));
            return 1;
        }
        _ => {}
    }
    emit(
        &json!({"type":"item.completed","item":{"id":"comment","type":"agent_message",
        "text":"An intermediate commentary."}}),
    );
    let mut result = scenario["text"]
        .as_str()
        .unwrap_or("Complete authored document.")
        .to_owned();
    if name == "echo" {
        result = user.clone();
        if system.contains("MARKITAI_DOCUMENT_JSON_V1")
            || system.contains("MARKITAI_VISION_JSON_V1")
        {
            result = json!({"cleaned_markdown":result,"frontmatter":{
                "description":"Authored Codex fixture.","tags":["fixture"]}})
            .to_string();
        }
    }
    emit(
        &json!({"type":"item.completed","item":{"id":"answer","type":"agent_message","text":result}}),
    );
    if name == "truncated" {
        return 0;
    }
    let mut usage = json!({"input_tokens":11,"cached_input_tokens":3,"cache_write_input_tokens":0,
        "output_tokens":5,"reasoning_output_tokens":2});
    if name == "zero" {
        for value in usage.as_object_mut().unwrap().values_mut() {
            *value = json!(0);
        }
    }
    if name == "bad-count" {
        usage["input_tokens"] = json!(-1);
    }
    emit(&json!({"type":"turn.completed","usage":usage}));
    if name == "after-terminal" {
        emit(&json!({"type":"turn.failed","error":{"message":"sk-fake-never-expose"}}));
    }
    if name == "nonzero" { 7 } else { 0 }
}
