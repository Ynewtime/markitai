//! Static fetch through one environment proxy, using authored loopback servers.
//! The isolated child reads its proxy from a private `.env`, so no process-wide
//! environment is mutated and no operating-system proxy setting is consulted.
use markitai_core::{ConvertOptions, convert};
use serde_json::json;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

fn isolated(name: &str) -> bool {
    let exact = format!("proxy::{name}");
    if std::env::var("MARKITAI_PROXY_TEST").as_deref() == Ok(&exact) {
        return false;
    }
    let root = tempfile::tempdir().unwrap();
    let mut command = Command::new(std::env::current_exe().unwrap());
    command
        .args(["--exact", &exact, "--nocapture"])
        .env_clear()
        .env("MARKITAI_PROXY_TEST", &exact)
        .env("MARKITAI_HOME", root.path().join("state"))
        .current_dir(root.path())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for key in ["HOME", "PATH", "LANG", "TMPDIR"] {
        if let Some(value) = std::env::var_os(key) {
            command.env(key, value);
        }
    }
    let child = command.spawn().unwrap();
    let started = Instant::now();
    let output = child.wait_with_output().unwrap();
    assert!(started.elapsed() < Duration::from_secs(90));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success() && stdout.contains("1 passed"),
        "{stdout}{}",
        String::from_utf8_lossy(&output.stderr)
    );
    true
}

/// Records each request head; replies with `respond(first line)`.
fn server(respond: fn(&str) -> String) -> (u16, Arc<Mutex<Vec<String>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let record = seen.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut head = Vec::new();
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                    break;
                }
                head.push(line.trim_end().to_owned());
            }
            let Some(first) = head.first().cloned() else {
                continue;
            };
            record.lock().unwrap().push(head.join("\n"));
            let _ = stream.write_all(respond(&first).as_bytes());
        }
    });
    (port, seen)
}

fn html(body: &str) -> String {
    let page = format!(
        "<!doctype html><html><head><title>{body}</title></head><body><main><h1>{body}</h1><p>Paragraph for {body}.</p></main></body></html>"
    );
    format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{page}",
        page.len()
    )
}

#[test]
fn static_fetch_uses_the_single_environment_proxy_with_reference_bypass() {
    if isolated("static_fetch_uses_the_single_environment_proxy_with_reference_bypass") {
        return;
    }
    let (proxy, proxied) = server(|first| {
        if first.starts_with("GET http://proxied.test/page ") {
            html("Via proxy")
        } else if first.starts_with("CONNECT ") {
            "HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".into()
        } else {
            "HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".into()
        }
    });
    let (origin, direct) = server(|_| html("Direct origin"));
    // HTTPS_PROXY is blank: the first nonempty variable is HTTP_PROXY, and it is
    // used for HTTPS as well. no_proxy is ignored because NO_PROXY is nonempty.
    std::fs::write(
        ".env",
        format!(
            "HTTPS_PROXY=\"  \"\nHTTP_PROXY=http://127.0.0.1:{proxy}\nNO_PROXY=bypass.test\nno_proxy=proxied.test\n"
        ),
    )
    .unwrap();
    let options = || ConvertOptions {
        config: Some(
            json!({"fetch":{"strategy":"static"},"cache":{"enabled":false},"llm":{"enabled":false},
            "history":{"record":false},"log":{"dir":null}}),
        ),
        ..Default::default()
    };
    let result = convert("http://proxied.test/page", options()).unwrap();
    assert!(result.markdown.contains("Via proxy"), "{}", result.markdown);
    let local = convert(&format!("http://127.0.0.1:{origin}/direct"), options()).unwrap();
    assert!(local.markdown.contains("Direct origin"));
    assert!(convert("http://bypass.test/page", options()).is_err());
    assert!(convert("https://secure.test/page", options()).is_err());

    let proxied = proxied.lock().unwrap().clone();
    let firsts: Vec<_> = proxied
        .iter()
        .map(|head| head.lines().next().unwrap().to_owned())
        .collect();
    assert!(
        firsts
            .iter()
            .any(|line| line == "GET http://proxied.test/page HTTP/1.1"),
        "{firsts:?}"
    );
    assert!(
        firsts
            .iter()
            .any(|line| line.starts_with("CONNECT secure.test:443 ")),
        "{firsts:?}"
    );
    assert!(
        !proxied
            .iter()
            .any(|head| head.contains("bypass.test") || head.contains("/direct")),
        "{proxied:?}"
    );
    assert_eq!(direct.lock().unwrap().len(), 1);
    assert!(direct.lock().unwrap()[0].starts_with("GET /direct HTTP/1.1"));
}
