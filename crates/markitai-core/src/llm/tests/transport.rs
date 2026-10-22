//! A failed request names its kind, the provider's own message when there is
//! one, and the deployment and host that failed; never the credential, the
//! prompts or the document (review finding R8).
use super::*;

/// A loopback server that handles one connection with `respond`.
fn serve(respond: impl FnOnce(TcpStream) + Send + 'static) -> (String, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap().to_string();
    let thread = thread::spawn(move || {
        let (stream, _) = listener.accept().unwrap();
        respond(stream);
    });
    (address, thread)
}

/// Closes the write side and reads until the client closes. macOS and
/// Windows reset a connection closed with unread request bytes, and the
/// client then sees the reset before the response.
fn drain(mut stream: TcpStream) {
    let _ = stream.shutdown(std::net::Shutdown::Write);
    let _ = stream.set_read_timeout(Some(Duration::from_secs(3)));
    let _ = std::io::copy(&mut stream, &mut std::io::sink());
}

fn failure(base: &str, timeout: u64) -> String {
    let mut cfg = cfg("openai/gpt-test", base);
    cfg["llm"]["router_settings"]["num_retries"] = json!(0);
    cfg["llm"]["router_settings"]["timeout"] = json!(timeout);
    run(&plain(), &cfg, &HashMap::new(), &mut |_| {})
        .unwrap_err()
        .to_string()
}

#[test]
fn a_refused_connection_is_named_with_its_deployment() {
    let address = TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .to_string();
    assert_eq!(
        failure(&format!("http://{address}/v1"), 3),
        format!("LLM request failed: connection refused (deployment openai/gpt-test at {address})")
    );
}

#[test]
fn a_failed_tls_handshake_is_named() {
    // A plain HTTP server behind an https:// base: the handshake reads an
    // HTTP answer instead of a TLS record.
    let (address, server) = serve(|mut stream| {
        let _ = stream.write_all(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n");
        drain(stream);
    });
    let error = failure(&format!("https://{address}/v1"), 3);
    server.join().unwrap();
    assert!(
        error.starts_with("LLM request failed: TLS handshake failed (")
            && error.ends_with(&format!(") (deployment openai/gpt-test at {address})")),
        "{error}"
    );
}

#[test]
fn a_handshake_cut_off_by_the_other_end_points_at_a_proxy_or_the_network() {
    // The peer closes the connection without a word, as a blocked route does.
    let (address, server) = serve(|mut stream| {
        use std::io::Read;
        let mut hello = [0u8; 4096];
        let _ = stream.read(&mut hello);
        let _ = stream.shutdown(std::net::Shutdown::Write);
        drain(stream);
    });
    let error = failure(&format!("https://{address}/v1"), 3);
    server.join().unwrap();
    assert!(
        error.starts_with("LLM request failed: TLS handshake failed (")
            && error.contains("likely a proxy or network problem")
            && error.contains("set HTTPS_PROXY or turn on the system proxy")
            && error.contains("within 30 s")
            && !error.contains("docs.rs")
            && error.ends_with(&format!(") (deployment openai/gpt-test at {address})")),
        "{error}"
    );
}

#[test]
fn the_proxy_not_opening_a_tunnel_is_named() {
    use crate::llm::diagnosis;
    // The proxy answers the CONNECT with a refusal.
    let (address, server) = serve(|mut stream| {
        use std::io::Read;
        let mut request = [0u8; 2048];
        let _ = stream.read(&mut request);
        let _ = stream.write_all(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n");
        drain(stream);
    });
    let proxy = reqwest::Proxy::all(format!("http://{address}")).unwrap();
    let client = crate::trust::bundled_builder()
        .no_proxy()
        .proxy(proxy)
        .build()
        .unwrap();
    let error = client.get("https://example.invalid/").send().unwrap_err();
    server.join().unwrap();
    let said = diagnosis::cause(error, None).unwrap();
    assert!(
        said.contains("the proxy did not open a connection to the provider"),
        "{said}"
    );
}

#[test]
fn a_timeout_names_its_limit() {
    let (address, server) = serve(|mut stream| {
        read_request(&mut stream);
        thread::sleep(Duration::from_millis(1500));
    });
    let error = failure(&format!("http://{address}/v1"), 1);
    server.join().unwrap();
    assert_eq!(
        error,
        format!(
            "LLM request timed out: no response within 1 s (deployment openai/gpt-test at {address})"
        )
    );
}

#[test]
fn a_connection_closed_before_the_response_is_named() {
    let (address, server) = serve(|mut stream| {
        read_request(&mut stream);
        drop(stream);
    });
    let error = failure(&format!("http://{address}/v1"), 3);
    server.join().unwrap();
    assert!(
        error.starts_with("LLM request failed: connection closed without a response (")
            && error.ends_with(&format!("(deployment openai/gpt-test at {address})")),
        "{error}"
    );
}

#[test]
fn http_errors_carry_the_provider_message_without_secrets_or_document_text() {
    let cases = [
        (
            400,
            json!({"error":{"message":"Invalid 'max_tokens': integer above maximum value. Expected a value <= 4096, but got 8192 instead.","type":"invalid_request_error","param":"max_tokens","code":"integer_above_max_value"}}),
            "LLM returned HTTP 400 (invalid_request_error/integer_above_max_value): Invalid 'max_tokens': integer above maximum value. Expected a value <= 4096, but got 8192 instead.",
        ),
        (
            500,
            json!({"error":{"message":"The server had an error while processing your request.","type":"server_error"}}),
            "LLM returned HTTP 500 (server_error): The server had an error while processing your request.",
        ),
        (
            401,
            json!({"error":{"message":"Incorrect API key provided: fake-test-key. You can find your API key at https://platform.example.test/account/api-keys?token=abc.","type":"invalid_request_error","code":"invalid_api_key"}}),
            "LLM returned HTTP 401 (invalid_request_error/invalid_api_key): the api_key configured for this model was refused; check that key. Provider message: Incorrect API key provided: [REDACTED]. You can find your API key at https://platform.example.test/account/api-keys.",
        ),
        (
            400,
            // The provider quotes the document it refused (plain() carries it).
            json!({"error":{"message":"Unexpected text near: # input {source} is literal","type":"invalid_request_error"}}),
            "LLM returned HTTP 400 (invalid_request_error): [message withheld: it repeats the request]",
        ),
    ];
    for (status, body, expected) in cases {
        let server = Mock::new(vec![(status, body)]);
        let address = server
            .base
            .trim_start_matches("http://")
            .trim_end_matches("/v1")
            .to_owned();
        let error = failure(&server.base, 3);
        server.finish();
        assert_eq!(
            error,
            format!("{expected} (deployment openai/gpt-test at {address})")
        );
    }
}

#[test]
fn a_body_that_is_not_json_names_its_type_and_size() {
    let page = "<html><head><title>Sign in</title></head><body>gateway</body></html>";
    let (address, server) = serve(move |mut stream| {
        read_request(&mut stream);
        let _ = write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{page}",
            page.len()
        );
    });
    let error = failure(&format!("http://{address}/v1"), 3);
    server.join().unwrap();
    assert_eq!(
        error,
        format!(
            "LLM response is not valid JSON (HTTP 200, text/html, {} bytes): Sign in (deployment openai/gpt-test at {address})",
            page.len()
        )
    );
}

#[test]
fn a_retried_failure_is_warned_with_its_cause() {
    let address = TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .to_string();
    let mut cfg = cfg("openai/gpt-test", &format!("http://{address}/v1"));
    cfg["llm"]["router_settings"]["num_retries"] = json!(1);
    let scope = DocumentScope::new(&cfg);
    let error = run(&plain(), &cfg, &HashMap::new(), &mut |_| {}).unwrap_err();
    assert!(
        error
            .to_string()
            .starts_with("LLM request failed: connection refused (deployment")
    );
    assert_eq!(
        scope.take_warnings(),
        [
            "LLM request to openai/gpt-test failed (LLM request failed: connection refused) and was sent again"
        ]
    );
}

#[test]
fn a_refusing_proxy_is_named_without_its_credentials_and_the_provider_is_not_blamed() {
    // Nothing listens on port 9 of loopback.
    let proxy = reqwest::Proxy::all("http://alice:secret@127.0.0.1:9").unwrap();
    let client = crate::trust::bundled_builder()
        .no_proxy()
        .proxy(proxy)
        .build()
        .unwrap();
    let error = client
        .get("https://api.example.test/v1")
        .send()
        .unwrap_err();
    let said = diagnosis::cause(error, Some("127.0.0.1:9 (HTTPS_PROXY)")).unwrap();
    assert_eq!(
        said,
        "connection refused by the proxy 127.0.0.1:9 (HTTPS_PROXY); check that it is running"
    );
    assert!(!said.contains("alice") && !said.contains("secret"));
    // Without a proxy the wording is unchanged.
    let direct = crate::trust::bundled_builder().no_proxy().build().unwrap();
    let error = direct.get("http://127.0.0.1:9/").send().unwrap_err();
    assert_eq!(diagnosis::cause(error, None).unwrap(), "connection refused");
}

#[test]
fn a_user_model_never_reaches_this_machine() {
    // Never accepted: a user model's request must not connect here.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let port = listener.local_addr().unwrap().port();
    for base in [
        format!("http://localhost:{port}/v1"),
        format!("http://127.0.0.1:{port}/v1"),
    ] {
        let mut cfg = cfg("openai/gpt-test", &base);
        cfg["llm"]["model_list"][0]["model_info"] = json!({"public_only": true});
        cfg["llm"]["router_settings"]["num_retries"] = json!(2);
        let error = run(&plain(), &cfg, &HashMap::new(), &mut |_| {})
            .unwrap_err()
            .to_string();
        assert!(error.contains("not on the public internet"), "{error}");
    }
    assert!(listener.accept().is_err(), "a request reached loopback");
}

#[test]
fn retry_after_reads_the_forms_providers_send() {
    assert_eq!(retry_delay("7"), Some(7));
    assert_eq!(retry_delay("1.204"), Some(2));
    assert_eq!(retry_delay("1.204s"), Some(2));
    assert_eq!(retry_delay("750ms"), Some(1));
    assert_eq!(retry_delay("6m0s"), Some(360));
    assert_eq!(retry_delay("20 seconds"), Some(20));
    assert_eq!(retry_delay("Wed, 21 Oct 2015 07:28:00 GMT"), Some(0));
    assert_eq!(retry_delay("soon"), None);
    // In a rate limit's words, only after a phrase that introduces a delay.
    let gemini = r#"[{"error":{"code":429,"details":[{"@type":"type.googleapis.com/google.rpc.retryinfo","retrydelay":"1.204s"}]}}]"#;
    assert_eq!(body_retry_delay(gemini), Some(2));
    assert_eq!(
        body_retry_delay("rate limit reached: limit 30, used 30. please try again in 750ms."),
        Some(1)
    );
    assert_eq!(body_retry_delay("limit 30s window, used 30"), None);
}

#[test]
fn a_delay_in_the_rate_limit_body_is_waited_for() {
    let limited = json!({"error":{"message":"Rate limit reached. Please try again in 3.5s.","code":"rate_limit_exceeded"}});
    let server = Mock::new(vec![(429, limited), (200, success("after the wait"))]);
    let mut waited = Vec::new();
    let (text, _) = run(
        &plain(),
        &cfg("openai/test", &server.base),
        &HashMap::new(),
        &mut |delay| waited.push(delay),
    )
    .unwrap();
    assert_eq!(text, "after the wait");
    assert_eq!(waited, [Duration::from_secs(4)]);
    server.finish();
}

#[test]
fn a_spent_request_budget_keeps_the_failure_that_spent_it() {
    let failing = json!({"error":{"message":"upstream overloaded","type":"server_error"}});
    let server = Mock::new(vec![(503, failing)]);
    let mut cfg = cfg("openai/test", &server.base);
    cfg["llm"]["max_requests_per_document"] = json!(1);
    let error = run(&plain(), &cfg, &HashMap::new(), &mut |_| {})
        .unwrap_err()
        .to_string();
    assert!(
        error.starts_with(
            "LLM per-document request budget exhausted after a failed request: LLM returned HTTP 503 (server_error): upstream overloaded"
        ),
        "{error}"
    );
    server.finish();
}
