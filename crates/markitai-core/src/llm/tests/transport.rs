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
            && error.contains("usually means a proxy or network problem")
            && error.contains("set HTTPS_PROXY or turn on the system proxy")
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
    let client = reqwest::blocking::Client::builder()
        .no_proxy()
        .proxy(proxy)
        .build()
        .unwrap();
    let error = client.get("https://example.invalid/").send().unwrap_err();
    server.join().unwrap();
    let said = diagnosis::cause(error).unwrap();
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
            "LLM returned HTTP 401 (invalid_request_error/invalid_api_key): Incorrect API key provided: [REDACTED]. You can find your API key at https://platform.example.test/account/api-keys.",
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
