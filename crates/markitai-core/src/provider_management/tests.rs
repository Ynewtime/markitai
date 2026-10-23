use super::*;
use std::{
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    sync::{
        Arc, Barrier, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    thread,
    time::Duration,
};

struct Mock {
    base: String,
    requests: Arc<Mutex<Vec<String>>>,
    stop: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<()>>,
}
impl Mock {
    fn new(reply: impl Fn(usize) -> String + Send + 'static) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let seen = requests.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let stopping = stop.clone();
        let worker = thread::spawn(move || {
            while !stopping.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        stream.set_nonblocking(false).unwrap();
                        stream
                            .set_read_timeout(Some(Duration::from_secs(3)))
                            .unwrap();
                        stream
                            .set_write_timeout(Some(Duration::from_secs(3)))
                            .unwrap();
                        let request = read(&mut stream);
                        let count = seen.lock().unwrap().len();
                        seen.lock().unwrap().push(request);
                        let _ = stream.write_all(reply(count).as_bytes());
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2))
                    }
                    Err(error) => panic!("mock accept: {error}"),
                }
            }
        });
        Self {
            base,
            requests,
            stop,
            worker: Some(worker),
        }
    }
    fn requests(&self) -> Vec<String> {
        self.requests.lock().unwrap().clone()
    }
}
impl Drop for Mock {
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
fn read(stream: &mut TcpStream) -> String {
    let mut request_reader = bounded_fixture_io::Reader::new(
        stream,
        std::time::Instant::now() + std::time::Duration::from_secs(3),
    );
    let mut bytes = Vec::new();
    let mut block = [0; 2048];
    loop {
        let n = request_reader.read(&mut block).unwrap();
        assert!(n > 0);
        bytes.extend_from_slice(&block[..n]);
        assert!(bytes.len() < 65536);
        if let Some(end) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
            let header = String::from_utf8_lossy(&bytes[..end]);
            let size = header
                .lines()
                .find_map(|line| {
                    line.to_ascii_lowercase()
                        .strip_prefix("content-length:")
                        .and_then(|s| s.trim().parse::<usize>().ok())
                })
                .unwrap_or(0);
            if bytes.len() >= end + 4 + size {
                return String::from_utf8(bytes).unwrap();
            }
        }
    }
}
fn reply(status: u16, body: &str) -> String {
    format!(
        "HTTP/1.1 {status} result\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
}

#[test]
fn discovery_uses_provider_specific_paths_credentials_and_typed_models() {
    for (provider, path, header, body, model, vision) in [
        (
            "openai",
            "/base/models",
            "authorization: Bearer private-test-key",
            r#"{"data":[{"id":"gpt-test"}]}"#,
            "openai/gpt-test",
            false,
        ),
        (
            "anthropic",
            "/base/models?limit=1000",
            "x-api-key: private-test-key",
            r#"{"data":[{"id":"claude-test","display_name":"Claude Test"}]}"#,
            "anthropic/claude-test",
            true,
        ),
        (
            "gemini",
            "/base/models?pageSize=1000",
            "x-goog-api-key: private-test-key",
            r#"{"models":[{"name":"models/embedding","supportedGenerationMethods":["embedContent"]},{"name":"models/gemini-test","supportedGenerationMethods":["generateContent"]}]}"#,
            "gemini/gemini-test",
            false,
        ),
        (
            "ollama",
            "/base/api/tags",
            "",
            r#"{"models":[{"name":"llama:test"}]}"#,
            "ollama/llama:test",
            false,
        ),
        (
            "azure",
            "/base/openai/models?api-version=2024-10-21",
            "api-key: private-test-key",
            r#"{"data":[{"id":"regional-test"}]}"#,
            "azure/regional-test",
            false,
        ),
        (
            "custom",
            "/base/models",
            "authorization: Bearer private-test-key",
            r#"{"data":[{"id":"local/test"}]}"#,
            "openai/local/test",
            false,
        ),
        (
            "openrouter",
            "/base/models",
            "authorization: Bearer private-test-key",
            r#"{"data":[{"id":"org/test","architecture":{"input_modalities":["text","image"]}}]}"#,
            "openrouter/org/test",
            true,
        ),
        (
            "deepseek",
            "/base/models",
            "authorization: Bearer private-test-key",
            r#"{"data":[{"id":"deepseek-test"}]}"#,
            "deepseek/deepseek-test",
            false,
        ),
    ] {
        let server = Mock::new(move |_| reply(200, body));
        let result=discover(&json!({"provider":provider,"api_base":format!("{}/base",server.base),"api_key":"private-test-key"})).unwrap();
        assert_eq!(result["models"].as_array().unwrap().len(), 1);
        assert_eq!(result["models"][0]["model"], model);
        assert_eq!(result["models"][0]["supports_vision"], vision);
        assert_eq!(result["authoritative"], provider != "azure");
        let requests = server.requests();
        assert_eq!(requests.len(), 1);
        assert!(
            requests[0].starts_with(&format!("GET {path} HTTP/1.1")),
            "{}",
            requests[0]
        );
        assert!(requests[0].contains(header));
        if provider == "anthropic" {
            assert!(requests[0].contains("anthropic-version: 2023-06-01"));
        }
        if matches!(provider, "gemini" | "ollama") {
            assert!(!requests[0].contains("authorization:"));
        }
        // The key never travels in the request line, where logs record it.
        let line = requests[0].lines().next().unwrap();
        assert!(!line.contains("private-test-key"), "{line}");
    }
}

#[test]
fn discovery_v1_azure_and_pagination_do_not_claim_deployment_authority() {
    let server = Mock::new(|_| reply(200, r#"{"data":[{"id":"base"}]}"#));
    let result=discover(&json!({"provider":"azure","api_base":format!("{}/openai/v1",server.base),"api_key":"test"})).unwrap();
    assert_eq!(result["status"], "partial");
    assert!(server.requests()[0].starts_with("GET /openai/v1/models HTTP"));
    let paged = discovery::parse(
        "gemini",
        &json!({"models":[{"name":"models/a"}],"nextPageToken":"next"}),
    )
    .unwrap();
    assert_eq!(paged["authoritative"], false);
    assert_eq!(paged["status"], "partial");
}

#[test]
fn discovery_cache_is_credential_scoped_and_refresh_is_single_flight() {
    let cache = Arc::new(cache::Cache::default());
    let loads = Arc::new(AtomicUsize::new(0));
    let value = json!({"provider":"openai","status":"ok","cached":false,"stale":false,"models":[{"model":"openai/a"}]});
    let get = |key: &str, refresh: bool| {
        cache
            .discover("openai", "http://local/v1", Some(key), refresh, || {
                loads.fetch_add(1, Ordering::SeqCst);
                Ok(value.clone())
            })
            .unwrap()
    };
    assert_eq!(get("one", false)["cached"], false);
    assert_eq!(get("one", false)["cached"], true);
    get("two", false);
    assert_eq!(loads.load(Ordering::SeqCst), 2);
    let start = Arc::new(Barrier::new(8));
    thread::scope(|scope| {
        for _ in 0..8 {
            let (cache, loads, start, value) =
                (cache.clone(), loads.clone(), start.clone(), value.clone());
            scope.spawn(move || {
                start.wait();
                cache
                    .discover("openai", "http://local/v1", Some("one"), true, || {
                        loads.fetch_add(1, Ordering::SeqCst);
                        thread::sleep(Duration::from_millis(120));
                        Ok(value)
                    })
                    .unwrap()
            });
        }
    });
    assert_eq!(loads.load(Ordering::SeqCst), 3);
}

#[test]
fn cache_ttl_stale_boundary_and_failure_do_not_extend_good_age() {
    let cache = cache::Cache::default();
    let value = json!({"provider":"custom","status":"ok","cached":false,"stale":false,"models":[{"model":"openai/saved"}]});
    let good = || Ok(value.clone());
    let failure = discovery::failed(
        "custom",
        crate::llm::ProbeReason::Network,
        "Model discovery request failed: connection refused",
        None,
    );
    let bad = || Err(failure.clone());
    cache
        .discover("custom", "http://local", None, false, good)
        .unwrap();
    cache.age("custom", "http://local", None, 301);
    let old = cache
        .discover("custom", "http://local", None, false, bad)
        .unwrap();
    assert_eq!(old["status"], "partial");
    assert_eq!(old["stale"], true);
    assert_eq!(old["models"], value["models"]);
    assert!(!old.to_string().contains("SECRET"));
    cache.age("custom", "http://local", None, 24 * 60 * 60 + 1);
    let gone = cache
        .discover("custom", "http://local", None, false, bad)
        .unwrap();
    // With no good list left, the failure says why, and it is not cached.
    assert_eq!(gone, failure);
    assert_eq!(gone["reason"], "network");
    let waiting = cache
        .discover("custom", "http://other", None, false, bad)
        .unwrap();
    assert_eq!(waiting["cached"], false);
    assert_eq!(
        cache
            .discover("custom", "http://other", None, false, good)
            .unwrap()["cached"],
        false,
        "a failure is never served as a cached list"
    );
}

/// Discover through a mock that answers every request with `answer`.
fn discovered(request: Value, answer: String) -> (Value, Vec<String>) {
    let server = Mock::new(move |_| answer.clone());
    let mut request = request;
    request["api_base"] = format!("{}/v1", server.base).into();
    request["refresh"] = true.into();
    let result = discover_explicit(&request).unwrap();
    (result, server.requests())
}

#[test]
fn a_failed_discovery_says_why_in_the_probes_reasons() {
    let keyed = json!({"provider":"openai","api_key":"sk-secret-discovery-key"});
    for (status, body, reason, detail) in [
        (
            401,
            r#"{"error":{"message":"Incorrect API key provided: sk-secret-discovery-key","type":"invalid_request_error"}}"#,
            "key_rejected",
            "Model discovery returned HTTP 401: Incorrect API key provided: …",
        ),
        (
            403,
            r#"{"message":"Forbidden here"}"#,
            "key_rejected",
            "Model discovery returned HTTP 403: Forbidden here",
        ),
        (
            429,
            r#"{"error":"Slow down"}"#,
            "rate_limited",
            "Model discovery returned HTTP 429: Slow down",
        ),
        (
            404,
            "<html><head><title>404</title></head><body><h1>Not   Found</h1></body></html>",
            "provider_error",
            "Model discovery returned HTTP 404: 404 Not Found",
        ),
        (
            500,
            r#"{"error":{"code":1}}"#,
            "provider_error",
            "Model discovery returned HTTP 500",
        ),
        (
            200,
            "not JSON",
            "provider_error",
            "Model discovery response is not JSON",
        ),
        (
            200,
            r#"{"error":"secret"}"#,
            "provider_error",
            "Model discovery response has no model list",
        ),
    ] {
        let (result, requests) = discovered(keyed.clone(), reply(status, body));
        assert_eq!(requests.len(), 1, "{status}");
        assert_eq!(
            result,
            json!({"provider":"openai","status":"unavailable","source":"live_api","authoritative":false,
                "cached":false,"stale":false,"models":[],"detail":detail,"reason":reason}),
            "{status} {body}"
        );
        assert!(!result.to_string().contains("sk-secret-discovery-key"));
    }
    // A long message is cut to 300 characters.
    let long = format!(r#"{{"message":"{}"}}"#, "x".repeat(400));
    let (result, _) = discovered(keyed.clone(), reply(400, &long));
    let detail = result["detail"].as_str().unwrap();
    assert_eq!(
        detail,
        format!("Model discovery returned HTTP 400: {}…", "x".repeat(300))
    );
    // A redirect is never followed: no HTTP answer from the provider.
    let (result, _) = discovered(
        keyed.clone(),
        "HTTP/1.1 302 Found
Location: http://127.0.0.1:9/
Content-Length: 0
Connection: close

"
        .into(),
    );
    assert_eq!(result["reason"], "network");
    assert!(
        result["detail"]
            .as_str()
            .unwrap()
            .starts_with("Model discovery request failed: ")
    );
    // A keyless request refused for want of a key: the key is what is missing.
    let (result, requests) = discovered(json!({"provider":"deepseek"}), reply(401, "{}"));
    assert_eq!(requests.len(), 1);
    assert_eq!(result["reason"], "no_key");
    assert_eq!(result["detail"], "Model discovery returned HTTP 401");
    assert!(
        result.get("key_variable").is_none(),
        "an explicit discovery reads no variable"
    );
    // Nobody listening.
    let closed = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = closed.local_addr().unwrap();
    drop(closed);
    let result = discover_explicit(
        &json!({"provider":"openai","api_key":"k","api_base":format!("http://{address}/v1"),"refresh":true}),
    )
    .unwrap();
    assert_eq!(result["reason"], "network");
    assert_eq!(
        result["detail"],
        "Model discovery request failed: connection refused"
    );
}

#[test]
fn a_discovery_that_gets_no_answer_in_time_says_so() {
    let (result, _) = discovered(json!({"provider":"openai","api_key":"k"}), String::new());
    // An empty answer closes the connection without a response.
    assert_eq!(result["reason"], "network");
    let server = Mock::new(|_| {
        thread::sleep(discovery::TIMEOUT + Duration::from_secs(1));
        reply(200, r#"{"data":[]}"#)
    });
    let result = discover_explicit(
        &json!({"provider":"openai","api_key":"k","api_base":server.base,"refresh":true}),
    )
    .unwrap();
    assert_eq!(result["reason"], "timeout");
    assert_eq!(result["detail"], "Model discovery timed out");
}

#[test]
fn a_missing_discovery_key_is_reported_before_any_request() {
    let server = Mock::new(|_| reply(200, r#"{"data":[{"id":"listed"}]}"#));
    let unset = "MARKITAI_DISCOVERY_TEST_UNSET_KEY_2B9D";
    assert!(!crate::config::environment().contains_key(unset));
    for request in [
        json!({"provider":"openai","api_key":format!("env:{unset}"),"api_base":server.base}),
        json!({"provider":"ollama","api_key":format!("env:{unset}"),"api_base":server.base}),
    ] {
        for result in [
            discover(&request).unwrap(),
            discover_explicit(&request).unwrap(),
        ] {
            assert_eq!(
                result,
                json!({"provider":request["provider"],"status":"unavailable","source":"live_api",
                    "authoritative":false,"cached":false,"stale":false,"models":[],"reason":"no_key",
                    "key_variable":unset,"detail":format!("{unset} is not set on the server, so nothing was sent")})
            );
        }
    }
    // A provider that needs a key, at its own endpoint, with none: an
    // explicit discovery reads no variable, so none is named.
    let result = discover_explicit(&json!({"provider":"deepseek"})).unwrap();
    assert_eq!(result["reason"], "no_key");
    assert!(result.get("key_variable").is_none());
    assert_eq!(
        result["detail"],
        "Model discovery needs an API key, and none is set, so nothing was sent"
    );
    assert!(server.requests().is_empty());
    // Local servers, OpenAI-compatible endpoints, the keyless placeholder and
    // an address of the caller's own are tried.
    for request in [
        json!({"provider":"ollama","api_base":server.base}),
        json!({"provider":"lm_studio","api_base":format!("{}/v1",server.base)}),
        json!({"provider":"custom","api_base":server.base}),
        json!({"provider":"deepseek","api_key":NO_KEY,"api_base":server.base}),
        json!({"provider":"openai","api_base":format!("{}/v1",server.base)}),
    ] {
        let mut request = request;
        request["refresh"] = true.into();
        let result = discover_explicit(&request).unwrap();
        assert!(
            result.get("reason").is_none() || result["reason"] != "no_key",
            "{request}: {result}"
        );
    }
    assert_eq!(server.requests().len(), 5);
}

#[test]
fn discovery_refuses_redirects_and_oversized_or_malformed_responses() {
    let destination = Mock::new(|_| reply(200, r#"{"data":[{"id":"leaked"}]}"#));
    let location = destination.base.clone();
    let redirect = Mock::new(move |_| {
        format!(
            "HTTP/1.1 302 Found\r\nLocation: {location}/models\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
        )
    });
    let result =
        discover(&json!({"provider":"custom","api_base":redirect.base,"api_key":"secret"}))
            .unwrap();
    assert_eq!(result["status"], "unavailable");
    assert!(destination.requests().is_empty());
    for body in ["not JSON", r#"{"error":"secret"}"#] {
        let server = Mock::new(move |_| reply(200, body));
        let result = discover(&json!({"provider":"custom","api_base":server.base})).unwrap();
        assert_eq!(result["status"], "unavailable");
        assert!(!result.to_string().contains("secret"));
    }
    let server = Mock::new(|_| {
        "HTTP/1.1 200 OK\r\nContent-Length: 2097153\r\nConnection: close\r\n\r\n".into()
    });
    assert_eq!(
        discover(&json!({"provider":"custom","api_base":server.base})).unwrap()["status"],
        "unavailable"
    );
    assert!(discovery::parse("openai", &json!({"data":vec![json!({"id":"x"});1001]})).is_err());
}

#[test]
fn probe_uses_exact_protocol_single_deployment_and_short_payload() {
    for (model, path, auth, response) in [
        (
            "openai/gpt-5-test",
            "/v1/chat/completions",
            "authorization: Bearer private-test-key",
            r#"{"choices":[{"message":{"content":"A real response"}}]}"#,
        ),
        (
            "anthropic/claude-test",
            "/v1/messages",
            "x-api-key: private-test-key",
            r#"{"content":[{"type":"text","text":"OK"}]}"#,
        ),
        (
            "azure/deployment-test",
            "/v1/openai/deployments/deployment-test/chat/completions?api-version=",
            "api-key: private-test-key",
            r#"{"choices":[{"message":{"content":"OK"}}]}"#,
        ),
        (
            "gemini/gemini-test",
            "/v1/chat/completions",
            "authorization: Bearer private-test-key",
            r#"{"choices":[{"message":{"content":"OK"}}]}"#,
        ),
        (
            "ollama/llama:test",
            "/v1/chat/completions",
            "authorization: Bearer private-test-key",
            r#"{"choices":[{"message":{"content":"OK"}}]}"#,
        ),
    ] {
        let server = Mock::new(move |_| reply(200, response));
        assert_eq!(probe(&json!({"model":model,"api_base":format!("{}/v1",server.base),"api_key":"private-test-key"})).unwrap()["ok"],true);
        let requests = server.requests();
        assert_eq!(requests.len(), 1);
        assert!(requests[0].starts_with(&format!("POST {path}")));
        assert!(requests[0].contains(auth));
        let body: Value =
            serde_json::from_str(requests[0].split_once("\r\n\r\n").unwrap().1).unwrap();
        assert_eq!(
            body["messages"],
            json!([{"role":"user","content":"Reply with exactly OK."}])
        );
        assert_eq!(body["model"], model.split_once('/').unwrap().1);
        assert_eq!(
            body.get("max_tokens")
                .or_else(|| body.get("max_completion_tokens")),
            Some(&json!(16))
        );
        assert!(body.get("system").is_none());
    }
}

#[test]
fn probe_errors_are_sanitized_and_never_retry_or_follow_redirects() {
    let secret = "DO-NOT-EXPOSE-THIS-KEY";
    let server = Mock::new(move |_| reply(429, secret));
    let result =
        probe(&json!({"model":"openai/specific-model","api_key":secret,"api_base":server.base}))
            .unwrap();
    assert_eq!(result["ok"], false);
    assert!(result["detail"].as_str().unwrap().contains("429"));
    assert!(!result.to_string().contains(secret));
    assert_eq!(server.requests().len(), 1);
    let destination = Mock::new(|_| reply(200, r#"{"choices":[{"message":{"content":"OK"}}]}"#));
    let base = destination.base.clone();
    let server = Mock::new(move |_| {
        format!(
            "HTTP/1.1 307 Redirect\r\nLocation: {base}/chat/completions\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
        )
    });
    assert_eq!(
        probe(&json!({"model":"openai/specific-model","api_key":secret,"api_base":server.base}))
            .unwrap()["ok"],
        false
    );
    assert!(destination.requests().is_empty());
    assert_eq!(
        probe(&json!({"model":"copilot/requested","api_base":destination.base})).unwrap()["ok"],
        false
    );
    assert!(destination.requests().is_empty());
}

#[test]
fn failed_probes_report_a_category_from_the_status_or_transport() {
    let answered = |status: u16, body: &'static str| {
        let server = Mock::new(move |_| reply(status, body));
        let result = probe_explicit(
            &json!({"model":"openai/category-test","api_key":"category-test-key","api_base":server.base}),
        )
        .unwrap();
        assert_eq!(server.requests().len(), 1, "{result}");
        assert!(result.get("key_variable").is_none(), "{result}");
        result
    };
    let ok = answered(200, r#"{"choices":[{"message":{"content":"OK"}}]}"#);
    assert_eq!(ok["ok"], true);
    assert!(ok.get("reason").is_none(), "{ok}");
    for (status, body, reason) in [
        (401, "{}", "key_rejected"),
        (
            403,
            r#"{"error":{"code":"permission_denied"}}"#,
            "key_rejected",
        ),
        (404, "{}", "model_not_found"),
        (
            400,
            r#"{"error":{"code":"model_not_found"}}"#,
            "model_not_found",
        ),
        (429, "{}", "rate_limited"),
        (
            402,
            r#"{"error":{"type":"insufficient_balance"}}"#,
            "rate_limited",
        ),
        (
            403,
            r#"{"error":{"message":"User location is not supported"}}"#,
            "provider_error",
        ),
        (500, "{}", "provider_error"),
        (307, "", "network"),
        (200, r#"{"choices":[]}"#, "provider_error"),
    ] {
        let result = answered(status, body);
        assert_eq!(
            (&result["ok"], &result["reason"]),
            (&json!(false), &json!(reason)),
            "{status} {body}: {result}"
        );
    }
    let refused = TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap();
    let result = probe_explicit(&json!({"model":"openai/category-test","api_key":"category-test-key","api_base":format!("http://{refused}/v1")})).unwrap();
    assert_eq!(result["reason"], "network", "{result}");
}

#[test]
fn a_missing_key_is_reported_before_any_request_and_keyless_servers_are_tried() {
    let server = Mock::new(|_| reply(200, r#"{"choices":[{"message":{"content":"OK"}}]}"#));
    // A synthetic variable no environment or dotenv file sets.
    let unset = "MARKITAI_PROBE_TEST_UNSET_KEY_5C1E";
    assert!(!crate::config::environment().contains_key(unset));
    for request in [
        json!({"model":"openai/missing-key","api_key":format!("env:{unset}"),"api_base":server.base}),
        json!({"model":"ollama/missing-key","api_key":format!("env:{unset}"),"api_base":server.base}),
    ] {
        for result in [probe(&request).unwrap(), probe_explicit(&request).unwrap()] {
            assert_eq!(
                result,
                json!({"ok":false,"reason":"no_key","key_variable":unset,
                    "detail":format!("{unset} is not set on the server")})
            );
        }
    }
    // A provider that needs a key, at its own endpoint, with none supplied:
    // an explicit test reads no variable, so none is named.
    assert_eq!(
        probe_explicit(&json!({"model":"deepseek/missing-key"})).unwrap(),
        json!({"ok":false,"reason":"no_key","detail":"This model needs an API key, and none is set"})
    );
    assert!(server.requests().is_empty());
    // Local servers and the keyless placeholder at an address are tried.
    for request in [
        json!({"model":"ollama/keyless","api_base":server.base}),
        json!({"model":"lm_studio/keyless","api_base":server.base}),
        json!({"model":"deepseek/keyless","api_key":NO_KEY,"api_base":server.base}),
        json!({"model":"openai/keyless","api_base":server.base}),
    ] {
        let result = probe_explicit(&request).unwrap();
        assert_eq!(result["ok"], true, "{request}: {result}");
    }
    assert_eq!(server.requests().len(), 4);
    // An address that refuses the keyless request: the key is what is missing.
    let refusing = Mock::new(|_| reply(401, "{}"));
    assert_eq!(
        probe_explicit(&json!({"model":"deepseek/keyless","api_base":refusing.base})).unwrap()["reason"],
        "no_key"
    );
    assert_eq!(refusing.requests().len(), 1);
}

#[test]
fn requests_validate_types_urls_without_echoing_values() {
    for request in [
        json!({"provider":"custom","api_base":"file:///secret"}),
        json!({"provider":"custom","api_base":"http://user:secret@localhost"}),
        json!({"provider":"openai","refresh":"true"}),
        json!({"provider":"openai","api_key":42}),
        json!({"provider":"openai","unknown":"secret"}),
    ] {
        assert!(discover(&request).is_err());
    }
    assert!(probe(&json!({"model":"","api_key":"secret"})).is_err());
    assert!(probe(&json!({"model":"openai/x","fallbacks":["openai/y"]})).is_err());
    // A subscription provider says why it has no list (see `subscription_runtimes_say_why_they_list_no_models`).
    let subscription = discover(&json!({"provider":"claude-agent"})).unwrap();
    assert_eq!(subscription["status"], "unavailable");
    assert!(subscription["reason"].is_string());
}

/// A fake runtime's home: the adapters find it by the variable, the stand-in plays the scenario.
fn fake_runtime_env(
    path_variable: &str,
    home_variable: &str,
    scenario: Value,
) -> (tempfile::TempDir, HashMap<String, String>) {
    let home = tempfile::tempdir().unwrap();
    crate::subscription::fake_runtime::install(home.path(), &scenario);
    let env = HashMap::from([
        (
            path_variable.to_owned(),
            crate::subscription::fake_runtime::program()
                .to_string_lossy()
                .into_owned(),
        ),
        (
            home_variable.to_owned(),
            home.path().to_string_lossy().into_owned(),
        ),
    ]);
    (home, env)
}

#[test]
fn subscription_runtimes_say_why_they_list_no_models() {
    // Not installed: nothing named on PATH, and a path that leads nowhere.
    let empty = tempfile::tempdir().unwrap();
    let on_path = HashMap::from([(
        "PATH".to_owned(),
        empty.path().to_string_lossy().into_owned(),
    )]);
    for (provider, command) in [
        (
            "claude-agent",
            "npm install -g @anthropic-ai/claude-code@2.1.284",
        ),
        ("copilot", "npm install -g @github/copilot@1.0.90-2"),
        ("chatgpt", "npm install -g @openai/codex@0.159.0"),
    ] {
        let missing = runtime::discover(provider, &on_path);
        assert_eq!(missing["status"], "unavailable", "{provider}");
        assert_eq!(missing["reason"], "runtime_missing", "{provider}");
        assert_eq!(
            missing["next_step"],
            json!({"kind":"install","command":command}),
            "{provider}"
        );
        assert!(
            missing["detail"].as_str().unwrap().contains("PATH"),
            "{provider}"
        );
        assert!(missing["runtime"]["name"].is_string());
    }
    let nowhere = HashMap::from([(
        "CLAUDE_CLI_PATH".to_owned(),
        empty.path().join("claude").to_string_lossy().into_owned(),
    )]);
    let missing = runtime::discover("claude-agent", &nowhere);
    assert_eq!(missing["reason"], "runtime_missing");
    assert!(
        missing["detail"]
            .as_str()
            .unwrap()
            .contains("CLAUDE_CLI_PATH")
    );

    // Installed, but another release: the installed one is named.
    let (_home, env) = fake_runtime_env(
        "CLAUDE_CLI_PATH",
        "CLAUDE_CONFIG_DIR",
        json!({"mode":"wrong-version"}),
    );
    let old = runtime::discover("claude-agent", &env);
    assert_eq!(old["reason"], "runtime_unsupported");
    assert_eq!(old["runtime"]["installed"], "2.1.283");
    assert!(old["detail"].as_str().unwrap().contains("2.1.283"));
    assert_eq!(old["next_step"]["kind"], "install");
    let (_home, env) = fake_runtime_env("CODEX_CLI_PATH", "CODEX_HOME", json!({"name":"version"}));
    assert_eq!(
        runtime::discover("chatgpt", &env)["reason"],
        "runtime_unsupported"
    );

    // Installed, but not signed in: the next step is the sign-in command.
    let (_home, env) =
        fake_runtime_env("CODEX_CLI_PATH", "CODEX_HOME", json!({"name":"auth-none"}));
    let signed_out = runtime::discover("chatgpt", &env);
    assert_eq!(signed_out["reason"], "not_signed_in", "{signed_out}");
    assert_eq!(
        signed_out["next_step"],
        json!({"kind":"sign_in","command":"markitai auth chatgpt login"})
    );
    let cache = tempfile::tempdir().unwrap();
    let (_home, mut env) = fake_runtime_env(
        "COPILOT_CLI_PATH",
        "COPILOT_HOME",
        json!({"mode":"unauth","home":std::env::var("HOME").ok(),"cache_home":cache.path()}),
    );
    env.insert(
        "COPILOT_CACHE_HOME".into(),
        cache.path().to_string_lossy().into_owned(),
    );
    let signed_out = runtime::discover("copilot", &env);
    assert_eq!(signed_out["reason"], "not_signed_in", "{signed_out}");
    assert_eq!(
        signed_out["next_step"]["command"],
        "markitai auth copilot login"
    );

    let (_home, env) = fake_runtime_env(
        "CLAUDE_CLI_PATH",
        "CLAUDE_CONFIG_DIR",
        json!({"mode":"signed-out"}),
    );
    let signed_out = runtime::discover("claude-agent", &env);
    assert_eq!(signed_out["reason"], "not_signed_in", "{signed_out}");
    assert_eq!(
        signed_out["next_step"]["command"],
        "markitai auth claude login"
    );
    assert_eq!(signed_out["detail"], "Claude CLI is not signed in");

    // Working runtimes still list.
    let (_home, env) =
        fake_runtime_env("CLAUDE_CLI_PATH", "CLAUDE_CONFIG_DIR", json!({"mode":"ok"}));
    assert_eq!(runtime::discover("claude-agent", &env)["status"], "ok");
    let (_home, env) = fake_runtime_env("CODEX_CLI_PATH", "CODEX_HOME", json!({"name":"ok"}));
    assert_eq!(runtime::discover("chatgpt", &env)["status"], "ok");
}

#[test]
fn a_runtime_answering_in_a_code_page_fails_only_its_own_discovery() {
    // `--version` printed in GBK (code page 936), as a program writing the
    // ANSI code page on a Chinese Windows would: "已就绪" is not UTF-8.
    let gbk = |prefix: &str| {
        let mut bytes = prefix.as_bytes().to_vec();
        bytes.extend_from_slice(&[0xd2, 0xd1, 0xbe, 0xcd, 0xd0, 0xf7, b'\r', b'\n']);
        json!(bytes)
    };
    let (_home, env) = fake_runtime_env(
        "CLAUDE_CLI_PATH",
        "CLAUDE_CONFIG_DIR",
        json!({"mode":"ok","version_bytes":gbk("2.1.284 (Claude Code) ")}),
    );
    let claude = runtime::discover("claude-agent", &env);
    assert_eq!(claude["status"], "unavailable", "{claude}");
    assert_eq!(claude["reason"], "runtime_failed", "{claude}");
    let (_home, env) = fake_runtime_env(
        "CODEX_CLI_PATH",
        "CODEX_HOME",
        json!({"name":"ok","version_bytes":gbk("codex-cli 0.159.0 ")}),
    );
    let codex = runtime::discover("chatgpt", &env);
    assert_eq!(codex["status"], "unavailable", "{codex}");
    assert_eq!(codex["reason"], "runtime_unsupported", "{codex}");
    for result in [&claude, &codex] {
        // No undecoded bytes or replacement characters reach the message.
        assert!(!result.to_string().contains('\u{fffd}'), "{result}");
        assert_eq!(result["models"], json!([]), "{result}");
    }

    // The same runtimes answering in UTF-8 list their models.
    let (_home, env) =
        fake_runtime_env("CLAUDE_CLI_PATH", "CLAUDE_CONFIG_DIR", json!({"mode":"ok"}));
    assert_eq!(runtime::discover("claude-agent", &env)["status"], "ok");
    let (_home, env) = fake_runtime_env("CODEX_CLI_PATH", "CODEX_HOME", json!({"name":"ok"}));
    assert_eq!(runtime::discover("chatgpt", &env)["status"], "ok");
}

#[test]
fn the_catalog_offers_every_routed_prefix_once_in_picker_groups() {
    let catalog = catalog();
    let names: Vec<_> = catalog.iter().map(|entry| entry.provider).collect();
    assert_eq!(
        &names[..8],
        [
            "openai",
            "anthropic",
            "gemini",
            "deepseek",
            "openrouter",
            "azure",
            "ollama",
            "mistral"
        ]
    );
    assert!(catalog[..8].iter().all(|entry| entry.group == "common"));
    assert!(catalog[8..].iter().all(|entry| entry.group != "common"));
    // Every HTTP prefix, the clouds and the subscription runtimes appear once,
    // as a provider or as an alias routed the same way.
    let mut routed: Vec<_> = crate::llm::providers::listings()
        .map(|listing| listing.prefix)
        .chain([
            "bedrock",
            "vertex_ai",
            "vertex_ai_beta",
            "copilot",
            "claude-agent",
            "chatgpt",
        ])
        .collect();
    routed.sort_unstable();
    let mut offered: Vec<_> = catalog
        .iter()
        .flat_map(|entry| {
            std::iter::once(entry.provider).chain(entry.aliases.iter().copied().filter(|alias| {
                crate::llm::providers::supported(alias)
                    && !catalog.iter().any(|other| other.provider == *alias)
            }))
        })
        .filter(|name| *name != "custom")
        .collect();
    offered.sort_unstable();
    let before = offered.len();
    offered.dedup();
    assert_eq!(offered.len(), before, "a prefix is offered twice");
    assert_eq!(offered, routed);
    let mut groups = PROVIDER_GROUPS.to_vec();
    for entry in &catalog {
        assert!(PROVIDER_GROUPS.contains(&entry.group), "{entry:?}");
        groups.retain(|group| *group != entry.group);
        assert_eq!(provider_label(entry.provider), entry.label);
        assert!(entry.icon.is_none_or(|icon| {
            !icon.is_empty()
                && icon
                    .bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
        }));
        if matches!(
            entry.provider,
            "custom" | "vertex_ai" | "copilot" | "claude-agent" | "chatgpt"
        ) {
            assert!(entry.key_variables.is_empty() && entry.default_base.is_none());
            continue;
        }
        assert!(!entry.key_variables.is_empty(), "{entry:?}");
        assert_eq!(
            entry.discovery,
            !["perplexity", "zai", "fireworks_ai", "bedrock"].contains(&entry.provider),
            "{entry:?}"
        );
    }
    assert!(groups.is_empty(), "{groups:?} hold no provider");
    // Each group lists its providers by label after the common eight.
    for group in &PROVIDER_GROUPS[1..] {
        let labels: Vec<_> = catalog
            .iter()
            .filter(|entry| entry.group == *group)
            .map(|entry| entry.label.to_lowercase())
            .collect();
        let mut sorted = labels.clone();
        sorted.sort();
        assert_eq!(labels, sorted, "{group}");
    }
    let compatible = |name: &str| {
        catalog
            .iter()
            .any(|entry| entry.provider == name && entry.compatible)
    };
    assert!(compatible("mistral") && compatible("groq") && compatible("huggingface"));
    assert!(
        !compatible("openai")
            && !compatible("custom")
            && !compatible("bedrock")
            && !compatible("chatgpt")
    );
    assert_eq!(provider_label("ollama_chat"), "Ollama");
    assert_eq!(provider_label("cohere_chat"), "Cohere");
    assert_eq!(provider_label("claude"), "Unknown provider");
    let find = |name: &str| catalog.iter().find(|entry| entry.provider == name).unwrap();
    assert_eq!(
        find("groq").default_base,
        Some("https://api.groq.com/openai/v1")
    );
    assert_eq!(find("together_ai").key_variables[0], "TOGETHER_API_KEY");
    assert!(find("lm_studio").key_optional && find("hosted_vllm").key_optional);
    assert_eq!(find("hosted_vllm").default_base, None);
    assert!(!find("groq").key_optional);
    // Discovery lists `/models` beside the inference endpoint of the added prefixes.
    assert_eq!(
        provider_default_base("nvidia_nim"),
        Some("https://integrate.api.nvidia.com/v1")
    );
    assert_eq!(provider_default_base("hosted_vllm"), None);
    assert_eq!(provider_default_base("ollama_chat"), None);
}

#[test]
fn compatible_prefixes_discover_through_models_or_ask_for_manual_ids() {
    for (provider, body, model) in [
        (
            "groq",
            r#"{"object":"list","data":[{"id":"llama-test","object":"model"}]}"#,
            "groq/llama-test",
        ),
        // Together AI's list is a bare array; its IDs keep their slashes.
        (
            "together_ai",
            r#"[{"id":"meta-llama/Test-Turbo","display_name":"Test Turbo","type":"chat"}]"#,
            "together_ai/meta-llama/Test-Turbo",
        ),
        (
            "lm_studio",
            r#"{"data":[{"id":"local-test"}]}"#,
            "lm_studio/local-test",
        ),
        // A prefix offered since the picker lists every routed provider.
        (
            "huggingface",
            r#"{"object":"list","data":[{"id":"org/model-test"}]}"#,
            "huggingface/org/model-test",
        ),
    ] {
        let server = Mock::new(move |_| reply(200, body));
        let mut request = json!({"provider":provider,"api_base":format!("{}/v1",server.base)});
        if provider != "lm_studio" {
            request["api_key"] = json!("private-test-key");
        }
        let result = discover(&request).unwrap();
        assert_eq!(result["status"], "ok", "{provider}: {result}");
        assert_eq!(result["models"][0]["model"], model);
        assert_eq!(result["models"][0]["supports_vision"], false);
        let requests = server.requests();
        assert!(
            requests[0].starts_with("GET /v1/models HTTP/1.1"),
            "{}",
            requests[0]
        );
        assert_eq!(
            requests[0].contains("authorization: Bearer private-test-key"),
            provider != "lm_studio",
            "{provider}"
        );
    }
    // A bare array is accepted only from the provider documented to send one.
    let server = Mock::new(|_| reply(200, r#"[{"id":"x"}]"#));
    let refused =
        discover(&json!({"provider":"groq","api_base":format!("{}/v1",server.base),"api_key":"k"}))
            .unwrap();
    assert_eq!(refused["status"], "unavailable");
    assert!(refused["models"].as_array().unwrap().is_empty());
    // No model list is documented: no request, an explicit manual-entry answer.
    for provider in ["perplexity", "zai", "fireworks_ai", "bedrock", "vertex_ai"] {
        let result = discover(&json!({"provider":provider,"api_key":"private-test-key"})).unwrap();
        assert_eq!(result["status"], "unavailable");
        assert_eq!(result["source"], "manual");
        assert!(result["models"].as_array().unwrap().is_empty());
        assert!(!result.to_string().contains("private-test-key"));
    }
    assert!(discover(&json!({"provider":"zai","api_base":"file:///x"})).is_err());
    // vLLM has no default address.
    assert_eq!(
        discover(&json!({"provider":"hosted_vllm"})).unwrap()["detail"],
        "An API endpoint is required"
    );
    assert!(discover(&json!({"provider":"ollama_chat"})).is_err());
}

#[cfg(test)]
mod bounded_fixture_io {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/support/bounded_read.rs"
    ));
}

#[test]
fn discovery_environment_endpoints_follow_all_provider_catalog_entries() {
    for provider in catalog() {
        for variable in provider.base_variables {
            let env = HashMap::from([(
                (*variable).to_owned(),
                "http://127.0.0.1:9911/v1".to_owned(),
            )]);
            assert_eq!(
                discovery_base_variable(provider.provider, &env),
                Some(*variable),
                "{}",
                provider.provider
            );
        }
    }
    let mut env = HashMap::from([
        (
            "OPENAI_API_BASE".to_owned(),
            "http://127.0.0.1:9911/v1".to_owned(),
        ),
        (
            "OPENAI_BASE_URL".to_owned(),
            "http://127.0.0.1:9912/v1".to_owned(),
        ),
    ]);
    assert_eq!(
        discovery_base_variable("openai", &env),
        Some("OPENAI_API_BASE")
    );
    env.insert("OPENAI_API_BASE".to_owned(), "".to_owned());
    assert_eq!(
        discovery_base_variable("openai", &env),
        Some("OPENAI_BASE_URL")
    );
    env.insert("OPENAI_BASE_URL".to_owned(), " ".to_owned());
    assert_eq!(discovery_base_variable("openai", &env), None);
    // Explicit request mode passes an empty map and cannot inherit either base.
    assert_eq!(discovery_base_variable("openai", &HashMap::new()), None);
    assert_eq!(discovery_base_variable("unknown", &env), None);
}

#[test]
fn discovery_sends_no_key_for_the_keyless_placeholder() {
    assert!(real_key(Some("sk-real")) && real_key(Some("env:OPENAI_API_KEY")));
    assert!(!real_key(Some(NO_KEY)) && !real_key(Some("")) && !real_key(None));
    for provider in ["openai", "anthropic", "custom", "lm_studio"] {
        let server = Mock::new(|_| reply(200, r#"{"data":[{"id":"listed"}]}"#));
        // Both modes: the placeholder is an explicit value, so no variable is
        // read in its place, and it is not sent either.
        let request = json!({"provider":provider,"api_base":format!("{}/v1",server.base),"api_key":NO_KEY,"refresh":true});
        discover(&request).unwrap();
        discover_explicit(&request).unwrap();
        let requests = server.requests();
        assert_eq!(requests.len(), 2, "{provider}");
        for request in requests {
            let lower = request.to_ascii_lowercase();
            assert!(
                !lower.contains("authorization:") && !lower.contains("x-api-key:"),
                "{provider}: {request}"
            );
        }
    }
}

#[test]
fn a_users_discovery_and_test_never_reach_this_machine() {
    let server = Mock::new(|_| reply(200, r#"{"data":[{"id":"local-model"}]}"#));
    let named = server.base.replace("127.0.0.1", "localhost");
    for base in [server.base.clone(), named] {
        let found =
            discover_public(&json!({"provider":"custom","api_base":base,"api_key":"k"})).unwrap();
        assert_eq!(found["status"], "unavailable", "{found}");
        assert!(
            found["detail"]
                .as_str()
                .unwrap()
                .contains("not on the public internet"),
            "{found}"
        );
        let tested = probe_public(
            &json!({"model":"openai/x","api_base":format!("{base}/v1"),"api_key":"k"}),
        )
        .unwrap();
        assert_eq!(tested["ok"], false, "{tested}");
        assert!(
            tested["detail"]
                .as_str()
                .unwrap()
                .contains("not on the public internet"),
            "{tested}"
        );
    }
    assert!(server.requests().is_empty());
    // The admin's own discovery of the same address still reaches it.
    let found =
        discover_explicit(&json!({"provider":"custom","api_base":server.base,"api_key":"k"}))
            .unwrap();
    assert!(found.to_string().contains("local-model"), "{found}");
}
