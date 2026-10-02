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
            "/base/models?pageSize=1000&key=private-test-key",
            "",
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
    let bad = || Err(Error::Conversion("SECRET URL KEY must never escape".into()));
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
    assert_eq!(gone["status"], "unavailable");
    assert_eq!(gone["models"], json!([]));
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
    assert_eq!(
        discover(&json!({"provider":"claude-agent"})).unwrap()["status"],
        "unavailable"
    );
}

#[test]
fn the_catalog_lists_the_popular_providers_then_every_compatible_prefix() {
    let catalog = catalog();
    let names: Vec<_> = catalog.iter().map(|entry| entry.provider).collect();
    assert_eq!(&names[..8], &POPULAR);
    let compatible: Vec<_> = catalog
        .iter()
        .filter(|entry| entry.compatible)
        .map(|entry| entry.provider)
        .collect();
    assert_eq!(
        compatible,
        [
            "groq",
            "mistral",
            "xai",
            "together_ai",
            "perplexity",
            "cerebras",
            "fireworks_ai",
            "deepinfra",
            "nebius",
            "moonshot",
            "sambanova",
            "zai",
            "nvidia_nim",
            "novita",
            "hosted_vllm",
            "lm_studio",
        ]
    );
    for entry in &catalog {
        assert_ne!(provider_label(entry.provider), "Unknown provider");
        if entry.provider == "custom" {
            assert!(entry.key_variables.is_empty() && entry.default_base.is_none());
            continue;
        }
        assert!(
            crate::llm::providers::supported(entry.provider),
            "{entry:?}"
        );
        assert!(!entry.key_variables.is_empty(), "{entry:?}");
        assert_eq!(
            entry.discovery,
            !["perplexity", "zai", "fireworks_ai"].contains(&entry.provider),
            "{entry:?}"
        );
    }
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
    for provider in ["perplexity", "zai", "fireworks_ai"] {
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
