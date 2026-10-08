//! The provider table, the one-time image notice, the repetition guard on the
//! text, document and visual paths, and chunks sized by a declared input
//! window, over loopback HTTP.
use super::*;
use crate::fetch::consent::{Gate, RemoteFallback, RemoteNotice};
use std::sync::atomic::AtomicBool;

fn vars(pairs: &[(&str, &str)]) -> HashMap<String, String> {
    pairs
        .iter()
        .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
        .collect()
}

fn single(params: Value) -> Value {
    config::normalize(
        &json!({"llm":{"model_list":[{"model_name":"default","litellm_params":params}]}}),
    )
    .unwrap()
}

fn resolved(params: Value, env: &[(&str, &str)]) -> Result<Deployment> {
    deployments(&single(params), &vars(env)).map(|mut entries| entries.remove(0))
}

#[test]
fn a_deployment_debug_view_never_shows_its_key() {
    let mut entry = resolved(
        json!({"model":"openai/debug-test","api_key":"sk-debug-view-secret"}),
        &[],
    )
    .unwrap();
    entry.endpoint = "https://llm.example/v1/chat/completions?api_key=endpoint-secret".into();
    let shown = format!("{entry:?}");
    assert!(
        !shown.contains("sk-debug-view-secret") && !shown.contains("endpoint-secret"),
        "{shown}"
    );
    assert!(
        shown.contains("debug-test") && shown.contains("llm.example"),
        "{shown}"
    );
}

#[test]
fn openai_compatible_prefixes_use_their_documented_endpoints_and_key_variables() {
    let groq = resolved(
        json!({"model":"groq/llama-test"}),
        &[("GROQ_API_KEY", "groq-key")],
    )
    .unwrap();
    assert_eq!(
        groq.endpoint,
        "https://api.groq.com/openai/v1/chat/completions"
    );
    assert_eq!(
        (groq.key.as_deref(), groq.model.as_str()),
        (Some("groq-key"), "llama-test")
    );
    assert_eq!(groq.protocol, Protocol::Chat);
    // The model keeps every slash after the prefix; LiteLLM's other key
    // names still work.
    let together = resolved(
        json!({"model":"together_ai/meta-llama/Test-70B"}),
        &[("TOGETHERAI_API_KEY", "together-key")],
    )
    .unwrap();
    assert_eq!(together.model, "meta-llama/Test-70B");
    assert_eq!(together.key.as_deref(), Some("together-key"));
    assert_eq!(
        together.endpoint,
        "https://api.together.ai/v1/chat/completions"
    );
    for (model, endpoint) in [
        ("mistral/m", "https://api.mistral.ai/v1/chat/completions"),
        ("xai/m", "https://api.x.ai/v1/chat/completions"),
        (
            "perplexity/sonar",
            "https://api.perplexity.ai/router/v1/chat/completions",
        ),
        ("cerebras/m", "https://api.cerebras.ai/v1/chat/completions"),
        (
            "fireworks_ai/accounts/fireworks/models/m",
            "https://api.fireworks.ai/inference/v1/chat/completions",
        ),
        (
            "deepinfra/m",
            "https://api.deepinfra.com/v1/openai/chat/completions",
        ),
        (
            "nebius/m",
            "https://api.tokenfactory.nebius.com/v1/chat/completions",
        ),
        ("moonshot/m", "https://api.moonshot.ai/v1/chat/completions"),
        (
            "sambanova/m",
            "https://api.sambanova.ai/v1/chat/completions",
        ),
        ("zai/m", "https://api.z.ai/api/paas/v4/chat/completions"),
        (
            "nvidia_nim/m",
            "https://integrate.api.nvidia.com/v1/chat/completions",
        ),
        ("novita/m", "https://api.novita.ai/openai/chat/completions"),
        ("lm_studio/m", "http://localhost:1234/v1/chat/completions"),
    ] {
        assert_eq!(
            resolved(json!({"model":model}), &[]).unwrap().endpoint,
            endpoint
        );
    }
    // Base variables follow LiteLLM's names; api_base wins over them.
    let fireworks = resolved(
        json!({"model":"fireworks_ai/m"}),
        &[("FIREWORKS_API_BASE", "https://gateway.example/v1")],
    )
    .unwrap();
    assert_eq!(
        fireworks.endpoint,
        "https://gateway.example/v1/chat/completions"
    );
    let pinned = resolved(
        json!({"model":"groq/m","api_base":"https://pinned.example/openai/v1"}),
        &[("GROQ_API_BASE", "https://ignored.example/v1")],
    )
    .unwrap();
    assert_eq!(
        pinned.endpoint,
        "https://pinned.example/openai/v1/chat/completions"
    );
    let ollama_chat = resolved(
        json!({"model":"ollama_chat/m"}),
        &[("OLLAMA_API_BASE", "http://127.0.0.1:11500/v1")],
    )
    .unwrap();
    assert_eq!(
        ollama_chat.endpoint,
        "http://127.0.0.1:11500/v1/chat/completions"
    );
    // vLLM has no default address.
    let Err(Error::Config(message)) = resolved(json!({"model":"hosted_vllm/m"}), &[]) else {
        panic!("hosted_vllm needs an address");
    };
    assert!(message.contains("HOSTED_VLLM_API_BASE"), "{message}");
    let vllm = resolved(
        json!({"model":"hosted_vllm/m"}),
        &[("HOSTED_VLLM_API_BASE", "http://gpu.internal:8000/v1")],
    )
    .unwrap();
    assert_eq!(
        vllm.endpoint,
        "http://gpu.internal:8000/v1/chat/completions"
    );
    let Err(Error::Config(message)) = resolved(json!({"model":"azure/deployment"}), &[]) else {
        panic!("azure needs an address");
    };
    assert!(message.contains("AZURE_API_BASE"), "{message}");
}

#[test]
fn local_servers_route_without_a_key_and_hosted_apis_need_one() {
    let lm_studio = single(json!({"model":"lm_studio/qwen-test"}));
    assert!(capabilities(&lm_studio, &HashMap::new()).routable);
    assert_eq!(
        vision_models(&lm_studio, &HashMap::new()),
        ["lm_studio/qwen-test"]
    );
    let vllm = single(json!({"model":"hosted_vllm/m","api_base":"http://127.0.0.1:8000/v1"}));
    assert!(capabilities(&vllm, &HashMap::new()).routable);
    let groq = single(json!({"model":"groq/m"}));
    assert!(!capabilities(&groq, &HashMap::new()).routable);
    assert!(capabilities(&groq, &vars(&[("GROQ_API_KEY", "k")])).routable);
    assert!(vision_models(&groq, &HashMap::new()).is_empty());
}

#[test]
fn bedrock_and_vertex_fail_with_the_openai_compatible_route() {
    for model in [
        "bedrock/anthropic.claude-test",
        "vertex_ai/gemini-test",
        "cohere/command",
    ] {
        let cfg = single(json!({"model":model,"api_key":"k"}));
        let Err(Error::Unsupported(message)) = deployments(&cfg, &HashMap::new()) else {
            panic!("{model} must be unsupported");
        };
        assert!(
            message.contains("openai/<model> with api_base"),
            "{message}"
        );
        assert!(!capabilities(&cfg, &HashMap::new()).routable);
    }
    // A supported sibling still serves the pool.
    let cfg = config::normalize(&json!({"llm":{"model_list":[
        {"model_name":"default","litellm_params":{"model":"bedrock/x","api_key":"k"}},
        {"model_name":"default","litellm_params":{"model":"mistral/m","api_key":"k"}}
    ]}}))
    .unwrap();
    assert_eq!(deployments(&cfg, &HashMap::new()).unwrap().len(), 1);
    assert!(config::llm_provider_supported("groq") && !config::llm_provider_supported("bedrock"));
}

#[test]
fn a_groq_request_carries_the_key_from_its_variable() {
    let server = Mock::new(vec![(200, success("Answer from the mock"))]);
    let cfg = config::normalize(&json!({"llm":{"model_list":[{"model_name":"default","litellm_params":{"model":"groq/test-model","api_base":server.base}}],"router_settings":{"timeout":3,"num_retries":0}},"cache":{"enabled":false}})).unwrap();
    let (text, _) = run(
        &plain(),
        &cfg,
        &vars(&[("GROQ_API_KEY", "groq-loopback-key")]),
        &mut |_| {},
    )
    .unwrap();
    assert_eq!(text, "Answer from the mock");
    let requests = server.finish();
    assert!(
        requests[0]
            .0
            .to_ascii_lowercase()
            .contains("authorization: bearer groq-loopback-key")
    );
    assert_eq!(requests[0].1["model"], "test-model");
}

fn entry(provider: &str, endpoint: &str) -> Deployment {
    Deployment {
        id: format!("{provider}/model"),
        explicit_id: None,
        group: "default".into(),
        model: "model".into(),
        provider: provider.into(),
        weight: 1,
        key: None,
        endpoint: endpoint.into(),
        protocol: Protocol::Chat,
        max_tokens: None,
        supports_vision: None,
    }
}

#[test]
fn images_to_a_model_off_this_machine_are_announced_once() {
    for (provider, endpoint) in [
        ("ollama", "http://localhost:11434/v1/chat/completions"),
        ("lm_studio", "http://127.0.0.1:1234/v1/chat/completions"),
        ("hosted_vllm", "http://[::1]:8000/v1/chat/completions"),
        ("openai", "http://models.localhost:8080/v1/chat/completions"),
        ("openai", "http://LOCALHOST./v1/chat/completions"),
    ] {
        assert!(!leaves_machine(&entry(provider, endpoint)), "{endpoint}");
    }
    for (provider, endpoint) in [
        ("openai", "https://api.openai.com/v1/chat/completions"),
        (
            "hosted_vllm",
            "http://192.168.1.20:8000/v1/chat/completions",
        ),
        ("claude-agent", "claude-cli://0123"),
        ("chatgpt", "http://localhost:1/never-used"),
    ] {
        assert!(leaves_machine(&entry(provider, endpoint)), "{endpoint}");
    }
    let home = tempfile::tempdir().unwrap();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let recorder = seen.clone();
    let gate = Gate::new(
        Some(RemoteFallback {
            explicitly_always: false,
            explicit_fallback_patterns: false,
            ask: None,
            notify: Box::new(move |notice| {
                recorder.lock().unwrap().push(notice.clone());
                true
            }),
        }),
        Some(home.path().into()),
    );
    // One group: a local model, two remote ones and a second endpoint of the
    // first remote model.
    let group = [
        entry("ollama", "http://localhost:11434/v1"),
        entry("openai", "https://api.openai.com/v1"),
        entry("gemini", "https://generativelanguage.googleapis.com"),
        entry("openai", "https://eu.api.openai.com/v1"),
    ];
    let all = [0, 1, 2, 3];
    // Selecting the local model sends nothing off the machine yet.
    disclose_images(&gate, &group, &all, 0);
    assert!(seen.lock().unwrap().is_empty());
    for _ in 0..3 {
        disclose_images(&gate, &group, &all, 2);
        disclose_images(&gate, &group, &all, 1);
    }
    // Every remote model of the group is named once, whichever was selected.
    assert_eq!(
        *seen.lock().unwrap(),
        [RemoteNotice::Images {
            models: vec!["openai/model".into(), "gemini/model".into()]
        }]
    );
    assert!(home.path().join("remote-images").is_file());
}

const LOOP_LINE: &str = "Thank you for reading this report.\n";

fn document_answer(markdown: &str) -> Value {
    success(
        &json!({"cleaned_markdown":markdown,"frontmatter":{"description":"A report","tags":["report"]}})
            .to_string(),
    )
}

#[test]
fn a_looping_plain_answer_is_cut_warned_and_asked_again_next_time() {
    let root = tempfile::tempdir().unwrap();
    let looping = format!("# Title\n\nShort body.\n\n{}", LOOP_LINE.repeat(30));
    let server = Mock::new(vec![(200, success(&looping)), (200, success(&looping))]);
    let cfg = cached_cfg(root.path(), "openai/test", &server.base);
    for _ in 0..2 {
        let answer = cached_call("# Title\n\nShort body.", "doc.md", "doc.md", &cfg).unwrap();
        assert!(!answer.cache_hit);
        assert_eq!(
            answer.markdown,
            "# Title\n\nShort body.\n\nThank you for reading this report."
        );
        assert!(
            answer
                .warnings
                .iter()
                .any(|warning| warning.contains("repeating one passage 30 times")),
            "{:?}",
            answer.warnings
        );
    }
    // Both runs asked the model: the salvaged answer was never cached.
    assert_eq!(server.finish().len(), 2);
}

#[test]
fn a_looping_document_chunk_is_salvaged_and_not_cached() {
    let root = tempfile::tempdir().unwrap();
    let source = "# Report\n\nThe quarterly figures improved across every region.";
    let looping = document_answer(&format!("{source}\n\n{}", LOOP_LINE.repeat(40)));
    let server = Mock::new(vec![(200, looping.clone()), (200, looping)]);
    let cfg = cached_cfg(root.path(), "openai/test", &server.base);
    for _ in 0..2 {
        let enhanced =
            process_document_with_runtime(source, "report.md", "report.md", false, &cfg, None)
                .unwrap();
        assert!(!enhanced.cache_hit);
        assert_eq!(
            enhanced.markdown,
            format!("{source}\n\nThank you for reading this report.")
        );
        assert!(
            enhanced
                .warnings
                .iter()
                .any(|warning| warning.contains("not cached")),
            "{:?}",
            enhanced.warnings
        );
    }
    assert_eq!(server.finish().len(), 2);
    // A clean answer is cached as before.
    let clean = Mock::new(vec![(200, document_answer(source))]);
    let cfg = cached_cfg(root.path(), "openai/test", &clean.base);
    let first =
        process_document_with_runtime(source, "report.md", "report.md", false, &cfg, None).unwrap();
    assert!(first.warnings.is_empty(), "{:?}", first.warnings);
    let second =
        process_document_with_runtime(source, "report.md", "report.md", false, &cfg, None).unwrap();
    assert!(second.cache_hit);
    assert_eq!(clean.finish().len(), 1);
}

#[test]
fn a_looping_visual_batch_is_salvaged_and_not_cached() {
    let root = tempfile::tempdir().unwrap();
    let looping = document_answer(&format!("Scanned page text.\n\n{}", LOOP_LINE.repeat(25)));
    let server = Mock::new(vec![(200, looping.clone()), (200, looping)]);
    let cfg = cached_cfg(root.path(), "openai/test", &server.base);
    let frames = [VisionFrame {
        number: 1,
        mime: "image/png",
        bytes: b"authored image bytes",
    }];
    for _ in 0..2 {
        let enhanced = process_vision_with_runtime(
            VisionRequest {
                markdown: "",
                source_label: "scan.png",
                cache_context: "scan.png",
                kind: VisionKind::PagedDocument,
                frames: &frames,
            },
            &cfg,
            None,
        )
        .unwrap();
        assert!(!enhanced.cache_hit);
        assert_eq!(
            enhanced.markdown,
            "Scanned page text.\n\nThank you for reading this report."
        );
        assert!(
            enhanced
                .warnings
                .iter()
                .any(|warning| warning.contains("repeating one passage 25 times")),
            "{:?}",
            enhanced.warnings
        );
    }
    assert_eq!(server.finish().len(), 2);
}

/// Answers each document request with its own user content as the cleaned
/// Markdown, so any chunking is accepted.
struct Echo {
    base: String,
    count: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}
impl Echo {
    fn new() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let base = format!("http://{}/v1", listener.local_addr().unwrap());
        let count = Arc::new(AtomicUsize::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let (seen, done) = (count.clone(), stop.clone());
        let thread = thread::spawn(move || {
            while !done.load(Ordering::Acquire) {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        let (_, request) = read_request(&mut stream);
                        seen.fetch_add(1, Ordering::AcqRel);
                        let content = request["messages"][1]["content"].as_str().unwrap_or("");
                        let body = document_answer(content).to_string();
                        write!(stream, "HTTP/1.1 200 Mock\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2))
                    }
                    Err(error) => panic!("echo listener failed: {error}"),
                }
            }
        });
        Self {
            base,
            count,
            stop,
            thread: Some(thread),
        }
    }
    fn count(&self) -> usize {
        self.count.load(Ordering::Acquire)
    }
}
impl Drop for Echo {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[test]
fn a_declared_input_window_splits_a_document_into_more_requests() {
    let paragraph = |n: usize| {
        format!(
            "Paragraph {n} describes the quarterly figures for one region in plain words. {}",
            "The numbers rose steadily and the team expects the same next year. ".repeat(6)
        )
    };
    // Chunk answers are merged trimmed, so no paragraph ends in a space.
    let markdown = (0..14)
        .map(|n| paragraph(n).trim_end().to_owned())
        .collect::<Vec<_>>()
        .join("\n\n");
    assert!(markdown.chars().count() > 6_000 && markdown.chars().count() < chunks::LIMIT);
    let with_window = |window: Option<u64>, base: &str| {
        let mut cfg = cfg("openai/test", base);
        if let Some(window) = window {
            cfg["llm"]["model_list"][0]["model_info"] = json!({"max_input_tokens":window});
        }
        cfg
    };
    for window in [None, Some(1_000_000), Some(2_000)] {
        let echo = Echo::new();
        let cfg = with_window(window, &echo.base);
        let enhanced =
            process_document_with_runtime(&markdown, "report.md", "report.md", false, &cfg, None)
                .unwrap();
        assert_eq!(enhanced.markdown, markdown, "{window:?}");
        // One request per chunk the rule plans: one without a window or with
        // a large one, several for a 2,000-token window.
        let planned = chunks::Protected::new(&markdown)
            .split_within(
                chunks::limit(&cfg, || document::prompt_tokens("report.md", false, &cfg)).unwrap(),
            )
            .len();
        assert_eq!(echo.count(), planned, "{window:?}");
        assert_eq!(planned > 1, window == Some(2_000), "{window:?}");
    }
    // A window the prompt alone fills is refused before any request.
    let echo = Echo::new();
    let Err(Error::Config(message)) = process_document_with_runtime(
        &markdown,
        "report.md",
        "report.md",
        false,
        &with_window(Some(300), &echo.base),
        None,
    ) else {
        panic!("a window without room must be refused");
    };
    assert!(message.contains("max_input_tokens is 300"), "{message}");
    assert_eq!(echo.count(), 0);
}

#[test]
fn browser_literal_connections_ignore_canary_environment_after_normalize() {
    let vars = [
        ("OLLAMA_API_KEY", "authored-server-canary"),
        ("OLLAMA_API_BASE", "http://127.0.0.1:9/stolen"),
    ];
    let explicit = resolved(
        json!({"model":"ollama/test","use_environment_credentials":false}),
        &vars,
    )
    .unwrap();
    assert!(explicit.key.is_none());
    assert!(!explicit.endpoint.contains(":9/"));
    let explicit = resolved(json!({"model":"ollama/test","api_base":"http://127.0.0.1:9911/v1","api_key":"own-literal-key","use_environment_credentials":false}), &vars).unwrap();
    assert_eq!(explicit.key.as_deref(), Some("own-literal-key"));
    assert!(explicit.endpoint.starts_with("http://127.0.0.1:9911/"));
    let legacy = resolved(json!({"model":"ollama/test"}), &vars).unwrap();
    assert_eq!(legacy.key.as_deref(), Some("authored-server-canary"));
}

#[test]
fn browser_literal_provider_mode_survives_configuration_reload() {
    let raw = json!({"llm":{"providers":[{"id":"local","provider":"ollama","api_base":"http://127.0.0.1:9911","use_environment_credentials":false}],"model_list":[{"model_name":"default","litellm_params":{"model":"ollama/test"},"model_info":{"provider_id":"local"}}]}});
    let saved = serde_json::to_string(&config::normalize(&raw).unwrap()).unwrap();
    let cfg: Value = serde_json::from_str(&saved).unwrap();
    let entries =
        deployments(&cfg, &vars(&[("OLLAMA_API_KEY", "authored-server-canary")])).unwrap();
    assert!(entries[0].key.is_none());
    assert!(entries[0].endpoint.starts_with("http://127.0.0.1:9911/"));
    let mut invalid = raw;
    invalid["llm"]["providers"][0]["use_environment_credentials"] = json!("false");
    assert_eq!(
        config::normalize(&invalid).unwrap()["llm"]["providers"][0]["use_environment_credentials"],
        false,
    );
    invalid["llm"]["providers"][0]["use_environment_credentials"] = json!("not-a-boolean");
    assert!(config::normalize(&invalid).is_err());
}
