//! A deployment refused for authentication is skipped for the rest of a run
//! while a sibling in its group can still serve requests.
use super::*;
use std::sync::atomic::AtomicBool;

// Weighted selection then picks the refused deployment first with
// probability 1 - 1e-12, so a missing exclusion shows up as a repeat visit.
const FIRST: u64 = 1_000_000_000_000;

/// Answers every request with one scripted response and counts them.
struct Repeat {
    base: String,
    count: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}
impl Repeat {
    fn new(status: u16, body: Value) -> Self {
        Self::slow(status, body, Duration::ZERO)
    }
    /// Answers after `delay`, keeping the caller's runtime permit busy.
    fn slow(status: u16, body: Value, delay: Duration) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let base = format!("http://{}/v1", listener.local_addr().unwrap());
        let count = Arc::new(AtomicUsize::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let (seen, done) = (count.clone(), stop.clone());
        let thread = thread::spawn(move || {
            let body = body.to_string();
            while !done.load(Ordering::Acquire) {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        read_request(&mut stream);
                        seen.fetch_add(1, Ordering::AcqRel);
                        thread::sleep(delay);
                        write!(stream, "HTTP/1.1 {status} Mock\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2))
                    }
                    Err(error) => panic!("mock LLM accept failed: {error}"),
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
impl Drop for Repeat {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            let result = thread.join();
            if !thread::panicking() {
                result.unwrap();
            }
        }
    }
}

fn pool(rows: &[(&str, &str, u64)]) -> Value {
    let mut cfg = cfg(rows[0].0, rows[0].1);
    cfg["llm"]["model_list"] = rows
        .iter()
        .map(|(model, base, weight)| {
            json!({"model_name":"default","litellm_params":{"model":model,"api_key":"fake-test-key","api_base":base,"weight":weight}})
        })
        .collect();
    cfg
}
fn refused() -> Value {
    json!({"error":{"message":"private-token rejected"}})
}
fn skipped(model: &str) -> String {
    format!("LLM deployment {model} failed authentication and is skipped for this run")
}

/// One document of a run: its own accounting scope, the run's shared runtime.
fn document_with(
    cfg: &Value,
    env: &HashMap<String, String>,
    runtime: &LlmRuntime,
) -> (Result<(String, ConversionUsage)>, Vec<String>) {
    let _scope = DocumentScope::new(cfg);
    let result = run_with_runtime(
        &plain(),
        cfg,
        env,
        &mut |_| panic!("an authentication fallback never sleeps"),
        Some(runtime),
    );
    (result, take_document_warnings())
}
fn document(cfg: &Value, runtime: &LlmRuntime) -> (Result<(String, ConversionUsage)>, Vec<String>) {
    document_with(cfg, &HashMap::new(), runtime)
}

#[test]
fn authentication_failure_moves_to_a_sibling_at_once_and_stays_excluded_for_the_run() {
    for status in [401, 403] {
        let bad = Repeat::new(status, refused());
        let good = Repeat::new(200, success("# cleaned"));
        let mut cfg = pool(&[
            ("openai/bad", &bad.base, FIRST),
            ("openai/good", &good.base, 1),
        ]);
        // No transport retry is needed to reach the sibling.
        cfg["llm"]["router_settings"]["num_retries"] = json!(0);
        let runtime = LlmRuntime::new(2).unwrap();
        let (result, warnings) = document(&cfg, &runtime);
        let (text, usage) = result.unwrap();
        assert_eq!(text, "# cleaned");
        assert_eq!(
            (usage.requests, usage.input_tokens, usage.output_tokens),
            (1, 11, 7)
        );
        assert_eq!(warnings, vec![skipped("openai/bad")]);
        for private in ["private-token", "fake-test-key", "127.0.0.1", "http"] {
            assert!(!warnings[0].contains(private), "{private}");
        }
        assert_eq!((bad.count(), good.count()), (1, 1));
        // Later documents of the same run skip it and do not warn again.
        for served in 2..=3 {
            let (result, warnings) = document(&cfg, &runtime);
            assert_eq!(result.unwrap().0, "# cleaned");
            assert!(warnings.is_empty());
            assert_eq!((bad.count(), good.count()), (1, served));
        }
        // An independent run starts without exclusions.
        let (result, warnings) = document(&cfg, &LlmRuntime::new(1).unwrap());
        assert!(result.is_ok());
        assert_eq!(warnings, vec![skipped("openai/bad")]);
        assert_eq!((bad.count(), good.count()), (2, 4));
    }
}

#[test]
fn every_deployment_refused_leaves_the_error_to_configured_fallbacks() {
    let first = Repeat::new(401, refused());
    let second = Repeat::new(403, refused());
    let backup = Repeat::new(200, success("backup"));
    let mut cfg = pool(&[
        ("openai/first", &first.base, 1),
        ("openai/second", &second.base, 1),
    ]);
    let runtime = LlmRuntime::new(1).unwrap();
    let (result, mut warnings) = document(&cfg, &runtime);
    let error = result.unwrap_err().to_string();
    assert!(error.contains("LLM returned HTTP 40"), "{error}");
    assert!(!error.contains("private-token"));
    warnings.sort();
    assert_eq!(
        warnings,
        vec![skipped("openai/first"), skipped("openai/second")]
    );
    assert_eq!((first.count(), second.count()), (1, 1));
    // The excluded group fails at once on a later document, without HTTP.
    let (result, warnings) = document(&cfg, &runtime);
    assert!(result.unwrap_err().to_string().contains(
        "Every LLM deployment of model group 'default' failed authentication earlier in this run"
    ));
    assert!(warnings.is_empty());
    assert_eq!((first.count(), second.count()), (1, 1));

    cfg["llm"]["model_list"].as_array_mut().unwrap().push(json!({"model_name":"backup","litellm_params":{"model":"openai/backup","api_key":"fake-test-key","api_base":backup.base}}));
    cfg["llm"]["router_settings"]["fallbacks"] = json!([{"default":["backup"]}]);
    let (result, _) = document(&cfg, &runtime);
    assert_eq!(result.unwrap().0, "backup");
    assert_eq!((first.count(), second.count(), backup.count()), (1, 1, 1));
    // A fresh run visits each refused deployment once, then the fallback group.
    let (result, warnings) = document(&cfg, &LlmRuntime::new(1).unwrap());
    assert_eq!(result.unwrap().0, "backup");
    assert_eq!(warnings.len(), 2);
    assert_eq!((first.count(), second.count(), backup.count()), (2, 2, 2));
}

#[test]
fn billing_failures_stay_terminal_and_exclude_nothing() {
    for (status, body) in [
        (402, json!({"error":"payment required"})),
        (403, json!({"error":{"code":"insufficient_quota"}})),
        (401, json!({"error":"billing account is inactive"})),
    ] {
        let bad = Repeat::new(status, body);
        let good = Repeat::new(200, success("unused"));
        let cfg = pool(&[
            ("openai/bad", &bad.base, FIRST),
            ("openai/good", &good.base, 1),
        ]);
        let runtime = LlmRuntime::new(1).unwrap();
        for visits in 1..=2 {
            let (result, warnings) = document(&cfg, &runtime);
            let error = result.unwrap_err().to_string();
            assert!(error.contains(&format!("HTTP {status}")), "{error}");
            assert!(warnings.is_empty());
            assert_eq!((bad.count(), good.count()), (visits, 0), "{status}");
        }
    }
}

#[test]
fn recognized_refusals_name_a_fixed_cause_but_never_the_provider_wording() {
    // OpenRouter answers a regional block with 403, which is not a credential problem.
    let region = json!({"error":{"message":"This model is not available in your region. private-token","code":403}});
    let blocked = Repeat::new(403, region.clone());
    let good = Repeat::new(200, success("# cleaned"));
    let mut cfg = pool(&[
        ("openai/blocked", &blocked.base, FIRST),
        ("openai/good", &good.base, 1),
    ]);
    cfg["llm"]["router_settings"]["num_retries"] = json!(0);
    let (result, warnings) = document(&cfg, &LlmRuntime::new(1).unwrap());
    assert_eq!(result.unwrap().0, "# cleaned");
    assert_eq!(
        warnings,
        [
            "LLM deployment openai/blocked is not available in this region and is skipped for this run"
        ]
    );

    let quota = "the account's quota or billing does not allow this request";
    for (status, body, reason) in [
        (403, region, REGION_UNAVAILABLE),
        (
            403,
            json!({"error":{"code":"unsupported_country_region_territory","message":"Country, region, or territory not supported"}}),
            REGION_UNAVAILABLE,
        ),
        (
            400,
            json!({"error":{"status":"FAILED_PRECONDITION","message":"User location is not supported for the API use."}}),
            REGION_UNAVAILABLE,
        ),
        (
            429,
            json!({"error":{"code":"insufficient_quota","message":"You exceeded your current quota. private-token"}}),
            quota,
        ),
        (
            404,
            json!({"error":{"code":"model_not_found","message":"The model private-token does not exist"}}),
            "the model is unavailable",
        ),
    ] {
        let only = Repeat::new(status, body);
        let mut cfg = pool(&[("openai/only", &only.base, 1)]);
        cfg["llm"]["router_settings"]["num_retries"] = json!(0);
        let (result, _) = document(&cfg, &LlmRuntime::new(1).unwrap());
        let error = result.unwrap_err().to_string();
        assert!(
            error.contains(&format!("LLM returned HTTP {status}: {reason}")),
            "{error}"
        );
        assert!(!error.contains("private-token"), "{error}");
    }
    // Unrecognized bodies keep the bare status.
    let only = Repeat::new(401, refused());
    let (result, _) = document(
        &pool(&[("openai/only", &only.base, 1)]),
        &LlmRuntime::new(1).unwrap(),
    );
    assert!(
        result
            .unwrap_err()
            .to_string()
            .ends_with("LLM returned HTTP 401")
    );
}

#[test]
fn a_group_of_one_deployment_keeps_its_previous_policy() {
    let only = Repeat::new(401, refused());
    let cfg = pool(&[("openai/only", &only.base, 1)]);
    let runtime = LlmRuntime::new(1).unwrap();
    for visits in 1..=2 {
        let (result, warnings) = document(&cfg, &runtime);
        assert!(result.unwrap_err().to_string().contains("HTTP 401"));
        assert!(warnings.is_empty());
        assert_eq!(only.count(), visits);
    }
    // Identical entries are one deployment identity, not a sibling.
    let duplicate = pool(&[
        ("openai/only", &only.base, 1),
        ("openai/only", &only.base, 1),
    ]);
    let (result, warnings) = document(&duplicate, &runtime);
    assert!(result.unwrap_err().to_string().contains("HTTP 401"));
    assert!(warnings.is_empty());
    assert_eq!(only.count(), 3);
    // An unavailable model cannot succeed on retry, while a temporary status
    // keeps its backoff and retry allowance even when its body names the model.
    for (status, pauses_expected, visits) in [
        (404, vec![], 1),
        (403, vec![], 1),
        (503, vec![Duration::from_secs(1), Duration::from_secs(2)], 3),
    ] {
        let unavailable = Repeat::new(status, json!({"error":{"code":"model_not_found"}}));
        let cfg = pool(&[("openai/only", &unavailable.base, 1)]);
        let _scope = DocumentScope::new(&cfg);
        let mut pauses = Vec::new();
        let error = run_with_runtime(
            &plain(),
            &cfg,
            &HashMap::new(),
            &mut |pause| pauses.push(pause),
            Some(&runtime),
        )
        .unwrap_err();
        assert!(error.to_string().contains(&format!("HTTP {status}")));
        assert_eq!(pauses, pauses_expected, "{status}");
        assert_eq!(unavailable.count(), visits, "{status}");
        assert!(take_document_warnings().is_empty());
    }
}

#[test]
fn an_unavailable_model_moves_to_a_sibling_at_once_and_stays_excluded_for_the_run() {
    for body in [
        json!({"error":{"code":"model_not_found","message":"private-token"}}),
        json!({"error":{"status":"FAILED_PRECONDITION","message":"User location is not supported for the API use."}}),
    ] {
        let missing = Repeat::new(400, body.clone());
        let good = Repeat::new(200, success("# cleaned"));
        let mut cfg = pool(&[
            ("openai/missing", &missing.base, FIRST),
            ("openai/good", &good.base, 1),
        ]);
        cfg["llm"]["router_settings"]["num_retries"] = json!(0);
        let runtime = LlmRuntime::new(1).unwrap();
        for served in 1..=2 {
            let (result, warnings) = document(&cfg, &runtime);
            assert_eq!(result.unwrap().0, "# cleaned");
            if served == 1 {
                let refusal = if body["error"]["code"] == "model_not_found" {
                    "is unavailable"
                } else {
                    "is not available in this region"
                };
                assert_eq!(
                    warnings,
                    [format!(
                        "LLM deployment openai/missing {refusal} and is skipped for this run"
                    )]
                );
            } else {
                assert!(warnings.is_empty());
            }
            assert_eq!((missing.count(), good.count()), (1, served));
        }
    }
}

#[test]
fn routing_strategies_never_select_an_excluded_deployment() {
    for strategy in [
        "simple-shuffle",
        "least-busy",
        "usage-based-routing",
        "latency-based-routing",
    ] {
        let bad = Repeat::new(401, refused());
        let good = Repeat::new(200, success("# routed"));
        let mut cfg = pool(&[
            ("openai/bad", &bad.base, FIRST),
            ("openai/good", &good.base, 1),
        ]);
        cfg["llm"]["router_settings"]["routing_strategy"] = json!(strategy);
        cfg["llm"]["router_settings"]["num_retries"] = json!(0);
        let runtime = LlmRuntime::new(2).unwrap();
        for _ in 0..4 {
            let (result, _) = document(&cfg, &runtime);
            let (text, usage) = result.unwrap();
            assert_eq!(text, "# routed");
            assert_eq!(usage.requests, 1);
        }
        // Least-busy and usage ties prefer the refused first candidate, and an
        // unmeasured latency score of zero beats the served sibling, so each
        // strategy would revisit it on every document without the exclusion.
        assert_eq!((bad.count(), good.count()), (1, 4), "{strategy}");
    }
}

#[test]
fn successful_usage_matches_a_healthy_pool_and_paid_refusals_still_count() {
    let good = Repeat::new(200, success("# cleaned"));
    let (healthy, _) = document(
        &pool(&[("openai/good", &good.base, 1)]),
        &LlmRuntime::new(1).unwrap(),
    );
    let healthy = serde_json::to_value(healthy.unwrap().1).unwrap();
    let bad = Repeat::new(401, refused());
    let (rerouted, _) = document(
        &pool(&[
            ("openai/bad", &bad.base, FIRST),
            ("openai/good", &good.base, 1),
        ]),
        &LlmRuntime::new(1).unwrap(),
    );
    assert_eq!(serde_json::to_value(rerouted.unwrap().1).unwrap(), healthy);
    assert_eq!(bad.count(), 1);
    // Structured usage on the refused response is recorded exactly as before.
    let paid = Repeat::new(
        401,
        json!({"error":"private-token","usage":{"prompt_tokens":3,"completion_tokens":1}}),
    );
    let (result, _) = document(
        &pool(&[
            ("openai/paid", &paid.base, FIRST),
            ("openai/good", &good.base, 1),
        ]),
        &LlmRuntime::new(1).unwrap(),
    );
    let usage = result.unwrap().1;
    assert_eq!(
        (usage.requests, usage.input_tokens, usage.output_tokens),
        (2, 14, 8)
    );
    assert_eq!(usage.by_model["openai/paid"]["requests"], 1);
    assert_eq!(usage.by_model["actual-model"]["requests"], 1);
    assert_eq!(good.count(), 3);
}

#[test]
fn visual_cancellation_is_published_only_when_no_sibling_can_serve() {
    let bad = Repeat::new(401, refused());
    let good = Repeat::new(200, success("# page"));
    let cfg = pool(&[
        ("openai/bad", &bad.base, FIRST),
        ("openai/good", &good.base, 1),
    ]);
    let stop = AtomicBool::new(false);
    let _scope = DocumentScope::new(&cfg);
    let (text, _) = run_controlled(
        &plain(),
        &cfg,
        &HashMap::new(),
        &mut |_| panic!("no backoff"),
        Some(&LlmRuntime::new(1).unwrap()),
        Some(&stop),
    )
    .unwrap();
    assert_eq!(text, "# page");
    assert!(!stop.load(Ordering::Acquire));
    let other = Repeat::new(403, refused());
    let failing = pool(&[
        ("openai/bad", &bad.base, 1),
        ("openai/other", &other.base, 1),
    ]);
    let stop = AtomicBool::new(false);
    run_controlled(
        &plain(),
        &failing,
        &HashMap::new(),
        &mut |_| panic!("no backoff"),
        Some(&LlmRuntime::new(1).unwrap()),
        Some(&stop),
    )
    .unwrap_err();
    assert!(stop.load(Ordering::Acquire));
    assert_eq!((bad.count(), other.count(), good.count()), (2, 1, 1));
}

#[test]
fn typed_documents_report_the_exclusion_once_per_run() {
    let answer = json!({"cleaned_markdown":"# Title\n\nShort body.","frontmatter":{"description":"A short note","tags":["note"]}});
    let bad = Repeat::new(401, refused());
    let good = Repeat::new(200, success(&answer.to_string()));
    let cfg = pool(&[
        ("openai/bad", &bad.base, FIRST),
        ("openai/good", &good.base, 1),
    ]);
    let runtime = LlmRuntime::new(2).unwrap();
    let markdown = "# Title\n\nShort body.";
    let first =
        process_document_with_runtime(markdown, "note.md", "note.md", false, &cfg, Some(&runtime))
            .unwrap();
    assert_eq!(first.markdown, markdown);
    assert_eq!(first.warnings, vec![skipped("openai/bad")]);
    let second = process_document_with_runtime(
        markdown,
        "other.md",
        "other.md",
        false,
        &cfg,
        Some(&runtime),
    )
    .unwrap();
    assert!(second.warnings.is_empty());
    assert_eq!((bad.count(), good.count()), (1, 2));
}

#[cfg(unix)]
#[test]
fn a_refused_subscription_account_moves_to_an_api_deployment() {
    use std::os::unix::fs::PermissionsExt;
    let root = tempfile::tempdir().unwrap();
    let executable = root.path().join("fake claude");
    std::fs::write(
        &executable,
        include_str!("../../../tests/conversion/claude_fixture.py"),
    )
    .unwrap();
    std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700)).unwrap();
    let home = root.path().join("home");
    std::fs::create_dir(&home).unwrap();
    // The runtime answers, but its account is not a first-party subscription.
    std::fs::write(
        root.path().join("fixture.json"),
        json!({"mode":"wrong-provider","home":home}).to_string(),
    )
    .unwrap();
    let mut env = HashMap::new();
    env.insert("CLAUDE_CLI_PATH".into(), executable.display().to_string());
    env.insert(
        "CLAUDE_CONFIG_DIR".into(),
        root.path().display().to_string(),
    );
    env.insert("HOME".into(), home.display().to_string());
    env.insert("PATH".into(), std::env::var("PATH").unwrap_or_default());
    let good = Repeat::new(200, success("# served"));
    let mut cfg = pool(&[("openai/good", &good.base, 1)]);
    cfg["llm"]["model_list"] = json!([
        {"model_name":"default","litellm_params":{"model":"claude-agent/sonnet"}},
        {"model_name":"default","litellm_params":{"model":"openai/good","api_key":"fake-test-key","api_base":good.base}}
    ]);
    // Least-busy ties keep candidate order, so the subscription runs first.
    cfg["llm"]["router_settings"]["routing_strategy"] = json!("least-busy");
    cfg["llm"]["router_settings"]["num_retries"] = json!(0);
    let runtime = LlmRuntime::new(1).unwrap();
    let launches = || {
        std::fs::read_to_string(root.path().join("requests.jsonl"))
            .unwrap_or_default()
            .lines()
            .count()
    };
    let (result, warnings) = document_with(&cfg, &env, &runtime);
    assert_eq!(result.unwrap().0, "# served");
    assert_eq!(warnings, vec![skipped("claude-agent/sonnet")]);
    let started = launches();
    assert!(started > 0);
    let (result, warnings) = document_with(&cfg, &env, &runtime);
    assert_eq!(result.unwrap().0, "# served");
    assert!(warnings.is_empty());
    assert_eq!(launches(), started);
    assert_eq!(good.count(), 2);
}

#[test]
fn concurrent_refusals_of_one_deployment_warn_once() {
    // Hold both refusals until both requests have selected the deployment, so
    // two documents observe the same authentication failure at the same time.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let base = format!("http://{}/v1", listener.local_addr().unwrap());
    let held = thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut streams = Vec::new();
        while streams.len() < 2 {
            match listener.accept() {
                Ok((mut stream, _)) => {
                    read_request(&mut stream);
                    streams.push(stream);
                }
                Err(error)
                    if error.kind() == std::io::ErrorKind::WouldBlock
                        && Instant::now() < deadline =>
                {
                    thread::sleep(Duration::from_millis(2))
                }
                other => panic!("both refused requests must arrive: {other:?}"),
            }
        }
        let body = refused().to_string();
        for mut stream in streams {
            write!(stream, "HTTP/1.1 401 Mock\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
        }
    });
    let good = Repeat::new(200, success("# both"));
    let cfg = pool(&[("openai/bad", &base, FIRST), ("openai/good", &good.base, 1)]);
    let runtime = LlmRuntime::new(2).unwrap();
    let start = Barrier::new(2);
    let warnings: Vec<_> = thread::scope(|scope| {
        let workers: Vec<_> = (0..2)
            .map(|_| {
                scope.spawn(|| {
                    start.wait();
                    let (result, warnings) = document(&cfg, &runtime);
                    assert_eq!(result.unwrap().0, "# both");
                    warnings
                })
            })
            .collect();
        workers
            .into_iter()
            .flat_map(|worker| worker.join().unwrap())
            .collect()
    });
    held.join().unwrap();
    assert_eq!(warnings, vec![skipped("openai/bad")]);
    assert_eq!(good.count(), 2);
}

#[test]
fn documents_queued_behind_a_refusal_never_select_the_refused_deployment() {
    // One permit: the second document waits while the first holds the slow
    // refusal, as concurrent batch files do behind a capped runtime.
    let bad = Repeat::slow(401, refused(), Duration::from_millis(300));
    let good = Repeat::new(200, success("# queued"));
    let cfg = pool(&[
        ("openai/bad", &bad.base, FIRST),
        ("openai/good", &good.base, 1),
    ]);
    let runtime = LlmRuntime::new(1).unwrap();
    let start = Barrier::new(2);
    let warnings: Vec<_> = thread::scope(|scope| {
        let workers: Vec<_> = (0..2)
            .map(|_| {
                scope.spawn(|| {
                    start.wait();
                    let (result, warnings) = document(&cfg, &runtime);
                    assert_eq!(result.unwrap().0, "# queued");
                    warnings
                })
            })
            .collect();
        workers
            .into_iter()
            .flat_map(|worker| worker.join().unwrap())
            .collect()
    });
    assert_eq!(warnings, vec![skipped("openai/bad")]);
    assert_eq!((bad.count(), good.count()), (1, 2));
}

#[test]
fn visual_documents_report_the_exclusion_once_per_run() {
    let answer = json!({"cleaned_markdown":"Page text.","frontmatter":{"description":"A page","tags":["page"]}});
    let bad = Repeat::new(403, refused());
    let good = Repeat::new(200, success(&answer.to_string()));
    let cfg = pool(&[
        ("openai/bad", &bad.base, FIRST),
        ("openai/good", &good.base, 1),
    ]);
    let runtime = LlmRuntime::new(2).unwrap();
    let frames = [VisionFrame {
        number: 1,
        mime: "image/png",
        bytes: b"authored image bytes",
    }];
    let request = || VisionRequest {
        markdown: "Page text.",
        source_label: "page.pdf",
        cache_context: "page.pdf",
        kind: VisionKind::PagedDocument,
        frames: &frames,
    };
    let first = process_vision_with_runtime(request(), &cfg, Some(&runtime)).unwrap();
    assert_eq!(first.warnings, vec![skipped("openai/bad")]);
    let second = process_vision_with_runtime(request(), &cfg, Some(&runtime)).unwrap();
    assert!(second.warnings.is_empty());
    assert_eq!((bad.count(), good.count()), (1, 2));
}
