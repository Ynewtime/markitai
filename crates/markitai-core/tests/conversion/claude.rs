#![cfg(unix)]
use markitai_core::{
    ConvertContext, ConvertOptions, LlmRuntime, convert_detailed, convert_json,
    convert_with_context_detailed, provider_management,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

fn isolated(name: &str) -> bool {
    let exact = format!("claude::{name}");
    if std::env::var("MARKITAI_CLAUDE_TEST").as_deref() == Ok(&exact) {
        return false;
    }
    let root = tempfile::tempdir().unwrap();
    let executable = root.path().join("fake claude");
    std::fs::write(&executable, include_str!("claude_fixture.py")).unwrap();
    std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700)).unwrap();
    let stdout = root.path().join("stdout");
    let stderr = root.path().join("stderr");
    let mut command = Command::new(std::env::current_exe().unwrap());
    command
        .args(["--exact", &exact, "--nocapture"])
        .env_clear()
        .env("MARKITAI_CLAUDE_TEST", &exact)
        .env("MARKITAI_HOME", root.path().join("state"))
        .env("CLAUDE_CONFIG_DIR", root.path())
        .env("CLAUDE_CLI_PATH", executable)
        .env("CLAUDE_CODE_OAUTH_TOKEN", "must-not-enter-child-runtime")
        .env("OPENAI_API_KEY", "must-not-enter-child-runtime")
        .current_dir(root.path())
        .stdout(Stdio::from(std::fs::File::create(&stdout).unwrap()))
        .stderr(Stdio::from(std::fs::File::create(&stderr).unwrap()));
    for key in ["HOME", "PATH", "LANG", "LC_ALL", "TMPDIR"] {
        if let Some(value) = std::env::var_os(key) {
            command.env(key, value);
        }
    }
    let mut child = command.spawn().unwrap();
    let start = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if start.elapsed() > Duration::from_secs(60) {
            let _ = child.kill();
            let _ = child.wait();
            panic!("private Claude test timed out: {exact}");
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    let out = std::fs::read_to_string(stdout).unwrap();
    let err = std::fs::read_to_string(stderr).unwrap();
    assert!(status.success(), "{status}\n{out}\n{err}");
    assert!(out.contains("1 passed"), "{out}");
    true
}
fn root() -> PathBuf {
    PathBuf::from(std::env::var_os("CLAUDE_CONFIG_DIR").unwrap())
}
fn mode(value: &str) {
    std::fs::write(
        root().join("fixture.json"),
        json!({"mode":value,"home":std::env::var("HOME").ok()}).to_string(),
    )
    .unwrap();
}
fn config() -> Value {
    json!({"log":{"dir":null},"prompts":{"dir":root().join("prompts")},"history":{"record":false},"cache":{"enabled":true,"global_dir":root().join("cache")},"image":{"compress":false,"alt_enabled":false,"desc_enabled":false},"llm":{"enabled":true,"keep_base":true,"on_failure":"fail","router_settings":{"num_retries":2,"timeout":5},"model_list":[{"model_name":"default","litellm_params":{"model":"claude-agent/sonnet"},"model_info":{"supports_vision":true}}]}})
}
fn options(cfg: Value) -> ConvertOptions {
    ConvertOptions {
        config: Some(cfg),
        ..Default::default()
    }
}
fn source() -> PathBuf {
    let path = root().join("source.md");
    std::fs::write(&path,"# Authored title\n\n完整段落🚀。\n\n```rust\nlet secret = \"literal\";\n```\n\n[kept](https://example.test/path)\n\nTHE END\n").unwrap();
    path
}
fn calls() -> Vec<Value> {
    std::fs::read_to_string(root().join("requests.jsonl"))
        .unwrap_or_default()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}
fn sends() -> usize {
    calls()
        .iter()
        .filter(|call| call.get("request").is_some())
        .count()
}
fn assert_unknown(usage: &markitai_core::ConversionUsage, requests: u64) {
    assert_eq!(usage.requests, requests);
    assert_eq!(usage.input_tokens, requests * 16);
    assert_eq!(usage.output_tokens, requests * 7);
    assert_eq!(usage.cost_usd, 0.0);
    let row = &usage.by_model["claude-agent/claude-fixture-1"];
    assert_eq!(row["unpriced_requests"], requests);
    assert_eq!(row["cost_status"], "unknown");
    assert_eq!(row["incomplete_request_observations"], requests);
    assert!(!usage.cost_complete());
    assert!(row.get("pricing_snapshot").is_none());
}

#[test]
fn text_pure_and_document_processing_use_runtime_without_api_key() {
    if isolated("text_pure_and_document_processing_use_runtime_without_api_key") {
        return;
    }
    mode("ok");
    let path = source();
    let result = convert_detailed(path.to_str().unwrap(), options(config())).unwrap();
    let output = result.llm_markdown.unwrap();
    assert!(output.contains("完整段落🚀"));
    assert!(output.contains("let secret = \"literal\";"));
    assert!(output.contains("[kept](https://example.test/path)"));
    assert!(output.contains("THE END"));
    assert_eq!(result.frontmatter["title"], "Authored title");
    assert_eq!(
        result.frontmatter["description"],
        "Authored subscription fixture."
    );
    assert_unknown(&result.usage, 1);
    let mut cfg = config();
    cfg["llm"]["pure"] = json!(true);
    let pure = convert_detailed(path.to_str().unwrap(), options(cfg)).unwrap();
    assert!(pure.llm_markdown.unwrap().contains("THE END"));
    assert_unknown(&pure.usage, 1);
    assert_eq!(sends(), 2);
    let systems: Vec<_> = calls()
        .into_iter()
        .filter_map(|call| {
            call.pointer("/request/system")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .collect();
    assert!(systems[0].contains("MARKITAI_DOCUMENT_JSON_V1"));
    assert!(!systems[1].contains("MARKITAI_DOCUMENT_JSON_V1"));
}
#[test]
fn repeated_and_same_runtime_conversions_do_not_cache_or_merge_accounts() {
    if isolated("repeated_and_same_runtime_conversions_do_not_cache_or_merge_accounts") {
        return;
    }
    mode("ok");
    let path = source();
    let runtime = LlmRuntime::new(2).unwrap();
    for _ in 0..2 {
        let result = convert_with_context_detailed(
            path.to_str().unwrap(),
            options(config()),
            ConvertContext {
                llm_runtime: Some(&runtime),
                ..Default::default()
            },
        )
        .unwrap();
        assert_unknown(&result.usage, 1);
        assert_ne!(result.skip_reason.as_deref(), Some("llm_cache"));
    }
    let results = std::thread::scope(|scope| {
        let a = scope.spawn(|| {
            convert_with_context_detailed(
                path.to_str().unwrap(),
                options(config()),
                ConvertContext {
                    llm_runtime: Some(&runtime),
                    ..Default::default()
                },
            )
            .unwrap()
        });
        let b = scope.spawn(|| {
            convert_with_context_detailed(
                path.to_str().unwrap(),
                options(config()),
                ConvertContext {
                    llm_runtime: Some(&runtime),
                    ..Default::default()
                },
            )
            .unwrap()
        });
        [a.join().unwrap(), b.join().unwrap()]
    });
    for result in results {
        assert_unknown(&result.usage, 1);
    }
    assert_eq!(sends(), 4);
}
#[test]
fn runtime_terminal_error_and_semantic_retries_retain_observed_usage() {
    if isolated("runtime_terminal_error_and_semantic_retries_retain_observed_usage") {
        return;
    }
    mode("paid-auth-error");
    let path = source();
    let result: Value = serde_json::from_str(&convert_json(
        &json!({"source":path,"options":{"config":config()}}).to_string(),
    ))
    .unwrap();
    assert_eq!(result["ok"], false);
    assert_eq!(result["error"]["usage"]["requests"], 1);
    assert_eq!(result["error"]["usage"]["input_tokens"], 16);
    assert_eq!(sends(), 1);
    assert!(!result.to_string().contains("DO NOT ECHO SECRET"));
    mode("invalid");
    let failed = convert_detailed(path.to_str().unwrap(), options(config())).unwrap_err();
    assert_unknown(&failed.usage, 3);
    assert_eq!(sends(), 4);
}
#[test]
fn unknown_tariff_budget_blocks_before_runtime_start() {
    if isolated("unknown_tariff_budget_blocks_before_runtime_start") {
        return;
    }
    mode("ok");
    let path = source();
    let mut cfg = config();
    cfg["llm"]["max_cost_per_document_usd"] = json!(1.0);
    let failed = convert_detailed(path.to_str().unwrap(), options(cfg)).unwrap_err();
    assert!(failed.error.to_string().contains("verified tariff"));
    assert_eq!(failed.usage.requests, 0);
    assert!(calls().is_empty());
}
#[test]
fn image_bytes_reach_runtime_as_ordered_png_blobs() {
    if isolated("image_bytes_reach_runtime_as_ordered_png_blobs") {
        return;
    }
    mode("ok");
    let path = root().join("pixels.png");
    image::RgbImage::from_pixel(4, 3, image::Rgb([91, 42, 223]))
        .save(&path)
        .unwrap();
    let data = std::fs::read(&path).unwrap();
    let mut fixture: Value =
        serde_json::from_slice(&std::fs::read(root().join("fixture.json")).unwrap()).unwrap();
    fixture["image_sha256"] = json!(markitai_core::hex(Sha256::digest(&data)));
    std::fs::write(root().join("fixture.json"), fixture.to_string()).unwrap();
    let result = convert_detailed(path.to_str().unwrap(), options(config())).unwrap();
    assert!(result.llm_markdown.is_some());
    assert_unknown(&result.usage, 1);
    let request = calls()
        .into_iter()
        .find(|call| call.get("request").is_some())
        .unwrap();
    assert_eq!(request["request"]["images"].as_array().unwrap().len(), 1);
    assert_eq!(request["request"]["images"][0]["mime"], "image/png");
    assert_eq!(
        request["request"]["images"][0]["sha256"],
        markitai_core::hex(Sha256::digest(&data))
    );
}
#[test]
fn discovery_and_probe_use_official_protocol_without_http_or_cached_identity() {
    if isolated("discovery_and_probe_use_official_protocol_without_http_or_cached_identity") {
        return;
    }
    mode("ok");
    for _ in 0..2 {
        let found = provider_management::discover(&json!({"provider":"claude-agent"})).unwrap();
        assert_eq!(found["status"], "ok");
        assert_eq!(found["source"], "official_cli");
        assert_eq!(found["cached"], false);
        assert_eq!(found["models"][0]["model"], "claude-agent/sonnet");
    }
    assert_eq!(
        calls()
            .iter()
            .filter(|call| call["args"]
                .as_array()
                .is_some_and(|args| args.iter().any(|arg| arg == "--print")))
            .count(),
        2
    );
    assert_eq!(
        provider_management::probe(&json!({"model":"claude-agent/sonnet"})).unwrap()["ok"],
        true
    );
    assert_eq!(sends(), 1);
    assert!(
        provider_management::discover(
            &json!({"provider":"claude-agent","api_base":"https://example.invalid"})
        )
        .is_err()
    );
    assert_eq!(sends(), 1);
}
#[test]
fn absent_runtime_usage_is_unknown_without_fabricated_paid_requests() {
    if isolated("absent_runtime_usage_is_unknown_without_fabricated_paid_requests") {
        return;
    }
    mode("unknown-usage");
    let path = source();
    let result = convert_detailed(path.to_str().unwrap(), options(config())).unwrap();
    assert_eq!(result.usage.requests, 0);
    assert!(result.usage.by_model.is_empty());
    assert!(
        result
            .warnings
            .iter()
            .any(|warning| warning.contains("unknown"))
    );
    assert_eq!(sends(), 1);
}

#[test]
fn aggregate_only_success_and_terminal_failure_retain_tokens_without_invented_requests() {
    if isolated(
        "aggregate_only_success_and_terminal_failure_retain_tokens_without_invented_requests",
    ) {
        return;
    }
    let path = source();
    for mode_name in ["aggregate-only", "aggregate-paid-error"] {
        mode(mode_name);
        let json: Value = serde_json::from_str(&convert_json(
            &json!({"source":path,"options":{"config":config()}}).to_string(),
        ))
        .unwrap();
        assert_eq!(json["ok"], mode_name == "aggregate-only");
        let usage = if mode_name == "aggregate-only" {
            &json["result"]["usage"]
        } else {
            &json["error"]["usage"]
        };
        assert_eq!(usage["requests"], 0);
        assert_eq!(usage["input_tokens"], 16);
        assert_eq!(usage["output_tokens"], 7);
        assert_eq!(
            usage["by_model"]["claude-agent/claude-fixture-1"]["incomplete_request_observations"],
            1
        );
        assert_eq!(
            usage["by_model"]["claude-agent/claude-fixture-1"]["cost_status"],
            "unknown"
        );
    }
    assert_eq!(sends(), 2);
}
#[test]
fn conflicting_runtime_totals_stop_without_replay_and_preserve_observed_usage() {
    if isolated("conflicting_runtime_totals_stop_without_replay_and_preserve_observed_usage") {
        return;
    }
    mode("aggregate-conflict");
    let path = source();
    let failure = convert_detailed(path.to_str().unwrap(), options(config())).unwrap_err();
    assert_unknown(&failure.usage, 1);
    assert!(
        failure
            .error
            .to_string()
            .contains("aggregate usage conflicts")
    );
    assert_eq!(sends(), 1);
}
#[test]
fn unsupported_subscription_overrides_fail_before_starting_official_runtime() {
    if isolated("unsupported_subscription_overrides_fail_before_starting_official_runtime") {
        return;
    }
    mode("ok");
    let path = source();
    for key in ["api_base", "api_key", "max_tokens"] {
        let mut cfg = config();
        cfg["llm"]["model_list"][0]["litellm_params"][key] = if key == "max_tokens" {
            json!(40)
        } else {
            json!("unsupported")
        };
        let failed = convert_detailed(path.to_str().unwrap(), options(cfg)).unwrap_err();
        assert!(failed.error.to_string().contains(key));
        assert_eq!(failed.usage.requests, 0);
    }
    assert!(calls().is_empty());
}

#[test]
fn preparation_retains_base_and_paid_failure_without_early_publication() {
    if isolated("preparation_retains_base_and_paid_failure_without_early_publication") {
        return;
    }
    struct Writer;
    impl markitai_core::output::Publication for Writer {
        fn skip_existing(&self) -> bool {
            false
        }
        fn publish(&self, path: &std::path::Path, bytes: &[u8]) -> markitai_core::Result<()> {
            std::fs::write(path, bytes)?;
            Ok(())
        }
    }
    mode("aggregate-paid-error");
    let path = source();
    let make = |directory: &str| {
        markitai_core::prepare_with_publication(
            path.to_str().unwrap(),
            ConvertOptions {
                output_dir: Some(root().join(directory)),
                config: Some(config()),
                ..Default::default()
            },
            ConvertContext::default(),
            &Writer,
        )
    };
    let mut prepared = make("deferred");
    assert_eq!(sends(), 1);
    assert_eq!(prepared.members().len(), 1);
    assert!(!prepared.members()[0].path.exists());
    let members = prepared.take_members();
    for member in &members {
        std::fs::write(&member.path, &member.bytes).unwrap();
    }
    let failure = prepared.finish_after_publication().unwrap_err();
    assert_eq!(
        (
            failure.usage.requests,
            failure.usage.input_tokens,
            failure.usage.output_tokens
        ),
        (0, 16, 7)
    );
    assert_eq!(
        failure.usage.by_model["claude-agent/claude-fixture-1"]["incomplete_request_observations"],
        1
    );
    assert!(!failure.usage.cost_complete());
    assert!(
        std::fs::read_to_string(&members[0].path)
            .unwrap()
            .contains("THE END")
    );

    let prepared = make("publication-failed");
    let destination = prepared.members()[0].path.clone();
    let failure = prepared.fail_publication(markitai_core::Error::Conversion(
        "authored publication refusal".into(),
    ));
    assert!(
        failure
            .error
            .to_string()
            .contains("authored publication refusal")
    );
    assert_eq!(
        (
            failure.usage.requests,
            failure.usage.input_tokens,
            failure.usage.output_tokens
        ),
        (0, 16, 7)
    );
    assert!(!destination.exists());

    let prepared = make("immediate");
    let destination = prepared.members()[0].path.clone();
    let failure = prepared.publish_immediately(&Writer).unwrap_err();
    assert_eq!(
        (
            failure.usage.requests,
            failure.usage.input_tokens,
            failure.usage.output_tokens
        ),
        (0, 16, 7)
    );
    assert!(
        std::fs::read_to_string(destination)
            .unwrap()
            .contains("THE END")
    );
    assert_eq!(sends(), 3);
}
