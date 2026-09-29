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
    let exact = format!("copilot::{name}");
    if std::env::var("MARKITAI_COPILOT_TEST").as_deref() == Ok(&exact) {
        return false;
    }
    let root = tempfile::tempdir().unwrap();
    let executable = root.path().join("fake copilot");
    std::fs::write(&executable, include_str!("copilot_fixture.py")).unwrap();
    std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700)).unwrap();
    let stdout = root.path().join("stdout");
    let stderr = root.path().join("stderr");
    let mut command = Command::new(std::env::current_exe().unwrap());
    command
        .args(["--exact", &exact, "--nocapture"])
        .env_clear()
        .env("MARKITAI_COPILOT_TEST", &exact)
        .env("MARKITAI_HOME", root.path().join("state"))
        .env("COPILOT_HOME", root.path())
        .env("COPILOT_CLI_PATH", executable)
        .env("COPILOT_GITHUB_TOKEN", "fixture-token")
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
            panic!("private Copilot test timed out: {exact}");
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
    PathBuf::from(std::env::var_os("COPILOT_HOME").unwrap())
}
fn mode(value: &str) {
    std::fs::write(
        root().join("fixture.json"),
        json!({"mode":value,"home":std::env::var("HOME").ok()}).to_string(),
    )
    .unwrap();
}
fn config() -> Value {
    json!({"log":{"dir":null},"prompts":{"dir":root().join("prompts")},"history":{"record":false},"cache":{"enabled":true,"global_dir":root().join("cache")},"image":{"compress":false,"alt_enabled":false,"desc_enabled":false},"llm":{"enabled":true,"keep_base":true,"on_failure":"fail","router_settings":{"num_retries":2,"timeout":5},"model_list":[{"model_name":"default","litellm_params":{"model":"copilot/fixture"},"model_info":{"supports_vision":true}}]}})
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
        .filter(|call| call["method"] == "session.send")
        .count()
}
fn assert_unknown(usage: &markitai_core::ConversionUsage, requests: u64) {
    assert_eq!(usage.requests, requests);
    assert_eq!(usage.input_tokens, requests * 7);
    assert_eq!(usage.output_tokens, requests * 3);
    assert_eq!(usage.cost_usd, 0.0);
    let row = &usage.by_model["copilot/fixture"];
    assert_eq!(row["unpriced_requests"], requests);
    assert_eq!(row["cost_status"], "unknown");
    assert!(row.get("pricing_snapshot").is_none());
}

#[test]
fn text_pure_and_document_processing_use_runtime_without_api_key() {
    if isolated("text_pure_and_document_processing_use_runtime_without_api_key") {
        return;
    }
    mode("normal");
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
        .filter(|call| call["method"] == "session.create")
        .map(|call| {
            call["params"]["systemMessage"]["content"]
                .as_str()
                .unwrap()
                .to_owned()
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
    mode("normal");
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
    mode("paid_error");
    let path = source();
    let result: Value = serde_json::from_str(&convert_json(
        &json!({"source":path,"options":{"config":config()}}).to_string(),
    ))
    .unwrap();
    assert_eq!(result["ok"], false);
    assert_eq!(result["error"]["usage"]["requests"], 1);
    assert_eq!(result["error"]["usage"]["input_tokens"], 7);
    assert_eq!(sends(), 1);
    assert!(!result.to_string().contains("do-not-disclose"));
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
    mode("normal");
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
    mode("normal");
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
        .find(|call| call["method"] == "session.send")
        .unwrap();
    assert_eq!(
        request["params"]["attachments"].as_array().unwrap().len(),
        1
    );
    assert_eq!(request["params"]["attachments"][0]["mimeType"], "image/png");
}
#[test]
fn discovery_and_probe_use_official_protocol_without_http_or_cached_identity() {
    if isolated("discovery_and_probe_use_official_protocol_without_http_or_cached_identity") {
        return;
    }
    mode("normal");
    for _ in 0..2 {
        let found = provider_management::discover(&json!({"provider":"copilot"})).unwrap();
        assert_eq!(found["status"], "ok");
        assert_eq!(found["source"], "official_cli");
        assert_eq!(found["cached"], false);
        assert_eq!(found["models"][0]["model"], "copilot/fixture");
    }
    assert_eq!(
        calls()
            .iter()
            .filter(|call| call["method"] == "models.list")
            .count(),
        2
    );
    assert_eq!(
        provider_management::probe(&json!({"model":"copilot/fixture"})).unwrap()["ok"],
        true
    );
    assert_eq!(sends(), 1);
    assert!(
        provider_management::discover(
            &json!({"provider":"copilot","api_base":"https://example.invalid"})
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
    mode("no_usage");
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
