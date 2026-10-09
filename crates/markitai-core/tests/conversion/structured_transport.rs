use super::vision_processing::{Server, cfg as base_cfg, reply};
use super::*;
use std::path::Path;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

fn isolated(name: &str) -> bool {
    let exact = format!("structured_transport::{name}");
    if std::env::var("MARKITAI_STRUCTURED_TEST").as_deref() == Ok(&exact) {
        return false;
    }
    let dir = tempfile::tempdir().unwrap();
    let state = dir.path().join("state");
    std::fs::create_dir(&state).unwrap();
    let out = dir.path().join("stdout");
    let err = dir.path().join("stderr");
    let mut cmd = Command::new(std::env::current_exe().unwrap());
    cmd.args(["--exact", &exact, "--nocapture"])
        .env_clear()
        .env("MARKITAI_STRUCTURED_TEST", &exact)
        .env("MARKITAI_HOME", state)
        .current_dir(dir.path())
        .stdout(Stdio::from(std::fs::File::create(&out).unwrap()))
        .stderr(Stdio::from(std::fs::File::create(&err).unwrap()));
    for name in [
        "HOME",
        "PATH",
        "USERPROFILE",
        "SYSTEMROOT",
        "WINDIR",
        "LANG",
        "LC_ALL",
        "TZ",
    ] {
        if let Some(value) = std::env::var_os(name) {
            cmd.env(name, value);
        }
    }
    let mut child = cmd.spawn().unwrap();
    let deadline = Instant::now() + Duration::from_secs(90);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if Instant::now() > deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!("isolated test timed out: {exact}");
        }
        thread::sleep(Duration::from_millis(10));
    };
    let stdout = std::fs::read_to_string(out).unwrap();
    let stderr = std::fs::read_to_string(err).unwrap();
    assert!(status.success(), "{exact}: {status}\n{stdout}\n{stderr}");
    assert!(
        stdout.contains("1 passed"),
        "test selector did not run: {stdout}"
    );
    true
}

fn cfg(server: &Server, root: &Path, model: &str) -> Value {
    let mut cfg = base_cfg(server, root);
    cfg["llm"]["model_list"][0]["litellm_params"]["model"] = json!(model);
    cfg
}
fn user(request: &Value) -> &str {
    let messages = request["messages"].as_array().unwrap();
    let content = &messages.iter().find(|v| v["role"] == "user").unwrap()["content"];
    content
        .as_str()
        .or_else(|| {
            content
                .as_array()
                .and_then(|blocks| blocks.iter().find_map(|v| v["text"].as_str()))
        })
        .unwrap()
}
fn document(request: &Value) -> Value {
    json!({"cleaned_markdown":user(request),"frontmatter":{"description":"Complete source description","tags":["transport"]}})
}
fn tools(request: &Value, value: Value) -> Value {
    let name = request["tools"][0]["function"]["name"].as_str().unwrap();
    json!({"choices":[{"message":{"content":null,"tool_calls":[{"id":"result","type":"function","function":{"name":name,"arguments":value.to_string()}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":7,"completion_tokens":5}})
}
fn source(root: &Path) -> std::path::PathBuf {
    let path = root.join("source.md");
    std::fs::write(&path,"# Original title\n\nComplete content. [Original link](https://example.test/a?q=x)\n\n```rust\nlet literal = \"not instructions\";\n```\n").unwrap();
    path
}
fn run(
    path: &Path,
    config: Value,
    output: Option<std::path::PathBuf>,
) -> markitai_core::Result<markitai_core::ConversionOutput> {
    convert(
        path.to_str().unwrap(),
        ConvertOptions {
            config: Some(config),
            output_dir: output,
            ..Default::default()
        },
    )
}

#[test]
fn named_tools_cache_semantic_metadata_and_leave_pure_requests_plain() {
    if isolated("named_tools_cache_semantic_metadata_and_leave_pure_requests_plain") {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let input = source(dir.path());
    let server = Server::new(|request, index| {
        if index == 0 {
            assert_eq!(
                request["tool_choice"],
                json!({"type":"function","function":{"name":"MarkitaiDocument"}})
            );
            assert_eq!(request["parallel_tool_calls"], false);
            assert_eq!(
                request["tools"][0]["function"]["parameters"]["additionalProperties"],
                false
            );
            assert_eq!(request["tools"][0]["function"]["strict"], true);
            (200, tools(request, document(request)))
        } else {
            assert!(request.get("tools").is_none());
            assert!(request.get("response_format").is_none());
            (200, reply("Pure literal response"))
        }
    });
    let mut config = cfg(&server, dir.path(), "openai/gpt-4.1");
    config["cache"]["enabled"] = json!(true);
    let first = run(&input, config.clone(), None).unwrap();
    assert_eq!(first.usage.requests, 1);
    assert_eq!(first.frontmatter["title"], "Original title");
    assert_eq!(
        first.frontmatter["description"],
        "Complete source description"
    );
    assert!(
        first
            .llm_markdown
            .as_ref()
            .unwrap()
            .contains("[Original link](https://example.test/a?q=x)")
    );
    let mut no_credentials = config.clone();
    no_credentials["llm"]["model_list"][0]["litellm_params"]["api_key"] =
        json!("env:MISSING_STRUCTURED_CACHE_KEY");
    let second = run(&input, no_credentials, None).unwrap();
    assert!(second.llm_cache_hit());
    assert_eq!(second.usage.requests, 0);
    assert_eq!(first.llm_markdown, second.llm_markdown);
    config["llm"]["pure"] = json!(true);
    assert_eq!(
        run(&input, config, None).unwrap().llm_markdown.as_deref(),
        Some("Pure literal response")
    );
    assert_eq!(server.count(), 2);
}

#[test]
fn rejected_wire_fields_descend_once_and_keep_all_paid_usage() {
    if isolated("rejected_wire_fields_descend_once_and_keep_all_paid_usage") {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let input = source(dir.path());
    let server = Server::new(|request, index| match index {
        0 => {
            assert!(request.get("tools").is_some());
            (
                400,
                json!({"error":{"param":"tool_choice","message":"Unsupported tool choice"},"usage":{"prompt_tokens":2,"completion_tokens":1}}),
            )
        }
        1 => {
            assert!(request.get("tools").is_none());
            assert_eq!(request["response_format"]["type"], "json_schema");
            (
                422,
                json!({"error":{"param":"response_format.json_schema","message":"Not supported"},"usage":{"prompt_tokens":3,"completion_tokens":2}}),
            )
        }
        _ => {
            assert!(request.get("tools").is_none());
            assert!(request.get("response_format").is_none());
            (200, reply(&document(request).to_string()))
        }
    });
    let mut config = cfg(&server, dir.path(), "openai/gpt-4.1-2025-04-14");
    config["llm"]["router_settings"]["num_retries"] = json!(5);
    config["cache"]["enabled"] = json!(true);
    let first = run(&input, config.clone(), None).unwrap();
    assert_eq!(server.count(), 3);
    assert_eq!(first.usage.requests, 3);
    assert_eq!(first.usage.input_tokens, 12);
    assert_eq!(first.usage.output_tokens, 8);
    let calls = server.requests.lock().unwrap();
    assert_eq!(calls[0]["messages"], calls[1]["messages"]);
    assert_eq!(calls[1]["messages"], calls[2]["messages"]);
    drop(calls);
    let cached = run(&input, config, None).unwrap();
    assert!(cached.llm_cache_hit());
    assert_eq!(cached.usage.requests, 0);
    assert_eq!(cached.llm_markdown, first.llm_markdown);
    assert_eq!(server.count(), 3);
}

#[test]
fn anthropic_native_schema_uses_native_shapes_and_tokens() {
    if isolated("anthropic_native_schema_uses_native_shapes_and_tokens") {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let input = source(dir.path());
    // These Anthropic entries start at native JSON schema; Haiku 5.5 starts
    // at a forced tool (below).
    let server = Server::new(|request, _| {
        assert!(request.get("response_format").is_none());
        assert!(request.get("tools").is_none());
        assert_eq!(request["output_config"]["format"]["type"], "json_schema");
        (
            200,
            json!({"content":[{"type":"text","text":document(request).to_string()}],"stop_reason":"end_turn","usage":{"input_tokens":5,"output_tokens":4}}),
        )
    });
    for model in [
        "anthropic/claude-haiku-4-5-20251001",
        "anthropic/claude-sonnet-5-5",
    ] {
        let result = run(&input, cfg(&server, dir.path(), model), None).unwrap();
        assert_eq!(result.usage.input_tokens, 5);
        assert_eq!(result.usage.requests, 1);
        assert_eq!(result.frontmatter["tags"], json!(["transport"]));
    }
    assert_eq!(server.count(), 2);
}

#[test]
fn claude_haiku_5_5_starts_with_a_forced_native_tool() {
    if isolated("claude_haiku_5_5_starts_with_a_forced_native_tool") {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let input = source(dir.path());
    let server = Server::new(|request, _| {
        assert_eq!(request["model"], "claude-haiku-5-5");
        assert!(request.get("output_config").is_none());
        assert_eq!(
            request["tool_choice"],
            json!({"type":"tool","name":"MarkitaiDocument","disable_parallel_tool_use":true})
        );
        (
            200,
            json!({"content":[{"type":"tool_use","id":"toolu_1","name":"MarkitaiDocument","input":document(request)}],"stop_reason":"tool_use","usage":{"input_tokens":5,"output_tokens":4}}),
        )
    });
    let result = run(
        &input,
        cfg(&server, dir.path(), "anthropic/claude-haiku-5-5"),
        None,
    )
    .unwrap();
    assert_eq!(result.usage.requests, 1);
    assert_eq!(result.frontmatter["tags"], json!(["transport"]));
    // Accepted at once: no rejection and no second request (the only
    // warning is the unknown price).
    assert!(
        result
            .warnings
            .iter()
            .all(|warning| !warning.contains("rejected") && !warning.contains("sent again")),
        "{:?}",
        result.warnings
    );
    assert_eq!(server.count(), 1);
}

#[test]
fn reachable_mixed_pool_uses_text_but_disabled_and_unresolved_members_do_not_lower_it() {
    if isolated(
        "reachable_mixed_pool_uses_text_but_disabled_and_unresolved_members_do_not_lower_it",
    ) {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let input = source(dir.path());
    let server = Server::new(|request, index| {
        if index == 0 {
            assert!(request.get("tools").is_none());
            assert!(request.get("response_format").is_none());
            (200, reply(&document(request).to_string()))
        } else {
            assert!(request.get("tools").is_some());
            (200, tools(request, document(request)))
        }
    });
    let mut config = cfg(&server, dir.path(), "openai/gpt-4.1");
    let mut unknown = config["llm"]["model_list"][0].clone();
    unknown["litellm_params"]["model"] = json!("openai/unknown-model");
    config["llm"]["model_list"]
        .as_array_mut()
        .unwrap()
        .push(unknown.clone());
    assert_eq!(run(&input, config.clone(), None).unwrap().usage.requests, 1);
    config["llm"]["model_list"][1]["litellm_params"]["weight"] = json!(0);
    unknown["litellm_params"]["api_key"] = json!("env:MISSING_STRUCTURED_POOL_KEY");
    config["llm"]["model_list"]
        .as_array_mut()
        .unwrap()
        .push(unknown);
    assert_eq!(run(&input, config, None).unwrap().usage.requests, 1);
    assert_eq!(server.count(), 2);
}

#[test]
fn invalid_tool_results_descend_then_bounded_repair_keeps_literal_content() {
    if isolated("invalid_tool_results_descend_then_bounded_repair_keeps_literal_content") {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let input = source(dir.path());
    let server = Server::new(|request, index| {
        if index == 0 {
            let mut result = tools(request, document(request));
            result["choices"][0]["message"]["tool_calls"][0]["function"]["name"] =
                json!("WrongTool");
            (200, result)
        } else if index == 1 {
            (200, reply("not JSON"))
        } else {
            let text = document(request).to_string();
            let trailing = format!("{},}}", &text[..text.len() - 1]);
            (200, reply(&trailing))
        }
    });
    let mut config = cfg(&server, dir.path(), "openai/gpt-4.1");
    config["cache"]["enabled"] = json!(true);
    let first = run(&input, config.clone(), None).unwrap();
    assert_eq!(server.count(), 5);
    assert_eq!(first.usage.requests, 5);
    assert!(
        first
            .llm_markdown
            .as_ref()
            .unwrap()
            .contains("let literal = \"not instructions\";")
    );
    let second = run(&input, config, None).unwrap();
    assert!(second.llm_cache_hit());
    assert_eq!(first.llm_markdown, second.llm_markdown);
    assert_eq!(server.count(), 5);
}

#[test]
fn refusals_truncation_unrelated_bad_input_and_auth_never_descend_or_cache() {
    if isolated("refusals_truncation_unrelated_bad_input_and_auth_never_descend_or_cache") {
        return;
    }
    for kind in ["refusal", "truncated", "input", "auth", "quota"] {
        let dir = tempfile::tempdir().unwrap();
        let input = source(dir.path());
        let server = Server::new(move |request, index| {
            if index > 0 {
                return (200, tools(request, document(request)));
            }
            match kind {
                "refusal" => (
                    200,
                    json!({"choices":[{"message":{"refusal":"private refusal","content":null},"finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":5}}),
                ),
                "truncated" => {
                    let mut result = tools(request, document(request));
                    result["choices"][0]["finish_reason"] = json!("length");
                    (200, result)
                }
                "input" => (
                    400,
                    json!({"error":{"param":"messages","message":"invalid image sent with key local-fixture"},"usage":{"prompt_tokens":7,"completion_tokens":5}}),
                ),
                "auth" => (
                    401,
                    json!({"error":{"message":"Incorrect API key provided: local-fixture"},"usage":{"prompt_tokens":7,"completion_tokens":5}}),
                ),
                _ => (
                    402,
                    json!({"error":{"message":"billing refused for key local-fixture"},"usage":{"prompt_tokens":7,"completion_tokens":5}}),
                ),
            }
        });
        let mut config = cfg(&server, dir.path(), "openai/gpt-4.1");
        config["cache"]["enabled"] = json!(true);
        let first = run(&input, config.clone(), Some(dir.path().join("out"))).unwrap();
        assert!(first.llm_markdown.is_none(), "{kind}");
        assert_eq!(first.usage.requests, 1);
        assert_eq!(server.count(), 1);
        assert!(
            first
                .warnings
                .iter()
                .all(|value| !value.contains("local-fixture"))
        );
        let second = run(&input, config.clone(), None).unwrap();
        assert!(!second.llm_cache_hit());
        assert_eq!(second.usage.requests, 1);
        let third = run(&input, config, None).unwrap();
        assert!(third.llm_cache_hit());
        assert_eq!(third.usage.requests, 0);
        assert_eq!(server.count(), 2);
    }
}

#[test]
fn protocol_downgrade_does_not_reset_document_request_budget() {
    if isolated("protocol_downgrade_does_not_reset_document_request_budget") {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let input = source(dir.path());
    let server = Server::new(|_, _| {
        (
            400,
            json!({"error":{"param":"tool_choice","message":"unsupported"},"usage":{"prompt_tokens":7,"completion_tokens":5}}),
        )
    });
    let mut config = cfg(&server, dir.path(), "openai/gpt-4.1");
    config["llm"]["max_requests_per_document"] = json!(1);
    let result = run(&input, config, Some(dir.path().join("out"))).unwrap();
    assert!(result.llm_markdown.is_none());
    assert_eq!(result.usage.requests, 1);
    assert_eq!(server.count(), 1);
    assert!(
        result
            .warnings
            .iter()
            .any(|value| value.contains("budget exhausted"))
    );
}

#[test]
fn image_analysis_tools_publish_real_captions_and_sidecar() {
    if isolated("image_analysis_tools_publish_real_captions_and_sidecar") {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("chart.png");
    image::RgbImage::from_pixel(100, 100, image::Rgb([20, 40, 200]))
        .save(&input)
        .unwrap();
    let server = Server::new(|request, _| {
        assert_eq!(
            request["tools"][0]["function"]["name"],
            "MarkitaiImageAnalysis"
        );
        assert_eq!(
            request["tools"][0]["function"]["parameters"]["required"],
            json!(["caption", "description", "extracted_text"])
        );
        assert!(
            request["messages"][1]["content"]
                .as_array()
                .unwrap()
                .iter()
                .any(|v| v["type"] == "image_url")
        );
        (
            200,
            tools(
                request,
                json!({"caption":"Blue chart","description":"A blue square chart.","extracted_text":"Visible content"}),
            ),
        )
    });
    let mut config = cfg(&server, dir.path(), "openai/gpt-4.1");
    config["image"]["alt_enabled"] = json!(true);
    config["image"]["desc_enabled"] = json!(true);
    let result = run(&input, config, Some(dir.path().join("out"))).unwrap();
    assert_eq!(server.count(), 1);
    assert_eq!(result.usage.requests, 1);
    assert_eq!(result.images.len(), 1);
    assert!(result.llm_markdown.unwrap().contains("Blue chart"));
    assert!(Path::new(result.images[0]["asset"].as_str().unwrap()).is_file());
    let sidecar = Path::new(result.images[0]["asset"].as_str().unwrap())
        .parent()
        .unwrap()
        .join("images.json");
    let metadata: Value = serde_json::from_slice(&std::fs::read(sidecar).unwrap()).unwrap();
    assert_eq!(metadata["images"].as_array().unwrap().len(), 1);
}

#[test]
fn first_visual_batch_uses_tools_and_later_cleaners_remain_plain() {
    if isolated("first_visual_batch_uses_tools_and_later_cleaners_remain_plain") {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("pages.tiff");
    let mut data = std::io::Cursor::new(Vec::new());
    {
        let mut encoder = tiff::encoder::TiffEncoder::new(&mut data).unwrap();
        for n in 1..=11 {
            encoder
                .write_image::<tiff::encoder::colortype::RGB8>(
                    32,
                    32,
                    image::RgbImage::from_pixel(32, 32, image::Rgb([n, 40, 200])).as_raw(),
                )
                .unwrap();
        }
    }
    std::fs::write(&input, data.into_inner()).unwrap();
    let server = Server::new(|request, index| {
        if index == 0 {
            assert!(
                request["messages"][0]["content"]
                    .as_str()
                    .unwrap()
                    .contains("MARKITAI_VISION_JSON_V1")
            );
            assert_eq!(
                request["messages"][1]["content"].as_array().unwrap().len(),
                11
            );
            (200, tools(request, document(request)))
        } else {
            assert!(
                request["messages"][0]["content"]
                    .as_str()
                    .unwrap()
                    .contains("MARKITAI_VISION_CLEAN_V1")
            );
            assert!(request.get("tools").is_none());
            assert!(request.get("response_format").is_none());
            assert_eq!(
                request["messages"][1]["content"].as_array().unwrap().len(),
                2
            );
            (200, reply(user(request)))
        }
    });
    let result = run(&input, cfg(&server, dir.path(), "openai/gpt-4.1"), None).unwrap();
    assert_eq!(server.count(), 2);
    assert_eq!(result.usage.requests, 2);
    let body = result.llm_markdown.unwrap();
    for n in 1..=11 {
        assert_eq!(
            body.matches(&format!("<!-- Page number: {n} -->")).count(),
            1
        );
    }
}

#[test]
fn exhausted_transport_never_restarts_the_ladder_or_image_plain_fallback() {
    if isolated("exhausted_transport_never_restarts_the_ladder_or_image_plain_fallback") {
        return;
    }
    // Exercise both a tools-capable pool and an unknown/text-only pool: losing
    // the terminal classification on the latter used to restart image analysis.
    for model in ["openai/gpt-4.1", "openai/fixture"] {
        for image in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let input = if image {
                let path = dir.path().join("chart.png");
                image::RgbImage::from_pixel(100, 100, image::Rgb([20, 40, 200]))
                    .save(&path)
                    .unwrap();
                path
            } else {
                source(dir.path())
            };
            let server = Server::new(|_, _| {
                (
                    503,
                    json!({"error":{"message":"Temporary upstream outage"},"usage":{"prompt_tokens":7,"completion_tokens":5}}),
                )
            });
            let mut config = cfg(&server, dir.path(), model);
            config["llm"]["router_settings"]["num_retries"] = json!(0);
            config["image"]["alt_enabled"] = json!(image);
            config["image"]["desc_enabled"] = json!(image);
            let output = run(&input, config, Some(dir.path().join("out"))).unwrap();
            assert_eq!(server.count(), 1, "{model}, image={image}");
            assert_eq!(output.usage.requests, 1);
            assert_eq!(output.usage.input_tokens, 7);
            assert_eq!(output.usage.output_tokens, 5);
            assert!(output.llm_markdown.is_none());
            assert!(output.images.is_empty());
            assert!(
                output
                    .warnings
                    .iter()
                    .any(|warning| warning.contains("503"))
            );
        }
    }
}

#[test]
fn oversized_response_header_stops_before_body_read_or_protocol_retry() {
    if isolated("oversized_response_header_stops_before_body_read_or_protocol_retry") {
        return;
    }
    use std::sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    };
    let dir = tempfile::tempdir().unwrap();
    let input = source(dir.path());
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let stop = Arc::new(AtomicBool::new(false));
    let stopping = stop.clone();
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = calls.clone();
    let worker = thread::spawn(move || {
        while !stopping.load(Ordering::Acquire) {
            let (mut stream, _) = match listener.accept() {
                Ok(value) => value,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(2));
                    continue;
                }
                Err(error) => panic!("header fixture accept: {error}"),
            };
            stream.set_nonblocking(false).unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let mut request_reader = bounded_fixture_io::Reader::new(
                &stream,
                std::time::Instant::now() + Duration::from_secs(5),
            );
            stream
                .set_write_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let mut bytes = Vec::new();
            let mut buffer = [0u8; 8192];
            loop {
                let n = request_reader.read(&mut buffer).unwrap();
                assert!(n > 0);
                bytes.extend_from_slice(&buffer[..n]);
                if let Some(at) = bytes.windows(4).position(|v| v == b"\r\n\r\n") {
                    let length = String::from_utf8_lossy(&bytes[..at])
                        .lines()
                        .find_map(|line| {
                            line.split_once(':')
                                .filter(|(name, _)| name.eq_ignore_ascii_case("content-length"))
                                .map(|(_, length)| length.trim().parse::<usize>().unwrap())
                        })
                        .unwrap();
                    if bytes.len() >= at + 4 + length {
                        break;
                    }
                }
                assert!(bytes.len() < 1_000_000);
            }
            observed.fetch_add(1, Ordering::SeqCst);
            // No large allocation or payload is needed to verify preflight.
            write!(stream,"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",100*1024*1024+1).unwrap();
        }
    });
    let config = json!({"prompts":{"dir":dir.path().join("prompts")},"cache":{"enabled":false},"ocr":{"enabled":false},"image":{"alt_enabled":false,"desc_enabled":false},"llm":{"enabled":true,"on_failure":"fallback","router_settings":{"timeout":5,"num_retries":2},"model_list":[{"model_name":"default","litellm_params":{"model":"openai/gpt-4.1","api_key":"fixture","api_base":base}}]}});
    let result = run(&input, config, Some(dir.path().join("out")));
    stop.store(true, Ordering::Release);
    worker.join().unwrap();
    let output = result.unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(output.llm_markdown.is_none());
    assert_eq!(output.usage.requests, 0);
    assert!(
        output
            .warnings
            .iter()
            .any(|warning| warning.contains("100 MiB"))
    );
}

#[cfg(test)]
mod bounded_fixture_io {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/support/bounded_read.rs"
    ));
}
