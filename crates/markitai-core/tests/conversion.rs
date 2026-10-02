use markitai_core::{ConvertOptions, Error, convert, convert_json};
use serde_json::{Value, json};
use std::io::{Read, Write};
use std::net::TcpListener;

#[cfg(target_os = "macos")]
#[path = "conversion/pdf_media.rs"]
mod pdf_media;

#[path = "conversion/url_pdf.rs"]
mod url_pdf;

#[path = "conversion/multipage_image.rs"]
mod multipage_image;

#[path = "conversion/heif_image.rs"]
mod heif_image;

#[path = "conversion/image_enrichment.rs"]
mod image_enrichment;

#[path = "conversion/office_media.rs"]
mod office_media;

#[path = "conversion/numbers.rs"]
mod numbers;

#[path = "conversion/document_processing.rs"]
mod document_processing;

#[path = "conversion/vision_processing.rs"]
mod vision_processing;

#[path = "conversion/browser_auth.rs"]
mod browser_auth;

#[path = "conversion/structured_transport.rs"]
mod structured_transport;

#[path = "conversion/workbook_media.rs"]
mod workbook_media;

#[path = "conversion/browser_pdf.rs"]
mod browser_pdf;

#[path = "conversion/browser_identity.rs"]
mod browser_identity;

#[path = "conversion/terminal_usage.rs"]
mod terminal_usage;

#[path = "conversion/proxy.rs"]
mod proxy;

#[path = "conversion/stdout_assets.rs"]
mod stdout_assets;

#[path = "conversion/legacy_ppt_objects.rs"]
mod legacy_ppt_objects;

#[path = "conversion/html_text.rs"]
mod html_text;

#[path = "conversion/document_breaks.rs"]
mod document_breaks;

fn options() -> ConvertOptions {
    ConvertOptions {
        config: Some(json!({})),
        llm: Some(false),
        ..Default::default()
    }
}

#[test]
fn memory_and_disk_outputs_share_content_and_metadata() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("source.md");
    std::fs::write(
        &input,
        "# 标题\n\nA **paragraph** with a [link](https://example.com).\n",
    )
    .unwrap();
    let memory = convert(input.to_str().unwrap(), options()).unwrap();
    assert!(memory.output_path.is_none());
    assert!(memory.assets.is_empty());
    assert_eq!(memory.frontmatter["source"], "source.md");
    let output_dir = dir.path().join("out");
    let disk = convert(
        input.to_str().unwrap(),
        ConvertOptions {
            output_dir: Some(output_dir.clone()),
            ..options()
        },
    )
    .unwrap();
    assert_eq!(memory.markdown, disk.markdown);
    assert_eq!(disk.output_path, Some(output_dir.join("source.md.md")));
    let written = std::fs::read_to_string(disk.output_path.unwrap()).unwrap();
    assert!(written.starts_with("---\ntitle:"));
    assert!(written.ends_with(&disk.markdown));
}

#[test]
fn inline_data_images_become_assets_only_with_an_output_directory() {
    use base64::Engine;
    let dir = tempfile::tempdir().unwrap();
    let png = |color: [u8; 3]| {
        let mut bytes = std::io::Cursor::new(Vec::new());
        image::RgbImage::from_pixel(120, 80, image::Rgb(color))
            .write_to(&mut bytes, image::ImageFormat::Png)
            .unwrap();
        base64::engine::general_purpose::STANDARD.encode(bytes.into_inner())
    };
    let (red, blue) = (png([200, 40, 40]), png([40, 40, 200]));
    // Enrichment is off: the local file and URL must be neither read nor fetched.
    image::RgbImage::new(120, 80)
        .save(dir.path().join("local.png"))
        .unwrap();
    let text = format!(
        "# Inline\n\n![red](data:image/png;base64,{red})\n\n![again](data:image/png;base64,{red})\n\n\
         ![blue](data:image/png;base64,{blue})\n\n![broken](data:image/png;base64,@@@)\n\n\
         ![local](local.png)\n\n![remote](http://127.0.0.1:9/remote.png)\n"
    );
    let input = dir.path().join("inline.md");
    std::fs::write(&input, &text).unwrap();
    let memory = convert(input.to_str().unwrap(), options()).unwrap();
    assert!(
        memory
            .markdown
            .contains(&format!("![red](data:image/png;base64,{red})"))
    );
    assert!(memory.assets.is_empty());

    let out = dir.path().join("out");
    let disk = convert(
        input.to_str().unwrap(),
        ConvertOptions {
            output_dir: Some(out.clone()),
            ..options()
        },
    )
    .unwrap();
    assert!(!disk.markdown.contains(&red) && !disk.markdown.contains(&blue));
    // Duplicate bytes share one asset; both kept images are real files.
    assert_eq!(disk.assets.len(), 2, "{:?}", disk.assets);
    for asset in &disk.assets {
        // The file path uses the platform's separator throughout; the link to
        // it in Markdown is `/`-separated everywhere.
        #[cfg(windows)]
        assert!(
            !asset.to_string_lossy().contains('/'),
            "{}",
            asset.display()
        );
        let relative = asset
            .strip_prefix(&out)
            .unwrap()
            .iter()
            .map(|part| part.to_str().unwrap())
            .collect::<Vec<_>>()
            .join("/");
        assert!(relative.starts_with(".markitai/assets/"), "{relative}");
        assert!(disk.markdown.contains(&relative), "{}", disk.markdown);
        image::open(asset).unwrap();
    }
    let red_target = disk
        .markdown
        .split("![red](")
        .nth(1)
        .unwrap()
        .split(')')
        .next()
        .unwrap();
    assert!(disk.markdown.contains(&format!("![again]({red_target})")));
    assert!(
        disk.markdown
            .contains("![broken](data:image/png;base64,@@@)")
    );
    assert!(
        disk.warnings
            .iter()
            .any(|w| w.contains("embedded data image"))
    );
    assert!(disk.markdown.contains("![local](local.png)"));
    assert!(
        disk.markdown
            .contains("![remote](http://127.0.0.1:9/remote.png)")
    );
}

#[test]
fn native_request_rejects_unknown_options_and_preserves_error_codes() {
    let missing: Value = serde_json::from_str(&convert_json(
        r#"{"source":"/missing-markitai-fixture.md","options":{"config":{}}}"#,
    ))
    .unwrap();
    assert_eq!(missing["error"]["code"], "not_found");
    let unknown: Value = serde_json::from_str(&convert_json(
        r#"{"source":"x","options":{"silently_ignored":true}}"#,
    ))
    .unwrap();
    assert_eq!(unknown["ok"], false);
}

#[test]
fn image_api_requires_extraction_and_vision_uses_binary_content() {
    use base64::Engine;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("图像.png");
    image::DynamicImage::new_rgb8(80, 80).save(&path).unwrap();
    let error = convert(path.to_str().unwrap(), options()).unwrap_err();
    assert_eq!(error.code(), "conversion_error");
    assert!(error.to_string().contains("llm=True"));
    let (base, server) = llm_server(
        200,
        r##"{"choices":[{"message":{"content":"# 图像\n\nRead text from image."}}]}"##,
    );
    let output = convert(path.to_str().unwrap(), ConvertOptions {
        config: Some(json!({"cache":{"enabled":false},"image":{"compress":false},"llm":{"enabled":true,"router_settings":{"num_retries":0},"model_list":[{"model_name":"vision","litellm_params":{"model":"openai/test-vision","api_base":base,"api_key":"test-key"},"model_info":{"supports_vision":true}}]}})),
        output_dir: Some(dir.path().join("output")),
        ..Default::default()
    }).unwrap();
    let request = server.join().unwrap();
    let content = request["messages"][1]["content"].as_array().unwrap();
    let expected = format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(std::fs::read(&path).unwrap())
    );
    assert!(content.iter().any(|block| {
        block["image_url"]["url"]
            .as_str()
            .is_some_and(|url| url == expected)
    }));
    assert_eq!(output.usage.requests, 1);
    assert!(
        output
            .llm_markdown
            .unwrap()
            .contains("Read text from image.")
    );
    assert_eq!(output.assets.len(), 1);
    assert_eq!(
        std::fs::read(&output.assets[0]).unwrap(),
        std::fs::read(&path).unwrap()
    );
}

#[test]
fn unrelated_features_do_not_reject_text_or_literal_image_examples() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("example.md");
    std::fs::write(&input, "# Text\n\n`![literal](not-an-image.png)`\n").unwrap();
    let cfg = json!({"cache":{"enabled":false},"security":{"pdf_sanitize":"remove"},"ocr":{"enabled":true},"screenshot":{"enabled":true},"image":{"alt_enabled":true,"desc_enabled":true}});
    let plain = convert(
        input.to_str().unwrap(),
        ConvertOptions {
            config: Some(cfg.clone()),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(plain.markdown.contains("![literal]"));
    let (base, server) = llm_server(
        200,
        r##"{"choices":[{"message":{"content":"{protected_input}\n\nKept example."}}]}"##,
    );
    let mut cfg = cfg;
    cfg["llm"] = json!({"enabled":true,"router_settings":{"num_retries":0},"model_list":[{"model_name":"test","litellm_params":{"model":"openai/test","api_base":base,"api_key":"test"}}]});
    let result = convert(
        input.to_str().unwrap(),
        ConvertOptions {
            config: Some(cfg),
            ..Default::default()
        },
    )
    .unwrap();
    server.join().unwrap();
    assert!(result.llm_markdown.unwrap().contains("Kept example."));
}

// Only successful mocks for the explicitly typed protocols acquire metadata.
// Pure calls and deliberately invalid/failing responses retain their old payloads.
fn mock_model_content(request: &Value, markdown: &str) -> String {
    let system = request["messages"][0]["content"].as_str().unwrap_or("");
    let typed_text = system.contains("MARKITAI_DOCUMENT_JSON_V1");
    let typed_vision = system.contains("MARKITAI_VISION_JSON_V1");
    let clean_vision = system.contains("MARKITAI_VISION_CLEAN_V1");
    if !(typed_text || typed_vision || clean_vision) {
        return markdown.to_owned();
    }
    let user = &request["messages"][1]["content"];
    let source = user
        .as_str()
        .or_else(|| {
            user.as_array()?
                .iter()
                .find(|item| item["type"] == "text")?["text"]
                .as_str()
        })
        .unwrap_or("");
    let body = if typed_vision || clean_vision {
        if markdown.contains("{protected_input}") {
            markdown.replace("{protected_input}", source)
        } else {
            format!("{source}\n\n{markdown}")
        }
    } else {
        markdown.replace("{protected_input}", source)
    };
    if clean_vision {
        body
    } else {
        json!({"cleaned_markdown":body,"frontmatter":{
            "description":"Mock document description", "tags":["mock"]
        }})
        .to_string()
    }
}

fn llm_server(status: u16, payload: &'static str) -> (String, std::thread::JoinHandle<Value>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let handle = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(std::time::Duration::from_secs(5)))
            .unwrap();
        let mut request_reader = bounded_fixture_io::Reader::new(
            &stream,
            std::time::Instant::now() + std::time::Duration::from_secs(5),
        );
        let mut bytes = Vec::new();
        let request = loop {
            let mut chunk = [0; 4096];
            let count = request_reader.read(&mut chunk).unwrap();
            assert!(count > 0);
            bytes.extend_from_slice(&chunk[..count]);
            if let Some(offset) = bytes.windows(4).position(|s| s == b"\r\n\r\n") {
                let header = String::from_utf8_lossy(&bytes[..offset]);
                let length = header
                    .lines()
                    .find_map(|line| {
                        line.to_ascii_lowercase()
                            .strip_prefix("content-length: ")
                            .map(str::to_owned)
                    })
                    .unwrap()
                    .parse::<usize>()
                    .unwrap();
                if bytes.len() >= offset + 4 + length {
                    assert!(header.starts_with("POST /v1/chat/completions "));
                    break serde_json::from_slice::<Value>(&bytes[offset + 4..offset + 4 + length])
                        .unwrap();
                }
            }
        };
        let mut payload: Value = serde_json::from_str(payload).unwrap();
        if status == 200
            && let Some(content) = payload.pointer_mut("/choices/0/message/content")
            && let Some(markdown) = content.as_str()
        {
            *content = json!(mock_model_content(&request, markdown));
        }
        let payload = payload.to_string();
        write!(stream,"HTTP/1.1 {status} Mock\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",payload.len()).unwrap();
        request
    });
    (format!("http://{address}/v1"), handle)
}

#[test]
fn llm_native_http_contract_usage_and_failure_policy() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.txt");
    std::fs::write(&input, "Body with facts.").unwrap();
    for (status, payload, policy) in [
        (
            200,
            r##"{"choices":[{"message":{"content":"# Enhanced\n\nBody with facts."}}],"usage":{"prompt_tokens":12,"completion_tokens":9}}"##,
            "fallback",
        ),
        (401, r#"{"error":{"message":"denied"}}"#, "fallback"),
        (500, r#"{"error":{"message":"unavailable"}}"#, "fail"),
    ] {
        let (base, server) = llm_server(status, payload);
        let cfg = json!({"cache":{"enabled":false},"llm":{"enabled":true,"on_failure":policy,"model_list":[{"model_name":"test","litellm_params":{"model":"openai/test-model","api_base":base,"api_key":"test-key"}}]}});
        let result = convert(
            input.to_str().unwrap(),
            ConvertOptions {
                config: Some(cfg),
                output_dir: Some(dir.path().join(format!("out-{status}"))),
                ..Default::default()
            },
        );
        let request = server.join().unwrap();
        assert_eq!(request["model"], "test-model");
        assert_eq!(request["messages"][1]["content"], "Body with facts.\n");
        if status == 200 {
            let result = result.unwrap();
            assert_eq!(result.usage.input_tokens, 12);
            assert_eq!(result.usage.output_tokens, 9);
            assert!(result.output_path.is_none());
            assert!(result.llm_output_path.unwrap().is_file());
        } else if policy == "fallback" {
            let result = result.unwrap();
            assert!(result.llm_markdown.is_none());
            assert!(!result.warnings.is_empty());
            assert!(result.output_path.unwrap().is_file());
        } else {
            assert!(matches!(result, Err(Error::Conversion(_))));
            assert!(
                dir.path()
                    .join(format!("out-{status}/input.txt.md"))
                    .is_file()
            );
        }
    }
}

#[test]
fn persistent_document_cache_keeps_host_json_stable_and_new_usage_zero() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("source.md");
    std::fs::write(&input, "# Original\n\nCache contract.\n").unwrap();
    let (base, server) = llm_server(
        200,
        r##"{"choices":[{"message":{"content":"# Enhanced\n\nCached body."},"finish_reason":"stop"}],"usage":{"prompt_tokens":12,"completion_tokens":8}}"##,
    );
    let mut cfg = json!({"cache":{"enabled":true,"global_dir":dir.path().join("state")},"llm":{"enabled":true,"on_failure":"fail","router_settings":{"num_retries":0},"model_list":[{"model_name":"cache","litellm_params":{"model":"openai/cache-contract","api_base":base,"api_key":"local-mock"}}]}});
    let first = convert(
        input.to_str().unwrap(),
        ConvertOptions {
            config: Some(cfg.clone()),
            ..Default::default()
        },
    )
    .unwrap();
    server.join().unwrap();
    assert!(!first.llm_cache_hit());
    assert_eq!(first.usage.requests, 1);
    cfg["llm"]["model_list"][0]["litellm_params"]["api_key"] =
        json!("os.environ/MARKITAI_MISSING_CACHE_CONTRACT_KEY");
    let second = convert(
        input.to_str().unwrap(),
        ConvertOptions {
            config: Some(cfg.clone()),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(second.llm_cache_hit());
    assert_eq!(first.llm_markdown, second.llm_markdown);
    assert_eq!(second.usage.requests, 0);
    assert_eq!(second.usage.input_tokens, 0);
    assert!(second.usage.by_model.is_empty());
    let request = json!({"source":input,"options":{"config":cfg}}).to_string();
    let response: Value = serde_json::from_str(&convert_json(&request)).unwrap();
    assert_eq!(response["ok"], true);
    assert!(response["result"].get("llm_cache_hit").is_none());
    assert!(response["result"].get("cache_hit").is_none());
    assert_eq!(response["result"]["usage"]["requests"], 0);
}

#[test]
fn pure_llm_preserves_raw_input_and_uses_metadata_from_retained_outputs() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.md");
    let original = "---\ntitle: Input\ncustom: original\n---\n\n# Body\ntext  ";
    std::fs::write(&input, original).unwrap();
    for (index, payload, keep_base, expected) in [
        (
            0,
            r#"{"choices":[{"message":{"content":"---\ntitle: Model\nsource: Model source\n---\n\nModel body  "}}]}"#,
            false,
            json!({"title":"Model","source":"Model source"}),
        ),
        (
            1,
            r#"{"choices":[{"message":{"content":"Model body  "}}]}"#,
            false,
            json!({}),
        ),
        (
            2,
            r#"{"choices":[{"message":{"content":"Model body  "}}]}"#,
            true,
            json!({"title":"Input","custom":"original"}),
        ),
    ] {
        let (base, server) = llm_server(200, payload);
        let result = convert(input.to_str().unwrap(), ConvertOptions {
            output_dir: Some(dir.path().join(format!("out-{index}"))),
            config: Some(json!({"cache":{"enabled":false},"llm":{"enabled":true,"pure":true,"keep_base":keep_base,"model_list":[{"model_name":"mock","litellm_params":{"model":"openai/mock","api_base":base}}]}})),
            ..Default::default()
        }).unwrap();
        let request = server.join().unwrap();
        assert_eq!(request["messages"][1]["content"], original);
        assert_eq!(serde_json::to_value(&result.frontmatter).unwrap(), expected);
        let response: Value = serde_json::from_str(payload).unwrap();
        let written = std::fs::read_to_string(result.llm_output_path.unwrap()).unwrap();
        assert_eq!(
            written,
            response["choices"][0]["message"]["content"]
                .as_str()
                .unwrap()
        );
        if keep_base {
            assert_eq!(
                std::fs::read_to_string(result.output_path.unwrap()).unwrap(),
                original
            );
        }
    }
}

#[test]
fn profiles_run_after_llm_and_each_retained_file_keeps_its_own_metadata() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.md");
    std::fs::write(&input, "# Base\n\n<!-- Page number: 1 -->\n\nBody").unwrap();
    let (base, server) = llm_server(
        200,
        r#"{"choices":[{"message":{"content":"{protected_input}\n\nNew body"}}]}"#,
    );
    let result = convert(input.to_str().unwrap(), ConvertOptions {
        output_dir: Some(dir.path().join("out")),
        config: Some(json!({"cache":{"enabled":false},"output":{"profile":"rag"},"llm":{"enabled":true,"keep_base":true,"model_list":[{"model_name":"mock","litellm_params":{"model":"openai/mock","api_base":base}}]}})),
        ..Default::default()
    }).unwrap();
    assert!(
        server.join().unwrap()["messages"][1]["content"]
            .as_str()
            .unwrap()
            .contains("⟦MKTI:")
    );
    assert_eq!(result.frontmatter["title"], "Base");
    assert_eq!(
        result.frontmatter["description"],
        "Mock document description"
    );
    assert!(result.markdown.contains("<!-- page: 1 -->"));
    assert!(result.llm_markdown.unwrap().contains("<!-- page: 1 -->"));
    let base = std::fs::read_to_string(result.output_path.unwrap()).unwrap();
    assert_eq!(
        markitai_core::output::split_frontmatter(&base).0["title"],
        "Base"
    );
    assert!(
        !markitai_core::output::split_frontmatter(&base)
            .0
            .contains_key("description")
    );
}

#[cfg(unix)]
#[test]
fn output_parent_symlinks_do_not_modify_external_files() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.md");
    std::fs::write(&input, "content").unwrap();
    let outside = dir.path().join("outside");
    std::fs::create_dir(&outside).unwrap();
    let link = dir.path().join("link");
    std::os::unix::fs::symlink(&outside, &link).unwrap();
    assert!(
        convert(
            input.to_str().unwrap(),
            ConvertOptions {
                output_dir: Some(link),
                ..options()
            }
        )
        .is_err()
    );
    assert_eq!(std::fs::read_dir(outside).unwrap().count(), 0);
}

#[test]
fn invalid_override_structure_is_an_error_and_skip_avoids_parsing_or_llm() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("broken.ipynb");
    std::fs::write(&input, "not JSON").unwrap();
    let invalid = convert(
        input.to_str().unwrap(),
        ConvertOptions {
            config: Some(json!({"output":false})),
            profile: Some("rag".into()),
            ..Default::default()
        },
    );
    assert!(matches!(invalid, Err(Error::Config(_))));
    let output = dir.path().join("out");
    std::fs::create_dir(&output).unwrap();
    std::fs::write(output.join("broken.ipynb.llm.md"), "must survive").unwrap();
    let skipped = convert(
        input.to_str().unwrap(),
        ConvertOptions {
            output_dir: Some(output.clone()),
            config: Some(json!({"output":{"on_conflict":"skip"},"llm":{"enabled":true}})),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(skipped.skip_reason.as_deref(), Some("exists"));
    assert!(skipped.markdown.is_empty());
    assert!(skipped.llm_markdown.is_none());
    assert_eq!(skipped.usage.requests, 0);
    assert_eq!(
        std::fs::read_to_string(output.join("broken.ipynb.llm.md")).unwrap(),
        "must survive"
    );
}

#[test]
fn url_names_and_explicit_pure_output_follow_existing_contracts() {
    for (url, name) in [
        ("https://example.com/page.html", "page.html"),
        ("https://example.com/path/to/doc", "doc"),
        ("https://example.com/", "example_com"),
        // A generic page name with an identifying query keeps the identity,
        // so two items of one site do not share a file.
        ("https://youtube.com/watch?v=abc", "youtube_com_watch_abc"),
        (
            "https://news.ycombinator.com/item?id=8863",
            "news_ycombinator_com_item_8863",
        ),
        (
            "https://news.ycombinator.com/item?id=1",
            "news_ycombinator_com_item_1",
        ),
        (
            "https://example.com/index.php?utm_source=x&id=7&sort=asc",
            "example_com_index.php_7",
        ),
        ("https://example.com/?p=123", "example_com_123"),
        ("https://example.com/8863?id=8863", "example_com_8863"),
        (
            "https://example.com/watch?v=a/b%20c",
            "example_com_watch_a-b-c",
        ),
        (
            "https://example.com/doc?token=zzz&id=5",
            "example_com_doc_5",
        ),
        ("https://example.com/p?id=../../etc", "example_com_p_etc"),
        // Searches, views, tracking and secrets are not identities.
        (
            "https://example.com:8080/search?q=x",
            "example_com_8080_search",
        ),
        (
            "https://example.com/page?session=abc&utm_source=a",
            "example_com_page",
        ),
        ("https://example.com/doc?id=", "example_com_doc"),
        ("https://example.com/doc?token=zzz", "example_com_doc"),
        // X and Twitter posts: user and post id, not the photo or video number.
        (
            "https://x.com/elonmusk/status/1234567890123456789/photo/1",
            "elonmusk-status-1234567890123456789",
        ),
        (
            "https://twitter.com/NASA/status/20?s=20&t=abc",
            "NASA-status-20",
        ),
        (
            "https://mobile.twitter.com/a_b/status/9/video/2",
            "a_b-status-9",
        ),
        ("https://www.x.com/user/status/5/analytics", "user-status-5"),
        ("https://fxtwitter.com/user/status/5", "user-status-5"),
        ("https://x.com/i/web/status/77", "i-status-77"),
        ("https://x.com/user", "user"),
        ("https://x.com/user/status/abc", "abc"),
        ("https://x.com/user/status/", "status"),
        ("https://example.com/user/status/12", "12"),
        // Non-ASCII paths, written plainly or percent-encoded, name their
        // output readably; an encoded slash or control character stays safe,
        // and bytes that are not UTF-8 keep their encoding.
        ("https://ynewtime.com/posts/人是什么单位", "人是什么单位"),
        (
            "https://ynewtime.com/posts/%E4%BA%BA%E6%98%AF%E4%BB%80%E4%B9%88%E5%8D%95%E4%BD%8D/",
            "人是什么单位",
        ),
        ("https://example.com/caf%C3%A9?lang=fr", "example_com_café"),
        ("https://example.com/a%2Fb", "a_b"),
        ("https://example.com/line%0Abreak", "line_break"),
        ("https://example.com/%FF%FE", "%FF%FE"),
        ("https://example.com/%2E%2E", "example_com"),
    ] {
        assert_eq!(
            markitai_core::output::url_name(url, &Default::default()),
            name
        );
    }
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.txt");
    std::fs::write(&input, "Pure body").unwrap();
    let output = dir.path().join("out");
    let result = convert(
        input.to_str().unwrap(),
        ConvertOptions {
            output_dir: Some(output.clone()),
            config: Some(json!({"output":{"filename":"chosen.md"},"llm":{"pure":true}})),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(result.output_path, Some(output.join("chosen.md")));
    assert_eq!(
        std::fs::read_to_string(output.join("chosen.md")).unwrap(),
        "Pure body"
    );
}

#[test]
fn url_names_from_queries_and_posts_stay_bounded_and_filesystem_safe() {
    let long = "a".repeat(500);
    for url in [
        format!("https://example.com/p?id={long}"),
        format!("https://example.com/{long}?id={long}"),
        format!("https://x.com/{long}/status/1"),
        "https://example.com/watch?v=%00%2F%5C:*%3F%22%3C%3E%7C".to_string(),
        "https://example.com/p?id=%E4%BA%BA%2F%E5%8D%95".to_string(),
    ] {
        let name = markitai_core::output::url_name(&url, &Default::default());
        assert!(!name.is_empty() && name.chars().count() <= 200, "{name}");
        assert!(
            !name.contains(['/', '\\', ':', '*', '?', '"', '<', '>', '|', '\0']),
            "{name}"
        );
        assert!(!name.starts_with('.') && !name.ends_with('.'), "{name}");
    }
    assert_eq!(
        markitai_core::output::url_name(
            &format!("https://example.com/p?id={long}"),
            &Default::default()
        ),
        format!("example_com_p_{}", "a".repeat(64))
    );
    assert_eq!(
        markitai_core::output::url_name(
            "https://example.com/p?id=%E4%BA%BA%2F%E5%8D%95",
            &Default::default()
        ),
        "example_com_p_人-单"
    );
}

#[test]
fn cached_url_reuses_extracted_content_without_changing_binding_json() {
    let root = tempfile::tempdir().unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/article", listener.local_addr().unwrap());
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(std::time::Duration::from_secs(5)))
            .unwrap();
        let mut request_reader = bounded_fixture_io::Reader::new(
            &stream,
            std::time::Instant::now() + std::time::Duration::from_secs(5),
        );
        let mut request = [0; 8192];
        let _ = request_reader.read(&mut request).unwrap();
        let body = "<html><title>Page title</title><article><p>Shared page content. 世界</p></article></html>";
        write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
    });
    let cfg = json!({"llm":{"enabled":false},"cache":{"global_dir":root.path().join("cache")},"fetch":{"strategy":"static"}});
    let first = convert(
        &url,
        ConvertOptions {
            config: Some(cfg.clone()),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(!first.fetch_cache_hit());
    server.join().unwrap();
    // The listener is gone, so any accidental second network request fails.
    let hit = convert(
        &url,
        ConvertOptions {
            config: Some(cfg.clone()),
            output_dir: Some(root.path().join("out")),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(hit.fetch_cache_hit());
    assert!(!hit.llm_cache_hit());
    assert_eq!(first.markdown, hit.markdown);
    assert_eq!(hit.frontmatter["title"], "Page title");
    assert!(hit.output_path.unwrap().is_file());
    let mut pure_cfg = cfg.clone();
    pure_cfg["llm"]["pure"] = json!(true);
    let pure = convert(
        &url,
        ConvertOptions {
            config: Some(pure_cfg),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(pure.fetch_cache_hit());
    assert!(pure.frontmatter.get("fetch_strategy").is_none());
    assert_eq!(pure.fetch_strategy(), Some("static"));
    assert!(
        serde_json::to_value(&pure)
            .unwrap()
            .get("fetch_strategy")
            .is_none()
    );
    let response: Value = serde_json::from_str(&convert_json(
        &json!({"source":url,"options":{"config":cfg}}).to_string(),
    ))
    .unwrap();
    assert_eq!(response["ok"], true);
    assert_eq!(response["result"]["markdown"], first.markdown);
    for internal in [
        "cache_hit",
        "llm_cache_hit",
        "fetch_cache_hit",
        "fetch_strategy",
        "explicit_fetch_strategy",
    ] {
        assert!(response["result"].get(internal).is_none());
    }
    let explicit = markitai_core::convert_with_context(
        &url,
        ConvertOptions {
            config: Some(cfg),
            ..Default::default()
        },
        markitai_core::ConvertContext {
            explicit_fetch_strategy: Some("static"),
            ..Default::default()
        },
    );
    assert!(
        matches!(explicit, Err(Error::Fetch(_))),
        "an explicit strategy must not share the configured/default scope"
    );
}

/// True, after saying why, when `result` is local OCR's explicit failure in a
/// process translated by Rosetta (an x86_64 build on Apple silicon), where
/// Vision text recognition fails on every system measured. The caller then
/// skips its assertions about recognized text. The translation is checked here
/// independently of the library, so a native process never skips, and any
/// other failure is left for the caller to report.
#[cfg(target_os = "macos")]
fn vision_unavailable_under_rosetta<T>(result: &markitai_core::Result<T>) -> bool {
    let mut translated: libc::c_int = 0;
    let mut size = std::mem::size_of::<libc::c_int>();
    // SAFETY: A read-only integer sysctl into a correctly sized buffer.
    let status = unsafe {
        libc::sysctlbyname(
            c"sysctl.proc_translated".as_ptr(),
            (&raw mut translated).cast(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    match result {
        Err(error)
            if status == 0
                && translated == 1
                && error
                    .to_string()
                    .contains("Vision text recognition failed under Rosetta translation") =>
        {
            eprintln!("skipping recognized-text assertions: {error}");
            true
        }
        _ => false,
    }
}

#[cfg(target_os = "macos")]
#[test]
fn local_image_ocr_reads_original_pixels_and_blank_images_are_explicit() {
    let dir = tempfile::tempdir().unwrap();
    let image = dir.path().join("recognition.png");
    std::fs::write(&image, include_bytes!("../src/ocr/fixtures/english.png")).unwrap();
    let cfg = json!({"cache":{"enabled":false},"ocr":{"enabled":true,"lang":"en"},
        "image":{"compress":true,"max_width":10,"max_height":10},"llm":{"enabled":false}});
    let recognized = convert(
        image.to_str().unwrap(),
        ConvertOptions {
            config: Some(cfg.clone()),
            output_dir: Some(dir.path().join("out")),
            ..Default::default()
        },
    );
    if vision_unavailable_under_rosetta(&recognized) {
        // The failed conversion publishes no Markdown.
        let published = std::fs::read_dir(dir.path().join("out"))
            .into_iter()
            .flatten()
            .any(|entry| entry.unwrap().path().extension() == Some("md".as_ref()));
        assert!(!published);
        return;
    }
    let recognized = recognized.unwrap();
    for line in include_str!("../src/ocr/fixtures/english.txt").lines() {
        assert!(
            recognized.markdown.contains(line),
            "missing {line}: {}",
            recognized.markdown
        );
    }
    assert!(recognized.llm_markdown.is_none());
    assert_eq!(recognized.assets.len(), 1);
    assert_eq!(recognized.usage.requests, 0);
    let preview = image::open(&recognized.assets[0]).unwrap();
    assert!(preview.width() <= 10 && preview.height() <= 10);
    let blank = dir.path().join("blank.png");
    image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
        100,
        100,
        image::Rgb([255, 255, 255]),
    ))
    .save(&blank)
    .unwrap();
    let result = convert(
        blank.to_str().unwrap(),
        ConvertOptions {
            config: Some(cfg),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(result.markdown.contains("![blank]"));
    assert!(
        result
            .warnings
            .iter()
            .any(|warning| warning.contains("no readable text"))
    );
    assert!(result.output_path.is_none() && result.assets.is_empty());
}

#[path = "conversion/request_coalescing.rs"]
mod request_coalescing;

#[path = "conversion/copilot.rs"]
mod copilot;

#[path = "conversion/chatgpt.rs"]
mod chatgpt;
#[path = "conversion/claude.rs"]
mod claude;

#[cfg(test)]
mod bounded_fixture_io {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/support/bounded_read.rs"
    ));
}

#[test]
fn office_ocr_without_a_renderer_keeps_the_documents_own_text() {
    // OCR over a folder must not fail its Office files where LibreOffice is
    // absent; explicitly requested screenshots still do.
    if markitai_core::office_render_available() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("report.docx");
    std::fs::write(
        &input,
        include_bytes!("../src/office_render/fixtures/blank-middle-three.docx"),
    )
    .unwrap();
    let base =
        json!({"cache":{"enabled":false},"history":{"record":false},"llm":{"enabled":false}});
    let mut ocr = base.clone();
    ocr["ocr"] = json!({"enabled": true});
    let result = convert(
        input.to_str().unwrap(),
        ConvertOptions {
            config: Some(ocr),
            output_dir: Some(dir.path().join("out")),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(
        result.markdown.contains("WORD FIRST PAGE"),
        "{}",
        result.markdown
    );
    assert!(
        result
            .warnings
            .iter()
            .any(|w| w.starts_with("OCR of Office pages needs LibreOffice")),
        "{:?}",
        result.warnings
    );
    assert!(result.screenshots.is_empty());
    assert!(
        result
            .warnings
            .iter()
            .any(|w| w.contains("brew install --cask libreoffice") && w.contains("--no-ocr")),
        "{:?}",
        result.warnings
    );

    // Screenshots degrade the same way: the text is published with one warning
    // that says how to get the pages or how to stop asking for them.
    let mut screenshots = base.clone();
    screenshots["screenshot"] = json!({"enabled": true});
    let result = convert(
        input.to_str().unwrap(),
        ConvertOptions {
            config: Some(screenshots.clone()),
            output_dir: Some(dir.path().join("out2")),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(
        result.markdown.contains("WORD FIRST PAGE"),
        "{}",
        result.markdown
    );
    assert!(result.screenshots.is_empty());
    let capture: Vec<_> = result
        .warnings
        .iter()
        .filter(|w| w.contains("LibreOffice"))
        .collect();
    assert_eq!(capture.len(), 1, "{:?}", result.warnings);
    assert!(
        capture[0].starts_with("Page screenshots of Office documents need LibreOffice")
            && capture[0].contains("brew install --cask libreoffice")
            && capture[0].contains("--no-screenshot"),
        "{capture:?}"
    );
    // Screenshots with OCR are still one warning, naming both.
    screenshots["ocr"] = json!({"enabled": true});
    let result = convert(
        input.to_str().unwrap(),
        ConvertOptions {
            config: Some(screenshots.clone()),
            output_dir: Some(dir.path().join("out3")),
            ..Default::default()
        },
    )
    .unwrap();
    let capture: Vec<_> = result
        .warnings
        .iter()
        .filter(|w| w.contains("LibreOffice"))
        .collect();
    assert_eq!(capture.len(), 1, "{:?}", result.warnings);
    assert!(
        capture[0].starts_with("Page screenshots and OCR of Office pages"),
        "{capture:?}"
    );

    // Only screenshots that are the whole output keep failing, and the error
    // names the way out.
    let mut only = screenshots;
    only["screenshot"] = json!({"enabled": true, "screenshot_only": true});
    only["ocr"] = json!({"enabled": false});
    let error = convert(
        input.to_str().unwrap(),
        ConvertOptions {
            config: Some(only),
            output_dir: Some(dir.path().join("out4")),
            ..Default::default()
        },
    )
    .unwrap_err()
    .to_string();
    assert!(
        error.contains("LibreOffice") && error.contains("brew install --cask libreoffice"),
        "{error}"
    );
    assert!(!dir.path().join("out4").join("report.docx.md").exists());
}

#[test]
fn numbers_capture_keeps_its_own_error_without_a_renderer() {
    // No LibreOffice installation would make Numbers capture work, so the
    // request still fails and says so instead of degrading to a warning.
    if markitai_core::office_render_available() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let package = dir.path().join("sheet.numbers");
    std::fs::write(
        &package,
        include_bytes!("../src/formats/numbers/fixtures/test-1.numbers"),
    )
    .unwrap();
    let error = convert(
        package.to_str().unwrap(),
        ConvertOptions {
            config: Some(json!({
                "cache":{"enabled":false},"history":{"record":false},
                "llm":{"enabled":false},"screenshot":{"enabled":true}
            })),
            output_dir: Some(dir.path().join("out")),
            ..Default::default()
        },
    )
    .unwrap_err()
    .to_string();
    // Without the native page renderer (Linux, Windows) capture fails earlier,
    // naming the renderer; either way it fails without the LibreOffice hint.
    assert!(
        (error.contains("Numbers") || error.contains("native PDF page renderer"))
            && !error.contains("brew install"),
        "{error}"
    );
}
