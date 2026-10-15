//! Actual optional LibreOffice acceptance; never counts an absent renderer as success.
#![cfg(target_os = "macos")]
use super::*;
use std::path::{Path, PathBuf};
const PPTX: &[u8] = include_bytes!("../../src/office_render/fixtures/hidden-blank-three.pptx");
const DOCX: &[u8] = include_bytes!("../../src/office_render/fixtures/blank-middle-three.docx");

fn source(directory: &Path, extension: &str, bytes: &[u8]) -> PathBuf {
    let path = directory.join(format!("演示 full document.{extension}"));
    std::fs::write(&path, bytes).unwrap();
    path
}
fn cfg() -> Value {
    json!({"cache":{"enabled":false},"history":{"record":false},"llm":{"enabled":false},"ocr":{"enabled":false},"image":{"alt_enabled":false,"desc_enabled":false,"format":"png","max_width":720,"max_height":540},"screenshot":{"enabled":true}})
}

#[test]
#[ignore = "Requires installed LibreOffice and native macOS renderer; run explicitly"]
fn office_capture_preserves_native_body_and_exports_every_slide_and_word_page() {
    for (extension, bytes, body_tokens) in [
        ("pptx", PPTX, vec!["VISIBLE FIRST", "HIDDEN SECOND"]),
        ("docx", DOCX, vec!["WORD FIRST PAGE", "WORD THIRD PAGE"]),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let input = source(dir.path(), extension, bytes);
        let output = dir.path().join("output");
        let mut base_cfg = cfg();
        base_cfg["screenshot"]["enabled"] = json!(false);
        let native = convert(
            input.to_str().unwrap(),
            ConvertOptions {
                config: Some(base_cfg),
                ..Default::default()
            },
        )
        .unwrap();
        let captured = convert(
            input.to_str().unwrap(),
            ConvertOptions {
                config: Some(cfg()),
                output_dir: Some(output),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(
            captured.markdown.starts_with(&native.markdown),
            "native body was altered"
        );
        assert_eq!(captured.screenshots.len(), 3);
        for token in body_tokens {
            assert!(captured.markdown.contains(token));
        }
        for path in &captured.screenshots {
            let image = std::fs::read(path).unwrap();
            assert!(image.starts_with(b"\x89PNG\r\n\x1a\n"));
            let name = path.file_name().unwrap().to_string_lossy();
            let encoded: String = name
                .as_bytes()
                .iter()
                .map(|&b| {
                    if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
                        (b as char).to_string()
                    } else {
                        format!("%{b:02X}")
                    }
                })
                .collect();
            assert!(captured.markdown.contains(&encoded));
        }
        assert_eq!(std::fs::read(input).unwrap(), bytes);
    }
}

#[test]
#[ignore = "Requires installed LibreOffice and native macOS renderer; run explicitly"]
fn office_all_page_vision_limit_fails_before_any_model_request() {
    let dir = tempfile::tempdir().unwrap();
    let input = source(dir.path(), "pptx", PPTX);
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let mut config = cfg();
    config["llm"] = json!({"enabled":true,"max_vision_pages_per_document":2,"router_settings":{"num_retries":0,"timeout":2},"model_list":[{"model_name":"mock","litellm_params":{"model":"openai/mock","api_key":"test-only","api_base":format!("http://{}/v1",listener.local_addr().unwrap())},"model_info":{"supports_vision":true}}]});
    let error = convert(
        input.to_str().unwrap(),
        ConvertOptions {
            config: Some(config),
            output_dir: Some(dir.path().join("out")),
            ..Default::default()
        },
    )
    .expect_err("three slides exceed a two-page vision budget");
    assert!(error.to_string().contains("vision"), "{error}");
    assert!(matches!(listener.accept(),Err(e) if e.kind()==std::io::ErrorKind::WouldBlock));
}

#[test]
fn malformed_workbook_cannot_succeed_when_full_page_capture_is_requested() {
    let dir = tempfile::tempdir().unwrap();
    let input = source(dir.path(), "xlsx", b"malformed-workbook-fixture");
    let error = convert(
        input.to_str().unwrap(),
        ConvertOptions {
            config: Some(cfg()),
            output_dir: Some(dir.path().join("out")),
            ..Default::default()
        },
    )
    .err()
    .unwrap();
    assert!(!error.to_string().is_empty());
    assert!(!dir.path().join("out").exists());
}

#[test]
#[ignore = "Requires installed LibreOffice and native macOS renderer; run explicitly"]
fn office_pure_and_explicit_only_route_every_page_without_losing_canonical_title() {
    let dir = tempfile::tempdir().unwrap();
    let input = source(dir.path(), "pptx", PPTX);
    for (pure, only) in [(false, false), (true, false), (true, true)] {
        let (base, server) = llm_server(
            200,
            r##"{"choices":[{"message":{"content":"# Reviewed\n\nAll three slides."}}]}"##,
        );
        let mut config = cfg();
        config["llm"] = json!({"enabled":true,"pure":pure,"keep_base":true,"router_settings":{"num_retries":0,"timeout":5},"model_list":[{"model_name":"mock","litellm_params":{"model":"openai/mock","api_base":base,"api_key":"test-only"},"model_info":{"supports_vision":true}}]});
        config["screenshot"]["screenshot_only"] = json!(only);
        let output = convert(
            input.to_str().unwrap(),
            ConvertOptions {
                config: Some(config),
                output_dir: Some(dir.path().join(format!("out-{pure}-{only}"))),
                ..Default::default()
            },
        )
        .unwrap();
        let request = server.join().unwrap();
        let content = &request["messages"][1]["content"];
        if pure && !only {
            assert!(content.as_str().unwrap().contains("VISIBLE FIRST"));
            assert!(
                !output
                    .llm_markdown
                    .as_ref()
                    .unwrap()
                    .contains("<!-- ![Slide ")
            );
        } else {
            let blocks = content.as_array().unwrap();
            assert_eq!(
                blocks
                    .iter()
                    .filter(|block| block["type"] == "image_url")
                    .count(),
                3
            );
            let text = blocks
                .iter()
                .find_map(|block| block["text"].as_str())
                .unwrap_or("");
            assert_eq!(text.contains("VISIBLE FIRST"), !only);
            assert_eq!(
                output
                    .llm_markdown
                    .as_ref()
                    .unwrap()
                    .matches("<!-- ![Slide ")
                    .count(),
                3
            );
        }
        assert_eq!(output.screenshots.len(), 3);
        assert!(output.markdown.contains("VISIBLE FIRST"));
    }
}

#[test]
#[ignore = "Requires installed LibreOffice and native macOS OCR; run explicitly"]
fn office_local_ocr_adds_supplement_without_replacing_native_body() {
    let dir = tempfile::tempdir().unwrap();
    let input = source(dir.path(), "docx", DOCX);
    let mut config = cfg();
    config["screenshot"]["enabled"] = json!(false);
    let native = convert(
        input.to_str().unwrap(),
        ConvertOptions {
            config: Some(config.clone()),
            ..Default::default()
        },
    )
    .unwrap();
    config["ocr"] = json!({"enabled":true});
    let result = convert(
        input.to_str().unwrap(),
        ConvertOptions {
            config: Some(config),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(result.markdown.starts_with(&native.markdown));
    assert!(result.markdown.contains("## Rendered-page OCR supplement"));
    assert!(result.markdown.contains("### Page 1"));
    assert!(result.markdown.contains("### Page 3"));
    assert!(!result.markdown.contains("### Page 2"));
    assert!(result.screenshots.is_empty());
    assert!(
        result
            .warnings
            .iter()
            .any(|warning| warning.starts_with("Office page 2 (1 of ")
                && warning.ends_with("): OCR found no text; the OCR supplement leaves it out."))
    );
}
