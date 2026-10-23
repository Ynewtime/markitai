use super::*;
use base64::Engine;
use std::path::{Path, PathBuf};

const MIXED: &[u8] = include_bytes!("../../src/pdf_raster/fixtures/mixed-native-scanned-blank.pdf");

fn source(dir: &Path) -> PathBuf {
    let path = dir.join("报告 (mixed).pdf");
    std::fs::write(&path, MIXED).unwrap();
    path
}

fn cfg() -> Value {
    // Page-order assertions read the optional page comments.
    json!({"output":{"page_markers":true},"cache":{"enabled":false},"history":{"record":false},"llm":{"enabled":false},"image":{"alt_enabled":false,"desc_enabled":false}})
}

fn run(
    path: &Path,
    cfg: Value,
    output_dir: Option<PathBuf>,
) -> markitai_core::Result<markitai_core::ConversionOutput> {
    convert(
        path.to_str().unwrap(),
        ConvertOptions {
            config: Some(cfg),
            output_dir,
            ..Default::default()
        },
    )
}

fn model(cfg: &mut Value, base: &str) {
    cfg["llm"] = json!({"enabled":true,"keep_base":true,"router_settings":{"num_retries":0,"timeout":5},"model_list":[{"model_name":"mock","litellm_params":{"model":"openai/mock","api_base":base,"api_key":"test"},"model_info":{"supports_vision":true}}]});
}

#[test]
fn pdf_local_ocr_preserves_native_page_and_records_blank_outcome() {
    let dir = tempfile::tempdir().unwrap();
    let path = source(dir.path());
    let base = run(&path, cfg(), None).unwrap();
    let mut config = cfg();
    config["ocr"] = json!({"enabled":true});
    let result = run(&path, config.clone(), None);
    if super::vision_unavailable_under_rosetta(&result) {
        return;
    }
    let result = result.unwrap();
    let native = base
        .markdown
        .split("<!-- Page number: 2 -->")
        .next()
        .unwrap();
    assert!(result.markdown.starts_with(native));
    assert_eq!(result.markdown.matches("<!-- Page number:").count(), 3);
    for token in ["MARKITAI", "OCR", "LOCAL", "TEXT", "ONLY"] {
        assert!(
            result.markdown.contains(token),
            "{token}: {}",
            result.markdown
        );
    }
    assert!(
        result
            .warnings
            .iter()
            .any(|w| w == "PDF page 3 (1 of 3): OCR found no text; it is blank in the output.")
    );
    assert!(result.screenshots.is_empty());
    config["ocr"]["per_page_routing"] = json!(false);
    let forced = run(&path, config, None).unwrap();
    assert!(forced.markdown.contains("NATIVE PAGE ONE"));
    assert_eq!(forced.markdown.matches("<!-- Page number:").count(), 3);
}

#[test]
fn pdf_scan_exact_transcript() {
    let dir = tempfile::tempdir().unwrap();
    let path = source(dir.path());
    let mut config = cfg();
    config["ocr"] = json!({"enabled":true});
    let result = run(&path, config, None).unwrap();
    let page = result
        .markdown
        .split("<!-- Page number: 2 -->")
        .nth(1)
        .unwrap()
        .split("<!-- Page number: 3 -->")
        .next()
        .unwrap();
    // A scanned page OCR read text on is its text alone; its scan is not
    // appended as a picture again.
    assert!(!page.contains("![Image on page 2]"), "{page}");
    let expected = include_str!("../../src/ocr/fixtures/english.txt");
    assert_eq!(
        page.split_whitespace().collect::<Vec<_>>(),
        expected.split_whitespace().collect::<Vec<_>>()
    );
}

#[test]
fn pdf_capture_publishes_final_names_and_reuses_identical_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let path = source(dir.path());
    let mut config = cfg();
    config["screenshot"] = json!({"enabled":true,"screenshot_only":true});
    config["image"]["format"] = json!("png");
    config["output"]["on_conflict"] = json!("overwrite");
    let output = dir.path().join("output");
    let captures = output.join(".markitai/screenshots");
    std::fs::create_dir_all(&captures).unwrap();
    let collision = captures.join("报告 (mixed).pdf.page0001.png");
    std::fs::write(&collision, b"older capture retained").unwrap();
    let result = run(&path, config.clone(), Some(output.clone())).unwrap();
    assert!(result.markdown.contains("NATIVE PAGE ONE"));
    assert_eq!(result.screenshots.len(), 3);
    assert!(result.screenshots[0].ends_with("报告 (mixed).pdf.page0001.v2.png"));
    assert!(
        result
            .markdown
            .contains("%E6%8A%A5%E5%91%8A%20%28mixed%29.pdf.page0001.v2.png")
    );
    for capture in &result.screenshots {
        assert_eq!(
            image::guess_format(&std::fs::read(capture).unwrap()).unwrap(),
            image::ImageFormat::Png
        );
    }
    assert_eq!(std::fs::read(collision).unwrap(), b"older capture retained");
    let second = run(&path, config.clone(), Some(output)).unwrap();
    assert_eq!(second.screenshots, result.screenshots);
    let memory = run(&path, config, None).unwrap();
    assert!(memory.screenshots.is_empty() && memory.output_path.is_none());
    assert_eq!(memory.markdown.matches("<!-- ![Page ").count(), 3);
}

#[test]
fn pdf_config_only_flag_does_not_enable_capture() {
    let dir = tempfile::tempdir().unwrap();
    let path = source(dir.path());
    let base = run(&path, cfg(), None).unwrap();
    let mut config = cfg();
    config["screenshot"] = json!({"screenshot_only":true,"enabled":false});
    let result = run(&path, config.clone(), None).unwrap();
    assert_eq!(result.markdown, base.markdown);
    assert!(result.screenshots.is_empty());
    let disk = run(&path, config, Some(dir.path().join("output"))).unwrap();
    assert!(disk.screenshots.is_empty());
}

#[test]
fn pdf_vlm_sends_every_page_with_real_mime_in_memory() {
    let dir = tempfile::tempdir().unwrap();
    let path = source(dir.path());
    // WebP below quality 100 is JPEG (no lossy WebP encoder); at 100 it is
    // lossless WebP. The MIME type follows the bytes either way.
    for (quality, mime, format) in [
        (75, "image/jpeg", image::ImageFormat::Jpeg),
        (100, "image/webp", image::ImageFormat::WebP),
    ] {
        let (base, server) = llm_server(
            200,
            r##"{"choices":[{"message":{"content":"# Verified pages\n\nAll pages reviewed."}}]}"##,
        );
        let mut config = cfg();
        model(&mut config, &base);
        config["ocr"] = json!({"enabled":true});
        config["image"]["format"] = json!("webp");
        config["image"]["quality"] = json!(quality);
        let result = run(&path, config, None).unwrap();
        let request = server.join().unwrap();
        let blocks = request["messages"][1]["content"].as_array().unwrap();
        let images = blocks
            .iter()
            .filter_map(|block| block["image_url"]["url"].as_str())
            .collect::<Vec<_>>();
        assert_eq!(images.len(), 3);
        for image in images {
            let encoded = image.strip_prefix(&format!("data:{mime};base64,")).unwrap();
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(encoded)
                .unwrap();
            assert_eq!(image::guess_format(&bytes).unwrap(), format);
        }
        assert!(result.screenshots.is_empty());
        let enhanced = result.llm_markdown.unwrap();
        assert!(enhanced.contains("All pages reviewed."));
        assert_eq!(enhanced.matches("<!-- Page number:").count(), 3);
        assert_eq!(enhanced.matches("<!-- ![Page ").count(), 3);
    }
}

#[test]
fn pdf_pure_is_text_only_except_explicit_file_screenshot_only() {
    let dir = tempfile::tempdir().unwrap();
    let path = source(dir.path());
    for only in [false, true] {
        let (base, server) = llm_server(
            200,
            r##"{"choices":[{"message":{"content":"# Reviewed\n\nBody."}}]}"##,
        );
        let mut config = cfg();
        model(&mut config, &base);
        config["llm"]["pure"] = json!(true);
        config["ocr"] = json!({"enabled":true});
        config["screenshot"] = json!({"screenshot_only":only});
        let result = run(&path, config, None).unwrap();
        assert_eq!(
            result
                .llm_markdown
                .unwrap()
                .contains("<!-- Page images for reference -->"),
            only
        );
        let request = server.join().unwrap();
        let content = &request["messages"][1]["content"];
        if only {
            let content = content.as_array().unwrap();
            assert_eq!(
                content
                    .iter()
                    .filter(|block| block["type"] == "image_url")
                    .count(),
                3
            );
            assert!(!content.iter().any(|block| {
                block["text"]
                    .as_str()
                    .is_some_and(|text| text.contains("NATIVE PAGE ONE"))
            }));
        } else {
            assert!(content.as_str().unwrap().contains("NATIVE PAGE ONE"));
        }
    }
}

#[test]
fn pdf_vision_page_cap_fails_before_model_or_publication() {
    let dir = tempfile::tempdir().unwrap();
    let path = source(dir.path());
    let mut config = cfg();
    model(&mut config, "http://127.0.0.1:1/v1");
    config["ocr"] = json!({"enabled":true});
    config["llm"]["max_vision_pages_per_document"] = json!(2);
    let output = dir.path().join("output");
    let error = run(&path, config, Some(output.clone())).unwrap_err();
    assert_eq!(error.code(), "invalid_input");
    assert!(error.to_string().contains("no pages were rendered or sent"));
    assert!(!output.exists());
}

/// The parity corpus's scanned page of Chinese, Japanese, Korean and a table
/// (`corpus/ocr/cjk-scanned.pdf`), read by the default engine (the portable
/// one; Vision does not run in a macOS virtual machine, so the runners skip
/// this test by name); it needs `MARKITAI_TEST_CJK_SCANNED_PDF` to name the
/// file and passes without it. `ocr_corpus.rs` holds the same file's line, Korean
/// and table checks for every platform.
#[cfg(target_os = "macos")]
#[test]
fn vision_reads_the_corpus_scanned_cjk_page() {
    let Some(path) = std::env::var_os("MARKITAI_TEST_CJK_SCANNED_PDF") else {
        return;
    };
    let mut config = cfg();
    config["ocr"] = json!({"enabled":true});
    let result = run(Path::new(&path), config, None);
    if super::vision_unavailable_under_rosetta(&result) {
        return;
    }
    let result = result.unwrap();
    let text: String = result.markdown.split_whitespace().collect();
    for phrase in [
        "第一章概述",
        "本文档用于测试中文文本提取",
        "日本語のテキストも含まれています",
        "日本の首都です",
        "项目数量单价",
        "苹果",
        "所有字符都应该被正确保留",
    ] {
        assert!(text.contains(phrase), "{phrase}: {}", result.markdown);
    }
    assert!(
        !result.warnings.iter().any(|w| w.contains("could not read")),
        "{:?}",
        result.warnings
    );
}
