use super::*;
use std::path::{Path, PathBuf};

const MIXED: &[u8] = include_bytes!("../../src/pdf_raster/fixtures/mixed-native-scanned-blank.pdf");

fn source(dir: &Path) -> PathBuf {
    let path = dir.join("mixed.pdf");
    std::fs::write(&path, MIXED).unwrap();
    path
}

fn cfg() -> Value {
    json!({"cache":{"enabled":false},"history":{"record":false},"llm":{"enabled":false},"image":{"alt_enabled":false,"desc_enabled":false}})
}

fn run(path: &Path, cfg: Value) -> markitai_core::Result<markitai_core::ConversionOutput> {
    convert(
        path.to_str().unwrap(),
        ConvertOptions {
            config: Some(cfg),
            ..Default::default()
        },
    )
}

fn model(cfg: &mut Value, base: &str) {
    cfg["llm"] = json!({"enabled":true,"keep_base":true,"router_settings":{"num_retries":0,"timeout":5},"model_list":[{"model_name":"mock","litellm_params":{"model":"openai/mock","api_base":base,"api_key":"test"},"model_info":{"supports_vision":true}}]});
}

#[test]
fn pdf_vision_is_told_to_read_scanned_pages_and_the_notice_stays_for_a_page_it_left_blank() {
    let dir = tempfile::tempdir().unwrap();
    let path = source(dir.path());
    // The answer has text only after page 3's own tokens: page 2 was skipped.
    let (base, server) = llm_server(
        200,
        r##"{"choices":[{"message":{"content":"# Verified pages\n\nAll pages reviewed."}}]}"##,
    );
    let mut config = cfg();
    model(&mut config, &base);
    // A preset with screenshots, without --ocr: the page images are sent.
    config["screenshot"] = json!({"enabled":true});
    let result = run(&path, config).unwrap();
    let request = server.join().unwrap();
    let system = request["messages"][0]["content"].as_str().unwrap();
    assert!(system.contains("Frames 2, 3 are scanned pages"), "{system}");
    // Page 2 has no text of its own in the answer, so it is still unread.
    assert!(
        result
            .warnings
            .iter()
            .any(|w| w.contains("native text was not recovered") && w.contains("--ocr")),
        "{:?}",
        result.warnings
    );
    let markdown = result.llm_markdown.as_deref().unwrap_or(&result.markdown);
    // Its scan stays visible; page 3's text is there.
    assert!(markdown.contains("![Image on page 2]"), "{markdown}");
    assert!(markdown.contains("All pages reviewed."), "{markdown}");
}

#[test]
fn pdf_scan_notice_stays_when_no_model_reads_the_pages() {
    let dir = tempfile::tempdir().unwrap();
    let path = source(dir.path());
    let mut config = cfg();
    config["screenshot"] = json!({"enabled":true});
    let result = run(&path, config).unwrap();
    assert!(
        result
            .warnings
            .iter()
            .any(|w| w.contains("native text was not recovered") && w.contains("--ocr")),
        "{:?}",
        result.warnings
    );
}

#[test]
fn text_only_pdf_is_its_text_without_pictures_or_page_images() {
    let dir = tempfile::tempdir().unwrap();
    let path = source(dir.path());
    let (base, server) = llm_server(
        200,
        r##"{"choices":[{"message":{"content":"# Verified pages\n\nAll pages reviewed.\n\n![Image on page 2](.markitai/assets/scan.jpg)"}}]}"##,
    );
    let mut config = cfg();
    model(&mut config, &base);
    config["screenshot"] = json!({"enabled":true});
    config["output"] = json!({"text_only":true});
    let result = run(&path, config).unwrap();
    server.join().unwrap();
    for markdown in [result.llm_markdown.as_deref().unwrap(), &result.markdown] {
        assert!(!markdown.contains("!["), "{markdown}");
        assert!(
            !markdown.contains("Page images for reference"),
            "{markdown}"
        );
        assert!(!markdown.starts_with("---"), "{markdown}");
    }
    assert!(result.llm_markdown.unwrap().contains("All pages reviewed."));
    // Without a model: the native text of the PDF, pictures left out.
    let mut config = cfg();
    config["output"] = json!({"text_only":true});
    let plain = run(&path, config).unwrap();
    assert!(!plain.markdown.contains("!["), "{}", plain.markdown);
    assert!(plain.images.is_empty());
}
