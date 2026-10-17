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
fn pdf_vision_reads_scanned_pages_and_drops_the_ocr_notice() {
    let dir = tempfile::tempdir().unwrap();
    let path = source(dir.path());
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
    assert!(
        !result
            .warnings
            .iter()
            .any(|w| w.contains("native text was not recovered") || w.contains("--ocr")),
        "{:?}",
        result.warnings
    );
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
