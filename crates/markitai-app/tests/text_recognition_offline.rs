//! Through the real core: local OCR that has no models and cannot download them is
//! `text_recognition_error` and not a kind of file that cannot be converted, and a language
//! it has no models for is the configuration's to fix, for a retry would not change it
//! (desktop/DESIGN.md, section 9). One test, for it sets this process's environment before
//! anything else runs.
use serde_json::{Value, json};
use std::path::Path;

fn convert(source: &Path, config: Value) -> Value {
    let request =
        json!({"source": source, "options": {"config": config, "llm": false, "ocr": true}});
    serde_json::from_str(&markitai_app::convert_json(&request.to_string())).unwrap()
}

#[test]
fn ocr_that_cannot_run_says_what_the_user_can_do_about_it() {
    let home = tempfile::tempdir().unwrap();
    let closed = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let proxy = format!("http://{}", closed.local_addr().unwrap());
    drop(closed);
    // SAFETY: The only test of this program, and nothing it starts has started yet.
    unsafe {
        for name in ["HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA"] {
            std::env::set_var(name, home.path());
        }
        std::env::set_var("MARKITAI_HOME", home.path().join("state"));
        std::env::set_var("MARKITAI_OCR_BACKEND", "paddle");
        for name in ["HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"] {
            std::env::set_var(name, &proxy);
        }
        std::env::remove_var("NO_PROXY");
        std::env::remove_var("no_proxy");
    }
    let core = Path::new(env!("CARGO_MANIFEST_DIR")).join("../markitai-core");
    let image = core.join("src/ocr/fixtures/english.png");
    let scan = core.join("src/pdf_raster/fixtures/mixed-native-scanned-blank.pdf");

    let response = convert(&image, json!({}));
    assert_eq!(
        response["error"]["code"], "text_recognition_error",
        "{response}"
    );
    let message = response["error"]["message"].as_str().unwrap();
    assert!(message.starts_with("Local OCR download of "), "{message}");

    for source in [&image, &scan] {
        let response = convert(source, json!({"ocr": {"lang": "tlh"}}));
        assert_eq!(response["error"]["code"], "config_error", "{response}");
        let message = response["error"]["message"].as_str().unwrap();
        assert!(message.starts_with("ocr.lang is not "), "{message}");
    }
}
