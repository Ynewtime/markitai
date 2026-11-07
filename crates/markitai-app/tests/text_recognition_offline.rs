//! Through the real core: local OCR that has no models and cannot download them, or has
//! none for the language set, is `text_recognition_error` and not a kind of file that
//! cannot be converted (desktop/DESIGN.md, section 9). One test, for it sets this
//! process's environment before anything else runs.
use serde_json::{Value, json};
use std::path::Path;

fn convert(source: &Path, config: Value) -> Value {
    let request =
        json!({"source": source, "options": {"config": config, "llm": false, "ocr": true}});
    serde_json::from_str(&markitai_app::convert_json(&request.to_string())).unwrap()
}

#[test]
fn ocr_that_cannot_get_a_model_is_not_a_file_that_cannot_be_converted() {
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
    let image =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../markitai-core/src/ocr/fixtures/english.png");

    let response = convert(&image, json!({}));
    assert_eq!(
        response["error"]["code"], "text_recognition_error",
        "{response}"
    );
    let message = response["error"]["message"].as_str().unwrap();
    assert!(message.starts_with("Local OCR download of "), "{message}");

    let response = convert(&image, json!({"ocr": {"lang": "tlh"}}));
    assert_eq!(
        response["error"]["code"], "text_recognition_error",
        "{response}"
    );
    let message = response["error"]["message"].as_str().unwrap();
    assert!(message.starts_with("ocr.lang is not "), "{message}");
}
