//! Through the real core and Vision on macOS, where the portable engine is the default and
//! Vision reads when its models cannot be downloaded: a language the portable models read
//! and Vision does not is still a failed download, for the setting is good (desktop/DESIGN.md,
//! section 9). One test, for it sets this process's environment before anything else runs.
#![cfg(target_os = "macos")]
use serde_json::{Value, json};
use std::path::Path;

#[test]
fn a_language_only_vision_lacks_stays_a_failed_download() {
    let home = tempfile::tempdir().unwrap();
    let closed = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let proxy = format!("http://{}", closed.local_addr().unwrap());
    drop(closed);
    // SAFETY: The only test of this program, and nothing it starts has started yet.
    unsafe {
        std::env::set_var("HOME", home.path());
        std::env::set_var("MARKITAI_HOME", home.path().join("state"));
        std::env::remove_var("MARKITAI_OCR_BACKEND");
        for name in ["HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"] {
            std::env::set_var(name, &proxy);
        }
        std::env::remove_var("NO_PROXY");
        std::env::remove_var("no_proxy");
    }
    let image =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../markitai-core/src/ocr/fixtures/english.png");
    let request = json!({
        "source": image,
        "options": {"config": {"ocr": {"lang": "el"}}, "llm": false, "ocr": true},
    });
    let response: Value =
        serde_json::from_str(&markitai_app::convert_json(&request.to_string())).unwrap();

    assert_eq!(
        response["error"]["code"], "text_recognition_error",
        "{response}"
    );
    let message = response["error"]["message"].as_str().unwrap();
    assert!(message.starts_with("Local OCR download of "), "{message}");
}
