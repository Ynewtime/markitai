//! Through the real core: local OCR that cannot read is `text_recognition_error`, and a
//! file that cannot be converted stays `conversion_error`, which is what the apps'
//! headlines follow (desktop/DESIGN.md, section 9). One test, for it sets this process's
//! environment before anything else runs.
use serde_json::{Value, json};
use std::path::Path;

fn convert(source: &Path, ocr: bool) -> Value {
    let request = json!({"source": source, "options": {"config": {}, "llm": false, "ocr": ocr}});
    serde_json::from_str(&markitai_app::convert_json(&request.to_string())).unwrap()
}

/// Leave every OCR model in place and damaged: OCR says so and downloads nothing.
fn damage_ocr_models(state: &Path) {
    let manifest: Value = serde_json::from_str(include_str!(
        "../../markitai-core/src/ocr/paddle/models.json"
    ))
    .unwrap();
    for model in manifest["models"].as_array().unwrap() {
        let name = model["name"].as_str().unwrap();
        let digest = model["sha256"].as_str().unwrap();
        let directory = state
            .join("models")
            .join("ocr")
            .join(format!("{name}-{}", &digest[..8]));
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join(model["file"].as_str().unwrap()), "damaged").unwrap();
    }
}

#[test]
fn ocr_that_cannot_read_is_told_from_a_file_that_cannot_be_converted() {
    let home = tempfile::tempdir().unwrap();
    let state = home.path().join("state");
    // SAFETY: The only test of this program, and nothing it starts has started yet.
    unsafe {
        for name in ["HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA"] {
            std::env::set_var(name, home.path());
        }
        std::env::set_var("MARKITAI_HOME", &state);
        std::env::set_var("MARKITAI_OCR_BACKEND", "paddle");
    }
    damage_ocr_models(&state);
    let fixture = |relative: &str| Path::new(env!("CARGO_MANIFEST_DIR")).join(relative);

    // An image read, and a scanned page, which the PDF path words in its own way.
    let image = fixture("../markitai-core/src/ocr/fixtures/english.png");
    let response = convert(&image, true);
    assert_eq!(
        response["error"]["code"], "text_recognition_error",
        "{response}"
    );
    let message = response["error"]["message"].as_str().unwrap();
    assert!(message.starts_with("Local OCR models: "), "{message}");
    let scan = fixture("../markitai-core/src/pdf_raster/fixtures/mixed-native-scanned-blank.pdf");
    let response = convert(&scan, true);
    assert_eq!(
        response["error"]["code"], "text_recognition_error",
        "{response}"
    );
    let message = response["error"]["message"].as_str().unwrap();
    assert!(message.contains("local OCR failed on page"), "{message}");

    // Files that cannot be read at all, with OCR on as well: OCR did not fail at reading.
    let cut_off = &std::fs::read(&image).unwrap()[..64];
    for (name, content) in [
        ("broken.docx", &b"not a zip"[..]),
        ("not-an-image.png", &b"not an image"[..]),
        ("cut-off.png", cut_off),
    ] {
        let source = home.path().join(name);
        std::fs::write(&source, content).unwrap();
        let response = convert(&source, true);
        assert_eq!(response["error"]["code"], "conversion_error", "{response}");
    }
}
