use super::*;
use base64::Engine;
#[cfg(target_os = "macos")]
use image::DynamicImage;
use image::{Rgb, RgbImage};
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
use tiff::{
    encoder::{TiffEncoder, colortype},
    tags::Tag,
};

fn isolated(name: &str, optout: bool) -> bool {
    let exact = format!("multipage_image::{name}");
    if std::env::var("MARKITAI_TIFF_TEST").as_deref() == Ok(exact.as_str()) {
        return false;
    }
    let directory = tempfile::tempdir().unwrap();
    let stdout = directory.path().join("stdout");
    let stderr = directory.path().join("stderr");
    let mut command = Command::new(std::env::current_exe().unwrap());
    command
        .args(["--exact", &exact, "--nocapture"])
        .env_clear()
        .env("MARKITAI_TIFF_TEST", &exact)
        .env("MARKITAI_HOME", directory.path().join("state"))
        .env("MARKITAI_NO_VLM_OCR", if optout { "true" } else { "false" })
        .env("PYTHON_DOTENV_DISABLED", "1")
        .current_dir(directory.path())
        .stdout(Stdio::from(std::fs::File::create(&stdout).unwrap()))
        .stderr(Stdio::from(std::fs::File::create(&stderr).unwrap()));
    for key in [
        "PATH",
        "HOME",
        "USERPROFILE",
        "SYSTEMROOT",
        "WINDIR",
        "LANG",
        "LC_ALL",
        "TZ",
    ] {
        if let Some(value) = std::env::var_os(key) {
            command.env(key, value);
        }
    }
    let mut child = command.spawn().unwrap();
    let deadline = Instant::now() + Duration::from_secs(120);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!("TIFF test timed out: {exact}");
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    let output = std::fs::read_to_string(stdout).unwrap();
    let errors = std::fs::read_to_string(stderr).unwrap();
    assert!(status.success(), "{exact}: {status}\n{output}\n{errors}");
    assert!(
        output.contains("1 passed"),
        "test selector missed {exact}: {output}"
    );
    true
}

fn tiff_file(directory: &Path, name: &str, pages: &[(RgbImage, u16)]) -> PathBuf {
    let mut output = Cursor::new(Vec::new());
    let mut encoder = TiffEncoder::new(&mut output).unwrap();
    for (pixels, orientation) in pages {
        let mut page = encoder
            .new_image::<colortype::RGB8>(pixels.width(), pixels.height())
            .unwrap();
        page.encoder()
            .write_tag(Tag::Orientation, *orientation)
            .unwrap();
        page.write_data(pixels.as_raw()).unwrap();
    }
    let path = directory.join(name);
    std::fs::write(&path, output.into_inner()).unwrap();
    path
}
fn colors() -> Vec<(RgbImage, u16)> {
    vec![
        (RgbImage::from_pixel(80, 60, Rgb([255, 0, 0])), 1),
        (RgbImage::from_pixel(90, 70, Rgb([0, 255, 0])), 6),
        (RgbImage::from_pixel(100, 80, Rgb([0, 0, 255])), 3),
    ]
}
fn cfg(directory: &Path) -> Value {
    // Page-order assertions read the optional page comments.
    json!({"output":{"page_markers":true},"cache":{"enabled":false},"history":{"record":false},"prompts":{"dir":directory.join("prompts")},"log":{"dir":null},
        "ocr":{"enabled":false},"image":{"compress":false,"alt_enabled":false,"desc_enabled":false},"llm":{"enabled":false}})
}
fn model(config: &mut Value, base: &str) {
    config["llm"] = json!({"enabled":true,"keep_base":true,"failure_policy":"fail","router_settings":{"num_retries":0,"timeout":5},
        "model_list":[{"model_name":"tiff-local","litellm_params":{"model":"openai/mock","api_base":base,"api_key":"fixture-only"},"model_info":{"supports_vision":true}}]});
}
fn run(
    path: &Path,
    config: Value,
    output_dir: Option<PathBuf>,
) -> markitai_core::Result<markitai_core::ConversionOutput> {
    convert(
        path.to_str().unwrap(),
        ConvertOptions {
            config: Some(config),
            output_dir,
            ..Default::default()
        },
    )
}

#[test]
fn all_tiff_pages_reach_vision_in_order_and_original_is_saved() {
    if isolated(
        "all_tiff_pages_reach_vision_in_order_and_original_is_saved",
        false,
    ) {
        return;
    }
    let directory = tempfile::tempdir().unwrap();
    // Content wins over the misleading filename.
    let path = tiff_file(directory.path(), "scan.png", &colors());
    let original = std::fs::read(&path).unwrap();
    let (base, server) = llm_server(
        200,
        r#"{"choices":[{"message":{"content":"All three pages read."}}]}"#,
    );
    let mut config = cfg(directory.path());
    model(&mut config, &base);
    config["ocr"]["enabled"] = true.into();
    let result = run(&path, config, Some(directory.path().join("out"))).unwrap();
    let request = server.join().unwrap();
    let blocks = request["messages"][1]["content"].as_array().unwrap();
    let images: Vec<_> = blocks
        .iter()
        .filter_map(|block| block["image_url"]["url"].as_str())
        .collect();
    assert_eq!(images.len(), 3);
    let expected = [
        ((80, 60), [255, 0, 0]),
        ((70, 90), [0, 255, 0]),
        ((100, 80), [0, 0, 255]),
    ];
    for (url, (dimensions, color)) in images.iter().zip(expected) {
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(url.strip_prefix("data:image/png;base64,").unwrap())
            .unwrap();
        let image = image::load_from_memory(&bytes).unwrap().to_rgb8();
        assert_eq!(image.dimensions(), dimensions);
        assert_eq!(image.get_pixel(0, 0).0, color);
        assert!(
            result
                .assets
                .iter()
                .any(|path| std::fs::read(path).unwrap() == bytes)
        );
    }
    assert_eq!(result.usage.requests, 1);
    assert_eq!(result.assets.len(), 4);
    assert!(
        result
            .assets
            .iter()
            .any(|path| std::fs::read(path).unwrap() == original)
    );
    assert_eq!(std::fs::read(&path).unwrap(), original);
    for page in 1..=3 {
        assert!(
            result
                .markdown
                .contains(&format!("<!-- Page number: {page} -->"))
        );
    }
    assert!(result.markdown.contains("[Original TIFF]"));
    assert!(result.llm_markdown.unwrap().contains("All three pages"));
}

#[test]
fn page_limit_fails_before_any_model_request_or_output_publication() {
    if isolated(
        "page_limit_fails_before_any_model_request_or_output_publication",
        false,
    ) {
        return;
    }
    let directory = tempfile::tempdir().unwrap();
    let path = tiff_file(directory.path(), "scan.tif", &colors());
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let mut config = cfg(directory.path());
    model(
        &mut config,
        &format!("http://{}/v1", listener.local_addr().unwrap()),
    );
    config["llm"]["max_vision_pages_per_document"] = 2.into();
    let out = directory.path().join("out");
    let error = run(&path, config, Some(out.clone())).unwrap_err();
    assert!(error.to_string().contains("max_vision_pages_per_document"));
    assert!(
        matches!(listener.accept(), Err(error) if error.kind() == std::io::ErrorKind::WouldBlock)
    );
    assert!(!out.join("scan.md").exists());
    assert!(!out.join(".markitai/assets").exists());
}

#[cfg(target_os = "macos")]
fn text_pages() -> Vec<(RgbImage, u16)> {
    let image = image::load_from_memory(include_bytes!("../../src/ocr/fixtures/english.png"))
        .unwrap()
        .to_rgb8();
    let rotated = DynamicImage::ImageRgb8(image.clone()).rotate270().to_rgb8();
    vec![
        (image, 1),
        (rotated, 6),
        (RgbImage::from_pixel(300, 200, Rgb([255, 255, 255])), 1),
    ]
}

#[cfg(target_os = "macos")]
#[test]
fn local_ocr_uses_every_upright_page_and_retains_blank_pages() {
    if isolated(
        "local_ocr_uses_every_upright_page_and_retains_blank_pages",
        false,
    ) {
        return;
    }
    let directory = tempfile::tempdir().unwrap();
    let path = tiff_file(directory.path(), "letters.tiff", &text_pages());
    let mut config = cfg(directory.path());
    config["ocr"]["enabled"] = true.into();
    // A model-specific budget must not truncate local OCR pages.
    config["llm"]["max_vision_pages_per_document"] = 1.into();
    let result = run(&path, config, Some(directory.path().join("out")));
    if super::vision_unavailable_under_rosetta(&result) {
        return;
    }
    let result = result.unwrap();
    let pages: Vec<_> = result.markdown.split("<!-- Page number:").skip(1).collect();
    assert_eq!(pages.len(), 3);
    for page in &pages[..2] {
        assert!(page.contains("MARKITAI OCR"), "{page}");
        assert!(page.contains("LOCAL TEXT ONLY"));
    }
    assert!(!pages[2].contains("MARKITAI OCR"));
    assert!(pages[2].contains("![Page 3]"));
    assert!(
        result
            .warnings
            .iter()
            .any(|warning| warning.contains("TIFF page 3"))
    );
    // The first two pages become identical upright PNGs and share one stored
    // asset; the original TIFF and blank preview are the other two files.
    assert_eq!(result.assets.len(), 3);
    let destinations: Vec<_> = pages
        .iter()
        .map(|page| page.split_once("](").unwrap().1.split_once(')').unwrap().0)
        .collect();
    assert_eq!(destinations[0], destinations[1]);
    assert_ne!(destinations[1], destinations[2]);
    for destination in destinations {
        let bytes = std::fs::read(directory.path().join("out").join(destination)).unwrap();
        assert!(image::load_from_memory(&bytes).is_ok());
    }
    let original = std::fs::read(&path).unwrap();
    assert!(
        result
            .assets
            .iter()
            .any(|asset| std::fs::read(asset).unwrap() == original)
    );
    assert_eq!(result.usage.requests, 0);
}

#[cfg(target_os = "macos")]
#[test]
fn vlm_optout_sends_all_recognized_text_without_image_blocks() {
    if isolated(
        "vlm_optout_sends_all_recognized_text_without_image_blocks",
        true,
    ) {
        return;
    }
    let directory = tempfile::tempdir().unwrap();
    let path = tiff_file(directory.path(), "letters.tif", &text_pages()[..2]);
    let (base, server) = llm_server(
        200,
        r#"{"choices":[{"message":{"content":"{protected_input}"}}]}"#,
    );
    let mut config = cfg(directory.path());
    model(&mut config, &base);
    config["ocr"]["enabled"] = true.into();
    config["llm"]["max_vision_pages_per_document"] = 1.into();
    let result = run(&path, config, None);
    if super::vision_unavailable_under_rosetta(&result) {
        // Local recognition fails before any model request; the fixture
        // server's thread ends with this test process.
        return;
    }
    let result = result.unwrap();
    let request = server.join().unwrap();
    let content = &request["messages"][1]["content"];
    assert!(content.is_string(), "{content}");
    let text = content.as_str().unwrap();
    assert_eq!(text.matches("MARKITAI OCR").count(), 2);
    assert!(text.contains("⟦MKTI:"));
    assert!(
        result
            .llm_markdown
            .as_deref()
            .unwrap()
            .contains("<!-- Page number: 2 -->")
    );
    assert_eq!(result.usage.requests, 1);
    assert!(!request.to_string().contains("data:image/"));
    assert!(
        result
            .warnings
            .iter()
            .any(|warning| warning.contains("VLM OCR is disabled"))
    );
}
