use super::*;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

fn isolated(name: &str, optout: bool) -> bool {
    let exact = format!("heif_image::{name}");
    if std::env::var("MARKITAI_HEIF_TEST").as_deref() == Ok(&exact) {
        return false;
    }
    let directory = tempfile::tempdir().unwrap();
    let stdout = directory.path().join("stdout");
    let stderr = directory.path().join("stderr");
    let mut command = Command::new(std::env::current_exe().unwrap());
    command
        .args(["--exact", &exact, "--nocapture"])
        .env_clear()
        .env("MARKITAI_HEIF_TEST", &exact)
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
            panic!("HEIF test timed out: {exact}");
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

fn cfg(directory: &Path) -> Value {
    json!({"cache":{"enabled":false},"history":{"record":false},"prompts":{"dir":directory.join("prompts")},"log":{"dir":null},
        "ocr":{"enabled":false},"image":{"compress":false,"alt_enabled":false,"desc_enabled":false},"llm":{"enabled":false}})
}
fn model(config: &mut Value, base: &str) {
    config["llm"] = json!({"enabled":true,"keep_base":true,"failure_policy":"fail","router_settings":{"num_retries":0,"timeout":5},
        "model_list":[{"model_name":"heif-local","litellm_params":{"model":"openai/mock","api_base":base,"api_key":"fixture-only"},"model_info":{"supports_vision":true}}]});
}

/// HEIC everywhere (ImageIO on macOS, the built-in decoder elsewhere); AVIF
/// where ImageIO decodes it.
#[test]
fn actual_heic_and_avif_reach_vision_as_png_with_upright_pixels() {
    use base64::Engine;
    if isolated(
        "actual_heic_and_avif_reach_vision_as_png_with_upright_pixels",
        false,
    ) {
        return;
    }
    let directory = tempfile::tempdir().unwrap();
    #[cfg_attr(not(target_os = "macos"), allow(unused_mut))]
    let mut cases = vec![(
        "rotated.svg",
        include_bytes!("../../src/images/fixtures/heif/quadrants-orientation6.heic").as_slice(),
        (80, 120),
    )];
    #[cfg(target_os = "macos")]
    cases.push((
        "tiny.png",
        include_bytes!("../../src/images/fixtures/heif/white_1x1.avif").as_slice(),
        (1, 1),
    ));
    for (name, bytes, dimensions) in cases {
        let path = directory.path().join(name);
        std::fs::write(&path, bytes).unwrap();
        let (base, server) = llm_server(
            200,
            r#"{"choices":[{"message":{"content":"Decoded native image."}}]}"#,
        );
        let mut config = cfg(directory.path());
        model(&mut config, &base);
        let result = convert(
            path.to_str().unwrap(),
            ConvertOptions {
                config: Some(config),
                output_dir: Some(directory.path().join(name).with_extension("out")),
                ..Default::default()
            },
        )
        .unwrap();
        let request = server.join().unwrap();
        let images: Vec<_> = request["messages"][1]["content"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|v| v["image_url"]["url"].as_str())
            .collect();
        assert_eq!(images.len(), 1);
        let png = base64::engine::general_purpose::STANDARD
            .decode(images[0].strip_prefix("data:image/png;base64,").unwrap())
            .unwrap();
        let decoded = image::load_from_memory(&png).unwrap().to_rgba8();
        assert_eq!(decoded.dimensions(), dimensions);
        assert_eq!(result.assets.len(), 1);
        assert_eq!(std::fs::read(&result.assets[0]).unwrap(), png);
        assert_eq!(std::fs::read(path).unwrap(), bytes);
        assert_eq!(result.usage.requests, 1);
        assert!(result.output_path.unwrap().exists());
    }
}

#[cfg(target_os = "macos")]
#[test]
fn heic_local_ocr_and_optout_use_original_resolution_without_image_upload() {
    if isolated(
        "heic_local_ocr_and_optout_use_original_resolution_without_image_upload",
        true,
    ) {
        return;
    }
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("letters.heic");
    std::fs::write(
        &path,
        include_bytes!("../../src/images/fixtures/heif/english.heic"),
    )
    .unwrap();
    let mut config = cfg(directory.path());
    config["ocr"]["enabled"] = true.into();
    let local = convert(
        path.to_str().unwrap(),
        ConvertOptions {
            config: Some(config.clone()),
            output_dir: Some(directory.path().join("out")),
            ..Default::default()
        },
    );
    // A recognition failure means ImageIO had already decoded the HEIF.
    if super::vision_unavailable_under_rosetta(&local) {
        return;
    }
    let local = local.unwrap();
    assert!(
        local.markdown.contains("MARKITAI OCR"),
        "{}",
        local.markdown
    );
    assert!(local.markdown.contains("LOCAL TEXT ONLY"));
    assert_eq!(local.usage.requests, 0);
    let png = image::open(&local.assets[0]).unwrap();
    assert_eq!((png.width(), png.height()), (819, 301));
    let (base, server) = llm_server(
        200,
        r#"{"choices":[{"message":{"content":"{protected_input}"}}]}"#,
    );
    model(&mut config, &base);
    let enhanced = convert(
        path.to_str().unwrap(),
        ConvertOptions {
            config: Some(config),
            ..Default::default()
        },
    )
    .unwrap();
    let request = server.join().unwrap();
    let text = request["messages"][1]["content"]
        .as_str()
        .expect("opt-out must send only text");
    assert!(text.contains("MARKITAI OCR") && text.contains("LOCAL TEXT ONLY"));
    assert_eq!(enhanced.usage.requests, 1);
}

#[test]
fn corrupt_heif_fails_before_a_model_request_or_output() {
    if isolated("corrupt_heif_fails_before_a_model_request_or_output", false) {
        return;
    }
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("bad.heic");
    let bytes = include_bytes!("../../src/images/fixtures/heif/quadrants-orientation1.heic");
    std::fs::write(&path, &bytes[..bytes.len() / 2]).unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let mut config = cfg(directory.path());
    model(
        &mut config,
        &format!("http://{}/v1", listener.local_addr().unwrap()),
    );
    let output = directory.path().join("out");
    assert!(
        convert(
            path.to_str().unwrap(),
            ConvertOptions {
                config: Some(config),
                output_dir: Some(output.clone()),
                ..Default::default()
            }
        )
        .is_err()
    );
    assert!(matches!(listener.accept(),Err(e) if e.kind()==std::io::ErrorKind::WouldBlock));
    assert!(!output.join(".markitai/assets").exists());
}
