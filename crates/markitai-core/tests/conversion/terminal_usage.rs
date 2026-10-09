use super::vision_processing::{Server, cfg as base_cfg, reply, typed};
use super::*;
use markitai_core::{
    ConversionUsage, ConvertContext, LlmRuntime, convert_detailed, convert_with_context_detailed,
    convert_with_publication_detailed,
};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::{
    Arc, Barrier,
    atomic::{AtomicUsize, Ordering},
};
use std::thread;
use std::time::{Duration, Instant};

fn isolated(name: &str) -> bool {
    let exact = format!("terminal_usage::{name}");
    if std::env::var("MARKITAI_TERMINAL_USAGE_TEST").as_deref() == Ok(&exact) {
        return false;
    }
    let dir = tempfile::tempdir().unwrap();
    let state = dir.path().join("state");
    std::fs::create_dir(&state).unwrap();
    let out = dir.path().join("stdout");
    let err = dir.path().join("stderr");
    let mut cmd = Command::new(std::env::current_exe().unwrap());
    cmd.args(["--exact", &exact, "--nocapture"])
        .env_clear()
        .env("MARKITAI_TERMINAL_USAGE_TEST", &exact)
        .env("MARKITAI_HOME", state)
        .current_dir(dir.path())
        .stdout(Stdio::from(std::fs::File::create(&out).unwrap()))
        .stderr(Stdio::from(std::fs::File::create(&err).unwrap()));
    for name in [
        "HOME",
        "PATH",
        "USERPROFILE",
        "SYSTEMROOT",
        "WINDIR",
        "LANG",
        "LC_ALL",
        "TZ",
    ] {
        if let Some(value) = std::env::var_os(name) {
            cmd.env(name, value);
        }
    }
    let mut child = cmd.spawn().unwrap();
    let deadline = Instant::now() + Duration::from_secs(90);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if Instant::now() > deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!("isolated test timed out: {exact}");
        }
        thread::sleep(Duration::from_millis(10));
    };
    let stdout = std::fs::read_to_string(out).unwrap();
    let stderr = std::fs::read_to_string(err).unwrap();
    assert!(status.success(), "{exact}: {status}\n{stdout}\n{stderr}");
    assert!(
        stdout.contains("1 passed"),
        "test selector did not run: {stdout}"
    );
    true
}

fn cfg(server: &Server, root: &Path) -> Value {
    let mut value = base_cfg(server, root);
    value["llm"]["on_failure"] = json!("fail");
    value
}
fn source(root: &Path) -> std::path::PathBuf {
    let path = root.join("source.md");
    std::fs::write(&path, "# Usage audit\n\nA complete authored paragraph.\n").unwrap();
    path
}
fn options(config: Value) -> ConvertOptions {
    ConvertOptions {
        config: Some(config),
        ..Default::default()
    }
}
fn user(request: &Value) -> &str {
    request["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|message| message["role"] == "user")
        .unwrap()["content"]
        .as_str()
        .unwrap()
}
fn assert_usage(usage: &ConversionUsage, requests: u64, input: u64, output: u64) {
    assert_eq!(usage.requests, requests);
    assert_eq!(usage.input_tokens, input);
    assert_eq!(usage.output_tokens, output);
    assert_eq!(
        usage
            .by_model
            .values()
            .map(|model| model["requests"].as_u64().unwrap())
            .sum::<u64>(),
        requests
    );
    assert_eq!(
        usage
            .by_model
            .values()
            .map(|model| model["input_tokens"].as_u64().unwrap())
            .sum::<u64>(),
        input
    );
    assert_eq!(
        usage
            .by_model
            .values()
            .map(|model| model["output_tokens"].as_u64().unwrap())
            .sum::<u64>(),
        output
    );
}

#[test]
fn invalid_paid_responses_retain_all_attempts_and_old_error_category() {
    if isolated("invalid_paid_responses_retain_all_attempts_and_old_error_category") {
        return;
    }
    let root = tempfile::tempdir().unwrap();
    let path = source(root.path());
    let server = Server::new(|_, _| (200, reply("not the required document object")));
    let config = cfg(&server, root.path());
    let failure = convert_detailed(path.to_str().unwrap(), options(config.clone())).unwrap_err();
    assert!(matches!(failure.error, Error::Conversion(_)));
    assert_usage(&failure.usage, 3, 21, 15);
    let old = convert(path.to_str().unwrap(), options(config)).unwrap_err();
    assert!(matches!(old, Error::Conversion(_)));
    assert_eq!(old.code(), failure.code());
    assert_eq!(old.to_string(), failure.to_string());
    assert_eq!(server.count(), 6);
}

#[test]
fn auth_and_quota_failure_json_retains_paid_zero_and_nonzero_usage() {
    if isolated("auth_and_quota_failure_json_retains_paid_zero_and_nonzero_usage") {
        return;
    }
    let root = tempfile::tempdir().unwrap();
    let path = source(root.path());
    for (status, kind, input, output) in [
        (401, "invalid_api_key", 0, 0),
        (429, "insufficient_quota", 13, 2),
    ] {
        let server = Server::new(move |_, _| {
            (
                status,
                json!({"error":{"message":"Authored terminal failure","type":kind,"code":kind},"usage":{"prompt_tokens":input,"completion_tokens":output}}),
            )
        });
        let response: Value = serde_json::from_str(&convert_json(
            &json!({"source":path,"options":{"config":cfg(&server,root.path())}}).to_string(),
        ))
        .unwrap();
        assert_eq!(response["ok"], false);
        assert_eq!(response["error"]["code"], "conversion_error");
        assert!(response["error"]["message"].is_string());
        let usage = &response["error"]["usage"];
        assert_eq!(usage["requests"], 1);
        assert_eq!(usage["input_tokens"], input);
        assert_eq!(usage["output_tokens"], output);
        assert_eq!(server.count(), 1);
        assert!(!response.to_string().contains("local-fixture"));
    }
}

#[test]
fn early_input_and_configuration_failures_have_no_paid_usage() {
    if isolated("early_input_and_configuration_failures_have_no_paid_usage") {
        return;
    }
    let failure = convert_detailed("", options(json!({}))).unwrap_err();
    assert!(matches!(failure.error, Error::InvalidInput(_)));
    assert_usage(&failure.usage, 0, 0, 0);
    for request in [
        json!({"source":"","options":{"config":{}}}).to_string(),
        "invalid json".into(),
    ] {
        let response: Value = serde_json::from_str(&convert_json(&request)).unwrap();
        assert_eq!(response["ok"], false);
        assert!(response["error"].get("usage").is_none());
    }
    let root = tempfile::tempdir().unwrap();
    let path = source(root.path());
    let failure = convert_detailed(path.to_str().unwrap(), options(json!({"llm":{"enabled":true,"model_list":[]},"cache":{"enabled":false},"prompts":{"dir":root.path().join("prompts")}}))).unwrap_err();
    assert!(matches!(failure.error, Error::NoModelConfigured));
    assert_usage(&failure.usage, 0, 0, 0);
}

struct RefusePublication(AtomicUsize);
impl markitai_core::output::Publication for RefusePublication {
    fn skip_existing(&self) -> bool {
        false
    }
    fn publish(&self, _path: &Path, _bytes: &[u8]) -> markitai_core::Result<()> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Err(Error::Io(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "authored publication denied",
        )))
    }
}
#[test]
fn successful_paid_processing_followed_by_publication_failure_keeps_usage() {
    if isolated("successful_paid_processing_followed_by_publication_failure_keeps_usage") {
        return;
    }
    let root = tempfile::tempdir().unwrap();
    let path = source(root.path());
    let server =
        Server::new(|request, _| (200, typed(user(request), "A complete authored document")));
    let publication = RefusePublication(AtomicUsize::new(0));
    let mut options = options(cfg(&server, root.path()));
    options.output_dir = Some(root.path().join("output"));
    let failure = convert_with_publication_detailed(
        path.to_str().unwrap(),
        options,
        ConvertContext::default(),
        Some(&publication),
    )
    .unwrap_err();
    assert!(
        matches!(&failure.error, Error::Io(error) if error.kind() == std::io::ErrorKind::PermissionDenied)
    );
    assert_usage(&failure.usage, 1, 7, 5);
    assert_eq!(publication.0.load(Ordering::SeqCst), 1);
    assert_eq!(server.count(), 1);
    assert!(!root.path().join("output/source.md.md").exists());
}

#[test]
fn image_analysis_failure_keeps_main_document_and_image_usage_once() {
    if isolated("image_analysis_failure_keeps_main_document_and_image_usage_once") {
        return;
    }
    let root = tempfile::tempdir().unwrap();
    let mut pixels = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(image::RgbImage::from_fn(100, 100, |x, y| {
        image::Rgb([x as u8, y as u8, 71])
    }))
    .write_to(&mut pixels, image::ImageFormat::Png)
    .unwrap();
    let path = root.path().join("source.epub");
    let bytes = pixels.into_inner();
    let mut archive = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    for (name, body) in [
        ("META-INF/container.xml", b"<container><rootfiles><rootfile full-path='book/package.opf'/></rootfiles></container>".as_slice()),
        ("book/package.opf", b"<package><metadata><title>Embedded image</title></metadata><manifest><item id='chapter' href='chapter.xhtml' media-type='application/xhtml+xml'/><item id='figure' href='chart.png' media-type='image/png'/></manifest><spine><itemref idref='chapter'/></spine></package>".as_slice()),
        ("book/chapter.xhtml", b"<html xmlns='http://www.w3.org/1999/xhtml'><body><p>Embedded figure.</p><p><img src='chart.png' alt='Author caption'/></p></body></html>".as_slice()),
        ("book/chart.png", bytes.as_slice()),
    ] {
        archive.start_file(name, zip::write::SimpleFileOptions::default()).unwrap();
        archive.write_all(body).unwrap();
    }
    std::fs::write(&path, archive.finish().unwrap().into_inner()).unwrap();
    // Image analysis runs beside enhancement: answer by request kind.
    let server = Server::new(|request, _| {
        if request["messages"][1]["content"].is_string() {
            (200, typed(user(request), "A complete authored document"))
        } else {
            (
                401,
                json!({"error":{"message":"Image model denied","code":"invalid_api_key"},"usage":{"prompt_tokens":11,"completion_tokens":3}}),
            )
        }
    });
    let mut config = cfg(&server, root.path());
    config["image"]["alt_enabled"] = json!(true);
    config["image"]["desc_enabled"] = json!(true);
    let output = convert_detailed(
        path.to_str().unwrap(),
        ConvertOptions {
            output_dir: Some(root.path().join("output")),
            ..options(config.clone())
        },
    )
    .unwrap();
    assert_usage(&output.usage, 2, 18, 8);
    assert!(
        output
            .warnings
            .iter()
            .any(|warning| warning.contains("Image analysis failed"))
    );
    assert!(
        output
            .llm_markdown
            .as_ref()
            .unwrap()
            .contains("Author caption")
    );
    let publication = RefusePublication(AtomicUsize::new(0));
    let failure = convert_with_publication_detailed(
        path.to_str().unwrap(),
        ConvertOptions {
            output_dir: Some(root.path().join("denied")),
            ..options(config)
        },
        ConvertContext::default(),
        Some(&publication),
    )
    .unwrap_err();
    assert!(matches!(failure.error, Error::Io(_)));
    assert_usage(&failure.usage, 2, 18, 8);
    assert_eq!(server.count(), 4);
}

#[test]
fn concurrent_shared_runtime_keeps_each_failed_documents_usage_separate() {
    if isolated("concurrent_shared_runtime_keeps_each_failed_documents_usage_separate") {
        return;
    }
    let root = tempfile::tempdir().unwrap();
    let path = source(root.path());
    let barrier = Arc::new(Barrier::new(2));
    let waiting = barrier.clone();
    let server = Server::new(move |request, _| {
        waiting.wait();
        let tokens = if request["model"] == "fixture-a" {
            17
        } else {
            31
        };
        (
            401,
            json!({"error":{"message":"Separate paid failure","code":"invalid_api_key"},"usage":{"prompt_tokens":tokens,"completion_tokens":1}}),
        )
    });
    let runtime = LlmRuntime::new(2).unwrap();
    let results = thread::scope(|scope| {
        let handles: Vec<_> = [("fixture-a", 17), ("fixture-b", 31)]
            .into_iter()
            .map(|(name, tokens)| {
                let runtime = &runtime;
                let path = &path;
                let mut config = cfg(&server, root.path());
                config["llm"]["model_list"][0]["litellm_params"]["model"] =
                    json!(format!("openai/{name}"));
                scope.spawn(move || {
                    let result = convert_with_context_detailed(
                        path.to_str().unwrap(),
                        options(config),
                        ConvertContext {
                            llm_runtime: Some(runtime),
                            ..Default::default()
                        },
                    )
                    .unwrap_err();
                    assert_usage(&result.usage, 1, tokens, 1);
                    result
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>()
    });
    assert_eq!(
        results.iter().map(|r| r.usage.input_tokens).sum::<u64>(),
        48
    );
    assert_eq!(server.count(), 2);
}

#[test]
fn standalone_image_terminal_failure_has_its_recorded_usage() {
    if isolated("standalone_image_terminal_failure_has_its_recorded_usage") {
        return;
    }
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("image.png");
    image::DynamicImage::new_rgb8(100, 100).save(&path).unwrap();
    let server = Server::new(|_, _| {
        (
            401,
            json!({"error":{"message":"Image denied","code":"invalid_api_key"},"usage":{"prompt_tokens":11,"completion_tokens":3}}),
        )
    });
    let mut config = cfg(&server, root.path());
    config["image"]["alt_enabled"] = json!(true);
    config["image"]["desc_enabled"] = json!(true);
    let failure = convert_detailed(path.to_str().unwrap(), options(config)).unwrap_err();
    assert!(matches!(failure.error, Error::Conversion(_)));
    assert_usage(&failure.usage, 1, 11, 3);
    assert_eq!(server.count(), 1);
}
