#![cfg(unix)]
use super::*;
use std::os::unix::fs::PermissionsExt;

struct Fixture {
    root: tempfile::TempDir,
    config: Config,
}
impl Fixture {
    fn new(name: &str) -> Self {
        let root = tempfile::tempdir().unwrap();
        let executable = root.path().join("codex");
        std::fs::write(&executable, include_bytes!("fake_exec.py")).unwrap();
        std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700)).unwrap();
        std::fs::write(
            root.path().join("scenario.json"),
            json!({"name":name}).to_string(),
        )
        .unwrap();
        let config = Config::from_env(&HashMap::from([
            (
                "CODEX_CLI_PATH".into(),
                executable.to_string_lossy().into_owned(),
            ),
            ("PATH".into(), std::env::var("PATH").unwrap()),
            ("OPENAI_API_KEY".into(), "must-not-forward".into()),
            ("OPENAI_BASE_URL".into(), "http://forbidden.invalid".into()),
            ("HTTP_PROXY".into(), "http://forbidden.invalid".into()),
        ]))
        .unwrap();
        Self { root, config }
    }
    fn request<'a>(&self, model: &'a str, images: &'a [(&'a str, &'a [u8])]) -> Request<'a> {
        Request {
            model,
            system: "Document instructions only.",
            user: "Document body\nwith Unicode 中文.",
            images,
            timeout: Duration::from_secs(10),
            cancel: None,
        }
    }
    fn calls(&self) -> Vec<Value> {
        std::fs::read_to_string(self.root.path().join("calls.jsonl"))
            .unwrap_or_default()
            .lines()
            .map(|s| serde_json::from_str(s).unwrap())
            .collect()
    }
}
fn png(color: [u8; 3]) -> Vec<u8> {
    let image = image::RgbImage::from_pixel(2, 1, image::Rgb(color));
    let mut cursor = std::io::Cursor::new(Vec::new());
    image
        .write_to(&mut cursor, image::ImageFormat::Png)
        .unwrap();
    cursor.into_inner()
}
#[test]
fn subscription_status_rejects_api_key_without_exposing_or_changing_auth() {
    for name in ["ok", "auth-api", "auth-none"] {
        let f = Fixture::new(name);
        let status = status(&f.config, Duration::from_secs(5)).unwrap();
        assert_eq!(status.authenticated, name == "ok");
        assert_eq!(status.provider, "chatgpt");
        assert!(!serde_json::to_string(&status).unwrap().contains("sk-fake"));
        assert_eq!(f.calls().len(), 2);
        if name != "ok" {
            let e = complete(&f.config, f.request("chatgpt/gpt-5.5", &[])).unwrap_err();
            assert_eq!(e.kind, FailureKind::Authentication);
            assert!(e.usage.aggregate.is_none());
            assert!(!e.error.to_string().contains("sk-fake"));
            assert!(!f.root.path().join("request.json").exists());
        }
    }
}
#[test]
fn text_and_ordered_images_use_private_files_custom_system_and_terminal_totals() {
    use sha2::{Digest, Sha256};
    let f = Fixture::new("ok");
    let a = png([255, 0, 0]);
    let b = png([0, 0, 255]);
    let out = complete(
        &f.config,
        f.request("chatgpt/gpt-5.5", &[("image/png", &a), ("image/png", &b)]),
    )
    .unwrap();
    assert_eq!(out.text, "Complete authored document.");
    assert_eq!(
        out.usage.aggregate.as_ref().unwrap(),
        &TokenTotals {
            input_tokens: 11,
            output_tokens: 5,
            cached_input_tokens: 3,
            cache_creation_input_tokens: 0,
            reasoning_output_tokens: 2
        }
    );
    let request: Value =
        serde_json::from_slice(&std::fs::read(f.root.path().join("request.json")).unwrap())
            .unwrap();
    assert_eq!(request["system"], "Document instructions only.");
    assert_eq!(request["user"], "Document body\nwith Unicode 中文.");
    assert_eq!(
        request["image_hashes"],
        json!([
            crate::hex(Sha256::digest(&a)),
            crate::hex(Sha256::digest(&b))
        ])
    );
    assert_eq!(request["workspace_mode"], 0o700);
    assert!(
        request["file_modes"]
            .as_array()
            .unwrap()
            .iter()
            .all(|mode| mode == 0o600)
    );
    assert!(!Path::new(request["workspace"].as_str().unwrap()).exists());
}
#[test]
fn unknown_version_model_and_mime_fail_without_document_execution() {
    let f = Fixture::new("version");
    let e = complete(&f.config, f.request("gpt-5.5", &[])).unwrap_err();
    assert_eq!(e.kind, FailureKind::Unsupported);
    assert_eq!(f.calls().len(), 1);
    let f = Fixture::new("ok");
    let e = complete(&f.config, f.request("gpt-6-sol", &[])).unwrap_err();
    assert_eq!(e.kind, FailureKind::Unsupported);
    assert!(f.calls().is_empty());
    let e = complete(
        &f.config,
        f.request("gpt-5.5", &[("image/png", b"not an image")]),
    )
    .unwrap_err();
    assert_eq!(e.kind, FailureKind::InvalidRequest);
    assert!(!f.root.path().join("request.json").exists());
}
#[test]
fn missing_terminal_errors_and_tool_events_never_publish_partial_text() {
    for (name, kind) in [
        ("truncated", FailureKind::Truncated),
        ("failed", FailureKind::Transport),
        ("tool", FailureKind::Permission),
        ("bad-count", FailureKind::Protocol),
    ] {
        let f = Fixture::new(name);
        let e = complete(&f.config, f.request("gpt-5.5", &[])).unwrap_err();
        assert_eq!(e.kind, kind, "{name}");
        assert!(e.usage.aggregate.is_none());
        assert!(!e.error.to_string().contains("sk-fake"));
    }
}
#[test]
fn observed_aggregate_survives_post_terminal_protocol_or_exit_failure() {
    for name in ["after-terminal", "nonzero"] {
        let f = Fixture::new(name);
        let e = complete(&f.config, f.request("gpt-5.5", &[])).unwrap_err();
        let totals = e.usage.aggregate.unwrap();
        assert_eq!((totals.input_tokens, totals.output_tokens), (11, 5));
    }
    let f = Fixture::new("zero");
    let out = complete(&f.config, f.request("gpt-5.5", &[])).unwrap();
    assert_eq!(out.usage.aggregate, Some(TokenTotals::default()));
    assert!(out.warnings.iter().any(|s| s.contains("unavailable")));
}
#[test]
fn cancellation_and_deadline_stop_private_process_tree_without_stale_files() {
    use std::sync::atomic::Ordering;
    let f = Fixture::new("sleep");
    let stop = AtomicBool::new(false);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            let until = Instant::now() + Duration::from_secs(5);
            while !f.root.path().join("grandchild.pid").exists() && Instant::now() < until {
                std::thread::sleep(Duration::from_millis(10));
            }
            stop.store(true, Ordering::Release);
        });
        let mut request = f.request("gpt-5.5", &[]);
        request.cancel = Some(&stop);
        let e = complete(&f.config, request).unwrap_err();
        assert_eq!(e.kind, FailureKind::Cancelled);
    });
    let pid: i32 = std::fs::read_to_string(f.root.path().join("grandchild.pid"))
        .unwrap()
        .parse()
        .unwrap();
    // A killed descendant may briefly remain a zombie until the OS parent reaps it.
    let status = std::process::Command::new("ps")
        .args(["-p", &pid.to_string(), "-o", "stat="])
        .output()
        .unwrap();
    let state = String::from_utf8_lossy(&status.stdout);
    assert!(
        state.trim().is_empty() || state.trim().starts_with('Z'),
        "{state}"
    );
    let request: Value =
        serde_json::from_slice(&std::fs::read(f.root.path().join("request.json")).unwrap())
            .unwrap();
    assert!(!Path::new(request["workspace"].as_str().unwrap()).exists());
    let f = Fixture::new("sleep");
    let mut request = f.request("gpt-5.5", &[]);
    request.timeout = Duration::from_millis(500);
    assert_eq!(
        complete(&f.config, request).unwrap_err().kind,
        FailureKind::Timeout
    );
}
#[test]
fn stderr_is_bounded_and_never_appears_in_public_errors() {
    let f = Fixture::new("stderr");
    let e = complete(&f.config, f.request("gpt-5.5", &[])).unwrap_err();
    assert_eq!(e.kind, FailureKind::ResourceLimit);
    assert!(e.error.to_string().len() < 256);
}
#[test]
fn catalog_list_is_a_restricted_capability_allowlist_not_entitlement_discovery() {
    let f = Fixture::new("ok");
    let list = models(&f.config, Duration::from_secs(5)).unwrap();
    assert_eq!(list.len(), 1);
    assert_eq!(list[0].model, "chatgpt/gpt-5.5");
    assert!(list[0].supports_vision);
}
