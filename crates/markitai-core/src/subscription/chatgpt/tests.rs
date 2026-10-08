use super::*;
use crate::subscription::fake_runtime;

struct Fixture {
    root: tempfile::TempDir,
    config: Config,
}
impl Fixture {
    fn new(name: &str) -> Self {
        Self::with_program(name, &fake_runtime::program())
    }
    /// A runtime started through `program`: the stand-in itself, or a script
    /// that runs it.
    fn with_program(name: &str, program: &Path) -> Self {
        let root = tempfile::tempdir().unwrap();
        fake_runtime::install(root.path(), &json!({"name":name}));
        let config = Config::from_env(&HashMap::from([
            (
                "CODEX_CLI_PATH".into(),
                program.to_string_lossy().into_owned(),
            ),
            (
                "CODEX_HOME".into(),
                root.path().to_string_lossy().into_owned(),
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
    #[cfg(unix)]
    {
        assert_eq!(request["workspace_mode"], 0o700);
        let modes = request["file_modes"].as_array().unwrap();
        assert_eq!(modes.len(), 4);
        assert!(modes.iter().all(|mode| mode == 0o600));
    }
    assert!(!Path::new(request["workspace"].as_str().unwrap()).exists());
}
/// npm installs `codex.cmd`. The command processor must hand the runtime
/// the same arguments, including the JSON configuration values with quotes.
#[cfg(windows)]
#[test]
fn a_command_script_shim_runs_the_runtime_with_its_arguments_intact() {
    let shims = tempfile::Builder::new()
        .prefix("npm shim space ")
        .tempdir()
        .unwrap();
    std::fs::write(
        shims.path().join("codex.cmd"),
        format!("@\"{}\" %*\r\n", fake_runtime::program().display()),
    )
    .unwrap();
    let f = Fixture::with_program("ok", &shims.path().join("codex"));
    assert!(
        status(&f.config, Duration::from_secs(10))
            .unwrap()
            .authenticated
    );
    let out = complete(
        &f.config,
        f.request("chatgpt/gpt-5.5", &[("image/png", &png([1, 2, 3]))]),
    )
    .unwrap();
    assert_eq!(out.text, "Complete authored document.");
    let request: Value =
        serde_json::from_slice(&std::fs::read(f.root.path().join("request.json")).unwrap())
            .unwrap();
    assert_eq!(request["user"], "Document body\nwith Unicode 中文.");
    assert_eq!(request["image_hashes"].as_array().unwrap().len(), 1);
}
/// Whether `pid` names a process that has not ended (a Unix zombie has).
fn running(pid: u32) -> bool {
    #[cfg(unix)]
    {
        let output = std::process::Command::new("ps")
            .args(["-p", &pid.to_string(), "-o", "stat="])
            .output()
            .unwrap();
        let state = String::from_utf8_lossy(&output.stdout);
        !state.trim().is_empty() && !state.trim().starts_with('Z')
    }
    #[cfg(windows)]
    {
        use windows_sys::Win32::Foundation::{CloseHandle, FALSE, WAIT_TIMEOUT};
        use windows_sys::Win32::System::Threading::{
            OpenProcess, PROCESS_SYNCHRONIZE, WaitForSingleObject,
        };
        let process = unsafe { OpenProcess(PROCESS_SYNCHRONIZE, FALSE, pid) };
        if process.is_null() {
            return false;
        }
        let alive = unsafe { WaitForSingleObject(process, 0) } == WAIT_TIMEOUT;
        unsafe { CloseHandle(process) };
        alive
    }
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
    let pid: u32 = std::fs::read_to_string(f.root.path().join("grandchild.pid"))
        .unwrap()
        .parse()
        .unwrap();
    // A killed descendant may briefly remain a zombie until the OS parent reaps it.
    assert!(!running(pid), "descendant {pid} survived");
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
