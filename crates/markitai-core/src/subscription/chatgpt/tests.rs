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
    // A name that cannot be a model is refused before Codex is started.
    let e = complete(&f.config, f.request("gpt 6; rm", &[])).unwrap_err();
    assert_eq!(e.kind, FailureKind::InvalidRequest);
    assert!(f.calls().is_empty());
    // A model the installed Codex does not offer is refused before any
    // document request, naming the ones it offers.
    let e = complete(&f.config, f.request("gpt-4-retired", &[])).unwrap_err();
    assert_eq!(e.kind, FailureKind::Unsupported);
    assert_eq!(
        e.error.to_string(),
        "The installed Codex does not offer chatgpt/gpt-4-retired; it offers chatgpt/gpt-6.2-nova, chatgpt/gpt-6-astra, chatgpt/gpt-6-luna, chatgpt/gpt-5.5"
    );
    assert!(!f.root.path().join("request.json").exists());
    let e = complete(
        &f.config,
        f.request("gpt-5.5", &[("image/png", b"not an image")]),
    )
    .unwrap_err();
    assert_eq!(e.kind, FailureKind::InvalidRequest);
    assert!(!f.root.path().join("request.json").exists());
}
#[test]
fn a_newer_codex_release_is_used_and_an_older_one_refused() {
    let f = Fixture::new("ok");
    fake_runtime::install(f.root.path(), &json!({"name":"ok","version":"0.172.1"}));
    let status = status(&f.config, Duration::from_secs(5)).unwrap();
    assert_eq!(status.details["cli_version"], "0.172.1");
    assert_eq!(status.details["newer_than_verified"], true);
    assert!(complete(&f.config, f.request("gpt-5.5", &[])).is_ok());
    fake_runtime::install(f.root.path(), &json!({"name":"ok","version":"0.158.9"}));
    let e = complete(&f.config, f.request("gpt-5.5", &[])).unwrap_err();
    assert_eq!(e.kind, FailureKind::Unsupported);
}
#[test]
fn missing_terminal_errors_and_tool_events_never_publish_partial_text() {
    for (name, kind) in [
        ("truncated", FailureKind::Truncated),
        ("failed", FailureKind::Transport),
        ("rate-limited", FailureKind::RateLimit),
        ("usage-limit", FailureKind::ResourceLimit),
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
fn the_models_are_the_ones_the_installed_codex_offers_its_default_first() {
    let f = Fixture::new("ok");
    let list = models(&f.config, Duration::from_secs(5)).unwrap();
    let names: Vec<&str> = list.iter().map(|model| model.model.as_str()).collect();
    // The account's list, hidden entries left out, in Codex's own order.
    assert_eq!(
        names,
        [
            "chatgpt/gpt-6.2-nova",
            "chatgpt/gpt-6-astra",
            "chatgpt/gpt-6-luna",
            "chatgpt/gpt-5.5"
        ]
    );
    assert_eq!(list[2].label, "GPT-6-LUNA");
    assert!(list.iter().all(|model| model.supports_vision));
    // The default is the lightest model of the binary's own list, not its
    // first (gpt-6-astra).
    assert_eq!(
        default_model(&f.config, Duration::from_secs(5)),
        "gpt-6-luna"
    );
    // A Codex that cannot list its models: no list, and Python's default.
    let f = Fixture::new("ok");
    fake_runtime::install(f.root.path(), &json!({"name":"ok","catalog":"none"}));
    let e = models(&f.config, Duration::from_secs(5)).unwrap_err();
    assert_eq!(e.kind, FailureKind::Protocol);
    assert!(e.error.to_string().contains("cannot list its models"));
    assert_eq!(
        default_model(&f.config, Duration::from_secs(5)),
        FALLBACK_MODEL
    );
}
#[test]
fn each_offered_model_runs_with_its_own_restricted_catalog_entry() {
    let f = Fixture::new("ok");
    // In the binary's list: no account refresh is needed.
    complete(&f.config, f.request("gpt-6-luna", &[])).unwrap();
    let sent: Value =
        serde_json::from_slice(&std::fs::read(f.root.path().join("request.json")).unwrap())
            .unwrap();
    assert_eq!(
        (sent["model"].as_str(), sent["catalog"].as_bool()),
        (Some("gpt-6-luna"), Some(true))
    );
    assert!(!f.root.path().join("refreshes.jsonl").exists());
    // Newer than the binary: found in the account's list.
    complete(&f.config, f.request("chatgpt/gpt-6.2-nova", &[])).unwrap();
    let sent: Value =
        serde_json::from_slice(&std::fs::read(f.root.path().join("request.json")).unwrap())
            .unwrap();
    assert_eq!(sent["model"], "gpt-6.2-nova");
    assert!(f.root.path().join("refreshes.jsonl").exists());
    // A Codex that cannot list its models: Codex's own catalog for every
    // model (this build compiles none in).
    let f = Fixture::new("ok");
    fake_runtime::install(f.root.path(), &json!({"name":"ok","catalog":"none"}));
    for model in ["gpt-5.5", "gpt-6-luna"] {
        complete(&f.config, f.request(model, &[])).unwrap();
        let sent: Value =
            serde_json::from_slice(&std::fs::read(f.root.path().join("request.json")).unwrap())
                .unwrap();
        assert_eq!(sent["catalog"], false, "{model}");
    }
}
#[test]
fn the_default_is_the_lightest_current_model_codex_lists() {
    let entry = |slug: &str, description: &str, upgrade: Value| json!({"slug":slug,"description":description,"upgrade":upgrade});
    let pick = |list: Vec<Value>| lightest(&list).map(str::to_owned);
    // Codex 0.163.0's order: the newest current luna, not an older one that
    // points to it, nor the first entry.
    let codex = vec![
        entry("gpt-6.1-sol", "Latest workhorse model", Value::Null),
        entry("gpt-6-astra", "Frontier intelligence", Value::Null),
        entry(
            "gpt-5.6-luna",
            "Older fast and efficient model.",
            json!({"model":"gpt-6-luna"}),
        ),
        entry(
            "gpt-6-luna",
            "Fast and affordable model for easier tasks.",
            Value::Null,
        ),
    ];
    assert_eq!(pick(codex).as_deref(), Some("gpt-6-luna"));
    // No luna: a small name, or a description that says it is affordable.
    let small = vec![
        entry("gpt-7", "Frontier", Value::Null),
        entry("gpt-7-mini", "Small model", Value::Null),
    ];
    assert_eq!(pick(small).as_deref(), Some("gpt-7-mini"));
    let said = vec![
        entry("gpt-7", "Frontier", Value::Null),
        entry(
            "gpt-7-swift",
            "A lower cost model for simple work",
            Value::Null,
        ),
    ];
    assert_eq!(pick(said).as_deref(), Some("gpt-7-swift"));
    // Only an outdated luna: still the lightest.
    let old = vec![
        entry("gpt-7", "Frontier", Value::Null),
        entry("gpt-5.6-luna", "Older", json!({"model":"gpt-7"})),
    ];
    assert_eq!(pick(old).as_deref(), Some("gpt-5.6-luna"));
    // Nothing marked light: Codex's own default.
    let plain = vec![
        entry("gpt-7", "Frontier", Value::Null),
        entry("gpt-7-pro", "Pro", Value::Null),
    ];
    assert_eq!(pick(plain).as_deref(), Some("gpt-7"));
    assert_eq!(pick(Vec::new()), None);
    // "lunar" is not the luna family.
    let lunar = vec![
        entry("gpt-7", "x", Value::Null),
        entry("gpt-lunar", "y", Value::Null),
    ];
    assert_eq!(pick(lunar).as_deref(), Some("gpt-7"));
}
