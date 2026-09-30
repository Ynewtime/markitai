use super::*;
#[cfg_attr(not(unix), allow(unused_imports))]
use serde_json::{Value, json};
use std::io::Cursor;

#[test]
fn framing_rejects_ambiguity_truncation_and_budgets() {
    for packet in [
        b"Content-Length: 2\r\nContent-Length: 2\r\n\r\n{}".as_slice(),
        b"Content-Length: 3\r\n\r\n{}",
        b"Content-Length: +2\r\n\r\n{}",
        b"Content-Length: 999999999999\r\n\r\n",
        b"Content-Length: 2\r\n\r\nxx",
    ] {
        assert!(process::read_frame(&mut Cursor::new(packet), &mut 0).is_err());
    }
    let body = serde_json::to_vec(&json!({"text":"结束🚀"})).unwrap();
    let mut packet = format!(
        "Content-Length: {}\r\nContent-Type: application/vscode-jsonrpc; charset=utf-8\r\n\r\n",
        body.len()
    )
    .into_bytes();
    packet.extend(body);
    assert_eq!(
        process::read_frame(&mut Cursor::new(packet), &mut 0).unwrap()["text"],
        "结束🚀"
    );
}

#[cfg(unix)]
mod unix {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };
    use std::time::Instant;
    struct Fixture {
        root: tempfile::TempDir,
        cfg: CopilotConfig,
    }
    impl Fixture {
        fn new(mode: &str) -> Self {
            let root = tempfile::Builder::new()
                .prefix("copilot fake space ")
                .tempdir()
                .unwrap();
            let executable = root.path().join("copilot fixture");
            std::fs::write(&executable, include_str!("fake_cli.py")).unwrap();
            std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700)).unwrap();
            std::fs::write(
                root.path().join("fixture.json"),
                serde_json::to_vec(&json!({"mode":mode,"home":std::env::var("HOME").ok(),"cache_home":root.path().join("private cache")}))
                    .unwrap(),
            )
            .unwrap();
            let env = HashMap::from([
                (
                    "COPILOT_CLI_PATH".into(),
                    executable.to_string_lossy().into_owned(),
                ),
                (
                    "COPILOT_HOME".into(),
                    root.path().to_string_lossy().into_owned(),
                ),
                (
                    "COPILOT_CACHE_HOME".into(),
                    root.path()
                        .join("private cache")
                        .to_string_lossy()
                        .into_owned(),
                ),
                ("COPILOT_GITHUB_TOKEN".into(), "test-only-secret".into()),
                ("GH_TOKEN".into(), "lower-priority-secret".into()),
                ("OPENAI_API_KEY".into(), "must-not-leak-to-child".into()),
                (
                    "COPILOT_PROVIDER_API_KEY".into(),
                    "must-not-select-BYOK".into(),
                ),
            ]);
            let cfg = CopilotConfig::from_env(&env).unwrap();
            Self { root, cfg }
        }
        fn request<'a>(&self, text: &'a str, cancel: Option<&'a AtomicBool>) -> Request<'a> {
            Request {
                model: "copilot/fixture",
                system: "fixed system",
                user: text,
                images: &[],
                timeout: Duration::from_secs(5),
                cancel,
            }
        }
        fn calls(&self) -> Vec<Value> {
            std::fs::read_to_string(self.root.path().join("requests.jsonl"))
                .unwrap_or_default()
                .lines()
                .map(|line| serde_json::from_str(line).unwrap())
                .collect()
        }
        fn assert_private_cwd_removed(&self) {
            let cwd = std::fs::read_to_string(self.root.path().join("cwd")).unwrap();
            assert!(!std::path::Path::new(&cwd).exists());
            assert_eq!(
                std::fs::read_to_string(self.root.path().join("cwd-mode")).unwrap(),
                "0o700"
            );
        }
    }
    #[test]
    fn official_status_and_models_are_not_estimated_from_path_presence() {
        let fixture = Fixture::new("normal");
        let observed_status = status(&fixture.cfg, Duration::from_secs(5)).unwrap();
        assert!(observed_status.authenticated);
        assert_eq!(observed_status.user.as_deref(), Some("fixture-user"));
        assert_eq!(
            observed_status.details["verification"],
            "runtime-auth-status"
        );
        let models = models(&fixture.cfg, Duration::from_secs(5)).unwrap();
        assert_eq!(
            models
                .iter()
                .map(|model| model.model.as_str())
                .collect::<Vec<_>>(),
            ["copilot/a", "copilot/z"]
        );
        assert!(!models[0].supports_vision);
        assert!(models[1].supports_vision);
        fixture.assert_private_cwd_removed();
        let fixture = Fixture::new("unauth");
        assert!(
            !status(&fixture.cfg, Duration::from_secs(5))
                .unwrap()
                .authenticated
        );
        assert_eq!(
            complete(&fixture.cfg, fixture.request("text", None))
                .unwrap_err()
                .kind,
            FailureKind::Authentication
        );
        assert!(
            !fixture
                .calls()
                .iter()
                .any(|call| call["method"] == "session.send")
        );
    }
    #[test]
    fn connect_rejects_legacy_display_fields_and_preserves_private_cache_override() {
        let fixture = Fixture::new("normal");
        {
            let mut process =
                process::Process::spawn(&fixture.cfg, Duration::from_secs(5), None).unwrap();
            let failure = process.call("connect", json!({"clientInfo":{"name":"markitai","version":"0.1.0"},"supportedTaskKinds":[]}), None, &mut |_| Ok(())).unwrap_err();
            assert_eq!(failure.kind, FailureKind::Protocol);
        }
        assert!(
            status(&fixture.cfg, Duration::from_secs(5))
                .unwrap()
                .authenticated
        );
        assert_eq!(
            std::fs::read_to_string(
                fixture
                    .root
                    .path()
                    .join("private cache/fixture-cache-access")
            )
            .unwrap(),
            "private cache preserved"
        );
        let calls = fixture.calls();
        let valid = calls
            .iter()
            .rev()
            .find(|v| v["method"] == "connect")
            .unwrap();
        assert_eq!(valid["params"]["clientInfo"]["editorName"], "markitai");
        assert!(valid["params"]["clientInfo"].get("name").is_none());
        assert!(valid["params"]["clientInfo"].get("version").is_none());
        fixture.assert_private_cwd_removed();
    }
    #[test]
    fn complete_preserves_unicode_chunks_and_ordered_memory_images() {
        let fixture = Fixture::new("split");
        let text =
            "first\n\n```rust\nlet s = \"结束🚀\";\n```\n[link](target)\nPage number: 21\nEND";
        let images = [
            ("image/png", b"\x89PNG\r\n\x1a\nfixture".as_slice()),
            ("image/png", b"\x89PNG\r\n\x1a\nfixture".as_slice()),
        ];
        let mut request = fixture.request(text, None);
        request.images = &images;
        let result = complete(&fixture.cfg, request).unwrap();
        assert_eq!(result.text, text);
        assert_eq!(result.usage.calls.len(), 1);
        assert_eq!(result.usage.calls[0].input_tokens, Some(7));
        assert_eq!(result.usage.calls[0].output_tokens, Some(3));
        assert!(
            !serde_json::to_string(&result.usage)
                .unwrap()
                .contains("12.5")
        );
        fixture.assert_private_cwd_removed();
    }
    #[test]
    fn duplicate_usage_events_and_api_ids_do_not_double_count() {
        let fixture = Fixture::new("duplicate");
        let result = complete(&fixture.cfg, fixture.request("text", None)).unwrap();
        assert_eq!(result.usage.calls.len(), 1);
        let fixture = Fixture::new("conflict");
        let failure = complete(&fixture.cfg, fixture.request("text", None)).unwrap_err();
        assert_eq!(failure.kind, FailureKind::Protocol);
        assert_eq!(failure.usage.calls.len(), 1);
        assert_eq!(failure.usage.calls[0].input_tokens, Some(7));
    }
    #[test]
    fn no_usage_is_unknown_and_paid_terminal_error_retains_observations() {
        let fixture = Fixture::new("no_usage");
        let result = complete(&fixture.cfg, fixture.request("text", None)).unwrap();
        assert!(result.usage.calls.is_empty());
        assert!(result.warnings[0].contains("unknown"));
        let fixture = Fixture::new("paid_error");
        let failure = complete(&fixture.cfg, fixture.request("text", None)).unwrap_err();
        assert_eq!(failure.kind, FailureKind::Authentication);
        assert_eq!(failure.usage.calls.len(), 1);
        assert!(!failure.error.to_string().contains("credential-must"));
        assert_eq!(
            fixture
                .calls()
                .iter()
                .filter(|call| call["method"] == "session.send")
                .count(),
            1
        );
    }
    #[test]
    fn missing_or_unsuccessful_terminal_never_publishes_partial_text() {
        for (mode, kind) in [
            ("eof", FailureKind::Transport),
            ("empty", FailureKind::Truncated),
            ("aborted", FailureKind::Cancelled),
            ("length", FailureKind::Truncated),
        ] {
            let fixture = Fixture::new(mode);
            let failure = complete(
                &fixture.cfg,
                fixture.request("must not become success", None),
            )
            .unwrap_err();
            assert_eq!(failure.kind, kind, "{mode}");
            assert_eq!(failure.usage.calls.len(), 1);
            fixture.assert_private_cwd_removed();
        }
    }
    #[test]
    fn callback_and_permission_requests_are_denied_without_replay() {
        for mode in ["callback", "permission"] {
            let fixture = Fixture::new(mode);
            let failure = complete(&fixture.cfg, fixture.request("text", None)).unwrap_err();
            assert_eq!(failure.kind, FailureKind::Permission);
            assert!(!failure.error.to_string().contains("credential-must"));
            assert_eq!(
                fixture
                    .calls()
                    .iter()
                    .filter(|call| call["method"] == "session.send")
                    .count(),
                1
            );
            fixture.assert_private_cwd_removed();
        }
    }
    #[test]
    fn model_and_billing_route_switches_fail_but_preserve_actual_usage() {
        for (mode, kind) in [
            ("subagent", FailureKind::Permission),
            ("byok", FailureKind::Unsupported),
            ("wrong_model", FailureKind::Unsupported),
        ] {
            let fixture = Fixture::new(mode);
            let failure = complete(&fixture.cfg, fixture.request("text", None)).unwrap_err();
            assert_eq!(failure.kind, kind, "{mode}");
            assert_eq!(failure.usage.calls.len(), 1);
        }
    }
    #[test]
    fn protocol_version_and_response_limits_fail_closed() {
        for (mode, kind) in [
            ("wrong_protocol", FailureKind::Unsupported),
            ("oversized", FailureKind::ResourceLimit),
        ] {
            let fixture = Fixture::new(mode);
            assert_eq!(
                complete(&fixture.cfg, fixture.request("text", None))
                    .unwrap_err()
                    .kind,
                kind
            );
            fixture.assert_private_cwd_removed();
        }
    }
    #[test]
    fn timeout_and_cancellation_terminate_descendant_pipe_holders() {
        for cancelled in [false, true] {
            let fixture = Fixture::new("hang");
            let cancel = Arc::new(AtomicBool::new(false));
            let notify = cancel.clone();
            let ready = fixture.root.path().join("descendant");
            let sender = cancelled.then(|| {
                std::thread::spawn(move || {
                    let wait = Instant::now();
                    while !ready.exists() && wait.elapsed() < Duration::from_secs(2) {
                        std::thread::sleep(Duration::from_millis(10));
                    }
                    notify.store(true, Ordering::Release);
                })
            });
            let mut request = fixture.request("text", Some(&cancel));
            request.timeout = Duration::from_secs(if cancelled { 3 } else { 1 });
            let started = Instant::now();
            let failure = complete(&fixture.cfg, request).unwrap_err();
            assert_eq!(
                failure.kind,
                if cancelled {
                    FailureKind::Cancelled
                } else {
                    FailureKind::Timeout
                }
            );
            assert!(started.elapsed() < Duration::from_secs(4));
            if let Some(sender) = sender {
                sender.join().unwrap();
            }
            fixture.assert_private_cwd_removed();
        }
    }
    #[test]
    fn request_validation_happens_before_spawn() {
        let fixture = Fixture::new("normal");
        let mut request = fixture.request("text", None);
        request.model = "fixture\ninvalid";
        assert_eq!(
            complete(&fixture.cfg, request).unwrap_err().kind,
            FailureKind::ResourceLimit
        );
        assert!(!fixture.root.path().join("pid").exists());
    }
}
