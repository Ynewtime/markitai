#![cfg(unix)]
use super::*;
use std::os::unix::fs::PermissionsExt;
use std::sync::atomic::Ordering;

struct Fixture {
    root: tempfile::TempDir,
    config: Config,
    record: PathBuf,
}
impl Fixture {
    fn new(mode: &str) -> Self {
        let root = tempfile::tempdir().unwrap();
        let script = root.path().join("fixture.py");
        std::fs::write(&script, include_bytes!("fake_cli.py")).unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o700)).unwrap();
        let record = root.path().join("calls.jsonl");
        let mut environment = HashMap::new();
        for key in ["HOME", "PATH", "TMPDIR", "LANG"] {
            if let Ok(v) = std::env::var(key) {
                environment.insert(key.into(), v);
            }
        }
        environment.insert("MARKITAI_CLAUDE_FIXTURE".into(), mode.into());
        environment.insert(
            "MARKITAI_CLAUDE_RECORD".into(),
            record.to_string_lossy().into_owned(),
        );
        Self {
            config: Config {
                executable: script,
                environment,
            },
            record,
            root,
        }
    }
    fn request<'a>(&self, user: &'a str) -> Request<'a> {
        Request {
            model: "claude-agent/sonnet",
            system: "system is separate",
            user,
            images: &[],
            timeout: Duration::from_secs(10),
            cancel: None,
        }
    }
    fn rows(&self) -> Vec<Value> {
        std::fs::read_to_string(&self.record)
            .unwrap_or_default()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }
    fn clean(&self) {
        for row in self.rows() {
            if let Some(path) = row["cwd"].as_str() {
                assert!(
                    !Path::new(path).exists(),
                    "private runtime workspace must be removed"
                );
                assert_eq!(row["mode"], 0o700);
            }
        }
        assert!(self.root.path().exists());
    }
}

#[test]
fn cli_status_does_not_read_credentials_or_accept_api_billing() {
    for (mode, logged) in [("ok", true), ("signed-out", false), ("byok-status", false)] {
        let f = Fixture::new(mode);
        let result = status(&f.config, Duration::from_secs(10)).unwrap();
        assert_eq!(result.authenticated, logged);
        assert_eq!(result.provider, "claude-agent");
        assert!(
            !serde_json::to_string(&result)
                .unwrap()
                .contains("do-not-echo")
        );
        assert_eq!(f.rows().len(), 2);
        f.clean();
    }
}
#[test]
fn catalog_uses_initialize_without_sending_a_prompt() {
    let f = Fixture::new("ok");
    let rows = models(&f.config, Duration::from_secs(10)).unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].model, "claude-agent/sonnet");
    assert!(
        !rows[0].supports_vision,
        "official catalog does not advertise this bit"
    );
    assert!(!f.rows().iter().any(|v| v.get("request").is_some()));
    f.clean();
}
#[test]
fn terminal_text_and_ordered_images_preserve_bytes_and_do_not_add_cost_estimates() {
    let f = Fixture::new("ok");
    let user = "首段🦀\n[asset](a%20b.png#x)\n```rust\nlet x = \"literal\";\n```\nTAIL";
    let images: &[(&str, &[u8])] = &[("image/png", b"png bytes"), ("image/jpeg", b"jpeg bytes")];
    let result = complete(
        &f.config,
        Request {
            images,
            ..f.request(user)
        },
    )
    .unwrap();
    assert_eq!(result.text, user);
    assert_eq!(result.usage.calls.len(), 1);
    assert_eq!(result.usage.calls[0].input_tokens, Some(11));
    assert_eq!(
        result.usage.aggregate.as_ref().unwrap().by_model["claude-fixture-1"].output_tokens,
        Some(7)
    );
    let json = serde_json::to_string(&result.usage).unwrap();
    assert!(!json.contains("9999"));
    assert!(!json.contains("cost"));
    let rows = f.rows();
    let sent = &rows.iter().find(|r| r.get("request").is_some()).unwrap()["request"];
    assert_eq!(sent["system"], "system is separate");
    assert_eq!(sent["text"], user);
    use sha2::Digest;
    for (i, (mime, data)) in images.iter().enumerate() {
        assert_eq!(sent["images"][i]["mime"], *mime);
        assert_eq!(
            sent["images"][i]["sha256"],
            crate::hex(sha2::Sha256::digest(data))
        );
    }
    f.clean();
}
#[test]
fn repeated_message_usage_is_deduplicated_and_conflict_keeps_first_evidence() {
    let f = Fixture::new("duplicate");
    assert_eq!(
        complete(&f.config, f.request("body"))
            .unwrap()
            .usage
            .calls
            .len(),
        1
    );
    f.clean();
    let f = Fixture::new("conflict");
    let failure = complete(&f.config, f.request("body")).unwrap_err();
    assert_eq!(failure.kind, FailureKind::Protocol);
    assert_eq!(failure.usage.calls.len(), 1);
    assert_eq!(failure.usage.calls[0].output_tokens, Some(7));
    f.clean();
}
#[test]
fn terminal_only_tokens_do_not_fabricate_requests_and_missing_usage_stays_unknown() {
    let f = Fixture::new("aggregate-only");
    let result = complete(&f.config, f.request("body")).unwrap();
    assert!(result.usage.calls.is_empty());
    assert_eq!(
        result.usage.aggregate.unwrap().totals.input_tokens,
        Some(11)
    );
    f.clean();
    let f = Fixture::new("unknown-usage");
    let result = complete(&f.config, f.request("body")).unwrap();
    assert!(result.usage.calls.is_empty());
    assert!(result.usage.aggregate.is_none());
    f.clean();
}
#[test]
fn paid_errors_never_return_partial_success_and_keep_observed_usage() {
    for (mode, kind) in [
        ("paid-auth-error", FailureKind::Authentication),
        ("missing-terminal", FailureKind::Protocol),
        ("aborted", FailureKind::Cancelled),
        ("truncated", FailureKind::Truncated),
        ("wrong-session", FailureKind::Protocol),
        ("extra-result", FailureKind::Protocol),
        ("bad-exit", FailureKind::Transport),
        ("malformed-paid", FailureKind::Protocol),
        ("output-limit", FailureKind::ResourceLimit),
    ] {
        let f = Fixture::new(mode);
        let failure = complete(&f.config, f.request("body")).unwrap_err();
        assert_eq!(failure.kind, kind, "{mode}");
        assert_eq!(failure.usage.calls.len(), 1, "{mode}");
        assert!(!failure.error.to_string().contains("SECRET"));
        f.clean();
    }
}
#[test]
fn callbacks_tools_and_model_switch_are_rejected_without_automatic_retry() {
    for (mode, kind, calls) in [
        ("callback-init", FailureKind::Permission, 0),
        ("paid-callback", FailureKind::Permission, 1),
        ("tool", FailureKind::Permission, 1),
        ("wrong-model", FailureKind::Unsupported, 1),
        ("wrong-provider", FailureKind::Authentication, 0),
    ] {
        let f = Fixture::new(mode);
        let failure = complete(&f.config, f.request("body")).unwrap_err();
        assert_eq!(failure.kind, kind, "{mode}");
        assert_eq!(failure.usage.calls.len(), calls, "{mode}");
        assert_eq!(
            f.rows()
                .iter()
                .filter(|r| r["args"]
                    .as_array()
                    .is_some_and(|a| a.iter().any(|v| v == "--print")))
                .count(),
            1
        );
        f.clean();
    }
}
#[test]
fn deadlines_and_stderr_limit_stop_processes_and_preserve_paid_evidence() {
    for mode in [
        "hang-init",
        "hang-paid",
        "stderr-flood",
        "stderr-after-terminal",
    ] {
        let f = Fixture::new(mode);
        let start = Instant::now();
        let failure = complete(
            &f.config,
            Request {
                timeout: Duration::from_secs(2),
                ..f.request("body")
            },
        )
        .unwrap_err();
        assert_eq!(
            failure.kind,
            if matches!(mode, "stderr-flood" | "stderr-after-terminal") {
                FailureKind::ResourceLimit
            } else {
                FailureKind::Timeout
            }
        );
        assert!(start.elapsed() < Duration::from_secs(6));
        if mode == "hang-paid" {
            assert_eq!(failure.usage.calls.len(), 1);
        }
        f.clean();
    }
}
#[test]
fn cancellation_preflight_and_wrong_version_do_not_dispatch_paid_input() {
    let f = Fixture::new("ok");
    let cancelled = AtomicBool::new(true);
    assert_eq!(
        complete(
            &f.config,
            Request {
                cancel: Some(&cancelled),
                ..f.request("body")
            }
        )
        .unwrap_err()
        .kind,
        FailureKind::Cancelled
    );
    assert!(f.rows().is_empty());
    cancelled.store(false, Ordering::Release);
    let f = Fixture::new("wrong-version");
    assert_eq!(
        complete(&f.config, f.request("body")).unwrap_err().kind,
        FailureKind::Unsupported
    );
    assert_eq!(f.rows().len(), 1);
    f.clean();
    let f = Fixture::new("ok");
    let oversized = "x".repeat(TEXT_LIMIT + 1);
    assert_eq!(
        complete(&f.config, f.request(&oversized)).unwrap_err().kind,
        FailureKind::ResourceLimit
    );
    assert!(f.rows().is_empty());
}
#[test]
fn config_does_not_capture_oauth_or_byok_overrides_and_keeps_home_unchanged() {
    let f = Fixture::new("ok");
    let original = std::env::var_os("HOME");
    let mut env = HashMap::from([
        (
            "CLAUDE_CLI_PATH".into(),
            f.config.executable.to_string_lossy().into_owned(),
        ),
        ("ANTHROPIC_API_KEY".into(), "not-a-real-key".into()),
        ("CLAUDE_CODE_OAUTH_TOKEN".into(), "not-a-real-token".into()),
        ("ANTHROPIC_BASE_URL".into(), "http://wrong.invalid".into()),
    ]);
    env.insert(
        "CLAUDE_CONFIG_DIR".into(),
        f.root
            .path()
            .join("private-official-config")
            .to_string_lossy()
            .into_owned(),
    );
    let config = Config::from_env(&env).unwrap();
    assert!(config.environment.contains_key("CLAUDE_CONFIG_DIR"));
    assert!(
        !config
            .environment
            .keys()
            .any(|k| k.starts_with("ANTHROPIC_") || k == "CLAUDE_CODE_OAUTH_TOKEN")
    );
    assert_eq!(std::env::var_os("HOME"), original);
}

#[test]
fn advertised_builtin_agent_metadata_does_not_grant_tools_or_background_authority() {
    let f = Fixture::new("catalog-only");
    let result = complete(&f.config, f.request("complete unchanged body")).unwrap();
    assert_eq!(result.text, "complete unchanged body");
    assert_eq!(result.usage.calls.len(), 1);
    f.clean();
    for (mode, kind) in [
        ("catalog-custom", FailureKind::Permission),
        ("catalog-agent-tool", FailureKind::Permission),
        ("catalog-task", FailureKind::Permission),
        ("catalog-callback", FailureKind::Permission),
        ("catalog-subagent", FailureKind::Permission),
        ("catalog-user-message", FailureKind::Protocol),
    ] {
        let f = Fixture::new(mode);
        let failure = complete(&f.config, f.request("body")).unwrap_err();
        assert_eq!(failure.kind, kind, "{mode}");
        assert_eq!(
            f.rows()
                .iter()
                .filter(|row| row.get("request").is_some())
                .count(),
            1
        );
        f.clean();
    }
}
