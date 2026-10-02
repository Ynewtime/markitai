//! `markitai auth` delegates status and login to the official runtimes,
//! played here by this test binary (see `fake_runtime`).
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

#[path = "../../markitai-core/src/subscription/fake_runtime.rs"]
mod fake_runtime;

/// Variables the CLI itself needs on each platform; everything else is cleared.
const SYSTEM: &[&str] = &[
    "HOME",
    "PATH",
    "TMPDIR",
    "LANG",
    "SystemRoot",
    "TEMP",
    "TMP",
    "USERPROFILE",
    "APPDATA",
    "LOCALAPPDATA",
    "PATHEXT",
    "ComSpec",
];

struct Fixture {
    root: tempfile::TempDir,
    executable: PathBuf,
}
impl Fixture {
    fn new(mode: &str) -> Self {
        let root = tempfile::Builder::new()
            .prefix("auth fixture ")
            .tempdir()
            .unwrap();
        let fixture = Self {
            root,
            executable: fake_runtime::program(),
        };
        fixture.state(mode, 0);
        std::fs::write(fixture.root.path().join("config.json"), b"{}\n").unwrap();
        fixture
    }
    /// One scenario for both runtimes; each finds it through its own home.
    fn state(&self, mode: &str, exit: i32) {
        fake_runtime::install(
            self.root.path(),
            &json!({
                "mode":mode,"home":std::env::var("HOME").ok(),"login_exit":exit,
                "cache_home":self.root.path().join("private cache")
            }),
        );
    }
    fn command(&self) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_markitai"));
        command.env_clear().current_dir(self.root.path());
        for key in SYSTEM {
            if let Some(value) = std::env::var_os(key) {
                command.env(key, value);
            }
        }
        command
            .env("MARKITAI_HOME", self.root.path().join("state"))
            .env("CLAUDE_CLI_PATH", self.root.path().join("missing claude"))
            .env("CODEX_CLI_PATH", self.root.path().join("missing codex"))
            .env("COPILOT_CLI_PATH", &self.executable)
            .env("COPILOT_HOME", self.root.path())
            .env("COPILOT_CACHE_HOME", self.root.path().join("private cache"))
            .env("COPILOT_GITHUB_TOKEN", "fixture-only")
            .env("OPENAI_API_KEY", "must-be-scrubbed")
            .env("COPILOT_PROVIDER_API_KEY", "must-be-scrubbed")
            .args(["-c", "config.json"]);
        command
    }
    fn status(&self, json: bool) -> std::process::Output {
        let mut command = self.command();
        command.args(["auth", "copilot", "status"]);
        if json {
            command.arg("--json");
        }
        command.output().unwrap()
    }
}

/// The spelling of a resolved runtime path that status reports: canonical,
/// without the Windows `\\?\` prefix.
fn reported(path: &Path) -> String {
    let canonical = path.canonicalize().unwrap();
    let text = canonical.to_str().unwrap();
    match text.strip_prefix(r"\\?\") {
        Some(rest) if cfg!(windows) && rest.as_bytes().get(1) == Some(&b':') => rest.into(),
        _ => text.into(),
    }
}

/// The login runs in the CLI's own process on Unix, which replaces itself
/// with the runtime, and in a child that it waits for on Windows.
fn assert_login_process(recorded: &Value, cli: u32) {
    if cfg!(unix) {
        assert_eq!(recorded["pid"], cli);
    } else {
        assert!(
            recorded["pid"]
                .as_u64()
                .is_some_and(|pid| pid != u64::from(cli))
        );
    }
}

#[test]
fn observed_status_preserves_public_shape_and_json_false_exit() {
    let fixture = Fixture::new("normal");
    let result = fixture.status(true);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let value: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(value["provider"], "copilot");
    assert_eq!(value["authenticated"], true);
    assert_eq!(value["user"], "fixture-user");
    assert!(value["expires_at"].is_null());
    assert!(value["error"].is_null());
    assert_eq!(value["details"]["protocol_version"], 3);
    fixture.state("unauth", 0);
    let result = fixture.status(true);
    assert!(result.status.success());
    assert_eq!(
        serde_json::from_slice::<Value>(&result.stdout).unwrap()["authenticated"],
        false
    );
    assert_eq!(fixture.status(false).status.code(), Some(1));
    let requests = std::fs::read_to_string(fixture.root.path().join("requests.jsonl")).unwrap();
    assert!(!requests.contains("session.send"));
    assert!(!fixture.root.path().join("login.json").exists());
}

#[test]
fn unavailable_runtime_and_other_adapters_remain_explicit() {
    let fixture = Fixture::new("normal");
    let result = fixture
        .command()
        .env("COPILOT_CLI_PATH", fixture.root.path().join("missing"))
        .args(["auth", "copilot", "status", "--json"])
        .output()
        .unwrap();
    assert!(result.status.success());
    let value: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(value["authenticated"], false);
    assert!(value["error"].is_string());
    for provider in ["claude", "chatgpt"] {
        let result = fixture
            .command()
            .args(["auth", provider, "status", "--json"])
            .output()
            .unwrap();
        assert!(result.status.success());
        let value: Value = serde_json::from_slice(&result.stdout).unwrap();
        assert_eq!(value["details"]["verification"], "unavailable");
        if provider == "claude" {
            assert_eq!(value["sdk_installed"], false);
            assert!(value["cli_path"].is_null());
        }
    }
    let result = fixture.command().arg("auth").output().unwrap();
    assert!(result.status.success());
    let text = String::from_utf8(result.stdout).unwrap();
    assert!(text.find("claude-agent:").unwrap() < text.find("chatgpt:").unwrap());
    assert!(text.find("chatgpt:").unwrap() < text.find("copilot:").unwrap());
}

#[test]
fn explicit_login_hands_over_and_preserves_exit_home_and_configuration() {
    let fixture = Fixture::new("normal");
    for code in [0, 7] {
        fixture.state("normal", code);
        let child = fixture
            .command()
            .args(["auth", "copilot", "login"])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let pid = child.id();
        let result = child.wait_with_output().unwrap();
        assert_eq!(
            result.status.code(),
            Some(code),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let login: Value =
            serde_json::from_slice(&std::fs::read(fixture.root.path().join("login.json")).unwrap())
                .unwrap();
        assert_login_process(&login, pid);
        assert_eq!(login["args"], json!(["login"]));
        assert_eq!(
            login["cache_home"],
            json!(fixture.root.path().join("private cache"))
        );
        assert_eq!(
            std::fs::read(fixture.root.path().join("config.json")).unwrap(),
            b"{}\n"
        );
        assert!(result.stdout.is_empty());
    }
}

#[test]
fn claude_status_and_login_delegate_to_official_runtime_with_private_auth_home() {
    let fixture = Fixture::new("normal");
    let executable = fake_runtime::program();
    let command = || {
        let mut command = fixture.command();
        command
            .env("CLAUDE_CLI_PATH", &executable)
            .env("CLAUDE_CONFIG_DIR", fixture.root.path())
            .env("ANTHROPIC_API_KEY", "must-be-scrubbed")
            .env("CLAUDE_CODE_OAUTH_TOKEN", "must-be-scrubbed");
        command
    };
    for (mode, authenticated) in [("normal", true), ("signed-out", false), ("byok", false)] {
        fixture.state(mode, 0);
        let output = command()
            .args(["auth", "claude", "status", "--json"])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let value: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(value["authenticated"], authenticated);
        assert_eq!(value["provider"], "claude-agent");
        assert_eq!(value["sdk_installed"], false);
        assert_eq!(value["details"]["native_adapter"], true);
        assert_eq!(value["cli_path"], reported(&executable));
        assert!(!String::from_utf8_lossy(&output.stdout).contains("fixture-secret"));
        assert_eq!(
            command()
                .args(["auth", "claude", "status"])
                .status()
                .unwrap()
                .code(),
            Some(if authenticated { 0 } else { 1 })
        );
    }
    // A signed-out runtime is one command away: the text status says which.
    fixture.state("signed-out", 0);
    let text = command()
        .args(["auth", "claude", "status"])
        .output()
        .unwrap();
    assert_eq!(text.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&text.stdout).contains("  Next: markitai auth claude login\n"),
        "{}",
        String::from_utf8_lossy(&text.stdout)
    );
    fixture.state("normal", 0);
    let signed_in = command()
        .args(["auth", "claude", "status"])
        .output()
        .unwrap();
    assert!(!String::from_utf8_lossy(&signed_in.stdout).contains("Next:"));
    let before = std::fs::read(fixture.root.path().join("config.json")).unwrap();
    for exit in [0, 7] {
        fixture.state("normal", exit);
        let child = command()
            .args(["auth", "claude", "login"])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let pid = child.id();
        let output = child.wait_with_output().unwrap();
        assert_eq!(output.status.code(), Some(exit));
        let record: Value = serde_json::from_slice(
            &std::fs::read(fixture.root.path().join("claude-login.json")).unwrap(),
        )
        .unwrap();
        assert_login_process(&record, pid);
        assert_eq!(record["home"], json!(std::env::var("HOME").ok()));
        assert_eq!(
            std::fs::read(fixture.root.path().join("config.json")).unwrap(),
            before
        );
    }
    let calls = std::fs::read_to_string(fixture.root.path().join("calls.jsonl")).unwrap();
    assert!(!calls.contains("--print"));
}
