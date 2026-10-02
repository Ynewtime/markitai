#![cfg(unix)]
use serde_json::{Value, json};
use std::os::unix::fs::PermissionsExt;
use std::process::{Command, Stdio};

struct Fixture {
    root: tempfile::TempDir,
    executable: std::path::PathBuf,
}
impl Fixture {
    fn new(mode: &str) -> Self {
        let root = tempfile::Builder::new()
            .prefix("auth fixture ")
            .tempdir()
            .unwrap();
        let executable = root.path().join("official fixture");
        let text = include_str!("../../markitai-core/src/subscription/fake_cli.py");
        let text = text.replace(
            "root = pathlib.Path(os.environ['COPILOT_HOME'])",
            r#"root = pathlib.Path(os.environ['COPILOT_HOME'])
if sys.argv[1:] == ['login']:
    assert 'OPENAI_API_KEY' not in os.environ
    assert 'COPILOT_PROVIDER_API_KEY' not in os.environ
    state = json.loads((root / 'fixture.json').read_text())
    assert os.environ.get('HOME') == state['home']
    assert os.environ.get('COPILOT_CACHE_HOME') == state['cache_home']
    (root / 'login.json').write_text(json.dumps({'pid':os.getpid(),'args':sys.argv[1:],'cache_home':os.environ.get('COPILOT_CACHE_HOME')}))
    sys.exit(state.get('login_exit', 0))"#,
        );
        std::fs::write(&executable, text).unwrap();
        std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700)).unwrap();
        let fixture = Self { root, executable };
        fixture.state(mode, 0);
        std::fs::write(fixture.root.path().join("config.json"), b"{}\n").unwrap();
        fixture
    }
    fn state(&self, mode: &str, exit: i32) {
        std::fs::write(
            self.root.path().join("fixture.json"),
            serde_json::to_vec(&json!({
                "mode":mode,"home":std::env::var("HOME").ok(),"login_exit":exit,"cache_home":self.root.path().join("private cache")
            }))
            .unwrap(),
        )
        .unwrap();
    }
    fn command(&self) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_markitai"));
        command.env_clear().current_dir(self.root.path());
        for key in ["HOME", "PATH", "TMPDIR", "LANG"] {
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
fn explicit_login_replaces_process_and_preserves_exit_home_and_configuration() {
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
        assert_eq!(result.status.code(), Some(code));
        let login: Value =
            serde_json::from_slice(&std::fs::read(fixture.root.path().join("login.json")).unwrap())
                .unwrap();
        assert_eq!(login["pid"], pid);
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
    let executable = fixture.root.path().join("claude fixture");
    std::fs::write(&executable, include_str!("claude_auth_fixture.py")).unwrap();
    std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700)).unwrap();
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
        assert_eq!(
            value["cli_path"],
            executable.canonicalize().unwrap().to_str().unwrap()
        );
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
        assert_eq!(record["pid"], pid);
        assert_eq!(record["home"], std::env::var("HOME").unwrap());
        assert_eq!(
            std::fs::read(fixture.root.path().join("config.json")).unwrap(),
            before
        );
    }
    let calls = std::fs::read_to_string(fixture.root.path().join("claude-calls.jsonl")).unwrap();
    assert!(!calls.contains("--print"));
}
