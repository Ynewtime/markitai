#![cfg(unix)]
#[path = "support/console.rs"]
mod console;

use serde_json::{Value, json};
use std::os::unix::fs::PermissionsExt;
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

struct Fixture {
    root: tempfile::TempDir,
    executable: std::path::PathBuf,
}
impl Fixture {
    fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        let executable = root.path().join("codex fixture");
        std::fs::write(&executable, include_bytes!("chatgpt_auth_fixture.py")).unwrap();
        std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700)).unwrap();
        std::fs::create_dir(root.path().join("codex-state")).unwrap();
        std::fs::write(
            root.path().join("codex-state/config.toml"),
            "# authored configuration must remain unchanged\n",
        )
        .unwrap();
        std::fs::write(root.path().join("config.json"), b"{}\n").unwrap();
        let f = Self { root, executable };
        f.state("normal", 0);
        f
    }
    fn state(&self, mode: &str, code: i32) {
        std::fs::write(self.root.path().join("state.json"),json!({"mode":mode,"login_exit":code,"home":std::env::var("HOME").ok(),"codex_home":self.root.path().join("codex-state")}).to_string()).unwrap();
    }
    fn command(&self) -> Command {
        let mut c = Command::new(env!("CARGO_BIN_EXE_markitai"));
        console::english(c.env_clear()).current_dir(self.root.path());
        for key in ["HOME", "PATH", "LANG", "LC_ALL", "TMPDIR"] {
            if let Some(value) = std::env::var_os(key) {
                c.env(key, value);
            }
        }
        c.env("MARKITAI_HOME", self.root.path().join("state"))
            .env("CODEX_CLI_PATH", &self.executable)
            .env("CODEX_HOME", self.root.path().join("codex-state"))
            .env(
                "MARKITAI_BROWSER_EXECUTABLE",
                self.root.path().join("no-browser"),
            )
            .env("OPENAI_API_KEY", "fixture-must-not-forward")
            .env("OPENAI_BASE_URL", "https://must-not-forward.invalid")
            .env("CODEX_API_KEY", "fixture-must-not-forward")
            .env("HTTPS_PROXY", "https://must-not-forward.invalid")
            .args(["-c", "config.json"]);
        c
    }
    fn run(&self, args: &[&str]) -> (u32, Output) {
        let out = self.root.path().join("stdout");
        let err = self.root.path().join("stderr");
        let mut c = self.command();
        c.args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::from(std::fs::File::create(&out).unwrap()))
            .stderr(Stdio::from(std::fs::File::create(&err).unwrap()));
        let mut child = c.spawn().unwrap();
        let pid = child.id();
        let deadline = Instant::now() + Duration::from_secs(40);
        let status = loop {
            if let Some(s) = child.try_wait().unwrap() {
                break s;
            }
            if Instant::now() >= deadline {
                let _ = child.kill();
                let _ = child.wait();
                panic!("CLI auth fixture exceeded deadline");
            }
            std::thread::sleep(Duration::from_millis(10));
        };
        (
            pid,
            Output {
                status,
                stdout: std::fs::read(out).unwrap(),
                stderr: std::fs::read(err).unwrap(),
            },
        )
    }
    fn calls(&self) -> Vec<Value> {
        std::fs::read_to_string(self.root.path().join("calls.jsonl"))
            .unwrap_or_default()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }
}
#[test]
fn codex_status_preserves_json_exit_contract_and_rejects_api_key_or_wrong_protocol() {
    let f = Fixture::new();
    let original = std::fs::read(f.root.path().join("codex-state/config.toml")).unwrap();
    for (mode, authenticated) in [
        ("normal", true),
        ("signed-out", false),
        ("api-key", false),
        ("malformed", false),
        ("wrong-version", false),
    ] {
        f.state(mode, 0);
        let (_, o) = f.run(&["auth", "chatgpt", "status", "--json"]);
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
        let v: Value = serde_json::from_slice(&o.stdout).unwrap();
        assert_eq!(v["provider"], "chatgpt");
        assert_eq!(v["authenticated"], authenticated);
        assert!(v["user"].is_null());
        assert!(v["expires_at"].is_null());
        assert_eq!(v["details"]["source"], "official_cli");
        assert!(!String::from_utf8_lossy(&o.stdout).contains("sk-fixture"));
        assert!(!String::from_utf8_lossy(&o.stderr).contains("sk-fixture"));
        assert_eq!(
            f.run(&["auth", "chatgpt", "status"]).1.status.code(),
            Some(if authenticated { 0 } else { 1 })
        );
    }
    assert!(
        f.calls()
            .iter()
            .all(|v| v["args"] == json!(["--version"]) || v["args"] == json!(["login", "status"]))
    );
    assert_eq!(
        std::fs::read(f.root.path().join("codex-state/config.toml")).unwrap(),
        original
    );
}
#[test]
fn explicit_codex_login_delegates_same_pid_and_exact_exit_without_post_login_inference() {
    let f = Fixture::new();
    for code in [0, 7] {
        f.state("normal", code);
        let (pid, o) = f.run(&["auth", "chatgpt", "login"]);
        assert_eq!(o.status.code(), Some(code));
        assert!(o.stdout.is_empty());
        assert!(o.stderr.is_empty());
        let calls = f.calls();
        let last = calls.last().unwrap();
        assert_eq!(last["args"], json!(["login"]));
        assert_eq!(last["pid"], pid);
        assert_eq!(last["home"], std::env::var("HOME").unwrap());
    }
    assert_eq!(f.calls().len(), 2);
    assert_eq!(
        std::fs::read(f.root.path().join("config.json")).unwrap(),
        b"{}\n"
    );
}
#[test]
fn doctor_checks_chatgpt_status_without_claiming_inference_or_model_entitlement() {
    let f = Fixture::new();
    std::fs::write(f.root.path().join("config.json"),json!({"llm":{"model_list":[{"model_name":"default","litellm_params":{"model":"chatgpt/gpt-5.5"}}]}}).to_string()).unwrap();
    for (mode, auth_status, exit) in [("normal", "ok", 0), ("api-key", "error", 1)] {
        f.state(mode, 0);
        let (_, o) = f.run(&["doctor", "--json"]);
        assert_eq!(
            o.status.code(),
            Some(exit),
            "{}",
            String::from_utf8_lossy(&o.stderr)
        );
        let v: Value = serde_json::from_slice(&o.stdout).unwrap();
        assert_eq!(v["chatgpt-runtime"]["status"], "ok");
        assert_eq!(v["chatgpt-auth"]["status"], auth_status);
        assert!(!v.to_string().contains("sk-fixture"));
        if mode == "normal" {
            assert!(
                v["chatgpt-auth"]["message"]
                    .as_str()
                    .unwrap()
                    .contains("not probed")
            );
        }
    }
    assert!(
        f.calls()
            .iter()
            .all(|v| v["args"] == json!(["--version"]) || v["args"] == json!(["login", "status"]))
    );
}
