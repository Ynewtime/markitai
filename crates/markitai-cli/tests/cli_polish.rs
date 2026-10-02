use serde_json::{Value, json};
use std::path::Path;
use std::process::{Command, Output};

fn command(root: &Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_markitai"));
    command.env_clear();
    for key in ["SYSTEMROOT", "WINDIR", "TEMP", "TMP", "ComSpec"] {
        if let Some(value) = std::env::var_os(key) {
            command.env(key, value);
        }
    }
    let bin = root.join("bin");
    std::fs::create_dir_all(&bin).unwrap();
    command
        .current_dir(root)
        .env("PATH", bin)
        .env("PATHEXT", ".EXE;.CMD;.BAT;.COM")
        .env("HOME", root.join("isolated-home"))
        .env("USERPROFILE", root.join("isolated-home"))
        .env("MARKITAI_HOME", root.join("state"))
        .env("MARKITAI_LANG", "en");
    command
}

fn output(root: &Path, args: &[&str]) -> Output {
    command(root).args(args).output().unwrap()
}

#[test]
fn configuration_warnings_are_advisory_explicit_and_separate_from_errors() {
    let root = tempfile::tempdir().unwrap();
    let defaults = output(root.path(), &["config", "validate"]);
    assert!(defaults.status.success());
    assert_eq!(defaults.stdout, b"Configuration is valid\n");
    assert!(defaults.stderr.is_empty());
    std::fs::write(
        root.path().join("explicit.json"),
        serde_json::to_vec(&json!({
            "batch":{"heavy_task_limit":2},"office":{"macos_fallback":false},
            "image":{"stdout_fetch_external":true}
        }))
        .unwrap(),
    )
    .unwrap();
    let explicit = output(root.path(), &["config", "validate", "explicit.json"]);
    assert!(explicit.status.success());
    assert_eq!(explicit.stdout, defaults.stdout);
    let warnings = String::from_utf8(explicit.stderr).unwrap();
    assert_eq!(warnings.lines().count(), 3);
    for key in [
        "batch.heavy_task_limit",
        "office.macos_fallback",
        "image.stdout_fetch_external",
    ] {
        assert!(
            warnings.contains(&format!("Warning: {key} has no effect")),
            "{warnings}"
        );
    }
    let chinese = command(root.path())
        .env("MARKITAI_LANG", "zh")
        .args(["config", "validate", "explicit.json"])
        .output()
        .unwrap();
    assert!(chinese.status.success());
    assert_eq!(chinese.stdout, "配置有效\n".as_bytes());
    assert!(
        String::from_utf8_lossy(&chinese.stderr)
            .contains("Warning: batch.heavy_task_limit 没有运行效果")
    );
    std::fs::write(
        root.path().join("invalid.json"),
        br#"{"batch":{"heavy_task_limit":-1}}"#,
    )
    .unwrap();
    let invalid = output(root.path(), &["config", "validate", "invalid.json"]);
    assert_eq!(invalid.status.code(), Some(1));
    assert!(invalid.stdout.is_empty());
    assert!(String::from_utf8_lossy(&invalid.stderr).starts_with("Error:"));
    assert!(!String::from_utf8_lossy(&invalid.stderr).contains("Warning:"));
}

#[test]
fn init_detects_runtime_commands_without_starting_them_or_adding_guessed_models() {
    let root = tempfile::tempdir().unwrap();
    let bin = root.path().join("bin");
    std::fs::create_dir(&bin).unwrap();
    for name in ["copilot", "claude", "codex"] {
        let path = bin.join(if cfg!(windows) {
            format!("{name}.cmd")
        } else {
            name.to_owned()
        });
        let script = if cfg!(windows) {
            "@echo off\r\necho forbidden>runtime-was-started\r\nexit /b 99\r\n"
        } else {
            "#!/bin/sh\nprintf forbidden > runtime-was-started\nexit 99\n"
        };
        std::fs::write(&path, script).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700)).unwrap();
        }
    }
    let init = output(root.path(), &["init", "--yes", "--local"]);
    assert!(
        init.status.success(),
        "{}",
        String::from_utf8_lossy(&init.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&init.stdout).lines().count(), 1);
    let hints = String::from_utf8(init.stderr).unwrap();
    for provider in ["copilot", "claude", "chatgpt"] {
        assert!(
            hints.contains(&format!("markitai auth {provider} status")),
            "{hints}"
        );
    }
    assert!(hints.contains("Login and version were not checked"));
    assert!(!root.path().join("runtime-was-started").exists());
    let saved: Value =
        serde_json::from_slice(&std::fs::read(root.path().join("markitai.json")).unwrap()).unwrap();
    assert_eq!(saved["llm"]["enabled"], false);
    assert!(saved["llm"].get("model_list").is_none());
}

#[test]
fn conversion_json_orders_success_failure_and_empty_envelopes() {
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("note.txt"), "Ordered JSON\n").unwrap();
    std::fs::create_dir(root.path().join("empty")).unwrap();
    for (input, success) in [("note.txt", true), ("missing.txt", false), ("empty", true)] {
        let result = output(root.path(), &[input, "-o", "out", "--json"]);
        assert_eq!(result.status.success(), success);
        let text = String::from_utf8(result.stdout).unwrap();
        assert!(
            text.starts_with("{\n  \"version\": \"1.0\",\n  \"ok\":"),
            "{text}"
        );
        let mut previous = 0;
        for key in ["version", "ok", "error", "batch", "items", "totals"] {
            let position = text.find(&format!("\n  \"{key}\":")).unwrap();
            assert!(position >= previous);
            previous = position;
        }
        let parsed: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(parsed["ok"], success);
        if input != "empty" {
            assert!(
                text.contains("\"kind\": \"file\",\n      \"source\":"),
                "{text}"
            );
        }
    }
}
