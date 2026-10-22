use serde_json::{Value, json};
use std::ffi::OsStr;
use std::path::Path;
use std::process::{Command, Output};

#[path = "support/console.rs"]
mod console;

fn command(root: &Path) -> Command {
    let mut command = isolated(env!("CARGO_BIN_EXE_markitai"), root);
    console::detached(&mut command);
    command
}

/// `program` in an environment that names only `root`, with English asked for.
fn isolated(program: impl AsRef<OsStr>, root: &Path) -> Command {
    let mut command = Command::new(program);
    console::english(command.env_clear());
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
    assert_eq!(defaults.stdout, "✓ Configuration is valid\n".as_bytes());
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
            warnings.contains(&format!("  ! {key} has no effect")),
            "{warnings}"
        );
    }
    let chinese = command(root.path())
        .env("MARKITAI_LANG", "zh")
        .args(["config", "validate", "explicit.json"])
        .output()
        .unwrap();
    assert!(chinese.status.success());
    assert_eq!(chinese.stdout, "✓ 配置有效\n".as_bytes());
    assert!(
        String::from_utf8_lossy(&chinese.stderr)
            .contains("  ! batch.heavy_task_limit 没有运行效果")
    );
    std::fs::write(
        root.path().join("invalid.json"),
        br#"{"batch":{"heavy_task_limit":-1}}"#,
    )
    .unwrap();
    let invalid = output(root.path(), &["config", "validate", "invalid.json"]);
    assert_eq!(invalid.status.code(), Some(1));
    assert!(invalid.stdout.is_empty());
    assert!(String::from_utf8_lossy(&invalid.stderr).starts_with("✗ "));
    assert!(!String::from_utf8_lossy(&invalid.stderr).contains("  ! "));
}

#[test]
fn a_dumb_terminal_gets_the_ascii_marks_in_a_pipe_too() {
    let root = tempfile::tempdir().unwrap();
    let dumb = command(root.path())
        .env("TERM", "dumb")
        .args(["config", "validate"])
        .output()
        .unwrap();
    assert!(dumb.status.success());
    assert_eq!(dumb.stdout, b"+ Configuration is valid\n");
}

/// Windows marks in a pipe or a file follow the output code page of the
/// console markitai is attached to, the page Windows PowerShell decodes a
/// program's output with: the ASCII set at 437, the Unicode marks at 65001
/// and without a console.
#[cfg(windows)]
#[test]
fn windows_redirected_marks_follow_the_console_code_page() {
    use std::os::windows::process::CommandExt;
    use std::process::Stdio;
    use windows_sys::Win32::System::Console::GetConsoleOutputCP;
    use windows_sys::Win32::System::Threading::CREATE_NO_WINDOW;
    let root = tempfile::tempdir().unwrap();
    let system = Path::new(&std::env::var_os("SYSTEMROOT").expect("SYSTEMROOT")).join("System32");
    let runner = unsafe { GetConsoleOutputCP() };
    for (page, expected) in [
        (437, "+ Configuration is valid\n"),
        (65001, "✓ Configuration is valid\n"),
    ] {
        let result = root.path().join(format!("validate-{page}.txt"));
        let active = root.path().join(format!("page-{page}.txt"));
        // cmd runs in a hidden console of its own, so the page it sets stays
        // there; markitai, started by cmd, shares that console and writes to
        // a file. The second chcp records the page that was in effect.
        let shell = isolated(system.join("cmd.exe"), root.path())
            .creation_flags(CREATE_NO_WINDOW)
            .raw_arg(format!(
                r#"/d /s /c ""{chcp}" {page} >nul && "{markitai}" config validate >"{result}" 2>&1 & "{chcp}" >"{active}""#,
                chcp = system.join("chcp.com").display(),
                markitai = env!("CARGO_BIN_EXE_markitai"),
                result = result.display(),
                active = active.display(),
            ))
            .stdin(Stdio::null())
            .output()
            .unwrap();
        let written = std::fs::read(&result).unwrap_or_default();
        assert_eq!(
            String::from_utf8_lossy(&written),
            expected,
            "code page {page}: {} {}",
            String::from_utf8_lossy(&std::fs::read(&active).unwrap_or_default()),
            String::from_utf8_lossy(&shell.stderr)
        );
    }
    assert_eq!(
        unsafe { GetConsoleOutputCP() },
        runner,
        "the code page of the test runner's own console changed"
    );
    // Without a console there is no page to follow.
    let detached = output(root.path(), &["config", "validate"]);
    assert_eq!(detached.stdout, "✓ Configuration is valid\n".as_bytes());
}

#[test]
fn init_names_installed_runtimes_that_are_not_signed_in_and_adds_no_model_for_them() {
    let root = tempfile::tempdir().unwrap();
    let bin = root.path().join("bin");
    std::fs::create_dir(&bin).unwrap();
    for name in ["copilot", "claude", "codex"] {
        let path = bin.join(if cfg!(windows) {
            format!("{name}.cmd")
        } else {
            name.to_owned()
        });
        // A runtime that answers nothing: asked for its version or sign-in,
        // it fails, so it counts as not signed in.
        let script = if cfg!(windows) {
            "@echo off\r\nexit /b 99\r\n"
        } else {
            "#!/bin/sh\nexit 99\n"
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
            hints.contains(&format!("markitai auth {provider} login")),
            "{hints}"
        );
    }
    assert!(
        hints.contains("not signed in with a subscription"),
        "{hints}"
    );
    let saved: Value =
        serde_json::from_slice(&std::fs::read(root.path().join("markitai.json")).unwrap()).unwrap();
    assert_eq!(saved["llm"]["enabled"], false);
    assert!(saved["llm"].get("model_list").is_none());
}

#[test]
fn a_single_file_is_named_by_its_file_name_and_an_unsupported_one_rejects_the_run() {
    let root = tempfile::tempdir().unwrap();
    std::fs::create_dir(root.path().join("in")).unwrap();
    std::fs::write(root.path().join("in/note.txt"), "Named JSON\n").unwrap();
    std::fs::write(root.path().join("in/thing.xyz"), "?").unwrap();
    let result = output(root.path(), &["in/note.txt", "-o", "out", "--json"]);
    assert!(result.status.success());
    let parsed: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(parsed["items"][0]["source"], "note.txt");
    assert_eq!(parsed["items"][0]["earlier_run"], false);
    assert_eq!(parsed["totals"]["this_run_cost_usd"], 0.0);
    let result = output(root.path(), &["in/thing.xyz", "-o", "out", "--json"]);
    assert_eq!(result.status.code(), Some(1));
    let parsed: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(parsed["items"], serde_json::json!([]));
    assert!(
        parsed["error"]
            .as_str()
            .unwrap()
            .starts_with("Unsupported file format: '.xyz'")
    );
}

#[cfg(unix)]
#[test]
fn a_folder_batch_names_the_symbolic_links_it_does_not_follow() {
    use std::os::unix::fs::symlink;
    let root = tempfile::tempdir().unwrap();
    let elsewhere = root.path().join("elsewhere");
    std::fs::create_dir_all(elsewhere.join("folder")).unwrap();
    std::fs::write(elsewhere.join("target.txt"), "Linked text\n").unwrap();
    std::fs::write(elsewhere.join("folder/inner.txt"), "Linked folder\n").unwrap();
    std::fs::write(elsewhere.join("data.bin"), [0u8; 4]).unwrap();
    let input = root.path().join("in");
    std::fs::create_dir(&input).unwrap();
    std::fs::write(input.join("own.txt"), "Own text\n").unwrap();
    symlink(elsewhere.join("target.txt"), input.join("link.txt")).unwrap();
    symlink(elsewhere.join("folder"), input.join("linked-folder")).unwrap();
    // Neither a document by its name nor a folder: not named.
    symlink(elsewhere.join("data.bin"), input.join("data.bin")).unwrap();
    symlink(elsewhere.join("absent.txt"), input.join("broken.txt")).unwrap();
    let result = output(root.path(), &["in", "-o", "out", "--json"]);
    assert!(result.status.success());
    let parsed: Value = serde_json::from_slice(&result.stdout).unwrap();
    let items = parsed["items"].as_array().unwrap();
    let skipped: Vec<_> = items
        .iter()
        .filter(|item| item["skip_reason"] == "symlink")
        .map(|item| {
            (
                item["source"].as_str().unwrap(),
                item["status"].as_str().unwrap(),
            )
        })
        .collect();
    assert_eq!(
        skipped,
        [("link.txt", "skipped"), ("linked-folder", "skipped")]
    );
    assert_eq!(parsed["totals"]["completed"], 1);
    assert_eq!(parsed["totals"]["skipped"], 2);
    assert!(parsed["ok"].as_bool().unwrap());
    // Not followed: nothing was written for them.
    assert!(!root.path().join("out/link.txt.md").exists());
    assert!(!root.path().join("out/linked-folder").exists());
    let result = output(root.path(), &["in", "-o", "out2"]);
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.contains("Done: 1 of 3 files converted, 2 skipped"),
        "{stderr}"
    );
    assert!(
        stderr.contains("Skipped 2 items (symbolic links, not followed): link.txt, linked-folder. Set output.allow_symlinks to true to convert what they point to."),
        "{stderr}"
    );
    // Only links: named, not "nothing found".
    let only = root.path().join("only");
    std::fs::create_dir(&only).unwrap();
    symlink(elsewhere.join("target.txt"), only.join("link.txt")).unwrap();
    let result = output(root.path(), &["only", "-o", "out3", "--json"]);
    let parsed: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(parsed["items"][0]["skip_reason"], "symlink");
    let result = output(root.path(), &["only", "-o", "out3"]);
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.contains("Skipped 1 item (symbolic links, not followed): link.txt."),
        "{stderr}"
    );
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
        if input == "note.txt" {
            assert!(
                text.contains("\"kind\": \"file\",\n      \"source\":"),
                "{text}"
            );
        }
    }
}
