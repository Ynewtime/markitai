//! Terminating signals must not leave external runtime process groups alive.
//!
//! The authored Codex protocol fixture starts a grandchild in its own process
//! group and then hangs. A terminal interrupt reaches only the CLI's group, so
//! these tests signal the CLI process alone, as a terminal-to-foreground
//! delivery would reach it and not the runtime.
#![cfg(unix)]
use serde_json::json;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

struct Fixture {
    root: tempfile::TempDir,
}

impl Fixture {
    fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        let runtime = root.path().join("runtime");
        std::fs::create_dir(&runtime).unwrap();
        let executable = runtime.join("codex");
        std::fs::write(
            &executable,
            include_bytes!("../../markitai-core/src/subscription/chatgpt/fake_exec.py"),
        )
        .unwrap();
        std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700)).unwrap();
        std::fs::write(
            runtime.join("scenario.json"),
            json!({"name":"sleep"}).to_string(),
        )
        .unwrap();
        std::fs::create_dir(root.path().join("inputs")).unwrap();
        std::fs::write(root.path().join("inputs/source.md"), "# Title\n\nBody.\n").unwrap();
        std::fs::write(
            root.path().join("config.json"),
            json!({"log":{"dir":null},"history":{"record":false},"cache":{"enabled":false},
                "image":{"alt_enabled":false,"desc_enabled":false},
                "llm":{"enabled":true,"on_failure":"fail","router_settings":{"num_retries":0,"timeout":120},
                    "model_list":[{"model_name":"default","litellm_params":{"model":"chatgpt/gpt-5.5"}}]}})
            .to_string(),
        )
        .unwrap();
        Self { root }
    }

    fn path(&self, name: &str) -> PathBuf {
        self.root.path().join(name)
    }

    fn spawn(&self, args: &[&str], stdin: Stdio) -> Child {
        let mut command = Command::new(env!("CARGO_BIN_EXE_markitai"));
        command.env_clear().current_dir(self.root.path());
        for key in ["HOME", "PATH", "LANG", "TMPDIR"] {
            if let Some(value) = std::env::var_os(key) {
                command.env(key, value);
            }
        }
        command
            .env("MARKITAI_HOME", self.path("home"))
            .env("CODEX_CLI_PATH", self.path("runtime/codex"))
            .args(["-c", "config.json"])
            .args(args)
            .stdin(stdin)
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap()
    }

    /// The runtime's grandchild, once the hanging turn has started.
    fn grandchild(&self) -> i32 {
        let path = self.path("runtime/grandchild.pid");
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            if let Ok(text) = std::fs::read_to_string(&path)
                && let Ok(pid) = text.trim().parse()
            {
                return pid;
            }
            assert!(Instant::now() < deadline, "runtime turn never started");
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

fn signal(child: &Child, signal: i32) {
    assert_eq!(unsafe { libc::kill(child.id() as i32, signal) }, 0);
}

fn wait(child: &mut Child) -> ExitStatus {
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            return status;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            panic!("CLI did not terminate after the signal");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// The grandchild is reparented after its group is killed; wait for removal.
fn assert_gone(pid: i32) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while unsafe { libc::kill(pid, 0) } == 0 {
        if Instant::now() >= deadline {
            unsafe {
                libc::kill(pid, libc::SIGKILL);
            }
            panic!("external runtime process outlived the CLI");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn single_conversion_interrupt_kills_runtime_group_and_keeps_signal_exit() {
    let f = Fixture::new();
    let mut child = f.spawn(&["inputs/source.md", "-o", "out"], Stdio::null());
    let grandchild = f.grandchild();
    signal(&child, libc::SIGINT);
    let status = wait(&mut child);
    assert_eq!(status.signal(), Some(libc::SIGINT), "{status:?}");
    assert_gone(grandchild);
}

#[test]
fn single_conversion_hangup_and_terminate_also_clean_up() {
    for sent in [libc::SIGHUP, libc::SIGTERM] {
        let f = Fixture::new();
        let mut child = f.spawn(&["inputs/source.md", "-o", "out"], Stdio::null());
        let grandchild = f.grandchild();
        signal(&child, sent);
        let status = wait(&mut child);
        assert_eq!(status.signal(), Some(sent), "{status:?}");
        assert_gone(grandchild);
    }
}

#[test]
fn batch_first_interrupt_waits_and_second_forces_exit_without_orphans() {
    let f = Fixture::new();
    let mut child = f.spawn(&["inputs", "-o", "out"], Stdio::null());
    let grandchild = f.grandchild();
    signal(&child, libc::SIGINT);
    // The first interrupt drains: active work keeps running.
    std::thread::sleep(Duration::from_millis(500));
    assert!(child.try_wait().unwrap().is_none());
    assert_eq!(unsafe { libc::kill(grandchild, 0) }, 0);
    signal(&child, libc::SIGINT);
    let status = wait(&mut child);
    assert_eq!(status.code(), Some(130), "{status:?}");
    assert_gone(grandchild);
}

#[test]
fn mcp_terminate_kills_runtime_group_started_by_a_tool_call() {
    let f = Fixture::new();
    let mut child = f.spawn(&["mcp"], Stdio::piped());
    let mut input = child.stdin.take().unwrap();
    let mut output = BufReader::new(child.stdout.take().unwrap());
    let mut send = |value: serde_json::Value| {
        writeln!(input, "{value}").unwrap();
        input.flush().unwrap();
    };
    send(
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"interrupt-test","version":"1"}}}),
    );
    let mut line = String::new();
    output.read_line(&mut line).unwrap();
    assert!(line.contains("\"id\":1"), "{line}");
    send(json!({"jsonrpc":"2.0","method":"notifications/initialized"}));
    let source = f.path("inputs/source.md");
    let out = f.path("mcp-out");
    send(
        json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"convert_document",
        "arguments":{"path":Path::new(&source),"output_dir":Path::new(&out),"llm":true}}}),
    );
    let grandchild = f.grandchild();
    signal(&child, libc::SIGTERM);
    let status = wait(&mut child);
    assert_eq!(status.signal(), Some(libc::SIGTERM), "{status:?}");
    assert_gone(grandchild);
    drop(input);
}
