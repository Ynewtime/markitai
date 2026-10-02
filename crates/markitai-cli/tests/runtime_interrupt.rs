//! Interrupts and termination must not leave external runtime trees alive.
//!
//! The authored Codex stand-in (this test binary, see `fake_runtime`) starts
//! a descendant in its own tree and then hangs. A terminal interrupt reaches
//! only the CLI: on Unix the runtime leads its own process group, and on
//! Windows it runs in a job and a hidden console of its own. These tests
//! therefore interrupt the CLI process alone, as the terminal would.
//!
//! The Unix CLI keeps a terminating signal that it inherits as ignored (a
//! shell gives that to `cmd &` for SIGINT, `nohup` for SIGHUP). The children
//! here therefore start from a known disposition instead of inheriting the
//! one the test runner happened to be launched with: a gate started as a
//! background job would otherwise run the CLI with SIGINT ignored.
use serde_json::json;
use std::io::{BufRead, BufReader, Write};
#[cfg(unix)]
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

#[path = "../../markitai-core/src/subscription/fake_runtime.rs"]
mod fake_runtime;
#[path = "support/interrupt.rs"]
mod interrupt;

struct Fixture {
    root: tempfile::TempDir,
}

impl Fixture {
    fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        fake_runtime::install(&root.path().join("runtime"), &json!({"name":"sleep"}));
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
        self.spawn_with(args, stdin, false)
    }

    /// Start the CLI so that only it receives the interrupt. On Unix it
    /// starts with default terminating-signal dispositions and an empty
    /// signal mask, or with SIGINT inherited as ignored.
    fn spawn_with(&self, args: &[&str], stdin: Stdio, ignore_sigint: bool) -> Child {
        let mut command = Command::new(env!("CARGO_BIN_EXE_markitai"));
        command.env_clear().current_dir(self.root.path());
        for key in [
            "HOME",
            "PATH",
            "LANG",
            "TMPDIR",
            "SystemRoot",
            "TEMP",
            "TMP",
            "USERPROFILE",
            "LOCALAPPDATA",
        ] {
            if let Some(value) = std::env::var_os(key) {
                command.env(key, value);
            }
        }
        command
            .env("MARKITAI_HOME", self.path("home"))
            .env("CODEX_CLI_PATH", fake_runtime::program())
            .env("CODEX_HOME", self.path("runtime"))
            .args(["-c", "config.json"])
            .args(args)
            .stdin(stdin)
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        interrupt::interruptible(&mut command);
        #[cfg(unix)]
        // SAFETY: the closure runs between fork and exec and calls only
        // async-signal-safe functions on stack data.
        unsafe {
            command.pre_exec(move || {
                let mut mask: libc::sigset_t = std::mem::zeroed();
                libc::sigemptyset(&mut mask);
                libc::sigprocmask(libc::SIG_SETMASK, &mask, std::ptr::null_mut());
                for signal in [libc::SIGINT, libc::SIGTERM, libc::SIGHUP] {
                    let ignored = ignore_sigint && signal == libc::SIGINT;
                    libc::signal(
                        signal,
                        if ignored {
                            libc::SIG_IGN
                        } else {
                            libc::SIG_DFL
                        },
                    );
                }
                Ok(())
            });
        }
        #[cfg(not(unix))]
        assert!(!ignore_sigint, "inherited dispositions are a Unix notion");
        command.spawn().unwrap()
    }

    /// The runtime's descendant, once the hanging turn has started.
    fn grandchild(&self) -> u32 {
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

#[cfg(unix)]
fn signal(child: &Child, signal: i32) {
    assert_eq!(unsafe { libc::kill(child.id() as i32, signal) }, 0);
}

/// Time allowed from delivering the interrupt to the CLI's exit. The handler
/// only kills registered trees and ends the process, so a healthy exit takes
/// milliseconds; the margin covers a workstation saturated by parallel builds
/// while a real hang still fails.
const EXIT_DEADLINE: Duration = Duration::from_secs(45);

/// Wait for the CLI to exit after the interrupt. `related` are the runtime
/// processes that should also be gone; they are listed if the CLI is not.
fn wait(child: &mut Child, related: &[u32]) -> ExitStatus {
    let sent = Instant::now();
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            return status;
        }
        if sent.elapsed() >= EXIT_DEADLINE {
            let mut pids = vec![child.id()];
            pids.extend_from_slice(related);
            let states = interrupt::process_states(&pids);
            let _ = child.kill();
            panic!(
                "CLI did not terminate within {:?} of the interrupt\n{states}",
                sent.elapsed()
            );
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// A Unix descendant is reparented after its group is killed; wait for its
/// removal.
fn assert_gone(pid: u32) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while interrupt::alive(pid) {
        if Instant::now() >= deadline {
            interrupt::kill(pid);
            panic!("external runtime process outlived the CLI");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// An interrupt ends the CLI as SIGINT would: by the signal on Unix, with
/// exit status 130 on Windows.
fn assert_interrupted(status: ExitStatus) {
    #[cfg(unix)]
    assert_eq!(status.signal(), Some(libc::SIGINT), "{status:?}");
    #[cfg(not(unix))]
    assert_eq!(status.code(), Some(130), "{status:?}");
}

#[test]
fn single_conversion_interrupt_kills_runtime_group_and_keeps_signal_exit() {
    let f = Fixture::new();
    let mut child = f.spawn(&["inputs/source.md", "-o", "out"], Stdio::null());
    let grandchild = f.grandchild();
    interrupt::interrupt(&child);
    let status = wait(&mut child, &[grandchild]);
    assert_interrupted(status);
    assert_gone(grandchild);
}

/// A signal inherited as ignored stays ignored, as for any Unix tool started
/// by `cmd &` or `nohup`; the other terminating signals still clean up.
#[cfg(unix)]
#[test]
fn inherited_ignored_interrupt_is_kept_and_terminate_still_cleans_up() {
    let f = Fixture::new();
    let mut child = f.spawn_with(&["inputs/source.md", "-o", "out"], Stdio::null(), true);
    let grandchild = f.grandchild();
    signal(&child, libc::SIGINT);
    std::thread::sleep(Duration::from_millis(500));
    assert!(
        child.try_wait().unwrap().is_none(),
        "SIGINT was not ignored"
    );
    assert!(interrupt::alive(grandchild));
    signal(&child, libc::SIGTERM);
    let status = wait(&mut child, &[grandchild]);
    assert_eq!(status.signal(), Some(libc::SIGTERM), "{status:?}");
    assert_gone(grandchild);
}

#[cfg(unix)]
#[test]
fn single_conversion_hangup_and_terminate_also_clean_up() {
    for sent in [libc::SIGHUP, libc::SIGTERM] {
        let f = Fixture::new();
        let mut child = f.spawn(&["inputs/source.md", "-o", "out"], Stdio::null());
        let grandchild = f.grandchild();
        signal(&child, sent);
        let status = wait(&mut child, &[grandchild]);
        assert_eq!(status.signal(), Some(sent), "{status:?}");
        assert_gone(grandchild);
    }
}

/// The draining first interrupt is the batch's guard, which Windows installs
/// only with the durable batch state that it does not have yet.
#[cfg(unix)]
#[test]
fn batch_first_interrupt_waits_and_second_forces_exit_without_orphans() {
    let f = Fixture::new();
    let mut child = f.spawn(&["inputs", "-o", "out"], Stdio::null());
    let grandchild = f.grandchild();
    interrupt::interrupt(&child);
    // The first interrupt drains: active work keeps running.
    std::thread::sleep(Duration::from_millis(500));
    assert!(child.try_wait().unwrap().is_none());
    assert!(interrupt::alive(grandchild));
    interrupt::interrupt(&child);
    let status = wait(&mut child, &[grandchild]);
    assert_eq!(status.code(), Some(130), "{status:?}");
    assert_gone(grandchild);
}

/// MCP drains on its own interrupt; termination (SIGTERM on Unix) or, on
/// Windows, the Ctrl-Break it does not own still kills the runtime tree that
/// a tool call started.
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
    #[cfg(unix)]
    signal(&child, libc::SIGTERM);
    #[cfg(not(unix))]
    interrupt::interrupt(&child);
    let status = wait(&mut child, &[grandchild]);
    #[cfg(unix)]
    assert_eq!(status.signal(), Some(libc::SIGTERM), "{status:?}");
    #[cfg(not(unix))]
    assert_eq!(status.code(), Some(130), "{status:?}");
    assert_gone(grandchild);
    drop(input);
}
