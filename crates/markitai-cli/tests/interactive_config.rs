use serde_json::{Value, json};
use std::path::Path;
use std::process::{Command, Stdio};

fn command(root: &Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_markitai"));
    command.env_clear();
    for key in ["HOME", "PATH", "SYSTEMROOT", "USERPROFILE", "TMPDIR"] {
        if let Some(value) = std::env::var_os(key) {
            command.env(key, value);
        }
    }
    command
        .current_dir(root)
        .env("MARKITAI_HOME", root.join("home"))
        // Runtime fixtures are independent of commands installed on the host.
        .env("COPILOT_CLI_PATH", root.join("missing-copilot"))
        .env("CLAUDE_CLI_PATH", root.join("missing-claude"))
        .env("CODEX_CLI_PATH", root.join("missing-codex"));
    command
}
#[test]
fn noninteractive_edit_and_wizard_refuse_but_quick_init_preserves_existing_settings() {
    let root = tempfile::tempdir().unwrap();
    for args in [vec!["config", "edit"], vec!["init"]] {
        let output = command(root.path())
            .args(args)
            .stdin(Stdio::null())
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(2));
        assert!(String::from_utf8_lossy(&output.stderr).contains("requires a terminal"));
    }
    assert!(!root.path().join("home").exists());
    let output = command(root.path())
        .args(["init", "--yes", "--local"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let path = root.path().join("markitai.json");
    let raw = json!({"unknown":{"preserve":1},"output":{"dir":"mine"},"llm":{"enabled":true,"model_list":[{"model_name":"mine","litellm_params":{"model":"openai/already","api_key":"env:MY_KEY"}}]}});
    std::fs::write(&path, serde_json::to_vec(&raw).unwrap()).unwrap();
    let mut last = Vec::new();
    for _ in 0..2 {
        let output = command(root.path())
            .env("MODEL", "openai/new")
            .env("OPENAI_API_KEY", "never-write-this-key")
            .args(["init", "-y", "--local"])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let bytes = std::fs::read(&path).unwrap();
        let saved: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(saved["unknown"], raw["unknown"]);
        assert_eq!(saved["output"], raw["output"]);
        assert_eq!(saved["llm"]["enabled"], true);
        assert_eq!(saved["llm"]["model_list"].as_array().unwrap().len(), 2);
        assert_eq!(saved["llm"]["model_list"][0], raw["llm"]["model_list"][0]);
        assert!(!String::from_utf8_lossy(&bytes).contains("never-write-this-key"));
        if !last.is_empty() {
            assert_eq!(bytes, last);
        }
        last = bytes;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            std::fs::metadata(path).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }
}

#[cfg(unix)]
fn terminal(root: &Path, args: &[&str], script: &[u8]) -> (std::process::ExitStatus, String) {
    use std::io::{Read, Write};
    use std::os::fd::{AsRawFd, FromRawFd};
    let (mut master, mut slave) = (-1, -1);
    assert_eq!(
        unsafe {
            libc::openpty(
                &mut master,
                &mut slave,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        },
        0
    );
    let mut master = unsafe { std::fs::File::from_raw_fd(master) };
    let slave = unsafe { std::fs::File::from_raw_fd(slave) };
    let mut reader = master.try_clone().unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
    let read = std::thread::spawn(move || {
        let mut all = Vec::new();
        let mut buf = [0; 4096];
        while std::time::Instant::now() < deadline {
            let mut descriptor = libc::pollfd {
                fd: reader.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            };
            let ready = unsafe { libc::poll(&mut descriptor, 1, 100) };
            if ready < 0 {
                if std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted {
                    continue;
                }
                break;
            }
            if ready == 0 {
                continue;
            }
            match reader.read(&mut buf) {
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Ok(0) | Err(_) => break,
                Ok(n) => all.extend_from_slice(&buf[..n]),
            }
        }
        all
    });
    let mut child = command(root)
        .args(args)
        .stdin(slave.try_clone().unwrap())
        .stderr(slave)
        .stdout(Stdio::null())
        .spawn()
        .unwrap();
    master.write_all(script).unwrap();
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if std::time::Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!("Interactive CLI did not finish");
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    };
    drop(master);
    let output = String::from_utf8_lossy(&read.join().unwrap()).into_owned();
    (status, output)
}

#[cfg(unix)]
#[test]
fn actual_terminal_editor_searches_rejects_invalid_values_saves_and_cancels() {
    let root = tempfile::tempdir().unwrap();
    let raw =
        json!({"custom":7,"image":{"quality":75},"fetch":{"jina":{"api_key":"hidden-credential"}}});
    std::fs::write(
        root.path().join("cfg.json"),
        serde_json::to_vec(&raw).unwrap(),
    )
    .unwrap();
    let (status, output) = terminal(
        root.path(),
        &["-c", "cfg.json", "config", "edit"],
        b"/imgql\n1\n500\n1\n82\noutput.dir\n:cancel\nq\n",
    );
    assert!(status.success(), "{output}");
    assert!(output.contains("Invalid value:"));
    assert!(output.contains("Saved image.quality = 82"));
    assert!(!output.contains("hidden-credential"));
    let saved: Value =
        serde_json::from_slice(&std::fs::read(root.path().join("cfg.json")).unwrap()).unwrap();
    assert_eq!(saved["image"]["quality"], 82);
    assert_eq!(saved["custom"], 7);
    assert_eq!(saved["fetch"], raw["fetch"]);
    assert!(saved.get("output").is_none());
}

#[cfg(unix)]
#[test]
fn actual_init_wizard_keeps_or_explicitly_replaces_a_malformed_config() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("cfg.json");
    std::fs::write(&path, "malformed but retained").unwrap();
    let (status, output) = terminal(root.path(), &["init", "-o", "cfg.json"], b"3\n");
    assert!(status.success(), "{output}");
    assert_eq!(
        std::fs::read_to_string(&path).unwrap(),
        "malformed but retained"
    );
    let (status, output) = terminal(root.path(), &["init", "-o", "cfg.json"], b"2\n");
    assert!(status.success(), "{output}");
    let saved: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    assert_eq!(saved["output"]["dir"], "./output");
    assert_eq!(saved["llm"]["enabled"], false);
}

#[cfg(unix)]
#[test]
fn interrupted_secret_entry_restores_actual_terminal_echo_without_saving() {
    use std::io::Write;
    use std::os::fd::{AsRawFd, FromRawFd};
    let root = tempfile::tempdir().unwrap();
    let (mut master, mut slave) = (-1, -1);
    assert_eq!(
        unsafe {
            libc::openpty(
                &mut master,
                &mut slave,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        },
        0
    );
    let mut master = unsafe { std::fs::File::from_raw_fd(master) };
    let slave = unsafe { std::fs::File::from_raw_fd(slave) };
    let flags = || {
        let mut terminal = std::mem::MaybeUninit::<libc::termios>::uninit();
        assert_eq!(
            unsafe { libc::tcgetattr(slave.as_raw_fd(), terminal.as_mut_ptr()) },
            0
        );
        unsafe { terminal.assume_init().c_lflag }
    };
    let original = flags();
    assert_ne!(original & libc::ECHO, 0);
    let mut child = command(root.path())
        .args(["config", "edit"])
        .stdin(slave.try_clone().unwrap())
        .stderr(slave.try_clone().unwrap())
        .stdout(Stdio::null())
        .spawn()
        .unwrap();
    master.write_all(b"fetch.jina.api_key\n").unwrap();
    let start = std::time::Instant::now();
    while flags() & libc::ECHO != 0 {
        if start.elapsed() > std::time::Duration::from_secs(10) {
            let _ = child.kill();
            let _ = child.wait();
            panic!("Secret prompt did not disable echo");
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert_eq!(
        unsafe { libc::kill(child.id() as libc::pid_t, libc::SIGINT) },
        0
    );
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if start.elapsed() > std::time::Duration::from_secs(15) {
            let _ = child.kill();
            let _ = child.wait();
            panic!("Interrupted editor did not finish");
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    };
    assert_eq!(status.code(), Some(130));
    assert_eq!(flags() & libc::ECHO, original & libc::ECHO);
    assert!(!root.path().join("home/config.json").exists());
}
