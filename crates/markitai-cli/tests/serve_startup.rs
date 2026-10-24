#![cfg(unix)]
//! What `markitai serve` prints when it starts, and why it refuses to.
#[path = "support/console.rs"]
mod console;

use std::{
    io::{BufRead, BufReader},
    net::TcpListener,
    path::Path,
    process::{Child, Command, Stdio},
    sync::mpsc::{Receiver, channel},
    time::{Duration, Instant},
};

const WAIT: Duration = Duration::from_secs(30);

struct Serve {
    child: Child,
    lines: Receiver<String>,
}

impl Serve {
    fn start(home: &Path, language: Option<&str>, args: &[&str]) -> Self {
        let mut command = Command::new(env!("CARGO_BIN_EXE_markitai"));
        console::english(command.env_clear());
        for key in ["PATH", "HOME", "TMPDIR"] {
            if let Some(value) = std::env::var_os(key) {
                command.env(key, value);
            }
        }
        if let Some(language) = language {
            command.env("MARKITAI_LANG", language);
        }
        let mut child = command
            .current_dir(home)
            .env("MARKITAI_HOME", home.join("home"))
            .env("NO_PROXY", "127.0.0.1,localhost")
            .arg("serve")
            .arg("--no-open")
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let reader = child.stderr.take().unwrap();
        let (sender, lines) = channel();
        std::thread::spawn(move || {
            for line in BufReader::new(reader).lines().map_while(Result::ok) {
                if sender.send(line).is_err() {
                    break;
                }
            }
        });
        Self { child, lines }
    }

    /// Everything printed up to and including the first line containing `last`.
    fn until(&self, last: &str) -> Vec<String> {
        let deadline = Instant::now() + WAIT;
        let mut seen = Vec::new();
        while let Ok(line) = self
            .lines
            .recv_timeout(deadline.saturating_duration_since(Instant::now()))
        {
            let done = line.contains(last);
            seen.push(line);
            if done {
                return seen;
            }
        }
        panic!("no line containing {last:?}; saw {seen:#?}");
    }

    fn finish(mut self) -> (Option<i32>, Vec<String>) {
        let deadline = Instant::now() + WAIT;
        let status = loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                break status;
            }
            assert!(Instant::now() < deadline, "server did not exit");
            std::thread::sleep(Duration::from_millis(10));
        };
        // The reader thread may still be forwarding the last lines; it ends at end of file.
        let mut lines = Vec::new();
        while let Ok(line) = self.lines.recv_timeout(WAIT) {
            lines.push(line);
        }
        (status.code(), lines)
    }
}

impl Drop for Serve {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn port_of(line: &str) -> u16 {
    line.rsplit(':').next().unwrap().parse().unwrap()
}

#[test]
fn a_local_server_prints_the_address_the_data_location_and_how_to_stop() {
    let temp = tempfile::tempdir().unwrap();
    let serve = Serve::start(temp.path(), None, &["--host", "127.0.0.1", "--port", "0"]);
    let lines = serve.until("Ctrl-C");
    let port = port_of(&lines[0]);
    // The title carries the address, then a blank line and the items.
    assert_eq!(
        lines[0],
        format!("◆ Markitai server listening on http://127.0.0.1:{port}")
    );
    assert_eq!(lines[1], "");
    let token = lines[2]
        .strip_prefix("  • Access token: ")
        .unwrap_or_else(|| panic!("{lines:?}"));
    assert_eq!(token.len(), 64, "{lines:?}");
    // The address to open sits on a line of its own, so it copies whole.
    assert_eq!(lines[3], "  → Open in your browser:");
    assert_eq!(
        lines[4],
        format!("    http://127.0.0.1:{port}/#token={token}")
    );
    assert_eq!(
        lines[5],
        format!(
            "  • Jobs and history: {}",
            temp.path().join("home/serve/jobs").display()
        )
    );
    assert!(lines[6].starts_with("  • Press Ctrl-C to stop"), "{lines:?}");
    assert_eq!(
        lines.len(),
        7,
        "a loopback server has nothing to warn about: {lines:?}"
    );
    // The data location is real: the service created it.
    assert!(temp.path().join("home/serve/jobs").is_dir());
}

#[test]
fn the_terminal_language_changes_every_line_but_not_where_the_values_are() {
    let temp = tempfile::tempdir().unwrap();
    let serve = Serve::start(
        temp.path(),
        Some("zh"),
        &["--host", "127.0.0.1", "--port", "0"],
    );
    let lines = serve.until("Ctrl-C");
    let port = port_of(&lines[0]);
    assert_eq!(
        lines[0],
        format!("◆ Markitai 服务正在监听 http://127.0.0.1:{port}"),
        "{lines:?}"
    );
    let token = lines[2]
        .strip_prefix("  • 访问令牌：")
        .unwrap_or_else(|| panic!("{lines:?}"));
    assert_eq!(token.len(), 64, "{lines:?}");
    assert_eq!(lines[3], "  → 在浏览器中打开：", "{lines:?}");
    assert_eq!(lines[4], format!("    http://127.0.0.1:{port}/#token={token}"));
    assert!(lines[5].starts_with("  • 任务和历史："), "{lines:?}");
    assert!(lines[6].starts_with("  • 按 Ctrl-C 停止"), "{lines:?}");
    assert!(
        lines.iter().all(|line| !line.contains("listening")
            && !line.contains("token:")
            && !line.contains("Open in")),
        "{lines:?}"
    );
}

#[test]
fn listening_beyond_this_computer_warns_and_hands_out_the_token_only_where_it_is_needed() {
    let temp = tempfile::tempdir().unwrap();
    let serve = Serve::start(temp.path(), None, &["--host", "0.0.0.0", "--port", "0"]);
    let lines = serve.until("Ctrl-C");
    let port = port_of(&lines[0]);
    assert_eq!(
        lines[0],
        format!("◆ Markitai server listening on http://0.0.0.0:{port}")
    );
    assert!(
        lines[2].starts_with(&format!(
            "  ! Binding to 0.0.0.0:{port} makes this server reachable"
        )),
        "{lines:?}"
    );
    assert!(
        lines[3].starts_with("    Other machines") && lines[3].contains("like a password"),
        "{lines:?}"
    );
    // Local and remote browsers both receive the token in a fragment.
    let token = lines[4]
        .strip_prefix("  • Access token: ")
        .unwrap_or_else(|| panic!("{lines:?}"));
    let local = lines
        .iter()
        .position(|line| line == "  → Open in your browser:")
        .unwrap_or_else(|| panic!("{lines:?}"));
    assert_eq!(
        lines[local + 1],
        format!("    http://127.0.0.1:{port}/#token={token}")
    );
    // Any address offered to other devices carries the token, and says so.
    for (index, line) in lines
        .iter()
        .enumerate()
        .filter(|(_, line)| line.starts_with("  → From another device"))
    {
        assert!(line.contains("contains the access token"), "{line}");
        assert!(
            lines[index + 1].starts_with("    http://")
                && lines[index + 1].ends_with(&format!("#token={token}")),
            "{lines:?}"
        );
    }
}

#[test]
fn a_port_that_is_taken_is_named_with_the_way_out() {
    let temp = tempfile::tempdir().unwrap();
    let taken = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = taken.local_addr().unwrap().port().to_string();
    for (language, expected) in [
        (None, "the port is already in use"),
        (Some("zh"), "端口已被占用"),
    ] {
        let serve = Serve::start(
            temp.path(),
            language,
            &["--host", "127.0.0.1", "--port", &port],
        );
        let (code, lines) = serve.finish();
        assert_eq!(code, Some(1), "{lines:?}");
        let text = lines.join("\n");
        assert!(text.contains(&format!("127.0.0.1:{port}")), "{text}");
        assert!(text.contains(expected), "{text}");
        assert!(text.contains("--port"), "{text}");
        assert!(!text.contains("os error"), "{text}");
        // Nothing was announced: the server never started.
        assert!(!text.contains("listening on"), "{text}");
    }
}
