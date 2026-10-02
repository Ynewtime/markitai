//! An interrupted batch says how many items it never started and how to
//! continue, in the terminal language.
//!
//! The interrupt itself is portable (`support/interrupt.rs`); the suite runs
//! where batches keep durable state for `--resume`, which Windows does not
//! have yet.
#![cfg(unix)]

use std::io::{BufRead, BufReader};
use std::net::TcpListener;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

#[path = "support/interrupt.rs"]
mod interrupt;

const WAIT: Duration = Duration::from_secs(20);

fn command(root: &Path, language: &str) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_markitai"));
    command.env_clear();
    for name in ["PATH", "SYSTEMROOT", "USERPROFILE", "TMPDIR", "TEMP", "TMP"] {
        if let Some(value) = std::env::var_os(name) {
            command.env(name, value);
        }
    }
    command
        .current_dir(root)
        .env("HOME", root)
        .env("MARKITAI_HOME", root.join("home"))
        .env("MARKITAI_LANG", language)
        .env("NO_PROXY", "127.0.0.1,localhost");
    command
}

/// Start a URL list of `count` addresses on a server that never answers, with
/// one conversion at a time, interrupt it while the first request is held, and
/// return the exit code with every stderr line.
fn interrupt_a_url_batch(root: &Path, language: &str, count: usize) -> (Option<i32>, Vec<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let address = listener.local_addr().unwrap();
    let list: String = (0..count)
        .map(|index| format!("http://{address}/page{index} page{index}\n"))
        .collect();
    std::fs::write(root.join("list.urls"), list).unwrap();
    let config = r#"{"cache":{"enabled":false},"history":{"record":false}}"#;
    let mut command = command(root, language);
    let mut child = interrupt::interruptible(&mut command)
        .args([
            "list.urls",
            "-o",
            "out",
            "--url-concurrency",
            "1",
            "--config-json",
            config,
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + WAIT;
    let held = loop {
        match listener.accept() {
            Ok((stream, _)) => break stream,
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                assert!(Instant::now() < deadline, "the CLI never asked for a URL");
                std::thread::sleep(Duration::from_millis(5));
            }
            Err(error) => panic!("accept failed: {error}"),
        }
    };
    let stderr = child.stderr.take().unwrap();
    let lines = std::thread::spawn(move || {
        BufReader::new(stderr)
            .lines()
            .map(Result::unwrap)
            .collect::<Vec<_>>()
    });
    interrupt::interrupt(&child);
    // The acknowledgement comes first; the held request is then released.
    std::thread::sleep(Duration::from_millis(500));
    drop(held);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if Instant::now() > deadline {
            let _ = child.kill();
            panic!("the interrupted CLI did not exit");
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    (status.code(), lines.join().unwrap())
}

#[test]
fn the_summary_counts_unstarted_items_and_suggests_resume() {
    let root = tempfile::tempdir().unwrap();
    let (code, lines) = interrupt_a_url_batch(root.path(), "en", 3);
    assert_eq!(code, Some(130));
    let line = lines
        .iter()
        .find(|line| line.starts_with("Not processed "))
        .unwrap_or_else(|| panic!("no unprocessed line: {lines:#?}"));
    assert!(
        line.starts_with("Not processed 2 items: http://127.0.0.1:"),
        "{line}"
    );
    assert!(line.contains("/page1"), "{line}");
    assert!(line.contains("/page2"), "{line}");
    assert!(
        !line.contains("/page0"),
        "the item that was running is not unprocessed: {line}"
    );
    assert!(
        line.ends_with("Run the same command with --resume to continue."),
        "{line}"
    );
    // The line sits between the totals and the output location.
    let at = |prefix: &str| {
        lines
            .iter()
            .position(|line| line.starts_with(prefix))
            .unwrap()
    };
    assert!(at("Done: ") < at("Not processed "), "{lines:#?}");
    assert!(at("Not processed ") < at("Output: "), "{lines:#?}");
}

#[test]
fn the_unstarted_items_are_what_resume_then_picks_up() {
    let root = tempfile::tempdir().unwrap();
    let (code, _) = interrupt_a_url_batch(root.path(), "en", 3);
    assert_eq!(code, Some(130));
    // The server is gone, so the resumed items fail quickly instead of waiting.
    let output = command(root.path(), "en")
        .args([
            "list.urls",
            "-o",
            "out",
            "--url-concurrency",
            "1",
            "--resume",
            "--json",
            "--config-json",
            r#"{"cache":{"enabled":false},"history":{"record":false}}"#,
        ])
        .stdin(Stdio::null())
        .output()
        .unwrap();
    let document: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let sources: Vec<_> = document["items"]
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item["source"].as_str().unwrap().to_owned())
        .collect();
    for page in ["/page1", "/page2"] {
        assert!(
            sources.iter().any(|source| source.ends_with(page)),
            "{page} was not picked up: {sources:?}"
        );
    }
}

#[test]
fn a_completed_batch_has_no_unprocessed_line_and_an_interrupted_one_has_it_in_chinese() {
    let root = tempfile::tempdir().unwrap();
    std::fs::create_dir(root.path().join("docs")).unwrap();
    std::fs::write(root.path().join("docs/a.txt"), "A\n").unwrap();
    let output = command(root.path(), "en")
        .args(["docs", "-o", "out"])
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(0));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("Not processed"), "{stderr}");
    assert!(!stderr.contains("--resume"), "{stderr}");

    let root = tempfile::tempdir().unwrap();
    let (code, lines) = interrupt_a_url_batch(root.path(), "zh", 3);
    assert_eq!(code, Some(130));
    let line = lines
        .iter()
        .find(|line| line.starts_with("未处理 "))
        .unwrap_or_else(|| panic!("no unprocessed line: {lines:#?}"));
    assert!(line.starts_with("未处理 2 项：http://127.0.0.1:"), "{line}");
    assert!(
        line.ends_with("。使用相同命令并加上 --resume 可继续。"),
        "{line}"
    );
    assert!(!lines.iter().any(|line| line.starts_with("Not processed")));
}
