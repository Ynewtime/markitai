//! macOS Vision reads in a process of its own, this program started again, which
//! is killed and started anew when a request stops (`docs/ocr.md`).
#![cfg(target_os = "macos")]
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const PAGE: &[u8] = include_bytes!("../../markitai-core/src/ocr/fixtures/english.png");

fn markitai(home: &Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_markitai"));
    command
        .env("HOME", home)
        .env("MARKITAI_HOME", home.join("state"))
        .env("MARKITAI_OCR_BACKEND", "vision")
        .env("MARKITAI_OCR_TRACE", "1")
        .current_dir(home);
    command
}

fn read_page(home: &Path) -> Command {
    let page = home.join("page.png");
    std::fs::write(&page, PAGE).unwrap();
    let mut command = markitai(home);
    command.arg(&page).args(["--ocr", "--no-llm", "--no-cache"]);
    command
}

/// An output without the time it was made at.
fn reading(output: &[u8]) -> String {
    String::from_utf8_lossy(output)
        .lines()
        .filter(|line| !line.starts_with("markitai_processed:"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// What the reader of the running program `parent` is, once it has one.
fn reader_of(parent: u32) -> u32 {
    let began = Instant::now();
    loop {
        let listing = Command::new("/usr/bin/pgrep")
            .args(["-P", &parent.to_string()])
            .output()
            .unwrap();
        if let Some(pid) = String::from_utf8_lossy(&listing.stdout).lines().next() {
            return pid.trim().parse().unwrap();
        }
        assert!(
            began.elapsed() < Duration::from_secs(60),
            "no reader started"
        );
        std::thread::sleep(Duration::from_millis(2));
    }
}

#[test]
fn a_vision_reader_goes_when_its_input_closes() {
    let home = tempfile::tempdir().unwrap();
    let done = markitai(home.path())
        .arg("__markitai-vision-reader")
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert!(done.status.success(), "{done:?}");
    assert!(done.stdout.is_empty(), "{done:?}");
}

/// A stopped process is to the program what a stopped request is: work and no CPU.
#[test]
fn vision_reads_the_same_after_its_reader_stops() {
    let home = tempfile::tempdir().unwrap();
    let plain = read_page(home.path()).output().unwrap();
    assert!(plain.status.success(), "{plain:?}");
    let text = reading(&plain.stdout);
    assert!(!text.trim().is_empty(), "{plain:?}");

    let running = read_page(home.path())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let reader = reader_of(running.id());
    let stopped = Command::new("/bin/kill")
        .args(["-STOP", &reader.to_string()])
        .status()
        .unwrap();
    assert!(stopped.success());
    let again = running.wait_with_output().unwrap();
    assert!(again.status.success(), "{again:?}");
    let trace = String::from_utf8_lossy(&again.stderr);
    assert!(trace.contains("vision reader restart=1"), "{trace}");
    assert_eq!(reading(&again.stdout), text);
}

/// Nobody reading any more is no reason to wait for ever: the trace of the restart
/// must not stop what restarts.
#[test]
fn vision_ends_after_its_reader_stops_with_nobody_reading_its_trace() {
    let home = tempfile::tempdir().unwrap();
    let mut running = read_page(home.path())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    drop(running.stderr.take());
    let reader = reader_of(running.id());
    let stopped = Command::new("/bin/kill")
        .args(["-STOP", &reader.to_string()])
        .status()
        .unwrap();
    assert!(stopped.success());
    let began = Instant::now();
    let status = loop {
        if let Some(status) = running.try_wait().unwrap() {
            break status;
        }
        if began.elapsed() > Duration::from_secs(280) {
            let _ = running.kill();
            panic!("still waiting");
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    assert!(status.success(), "{status:?}");
}
