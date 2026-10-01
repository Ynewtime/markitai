//! Standard output that nobody reads (`markitai note.txt | head -1`) ends a run
//! quietly, as with other Unix tools; a stream that cannot be written for any
//! other reason fails with a message that names it.
#![cfg(unix)]

use std::os::fd::{FromRawFd, OwnedFd};
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::process::{Command, Output, Stdio};
use std::sync::Mutex;

/// A pipe's descriptors are inherited by any child forked while they exist.
/// Tests of this file therefore start their children one at a time.
static CHILDREN: Mutex<()> = Mutex::new(());

fn command(root: &Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_markitai"));
    command.env_clear();
    for name in [
        "HOME",
        "PATH",
        "SYSTEMROOT",
        "USERPROFILE",
        "TMPDIR",
        "TEMP",
        "TMP",
    ] {
        if let Some(value) = std::env::var_os(name) {
            command.env(name, value);
        }
    }
    command
        .current_dir(root)
        .env("MARKITAI_HOME", root.join("home"));
    command
}

/// The write end of a pipe whose read end is already closed: the first write
/// fails with EPIPE, however small it is.
fn pipe_without_reader() -> Stdio {
    let mut descriptors = [0 as libc::c_int; 2];
    assert_eq!(unsafe { libc::pipe(descriptors.as_mut_ptr()) }, 0);
    for descriptor in descriptors {
        unsafe { libc::fcntl(descriptor, libc::F_SETFD, libc::FD_CLOEXEC) };
    }
    unsafe { libc::close(descriptors[0]) };
    Stdio::from(unsafe { OwnedFd::from_raw_fd(descriptors[1]) })
}

fn run_into_closed_pipe(root: &Path, args: &[&str]) -> Output {
    let _one_at_a_time = CHILDREN.lock().unwrap_or_else(|e| e.into_inner());
    let mut command = command(root);
    command
        .args(args)
        .stdin(Stdio::null())
        .stdout(pipe_without_reader())
        .stderr(Stdio::piped());
    command.spawn().unwrap().wait_with_output().unwrap()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

#[test]
fn a_reader_that_has_gone_ends_every_kind_of_output_quietly() {
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("note.txt"), "# Hello\n世界\n").unwrap();
    std::fs::create_dir(root.path().join("docs")).unwrap();
    for name in ["a.txt", "b.txt"] {
        std::fs::write(root.path().join("docs").join(name), "body\n").unwrap();
    }
    for args in [
        // The converted Markdown.
        vec!["note.txt", "--pure"],
        // Output printed line by line elsewhere in the program.
        vec!["config", "list"],
        vec!["config", "path"],
        vec!["docs", "-o", "out", "--dry-run", "--quiet"],
        // The help shown without input.
        vec![],
        vec!["--help"],
    ] {
        let output = run_into_closed_pipe(root.path(), &args);
        assert_eq!(
            output.status.code(),
            Some(0),
            "{args:?}: {}",
            stderr(&output)
        );
        assert!(
            output.stderr.is_empty(),
            "{args:?} must not complain: {}",
            stderr(&output)
        );
    }
}

#[test]
fn a_reader_that_has_gone_changes_neither_the_work_done_nor_a_failing_status() {
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("note.txt"), "# Hello\n").unwrap();
    // The JSON result has nowhere to go, but the file is still written.
    let output = run_into_closed_pipe(root.path(), &["note.txt", "-o", "out", "--json"]);
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    assert!(output.stderr.is_empty(), "{}", stderr(&output));
    let written = std::fs::read_to_string(root.path().join("out/note.txt.md")).unwrap();
    assert!(written.contains("# Hello"), "{written}");
    // A real failure keeps its status, and the closed pipe adds nothing to it:
    // the JSON result carries the error, so stderr stays empty.
    let output = run_into_closed_pipe(root.path(), &["missing.txt", "-o", "out", "--json"]);
    assert_eq!(output.status.code(), Some(1), "{}", stderr(&output));
    assert!(output.stderr.is_empty(), "{}", stderr(&output));
    // Without --json the error is printed on stderr as usual.
    let output = run_into_closed_pipe(root.path(), &["missing.txt"]);
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(
        stderr(&output),
        "Error: Input file not found: missing.txt\n"
    );
}

/// A regular file that may grow only to `limit` bytes: further writes fail with
/// EFBIG (SIGXFSZ ignored), as writes to `/dev/full` fail with ENOSPC on Linux.
fn run_into_file_with_limit(root: &Path, args: &[&str], limit: u64) -> (Output, u64) {
    let _one_at_a_time = CHILDREN.lock().unwrap_or_else(|e| e.into_inner());
    let target = root.join("stdout.bin");
    let mut command = command(root);
    command
        .args(args)
        .stdin(Stdio::null())
        .stdout(std::fs::File::create(&target).unwrap())
        .stderr(Stdio::piped());
    unsafe {
        command.pre_exec(move || {
            libc::signal(libc::SIGXFSZ, libc::SIG_IGN);
            let limits = libc::rlimit {
                rlim_cur: limit as libc::rlim_t,
                rlim_max: limit as libc::rlim_t,
            };
            if libc::setrlimit(libc::RLIMIT_FSIZE, &limits) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let output = command.spawn().unwrap().wait_with_output().unwrap();
    (output, std::fs::metadata(target).unwrap().len())
}

/// The device that fails every write with ENOSPC, where the system has it.
#[cfg(target_os = "linux")]
#[test]
fn dev_full_fails_with_a_message_naming_standard_output() {
    let _one_at_a_time = CHILDREN.lock().unwrap_or_else(|e| e.into_inner());
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("note.txt"), "# Hello\n").unwrap();
    let full = std::fs::OpenOptions::new()
        .write(true)
        .open("/dev/full")
        .unwrap();
    let output = command(root.path())
        .args(["note.txt", "--pure"])
        .stdin(Stdio::null())
        .stdout(full)
        .stderr(Stdio::piped())
        .spawn()
        .unwrap()
        .wait_with_output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1), "{}", stderr(&output));
    assert_eq!(
        stderr(&output),
        "Error: Cannot write to standard output: No space left on device (os error 28)\n"
    );
}

#[test]
fn a_stream_that_cannot_be_written_fails_with_a_message_naming_standard_output() {
    let root = tempfile::tempdir().unwrap();
    let body = "A line of text that makes the document longer.\n".repeat(8 * 1024);
    std::fs::write(root.path().join("big.txt"), &body).unwrap();
    let limit = 64 * 1024;
    let (output, size) = run_into_file_with_limit(root.path(), &["big.txt", "--pure"], limit);
    assert_eq!(output.status.code(), Some(1), "{}", stderr(&output));
    let text = stderr(&output);
    assert!(
        text.starts_with("Error: Cannot write to standard output: "),
        "{text}"
    );
    assert!(text.contains("File too large"), "{text}");
    assert_eq!(text.matches("Error:").count(), 1, "{text}");
    assert!(size <= limit, "the limit did not apply: {size} bytes");

    // Output that fits is unaffected by the same limit.
    std::fs::write(root.path().join("small.txt"), "short\n").unwrap();
    let (output, size) = run_into_file_with_limit(root.path(), &["small.txt", "--pure"], limit);
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    assert!(output.stderr.is_empty(), "{}", stderr(&output));
    assert_eq!(size, 6);

    // Other printed output reports it too.
    let (output, _) = run_into_file_with_limit(root.path(), &["config", "list"], 16);
    assert_eq!(output.status.code(), Some(1), "{}", stderr(&output));
    assert!(
        stderr(&output).contains("Cannot write to standard output"),
        "{}",
        stderr(&output)
    );
}
