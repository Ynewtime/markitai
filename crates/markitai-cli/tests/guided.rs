#[path = "support/console.rs"]
mod console;

#[cfg(unix)]
use serde_json::{Value, json};
use std::path::Path;
use std::process::{Command, Stdio};

fn command(root: &Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_markitai"));
    console::english(command.env_clear());
    for key in ["HOME", "PATH", "SYSTEMROOT", "USERPROFILE", "TMPDIR"] {
        if let Some(value) = std::env::var_os(key) {
            command.env(key, value);
        }
    }
    command
        .current_dir(root)
        .env("MARKITAI_HOME", root.join("home"))
        .env("NO_PROXY", "127.0.0.1,localhost")
        // No subscription runtime signed in on this machine is offered.
        .env("CLAUDE_CLI_PATH", root.join("absent"))
        .env("COPILOT_CLI_PATH", root.join("absent"))
        .env("CODEX_CLI_PATH", root.join("absent"));
    command
}

#[test]
fn missing_input_shows_help_and_nonterminal_or_json_wizards_fail_before_output() {
    let dir = tempfile::tempdir().unwrap();
    let output = command(dir.path())
        .args(["--quiet", "--no-llm"])
        .output()
        .unwrap();
    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).contains("Usage:"));
    for args in [vec!["-I"], vec!["-I", "--json", "-o", "out"]] {
        let output = command(dir.path())
            .args(args)
            .stdin(Stdio::null())
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(2));
        assert!(output.stdout.is_empty());
    }
    assert!(!dir.path().join("out").exists());
}

#[test]
fn presets_ignore_case_and_explicit_paired_flags_keep_their_precedence() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("source.txt"), "Unchanged source\n").unwrap();
    let output = command(dir.path())
        .args(["source.txt", "-p", "MiNiMaL", "--pure"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, b"Unchanged source\n");
    let output = command(dir.path())
        .args([
            "source.txt",
            "-p",
            "CuStOm",
            "--llm",
            "--no-llm",
            "--pure",
            "--config-json",
            r#"{"presets":{"custom":{"llm":true}}}"#,
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, b"Unchanged source\n");
}

#[cfg(unix)]
mod terminal {
    use super::*;
    use std::io::{Read, Write};
    use std::os::fd::{AsRawFd, FromRawFd};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    struct Session {
        child: std::process::Child,
        master: std::fs::File,
        slave: std::fs::File,
        transcript: Arc<Mutex<Vec<u8>>>,
        reader: Option<std::thread::JoinHandle<()>>,
        stop: Arc<std::sync::atomic::AtomicBool>,
    }
    impl Session {
        fn new(mut command: Command) -> Self {
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
            let master = unsafe { std::fs::File::from_raw_fd(master) };
            let slave = unsafe { std::fs::File::from_raw_fd(slave) };
            let child = command
                .stdin(slave.try_clone().unwrap())
                .stderr(slave.try_clone().unwrap())
                .stdout(Stdio::piped())
                .spawn()
                .unwrap();
            let transcript = Arc::new(Mutex::new(Vec::new()));
            let captured = transcript.clone();
            let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let finished = stop.clone();
            let mut reader = master.try_clone().unwrap();
            // Keep a slave for ECHO verification, so the reader needs an explicit
            // short poll deadline after the child has exited rather than EOF.
            let handle = std::thread::spawn(move || {
                let deadline = Instant::now() + Duration::from_secs(20);
                let mut bytes = [0; 4096];
                while Instant::now() < deadline {
                    let mut descriptor = libc::pollfd {
                        fd: reader.as_raw_fd(),
                        events: libc::POLLIN,
                        revents: 0,
                    };
                    let result = unsafe { libc::poll(&mut descriptor, 1, 100) };
                    if result > 0 {
                        match reader.read(&mut bytes) {
                            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {
                                continue;
                            }
                            Ok(0) | Err(_) => break,
                            Ok(n) => captured.lock().unwrap().extend_from_slice(&bytes[..n]),
                        }
                    } else if result < 0
                        && std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted
                    {
                        continue;
                    } else if result < 0 || finished.load(std::sync::atomic::Ordering::Relaxed) {
                        break;
                    }
                }
            });
            Self {
                child,
                master,
                slave,
                transcript,
                reader: Some(handle),
                stop,
            }
        }
        fn send(&mut self, script: &str) {
            self.master.write_all(script.as_bytes()).unwrap();
        }
        /// The transcript without the colours of the terminal style.
        fn text(&self) -> String {
            String::from_utf8_lossy(&self.transcript.lock().unwrap())
                .split("\u{1b}[")
                .enumerate()
                .map(|(index, part)| match part.split_once('m') {
                    Some((_, rest)) if index > 0 => rest,
                    _ => part,
                })
                .collect()
        }
        fn wait_for(&self, text: &str) {
            let start = Instant::now();
            while !self.text().contains(text) {
                assert!(
                    start.elapsed() < Duration::from_secs(15),
                    "Missing prompt {text}: {}",
                    self.text()
                );
                std::thread::sleep(Duration::from_millis(10));
            }
        }
        fn finish(&mut self) -> (std::process::ExitStatus, Vec<u8>) {
            let start = Instant::now();
            let status = loop {
                if let Some(status) = self.child.try_wait().unwrap() {
                    break status;
                }
                assert!(
                    start.elapsed() < Duration::from_secs(15),
                    "Wizard timed out: {}",
                    self.text()
                );
                std::thread::sleep(Duration::from_millis(10));
            };
            let mut stdout = Vec::new();
            self.child
                .stdout
                .take()
                .unwrap()
                .read_to_end(&mut stdout)
                .unwrap();
            self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
            self.reader.take().unwrap().join().unwrap();
            (status, stdout)
        }
    }
    impl Drop for Session {
        fn drop(&mut self) {
            let _ = self.child.kill();
            let _ = self.child.wait();
            // Signal reader termination without waiting for its overall timeout.
            if let Some(handle) = self.reader.take() {
                self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
                let _ = handle.join();
            }
        }
    }
    fn base() -> Value {
        json!({"llm":{"enabled":false},"ocr":{"enabled":false},"image":{"alt_enabled":false,"desc_enabled":false},"screenshot":{"enabled":false},"cache":{"enabled":false},"history":{"record":false},"output":{"report":true}})
    }

    #[test]
    fn file_wizard_uses_unicode_paths_effective_overrides_and_normal_reports_without_saving_config()
    {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("源 文档.txt"), "Preserved Unicode 世界\n").unwrap();
        let mut raw = base();
        raw["ocr"]["enabled"] = json!(true);
        let cfg_bytes = serde_json::to_vec(&raw).unwrap();
        std::fs::write(dir.path().join("config.json"), &cfg_bytes).unwrap();
        let mut cmd = command(dir.path());
        cmd.args([
            "-I",
            "-c",
            "config.json",
            "--config-json",
            r#"{"output":{"dir":"custom out"}}"#,
            "--no-ocr",
        ]);
        let mut session = Session::new(cmd);
        session.send("1\n源 文档.txt\n\nn\nn\n\nn\ny\n");
        let (status, stdout) = session.finish();
        assert!(status.success(), "{}", session.text());
        assert!(
            stdout.is_empty(),
            "file output must not also print Markdown"
        );
        let output = std::fs::read_to_string(dir.path().join("custom out/源 文档.txt.md")).unwrap();
        assert!(output.contains("Preserved Unicode 世界"));
        assert!(session.text().contains("OCR: disabled"));
        assert_eq!(
            std::fs::read(dir.path().join("config.json")).unwrap(),
            cfg_bytes
        );
        assert_eq!(
            std::fs::read_dir(dir.path().join("custom out/.markitai/reports"))
                .unwrap()
                .count(),
            1
        );
    }

    #[test]
    fn numbers_directory_package_defaults_to_file_and_uses_one_document_pipeline() {
        let dir = tempfile::tempdir().unwrap();
        let package = dir.path().join("预算.NuMbErS");
        std::fs::create_dir(&package).unwrap();
        let bytes =
            include_bytes!("../../markitai-core/src/formats/numbers/fixtures/test-1.numbers");
        let mut zip = zip::ZipArchive::new(std::io::Cursor::new(bytes)).unwrap();
        for index in 0..zip.len() {
            let mut entry = zip.by_index(index).unwrap();
            let path = package.join(entry.enclosed_name().unwrap());
            if entry.is_dir() {
                std::fs::create_dir_all(path).unwrap();
                continue;
            }
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            let mut data = Vec::new();
            entry.read_to_end(&mut data).unwrap();
            std::fs::write(path, data).unwrap();
        }
        let mut cmd = command(dir.path());
        cmd.args(["预算.NuMbErS", "-I", "--config-json", &base().to_string()]);
        let mut session = Session::new(cmd);
        session.wait_for("Input: 1 file, 2 directory, 3 URL [1]");
        session.send("\n\nout\nn\nn\nn\nn\ny\n");
        let (status, stdout) = session.finish();
        assert!(status.success(), "{}", session.text());
        assert!(stdout.is_empty());
        let markdown = std::fs::read_to_string(dir.path().join("out/预算.NuMbErS.md")).unwrap();
        assert!(markdown.contains("YYY\\_ROW\\_4"));
        assert!(!dir.path().join("out/.markitai/states").exists());
        let reports: Vec<_> = std::fs::read_dir(dir.path().join("out/.markitai/reports"))
            .unwrap()
            .collect();
        assert_eq!(reports.len(), 1);
    }

    #[test]
    fn directory_wizard_uses_the_normal_batch_pipeline_and_cancel_creates_nothing() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("inputs")).unwrap();
        for name in ["a.txt", "b.txt"] {
            std::fs::write(dir.path().join("inputs").join(name), name).unwrap();
        }
        let mut cmd = command(dir.path());
        cmd.args(["-I", "--config-json", &base().to_string()]);
        let mut session = Session::new(cmd);
        session.send("2\ninputs\n\nn\nn\nn\nn\ny\n");
        let (status, _) = session.finish();
        assert!(status.success(), "{}", session.text());
        assert!(dir.path().join("output/a.txt.md").is_file());
        assert!(dir.path().join("output/b.txt.md").is_file());
        assert_eq!(
            std::fs::read_dir(dir.path().join("output/.markitai/reports"))
                .unwrap()
                .count(),
            1
        );
        let mut cmd = command(dir.path());
        cmd.args(["-I", "--config-json", &base().to_string()]);
        let mut session = Session::new(cmd);
        session.send("2\ninputs\ncancelled\nn\nn\nn\nn\nn\n");
        let (status, _) = session.finish();
        assert!(status.success());
        assert!(!dir.path().join("cancelled").exists());
    }

    #[test]
    fn url_wizard_retains_explicit_model_config_and_makes_one_fetch_and_one_pure_request() {
        let dir = tempfile::tempdir().unwrap();
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let mut requests = Vec::new();
            let deadline = Instant::now() + Duration::from_secs(15);
            while requests.len() < 2 {
                let (mut stream, _) = match listener.accept() {
                    Ok(value) => value,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        assert!(Instant::now() < deadline, "Missing mock request");
                        std::thread::sleep(Duration::from_millis(5));
                        continue;
                    }
                    Err(error) => panic!("{error}"),
                };
                stream.set_nonblocking(false).unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let mut request_reader = bounded_fixture_io::Reader::new(
                    &stream,
                    std::time::Instant::now() + Duration::from_secs(5),
                );
                let mut bytes = Vec::new();
                let mut buf = [0; 4096];
                let split = loop {
                    let n = request_reader.read(&mut buf).unwrap();
                    assert!(n > 0);
                    bytes.extend_from_slice(&buf[..n]);
                    assert!(bytes.len() < 1024 * 1024);
                    if let Some(index) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
                        break index + 4;
                    }
                };
                let header = String::from_utf8_lossy(&bytes[..split]).to_string();
                let length = header
                    .lines()
                    .find_map(|line| {
                        line.split_once(':')
                            .filter(|(key, _)| key.eq_ignore_ascii_case("content-length"))
                            .map(|(_, value)| value.trim().parse::<usize>().unwrap())
                    })
                    .unwrap_or(0);
                while bytes.len() < split + length {
                    let n = request_reader.read(&mut buf).unwrap();
                    assert!(n > 0);
                    bytes.extend_from_slice(&buf[..n]);
                }
                let (content_type, body) = if header.starts_with("GET ") {
                    ("text/plain", "A preserved source article with enough words to establish the text extraction route and ensure the wizard retains the explicitly selected model configuration.\n".to_owned())
                } else {
                    let request: Value =
                        serde_json::from_slice(&bytes[split..split + length]).unwrap();
                    assert_eq!(request["model"], "wizard");
                    assert!(request["messages"][1]["content"].is_string());
                    assert!(
                        request["messages"][1]["content"]
                            .as_str()
                            .unwrap()
                            .contains("preserved source article")
                    );
                    ("application/json", json!({"choices":[{"message":{"content":"CLEAN WIZARD RESULT"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":4}}).to_string())
                };
                requests.push(header.lines().next().unwrap().to_owned());
                write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
            }
            requests
        });
        let mut cfg = base();
        cfg["llm"]["model_list"] = json!([{"model_name":"default","litellm_params":{"model":"openai/wizard","api_key":"not-a-real-key","api_base":format!("http://{address}/v1")}}]);
        std::fs::write(
            dir.path().join("explicit.json"),
            serde_json::to_vec(&cfg).unwrap(),
        )
        .unwrap();
        let mut cmd = command(dir.path());
        cmd.args(["-I", "-c", "explicit.json"]);
        let mut session = Session::new(cmd);
        session.send(&format!(
            "3\nhttp://{address}/article.txt\nout\ny\nn\nn\ny\nn\nn\ny\n"
        ));
        let (status, _) = session.finish();
        assert!(status.success(), "{}", session.text());
        let requests = server.join().unwrap();
        assert!(requests[0].starts_with("GET /article.txt "));
        assert!(requests[1].starts_with("POST /v1/chat/completions "));
        let outputs: Vec<_> = std::fs::read_dir(dir.path().join("out"))
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "md"))
            .collect();
        assert_eq!(outputs.len(), 1);
        assert_eq!(
            std::fs::read_to_string(outputs[0].path()).unwrap().trim(),
            "CLEAN WIZARD RESULT"
        );
    }

    #[test]
    fn cancelling_hidden_session_credential_restores_echo_and_does_not_write_configuration() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("source.txt"), "Do not convert\n").unwrap();
        let mut cmd = command(dir.path());
        cmd.args(["-I", "--config-json", &base().to_string()]);
        let mut session = Session::new(cmd);
        session.send("1\nsource.txt\nout\ny\n1\nopenai/session\n\n2\n");
        session.wait_for("API key (hidden; not saved):");
        let mut attributes = std::mem::MaybeUninit::<libc::termios>::uninit();
        assert_eq!(
            unsafe { libc::tcgetattr(session.slave.as_raw_fd(), attributes.as_mut_ptr()) },
            0
        );
        assert_eq!(unsafe { attributes.assume_init() }.c_lflag & libc::ECHO, 0);
        session.send("never-save-this-secret");
        assert_eq!(
            unsafe { libc::kill(session.child.id() as i32, libc::SIGINT) },
            0
        );
        let (status, _) = session.finish();
        assert_eq!(status.code(), Some(0));
        let mut attributes = std::mem::MaybeUninit::<libc::termios>::uninit();
        assert_eq!(
            unsafe { libc::tcgetattr(session.slave.as_raw_fd(), attributes.as_mut_ptr()) },
            0
        );
        assert_ne!(unsafe { attributes.assume_init() }.c_lflag & libc::ECHO, 0);
        assert!(!session.text().contains("never-save-this-secret"));
        assert!(!dir.path().join("out").exists());
        assert!(!dir.path().join("home").exists());
        assert!(!dir.path().join("markitai.json").exists());
    }
}

#[cfg(all(test, unix))]
mod bounded_fixture_io {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/support/bounded_read.rs"
    ));
}
