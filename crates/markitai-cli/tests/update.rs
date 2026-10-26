//! `markitai update`, the new-version notice and the server's web update
//! against a fake release host on this computer: no request leaves it.
//!
//! The debug build counts as a source build (it never checks); these tests
//! make it pose as a release with `MARKITAI_TEST_UPDATE_AS`.
#[path = "support/console.rs"]
mod console;
#[path = "support/interrupt.rs"]
mod interrupt;

use serde_json::{Value, json};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// A fake GitHub release host: `/releases/latest/download/X` redirects to the
/// latest version, `/releases/download/v<version>/<asset>` serves the files.
struct Releases {
    port: u16,
    assets: Arc<Mutex<HashMap<String, Vec<u8>>>>,
    latest: Arc<Mutex<String>>,
    /// Milliseconds every answer waits; `u64::MAX` never answers.
    stall: Arc<Mutex<u64>>,
    seen: Arc<Mutex<Vec<String>>>,
}

impl Releases {
    fn start(latest: &str) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let releases = Self {
            port,
            assets: Arc::default(),
            latest: Arc::new(Mutex::new(latest.into())),
            stall: Arc::default(),
            seen: Arc::default(),
        };
        let (assets, latest, seen, stall) = (
            releases.assets.clone(),
            releases.latest.clone(),
            releases.seen.clone(),
            releases.stall.clone(),
        );
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else { break };
                let (assets, latest, seen, stall) =
                    (assets.clone(), latest.clone(), seen.clone(), stall.clone());
                std::thread::spawn(move || {
                    let wait = *stall.lock().unwrap();
                    if wait == u64::MAX {
                        std::thread::sleep(Duration::from_secs(120));
                        return;
                    }
                    std::thread::sleep(Duration::from_millis(wait));
                    answer(stream, &assets, &latest, &seen)
                });
            }
        });
        releases
    }

    fn base(&self) -> String {
        format!("http://127.0.0.1:{}/releases", self.port)
    }

    fn put(&self, version: &str, name: &str, bytes: Vec<u8>) {
        self.assets
            .lock()
            .unwrap()
            .insert(format!("/releases/download/v{version}/{name}"), bytes);
    }

    /// A release whose archive holds `program` as `markitai`; `corrupt`
    /// lists another digest for it.
    fn publish(&self, version: &str, program: &str, corrupt: bool) {
        let target = target();
        let name = format!("markitai-v{version}-{target}.tar.gz");
        let archive = tar_gz(&[
            ("markitai", program.as_bytes(), 0o755),
            ("LICENSE", b"MIT\n", 0o644),
            ("docs/index.md", format!("# {version}\n").as_bytes(), 0o644),
        ]);
        let digest = if corrupt {
            "0".repeat(64)
        } else {
            hex(&<sha2::Sha256 as sha2::Digest>::digest(&archive))
        };
        self.put(version, &name, archive);
        self.put(
            version,
            "SHA256SUMS",
            format!("{digest}  {name}\n").into_bytes(),
        );
        self.put(
            version,
            "RELEASE-NOTES.md",
            b"Faster tables and a new command.\n\n### Added\n\n- One\n".to_vec(),
        );
        self.put(
            version,
            "RELEASE-NOTES.zh.md",
            "表格更快，并新增一个命令。\n\n### 新增\n\n- 一\n"
                .as_bytes()
                .to_vec(),
        );
    }

    /// A Windows release laid out as the real zip (release_build.py: the
    /// three EXEs, LICENSE, NOTICE): this binary as `markitai.exe`, posed as
    /// `version` by its pose file, which also says whether it fails to start.
    /// `mkai.exe` and `markitai-mcp.exe` are stand-ins: the update only
    /// requires them, and three debug copies would pass the archive limit.
    #[cfg(windows)]
    fn publish_exe(&self, version: &str, fail: bool) {
        use std::io::Write as _;
        let name = format!("markitai-v{version}-{}.zip", target());
        let pose = format!("{version}\n{}", if fail { "fail-start\n" } else { "" });
        let program = std::fs::read(env!("CARGO_BIN_EXE_markitai")).unwrap();
        let mut writer = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
        // Stored: the debug program is large, and this is no test of deflate.
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored)
            .large_file(program.len() > u32::MAX as usize / 2);
        for (entry, bytes) in [
            ("markitai.exe", program.as_slice()),
            ("mkai.exe", b"mkai stand-in".as_slice()),
            ("markitai-mcp.exe", b"markitai-mcp stand-in".as_slice()),
            (POSE, pose.as_bytes()),
            ("LICENSE", b"MIT\n".as_slice()),
            ("NOTICE", b"notices\n".as_slice()),
        ] {
            writer.start_file(entry, options).unwrap();
            writer.write_all(bytes).unwrap();
        }
        let archive = writer.finish().unwrap().into_inner();
        let digest = hex(&<sha2::Sha256 as sha2::Digest>::digest(&archive));
        self.put(version, &name, archive);
        self.put(
            version,
            "SHA256SUMS",
            format!("{digest}  {name}\n").into_bytes(),
        );
        self.put(version, "RELEASE-NOTES.md", b"Faster tables.\n".to_vec());
    }

    fn requests(&self) -> Vec<String> {
        self.seen.lock().unwrap().clone()
    }
}

fn answer(
    stream: TcpStream,
    assets: &Mutex<HashMap<String, Vec<u8>>>,
    latest: &Mutex<String>,
    seen: &Mutex<Vec<String>>,
) {
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut line = String::new();
    if reader.read_line(&mut line).is_err() {
        return;
    }
    let mut agent = String::new();
    loop {
        let mut header = String::new();
        if reader.read_line(&mut header).is_err() || header.trim().is_empty() {
            break;
        }
        if let Some(value) = header.to_ascii_lowercase().strip_prefix("user-agent:") {
            agent = value.trim().to_owned();
        }
    }
    let path = line.split_whitespace().nth(1).unwrap_or("").to_owned();
    seen.lock().unwrap().push(format!("{path} ({agent})"));
    let mut stream = stream;
    let (head, body) = if let Some(asset) = path.strip_prefix("/releases/latest/download/") {
        let version = latest.lock().unwrap().clone();
        (
            format!(
                "HTTP/1.1 302 Found\r\nlocation: https://github.com/Ynewtime/markitai/releases/download/v{version}/{asset}\r\ncontent-length: 0\r\n"
            ),
            Vec::new(),
        )
    } else if let Some(bytes) = assets.lock().unwrap().get(&path) {
        (
            format!("HTTP/1.1 200 OK\r\ncontent-length: {}\r\n", bytes.len()),
            bytes.clone(),
        )
    } else {
        (
            "HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\n".into(),
            Vec::new(),
        )
    };
    let _ = stream.write_all(format!("{head}connection: close\r\n\r\n").as_bytes());
    let _ = stream.write_all(&body);
}

fn target() -> &'static str {
    if cfg!(all(target_os = "linux", target_arch = "aarch64")) {
        "aarch64-unknown-linux-gnu"
    } else if cfg!(target_os = "linux") {
        "x86_64-unknown-linux-gnu"
    } else if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        "aarch64-apple-darwin"
    } else if cfg!(target_os = "macos") {
        "x86_64-apple-darwin"
    } else if cfg!(target_arch = "aarch64") {
        "aarch64-pc-windows-msvc"
    } else {
        "x86_64-pc-windows-msvc"
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn tar_gz(files: &[(&str, &[u8], u32)]) -> Vec<u8> {
    let mut builder = tar::Builder::new(flate2::write::GzEncoder::new(
        Vec::new(),
        flate2::Compression::fast(),
    ));
    for (name, bytes, mode) in files {
        let mut header = tar::Header::new_gnu();
        header.set_size(bytes.len() as u64);
        header.set_mode(*mode);
        builder.append_data(&mut header, name, *bytes).unwrap();
    }
    for link in ["mkai", "markitai-mcp"] {
        let mut header = tar::Header::new_gnu();
        header.set_entry_type(tar::EntryType::Symlink);
        header.set_size(0);
        builder.append_link(&mut header, link, "markitai").unwrap();
    }
    builder.into_inner().unwrap().finish().unwrap()
}

/// An isolated home; `program` is the markitai the commands run.
struct Home {
    root: tempfile::TempDir,
    program: PathBuf,
}

impl Home {
    /// The test binary itself, which no setup script installed.
    fn plain() -> Self {
        let root = tempfile::tempdir().unwrap();
        Self {
            root,
            program: PathBuf::from(env!("CARGO_BIN_EXE_markitai")),
        }
    }

    /// A setup.sh install of `version` whose `bin_dir/markitai` is (a link to)
    /// the test binary.
    #[cfg(unix)]
    fn setup_sh(version: &str) -> Self {
        let root = tempfile::tempdir().unwrap();
        let bin = root.path().join("home/.local/bin");
        let current = root.path().join("home/.local/share/markitai/current");
        std::fs::create_dir_all(&bin).unwrap();
        std::fs::create_dir_all(&current).unwrap();
        std::os::unix::fs::symlink(env!("CARGO_BIN_EXE_markitai"), bin.join("markitai")).unwrap();
        std::os::unix::fs::symlink("markitai", bin.join("mkai")).unwrap();
        std::fs::write(current.join("LICENSE"), "old\n").unwrap();
        std::fs::write(
            current.join("receipt"),
            format!(
                "version={version}\ntarget={}\nbin_dir={}\n",
                target(),
                bin.display()
            ),
        )
        .unwrap();
        Self {
            program: bin.join("markitai"),
            root,
        }
    }

    /// A setup.ps1 install of `version`: a copy of this binary posed by its
    /// pose file, in a folder whose receipt lists both.
    #[cfg(windows)]
    fn setup_ps1(version: &str) -> Self {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("home/AppData/Local/Programs/markitai");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::copy(env!("CARGO_BIN_EXE_markitai"), dir.join("markitai.exe")).unwrap();
        std::fs::write(dir.join(POSE), format!("{version}\n")).unwrap();
        for name in ["mkai.exe", "markitai-mcp.exe", "LICENSE", "NOTICE"] {
            std::fs::write(dir.join(name), "old\n").unwrap();
        }
        std::fs::write(
            dir.join("receipt"),
            format!(
                "version={version}\ntarget={}\nfile=LICENSE\nfile=NOTICE\nfile={POSE}\nfile=markitai-mcp.exe\nfile=markitai.exe\nfile=mkai.exe\n",
                target()
            ),
        )
        .unwrap();
        Self {
            program: dir.join("markitai.exe"),
            root,
        }
    }

    fn path(&self, name: &str) -> PathBuf {
        self.root.path().join(name)
    }

    fn command(&self, releases: &Releases, extra: &[(&str, &str)]) -> Command {
        let mut command = Command::new(&self.program);
        console::detached(&mut command);
        console::english(command.env_clear());
        for name in ["PATH", "SYSTEMROOT", "TMPDIR"] {
            if let Some(value) = std::env::var_os(name) {
                command.env(name, value);
            }
        }
        command
            .current_dir(self.root.path())
            .env("HOME", self.path("home"))
            .env("USERPROFILE", self.path("home"))
            .env("MARKITAI_HOME", self.path("mh"))
            .env("MARKITAI_TEST_UPDATE_AS", "2.1.0")
            .env("MARKITAI_DOWNLOAD_URL", releases.base())
            .envs(extra.iter().copied())
            .stdin(Stdio::null());
        command
    }

    fn run(&self, releases: &Releases, args: &[&str], extra: &[(&str, &str)]) -> Output {
        self.command(releases, extra).args(args).output().unwrap()
    }

    fn cache(&self) -> Value {
        serde_json::from_slice(&std::fs::read(self.path("mh/update-check.json")).unwrap()).unwrap()
    }
}

fn text(output: &Output) -> (String, String, i32) {
    (
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
        output.status.code().unwrap_or(-1),
    )
}

/// A server run from the setup install, with its token and port as it printed them.
struct Serve {
    child: std::process::Child,
    port: u16,
    token: String,
    log: Arc<Mutex<String>>,
}

/// A debug build's pose file (`update::TEST_POSE`).
#[cfg(windows)]
const POSE: &str = "markitai-test-pose";

#[cfg(unix)]
const TOKEN: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

impl Serve {
    /// On a free port with a known token.
    #[cfg(unix)]
    fn start(home: &Home, releases: &Releases) -> Self {
        Self::start_with(
            home,
            releases,
            &[("MARKITAI_SERVE_TOKEN", TOKEN)],
            Some(free_port()),
        )
    }

    /// With `extra` environment, on `port` (`None`: any).
    fn start_with(
        home: &Home,
        releases: &Releases,
        extra: &[(&str, &str)],
        port: Option<u16>,
    ) -> Self {
        let mut command = home.command(releases, extra);
        // Windows: a process group of its own on this console, for Ctrl-Break.
        interrupt::interruptible(&mut command);
        command
            .args([
                "serve",
                "--port",
                &port.unwrap_or(0).to_string(),
                "--no-open",
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::piped());
        let mut child = command.spawn().unwrap();
        let log = Arc::new(Mutex::new(String::new()));
        let (copy, stderr) = (log.clone(), child.stderr.take().unwrap());
        std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                copy.lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .push_str(&format!("{line}\n"));
            }
        });
        let started = Instant::now();
        let after = |text: &str, marker: &str, valid: fn(char) -> bool| {
            text.split(marker)
                .nth(1)
                .and_then(|rest| rest.split(|c: char| !valid(c)).next())
                .filter(|found| !found.is_empty())
                .map(str::to_owned)
        };
        let (port, token) = loop {
            let text = log.lock().unwrap().clone();
            let port = after(&text, "http://127.0.0.1:", |c| c.is_ascii_digit());
            let token = after(&text, "Access token: ", |c| c.is_ascii_alphanumeric());
            if let (Some(port), Some(token)) = (port, token) {
                break (port.parse().unwrap(), token);
            }
            assert!(started.elapsed() < Duration::from_secs(60), "{text}");
            std::thread::sleep(Duration::from_millis(20));
        };
        Self {
            child,
            port,
            token,
            log,
        }
    }

    fn call(&self, method: &str, path: &str, body: Option<Value>) -> Option<(u16, Value)> {
        let mut stream = TcpStream::connect(("127.0.0.1", self.port)).ok()?;
        stream
            .set_read_timeout(Some(Duration::from_secs(10)))
            .ok()?;
        let body = body.map(|body| body.to_string()).unwrap_or_default();
        let request = format!(
            "{method} {path} HTTP/1.1\r\nhost: 127.0.0.1:{}\r\nauthorization: Bearer {}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
            self.port,
            self.token,
            body.len()
        );
        stream.write_all(request.as_bytes()).ok()?;
        let mut reply = String::new();
        std::io::Read::read_to_string(&mut stream, &mut reply).ok()?;
        let status = reply.split_whitespace().nth(1)?.parse().ok()?;
        let body = reply.split_once("\r\n\r\n")?.1;
        Some((status, serde_json::from_str(body).unwrap_or(Value::Null)))
    }

    /// `GET /api/update` until `done` holds.
    fn wait(&self, label: &str, done: impl Fn(&Value) -> bool) -> Value {
        let started = Instant::now();
        loop {
            if let Some((200, view)) = self.call("GET", "/api/update", None)
                && done(&view)
            {
                return view;
            }
            assert!(
                started.elapsed() < Duration::from_secs(90),
                "{label}: {}",
                self.log.lock().unwrap_or_else(|error| error.into_inner())
            );
            std::thread::sleep(Duration::from_millis(50));
        }
    }
}

impl Drop for Serve {
    /// As Ctrl-C would; a Windows server's restarted child gets it too.
    fn drop(&mut self) {
        interrupt::interrupt(&self.child);
        let started = Instant::now();
        while started.elapsed() < Duration::from_secs(30) {
            if matches!(self.child.try_wait(), Ok(Some(_))) {
                return;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

#[test]
fn check_answers_with_its_exit_status_and_json() {
    let releases = Releases::start("2.2.0");
    releases.publish("2.2.0", "#!/bin/sh\n", false);
    let home = Home::plain();
    let (_, stderr, code) = text(&home.run(&releases, &["update", "--check"], &[]));
    assert_eq!(code, 10, "{stderr}");
    assert!(
        stderr.contains("New version 2.2.0 (current 2.1.0): markitai update"),
        "{stderr}"
    );
    let (stdout, _, code) = text(&home.run(&releases, &["update", "--check", "--json"], &[]));
    assert_eq!(code, 10);
    let answer: Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(
        (
            &answer["current"],
            &answer["latest"],
            &answer["newer"],
            &answer["action"]
        ),
        (
            &json!("2.1.0"),
            &json!("2.2.0"),
            &json!(true),
            &json!("none")
        )
    );
    assert_eq!(
        answer["notes_url"],
        format!("{}/tag/v2.2.0", releases.base())
    );
    // What was learnt is kept for the notice, with the release's description.
    assert_eq!(home.cache()["latest"], "2.2.0");
    assert_eq!(home.cache()["lead"], "Faster tables and a new command.");
    assert_eq!(home.cache()["lead_zh"], "表格更快，并新增一个命令。");
    // The same release: up to date.
    let (_, stderr, code) = text(&home.run(
        &releases,
        &["update", "--check"],
        &[("MARKITAI_TEST_UPDATE_AS", "2.2.0")],
    ));
    assert_eq!((code, stderr.trim()), (0, "✓ markitai 2.2.0 is up to date"));
    // Every request names itself `markitai` and nothing else.
    assert!(
        releases
            .requests()
            .iter()
            .all(|request| request.ends_with("(markitai)")),
        "{:?}",
        releases.requests()
    );
}

#[test]
fn a_failed_check_is_exit_one_and_a_source_build_never_asks() {
    let closed = TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap();
    let releases = Releases::start("2.2.0");
    let home = Home::plain();
    let offline = format!("http://{closed}/releases");
    let (_, stderr, code) = text(&home.run(
        &releases,
        &["update", "--check"],
        &[("MARKITAI_DOWNLOAD_URL", &offline)],
    ));
    assert_eq!(code, 1, "{stderr}");
    assert!(stderr.contains("Can't reach 127.0.0.1"), "{stderr}");
    // The failure starts the backoff.
    assert_eq!(home.cache()["failures"], 1);
    let (_, stderr, code) = text(&home.run(
        &releases,
        &["update"],
        &[("MARKITAI_LANG", "zh"), ("MARKITAI_DOWNLOAD_URL", &offline)],
    ));
    assert_eq!(code, 1);
    assert!(stderr.contains("✗ 无法连接 127.0.0.1"), "{stderr}");
    assert!(
        stderr.contains("• 未做改动") && stderr.contains("✗ 更新失败"),
        "{stderr}"
    );
    // Without posing as a release, the debug build is a source build.
    let mut command = home.command(&releases, &[]);
    command.env_remove("MARKITAI_TEST_UPDATE_AS");
    let (_, stderr, code) = text(&command.args(["update"]).output().unwrap());
    assert_eq!(code, 0);
    assert!(stderr.contains("Source build ("), "{stderr}");
    assert!(releases.requests().is_empty(), "{:?}", releases.requests());
    // A plain http base elsewhere than this computer is refused.
    let (_, stderr, code) = text(&home.run(
        &releases,
        &["update", "--check"],
        &[("MARKITAI_DOWNLOAD_URL", "http://mirror.example/releases")],
    ));
    assert_eq!(code, 1);
    assert!(stderr.contains("not HTTPS"), "{stderr}");
}

#[test]
fn a_copy_no_setup_script_installed_gets_the_setup_command_and_changes_nothing() {
    let releases = Releases::start("2.2.0");
    releases.publish("2.2.0", "#!/bin/sh\n", false);
    let home = Home::plain();
    let (_, stderr, code) = text(&home.run(&releases, &["update"], &[]));
    assert_eq!(code, 10, "{stderr}");
    assert!(stderr.contains("◆ markitai 2.1.0 → 2.2.0"), "{stderr}");
    assert!(
        stderr.contains("  Faster tables and a new command."),
        "{stderr}"
    );
    assert!(
        stderr.contains("! Can't tell how this copy was installed:"),
        "{stderr}"
    );
    assert!(stderr.contains("markitai.dev/setup."), "{stderr}");
    // Only the check and the notes were asked for: no archive.
    assert!(
        releases
            .requests()
            .iter()
            .all(|request| !request.contains(".tar.gz") && !request.contains(".zip")),
        "{:?}",
        releases.requests()
    );
    // In Chinese, the Chinese description (from the cache now).
    let (_, stderr, code) = text(&home.run(&releases, &["update"], &[("MARKITAI_LANG", "zh")]));
    assert_eq!(code, 10, "{stderr}");
    assert!(stderr.contains("  表格更快，并新增一个命令。"), "{stderr}");
    let (_, stderr, code) = text(&home.run(&releases, &["update", "--version", "1.2.0"], &[]));
    assert_eq!(code, 1);
    assert!(stderr.contains("Python package"), "{stderr}");
}

#[cfg(unix)]
mod unix {
    use super::*;
    use std::io::Read;

    #[test]
    fn a_setup_install_is_replaced_in_place_and_its_receipt_rewritten() {
        let releases = Releases::start("2.2.0");
        releases.publish("2.2.0", "#!/bin/sh\necho \"markitai 2.2.0\"\n", false);
        let home = Home::setup_sh("2.1.0");
        let (stdout, stderr, code) = text(&home.run(&releases, &["update", "--json"], &[]));
        assert_eq!(code, 0, "{stderr}");
        let answer: Value = serde_json::from_str(&stdout).unwrap();
        assert_eq!(
            (&answer["method"], &answer["action"]),
            (&json!("setup"), &json!("updated"))
        );
        let program = home.path("home/.local/bin/markitai");
        let version = Command::new(&program).arg("--version").output().unwrap();
        assert_eq!(
            String::from_utf8_lossy(&version.stdout).trim(),
            "markitai 2.2.0"
        );
        let current = home.path("home/.local/share/markitai/current");
        assert_eq!(
            std::fs::read_to_string(current.join("LICENSE")).unwrap(),
            "MIT\n"
        );
        let receipt = std::fs::read_to_string(current.join("receipt")).unwrap();
        for line in [
            "version=2.2.0".to_owned(),
            format!("base={}", releases.base()),
            "installer=setup.sh".into(),
            "format=1".into(),
        ] {
            assert!(
                receipt.lines().any(|found| found == line),
                "{line}: {receipt}"
            );
        }
        // The links stay, and nothing is left beside the program.
        let mut names: Vec<String> = std::fs::read_dir(home.path("home/.local/bin"))
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        assert_eq!(names, ["markitai", "markitai-mcp", "mkai"]);
    }

    #[test]
    fn a_mismatching_archive_changes_nothing() {
        let releases = Releases::start("2.2.0");
        releases.publish("2.2.0", "#!/bin/sh\necho \"markitai 2.2.0\"\n", true);
        let home = Home::setup_sh("2.1.0");
        let (_, stderr, code) = text(&home.run(&releases, &["update"], &[("MARKITAI_LANG", "zh")]));
        assert_eq!(code, 1, "{stderr}");
        assert!(
            stderr.contains("✗ SHA-256 不匹配，已删除下载的文件"),
            "{stderr}"
        );
        assert!(stderr.contains("• 未做改动"), "{stderr}");
        let program = home.path("home/.local/bin/markitai");
        assert!(
            std::fs::symlink_metadata(&program)
                .unwrap()
                .file_type()
                .is_symlink()
        );
        let receipt =
            std::fs::read_to_string(home.path("home/.local/share/markitai/current/receipt"))
                .unwrap();
        assert!(receipt.starts_with("version=2.1.0\n"), "{receipt}");
        assert_eq!(
            std::fs::read_dir(home.path("home/.local/bin"))
                .unwrap()
                .count(),
            2
        );
    }

    #[test]
    fn a_new_program_that_does_not_start_is_not_installed() {
        let releases = Releases::start("2.2.0");
        releases.publish("2.2.0", "#!/bin/sh\necho broken >&2\nexit 3\n", false);
        let home = Home::setup_sh("2.1.0");
        let (_, stderr, code) = text(&home.run(&releases, &["update"], &[]));
        assert_eq!(code, 1, "{stderr}");
        assert!(
            stderr.contains("The new version didn't start: broken"),
            "{stderr}"
        );
        assert!(stderr.contains("• Nothing changed"), "{stderr}");
        let program = home.path("home/.local/bin/markitai");
        assert!(
            std::fs::symlink_metadata(&program)
                .unwrap()
                .file_type()
                .is_symlink()
        );
    }

    #[test]
    fn a_downgrade_without_a_terminal_needs_yes() {
        let releases = Releases::start("2.2.0");
        releases.publish("2.0.1", "#!/bin/sh\necho \"markitai 2.0.1\"\n", false);
        let home = Home::setup_sh("2.1.0");
        let (_, stderr, code) = text(&home.run(&releases, &["update", "--version", "2.0.1"], &[]));
        assert_eq!(code, 2, "{stderr}");
        assert!(stderr.contains("Needs confirmation"), "{stderr}");
        let (_, stderr, code) =
            text(&home.run(&releases, &["update", "--version", "2.0.1", "--yes"], &[]));
        assert_eq!(code, 0, "{stderr}");
        assert!(stderr.contains("✓ Downgraded to 2.0.1"), "{stderr}");
    }

    /// Run with stderr on a pseudo-terminal; what it wrote.
    fn on_a_terminal(command: &mut Command) -> (String, i32) {
        use std::os::fd::{FromRawFd, OwnedFd};
        let (mut master, slave) = unsafe {
            let (mut master, mut slave) = (0, 0);
            let mut window = libc::winsize {
                ws_row: 24,
                ws_col: 100,
                ws_xpixel: 0,
                ws_ypixel: 0,
            };
            let window: *mut libc::winsize = &mut window;
            assert_eq!(
                libc::openpty(
                    &mut master,
                    &mut slave,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    window
                ),
                0
            );
            // Close-on-exec: only the run's stderr may hold the terminal, not
            // a background check it starts.
            libc::fcntl(master, libc::F_SETFD, libc::FD_CLOEXEC);
            libc::fcntl(slave, libc::F_SETFD, libc::FD_CLOEXEC);
            (
                std::fs::File::from(OwnedFd::from_raw_fd(master)),
                OwnedFd::from_raw_fd(slave),
            )
        };
        command.stdout(Stdio::null()).stderr(Stdio::from(slave));
        let mut child = command.spawn().unwrap();
        // Replacing the command's copy of the slave closes it, so the
        // master reads to the end once the child exits.
        command.stderr(Stdio::null());
        let reader = std::thread::spawn(move || {
            let mut bytes = Vec::new();
            let mut buffer = [0_u8; 4096];
            while let Ok(count) = master.read(&mut buffer) {
                if count == 0 {
                    break;
                }
                bytes.extend(&buffer[..count]);
            }
            bytes
        });
        let status = child.wait().unwrap();
        (
            String::from_utf8_lossy(&reader.join().unwrap()).into_owned(),
            status.code().unwrap_or(-1),
        )
    }

    #[test]
    fn the_notice_follows_a_run_on_a_terminal_once_a_day_and_never_elsewhere() {
        let releases = Releases::start("2.2.0");
        let home = Home::plain();
        std::fs::write(home.path("a.txt"), "Words.\n").unwrap();
        // Without colour, so the lines read as plain text.
        let convert = |extra: &[(&str, &str)]| {
            let mut command = home.command(&releases, extra);
            command.env("NO_COLOR", "1").args(["a.txt", "-o", "out"]);
            command
        };
        // Off a terminal: no check, no notice.
        let (_, stderr, code) = text(&convert(&[]).output().unwrap());
        assert_eq!(code, 0, "{stderr}");
        assert!(releases.requests().is_empty());
        // On one: checked in the background (the run does not wait), then
        // the next run names the release.
        let notice = "• 有新版本 2.2.0（当前 2.1.0）：";
        let (first, code) = on_a_terminal(&mut convert(&[("TERM", "xterm")]));
        assert_eq!(code, 0, "{first}");
        wait_for_cache(&home, |cache| cache["latest"] == "2.2.0");
        let (shown, code) = on_a_terminal(&mut convert(&[
            ("MARKITAI_LANG", "zh"),
            ("TERM", "xterm"),
            ("LANG", "zh_CN.UTF-8"),
        ]));
        assert_eq!(code, 0, "{shown}");
        assert!(
            shown.contains(notice) && shown.contains("markitai update"),
            "{shown:?}"
        );
        assert_eq!(releases.requests().len(), 2, "{:?}", releases.requests());
        // Not again today, and no new request either.
        let (again, _) = on_a_terminal(&mut convert(&[
            ("MARKITAI_LANG", "zh"),
            ("TERM", "xterm"),
            ("LANG", "zh_CN.UTF-8"),
        ]));
        assert!(!again.contains("有新版本"), "{again:?}");
        assert_eq!(releases.requests().len(), 2);
        // A newer release is news at once, unless checking is off, quiet, in CI or JSON.
        let mut cache = home.cache();
        cache["latest"] = json!("2.3.0");
        std::fs::write(home.path("mh/update-check.json"), cache.to_string()).unwrap();
        for extra in [
            vec![("MARKITAI_NO_UPDATE_CHECK", "1")],
            vec![("CI", "true")],
        ] {
            let (quiet, _) = on_a_terminal(&mut convert(&extra));
            assert!(!quiet.contains("New version"), "{extra:?}: {quiet:?}");
        }
        for args in [
            &["a.txt", "-o", "out", "-q"][..],
            &["a.txt", "-o", "out", "--json"],
            &["a.txt"],
        ] {
            let mut command = home.command(&releases, &[]);
            command.args(args);
            let (quiet, _) = on_a_terminal(&mut command);
            assert!(!quiet.contains("New version"), "{args:?}: {quiet:?}");
        }
        std::fs::write(home.path("cfg.json"), r#"{"update":{"check":false}}"#).unwrap();
        let (off, _) = on_a_terminal(convert(&[]).args(["-c", "cfg.json"]));
        assert!(!off.contains("New version"), "{off:?}");
        let (shown, _) = on_a_terminal(&mut convert(&[]));
        assert!(
            shown.contains("• New version 2.3.0 (current 2.1.0): markitai update"),
            "{shown:?}"
        );
    }

    fn wait_for_cache(home: &Home, done: impl Fn(&Value) -> bool) -> Value {
        let started = std::time::Instant::now();
        loop {
            if let Ok(bytes) = std::fs::read(home.path("mh/update-check.json"))
                && let Ok(cache) = serde_json::from_slice::<Value>(&bytes)
                && done(&cache)
            {
                return cache;
            }
            assert!(
                started.elapsed() < Duration::from_secs(20),
                "the background check never wrote the cache"
            );
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    /// One conversion on a terminal, and how long it took.
    fn timed(home: &Home, releases: &Releases) -> (String, Duration) {
        let mut command = home.command(releases, &[("TERM", "xterm"), ("NO_COLOR", "1")]);
        command.args(["a.txt", "-o", "out"]);
        let started = std::time::Instant::now();
        let (text, code) = on_a_terminal(&mut command);
        assert_eq!(code, 0, "{text}");
        (text, started.elapsed())
    }

    #[test]
    fn a_release_host_that_never_answers_costs_a_run_nothing_and_backs_off() {
        let releases = Releases::start("2.2.0");
        *releases.stall.lock().unwrap() = u64::MAX;
        let home = Home::plain();
        std::fs::write(home.path("a.txt"), "Words.\n").unwrap();
        let (_, quiet) = timed(&home, &releases);
        let (_, baseline) = {
            let mut off = home.command(
                &releases,
                &[("TERM", "xterm"), ("MARKITAI_NO_UPDATE_CHECK", "1")],
            );
            off.args(["a.txt", "-o", "out"]);
            let started = std::time::Instant::now();
            let result = on_a_terminal(&mut off);
            (result.0, started.elapsed())
        };
        assert!(
            quiet < baseline + Duration::from_millis(900),
            "{quiet:?} vs {baseline:?}"
        );
        // The attempt is recorded before the request: the next run does not ask again.
        let cache = wait_for_cache(&home, |cache| cache["retry_at"].is_i64());
        assert_eq!(cache["failures"], 1);
        let asked = releases.requests().len();
        let (_, again) = timed(&home, &releases);
        assert!(again < baseline + Duration::from_millis(900), "{again:?}");
        std::thread::sleep(Duration::from_millis(300));
        assert_eq!(
            releases.requests().len(),
            asked,
            "{:?}",
            releases.requests()
        );
    }

    #[test]
    fn a_slow_release_host_is_read_after_the_run_and_named_by_the_next() {
        let releases = Releases::start("2.2.0");
        releases.publish("2.2.0", "#!/bin/sh\n", false);
        *releases.stall.lock().unwrap() = 1200;
        let home = Home::plain();
        std::fs::write(home.path("a.txt"), "Words.\n").unwrap();
        let (first, took) = timed(&home, &releases);
        assert!(
            took < Duration::from_millis(1200),
            "the run waited for the check: {took:?}"
        );
        assert!(!first.contains("New version"), "{first:?}");
        // The version is saved as soon as it is known, before the description.
        let cache = wait_for_cache(&home, |cache| cache["latest"] == "2.2.0");
        assert_eq!(cache["failures"], 0);
        let (second, took) = timed(&home, &releases);
        assert!(took < Duration::from_millis(1200), "{took:?}");
        assert!(
            second.contains("New version 2.2.0 (current 2.1.0)"),
            "{second:?}"
        );
    }

    /// The new release's program: this binary posing as 2.2.0.
    pub(super) fn posing_program(extra: &str) -> String {
        format!(
            "#!/bin/sh\n[ \"$1\" = --version ] && {{ echo \"markitai 2.2.0\"; exit 0; }}\nMARKITAI_TEST_UPDATE_AS=2.2.0 {extra} exec '{}' \"$@\"\n",
            env!("CARGO_BIN_EXE_markitai")
        )
    }

    #[test]
    fn the_workbench_updates_a_loopback_server_which_restarts_in_place() {
        let releases = Releases::start("2.2.0");
        releases.publish("2.2.0", &posing_program(""), false);
        let home = Home::setup_sh("2.1.0");
        let server = Serve::start(&home, &releases);
        let pid = server.child.id();
        let view = server.wait("the daily check", |view| view["latest"] == "2.2.0");
        assert_eq!(
            (&view["can_apply"], &view["method"]),
            (&json!(true), &json!("setup"))
        );
        // The page cannot pick another version.
        let (status, refused) = server
            .call("POST", "/api/update", Some(json!({"version": "2.0.1"})))
            .unwrap();
        assert_eq!(
            (status, refused["reason"].as_str()),
            (409, Some("version_changed"))
        );
        let (status, started) = server
            .call("POST", "/api/update", Some(json!({"version": "2.2.0"})))
            .unwrap();
        assert_eq!(
            (status, started["state"].as_str()),
            (202, Some("downloading")),
            "{started}"
        );
        let view = server.wait("the restart", |view| view["updated_from"] == "2.1.0");
        assert_eq!(
            (&view["current"], &view["state"]),
            (&json!("2.2.0"), &json!("idle"))
        );
        // The same process, the same port and token; the old files are gone.
        assert_eq!(server.child.id(), pid);
        assert!(
            home.path("home/.local/bin")
                .read_dir()
                .unwrap()
                .all(|entry| {
                    !entry
                        .unwrap()
                        .file_name()
                        .to_string_lossy()
                        .starts_with(".markitai")
                })
        );
        assert!(!home.path("mh/update-pending.json").exists());
        let receipt =
            std::fs::read_to_string(home.path("home/.local/share/markitai/current/receipt"))
                .unwrap();
        assert!(receipt.starts_with("version=2.2.0\n"), "{receipt}");
        let log = server.log.lock().unwrap().clone();
        for line in [
            "→ Updating to 2.2.0 (from the web workbench)",
            "✓ Downloaded and verified 2.2.0",
            "→ Restarting",
            "◆ Markitai 2.2.0 restarted, listening on http://127.0.0.1:",
            "✓ Updated from 2.1.0; the access token is unchanged",
        ] {
            assert!(log.contains(line), "{line}: {log}");
        }
    }

    #[test]
    fn the_workbench_refuses_a_release_whose_program_is_another_version() {
        let releases = Releases::start("2.2.0");
        releases.publish("2.2.0", "#!/bin/sh\necho 'markitai 2.3.0'\n", false);
        let home = Home::setup_sh("2.1.0");
        let server = Serve::start(&home, &releases);
        server.wait("the daily check", |view| view["latest"] == "2.2.0");
        server
            .call("POST", "/api/update", Some(json!({"version": "2.2.0"})))
            .unwrap();
        let view = server.wait("the failure", |view| view["state"] == "failed");
        assert_eq!(view["error"]["code"], "start_failed", "{view}");
        assert_eq!(view["error"]["rolled_back"], false, "{view}");
        let reason = view["error"]["reason"].as_str().unwrap();
        assert!(reason.contains("it reports markitai 2.3.0"), "{reason}");
        assert!(
            std::fs::symlink_metadata(home.path("home/.local/bin/markitai"))
                .unwrap()
                .file_type()
                .is_symlink()
        );
    }

    #[test]
    fn a_new_version_that_fails_before_it_is_ready_puts_the_old_one_back() {
        let releases = Releases::start("2.2.0");
        releases.publish(
            "2.2.0",
            &posing_program("MARKITAI_TEST_UPDATE_FAIL_START=1"),
            false,
        );
        let home = Home::setup_sh("2.1.0");
        let server = Serve::start(&home, &releases);
        server.wait("the daily check", |view| view["latest"] == "2.2.0");
        server
            .call("POST", "/api/update", Some(json!({"version": "2.2.0"})))
            .unwrap();
        // The old version serves again and says why.
        let view = server.wait("the rollback", |view| {
            view["error"]["code"] == "start_failed"
        });
        assert_eq!(view["current"], "2.1.0");
        assert_eq!(view["error"]["rolled_back"], true);
        let program = home.path("home/.local/bin/markitai");
        assert!(
            std::fs::symlink_metadata(&program)
                .unwrap()
                .file_type()
                .is_symlink()
        );
        let receipt =
            std::fs::read_to_string(home.path("home/.local/share/markitai/current/receipt"))
                .unwrap();
        assert!(receipt.starts_with("version=2.1.0\n"), "{receipt}");
        let log = server.log.lock().unwrap().clone();
        for line in [
            "✗ 2.2.0 didn't start:",
            "→ Back on 2.1.0",
            "! The update to 2.2.0 failed; still 2.1.0",
        ] {
            assert!(log.contains(line), "{line}: {log}");
        }
    }
}

/// `markitai update` restarting the server it finds (R2). Windows installs
/// copies of this binary, each posed by its pose file; on Unix the setup.sh
/// install and a posing script stand in for them. Both versions are newer
/// than 2.1.0, so both take a handover.
mod restart_by_cli {
    use super::*;

    const OLD: &[(&str, &str)] = &[("MARKITAI_TEST_UPDATE_AS", "2.1.1")];

    /// A setup install of 2.1.1 and a release 2.2.0 that fails before it is
    /// ready when `fail`.
    fn installed(releases: &Releases, fail: bool) -> Home {
        #[cfg(unix)]
        {
            let extra = if fail {
                "MARKITAI_TEST_UPDATE_FAIL_START=1"
            } else {
                ""
            };
            releases.publish("2.2.0", &unix::posing_program(extra), false);
            Home::setup_sh("2.1.1")
        }
        #[cfg(windows)]
        {
            releases.publish_exe("2.2.0", fail);
            Home::setup_ps1("2.1.1")
        }
    }

    /// Unix keeps the socket, so any port stays; Windows binds the same
    /// port number again.
    fn port() -> Option<u16> {
        if cfg!(windows) {
            Some(free_port())
        } else {
            None
        }
    }

    /// `markitai update`, answering Enter (yes) at the restart prompt.
    fn update(home: &Home, releases: &Releases) -> (String, i32) {
        let mut command = home.command(
            releases,
            &[
                ("MARKITAI_TEST_UPDATE_AS", "2.1.1"),
                ("MARKITAI_TEST_UPDATE_INTERACTIVE", "1"),
            ],
        );
        let mut child = command
            .arg("update")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child.stdin.take().unwrap().write_all(b"\n").unwrap();
        let output = child.wait_with_output().unwrap();
        let (stdout, stderr, code) = text(&output);
        (format!("{stdout}{stderr}"), code)
    }

    #[test]
    fn markitai_update_restarts_the_server_on_its_port_with_its_token() {
        let releases = Releases::start("2.2.0");
        let home = installed(&releases, false);
        // The token is the server's own (none in the environment).
        let server = Serve::start_with(&home, &releases, OLD, port());
        let (said, code) = update(&home, &releases);
        assert_eq!(code, 0, "{said}");
        assert!(said.contains("Restart it now? [Y/n]"), "{said}");
        assert!(said.contains("markitai serve restarted: 2.2.0"), "{said}");
        // The same port and token answer, from the new version.
        let view = server.wait("the restart", |view| view["current"] == "2.2.0");
        assert_eq!(view["updated_from"], "2.1.1", "{view}");
        assert!(!home.path("mh/update-pending.json").exists());
    }

    #[test]
    fn a_new_version_that_does_not_start_leaves_the_old_one_serving() {
        let releases = Releases::start("2.2.0");
        let home = installed(&releases, true);
        let server = Serve::start_with(&home, &releases, OLD, port());
        let (said, code) = update(&home, &releases);
        assert_eq!(code, 1, "{said}");
        // The failure is the new version's start, nothing earlier: else
        // there is no rollback to wait for.
        assert!(said.contains("The new version didn't start"), "{said}");
        assert!(!said.contains("came back as"), "{said}");
        // The old version answers on the same port with the same token.
        let view = server.wait("the rollback", |view| {
            view["current"] == "2.1.1" && view["error"]["code"] == "start_failed"
        });
        assert_eq!(view["error"]["rolled_back"], true, "{view}");
        #[cfg(windows)]
        {
            let dir = home.program.parent().unwrap();
            assert_eq!(std::fs::read_to_string(dir.join(POSE)).unwrap(), "2.1.1\n");
            let receipt = std::fs::read_to_string(dir.join("receipt")).unwrap();
            assert!(receipt.starts_with("version=2.1.1\n"), "{receipt}");
        }
        #[cfg(unix)]
        assert!(
            std::fs::symlink_metadata(&home.program)
                .unwrap()
                .file_type()
                .is_symlink()
        );
    }
}
