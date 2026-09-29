use super::*;
use std::io::Cursor;

fn manifest(url: &str) -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!({"channels":{"Stable":{"version":"153.0.8000.1","downloads":{"chrome-headless-shell":[{"platform":"mac-arm64","url":url}]}}}})).unwrap()
}

fn archive(entries: &[(&str, &[u8], u32)]) -> Vec<u8> {
    let mut zip = zip::ZipWriter::new(Cursor::new(Vec::new()));
    for (name, bytes, mode) in entries {
        let options = zip::write::SimpleFileOptions::default().unix_permissions(*mode);
        zip.start_file(*name, options).unwrap();
        zip.write_all(bytes).unwrap();
    }
    zip.finish().unwrap().into_inner()
}

const EXE: &str = "chrome-headless-shell-mac-arm64/chrome-headless-shell";

#[test]
fn manifest_accepts_only_exact_official_stable_asset() {
    let official = "https://storage.googleapis.com/chrome-for-testing-public/153.0.8000.1/mac-arm64/chrome-headless-shell-mac-arm64.zip";
    assert_eq!(
        select(&manifest(official), "mac-arm64").unwrap().url,
        official
    );
    for unsafe_url in [
        "http://storage.googleapis.com/archive.zip",
        "https://example.invalid/archive.zip",
        &format!("{official}?redirect=1"),
    ] {
        assert!(select(&manifest(unsafe_url), "mac-arm64").is_err());
    }
    assert!(select(&manifest(official), "linux64").is_err());
    let mut value: Value = serde_json::from_slice(&manifest(official)).unwrap();
    let row = value["channels"]["Stable"]["downloads"]["chrome-headless-shell"][0].clone();
    value["channels"]["Stable"]["downloads"]["chrome-headless-shell"]
        .as_array_mut()
        .unwrap()
        .push(row);
    assert!(select(&serde_json::to_vec(&value).unwrap(), "mac-arm64").is_err());
    for value in [
        "../../escape",
        "1.2.3",
        "1.2.3.4294967296",
        "1.2..4",
        "1.2.3.4/other",
    ] {
        assert!(!version_valid(value));
    }
}

#[test]
fn zip_retains_executable_and_rejects_paths_collisions_and_expansion() {
    let dir = tempfile::tempdir().unwrap();
    let bytes = archive(&[(EXE, b"browser", 0o755)]);
    extract(Cursor::new(bytes.clone()), dir.path(), "mac-arm64", 7).unwrap();
    assert_eq!(fs::read(dir.path().join(EXE)).unwrap(), b"browser");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            fs::metadata(dir.path().join(EXE))
                .unwrap()
                .permissions()
                .mode()
                & 0o7777,
            0o700
        );
    }
    assert!(extract(Cursor::new(bytes), dir.path(), "mac-arm64", 7).is_err());
    for entries in [
        vec![("../escape", &b"bad"[..], 0o600)],
        vec![("other-root/browser", &b"bad"[..], 0o700)],
        vec![("chrome-headless-shell-mac-arm64/C:bad", &b"bad"[..], 0o600)],
        vec![
            (EXE, &b"browser"[..], 0o700),
            (
                "chrome-headless-shell-mac-arm64/CHROME-HEADLESS-SHELL",
                &b"bad"[..],
                0o700,
            ),
        ],
    ] {
        let dir = tempfile::tempdir().unwrap();
        assert!(extract(Cursor::new(archive(&entries)), dir.path(), "mac-arm64", 100).is_err());
    }
    let dir = tempfile::tempdir().unwrap();
    assert!(
        extract(
            Cursor::new(archive(&[(EXE, b"browser", 0o700)])),
            dir.path(),
            "mac-arm64",
            6
        )
        .is_err()
    );
}

#[test]
fn links_are_rejected_before_they_can_redirect_extraction() {
    let mut zip = zip::ZipWriter::new(Cursor::new(Vec::new()));
    zip.add_symlink(
        EXE,
        "../../outside",
        zip::write::SimpleFileOptions::default(),
    )
    .unwrap();
    let dir = tempfile::tempdir().unwrap();
    assert!(
        extract(
            Cursor::new(zip.finish().unwrap().into_inner()),
            dir.path(),
            "mac-arm64",
            100
        )
        .is_err()
    );
}

#[test]
fn installation_lock_excludes_concurrent_writers_and_releases_on_drop() {
    let dir = tempfile::tempdir().unwrap();
    let first = lock(dir.path()).unwrap();
    assert!(lock(dir.path()).is_err());
    drop(first);
    assert!(lock(dir.path()).is_ok());
}

fn installed_fixture(root: &Path) -> PathBuf {
    let receipt = Receipt {
        schema: 1,
        version: "153.0.8000.1".into(),
        platform: "mac-arm64".into(),
        directory: "headless-153.0.8000.1-mac-arm64-test".into(),
        archive_sha256: "test-only".into(),
        executable_sha256: "test-only".into(),
        source: "test-only".into(),
    };
    let directory = root.join(&receipt.directory);
    private_directory(&directory).unwrap();
    extract(
        Cursor::new(archive(&[(EXE, b"browser", 0o700)])),
        &directory,
        "mac-arm64",
        7,
    )
    .unwrap();
    fs::write(
        root.join("current.json"),
        serde_json::to_vec(&receipt).unwrap(),
    )
    .unwrap();
    directory.join(EXE)
}

#[test]
fn discovery_uses_complete_matching_receipt_and_ignores_staging() {
    let dir = tempfile::tempdir().unwrap();
    let executable = installed_fixture(dir.path());
    assert_eq!(
        installed_at(dir.path(), "mac-arm64"),
        Some(executable.clone())
    );
    assert!(installed_at(dir.path(), "mac-x64").is_none());
    let stage = tempfile::Builder::new()
        .prefix(".install-")
        .tempdir_in(dir.path())
        .unwrap();
    fs::write(stage.path().join("incomplete"), b"unfinished download").unwrap();
    assert_eq!(
        installed_at(dir.path(), "mac-arm64"),
        Some(executable.clone())
    );
    drop(stage);
    fs::remove_file(executable).unwrap();
    assert!(installed_at(dir.path(), "mac-arm64").is_none());
}

#[test]
fn failed_start_keeps_previous_receipt_and_success_publishes_verified_version() {
    let root = tempfile::tempdir().unwrap();
    let previous = installed_fixture(root.path());
    let prior_receipt = fs::read(root.path().join("current.json")).unwrap();
    for succeeds in [false, true] {
        let stage = tempfile::Builder::new()
            .prefix(".install-")
            .tempdir_in(root.path())
            .unwrap();
        extract(
            Cursor::new(archive(&[(EXE, b"next browser", 0o700)])),
            stage.path(),
            "mac-arm64",
            12,
        )
        .unwrap();
        let stage_path = stage.path().to_path_buf();
        let selected = Download {
            version: "153.0.8000.2".into(),
            url: "test-only".into(),
        };
        let result = publish(
            root.path(),
            stage,
            "mac-arm64",
            selected,
            "test-only".into(),
            |path| {
                assert_eq!(fs::read(path).unwrap(), b"next browser");
                if succeeds {
                    Ok(())
                } else {
                    Err(failure("test startup failure"))
                }
            },
        );
        assert!(!stage_path.exists());
        assert_eq!(fs::read(&previous).unwrap(), b"browser");
        if succeeds {
            let current = result.unwrap();
            assert_eq!(
                installed_at(root.path(), "mac-arm64"),
                Some(current.clone())
            );
            assert_eq!(fs::read(current).unwrap(), b"next browser");
        } else {
            assert!(result.is_err());
            assert_eq!(
                fs::read(root.path().join("current.json")).unwrap(),
                prior_receipt
            );
            assert_eq!(
                installed_at(root.path(), "mac-arm64"),
                Some(previous.clone())
            );
        }
    }
}

#[test]
#[cfg(unix)]
fn discovery_and_locks_reject_symlink_substitution() {
    use std::os::unix::fs::symlink;
    let dir = tempfile::tempdir().unwrap();
    let executable = installed_fixture(dir.path());
    let moved = dir.path().join("outside");
    fs::rename(&executable, &moved).unwrap();
    symlink(&moved, &executable).unwrap();
    assert!(installed_at(dir.path(), "mac-arm64").is_none());
    symlink(&moved, dir.path().join("install.lock")).unwrap();
    assert!(lock(dir.path()).is_err());
}

#[test]
fn streaming_download_checks_actual_bytes_and_records_digest() {
    use std::net::TcpListener;
    fn serve(bytes: &'static [u8]) -> (String, std::thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let task = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(3)))
                .unwrap();
            let mut request_reader = bounded_fixture_io::Reader::new(
                &stream,
                std::time::Instant::now() + Duration::from_secs(3),
            );
            let mut buffer = [0; 8192];
            let count = request_reader.read(&mut buffer).unwrap();
            assert!(count > 0);
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n")
                .unwrap();
            stream.write_all(bytes).unwrap();
        });
        (url, task)
    }
    let client = Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(3))
        .build()
        .unwrap();
    let (url, task) = serve(b"browser");
    let mut body = Vec::new();
    assert_eq!(
        download(&client, &url, 7, &mut body).unwrap(),
        crate::hex(Sha256::digest(b"browser"))
    );
    task.join().unwrap();
    assert_eq!(body, b"browser");
    let (url, task) = serve(b"oversized browser");
    assert!(download(&client, &url, 7, &mut Vec::new()).is_err());
    task.join().unwrap();
}

#[cfg(test)]
mod bounded_fixture_io {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/support/bounded_read.rs"
    ));
}
