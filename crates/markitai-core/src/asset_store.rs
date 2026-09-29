//! Shared content-addressed assets are reused only after exact byte validation.
use crate::{Error, Result};
use std::fs::{self, File, Metadata};
use std::io::{self, Read, Write};
use std::path::Path;

const BLOCK: usize = 32 * 1024;

/// The caller checks its configured symlink policy before entering this helper.
/// An allowed existing symlink is read for verification, never replaced or written
/// through. Document publication keeps its separate conflict/ownership policy.
pub(crate) fn insert_or_verify(path: &Path, bytes: &[u8]) -> Result<()> {
    if verify_existing(path, bytes)? {
        return Ok(());
    }
    publish_new(path, bytes)
}

fn mismatch(path: &Path) -> Error {
    Error::Conversion(format!(
        "Existing asset content does not match: {}",
        path.display()
    ))
}

/// False means the directory entry itself was absent, not a dangling symlink.
fn verify_existing(path: &Path, bytes: &[u8]) -> Result<bool> {
    let leaf = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error.into()),
    };
    let advertised = fs::metadata(path)?;
    if !advertised.is_file() {
        return Err(Error::Conversion(format!(
            "Asset is not a regular file: {}",
            path.display()
        )));
    }
    if advertised.len() != bytes.len() as u64 {
        return Err(mismatch(path));
    }
    let mut file = File::open(path)?;
    let before = file.metadata()?;
    if !before.is_file() || !same_version(&advertised, &before) {
        return Err(Error::Conversion(format!(
            "Asset changed during verification: {}",
            path.display()
        )));
    }
    let mut buffer = [0u8; BLOCK];
    for expected in bytes.chunks(BLOCK) {
        if let Err(error) = file.read_exact(&mut buffer[..expected.len()]) {
            return if error.kind() == io::ErrorKind::UnexpectedEof {
                Err(mismatch(path))
            } else {
                Err(error.into())
            };
        }
        if &buffer[..expected.len()] != expected {
            return Err(mismatch(path));
        }
    }
    if file.read(&mut buffer[..1])? != 0 {
        return Err(mismatch(path));
    }
    if !same_version(&before, &file.metadata()?)
        || !same_version(&before, &fs::metadata(path)?)
        || !same_version(&leaf, &fs::symlink_metadata(path)?)
    {
        return Err(Error::Conversion(format!(
            "Asset changed during verification: {}",
            path.display()
        )));
    }
    Ok(true)
}

fn same_version(left: &Metadata, right: &Metadata) -> bool {
    let common = left.file_type() == right.file_type()
        && left.len() == right.len()
        && left.modified().ok() == right.modified().ok();
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        // No-clobber publication may use hard-link/unlink on some filesystems;
        // that changes ctime/link count without changing the asset bytes.
        common && left.dev() == right.dev() && left.ino() == right.ino()
    }
    #[cfg(not(unix))]
    {
        common
    }
}

// Both competing writers may arrive here after observing a missing destination.
// Only this asset-specific path accepts AlreadyExists, and only after validating
// the winner's full content. Other I/O failures remain failures.
fn publish_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let mut staged = crate::output::deliverable_builder().tempfile_in(parent)?;
    staged.write_all(bytes)?;
    staged.as_file().sync_all()?;
    match staged.persist_noclobber(path) {
        Ok(_) => Ok(()),
        Err(error) if error.error.kind() == io::ErrorKind::AlreadyExists => {
            drop(error.file);
            if verify_existing(path, bytes)? {
                Ok(())
            } else {
                Err(Error::Conversion(format!(
                    "Competing asset disappeared during verification: {}",
                    path.display()
                )))
            }
        }
        Err(error) => Err(error.error.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::process::{Child, Command, Stdio};
    use std::time::{Duration, Instant};

    #[test]
    fn existing_asset_requires_exact_bytes_and_never_replaces_conflicting_content() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("asset.bin");
        let mut bytes = vec![0xa5; BLOCK * 2 + 7];
        bytes[BLOCK] = 0;
        insert_or_verify(&path, &bytes).unwrap();
        let before = fs::metadata(&path).unwrap();
        insert_or_verify(&path, &bytes).unwrap();
        assert!(same_version(&before, &fs::metadata(&path).unwrap()));
        for wrong in [
            {
                let mut wrong = bytes.clone();
                wrong[BLOCK] = 1;
                wrong
            },
            bytes[..bytes.len() - 1].to_vec(),
            {
                let mut wrong = bytes.clone();
                wrong.push(0);
                wrong
            },
        ] {
            assert!(insert_or_verify(&path, &wrong).is_err());
            assert_eq!(fs::read(&path).unwrap(), bytes);
        }
        let empty = root.path().join("empty.bin");
        insert_or_verify(&empty, &[]).unwrap();
        insert_or_verify(&empty, &[]).unwrap();
        assert!(insert_or_verify(&empty, &[0]).is_err());
        let directory = root.path().join("directory.bin");
        fs::create_dir(&directory).unwrap();
        assert!(insert_or_verify(&directory, &[]).is_err());
        assert!(directory.is_dir());
    }

    #[cfg(unix)]
    #[test]
    fn permitted_symlinks_are_only_verified_and_special_targets_are_rejected() {
        use std::os::unix::{fs::symlink, net::UnixListener};
        let root = tempfile::tempdir().unwrap();
        let target = root.path().join("target.bin");
        let alias = root.path().join("alias.bin");
        fs::write(&target, b"same").unwrap();
        symlink("target.bin", &alias).unwrap();
        insert_or_verify(&alias, b"same").unwrap();
        assert!(insert_or_verify(&alias, b"other").is_err());
        assert_eq!(fs::read(&target).unwrap(), b"same");
        assert_eq!(fs::read_link(&alias).unwrap(), Path::new("target.bin"));
        let dangling = root.path().join("dangling.bin");
        symlink("absent.bin", &dangling).unwrap();
        assert!(insert_or_verify(&dangling, b"new").is_err());
        assert_eq!(fs::read_link(&dangling).unwrap(), Path::new("absent.bin"));
        assert!(!root.path().join("absent.bin").exists());
        let socket = root.path().join("socket");
        let _listener = UnixListener::bind(&socket).unwrap();
        let socket_alias = root.path().join("socket.bin");
        symlink("socket", &socket_alias).unwrap();
        assert!(insert_or_verify(&socket_alias, &[]).is_err());
    }

    struct Worker(Child);
    impl Worker {
        fn start(root: &Path, role: &str, payload: &str) -> Self {
            let mut command = Command::new(std::env::current_exe().unwrap());
            command.env_clear();
            for name in ["PATH", "SYSTEMROOT", "TMPDIR", "TMP", "TEMP"] {
                if let Some(value) = std::env::var_os(name) {
                    command.env(name, value);
                }
            }
            Self(
                command
                    .args([
                        "--ignored",
                        "--exact",
                        "asset_store::tests::asset_race_child",
                        "--nocapture",
                    ])
                    .env("MARKITAI_HOME", root.join("private-home"))
                    .env("MARKITAI_TEST_ASSET_ROOT", root)
                    .env("MARKITAI_TEST_ASSET_ROLE", role)
                    .env("MARKITAI_TEST_ASSET_PAYLOAD", payload)
                    .stdin(Stdio::null())
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .spawn()
                    .unwrap(),
            )
        }
        fn wait(&mut self) {
            let deadline = Instant::now() + Duration::from_secs(15);
            loop {
                if let Some(status) = self.0.try_wait().unwrap() {
                    assert!(
                        status.success(),
                        "asset child exited unsuccessfully: {status}"
                    );
                    return;
                }
                assert!(Instant::now() < deadline, "asset child did not exit");
                std::thread::sleep(Duration::from_millis(5));
            }
        }
    }
    impl Drop for Worker {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
    fn wait_for(path: &Path) {
        let deadline = Instant::now() + Duration::from_secs(15);
        while !path.is_file() {
            assert!(
                Instant::now() < deadline,
                "child gate was not published: {}",
                path.display()
            );
            std::thread::sleep(Duration::from_millis(5));
        }
    }
    fn payload(name: &str) -> Vec<u8> {
        match name {
            "same" => vec![0xa5; BLOCK * 2 + 7],
            "left" => vec![0x11; BLOCK * 2 + 7],
            "right" => vec![0xee; BLOCK * 2 + 7],
            _ => panic!("unknown private asset fixture"),
        }
    }
    fn race(left: &str, right: &str) -> (tempfile::TempDir, String, String) {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir(root.path().join("assets")).unwrap();
        let mut a = Worker::start(root.path(), "left", left);
        let mut b = Worker::start(root.path(), "right", right);
        wait_for(&root.path().join("left.ready"));
        wait_for(&root.path().join("right.ready"));
        // Both children have observed absence before either can publish. They now
        // enter the same production publication branch in independent processes.
        assert!(!root.path().join("assets/shared.bin").exists());
        fs::write(root.path().join("go"), b"publish").unwrap();
        a.wait();
        b.wait();
        let left = fs::read_to_string(root.path().join("left.result")).unwrap();
        let right = fs::read_to_string(root.path().join("right.result")).unwrap();
        let files: Vec<_> = fs::read_dir(root.path().join("assets"))
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        assert_eq!(files, vec![std::ffi::OsString::from("shared.bin")]);
        (root, left, right)
    }

    #[test]
    fn identical_assets_competing_from_two_processes_both_succeed() {
        let (root, left, right) = race("same", "same");
        assert_eq!(left, "ok");
        assert_eq!(right, "ok");
        assert_eq!(
            fs::read(root.path().join("assets/shared.bin")).unwrap(),
            payload("same")
        );
    }

    #[test]
    fn competing_different_bytes_reject_the_loser_and_preserve_the_winner() {
        let (root, left, right) = race("left", "right");
        assert_eq!(
            [left.as_str(), right.as_str()]
                .iter()
                .filter(|status| **status == "ok")
                .count(),
            1
        );
        assert_eq!(
            [left.as_str(), right.as_str()]
                .iter()
                .filter(|status| status.starts_with("error:"))
                .count(),
            1
        );
        let winner = if left == "ok" { "left" } else { "right" };
        assert_eq!(
            fs::read(root.path().join("assets/shared.bin")).unwrap(),
            payload(winner)
        );
    }

    #[test]
    #[ignore = "isolated child invoked by asset publication process tests"]
    fn asset_race_child() {
        let root = PathBuf::from(
            std::env::var_os("MARKITAI_TEST_ASSET_ROOT").expect("private asset fixture root"),
        );
        let role = std::env::var("MARKITAI_TEST_ASSET_ROLE").unwrap();
        assert!(matches!(role.as_str(), "left" | "right"));
        let bytes = payload(&std::env::var("MARKITAI_TEST_ASSET_PAYLOAD").unwrap());
        let path = root.join("assets/shared.bin");
        assert!(!verify_existing(&path, &bytes).unwrap());
        fs::write(root.join(format!("{role}.ready")), b"observed absent").unwrap();
        wait_for(&root.join("go"));
        let result = match publish_new(&path, &bytes) {
            Ok(()) => "ok".to_owned(),
            Err(error) => format!("error:{error}"),
        };
        fs::write(root.join(format!("{role}.result")), result).unwrap();
    }
}
