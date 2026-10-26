//! Executable scripts for tests that start on their first try.
use std::os::unix::fs::PermissionsExt;
use std::path::Path;

/// Writes `body` to `path` as an executable, copied by `cp` rather than
/// written by this process: a test thread that forks while this process holds
/// the file open for writing makes its start fail (ETXTBSY) until that child
/// execs, longer on a busy machine.
pub(crate) fn write_executable(path: &Path, body: &str) {
    let mut source = path.as_os_str().to_owned();
    source.push(".source");
    std::fs::write(&source, body).unwrap();
    // A new file each time: one being run can't be written over.
    let _ = std::fs::remove_file(path);
    let copied = std::process::Command::new("cp")
        .arg(&source)
        .arg(path)
        .status()
        .unwrap();
    assert!(copied.success());
    std::fs::remove_file(&source).unwrap();
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700)).unwrap();
}
