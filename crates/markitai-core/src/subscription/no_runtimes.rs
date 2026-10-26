//! Test support: a test process never finds the official Copilot, Claude or
//! Codex runtime installed on the developer's machine.
//!
//! The CLIs and servers the tests start look these runtimes up on `PATH` (the
//! serve tests' servers ask each one for its sign-in state), so a developer's
//! own installation would run, and reach the network, in the middle of a test
//! run; hosted CI has none. A static initializer rewrites this process's
//! `PATH` before the test harness starts: a directory holding a runtime is
//! replaced by a directory of links to its other programs (left out on
//! Windows, where links need privileges), so children inherit or copy a `PATH`
//! without them. Tests that need a runtime play one with `fake_runtime`.
//!
//! The library unit tests and the CLI's integration tests include this file.
#![allow(dead_code)]

use std::ffi::OsString;
use std::path::{Path, PathBuf};

const RUNTIMES: [&str; 3] = ["copilot", "claude", "codex"];

#[used]
#[cfg_attr(
    target_vendor = "apple",
    unsafe(link_section = "__DATA,__mod_init_func")
)]
#[cfg_attr(
    all(unix, not(target_vendor = "apple")),
    unsafe(link_section = ".init_array")
)]
#[cfg_attr(windows, unsafe(link_section = ".CRT$XCU"))]
static HIDE: extern "C" fn() = hide;

extern "C" fn hide() {
    let Some(path) = std::env::var_os("PATH") else {
        return;
    };
    if let Some(cleaned) = without_runtimes(&path, &std::env::temp_dir()) {
        // SAFETY: static initializers run before main, so no other thread exists.
        unsafe { std::env::set_var("PATH", cleaned) };
    }
}

/// The executables named like a runtime in `directory`.
fn runtimes_in(directory: &Path) -> Vec<PathBuf> {
    let extensions: Vec<String> = if cfg!(windows) {
        std::env::var("PATHEXT")
            .unwrap_or_else(|_| ".COM;.EXE;.BAT;.CMD".into())
            .split(';')
            .filter(|extension| !extension.is_empty())
            .map(str::to_owned)
            .chain([String::new()])
            .collect()
    } else {
        vec![String::new()]
    };
    RUNTIMES
        .iter()
        .flat_map(|name| {
            extensions
                .iter()
                .map(move |extension| directory.join(format!("{name}{extension}")))
        })
        .filter(|candidate| candidate.is_file())
        .collect()
}

/// `path` without the runtimes (link directories made under `mirrors`), or
/// None when it holds none.
fn without_runtimes(path: &std::ffi::OsStr, mirrors: &Path) -> Option<OsString> {
    let mut changed = false;
    let mut entries = Vec::new();
    for directory in std::env::split_paths(path) {
        let found = runtimes_in(&directory);
        if found.is_empty() {
            entries.push(directory);
            continue;
        }
        changed = true;
        entries.extend(mirror(&directory, &found, mirrors));
    }
    changed
        .then(|| std::env::join_paths(entries).ok())
        .flatten()
}

/// A directory of links to everything in `directory` but `hidden`, shared by
/// test processes and made again when `directory` changes.
#[cfg(unix)]
fn mirror(directory: &Path, hidden: &[PathBuf], mirrors: &Path) -> Option<PathBuf> {
    use std::hash::{Hash, Hasher};
    let modified = std::fs::metadata(directory).ok()?.modified().ok()?;
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    (directory, modified).hash(&mut hasher);
    let target = mirrors.join(format!("markitai-test-path-{:016x}", hasher.finish()));
    if !target.is_dir() {
        let staging = target.with_extension(std::process::id().to_string());
        let _ = std::fs::remove_dir_all(&staging);
        std::fs::create_dir_all(&staging).ok()?;
        for entry in std::fs::read_dir(directory).ok()?.flatten() {
            if !hidden.contains(&entry.path()) {
                let _ = std::os::unix::fs::symlink(entry.path(), staging.join(entry.file_name()));
            }
        }
        // Another test process may have made it first; either copy will do.
        if std::fs::rename(&staging, &target).is_err() {
            let _ = std::fs::remove_dir_all(&staging);
        }
    }
    target.is_dir().then_some(target)
}

#[cfg(not(unix))]
fn mirror(_: &Path, _: &[PathBuf], _: &Path) -> Option<PathBuf> {
    None
}

#[test]
fn no_subscription_runtime_is_on_the_test_path() {
    let path = std::env::var_os("PATH").unwrap_or_default();
    for directory in std::env::split_paths(&path) {
        assert_eq!(runtimes_in(&directory), Vec::<PathBuf>::new());
    }
}

#[cfg(unix)]
#[test]
fn a_directory_with_a_runtime_keeps_its_other_programs() {
    let root = tempfile::tempdir().unwrap();
    let (tools, plain) = (root.path().join("tools"), root.path().join("plain"));
    for directory in [&tools, &plain] {
        std::fs::create_dir(directory).unwrap();
    }
    for name in ["claude", "codex", "uv"] {
        std::fs::write(tools.join(name), "").unwrap();
    }
    let path = std::env::join_paths([&plain, &tools]).unwrap();
    let cleaned = without_runtimes(&path, root.path()).unwrap();
    let entries: Vec<PathBuf> = std::env::split_paths(&cleaned).collect();
    assert_eq!(entries[0], plain);
    let mut kept: Vec<_> = std::fs::read_dir(&entries[1])
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect();
    kept.sort();
    assert_eq!(kept, ["uv"]);
    assert_eq!(without_runtimes(plain.as_os_str(), root.path()), None);
}
