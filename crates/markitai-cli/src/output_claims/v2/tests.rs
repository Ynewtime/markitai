use super::*;
use crate::output_claims::{MemberLeases, NamespaceBatch};

fn names(base: &str) -> Vec<String> {
    vec![format!("{base}.md"), format!("{base}.llm.md")]
}
fn parent(root: &Path) -> Arc<Parent> {
    let mut batch = NamespaceBatch::new();
    batch.prepare(root, false).unwrap();
    batch.commit().unwrap().validate(root, false).unwrap();
    Parent::open(&platform::canonicalize(root).unwrap()).unwrap()
}
fn count(parent: &Parent, name: &str) -> usize {
    fs::read_dir(parent.location(name)).unwrap().count()
}

#[test]
fn fresh_namespace_excludes_family_and_cleans_only_when_epoch_idle() {
    let root = tempfile::tempdir().unwrap();
    let parent = parent(root.path());
    let epoch = Epoch::new(Arc::clone(&parent)).unwrap();
    let first = Lease::acquire(Arc::clone(&epoch), &names("x")).unwrap();
    let independent = Lease::acquire(Arc::clone(&epoch), &names("y")).unwrap();
    assert!(matches!(
        Lease::acquire(Arc::clone(&epoch), &names("x.llm")),
        Err(Error::Busy)
    ));
    parent.cleanup().unwrap();
    assert!(count(&parent, "names-v2") >= 4);
    drop(first);
    assert!(count(&parent, "names-v2") >= 4);
    drop(independent);
    assert!(count(&parent, "names-v2") >= 4);
    assert!(count(&parent, "names-v2") >= 4);
    drop(epoch);
    assert_eq!(count(&parent, "names-v2"), 0);
    assert_eq!(fs::metadata(parent.location("members")).unwrap().len(), 0);
    assert!(parent.location("members").is_file());
}

#[test]
fn filesystem_aliases_follow_native_probe_identity() {
    let root = tempfile::tempdir().unwrap();
    let parent = parent(root.path());
    let epoch = Epoch::new(parent).unwrap();
    for (first, alias) in [("Case.md", "case.md"), ("caf\u{e9}.md", "cafe\u{301}.md")] {
        let a = vec![first.into()];
        let b = vec![alias.into()];
        let held = Lease::acquire(Arc::clone(&epoch), &a).unwrap();
        let aliases = epoch.keys(&b).unwrap() == held.keys();
        let second = Lease::acquire(Arc::clone(&epoch), &b);
        assert_eq!(matches!(second, Err(Error::Busy)), aliases);
        drop(second);
        drop(held);
    }
}

#[test]
fn active_legacy_members_are_never_migrated_or_deleted() {
    let root = tempfile::tempdir().unwrap();
    for name in [
        ".markitai",
        ".markitai/ownership",
        ".markitai/ownership/members",
    ] {
        platform::private_directory()
            .create(root.path().join(name))
            .unwrap();
    }
    let held = MemberLeases::acquire(root.path(), &names("x"), false).unwrap();
    assert!(held.epoch().is_none());
    assert!(matches!(
        MemberLeases::acquire(root.path(), &names("x"), false),
        Err(Error::Busy)
    ));
    let identities = held.keys();
    drop(held);
    assert_eq!(
        MemberLeases::acquire(root.path(), &names("x"), false)
            .unwrap()
            .keys(),
        identities
    );
    assert!(!root.path().join(".markitai/ownership/names-v2").exists());
}

#[cfg(unix)]
#[test]
fn cleanup_preserves_unknown_objects_and_external_bytes() {
    use std::os::unix::fs::symlink;
    for kind in ["symlink", "hardlink", "bytes", "directory"] {
        let root = tempfile::tempdir().unwrap();
        let parent = parent(root.path());
        let external = root.path().join("external");
        fs::write(&external, b"preserve").unwrap();
        let path = parent.location("names-v2/unknown.md");
        match kind {
            "symlink" => symlink(&external, &path).unwrap(),
            "hardlink" => fs::hard_link(&external, &path).unwrap(),
            "bytes" => fs::write(&path, b"unknown").unwrap(),
            _ => fs::create_dir(&path).unwrap(),
        }
        assert!(parent.cleanup().is_err());
        assert!(fs::symlink_metadata(&path).is_ok());
        assert_eq!(fs::read(&external).unwrap(), b"preserve");
    }
}

#[test]
fn stable_gate_replacement_blocks_publication_and_cleanup() {
    let root = tempfile::tempdir().unwrap();
    let parent = parent(root.path());
    let epoch = Epoch::new(Arc::clone(&parent)).unwrap();
    let held = Lease::acquire(epoch, &names("x")).unwrap();
    let gate = parent.location("members");
    fs::rename(&gate, root.path().join("original-gate")).unwrap();
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    platform::private_file(&mut options).open(&gate).unwrap();
    assert!(held.validate().is_err());
    assert!(parent.cleanup().is_err());
    drop(held);
    assert_eq!(count(&parent, "names-v2"), 2);
}

#[test]
fn killed_acquisition_and_cleanup_never_split_the_lock_domain() {
    use std::process::{Command, Stdio};
    use std::time::{Duration, Instant};
    for phase in ["probe", "writer", "active", "cleanup"] {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir(root.path().join("home-a")).unwrap();
        fs::create_dir(root.path().join("state-a")).unwrap();
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "output_claims::v2::tests::paused_process_helper",
                "--ignored",
                "--nocapture",
            ])
            .env("MARKITAI_V2_ROOT", root.path())
            .env("MARKITAI_V2_PAUSE", phase)
            .env("HOME", root.path().join("home-a"))
            .env("MARKITAI_HOME", root.path().join("state-a"))
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        while !root.path().join("paused").exists() {
            assert!(child.try_wait().unwrap().is_none());
            if Instant::now() >= deadline {
                child.kill().unwrap();
                child.wait().unwrap();
                panic!("pause {phase} timed out");
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        if phase == "active" {
            let observer =
                Epoch::new(Parent::open(&platform::canonicalize(root.path()).unwrap()).unwrap())
                    .unwrap();
            assert_eq!(observer.family_keys(&[names("x")]).unwrap()[0].len(), 2);
            drop(observer);
            assert!(matches!(
                MemberLeases::acquire(root.path(), &names("x"), false),
                Err(Error::Busy)
            ));
            let other = MemberLeases::acquire(root.path(), &names("y"), false).unwrap();
            drop(other);
        }
        child.kill().unwrap();
        child.wait().unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        let held = loop {
            match MemberLeases::acquire(root.path(), &names("x"), false) {
                Err(Error::Busy) if Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_millis(10))
                }
                result => break result.unwrap(),
            }
        };
        held.validate_member(&root.path().join("x.md")).unwrap();
        drop(held);
        let parent = Parent::open(&platform::canonicalize(root.path()).unwrap()).unwrap();
        parent.cleanup().unwrap();
        assert_eq!(count(&parent, "names-v2"), 0);
        assert!(!parent.location("writers-v2").exists());
    }
}

#[test]
#[ignore = "private kill-boundary subprocess fixture"]
fn paused_process_helper() {
    let Some(root) = std::env::var_os("MARKITAI_V2_ROOT") else {
        return;
    };
    let root = PathBuf::from(root);
    let lease = MemberLeases::acquire(&root, &names("x"), false).unwrap();
    checkpoint("active");
    if std::env::var("MARKITAI_V2_PAUSE").unwrap() == "cleanup" {
        drop(lease);
    }
    panic!("configured pause was not reached");
}

#[cfg(target_os = "linux")]
#[test]
fn explicit_release_unpins_fork_inherited_writers_and_epoch() {
    let root = tempfile::tempdir().unwrap();
    let parent = parent(root.path());
    let epoch = Epoch::new(Arc::clone(&parent)).unwrap();
    let held = Lease::acquire(Arc::clone(&epoch), &names("x")).unwrap();
    // SAFETY: child performs only async-signal-safe libc calls and never uses
    // Rust's allocator or inherited synchronization primitives after fork.
    let pid = unsafe { libc::fork() };
    assert!(pid >= 0);
    if pid == 0 {
        unsafe {
            loop {
                libc::pause();
            }
        }
    }
    struct Child(i32);
    impl Drop for Child {
        fn drop(&mut self) {
            unsafe {
                libc::kill(self.0, libc::SIGKILL);
                libc::waitpid(self.0, std::ptr::null_mut(), 0);
            }
        }
    }
    let _child = Child(pid);
    drop(held);
    drop(epoch);
    parent.cleanup().unwrap();
    assert_eq!(count(&parent, "names-v2"), 0);
    let _new = MemberLeases::acquire(root.path(), &names("x"), false).unwrap();
}

#[test]
fn replacing_a_probe_invalidates_the_held_claim_without_touching_documents() {
    let root = tempfile::tempdir().unwrap();
    let parent = parent(root.path());
    let epoch = Epoch::new(Arc::clone(&parent)).unwrap();
    let held = Lease::acquire(Arc::clone(&epoch), &["x.md".into()]).unwrap();
    let probe = parent.location("names-v2/x.md");
    fs::rename(&probe, root.path().join("old-probe")).unwrap();
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    platform::private_file(&mut options).open(&probe).unwrap();
    assert!(held.validate().is_err());
    let replacement = Lease::acquire(epoch, &["x.md".into()]).unwrap();
    replacement.validate().unwrap();
    drop(held);
    drop(replacement);
    assert!(!root.path().join("x.md").exists());
}

#[test]
fn metadata_only_observers_do_not_release_an_active_probe_lock() {
    let root = tempfile::tempdir().unwrap();
    let parent = parent(root.path());
    let epoch = Epoch::new(Arc::clone(&parent)).unwrap();
    let held = Lease::acquire(Arc::clone(&epoch), &names("x")).unwrap();
    assert_eq!(epoch.keys(&names("x")).unwrap(), held.keys());
    assert_eq!(epoch.family_keys(&[names("x")]).unwrap(), vec![held.keys()]);
    assert!(matches!(
        Lease::acquire(epoch, &names("x")),
        Err(Error::Busy)
    ));
    held.validate().unwrap();
}

#[test]
fn unknown_writers_namespace_is_rejected_and_preserved() {
    let root = tempfile::tempdir().unwrap();
    let parent = parent(root.path());
    platform::private_directory()
        .create(parent.location("writers-v2"))
        .unwrap();
    let unknown = parent.location("writers-v2/unknown");
    fs::write(&unknown, b"preserve").unwrap();
    let message = parent.validate().unwrap_err().to_string();
    assert!(message.contains("writers-v2") && !message.contains("unreleased"));
    assert!(NamespaceBatch::new().prepare(root.path(), false).is_err());
    assert_eq!(fs::read(unknown).unwrap(), b"preserve");
}

#[test]
#[ignore = "caller-owned 16-parent namespace / active claim descriptor fixture"]
fn maximum_namespace_window_helper() {
    let Some(root) = std::env::var_os("MARKITAI_V2_BENCH_ROOT") else {
        return;
    };
    let root = PathBuf::from(root);
    let mut batch = NamespaceBatch::new();
    for index in 0..super::super::namespace::MAX_NAMESPACE_PARENTS {
        batch.prepare(&root.join(index.to_string()), false).unwrap();
    }
    #[cfg(unix)]
    let count_fds = || {
        (0..1024)
            .filter(|&fd| unsafe { libc::fcntl(fd, libc::F_GETFD) } >= 0)
            .count()
    };
    #[cfg(unix)]
    let namespace_fds = count_fds();
    let ready = batch.commit().unwrap();
    let mut claims = Vec::new();
    for index in 0..super::super::namespace::MAX_NAMESPACE_PARENTS {
        let parent = root.join(index.to_string());
        ready.validate(&parent, false).unwrap();
        claims.push(MemberLeases::acquire(&parent, &names("x"), false).unwrap());
    }
    #[cfg(unix)]
    let active_fds = count_fds();
    drop(claims);
    drop(ready);
    let metrics = serde_json::json!({"parents":super::super::namespace::MAX_NAMESPACE_PARENTS});
    #[cfg(unix)]
    let metrics = {
        let mut metrics = metrics;
        metrics["namespace_fds"] = serde_json::json!(namespace_fds);
        metrics["active_fds"] = serde_json::json!(active_fds);
        metrics["final_fds"] = serde_json::json!(count_fds());
        metrics["peak_fds"] = serde_json::json!(super::fd_peak());
        metrics
    };
    fs::write(
        root.join("metrics.json"),
        serde_json::to_vec_pretty(&metrics).unwrap(),
    )
    .unwrap();
}

#[test]
fn publication_refuses_a_replaced_probe_and_preserves_existing_bytes() {
    use crate::output_claims::{Claim, Policy};
    use markitai_core::output::Publication;
    let root = tempfile::tempdir().unwrap();
    let leases = MemberLeases::acquire(root.path(), &["x.md".into()], false).unwrap();
    let document = leases.parent().join("x.md");
    fs::write(&document, b"existing authored bytes").unwrap();
    let claim = Claim::new(leases, None, Policy::Overwrite, false).unwrap();
    let probe = root.path().join(".markitai/ownership/names-v2/x.md");
    fs::rename(&probe, root.path().join("original-probe")).unwrap();
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    platform::private_file(&mut options).open(&probe).unwrap();
    let error = claim
        .publish(&document, b"forbidden replacement")
        .unwrap_err();
    assert!(error.to_string().contains("probe was replaced"), "{error}");
    assert_eq!(fs::read(&document).unwrap(), b"existing authored bytes");
}
