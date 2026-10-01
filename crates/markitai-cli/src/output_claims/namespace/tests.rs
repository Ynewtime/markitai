use super::*;
use crate::output_claims::MemberLeases;
use std::os::unix::fs::{MetadataExt, PermissionsExt, symlink};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

fn names() -> Vec<String> {
    vec!["document.md".into(), "document.llm.md".into()]
}

fn synchronized(directories: &[&Directory]) -> Result<()> {
    let mut sync = SyncGroup::new();
    for directory in directories {
        sync.stage(&directory.file)?;
    }
    sync.commit()?;
    Ok(())
}

#[test]
fn new_and_existing_chains_are_fenced_before_claims_without_touching_outputs() {
    let root = tempfile::tempdir().unwrap();
    let parent = root.path().join("out");
    let mut first = NamespaceBatch::new();
    first.prepare(&parent, false).unwrap();
    first.prepare(&parent, false).unwrap();
    assert_eq!(first.parents.len(), 1);
    let ready = first.commit().unwrap();
    ready.validate(&parent, false).unwrap();
    fs::write(parent.join("document.md"), "foreign output\n").unwrap();
    let leases = MemberLeases::acquire(&parent, &names(), false).unwrap();
    let keys = leases.keys();
    drop(leases);

    let mut second = NamespaceBatch::new();
    second.prepare(&parent, false).unwrap();
    let mut observed = Vec::new();
    let ready = second
        .commit_with(|directories| {
            observed = directories
                .iter()
                .map(|directory| directory.path.clone())
                .collect();
            synchronized(directories)
        })
        .unwrap();
    ready.validate(&parent, false).unwrap();
    assert_eq!(observed.len(), 5);
    let actual = fs::canonicalize(&parent).unwrap();
    assert_eq!(
        observed,
        vec![
            actual.join(".markitai/ownership/records"),
            actual.join(".markitai/ownership/members"),
            actual.join(".markitai/ownership"),
            actual.join(".markitai"),
            actual.clone(),
        ]
    );
    assert_eq!(
        MemberLeases::acquire(&parent, &names(), false)
            .unwrap()
            .keys(),
        keys
    );
    assert_eq!(
        fs::read_to_string(parent.join("document.md")).unwrap(),
        "foreign output\n"
    );
    for path in observed.iter().take(4) {
        assert_eq!(fs::metadata(path).unwrap().mode() & 0o077, 0);
    }
}

#[test]
fn a_sync_failure_never_yields_admission_authority_or_starts_work() {
    let root = tempfile::tempdir().unwrap();
    let parent = root.path().join("out");
    let mut batch = NamespaceBatch::new();
    batch.prepare(&parent, false).unwrap();
    let mut dispatched = 0;
    let result = batch
        .commit_with(|directories| {
            // Some host synchronization can succeed before a later operation fails.
            let mut sync = SyncGroup::new();
            sync.stage(&directories[0].file)?;
            Err(io::Error::other("injected namespace synchronization failure").into())
        })
        .map(|ready| {
            ready.validate(&parent, false).unwrap();
            dispatched += 1;
        });
    assert!(result.is_err());
    assert_eq!(dispatched, 0);
    assert!(!parent.join("document.md").exists());
    let mut retry = NamespaceBatch::new();
    retry.prepare(&parent, false).unwrap();
    retry.commit().unwrap().validate(&parent, false).unwrap();
}

#[test]
fn the_namespace_fence_orders_without_a_durable_flush_and_symlinks_still_fail() {
    use crate::output_claims::sync_group::take_commits;
    let root = tempfile::tempdir().unwrap();
    let real = fs::canonicalize(root.path()).unwrap();
    let parent = real.join("tree/out");
    let mut batch = NamespaceBatch::new();
    batch.prepare(&parent, false).unwrap();
    batch.prepare(&real.join("tree/second"), false).unwrap();
    take_commits();
    let ready = batch.commit().unwrap();
    // One commit for every parent of the window, ordered only: the admission
    // journal flush and each publication's acknowledgement make it durable.
    assert_eq!(take_commits(), ["ordered"]);
    ready.validate(&parent, false).unwrap();
    // A symbolic link substituted above the parent fails the next validation.
    fs::rename(real.join("tree"), real.join("moved")).unwrap();
    symlink(real.join("moved"), real.join("tree")).unwrap();
    let error = ready.validate(&parent, false).unwrap_err().to_string();
    assert!(error.contains("symlink policy"), "{error}");
    let mut batch = NamespaceBatch::new();
    assert!(batch.prepare(&parent, false).is_err());
}

#[test]
fn changed_namespace_is_rejected_before_and_after_the_fence() {
    for after_fence in [false, true] {
        let root = tempfile::tempdir().unwrap();
        let parent = root.path().join("out");
        let mut batch = NamespaceBatch::new();
        batch.prepare(&parent, false).unwrap();
        let members = parent.join(".markitai/ownership/members");
        let replace = || {
            fs::rename(&members, parent.join("old-members")).unwrap();
            let mut builder = fs::DirBuilder::new();
            use std::os::unix::fs::DirBuilderExt;
            builder.mode(0o700).create(&members).unwrap();
            fs::write(members.join("foreign"), "keep me").unwrap();
        };
        if after_fence {
            let ready = batch.commit().unwrap();
            replace();
            assert!(ready.validate(&parent, false).is_err());
        } else {
            let result = batch.commit_with(|directories| {
                synchronized(directories)?;
                replace();
                Ok(())
            });
            assert!(result.is_err());
        }
        assert_eq!(
            fs::read_to_string(members.join("foreign")).unwrap(),
            "keep me"
        );
    }
}

#[test]
fn one_bad_parent_does_not_poison_other_preparations_or_replace_its_bytes() {
    let root = tempfile::tempdir().unwrap();
    let bad = root.path().join("bad");
    fs::create_dir_all(bad.join(".markitai/ownership")).unwrap();
    fs::set_permissions(
        bad.join(".markitai/ownership"),
        fs::Permissions::from_mode(0o700),
    )
    .unwrap();
    let collision = bad.join(".markitai/ownership/records");
    fs::write(&collision, "unrelated metadata").unwrap();
    let good = root.path().join("good");
    let mut batch = NamespaceBatch::new();
    batch.prepare(&good, false).unwrap();
    assert!(batch.prepare(&bad, false).is_err());
    let ready = batch.commit().unwrap();
    ready.validate(&good, false).unwrap();
    assert!(ready.validate(&bad, false).is_err());
    assert_eq!(fs::read_to_string(collision).unwrap(), "unrelated metadata");
}

#[test]
fn parent_aliases_share_proof_but_metadata_symlinks_are_never_admitted() {
    let root = tempfile::tempdir().unwrap();
    let parent = root.path().join("out");
    fs::create_dir(&parent).unwrap();
    let alias = root.path().join("alias");
    symlink(&parent, &alias).unwrap();
    let mut batch = NamespaceBatch::new();
    assert!(batch.prepare(&alias, false).is_err());
    batch.prepare(&parent, false).unwrap();
    batch.prepare(&alias, true).unwrap();
    assert_eq!(batch.parents.len(), 1);
    let ready = batch.commit().unwrap();
    ready.validate(&alias, true).unwrap();
    assert!(ready.validate(&alias, false).is_err());

    let root2 = tempfile::tempdir().unwrap();
    let other = root2.path().join("other");
    fs::create_dir(&other).unwrap();
    symlink(&other, root2.path().join(".markitai")).unwrap();
    assert!(NamespaceBatch::new().prepare(root2.path(), true).is_err());
    assert_eq!(fs::read_dir(other).unwrap().count(), 0);
}

#[test]
fn descriptor_window_is_bounded_without_limiting_a_later_window() {
    let root = tempfile::tempdir().unwrap();
    let mut batch = NamespaceBatch::new();
    for index in 0..MAX_NAMESPACE_PARENTS {
        batch
            .prepare(&root.path().join(format!("out-{index}")), false)
            .unwrap();
    }
    let next = root.path().join("next");
    assert!(batch.prepare(&next, false).is_err());
    assert!(!next.exists());
    let _ready = batch.commit().unwrap();
    let mut second = NamespaceBatch::new();
    second.prepare(&next, false).unwrap();
    second.commit().unwrap().validate(&next, false).unwrap();
}

struct ChildGuard(Child);
impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
fn helper(root: &Path, mode: &str) -> ChildGuard {
    ChildGuard(
        Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "output_claims::namespace::tests::namespace_process_helper",
                "--ignored",
                "--nocapture",
            ])
            .env_clear()
            .env("MARKITAI_NAMESPACE_TEST_ROOT", root)
            .env("MARKITAI_NAMESPACE_TEST_MODE", mode)
            .env("MARKITAI_HOME", root.join("private-home"))
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap(),
    )
}
fn ready(child: &mut ChildGuard, root: &Path) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !root.join("ready").exists() {
        assert!(
            child.0.try_wait().unwrap().is_none(),
            "helper exited before readiness"
        );
        assert!(Instant::now() < deadline, "helper did not become ready");
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn another_initializer_can_die_before_fencing_without_lending_its_proof() {
    let root = tempfile::tempdir().unwrap();
    let parent = root.path().join("out");
    let mut child = helper(root.path(), "uncommitted");
    ready(&mut child, root.path());
    // The peer deliberately published no fence, though its directories are visible.
    let mut batch = NamespaceBatch::new();
    batch.prepare(&parent, false).unwrap();
    let mut objects_fenced = 0;
    let ready = batch
        .commit_with(|directories| {
            objects_fenced = directories.len();
            synchronized(directories)
        })
        .unwrap();
    assert_eq!(objects_fenced, 5);
    drop(child);
    ready.validate(&parent, false).unwrap();
    let lease = MemberLeases::acquire(&parent, &names(), false).unwrap();
    ready.validate(&parent, false).unwrap();
    assert_eq!(lease.keys().len(), 2);
}

#[test]
fn committed_namespace_keeps_cross_process_member_exclusion_and_kill_release() {
    let root = tempfile::tempdir().unwrap();
    let parent = root.path().join("out");
    let mut child = helper(root.path(), "held");
    ready(&mut child, root.path());
    let lock = parent.join(".markitai/ownership/members/document.md");
    let before = fs::metadata(&lock).unwrap();
    let mut batch = NamespaceBatch::new();
    batch.prepare(&parent, false).unwrap();
    let ready = batch.commit().unwrap();
    ready.validate(&parent, false).unwrap();
    assert!(matches!(
        MemberLeases::acquire(&parent, &names(), false),
        Err(Error::Busy)
    ));
    drop(child);
    let lease = MemberLeases::acquire(&parent, &names(), false).unwrap();
    let after = fs::metadata(lock).unwrap();
    assert_eq!((before.dev(), before.ino()), (after.dev(), after.ino()));
    assert_eq!(lease.keys().len(), 2);
}

#[test]
#[ignore = "subprocess fixture: invoked only with a private test root"]
fn namespace_process_helper() {
    let Some(root) = std::env::var_os("MARKITAI_NAMESPACE_TEST_ROOT") else {
        return;
    };
    let root = PathBuf::from(root);
    let parent = root.join("out");
    let mut batch = NamespaceBatch::new();
    batch.prepare(&parent, false).unwrap();
    let mode = std::env::var("MARKITAI_NAMESPACE_TEST_MODE").unwrap();
    let (_ready, _lease) = if mode == "held" {
        let ready = batch.commit().unwrap();
        let lease = MemberLeases::acquire(&parent, &names(), false).unwrap();
        ready.validate(&parent, false).unwrap();
        (Some(ready), Some(lease))
    } else {
        assert_eq!(mode, "uncommitted");
        // Keep the uncommitted descriptors alive until killed, without syncing them.
        fs::write(root.join("ready"), "uncommitted namespace visible").unwrap();
        std::thread::sleep(Duration::from_secs(20));
        drop(batch);
        panic!("parent failed to stop the uncommitted initializer");
    };
    fs::write(root.join("ready"), "member locks held").unwrap();
    std::thread::sleep(Duration::from_secs(20));
    panic!("parent failed to stop the held-claim helper");
}
