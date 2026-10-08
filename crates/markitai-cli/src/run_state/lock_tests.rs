use super::{Limits, Scope, StateStore};
use crate::file_lock_test::InheritedChild;
use crate::run_state::{Error, Mode};

#[test]
fn dropping_store_releases_ownership_while_fork_child_still_holds_the_descriptor() {
    let root = tempfile::tempdir().unwrap();
    let input = root.path().join("input");
    let output = root.path().join("output");
    std::fs::create_dir(&input).unwrap();
    std::fs::create_dir(&output).unwrap();
    let scope = Scope::new(Mode::Directory, &input, &output).unwrap();
    let owner = StateStore::open(scope.clone(), "abc123", false, Limits::default()).unwrap();
    assert!(matches!(
        StateStore::open(scope.clone(), "abc123", false, Limits::default()),
        Err(Error::Busy)
    ));
    let mut child = InheritedChild::start();
    drop(owner);
    let reopened = StateStore::open(scope, "abc123", false, Limits::default());
    // Release/reap the child even when the old drop-only implementation fails.
    child.finish();
    assert!(
        reopened.is_ok(),
        "parent ownership outlived its guard: {:?}",
        reopened.err()
    );
}
