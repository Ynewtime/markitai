//! Symlink policy and path resolution for one state validation operation.
//!
//! The codec validates every saved path against the filesystem, walking each
//! ancestor. One operation revisits the same ancestors many times (scope roots,
//! shared output parents), so within an explicit scope each `lstat` and
//! `readlink` result is observed once and reused. That is a single consistent
//! observation instead of many, never a weaker check; outside a scope every call
//! reads the filesystem directly. The rules replicate
//! `markitai_core::output::check_path` and `report_store::resolve_path` exactly;
//! on Windows each resolved path, which the file system spells, is reused.
use std::cell::RefCell;
use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy)]
enum Observed {
    /// NotFound, NotADirectory or PermissionDenied: kept as a plain component.
    Unavailable,
    Present {
        symlink: bool,
        root_owned: bool,
    },
}

#[derive(Default)]
struct Memo {
    depth: usize,
    observed: HashMap<PathBuf, Observed>,
    #[cfg(not(windows))]
    links: HashMap<PathBuf, PathBuf>,
    #[cfg(windows)]
    resolved: HashMap<PathBuf, PathBuf>,
}

thread_local! {
    static MEMO: RefCell<Option<Memo>> = const { RefCell::new(None) };
}

/// Reuse filesystem observations until this guard (and any outer one) drops.
/// Only scopes that do not modify the filesystem may hold it.
#[must_use]
pub(super) struct Scope(());

impl Scope {
    pub(super) fn enter() -> Self {
        MEMO.with(|memo| memo.borrow_mut().get_or_insert_with(Memo::default).depth += 1);
        Self(())
    }
}

/// Whether this thread currently reuses observations; writers assert it is not.
#[cfg(test)]
pub(super) fn active() -> bool {
    MEMO.with(|memo| memo.borrow().is_some())
}

impl Drop for Scope {
    fn drop(&mut self) {
        MEMO.with(|memo| {
            let mut memo = memo.borrow_mut();
            if let Some(active) = memo.as_mut() {
                active.depth -= 1;
                if active.depth == 0 {
                    *memo = None;
                }
            }
        });
    }
}

fn observe(path: &Path) -> io::Result<Observed> {
    if let Some(found) = MEMO.with(|memo| {
        memo.borrow()
            .as_ref()
            .and_then(|memo| memo.observed.get(path).copied())
    }) {
        return Ok(found);
    }
    let found = match fs::symlink_metadata(path) {
        Ok(metadata) => Observed::Present {
            symlink: metadata.file_type().is_symlink(),
            root_owned: markitai_core::platform::root_owned(&metadata),
        },
        Err(error)
            if matches!(
                error.kind(),
                io::ErrorKind::NotFound
                    | io::ErrorKind::NotADirectory
                    | io::ErrorKind::PermissionDenied
            ) =>
        {
            Observed::Unavailable
        }
        // Other errors are never cached; each caller handles them as before.
        Err(error) => return Err(error),
    };
    MEMO.with(|memo| {
        if let Some(memo) = memo.borrow_mut().as_mut() {
            memo.observed.insert(path.to_owned(), found);
        }
    });
    Ok(found)
}

#[cfg(not(windows))]
fn read_link(path: &Path) -> io::Result<PathBuf> {
    if let Some(target) = MEMO.with(|memo| {
        memo.borrow()
            .as_ref()
            .and_then(|memo| memo.links.get(path).cloned())
    }) {
        return Ok(target);
    }
    let target = fs::read_link(path)?;
    MEMO.with(|memo| {
        if let Some(memo) = memo.borrow_mut().as_mut() {
            memo.links.insert(path.to_owned(), target.clone());
        }
    });
    Ok(target)
}

/// `markitai_core::output::check_path`: no symlink ancestor unless allowed,
/// except root-owned system links above the final path.
pub(super) fn symlinks_permitted(path: &Path, allow_symlinks: bool) -> io::Result<bool> {
    if allow_symlinks {
        return Ok(true);
    }
    let absolute = std::path::absolute(path)?;
    for ancestor in absolute.ancestors() {
        // The original ignores lookup errors for this policy check.
        if let Ok(Observed::Present {
            symlink: true,
            root_owned,
        }) = observe(ancestor)
        {
            if ancestor != absolute && root_owned {
                continue;
            }
            return Ok(false);
        }
    }
    Ok(true)
}

/// `report_store::resolve_path`.
pub(super) fn resolve(path: &Path) -> io::Result<PathBuf> {
    #[cfg(windows)]
    {
        if let Some(found) = MEMO.with(|memo| {
            memo.borrow()
                .as_ref()
                .and_then(|memo| memo.resolved.get(path).cloned())
        }) {
            return Ok(found);
        }
        let resolved = markitai_core::platform::resolve(path)?;
        MEMO.with(|memo| {
            if let Some(memo) = memo.borrow_mut().as_mut() {
                memo.resolved.insert(path.to_owned(), resolved.clone());
            }
        });
        Ok(resolved)
    }
    #[cfg(not(windows))]
    {
        let absolute = std::path::absolute(path)?;
        let mut resolved = PathBuf::new();
        resolve_components(&absolute, &mut resolved, &mut Default::default(), 0)?;
        Ok(resolved)
    }
}

#[cfg(not(windows))]
fn resolve_components(
    path: &Path,
    resolved: &mut PathBuf,
    active: &mut std::collections::HashSet<PathBuf>,
    depth: usize,
) -> io::Result<()> {
    use std::path::Component;
    if depth > 256 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Report path has too many nested symbolic links",
        ));
    }
    for component in path.components() {
        match component {
            Component::Prefix(_) | Component::RootDir => resolved.push(component),
            Component::CurDir => (),
            Component::ParentDir => {
                resolved.pop();
            }
            Component::Normal(name) => {
                let candidate = resolved.join(name);
                if matches!(
                    observe(&candidate)?,
                    Observed::Present { symlink: true, .. }
                ) {
                    if !active.insert(candidate.clone()) {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            "Report path contains a symbolic link loop",
                        ));
                    }
                    let target = read_link(&candidate)?;
                    resolve_components(&target, resolved, active, depth + 1)?;
                    active.remove(&candidate);
                } else {
                    resolved.push(name);
                }
            }
        }
    }
    Ok(())
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::fs::symlink;

    #[test]
    fn scoped_results_match_the_original_resolver_and_policy() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        fs::create_dir_all(root.join("real/sub")).unwrap();
        symlink(root.join("real"), root.join("link")).unwrap();
        symlink("loop-b", root.join("loop-a")).unwrap();
        symlink("loop-a", root.join("loop-b")).unwrap();
        symlink("../real/sub", root.join("real/relative")).unwrap();
        let paths = [
            root.join("real/sub/file.md"),
            root.join("link/sub/file.md"),
            root.join("real/relative/x"),
            root.join("real/../link/./sub"),
            root.join("missing/deeper"),
            root.join("loop-a/x"),
        ];
        for scoped in [false, true] {
            let _scope = scoped.then(Scope::enter);
            for path in &paths {
                for _ in 0..2 {
                    let expected = crate::report_store::resolve_path(path).map_err(|e| e.kind());
                    assert_eq!(resolve(path).map_err(|e| e.kind()), expected, "{path:?}");
                    for allow in [false, true] {
                        assert_eq!(
                            symlinks_permitted(path, allow).unwrap(),
                            markitai_core::output::check_path(path, allow).is_ok(),
                            "{path:?} {allow}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn observations_are_reused_only_inside_a_scope() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("entry");
        fs::create_dir(&target).unwrap();
        {
            let _outer = Scope::enter();
            assert!(symlinks_permitted(&target, false).unwrap());
            fs::remove_dir(&target).unwrap();
            symlink(dir.path(), &target).unwrap();
            {
                let _nested = Scope::enter();
                // The operation keeps its first observation.
                assert!(symlinks_permitted(&target, false).unwrap());
            }
            assert!(symlinks_permitted(&target, false).unwrap());
        }
        // A new operation observes the replaced entry.
        assert!(!symlinks_permitted(&target, false).unwrap());
        MEMO.with(|memo| assert!(memo.borrow().is_none()));
    }
}
