use super::{Error, Result};
use std::ffi::OsStr;
use std::fs::{self, File, Metadata, OpenOptions, TryLockError};
use std::io;
use std::path::{Component, Path, PathBuf};

const MAX_MEMBERS: usize = 2;
const MAX_MEMBER_BYTES: usize = 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Identity {
    device: u64,
    inode: u64,
}

struct Directory {
    path: PathBuf,
    identity: Identity,
    private: bool,
}

struct MemberFile {
    path: PathBuf,
    identity: Identity,
    file: File,
    acquired: bool,
}

impl MemberFile {
    fn acquire(&mut self) -> Result<()> {
        self.file.try_lock().map_err(|error| match error {
            TryLockError::WouldBlock => Error::Busy,
            TryLockError::Error(error) => Error::Io(error),
        })?;
        self.acquired = true;
        Ok(())
    }
}

impl Drop for MemberFile {
    fn drop(&mut self) {
        // Planning-only descriptors never unlock. Partial acquisition and path
        // validation errors release exactly the members successfully claimed.
        if self.acquired {
            let _ = self.file.unlock();
        }
    }
}

/// Each descriptor excludes only the document names this conversion can publish.
/// Lock entries stay in place after release; replacing their inode would split
/// cooperating writers into separate lock domains.
pub(crate) struct MemberLeases {
    prepared: PreparedMembers,
}

impl MemberLeases {
    pub(crate) fn acquire(parent: &Path, members: &[String], allow_symlinks: bool) -> Result<Self> {
        let mut prepared = PreparedMembers::open(parent, members, allow_symlinks)?;
        for member in &mut prepared.files {
            member.acquire()?;
        }
        prepared.check_paths()?;
        Ok(Self { prepared })
    }

    pub(crate) fn parent(&self) -> &Path {
        &self.prepared.parent
    }

    pub(crate) fn members(&self) -> &[String] {
        &self.prepared.members
    }

    pub(crate) fn keys(&self) -> Vec<(u64, u64)> {
        self.prepared.keys()
    }

    pub(crate) fn validate_member(&self, path: &Path) -> Result<PathBuf> {
        self.prepared.validate_member(path)
    }
}

/// Return process-local reservation identities without claiming a queued task.
/// These keys are planning hints, never persisted publication authority.
pub(crate) fn reserve_keys(
    parent: &Path,
    members: &[String],
    allow_symlinks: bool,
) -> Result<Vec<(u64, u64)>> {
    let prepared = PreparedMembers::open(parent, members, allow_symlinks)?;
    Ok(prepared.keys())
}

struct PreparedMembers {
    original_parent: PathBuf,
    parent: PathBuf,
    parent_identity: Identity,
    members: Vec<String>,
    directories: Vec<Directory>,
    files: Vec<MemberFile>,
    allow_symlinks: bool,
}

impl PreparedMembers {
    fn open(parent: &Path, members: &[String], allow_symlinks: bool) -> Result<Self> {
        if members.is_empty() || members.len() > MAX_MEMBERS {
            return Err(Error::Invalid(
                "one or two output members are required".into(),
            ));
        }
        for member in members {
            validate_name(member)?;
        }
        // Native file identity is required even for an empty destination. Do not
        // start model work with a weaker, pathname-only ownership guarantee.
        if !cfg!(unix) {
            return Err(unsupported());
        }
        let original_parent = std::path::absolute(parent)?;
        let parent = prepare_namespace_parent(&original_parent, allow_symlinks)?;
        let parent_metadata = directory_metadata(&parent, false)?;
        let parent_identity = identity(&parent_metadata)?;
        let mut directories = Vec::with_capacity(3);
        let mut metadata_path = parent.clone();
        for (name, private) in [(".markitai", false), ("ownership", true), ("members", true)] {
            metadata_path.push(name);
            create_metadata_directory(&metadata_path)?;
            let metadata = directory_metadata(&metadata_path, private)?;
            let directory_identity = identity(&metadata)?;
            if directory_identity.device != parent_identity.device {
                return Err(Error::Invalid(
                    "output claim metadata must share the output filesystem".into(),
                ));
            }
            directories.push(Directory {
                path: metadata_path.clone(),
                identity: directory_identity,
                private,
            });
        }
        let mut members = members.to_vec();
        members.sort_unstable();
        members.dedup();
        let mut prepared = Self {
            original_parent,
            parent,
            parent_identity,
            members,
            directories,
            files: Vec::with_capacity(MAX_MEMBERS),
            allow_symlinks,
        };
        for member in &prepared.members {
            let path = metadata_path.join(member);
            // Check before opening so special files cannot block the acquisition.
            match fs::symlink_metadata(&path) {
                Ok(metadata) => {
                    validate_lock_metadata(&metadata, parent_identity.device)?;
                    let existing_identity = identity(&metadata)?;
                    if prepared
                        .files
                        .iter()
                        .any(|held| held.identity == existing_identity)
                    {
                        continue;
                    }
                }
                Err(error) if error.kind() == io::ErrorKind::NotFound => (),
                Err(error) => return Err(error.into()),
            }
            let mut options = OpenOptions::new();
            options.read(true).write(true).create(true).truncate(false);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            let file = options.open(&path)?;
            let metadata = file.metadata()?;
            validate_lock_metadata(&metadata, parent_identity.device)?;
            let file_identity = identity(&metadata)?;
            let path_metadata = fs::symlink_metadata(&path)?;
            validate_lock_metadata(&path_metadata, parent_identity.device)?;
            if identity(&path_metadata)? != file_identity {
                return Err(Error::Invalid(
                    "output lock changed during acquisition".into(),
                ));
            }
            // Case or Unicode aliases on this filesystem can name the same lock.
            // Do not acquire a second independent descriptor for our own inode.
            if prepared
                .files
                .iter()
                .any(|held| held.identity == file_identity)
            {
                continue;
            }
            prepared.files.push(MemberFile {
                path,
                identity: file_identity,
                file,
                acquired: false,
            });
        }
        prepared.check_paths()?;
        Ok(prepared)
    }

    fn keys(&self) -> Vec<(u64, u64)> {
        self.files
            .iter()
            .map(|member| (member.identity.device, member.identity.inode))
            .collect()
    }

    /// Resolve the parent only: a permitted leaf symlink remains the entry a
    /// publication replaces, rather than becoming an authority over its target.
    pub(crate) fn validate_member(&self, path: &Path) -> Result<PathBuf> {
        let name = path
            .file_name()
            .and_then(OsStr::to_str)
            .ok_or_else(|| Error::Invalid("output member has no valid filename".into()))?;
        validate_name(name)?;
        if !self.members.iter().any(|member| member == name) {
            return Err(Error::Invalid(
                "output path is outside the claimed member set".into(),
            ));
        }
        self.check_paths()?;
        check_policy(path, self.allow_symlinks)?;
        let requested_parent = path.parent().unwrap_or_else(|| Path::new(""));
        if crate::report_store::resolve_path(requested_parent)? != self.parent {
            return Err(Error::Invalid(
                "output path is outside the claimed parent".into(),
            ));
        }
        Ok(self.parent.join(name))
    }

    fn check_paths(&self) -> Result<()> {
        check_policy(&self.original_parent, self.allow_symlinks)?;
        if crate::report_store::resolve_path(&self.original_parent)? != self.parent
            || identity(&directory_metadata(&self.parent, false)?)? != self.parent_identity
        {
            return Err(Error::Invalid("claimed output parent changed".into()));
        }
        for directory in &self.directories {
            let metadata = directory_metadata(&directory.path, directory.private)?;
            if identity(&metadata)? != directory.identity {
                return Err(Error::Invalid("output claim directory changed".into()));
            }
        }
        for held in &self.files {
            let metadata = fs::symlink_metadata(&held.path)?;
            validate_lock_metadata(&metadata, self.parent_identity.device)?;
            if identity(&metadata)? != held.identity
                || identity(&held.file.metadata()?)? != held.identity
            {
                return Err(Error::Invalid("held output lock was replaced".into()));
            }
        }
        Ok(())
    }
}

fn validate_name(name: &str) -> Result<()> {
    let mut components = Path::new(name).components();
    if name.len() > MAX_MEMBER_BYTES
        || name.contains('\0')
        || !matches!(components.next(), Some(Component::Normal(value)) if value == OsStr::new(name))
        || components.next().is_some()
    {
        return Err(Error::Invalid(
            "output member must be a bounded direct filename".into(),
        ));
    }
    Ok(())
}

fn check_policy(path: &Path, allow_symlinks: bool) -> Result<()> {
    markitai_core::output::check_path(path, allow_symlinks)
        .map_err(|_| Error::Invalid("output path violates the symlink policy".into()))
}

fn unsupported() -> Error {
    Error::Invalid("output ownership requires supported native file identity".into())
}

fn identity(metadata: &Metadata) -> Result<Identity> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        Ok(Identity {
            device: metadata.dev(),
            inode: metadata.ino(),
        })
    }
    #[cfg(not(unix))]
    {
        let _ = metadata;
        Err(unsupported())
    }
}

fn private_metadata(metadata: &Metadata) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        metadata.permissions().mode() & 0o077 == 0
    }
    #[cfg(not(unix))]
    {
        let _ = metadata;
        false
    }
}

fn directory_metadata(path: &Path, private: bool) -> Result<Metadata> {
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.is_dir() || (private && !private_metadata(&metadata)) {
        return Err(Error::Invalid(
            "output claim directory is not a permitted regular directory".into(),
        ));
    }
    Ok(metadata)
}

fn validate_lock_metadata(metadata: &Metadata, device: u64) -> Result<()> {
    if !metadata.is_file() || !private_metadata(metadata) || identity(metadata)?.device != device {
        return Err(Error::Invalid(
            "output lock must be a private regular file on the output filesystem".into(),
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if metadata.nlink() != 1 {
            return Err(Error::Invalid(
                "output lock must not have multiple hard links".into(),
            ));
        }
    }
    Ok(())
}

// Shared only with namespace preparation. Ancestor creation keeps the original
// immediate synchronization path; the grouped path begins below this parent.
pub(crate) fn prepare_namespace_parent(parent: &Path, allow_symlinks: bool) -> Result<PathBuf> {
    let original_parent = std::path::absolute(parent)?;
    check_policy(&original_parent, allow_symlinks)?;
    let planned_parent = crate::report_store::resolve_path(&original_parent)?;
    create_output_directory(&planned_parent, &mut sync_new_directory)?;
    check_policy(&original_parent, allow_symlinks)?;
    let parent = fs::canonicalize(&original_parent)?;
    if parent != planned_parent {
        return Err(Error::Invalid(
            "output parent changed while acquiring a claim".into(),
        ));
    }
    Ok(parent)
}

fn create_metadata_directory(path: &Path) -> Result<()> {
    let mut builder = fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    match builder.create(path) {
        Ok(()) => sync_new_directory(path),
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => Ok(()),
        Err(error) => Err(error.into()),
    }
}

/// Create every missing output ancestor of these parents before any name probe,
/// claim or dispatch. Each newly created directory and its parent receive the
/// same host synchronization as the immediate path, but one media fence per
/// volume covers the whole set. Existing directories keep their original
/// treatment, and each parent's policy and resolved identity are rechecked. An
/// error, including a failed commit, acknowledges nothing and admits no work.
pub(crate) fn prepare_output_ancestors<'a>(
    parents: impl IntoIterator<Item = &'a Path>,
    allow_symlinks: bool,
) -> Result<()> {
    let mut group = super::sync_group::SyncGroup::new();
    stage_output_ancestors(parents, allow_symlinks, |path| {
        let mut options = fs::OpenOptions::new();
        options.read(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW);
        }
        Ok(group.stage(&options.open(path)?)?)
    })?;
    group.commit()?;
    Ok(())
}

fn stage_output_ancestors<'a>(
    parents: impl IntoIterator<Item = &'a Path>,
    allow_symlinks: bool,
    mut stage: impl FnMut(&Path) -> Result<()>,
) -> Result<()> {
    let mut created = Vec::new();
    let mut planned_parents = Vec::new();
    let mut prepared = std::collections::HashSet::new();
    for parent in parents {
        let original = std::path::absolute(parent)?;
        if !prepared.insert(original.clone()) {
            continue;
        }
        check_policy(&original, allow_symlinks)?;
        let planned = crate::report_store::resolve_path(&original)?;
        create_output_directory(&planned, &mut |directory| {
            created.push(directory.to_owned());
            Ok(())
        })?;
        check_policy(&original, allow_symlinks)?;
        if fs::canonicalize(&original)? != planned {
            return Err(Error::Invalid(
                "output parent changed while preparing ancestors".into(),
            ));
        }
        planned_parents.push(planned);
    }
    // Stage only after every creation, so each directory's synchronization
    // covers all entries created in it, including later nested children.
    // Existing directories on each chain below the parents' common prefix are
    // staged too: a crash before an earlier commit could have left them
    // unfenced, and their existence alone is not durability.
    if cfg!(unix) {
        let common = planned_parents.split_first().map(|(first, rest)| {
            rest.iter().fold(first.clone(), |prefix, path| {
                prefix
                    .components()
                    .zip(path.components())
                    .take_while(|(a, b)| a == b)
                    .map(|(a, _)| a)
                    .collect::<PathBuf>()
            })
        });
        let mut staged = std::collections::BTreeSet::new();
        for directory in &created {
            staged.insert(directory.clone());
            staged.extend(directory.parent().map(Path::to_owned));
        }
        if let Some(common) = &common {
            for parent in &planned_parents {
                staged.extend(
                    parent
                        .ancestors()
                        .take_while(|path| path.starts_with(common))
                        .map(Path::to_owned),
                );
            }
        }
        for path in &staged {
            stage(path)?;
        }
    }
    Ok(())
}

fn create_output_directory(
    path: &Path,
    created: &mut dyn FnMut(&Path) -> Result<()>,
) -> Result<()> {
    let mut missing = Vec::new();
    let mut current = path;
    loop {
        match fs::symlink_metadata(current) {
            Ok(metadata) if metadata.is_dir() => break,
            Ok(_) => return Err(Error::Invalid("output parent is not a directory".into())),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                missing.push(current.to_owned());
                current = current
                    .parent()
                    .ok_or_else(|| Error::Invalid("output has no directory root".into()))?;
            }
            Err(error) => return Err(error.into()),
        }
    }
    for directory in missing.into_iter().rev() {
        match fs::create_dir(&directory) {
            Ok(()) => created(&directory)?,
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                directory_metadata(&directory, false)?;
            }
            Err(error) => return Err(error.into()),
        }
    }
    Ok(())
}

fn sync_new_directory(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        File::open(path)?.sync_all()?;
        if let Some(parent) = path.parent() {
            File::open(parent)?.sync_all()?;
        }
    }
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::fs::{MetadataExt, PermissionsExt, symlink};
    use std::process::{Child, Command, Stdio};
    use std::time::{Duration, Instant};

    #[test]
    fn grouped_ancestors_stage_each_new_directory_and_parent_once_then_commit() {
        let dir = tempfile::tempdir().unwrap();
        let root = fs::canonicalize(dir.path()).unwrap();
        fs::create_dir(root.join("existing")).unwrap();
        let parents = [
            root.join("a/b/c"),
            root.join("a/b/d"),
            root.join("existing"),
            root.join("existing/new"),
            root.join("a/b/c"),
        ];
        let mut staged = Vec::new();
        stage_output_ancestors(parents.iter().map(PathBuf::as_path), false, |path| {
            // Every directory already exists: staging follows all creation.
            assert!(parents.iter().all(|parent| parent.is_dir()));
            staged.push(path.to_owned());
            Ok(())
        })
        .unwrap();
        let count = staged.len();
        staged.sort();
        staged.dedup();
        assert_eq!(staged.len(), count, "a directory was staged twice");
        let expected: std::collections::BTreeSet<_> =
            ["", "a", "a/b", "a/b/c", "a/b/d", "existing", "existing/new"]
                .iter()
                .map(|name| {
                    if name.is_empty() {
                        root.clone()
                    } else {
                        root.join(name)
                    }
                })
                .collect();
        assert_eq!(
            staged
                .into_iter()
                .collect::<std::collections::BTreeSet<_>>(),
            expected
        );
        for parent in &parents {
            assert!(parent.is_dir());
        }
        // A later run finds everything established and stages nothing.
        // A later run re-stages the established chains below the common
        // prefix: existence after a possible crash is not durability.
        let mut again = std::collections::BTreeSet::new();
        stage_output_ancestors(parents.iter().map(PathBuf::as_path), false, |path| {
            assert!(again.insert(path.to_owned()), "staged twice");
            Ok(())
        })
        .unwrap();
        assert_eq!(again, expected);
        // The chain stops at the parents' common prefix inside the output tree.
        let deep = [root.join("x/y/p1"), root.join("x/y/p2/q")];
        let mut bounded = std::collections::BTreeSet::new();
        stage_output_ancestors(deep.iter().map(PathBuf::as_path), false, |path| {
            bounded.insert(path.to_owned());
            Ok(())
        })
        .unwrap();
        let expected_deep: std::collections::BTreeSet<_> =
            ["x", "x/y", "x/y/p1", "x/y/p2", "x/y/p2/q"]
                .iter()
                .map(|name| root.join(name))
                .chain([root.clone()])
                .collect();
        // Newly created x and its parent root are staged; above the prefix
        // x/y nothing pre-existing is added on a rerun.
        assert_eq!(bounded, expected_deep);
        let mut rerun = std::collections::BTreeSet::new();
        stage_output_ancestors(deep.iter().map(PathBuf::as_path), false, |path| {
            rerun.insert(path.to_owned());
            Ok(())
        })
        .unwrap();
        let expected_rerun: std::collections::BTreeSet<_> = ["x/y", "x/y/p1", "x/y/p2", "x/y/p2/q"]
            .iter()
            .map(|name| root.join(name))
            .collect();
        assert_eq!(rerun, expected_rerun);
        // The real group completes its media fence for new directories.
        let real = root.join("real-group/x");
        prepare_output_ancestors([real.as_path()], false).unwrap();
        assert!(real.is_dir());
    }

    #[test]
    fn grouped_ancestors_fail_on_policy_file_or_staging_errors() {
        let dir = tempfile::tempdir().unwrap();
        let root = fs::canonicalize(dir.path()).unwrap();
        fs::create_dir(root.join("real")).unwrap();
        symlink(root.join("real"), root.join("link")).unwrap();
        fs::write(root.join("file"), b"x").unwrap();
        let linked = root.join("link/child");
        assert!(stage_output_ancestors([linked.as_path()], false, |_| Ok(())).is_err());
        assert!(!root.join("real/child").exists());
        assert!(prepare_output_ancestors([linked.as_path()], false).is_err());
        // Allowed symlinks resolve first, so staging never opens a link itself.
        prepare_output_ancestors([linked.as_path()], true).unwrap();
        assert!(root.join("real/child").is_dir());
        let under_file = root.join("file/child");
        assert!(stage_output_ancestors([under_file.as_path()], false, |_| Ok(())).is_err());
        let staging = root.join("staging-fails");
        let error = stage_output_ancestors([staging.as_path()], false, |_| {
            Err(Error::Invalid("injected host sync failure".into()))
        })
        .unwrap_err();
        assert!(error.to_string().contains("injected host sync failure"));
    }

    fn names(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    fn lock_path(parent: &Path, member: &str) -> PathBuf {
        parent.join(".markitai/ownership/members").join(member)
    }

    struct ChildGuard(Child);

    impl Drop for ChildGuard {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }

    #[test]
    fn independent_members_share_a_parent_but_overlapping_families_are_busy() {
        let temp = tempfile::tempdir().unwrap();
        let first =
            MemberLeases::acquire(temp.path(), &names(&["x.md", "x.llm.md"]), false).unwrap();
        let independent =
            MemberLeases::acquire(temp.path(), &names(&["y.md", "y.llm.md"]), false).unwrap();
        assert!(matches!(
            MemberLeases::acquire(temp.path(), &names(&["x.llm.md", "x.llm.llm.md"]), false),
            Err(Error::Busy)
        ));
        drop(first);
        assert!(
            MemberLeases::acquire(temp.path(), &names(&["x.llm.md", "x.llm.llm.md"]), false)
                .is_ok()
        );
        drop(independent);
    }

    #[test]
    fn partial_acquisition_releases_prior_locks_without_removing_entries() {
        let temp = tempfile::tempdir().unwrap();
        let _busy = MemberLeases::acquire(temp.path(), &names(&["z.md"]), false).unwrap();
        assert!(matches!(
            MemberLeases::acquire(temp.path(), &names(&["a.md", "z.md"]), false),
            Err(Error::Busy)
        ));
        let before = fs::metadata(lock_path(temp.path(), "a.md")).unwrap().ino();
        let available = MemberLeases::acquire(temp.path(), &names(&["a.md"]), false).unwrap();
        drop(available);
        assert_eq!(
            fs::metadata(lock_path(temp.path(), "a.md")).unwrap().ino(),
            before
        );
    }

    #[test]
    fn invalid_member_sets_do_not_create_the_output_tree() {
        let temp = tempfile::tempdir().unwrap();
        let output = temp.path().join("missing/deep/output");
        for invalid in [
            "",
            ".",
            "..",
            "../escape.md",
            "sub/file.md",
            "/absolute.md",
            "name/",
            "bad\0name",
        ] {
            assert!(MemberLeases::acquire(&output, &names(&[invalid]), false).is_err());
            assert!(!output.exists());
        }
        assert!(MemberLeases::acquire(&output, &[], false).is_err());
        assert!(MemberLeases::acquire(&output, &names(&["a", "b", "c"]), false).is_err());
        assert!(
            MemberLeases::acquire(&output, &["a".repeat(MAX_MEMBER_BYTES + 1)], false).is_err()
        );
        assert!(!output.exists());
    }

    #[test]
    fn parent_aliases_share_locks_and_original_spelling_obeys_policy() {
        let temp = tempfile::tempdir().unwrap();
        let physical = temp.path().join("physical");
        let alias = temp.path().join("alias");
        fs::create_dir(&physical).unwrap();
        symlink(&physical, &alias).unwrap();
        assert!(MemberLeases::acquire(&alias, &names(&["a.md"]), false).is_err());
        assert!(!physical.join(".markitai").exists());
        let lease = MemberLeases::acquire(&alias, &names(&["a.md"]), true).unwrap();
        assert_eq!(lease.parent(), fs::canonicalize(&physical).unwrap());
        assert!(matches!(
            MemberLeases::acquire(&physical, &names(&["a.md"]), false),
            Err(Error::Busy)
        ));
        assert_eq!(
            lease.validate_member(&alias.join("a.md")).unwrap(),
            lease.parent().join("a.md")
        );
    }

    #[test]
    fn metadata_symlinks_and_nonprivate_locks_never_gain_authority() {
        let temp = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        symlink(outside.path(), temp.path().join(".markitai")).unwrap();
        assert!(MemberLeases::acquire(temp.path(), &names(&["a.md"]), true).is_err());
        assert!(!outside.path().join("ownership").exists());
        fs::remove_file(temp.path().join(".markitai")).unwrap();
        let lease = MemberLeases::acquire(temp.path(), &names(&["a.md"]), false).unwrap();
        let path = lock_path(temp.path(), "a.md");
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        for suffix in [".markitai/ownership", ".markitai/ownership/members"] {
            assert_eq!(
                fs::metadata(temp.path().join(suffix))
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o700
            );
        }
        drop(lease);
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
        assert!(MemberLeases::acquire(temp.path(), &names(&["a.md"]), false).is_err());
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o644
        );
        fs::remove_file(&path).unwrap();
        let outside_file = outside.path().join("untouched");
        fs::write(&outside_file, b"private data").unwrap();
        symlink(&outside_file, &path).unwrap();
        assert!(MemberLeases::acquire(temp.path(), &names(&["a.md"]), true).is_err());
        assert_eq!(fs::read(outside_file).unwrap(), b"private data");
    }

    #[test]
    fn a_replaced_lock_or_unclaimed_path_is_rejected_before_publication() {
        let temp = tempfile::tempdir().unwrap();
        let lease = MemberLeases::acquire(temp.path(), &names(&["a.md"]), false).unwrap();
        assert!(lease.validate_member(&temp.path().join("b.md")).is_err());
        assert!(
            lease
                .validate_member(&temp.path().join("elsewhere/a.md"))
                .is_err()
        );
        let path = lock_path(temp.path(), "a.md");
        fs::rename(&path, path.with_extension("old")).unwrap();
        let replacement = tempfile::NamedTempFile::new_in(path.parent().unwrap()).unwrap();
        replacement.persist(&path).unwrap();
        assert!(lease.validate_member(&temp.path().join("a.md")).is_err());
    }

    #[test]
    fn permitted_leaf_symlinks_are_not_resolved_into_their_targets() {
        let temp = tempfile::tempdir().unwrap();
        let outside = tempfile::NamedTempFile::new().unwrap();
        let target = temp.path().join("a.md");
        symlink(outside.path(), &target).unwrap();
        let lease = MemberLeases::acquire(temp.path(), &names(&["a.md"]), true).unwrap();
        assert_eq!(
            lease.validate_member(&target).unwrap(),
            lease.parent().join("a.md")
        );
        drop(lease);
        let lease = MemberLeases::acquire(temp.path(), &names(&["a.md"]), false).unwrap();
        assert!(lease.validate_member(&target).is_err());
    }

    #[test]
    fn actual_filesystem_aliases_do_not_self_conflict() {
        let temp = tempfile::tempdir().unwrap();
        for pair in [["Alias.md", "alias.md"], ["caf\u{e9}.md", "cafe\u{301}.md"]] {
            let first = MemberLeases::acquire(temp.path(), &names(&[pair[0]]), false).unwrap();
            let first_identity =
                identity(&fs::metadata(lock_path(temp.path(), pair[0])).unwrap()).unwrap();
            let alias_identity = fs::metadata(lock_path(temp.path(), pair[1]))
                .ok()
                .map(|value| identity(&value).unwrap());
            if alias_identity == Some(first_identity) {
                assert_eq!(
                    reserve_keys(temp.path(), &names(&[pair[0]]), false).unwrap(),
                    reserve_keys(temp.path(), &names(&[pair[1]]), false).unwrap()
                );
                assert!(matches!(
                    MemberLeases::acquire(temp.path(), &names(&[pair[1]]), false),
                    Err(Error::Busy)
                ));
                drop(first);
                let both = MemberLeases::acquire(temp.path(), &names(&pair), false).unwrap();
                assert_eq!(both.keys().len(), 1);
                for name in pair {
                    assert_eq!(
                        both.validate_member(&temp.path().join(name)).unwrap(),
                        both.parent().join(name)
                    );
                }
            } else {
                let second = MemberLeases::acquire(temp.path(), &names(&[pair[1]]), false).unwrap();
                assert_ne!(first.keys(), second.keys());
                assert_ne!(
                    reserve_keys(temp.path(), &names(&[pair[0]]), false).unwrap(),
                    reserve_keys(temp.path(), &names(&[pair[1]]), false).unwrap()
                );
            }
        }
    }

    #[test]
    fn reservation_keys_exist_without_output_or_an_exclusive_lease() {
        let temp = tempfile::tempdir().unwrap();
        let members = names(&["planned.md", "planned.llm.md"]);
        let keys = reserve_keys(temp.path(), &members, false).unwrap();
        assert_eq!(keys.len(), 2);
        assert!(
            members
                .iter()
                .all(|member| !temp.path().join(member).exists())
        );
        let held = MemberLeases::acquire(temp.path(), &members, false).unwrap();
        assert_eq!(keys, held.keys());
        assert_eq!(reserve_keys(temp.path(), &members, false).unwrap(), keys);
        assert!(matches!(
            MemberLeases::acquire(temp.path(), &members, false),
            Err(Error::Busy)
        ));
        drop(held);
        assert_eq!(
            MemberLeases::acquire(temp.path(), &members, false)
                .unwrap()
                .keys(),
            keys
        );
    }

    #[test]
    fn killed_process_releases_the_same_stable_member_inode() {
        let temp = tempfile::tempdir().unwrap();
        let ready = temp.path().join("ready");
        let reservation = reserve_keys(temp.path(), &names(&["held.md"]), false).unwrap();
        let mut child = ChildGuard(
            Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "output_claims::leases::tests::child_holds_member",
                    "--ignored",
                    "--nocapture",
                ])
                .env("MARKITAI_TEST_MEMBER_PARENT", temp.path())
                .env("MARKITAI_TEST_MEMBER_READY", &ready)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap(),
        );
        let deadline = Instant::now() + Duration::from_secs(10);
        while !ready.exists() {
            if let Some(status) = child.0.try_wait().unwrap() {
                panic!("lease helper exited before readiness: {status}");
            }
            if Instant::now() >= deadline {
                panic!("lease helper did not become ready");
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        let before = fs::metadata(lock_path(temp.path(), "held.md"))
            .unwrap()
            .ino();
        assert_eq!(
            reserve_keys(temp.path(), &names(&["held.md"]), false).unwrap(),
            reservation
        );
        let busy = MemberLeases::acquire(temp.path(), &names(&["held.md"]), false);
        let independent = MemberLeases::acquire(temp.path(), &names(&["other.md"]), false);
        child.0.kill().unwrap();
        child.0.wait().unwrap();
        assert!(matches!(busy, Err(Error::Busy)));
        assert!(independent.is_ok());
        let reacquired = MemberLeases::acquire(temp.path(), &names(&["held.md"]), false).unwrap();
        assert_eq!(reacquired.keys(), reservation);
        assert_eq!(
            fs::metadata(lock_path(temp.path(), "held.md"))
                .unwrap()
                .ino(),
            before
        );
    }

    #[test]
    #[ignore = "subprocess helper invoked by killed_process_releases_the_same_stable_member_inode"]
    fn child_holds_member() {
        let Some(parent) = std::env::var_os("MARKITAI_TEST_MEMBER_PARENT") else {
            return;
        };
        let ready = std::env::var_os("MARKITAI_TEST_MEMBER_READY").unwrap();
        let _lease =
            MemberLeases::acquire(Path::new(&parent), &names(&["held.md"]), false).unwrap();
        fs::write(ready, b"ready").unwrap();
        loop {
            std::thread::sleep(Duration::from_secs(1));
        }
    }
}

#[cfg(all(test, target_os = "linux"))]
use crate::file_lock_test as fork_lock_test;

#[cfg(all(test, target_os = "linux"))]
mod inherited_lock_tests {
    use super::*;
    #[test]
    fn member_family_releases_before_inherited_child_exits() {
        let root = tempfile::tempdir().unwrap();
        let members = vec!["a.md".to_owned(), "a.llm.md".to_owned()];
        let owner = MemberLeases::acquire(root.path(), &members, false).unwrap();
        assert!(matches!(
            MemberLeases::acquire(root.path(), &members, false),
            Err(Error::Busy)
        ));
        // A reservation lookup must not release another active claim.
        reserve_keys(root.path(), &members, false).unwrap();
        assert!(matches!(
            MemberLeases::acquire(root.path(), &members, false),
            Err(Error::Busy)
        ));
        let mut child = fork_lock_test::InheritedChild::start();
        drop(owner);
        let reopened = MemberLeases::acquire(root.path(), &members, false);
        child.finish();
        assert!(reopened.is_ok(), "member locks outlived their owner");
    }
    #[test]
    fn partial_family_acquisition_releases_only_acquired_members() {
        let root = tempfile::tempdir().unwrap();
        let second = vec!["b.md".to_owned()];
        let blocker = MemberLeases::acquire(root.path(), &second, false).unwrap();
        let mut prepared =
            PreparedMembers::open(root.path(), &["a.md".into(), "b.md".into()], false).unwrap();
        prepared.files[0].acquire().unwrap();
        let mut child = fork_lock_test::InheritedChild::start();
        assert!(matches!(prepared.files[1].acquire(), Err(Error::Busy)));
        drop(prepared);
        let first = MemberLeases::acquire(root.path(), &["a.md".into()], false);
        let second_still_busy = matches!(
            MemberLeases::acquire(root.path(), &second, false),
            Err(Error::Busy)
        );
        child.finish();
        assert!(first.is_ok(), "failed acquisition pinned the first member");
        assert!(
            second_still_busy,
            "failed acquisition released someone else's member"
        );
        drop(blocker);
        MemberLeases::acquire(root.path(), &second, false).unwrap();
    }
}
