//! Establish the controlled claim namespace before any admission is dispatched.
use super::{Error, Result, leases, sync_group::SyncGroup};
use std::collections::BTreeMap;
use std::fs::{self, File, Metadata, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};

// This limit bounds extra descriptors, not configured conversion concurrency.
pub(crate) const MAX_NAMESPACE_PARENTS: usize = 16;

struct Directory {
    path: PathBuf,
    file: File,
    identity: (u64, u64),
    private: bool,
}

impl Directory {
    fn open(path: PathBuf, private: bool, device: Option<u64>) -> Result<Self> {
        let before = checked_metadata(&path, private, device)?;
        let mut options = OpenOptions::new();
        options.read(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            // Check the descriptor as well: a substituted FIFO must never block.
            options.custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_NONBLOCK);
        }
        let file = options.open(&path)?;
        let identity = identity(&before)?;
        if identity != self::identity(&file.metadata()?)? {
            return Err(invalid("namespace directory changed while opening"));
        }
        let directory = Self {
            path,
            file,
            identity,
            private,
        };
        directory.validate()?;
        Ok(directory)
    }

    fn validate(&self) -> Result<()> {
        let observed = checked_metadata(&self.path, self.private, Some(self.identity.0))?;
        if identity(&observed)? != self.identity
            || identity(&self.file.metadata()?)? != self.identity
        {
            return Err(invalid("namespace directory identity changed"));
        }
        Ok(())
    }
}

struct Namespace {
    // Parent first; its metadata children follow in parent-before-child order.
    directories: Vec<Directory>,
}

impl Namespace {
    fn prepare(parent: PathBuf) -> Result<Self> {
        let parent = Directory::open(parent, false, None)?;
        let device = parent.identity.0;
        let mut path = parent.path.clone();
        let mut directories = vec![parent];
        for (name, private) in [(".markitai", false), ("ownership", true), ("members", true)] {
            path.push(name);
            create_metadata_directory(&path)?;
            directories.push(Directory::open(path.clone(), private, Some(device))?);
        }
        path.pop();
        path.push("records");
        create_metadata_directory(&path)?;
        directories.push(Directory::open(path, true, Some(device))?);
        let namespace = Self { directories };
        namespace.validate()?;
        Ok(namespace)
    }

    fn validate(&self) -> Result<()> {
        for directory in &self.directories {
            directory.validate()?;
        }
        Ok(())
    }
}

/// A successful prepare creates no dispatch authority. Another initializer's
/// existing directories are retained and fenced exactly like our new ones.
#[must_use = "namespace preparation needs a successful commit before admission"]
pub(crate) struct NamespaceBatch {
    parents: BTreeMap<PathBuf, Namespace>,
}

impl NamespaceBatch {
    pub(crate) fn new() -> Self {
        Self {
            parents: BTreeMap::new(),
        }
    }

    pub(crate) fn prepare(&mut self, parent: &Path, allow_symlinks: bool) -> Result<()> {
        check_policy(parent, allow_symlinks)?;
        let planned = crate::report_store::resolve_path(parent)?;
        if !self.parents.contains_key(&planned) && self.parents.len() == MAX_NAMESPACE_PARENTS {
            return Err(invalid("namespace preparation window is full"));
        }
        // Keep the original output-ancestor creation and immediate barriers.
        // Only the controlled metadata chain below this parent is grouped.
        let parent = leases::prepare_namespace_parent(parent, allow_symlinks)?;
        if parent != planned {
            return Err(invalid(
                "output parent changed during namespace preparation",
            ));
        }
        if let Some(namespace) = self.parents.get(&parent) {
            return namespace.validate();
        }
        let namespace = Namespace::prepare(parent.clone())?;
        self.parents.insert(parent, namespace);
        Ok(())
    }

    pub(crate) fn commit(self) -> Result<PreparedNamespaces> {
        self.commit_with(|directories| {
            let mut sync = SyncGroup::new();
            for directory in directories {
                sync.stage(&directory.file)?;
            }
            sync.commit()?;
            Ok(())
        })
    }

    fn commit_with(
        self,
        synchronize: impl FnOnce(&[&Directory]) -> Result<()>,
    ) -> Result<PreparedNamespaces> {
        for namespace in self.parents.values() {
            namespace.validate()?;
        }
        // Each child and the parent containing its entry must be synchronized.
        // Include AlreadyExists observations; a peer may not yet have fenced them.
        let directories: Vec<_> = self
            .parents
            .values()
            .flat_map(|namespace| namespace.directories.iter().rev())
            .collect();
        synchronize(&directories)?;
        for namespace in self.parents.values() {
            namespace.validate()?;
        }
        Ok(PreparedNamespaces {
            parents: self.parents,
        })
    }
}

/// Own the checked descriptors until this small window has acquired its member
/// leases. This proof is never persisted or shared across unrelated CLI runs.
#[must_use = "validate the prepared parent before and after acquiring its claim"]
pub(crate) struct PreparedNamespaces {
    parents: BTreeMap<PathBuf, Namespace>,
}

impl PreparedNamespaces {
    pub(crate) fn validate(&self, parent: &Path, allow_symlinks: bool) -> Result<()> {
        check_policy(parent, allow_symlinks)?;
        let parent = crate::report_store::resolve_path(parent)?;
        self.parents
            .get(&parent)
            .ok_or_else(|| invalid("output parent was not prepared in this namespace window"))?
            .validate()
    }
}

fn create_metadata_directory(path: &Path) -> Result<()> {
    #[cfg_attr(not(unix), allow(unused_mut))] // Only Unix sets a mode.
    let mut builder = fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    match builder.create(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => Ok(()),
        Err(error) => Err(error.into()),
    }
}

fn checked_metadata(path: &Path, private: bool, device: Option<u64>) -> Result<Metadata> {
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return Err(invalid("namespace must contain regular directories"));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if (private && metadata.mode() & 0o077 != 0)
            || device.is_some_and(|device| metadata.dev() != device)
        {
            return Err(invalid(
                "namespace metadata must be private on the output filesystem",
            ));
        }
    }
    #[cfg(not(unix))]
    let _ = (private, device);
    Ok(metadata)
}

fn identity(metadata: &Metadata) -> Result<(u64, u64)> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        Ok((metadata.dev(), metadata.ino()))
    }
    #[cfg(not(unix))]
    {
        let _ = metadata;
        Err(invalid(
            "namespace preparation requires native file identity",
        ))
    }
}

fn check_policy(path: &Path, allow_symlinks: bool) -> Result<()> {
    markitai_core::output::check_path(path, allow_symlinks)
        .map_err(|_| invalid("namespace path violates the symlink policy"))
}

fn invalid(message: &str) -> Error {
    Error::Invalid(message.into())
}

#[cfg(all(test, unix))]
mod tests;
