//! Establish the controlled claim namespace before any admission is dispatched.
use super::{Error, Result, leases, sync_group::SyncGroup};
use markitai_core::platform::{self, FileId, Status};
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};

// This limit bounds extra descriptors, not configured conversion concurrency.
pub(crate) const MAX_NAMESPACE_PARENTS: usize = 16;

struct Directory {
    path: PathBuf,
    file: File,
    identity: FileId,
    private: bool,
    regular_file: bool,
}

impl Directory {
    fn open(path: PathBuf, private: bool, volume: Option<u64>) -> Result<Self> {
        let identity = checked_metadata(&path, private, volume)?.id();
        // Check the handle as well: a substituted link or FIFO is refused
        // without being followed or blocking.
        let file = platform::open_directory(&path)?;
        if identity != platform::file_status(&file)?.id() {
            return Err(invalid("namespace directory changed while opening"));
        }
        let directory = Self {
            path,
            file,
            identity,
            private,
            regular_file: false,
        };
        directory.validate()?;
        Ok(directory)
    }

    fn stable(path: PathBuf, volume: u64) -> Result<Self> {
        let (file, identity) = super::v2::open_checked(&path, volume, true)?;
        let entry = Self {
            path,
            file,
            identity,
            private: true,
            regular_file: true,
        };
        entry.validate()?;
        Ok(entry)
    }

    fn validate(&self) -> Result<()> {
        let observed = if self.regular_file {
            super::v2::checked_file(&self.path, self.identity.volume)?
        } else {
            checked_metadata(&self.path, self.private, Some(self.identity.volume))?
        };
        if observed.id() != self.identity
            || platform::file_status(&self.file)?.id() != self.identity
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
        let device = parent.identity.volume;
        let mut path = parent.path.clone();
        let mut directories = vec![parent];
        for (name, private) in [(".markitai", false), ("ownership", true)] {
            path.push(name);
            create_metadata_directory(&path)?;
            directories.push(Directory::open(path.clone(), private, Some(device))?);
        }
        let members = path.join("members");
        let legacy = match platform::status(&members) {
            Ok(status) if status.metadata().is_dir() => true,
            Ok(_) => false,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                let mut options = OpenOptions::new();
                options.read(true).write(true).create_new(true);
                platform::private_file(&mut options);
                match platform::open_no_follow(&options, &members) {
                    Ok(_) => false,
                    Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                        platform::status(&members)?.metadata().is_dir()
                    }
                    Err(error) => return Err(error.into()),
                }
            }
            Err(error) => return Err(error.into()),
        };
        if legacy {
            directories.push(Directory::open(members, true, Some(device))?);
        } else {
            directories.push(Directory::stable(members, device)?);
            directories.push(Directory::stable(path.join("epoch-v2"), device)?);
            if platform::status(&path.join("writers-v2")).is_ok() {
                return Err(invalid("unsupported unreleased v2 sidecar namespace"));
            }
            {
                let name = "names-v2";
                let child = path.join(name);
                create_metadata_directory(&child)?;
                directories.push(Directory::open(child, true, Some(device))?);
            }
        }
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
        let planned = leases::planned_namespace_parent(parent, allow_symlinks)?;
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

    /// One ordering fence per volume: no member lock, receipt or document in
    /// this namespace can reach the media before its directories. The
    /// admission journal's full flush and each document's durable
    /// acknowledgement fence on the same file system make them durable.
    pub(crate) fn commit(self) -> Result<PreparedNamespaces> {
        self.commit_with(|directories| {
            let mut sync = SyncGroup::new();
            for directory in directories {
                sync.stage(&directory.file)?;
                #[cfg(test)]
                super::v2::observe_fd_peak();
            }
            sync.commit_ordered()?;
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
        let parent = leases::planned_namespace_parent(parent, allow_symlinks)?;
        self.parents
            .get(&parent)
            .ok_or_else(|| invalid("output parent was not prepared in this namespace window"))?
            .validate()
    }
}

fn create_metadata_directory(path: &Path) -> Result<()> {
    match platform::private_directory().create(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => Ok(()),
        Err(error) => Err(error.into()),
    }
}

fn checked_metadata(path: &Path, private: bool, volume: Option<u64>) -> Result<Status> {
    let status = platform::status(path)?;
    if !status.metadata().is_dir() || status.metadata().file_type().is_symlink() {
        return Err(invalid("namespace must contain regular directories"));
    }
    if (private && (!status.private() || !status.owned_by_current_user()))
        || volume.is_some_and(|volume| status.id().volume != volume)
    {
        return Err(invalid(
            "namespace metadata must be private on the output filesystem",
        ));
    }
    Ok(status)
}

fn invalid(message: &str) -> Error {
    Error::Invalid(message.into())
}

#[cfg(test)]
mod tests;
