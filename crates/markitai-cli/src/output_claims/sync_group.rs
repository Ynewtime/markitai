//! Internal preparation fence. Staging is never a durable-publication acknowledgement.
use std::collections::BTreeMap;
use std::fs::File;
use std::io;

const MAX_VOLUMES: usize = 32;

/// Hold one descriptor per eligible volume until every staged object crosses a
/// media fence. Dropping this value does not imply that anything was committed.
#[must_use = "staged synchronization must be committed before publication is acknowledged"]
pub(crate) struct SyncGroup {
    inner: Group<System>,
}

impl SyncGroup {
    pub(crate) fn new() -> Self {
        Self {
            inner: Group::new(System),
        }
    }

    /// Flush this object's data/attributes to the device. The group still needs
    /// commit() before a later phase may depend on stable media contents.
    pub(crate) fn stage(&mut self, file: &File) -> io::Result<()> {
        self.inner.stage(file)
    }

    /// Complete all staged volume fences, or fail without issuing a success
    /// acknowledgement. Some earlier volumes may already be durable on failure.
    pub(crate) fn commit(self) -> io::Result<()> {
        self.inner.commit()
    }
}

trait Backend {
    type Handle;
    fn volume(&self, handle: &Self::Handle) -> io::Result<Option<u64>>;
    fn duplicate(&self, handle: &Self::Handle) -> io::Result<Self::Handle>;
    fn host_sync(&self, handle: &Self::Handle) -> io::Result<()>;
    fn media_sync(&self, handle: &Self::Handle) -> io::Result<()>;
    fn durable_sync(&self, handle: &Self::Handle) -> io::Result<()>;
}

struct Group<B: Backend> {
    backend: B,
    volumes: BTreeMap<u64, B::Handle>,
    poisoned: bool,
}

impl<B: Backend> Group<B> {
    fn new(backend: B) -> Self {
        Self {
            backend,
            volumes: BTreeMap::new(),
            poisoned: false,
        }
    }

    fn stage(&mut self, handle: &B::Handle) -> io::Result<()> {
        if self.poisoned {
            return Err(io::Error::other("synchronization group already failed"));
        }
        let result = self.stage_inner(handle);
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    fn stage_inner(&mut self, handle: &B::Handle) -> io::Result<()> {
        let Some(volume) = self.backend.volume(handle)? else {
            // Linux and unsupported/macOS remote filesystems keep their ordinary
            // per-object durability operation. No cross-filesystem equivalence.
            return self.backend.durable_sync(handle);
        };
        if !self.volumes.contains_key(&volume) {
            if self.volumes.len() == MAX_VOLUMES {
                // A bounded descriptor table cannot justify omitting a fence.
                return self.backend.durable_sync(handle);
            }
            let retained = self.backend.duplicate(handle)?;
            self.backend.host_sync(handle)?;
            self.volumes.insert(volume, retained);
        } else {
            self.backend.host_sync(handle)?;
        }
        Ok(())
    }

    fn commit(self) -> io::Result<()> {
        if self.poisoned {
            return Err(io::Error::other("synchronization group already failed"));
        }
        for handle in self.volumes.values() {
            self.backend.media_sync(handle)?;
        }
        Ok(())
    }
}

struct System;

impl Backend for System {
    type Handle = File;

    fn volume(&self, file: &File) -> io::Result<Option<u64>> {
        #[cfg(target_os = "macos")]
        {
            use std::os::fd::AsRawFd;
            use std::os::unix::fs::MetadataExt;
            let metadata = file.metadata()?;
            if !metadata.is_file() && !metadata.is_dir() {
                return Ok(None);
            }
            let mut filesystem = std::mem::MaybeUninit::<libc::statfs>::uninit();
            // SAFETY: fstatfs initializes the provided struct on success and the
            // descriptor is borrowed from a live File for this call.
            if unsafe { libc::fstatfs(file.as_raw_fd(), filesystem.as_mut_ptr()) } != 0 {
                return Err(io::Error::last_os_error());
            }
            // SAFETY: the preceding call succeeded.
            let filesystem = unsafe { filesystem.assume_init() };
            let name: Vec<u8> = filesystem
                .f_fstypename
                .iter()
                .take_while(|&&byte| byte != 0)
                .map(|&byte| byte as u8)
                .collect();
            let local = (filesystem.f_flags as u64 & libc::MNT_LOCAL as u64) != 0;
            if local && matches!(name.as_slice(), b"apfs" | b"hfs") {
                Ok(Some(metadata.dev()))
            } else {
                Ok(None)
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = file;
            Ok(None)
        }
    }

    fn duplicate(&self, file: &File) -> io::Result<File> {
        file.try_clone()
    }

    fn host_sync(&self, file: &File) -> io::Result<()> {
        #[cfg(target_os = "macos")]
        {
            use std::os::fd::AsRawFd;
            retry_interrupted(|| {
                // SAFETY: the descriptor remains open and fsync has no pointer arguments.
                unsafe { libc::fsync(file.as_raw_fd()) }
            })
        }
        #[cfg(not(target_os = "macos"))]
        file.sync_all()
    }

    fn media_sync(&self, file: &File) -> io::Result<()> {
        #[cfg(target_os = "macos")]
        {
            use std::os::fd::AsRawFd;
            retry_interrupted(|| {
                // SAFETY: F_FULLFSYNC ignores its optional argument; the file is
                // a retained descriptor on the same verified local volume.
                unsafe { libc::fcntl(file.as_raw_fd(), libc::F_FULLFSYNC) }
            })
        }
        #[cfg(not(target_os = "macos"))]
        file.sync_all()
    }

    fn durable_sync(&self, file: &File) -> io::Result<()> {
        file.sync_all()
    }
}

#[cfg(target_os = "macos")]
fn retry_interrupted(mut operation: impl FnMut() -> libc::c_int) -> io::Result<()> {
    loop {
        if operation() != -1 {
            return Ok(());
        }
        let error = io::Error::last_os_error();
        if error.kind() != io::ErrorKind::Interrupted {
            return Err(error);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    #[derive(Clone, Copy)]
    struct Handle {
        id: u32,
        volume: Option<u64>,
    }
    struct Fake {
        events: Rc<RefCell<Vec<(char, u32)>>>,
        fail: Option<(char, u32)>,
    }
    impl Fake {
        fn event(&self, kind: char, handle: &Handle) -> io::Result<()> {
            self.events.borrow_mut().push((kind, handle.id));
            if self.fail == Some((kind, handle.id)) {
                Err(io::Error::other("injected sync failure"))
            } else {
                Ok(())
            }
        }
    }
    impl Backend for Fake {
        type Handle = Handle;
        fn volume(&self, handle: &Handle) -> io::Result<Option<u64>> {
            Ok(handle.volume)
        }
        fn duplicate(&self, handle: &Handle) -> io::Result<Handle> {
            Ok(*handle)
        }
        fn host_sync(&self, handle: &Handle) -> io::Result<()> {
            self.event('h', handle)
        }
        fn media_sync(&self, handle: &Handle) -> io::Result<()> {
            self.event('m', handle)
        }
        fn durable_sync(&self, handle: &Handle) -> io::Result<()> {
            self.event('d', handle)
        }
    }
    fn handle(id: u32, volume: u64) -> Handle {
        Handle {
            id,
            volume: Some(volume),
        }
    }

    #[test]
    fn all_object_flushes_precede_separate_volume_media_fences() {
        let events = Rc::new(RefCell::new(Vec::new()));
        let mut group = Group::new(Fake {
            events: events.clone(),
            fail: None,
        });
        group.stage(&handle(1, 10)).unwrap();
        group.stage(&handle(2, 10)).unwrap();
        group.stage(&handle(3, 20)).unwrap();
        assert_eq!(*events.borrow(), [('h', 1), ('h', 2), ('h', 3)]);
        group.commit().unwrap();
        assert_eq!(
            *events.borrow(),
            [('h', 1), ('h', 2), ('h', 3), ('m', 1), ('m', 3)]
        );
    }

    #[test]
    fn failed_object_preparation_cannot_be_later_acknowledged() {
        let events = Rc::new(RefCell::new(Vec::new()));
        let mut group = Group::new(Fake {
            events: events.clone(),
            fail: Some(('h', 2)),
        });
        group.stage(&handle(1, 10)).unwrap();
        assert!(group.stage(&handle(2, 10)).is_err());
        assert!(group.stage(&handle(3, 10)).is_err());
        assert!(group.commit().is_err());
        assert_eq!(*events.borrow(), [('h', 1), ('h', 2)]);
    }

    #[test]
    fn media_failure_does_not_acknowledge_later_volumes() {
        let events = Rc::new(RefCell::new(Vec::new()));
        let mut group = Group::new(Fake {
            events: events.clone(),
            fail: Some(('m', 1)),
        });
        group.stage(&handle(1, 10)).unwrap();
        group.stage(&handle(2, 20)).unwrap();
        assert!(group.commit().is_err());
        assert_eq!(*events.borrow(), [('h', 1), ('h', 2), ('m', 1)]);
    }

    #[test]
    fn unsupported_or_over_capacity_volumes_keep_full_per_object_sync() {
        let events = Rc::new(RefCell::new(Vec::new()));
        let mut group = Group::new(Fake {
            events: events.clone(),
            fail: None,
        });
        group
            .stage(&Handle {
                id: 90,
                volume: None,
            })
            .unwrap();
        for id in 0..MAX_VOLUMES as u32 {
            group.stage(&handle(id, id as u64)).unwrap();
        }
        group.stage(&handle(91, 99)).unwrap();
        assert_eq!(group.volumes.len(), MAX_VOLUMES);
        group.commit().unwrap();
        assert_eq!(events.borrow()[0], ('d', 90));
        assert!(events.borrow().contains(&('d', 91)));
        assert_eq!(
            events
                .borrow()
                .iter()
                .filter(|(kind, _)| *kind == 'm')
                .count(),
            MAX_VOLUMES
        );
    }

    #[test]
    #[cfg(unix)]
    fn real_private_files_survive_descriptor_retention_and_fence() {
        use std::io::Write;
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("prepared");
        let mut file = File::create(&path).unwrap();
        file.write_all(b"complete staged bytes").unwrap();
        let mut group = SyncGroup::new();
        group.stage(&file).unwrap();
        drop(file);
        group.stage(&File::open(directory.path()).unwrap()).unwrap();
        group.commit().unwrap();
        assert_eq!(std::fs::read(path).unwrap(), b"complete staged bytes");
    }
}
