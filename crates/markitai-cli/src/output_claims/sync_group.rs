//! Internal preparation fence. Staging is never a durable-publication acknowledgement.
//!
//! Windows has no directory flush and no ordering barrier: every staged file
//! is flushed durably (`FlushFileBuffers`) when it is staged, staging a
//! directory only checks that it exists, and a renamed file is flushed after
//! its rename ([`SyncGroup::stage_renamed`]), which commits the NTFS log
//! records of that rename and of every earlier change on the volume.
use std::collections::BTreeMap;
use std::fs::File;
use std::io;
use std::path::Path;

const MAX_VOLUMES: usize = 32;

/// How a commit completes the fence of each verified volume.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fence {
    /// Staged objects reach stable media before any later write to the same
    /// device, but nothing is promised durable when the commit returns.
    Ordered,
    /// Staged objects are on stable media when the commit returns.
    Durable,
}

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
    /// a commit before a later phase may depend on stable media contents.
    pub(crate) fn stage(&mut self, file: &File) -> io::Result<()> {
        self.inner.stage(file)
    }

    /// Stage the entries of the directory `path` names, following a final
    /// link as an ordinary open does. On Windows only its existence is checked.
    pub(crate) fn stage_directory(&mut self, path: &Path) -> io::Result<()> {
        #[cfg(windows)]
        {
            self.inner
                .guard(|| markitai_core::platform::sync_directory(path))
        }
        #[cfg(not(windows))]
        {
            match File::open(path) {
                Ok(file) => self.stage(&file),
                // A directory that cannot be opened fails the group too.
                Err(error) => self.inner.guard(|| Err(error)),
            }
        }
    }

    /// Stage the directory entry `path` itself: a final symbolic link (or
    /// junction) is refused instead of followed.
    pub(crate) fn stage_directory_entry(&mut self, path: &Path) -> io::Result<()> {
        #[cfg(windows)]
        {
            self.inner
                .guard(|| markitai_core::platform::open_directory(path).map(drop))
        }
        #[cfg(not(windows))]
        {
            match markitai_core::platform::open_directory(path) {
                Ok(file) => self.stage(&file),
                Err(error) => self.inner.guard(|| Err(error)),
            }
        }
    }

    /// A rename just published `file` under its new name. Unix persists the
    /// name through the staged parent directory; Windows flushes the file.
    pub(crate) fn stage_renamed(&mut self, file: &File) -> io::Result<()> {
        #[cfg(windows)]
        {
            self.stage(file)
        }
        #[cfg(not(windows))]
        {
            let _ = file;
            self.inner.guard(|| Ok(()))
        }
    }

    /// [`Self::stage_renamed`] for the file a rename published inside a
    /// renamed directory.
    pub(crate) fn stage_renamed_path(&mut self, path: &Path) -> io::Result<()> {
        self.inner
            .guard(|| markitai_core::platform::sync_renamed_path(path))
    }

    /// Complete all staged volume fences durably, or fail without issuing a
    /// success acknowledgement. Some earlier volumes may already be durable on
    /// failure. A durable fence also persists every write that an earlier
    /// ordered commit staged on the same device.
    pub(crate) fn commit(self) -> io::Result<()> {
        #[cfg(test)]
        record(Fence::Durable);
        self.inner.commit(Fence::Durable)
    }

    /// Order every staged object before any later write to its device, without
    /// waiting for the device cache: what follows (a rename, a later phase)
    /// can never reach stable media without what was staged. This is not an
    /// acknowledgement; success still requires a durable commit on the volume.
    /// Off verified volumes every object was already synchronized durably when
    /// it was staged, which is at least as strong.
    pub(crate) fn commit_ordered(self) -> io::Result<()> {
        #[cfg(test)]
        record(Fence::Ordered);
        self.inner.commit(Fence::Ordered)
    }
}

#[cfg(test)]
thread_local! {
    static COMMITS: std::cell::RefCell<Vec<Fence>> = const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(test)]
fn record(fence: Fence) {
    COMMITS.with(|commits| commits.borrow_mut().push(fence));
}

/// Fence kinds committed by this test thread since the last call, in order.
#[cfg(test)]
pub(crate) fn take_commits() -> Vec<&'static str> {
    COMMITS.with(|commits| {
        commits
            .borrow_mut()
            .drain(..)
            .map(|fence| match fence {
                Fence::Ordered => "ordered",
                Fence::Durable => "durable",
            })
            .collect()
    })
}

trait Backend {
    type Handle;
    fn volume(&self, handle: &Self::Handle) -> io::Result<Option<u64>>;
    fn duplicate(&self, handle: &Self::Handle) -> io::Result<Self::Handle>;
    fn host_sync(&self, handle: &Self::Handle) -> io::Result<()>;
    fn media_sync(&self, handle: &Self::Handle, fence: Fence) -> io::Result<()>;
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

    /// Run a staging step that needs no handle of this group, with the same
    /// poisoning as [`Self::stage`].
    fn guard(&mut self, step: impl FnOnce() -> io::Result<()>) -> io::Result<()> {
        if self.poisoned {
            return Err(io::Error::other("synchronization group already failed"));
        }
        let result = step();
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

    fn commit(self, fence: Fence) -> io::Result<()> {
        if self.poisoned {
            return Err(io::Error::other("synchronization group already failed"));
        }
        for handle in self.volumes.values() {
            self.backend.media_sync(handle, fence)?;
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

    fn media_sync(&self, file: &File, fence: Fence) -> io::Result<()> {
        #[cfg(target_os = "macos")]
        {
            use std::os::fd::AsRawFd;
            let full = || {
                retry_interrupted(|| {
                    // SAFETY: F_FULLFSYNC ignores its optional argument; the file
                    // is a retained descriptor on the same verified local volume.
                    unsafe { libc::fcntl(file.as_raw_fd(), libc::F_FULLFSYNC) }
                })
            };
            if fence == Fence::Durable {
                return full();
            }
            // fcntl(2): after fsync of each descriptor, one barrier orders all of
            // them before any later I/O on the device. APFS and HFS implement it.
            let barrier = retry_interrupted(|| {
                // SAFETY: as above; F_BARRIERFSYNC also ignores its argument.
                unsafe { libc::fcntl(file.as_raw_fd(), libc::F_BARRIERFSYNC) }
            });
            match barrier {
                // A file system without the operation gets the stronger flush.
                Err(error)
                    if matches!(
                        error.raw_os_error(),
                        Some(libc::ENOTSUP | libc::ENOTTY | libc::EINVAL)
                    ) =>
                {
                    full()
                }
                result => result,
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = fence;
            file.sync_all()
        }
    }

    fn durable_sync(&self, file: &File) -> io::Result<()> {
        // A Windows directory handle cannot be flushed; its names become
        // durable with the next file flush on the volume.
        #[cfg(windows)]
        if file.metadata()?.is_dir() {
            return Ok(());
        }
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
        fn media_sync(&self, handle: &Handle, fence: Fence) -> io::Result<()> {
            // 'm' is the durable flush, 'b' the ordering barrier.
            self.event(if fence == Fence::Durable { 'm' } else { 'b' }, handle)
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
        group.commit(Fence::Durable).unwrap();
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
        assert!(group.commit(Fence::Durable).is_err());
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
        assert!(group.commit(Fence::Durable).is_err());
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
        group.commit(Fence::Durable).unwrap();
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
    fn ordered_commit_issues_one_barrier_per_volume_after_every_object_flush() {
        let events = Rc::new(RefCell::new(Vec::new()));
        let mut group = Group::new(Fake {
            events: events.clone(),
            fail: None,
        });
        group.stage(&handle(1, 10)).unwrap();
        group.stage(&handle(2, 20)).unwrap();
        group.stage(&handle(3, 10)).unwrap();
        group.commit(Fence::Ordered).unwrap();
        // A barrier, never a full flush, and only after all object flushes.
        assert_eq!(
            *events.borrow(),
            [('h', 1), ('h', 2), ('h', 3), ('b', 1), ('b', 2)]
        );
    }

    #[test]
    fn ordered_commit_cannot_succeed_after_a_failed_object_or_barrier() {
        let events = Rc::new(RefCell::new(Vec::new()));
        let mut group = Group::new(Fake {
            events: events.clone(),
            fail: Some(('h', 2)),
        });
        group.stage(&handle(1, 10)).unwrap();
        assert!(group.stage(&handle(2, 10)).is_err());
        assert!(group.commit(Fence::Ordered).is_err());
        assert_eq!(*events.borrow(), [('h', 1), ('h', 2)]);
        let events = Rc::new(RefCell::new(Vec::new()));
        let mut group = Group::new(Fake {
            events: events.clone(),
            fail: Some(('b', 1)),
        });
        group.stage(&handle(1, 10)).unwrap();
        group.stage(&handle(2, 20)).unwrap();
        assert!(group.commit(Fence::Ordered).is_err());
        assert_eq!(*events.borrow(), [('h', 1), ('h', 2), ('b', 1)]);
    }

    #[test]
    fn ordered_commit_keeps_durable_per_object_sync_off_verified_volumes() {
        let events = Rc::new(RefCell::new(Vec::new()));
        let mut group = Group::new(Fake {
            events: events.clone(),
            fail: None,
        });
        group
            .stage(&Handle {
                id: 7,
                volume: None,
            })
            .unwrap();
        group.commit(Fence::Ordered).unwrap();
        assert_eq!(*events.borrow(), [('d', 7)]);
    }

    #[test]
    fn real_private_files_survive_descriptor_retention_and_fence() {
        use std::io::Write;
        let directory = tempfile::tempdir().unwrap();
        for ordered in [false, true] {
            let path = directory.path().join(format!("prepared-{ordered}"));
            let mut file = File::create(&path).unwrap();
            file.write_all(b"complete staged bytes").unwrap();
            let mut group = SyncGroup::new();
            group.stage(&file).unwrap();
            drop(file);
            group.stage_directory(directory.path()).unwrap();
            group.stage_directory_entry(directory.path()).unwrap();
            group
                .stage(&markitai_core::platform::open_directory(directory.path()).unwrap())
                .unwrap();
            if ordered {
                group.commit_ordered().unwrap();
            } else {
                group.commit().unwrap();
            }
            assert_eq!(std::fs::read(path).unwrap(), b"complete staged bytes");
        }
    }

    #[test]
    fn a_renamed_file_and_a_missing_directory_follow_the_group_rules() {
        use std::io::Write;
        let directory = tempfile::tempdir().unwrap();
        let mut staged = tempfile::NamedTempFile::new_in(directory.path()).unwrap();
        staged.write_all(b"bytes").unwrap();
        let mut group = SyncGroup::new();
        group.stage(staged.as_file()).unwrap();
        let target = directory.path().join("published");
        let published = markitai_core::platform::persist_noclobber(staged, &target).unwrap();
        group.stage_renamed(&published).unwrap();
        group.stage_renamed_path(&target).unwrap();
        group.stage_directory(directory.path()).unwrap();
        group.commit().unwrap();
        // A vanished directory fails staging and poisons the group.
        let mut group = SyncGroup::new();
        assert!(
            group
                .stage_directory(&directory.path().join("missing"))
                .is_err()
        );
        assert!(group.stage_renamed(&published).is_err());
        assert!(group.commit().is_err());
        let mut group = SyncGroup::new();
        assert!(group.stage_directory_entry(&target).is_err());
        assert!(group.commit_ordered().is_err());
    }
}
