//! Short admission gates protect temporary filesystem alias probes. Epoch pins
//! keep their identities valid; only acquired writer locks outlive the gate.
use super::{Error, Result};
use markitai_core::platform::{self, FileId, Status};
use std::fs::{self, File, OpenOptions, TryLockError};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const CLEANUP_BLOCK: usize = 256;
/// No release writes `writers-v2`; it is unknown metadata, preserved and reported.
pub(super) const UNKNOWN_WRITERS: &str =
    "unknown ownership metadata .markitai/ownership/writers-v2 is preserved, not used";

fn invalid(message: &str) -> Error {
    Error::Invalid(message.into())
}

pub(super) fn checked_file(path: &Path, volume: u64) -> Result<Status> {
    let status = platform::status(path)?;
    check_file(&status, volume)?;
    Ok(status)
}

fn check_file(status: &Status, volume: u64) -> Result<()> {
    if !status.metadata().is_file()
        || !status.private()
        || !status.owned_by_current_user()
        || status.links() != 1
        || status.id().volume != volume
        || status.metadata().len() != 0
    {
        return Err(invalid(
            "v2 coordination entry must be an empty private single-link file on the output filesystem",
        ));
    }
    Ok(())
}

pub(super) fn open_checked(path: &Path, volume: u64, create: bool) -> Result<(File, FileId)> {
    match checked_file(path, volume) {
        Ok(_) => (),
        Err(Error::Io(error)) if create && error.kind() == io::ErrorKind::NotFound => (),
        Err(error) => return Err(error),
    }
    let mut options = OpenOptions::new();
    options
        .read(true)
        .write(true)
        .create(create)
        .truncate(false);
    platform::private_file(&mut options);
    let file = platform::open_no_follow(&options, path)?;
    let held = platform::file_status(&file)?;
    check_file(&held, volume)?;
    if checked_file(path, volume)?.id() != held.id() {
        return Err(invalid("v2 coordination entry changed while opening"));
    }
    Ok((file, held.id()))
}

/// No persistent directory descriptors: paths/identities are reobserved before
/// each gate. Stable coordination files are never deleted or replaced.
#[derive(Debug)]
pub(crate) struct Parent {
    path: PathBuf,
    identity: FileId,
    directories: Vec<(PathBuf, FileId, bool)>,
    gate: FileId,
    epoch: FileId,
}

impl Parent {
    pub(crate) fn open(path: &Path) -> Result<Arc<Self>> {
        let status = platform::status(path)?;
        if !status.metadata().is_dir() {
            return Err(invalid("v2 output parent is not a directory"));
        }
        let identity = status.id();
        let mut directories = Vec::new();
        for (name, private) in [
            (".markitai", false),
            (".markitai/ownership", true),
            (".markitai/ownership/names-v2", true),
            (".markitai/ownership/records", true),
        ] {
            let child = path.join(name);
            let status = platform::status(&child)?;
            if !status.metadata().is_dir()
                || (private && (!status.private() || !status.owned_by_current_user()))
                || status.id().volume != identity.volume
            {
                return Err(invalid(
                    "v2 metadata must be regular private directories on the output filesystem",
                ));
            }
            directories.push((child, status.id(), private));
        }
        let gate = checked_file(&path.join(".markitai/ownership/members"), identity.volume)?.id();
        let epoch = checked_file(&path.join(".markitai/ownership/epoch-v2"), identity.volume)?.id();
        let parent = Arc::new(Self {
            path: path.to_owned(),
            identity,
            directories,
            gate,
            epoch,
        });
        parent.validate()?;
        Ok(parent)
    }

    fn location(&self, name: &str) -> PathBuf {
        self.path.join(".markitai/ownership").join(name)
    }

    pub(crate) fn identity(&self) -> FileId {
        self.identity
    }

    pub(crate) fn validate(&self) -> Result<()> {
        match platform::status(&self.location("writers-v2")) {
            Err(error) if error.kind() == io::ErrorKind::NotFound => (),
            Err(error) => return Err(error.into()),
            Ok(_) => return Err(invalid(UNKNOWN_WRITERS)),
        }
        let status = platform::status(&self.path)?;
        if !status.metadata().is_dir() || status.id() != self.identity {
            return Err(invalid("v2 output parent changed"));
        }
        for (path, id, private) in &self.directories {
            let status = platform::status(path)?;
            if !status.metadata().is_dir()
                || (*private && (!status.private() || !status.owned_by_current_user()))
                || status.id() != *id
            {
                return Err(invalid("v2 metadata directory changed"));
            }
        }
        if checked_file(&self.location("members"), self.identity.volume)?.id() != self.gate
            || checked_file(&self.location("epoch-v2"), self.identity.volume)?.id() != self.epoch
        {
            return Err(invalid("v2 stable coordination file changed"));
        }
        Ok(())
    }

    fn gate(&self, nonblocking: bool) -> Result<Gate> {
        self.validate()?;
        let (file, id) = open_checked(&self.location("members"), self.identity.volume, false)?;
        if id != self.gate {
            return Err(invalid("v2 gate changed"));
        }
        #[cfg(test)]
        let waiting = std::time::Instant::now();
        let guard = Gate { file };
        if nonblocking {
            guard.file.try_lock().map_err(lock_error)?;
        } else {
            guard.file.lock()?;
        }
        #[cfg(test)]
        {
            GATE_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            GATE_WAIT_NS.fetch_add(
                waiting.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }
        self.validate()?;
        if platform::file_status(&guard.file)?.id() != self.gate {
            return Err(invalid("v2 held gate changed"));
        }
        Ok(guard)
    }

    fn epoch_file(&self) -> Result<File> {
        let (file, id) = open_checked(&self.location("epoch-v2"), self.identity.volume, false)?;
        if id != self.epoch {
            return Err(invalid("v2 epoch changed"));
        }
        Ok(file)
    }

    /// Gate held; descriptors never escape this method. A shared epoch held by
    /// the caller pins all successfully observed identities after we close.
    fn probes(&self, members: &[String]) -> Result<Vec<FileId>> {
        let mut keys = Vec::new();
        for member in members {
            super::leases::validate_name(member)?;
            let (file, key) = open_checked(
                &self.location("names-v2").join(member),
                self.identity.volume,
                true,
            )?;
            checkpoint("probe");
            drop(file);
            if !keys.contains(&key) {
                keys.push(key);
            }
        }
        keys.sort();
        Ok(keys)
    }

    /// A whole block requires gate + exclusive epoch. Each deleted object is
    /// opened no-follow, checked, closed, rechecked, then unlinked under gate.
    fn clean_block(&self) -> Result<usize> {
        let _gate = self.gate(true)?;
        let epoch = Gate {
            file: self.epoch_file()?,
        };
        epoch.file.try_lock().map_err(lock_error)?;
        self.validate()?;
        let mut removed = 0;
        for entry in fs::read_dir(self.location("names-v2"))?.take(CLEANUP_BLOCK) {
            let path = entry?.path();
            let name = path
                .file_name()
                .and_then(|value| value.to_str())
                .ok_or_else(|| invalid("unknown v2 cleanup filename"))?;
            super::leases::validate_name(name)?;
            let (file, id) = open_checked(&path, self.identity.volume, false)?;
            // Defence against a retained noncooperating probe descriptor.
            let held = Gate { file };
            held.file.try_lock().map_err(lock_error)?;
            held.file.unlock()?;
            drop(held);
            if checked_file(&path, self.identity.volume)?.id() != id {
                return Err(invalid("v2 cleanup entry changed"));
            }
            self.validate()?;
            checkpoint("cleanup");
            fs::remove_file(path)?;
            removed += 1;
        }
        Ok(removed)
    }

    /// Release locks between blocks so cleanup never monopolizes admission.
    pub(crate) fn cleanup(&self) -> Result<()> {
        loop {
            match self.clean_block() {
                Ok(CLEANUP_BLOCK) => (),
                Ok(_) | Err(Error::Busy) => return Ok(()),
                Err(error) => return Err(error),
            }
        }
    }
}

fn lock_error(error: TryLockError) -> Error {
    match error {
        TryLockError::WouldBlock => Error::Busy,
        TryLockError::Error(error) => Error::Io(error),
    }
}

struct Gate {
    file: File,
}
impl Drop for Gate {
    fn drop(&mut self) {
        let _ = self.file.unlock();
    }
}

/// One OS shared lock per Arc, explicitly unlocked on the last drop (also
/// releases a fork-inherited open-file-description). No member owns a pin fd.
pub(crate) struct Epoch {
    parent: Arc<Parent>,
    file: Option<File>,
}
impl Epoch {
    pub(crate) fn new(parent: Arc<Parent>) -> Result<Arc<Self>> {
        // Recover kill leftovers when idle, before establishing this index.
        parent.cleanup()?;
        let _gate = parent.gate(false)?;
        let file = parent.epoch_file()?;
        let pin = Self {
            parent: Arc::clone(&parent),
            file: Some(file),
        };
        pin.file.as_ref().expect("new epoch").lock_shared()?;
        parent.validate()?;
        Ok(Arc::new(pin))
    }

    pub(crate) fn parent(&self) -> &Arc<Parent> {
        &self.parent
    }

    pub(crate) fn validate(&self) -> Result<()> {
        self.parent.validate()?;
        if platform::file_status(self.file.as_ref().expect("live epoch"))?.id() != self.parent.epoch
        {
            return Err(invalid("v2 held epoch changed"));
        }
        Ok(())
    }

    pub(crate) fn family_keys(&self, families: &[Vec<String>]) -> Result<Vec<Vec<FileId>>> {
        let mut result = Vec::with_capacity(families.len());
        for chunk in families.chunks(128) {
            let _gate = self.parent.gate(false)?;
            self.validate()?;
            for members in chunk {
                result.push(self.parent.probes(members)?);
            }
        }
        Ok(result)
    }

    pub(crate) fn keys(&self, members: &[String]) -> Result<Vec<FileId>> {
        let _gate = self.parent.gate(false)?;
        self.validate()?;
        self.parent.probes(members)
    }
}
impl Drop for Epoch {
    fn drop(&mut self) {
        if let Some(file) = self.file.take() {
            let _ = file.unlock();
            drop(file);
        }
        // Nonblocking gate avoids recursion when acquisition fails in-gate;
        // recovery on the next open handles those safe retained probes.
        if let Err(error) = self.parent.cleanup() {
            eprintln!("Output coordination cleanup incomplete: {error}");
        }
    }
}

struct Writer {
    path: PathBuf,
    identity: FileId,
    file: Option<File>,
    acquired: bool,
}
impl Drop for Writer {
    fn drop(&mut self) {
        if self.acquired
            && let Some(file) = &self.file
        {
            let _ = file.unlock();
        }
    }
}

pub(super) struct Lease {
    epoch: Arc<Epoch>,
    keys: Vec<FileId>,
    writers: Vec<Writer>,
    probes: Vec<(PathBuf, FileId)>,
}
impl Lease {
    pub(super) fn acquire(epoch: Arc<Epoch>, members: &[String]) -> Result<Self> {
        let _gate = epoch.parent.gate(false)?;
        epoch.validate()?;
        let mut writers: Vec<Writer> = Vec::with_capacity(members.len());
        let mut probes = Vec::with_capacity(members.len());
        for member in members {
            super::leases::validate_name(member)?;
            let path = epoch.parent.location("names-v2").join(member);
            let (file, identity) = open_checked(&path, epoch.parent.identity.volume, true)?;
            checkpoint("probe");
            probes.push((path.clone(), identity));
            if writers.iter().any(|writer| writer.identity == identity) {
                drop(file);
                continue;
            }
            writers.push(Writer {
                path,
                identity,
                file: Some(file),
                acquired: false,
            });
        }
        writers.sort_by_key(|writer| writer.identity);
        let keys = writers.iter().map(|writer| writer.identity).collect();
        for writer in &mut writers {
            writer
                .file
                .as_ref()
                .expect("live writer")
                .try_lock()
                .map_err(lock_error)?;
            writer.acquired = true;
            checkpoint("writer");
        }
        epoch.validate()?;
        Ok(Self {
            epoch: Arc::clone(&epoch),
            keys,
            writers,
            probes,
        })
    }

    pub(super) fn keys(&self) -> Vec<FileId> {
        self.keys.clone()
    }
    pub(super) fn epoch(&self) -> Arc<Epoch> {
        Arc::clone(&self.epoch)
    }

    pub(super) fn validate(&self) -> Result<()> {
        self.epoch.validate()?;
        for (path, id) in &self.probes {
            if checked_file(path, self.epoch.parent.identity.volume)?.id() != *id {
                return Err(invalid("held v2 probe was replaced"));
            }
        }
        for writer in &self.writers {
            let held = platform::file_status(writer.file.as_ref().expect("live writer"))?;
            check_file(&held, self.epoch.parent.identity.volume)?;
            if checked_file(&writer.path, self.epoch.parent.identity.volume)?.id()
                != writer.identity
                || held.id() != writer.identity
            {
                return Err(invalid("held v2 writer was replaced"));
            }
        }
        Ok(())
    }
}
impl Drop for Lease {
    fn drop(&mut self) {
        let cleanup = match self.epoch.parent.gate(false) {
            Ok(_gate) => {
                let result = (|| -> Result<()> {
                    self.validate()?;
                    for writer in &mut self.writers {
                        let file = writer.file.as_ref().expect("live writer");
                        file.unlock()?;
                        writer.acquired = false;
                        drop(writer.file.take());
                        // Probe stays until the last shared epoch is released.
                    }
                    Ok(())
                })();
                // On validation, unlock or unlink failure, close every remaining
                // held writer before the verified gate guard leaves this arm.
                self.writers.clear();
                result
            }
            Err(error) => {
                // A changed namespace supplies no trusted gate. Acquired handles
                // are only released, never reused or deleted; no unacquired fd
                // escapes a gate. Explicit unlock still covers inherited fds.
                self.writers.clear();
                Err(error)
            }
        };
        if let Err(error) = cleanup {
            eprintln!("Output coordination cleanup incomplete: {error}");
        }
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
fn checkpoint(phase: &str) {
    observe_fd_peak();
    if std::env::var("MARKITAI_V2_PAUSE").ok().as_deref() == Some(phase)
        && let Some(root) = std::env::var_os("MARKITAI_V2_ROOT")
    {
        let root = PathBuf::from(root);
        fs::write(root.join("paused"), phase).unwrap();
        loop {
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    }
}
#[cfg(not(test))]
fn checkpoint(_phase: &str) {}

#[cfg(test)]
static GATE_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(test)]
static GATE_WAIT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(test)]
pub(crate) fn gate_metrics() -> (u64, u64) {
    (
        GATE_CALLS.load(std::sync::atomic::Ordering::Relaxed),
        GATE_WAIT_NS.load(std::sync::atomic::Ordering::Relaxed),
    )
}

#[cfg(all(test, unix))]
static FD_PEAK: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
pub(crate) fn observe_fd_peak() {
    #[cfg(unix)]
    if std::env::var_os("MARKITAI_V2_FD_OBSERVER").is_some() {
        // SAFETY: F_GETFD observes a descriptor without retaining or changing it.
        let count = (0..1024)
            .filter(|&fd| unsafe { libc::fcntl(fd, libc::F_GETFD) } >= 0)
            .count();
        FD_PEAK.fetch_max(count, std::sync::atomic::Ordering::Relaxed);
    }
}
#[cfg(all(test, unix))]
pub(crate) fn fd_peak() -> usize {
    FD_PEAK.load(std::sync::atomic::Ordering::Relaxed)
}
