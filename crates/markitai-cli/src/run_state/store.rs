#[path = "legacy_backup.rs"]
mod legacy_backup;
use super::{
    Checkpoint, Error, Event, Fence, ItemKey, Limits, LoadOutcome, Result, Scope, Snapshot, Status,
    codec,
};
use crate::output_claims::sync_group::SyncGroup;
use markitai_core::platform::{self, FileId};
use serde_json::Value;
use std::fs::{self, File, OpenOptions, TryLockError};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

/// Bytes an unfinished entry may still gain before the final compaction: its
/// output path and the usage diagnostics of its attempt.
const ENTRY_GROWTH: usize = 512;

/// The open lock descriptor owns one checkpoint for the lifetime of this store.
/// Buffered mutations require an explicit flush before work may rely on them.
pub(crate) struct StateStore {
    scope: Scope,
    directory: PathBuf,
    base: PathBuf,
    journal: PathBuf,
    _lock: File,
    allow_symlinks: bool,
    limits: Limits,
    snapshot: Option<Snapshot>,
    pending: Vec<Vec<u8>>,
    pending_bytes: usize,
    journal_bytes: usize,
    // Set only after the current journal entry and data are durably synced.
    journal_synced_identity: Option<FileId>,
    durable_sequence: u64,
    // Set while an ordered checkpoint or created states directory still awaits
    // the durable fence of its volume; `durable_sequence` stays behind until a
    // flush or durable compaction completes one.
    unfenced: bool,
    begun: bool,
    poisoned: bool,
    legacy_backup: Option<PathBuf>,
    #[cfg(test)]
    fault: Option<FaultPoint>,
}

impl Drop for StateStore {
    fn drop(&mut self) {
        // A fork child can retain this open file description until exec. Release
        // our lock explicitly when the owning store ends, without flushing work.
        let _ = self._lock.unlock();
    }
}

#[cfg(all(test, target_os = "linux"))]
#[path = "lock_tests.rs"]
mod lock_tests;

enum DiskBase {
    Missing,
    Valid(Box<Snapshot>),
    Corrupt(String),
}

impl StateStore {
    pub(crate) fn open(
        scope: Scope,
        task_hash: &str,
        allow_symlinks: bool,
        limits: Limits,
    ) -> Result<Self> {
        if task_hash.len() != 6
            || !task_hash
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(Error::Invalid(
                "task hash must contain six lowercase hexadecimal digits".into(),
            ));
        }
        if limits.base_bytes == 0
            || limits.journal_bytes == 0
            || limits.line_bytes == 0
            || limits.entries == 0
        {
            return Err(Error::Invalid("storage limits must be positive".into()));
        }
        let requested = scope.output_spelling().join(".markitai/states");
        let directory = scope.output.join(".markitai/states");
        {
            // One observation of these paths' shared ancestors per step.
            let _paths = super::paths::Scope::enter();
            check_policy(scope.output_spelling(), allow_symlinks)?;
            if super::paths::resolve(scope.output_spelling())? != scope.output {
                return Err(Error::ForeignScope(
                    "output path changed while planning recovery".into(),
                ));
            }
            check_policy(&requested, allow_symlinks)?;
            check_policy(&directory, allow_symlinks)?;
        }
        let unfenced = create_directory(&directory)?;
        let base = directory.join(format!("markitai.{task_hash}.state.json"));
        let journal = directory.join(format!("markitai.{task_hash}.state.jsonl"));
        let lock_path = directory.join(format!("markitai.{task_hash}.state.lock"));
        {
            let _paths = super::paths::Scope::enter();
            check_policy(&requested, allow_symlinks)?;
            check_policy(&directory, allow_symlinks)?;
            regular_file(&lock_path, allow_symlinks)?;
        }
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)?;
        if !lock.metadata()?.is_file() {
            return Err(Error::Invalid(
                "checkpoint lock is not a regular file".into(),
            ));
        }
        lock.try_lock().map_err(|error| match error {
            TryLockError::WouldBlock => Error::Busy,
            TryLockError::Error(error) => Error::Io(error),
        })?;
        Ok(Self {
            scope,
            directory,
            base,
            journal,
            _lock: lock,
            allow_symlinks,
            limits,
            snapshot: None,
            pending: Vec::new(),
            pending_bytes: 0,
            journal_bytes: 0,
            journal_synced_identity: None,
            durable_sequence: 0,
            unfenced,
            begun: false,
            poisoned: false,
            legacy_backup: None,
            #[cfg(test)]
            fault: None,
        })
    }

    pub(crate) fn load(&mut self) -> Result<LoadOutcome> {
        self.healthy()?;
        if !self.pending.is_empty() {
            return Err(Error::Invalid(
                "cannot reload buffered state mutations".into(),
            ));
        }
        self.begun = false;
        self.journal_synced_identity = None;
        self.snapshot = None;
        self.durable_sequence = 0;
        let mut snapshot = match self.read_base()? {
            DiskBase::Missing => return Ok(LoadOutcome::Missing),
            DiskBase::Corrupt(reason) => return Ok(LoadOutcome::Corrupt { reason }),
            DiskBase::Valid(snapshot) => *snapshot,
        };
        let mut warnings = Vec::new();
        if let Some(mut bytes) = read_limited(
            &self.journal,
            self.allow_symlinks,
            self.limits.journal_bytes,
        )? {
            let exceeded = bytes.len() > self.limits.journal_bytes;
            if exceeded {
                bytes.truncate(self.limits.journal_bytes);
            }
            self.journal_bytes = bytes.len();
            let watermark = snapshot
                .checkpoint
                .as_ref()
                .map_or(0, |checkpoint| checkpoint.applied_sequence);
            let mut consumed = 0;
            for line in bytes.split_inclusive(|byte| *byte == b'\n') {
                consumed += line.len();
                if exceeded && consumed == bytes.len() && !line.ends_with(b"\n") {
                    break;
                }
                let line = line.strip_suffix(b"\n").unwrap_or(line);
                if line.len() > self.limits.line_bytes {
                    warn(
                        &mut warnings,
                        "Recovery journal line limit reached; later events were not replayed",
                    );
                    break;
                }
                if std::str::from_utf8(line).is_err() {
                    warn(
                        &mut warnings,
                        "Recovery journal contains invalid UTF-8; later events were not replayed",
                    );
                    break;
                }
                if line.iter().all(u8::is_ascii_whitespace) {
                    continue;
                }
                let event = match codec::decode_event_limited(line, self.limits.line_bytes) {
                    Ok(Some(event)) => event,
                    Ok(None) => continue,
                    Err(Error::JsonSyntax(_)) => {
                        warn(&mut warnings, "Malformed recovery journal JSON was skipped");
                        continue;
                    }
                    Err(_) => {
                        warn(
                            &mut warnings,
                            "Invalid recovery journal event; later events were not replayed",
                        );
                        break;
                    }
                };
                match (&snapshot.checkpoint, &event.fence) {
                    (Some(checkpoint), Some(fence)) => {
                        if fence.generation != checkpoint.generation || fence.sequence <= watermark
                        {
                            continue;
                        }
                        if checkpoint.applied_sequence.checked_add(1) != Some(fence.sequence) {
                            warn(
                                &mut warnings,
                                "Recovery journal sequence is duplicate, reversed or incomplete; replay stopped",
                            );
                            break;
                        }
                        if !contains_key(&snapshot, &event.key) {
                            warn(
                                &mut warnings,
                                "Native recovery journal references an uncheckpointed item; replay stopped",
                            );
                            break;
                        }
                    }
                    (Some(_), None) => continue,
                    (None, Some(_)) => {
                        warn(
                            &mut warnings,
                            "Native journal event cannot be replayed over a legacy base; replay stopped",
                        );
                        break;
                    }
                    (None, None) => (),
                }
                if codec::apply_event(&mut snapshot, &event, &self.scope, self.allow_symlinks)
                    .is_err()
                {
                    warn(
                        &mut warnings,
                        "Invalid recovery journal mutation; later events were not replayed",
                    );
                    break;
                }
            }
            if exceeded {
                warn(
                    &mut warnings,
                    "Recovery journal byte limit reached; later events were not replayed",
                );
            }
        } else {
            self.journal_bytes = 0;
        }
        codec::normalize_interrupted(&mut snapshot);
        self.durable_sequence = snapshot
            .checkpoint
            .as_ref()
            .map_or(0, |checkpoint| checkpoint.applied_sequence);
        self.snapshot = Some(snapshot.clone());
        Ok(LoadOutcome::Loaded {
            snapshot: Box::new(snapshot),
            warnings,
        })
    }

    pub(crate) fn begin(&mut self, mut snapshot: Snapshot) -> Result<()> {
        self.healthy()?;
        if !self.pending.is_empty() {
            return Err(Error::Invalid(
                "cannot replace buffered state mutations".into(),
            ));
        }
        let disk = self.read_base()?;
        if let Some(checkpoint) = &snapshot.checkpoint {
            let retained = self
                .snapshot
                .as_ref()
                .and_then(|snapshot| snapshot.checkpoint.as_ref());
            if checkpoint.scope != self.scope || retained != Some(checkpoint) {
                return Err(Error::ForeignScope(
                    "native checkpoint must retain the loaded generation and sequence".into(),
                ));
            }
            if let DiskBase::Valid(previous) = &disk {
                if previous.checkpoint.as_ref().is_none_or(|old| {
                    old.generation != checkpoint.generation
                        || old.applied_sequence > checkpoint.applied_sequence
                }) {
                    return Err(Error::ForeignScope(
                        "checkpoint generation changed before publication".into(),
                    ));
                }
            } else {
                return Err(Error::ForeignScope(
                    "loaded checkpoint was removed or damaged before publication".into(),
                ));
            }
        } else {
            snapshot.checkpoint = Some(Checkpoint {
                generation: uuid::Uuid::new_v4().to_string(),
                applied_sequence: 0,
                scope: self.scope.clone(),
            });
        }
        let snapshot = codec::prepare_checkpoint(snapshot, &self.scope, self.allow_symlinks)?;
        let bytes = codec::encode(&snapshot, &self.scope, self.allow_symlinks, self.limits)?;
        // Refuse a batch whose finished state could not be compacted now,
        // before any paid work, instead of at its final compaction.
        let unfinished = snapshot
            .documents
            .values()
            .chain(snapshot.urls.values())
            .filter(|entry| entry.status != Status::Completed)
            .count();
        if bytes
            .len()
            .saturating_add(unfinished.saturating_mul(ENTRY_GROWTH))
            > self.limits.base_bytes
        {
            return Err(Error::Limit("projected base bytes"));
        }
        if matches!(disk, DiskBase::Corrupt(_)) {
            self.quarantine()?;
        } else if matches!(&disk, DiskBase::Valid(previous) if previous.checkpoint.is_none()) {
            self.check_paths()?;
            let generation = &snapshot.checkpoint.as_ref().unwrap().generation;
            self.legacy_backup = Some(legacy_backup::preserve(
                &self.directory,
                &self.base,
                &self.journal,
                generation,
                self.limits,
            )?);
        }
        self.snapshot = Some(snapshot);
        // Ordering suffices: nothing relies on this checkpoint before the
        // durable fence of the first flush (admission) or the final compaction.
        let result = self.replace_base(&bytes, Commit::Ordered);
        if result.is_err() {
            self.poisoned = true;
        } else {
            self.begun = true;
        }
        result
    }

    pub(crate) fn legacy_backup(&self) -> Option<&Path> {
        self.legacy_backup.as_deref()
    }

    pub(crate) fn snapshot(&self) -> Option<&Snapshot> {
        self.snapshot.as_ref()
    }

    pub(crate) fn record(&mut self, key: ItemKey, data: Value) -> Result<u64> {
        self.writable()?;
        let snapshot = self.snapshot.as_ref().unwrap();
        if !contains_key(snapshot, &key) {
            return Err(Error::Invalid(
                "cannot record an item absent from the checkpoint".into(),
            ));
        }
        let checkpoint = snapshot.checkpoint.as_ref().unwrap();
        let sequence = checkpoint
            .applied_sequence
            .checked_add(1)
            .ok_or_else(|| Error::Sequence("sequence exhausted".into()))?;
        let event = Event {
            key,
            data,
            fence: Some(Fence {
                generation: checkpoint.generation.clone(),
                sequence,
            }),
        };
        // Preparing and applying this event validate the same paths: one
        // observation serves both, unless compaction writes in between.
        let mut paths = Some(super::paths::Scope::enter());
        let event = codec::prepare_event(snapshot, event, &self.scope, self.allow_symlinks)?;
        let mut line = codec::encode_event_limited(&event, self.limits.line_bytes)?;
        if line.len() > self.limits.line_bytes {
            return Err(Error::Limit("journal line"));
        }
        line.push(b'\n');
        if line.len() > self.limits.journal_bytes {
            return Err(Error::Limit("journal bytes"));
        }
        if self
            .journal_bytes
            .saturating_add(self.pending_bytes)
            .saturating_add(line.len())
            > self.limits.journal_bytes
        {
            drop(paths.take());
            // The next flush completes this compaction's durable fence.
            self.compact_with(Commit::Ordered)?;
            paths = Some(super::paths::Scope::enter());
        }
        let snapshot = self.snapshot.as_mut().unwrap();
        if !codec::apply_event(snapshot, &event, &self.scope, self.allow_symlinks)? {
            return Err(Error::Invalid("state mutation was not applied".into()));
        }
        drop(paths);
        self.pending_bytes += line.len();
        self.pending.push(line);
        Ok(sequence)
    }

    pub(crate) fn flush(&mut self) -> Result<u64> {
        self.writable()?;
        if self.pending.is_empty() && !self.unfenced {
            return Ok(self.durable_sequence);
        }
        let result = self.append_pending();
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    /// Replace the base with the complete snapshot and remove the journal,
    /// durably: on success the compacted state is on stable storage.
    pub(crate) fn compact(&mut self) -> Result<()> {
        self.compact_with(Commit::Durable)
    }

    fn compact_with(&mut self, commit: Commit) -> Result<()> {
        self.writable()?;
        let bytes = codec::encode(
            self.snapshot.as_ref().unwrap(),
            &self.scope,
            self.allow_symlinks,
            self.limits,
        )?;
        let result = self.replace_base(&bytes, commit);
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    fn read_base(&self) -> Result<DiskBase> {
        let bytes = {
            let _paths = super::paths::Scope::enter();
            self.check_paths()?;
            read_limited(&self.base, self.allow_symlinks, self.limits.base_bytes)?
        };
        let Some(bytes) = bytes else {
            return Ok(DiskBase::Missing);
        };
        if bytes.len() > self.limits.base_bytes {
            return Ok(DiskBase::Corrupt("base byte limit exceeded".into()));
        }
        match codec::decode(&bytes, &self.scope, self.allow_symlinks, self.limits) {
            Ok(snapshot) => Ok(DiskBase::Valid(Box::new(snapshot))),
            Err(error @ (Error::ForeignScope(_) | Error::Io(_))) => Err(error),
            Err(error) => Ok(DiskBase::Corrupt(error.to_string())),
        }
    }

    fn append_pending(&mut self) -> Result<u64> {
        #[cfg(test)]
        assert!(
            !super::paths::active(),
            "journal write under an earlier observation"
        );
        if self.pending.is_empty() {
            // An ordered checkpoint or created directory awaits its fence.
            self.check_paths()?;
            let mut fence = Staged::new();
            fence.directory(&self.directory)?;
            fence.durable()?;
            return Ok(self.fenced());
        }
        {
            let _paths = super::paths::Scope::enter();
            self.check_paths()?;
            regular_file(&self.journal, self.allow_symlinks)?;
        }
        let mut options = OpenOptions::new();
        // Reading is never used; it lets every platform query the handle.
        options.create(true).append(true).read(true);
        let mut file = platform::private_file(&mut options).open(&self.journal)?;
        let opened = platform::file_status(&file)?;
        let metadata = opened.metadata();
        if !metadata.is_file() || metadata.len() != self.journal_bytes as u64 {
            return Err(Error::Invalid("journal changed outside its owner".into()));
        }
        #[cfg(test)]
        if self.take_fault(FaultPoint::PartialJournalAppend) {
            let first = &self.pending[0];
            file.write_all(&first[..first.len() / 2])?;
            return Err(injected_error());
        }
        for line in &self.pending {
            file.write_all(line)?;
        }
        file.flush()?;
        #[cfg(test)]
        self.fail_at(FaultPoint::BeforeJournalSync)?;
        let identity = Some(opened.id());
        let mut fence = Staged::new();
        fence.file(&file)?;
        // Appending changes file data, not an already durable directory entry.
        // A fresh/replaced journal, or one an ordered checkpoint still precedes,
        // also needs the namespace: its bytes are ordered before its entry,
        // then one durable fence covers both.
        if identity != self.journal_synced_identity || self.unfenced {
            fence.ordered()?;
            #[cfg(test)]
            self.fail_at(FaultPoint::BeforeJournalDirectorySync)?;
            fence = Staged::new();
            fence.directory(&self.directory)?;
        }
        fence.durable()?;
        self.journal_synced_identity = identity;
        self.journal_bytes += self.pending_bytes;
        self.pending_bytes = 0;
        self.pending.clear();
        Ok(self.fenced())
    }

    /// Everything applied so far is on stable storage.
    fn fenced(&mut self) -> u64 {
        self.unfenced = false;
        self.durable_sequence = self
            .snapshot
            .as_ref()
            .unwrap()
            .checkpoint
            .as_ref()
            .unwrap()
            .applied_sequence;
        self.durable_sequence
    }

    /// Publish `bytes` as the base and remove the journal it supersedes.
    ///
    /// The snapshot's bytes are ordered before its name, and its name before
    /// the journal's removal, so a crash leaves the old base with its journal,
    /// the new base with a stale journal (ignored by the replay fence) or the
    /// new base alone. A durable commit then fences the directory before
    /// returning; an ordered one leaves that to the next flush.
    fn replace_base(&mut self, bytes: &[u8], commit: Commit) -> Result<()> {
        // Every step of this write observes the paths afresh.
        #[cfg(test)]
        assert!(
            !super::paths::active(),
            "checkpoint write under an earlier observation"
        );
        {
            let _paths = super::paths::Scope::enter();
            self.check_paths()?;
            regular_file(&self.base, self.allow_symlinks)?;
            regular_file(&self.journal, self.allow_symlinks)?;
        }
        let mut temp = tempfile::Builder::new()
            .prefix(".markitai-state-")
            .suffix(".tmp")
            .tempfile_in(&self.directory)?;
        temp.write_all(bytes)?;
        let mut fence = Staged::new();
        fence.file(temp.as_file())?;
        fence.ordered()?;
        #[cfg(test)]
        self.fail_at(FaultPoint::AfterTempSync)?;
        self.check_paths()?;
        let installed =
            platform::persist(temp, &self.base).map_err(|error| Error::Io(error.error))?;
        let journal = regular_file(&self.journal, self.allow_symlinks)?;
        let mut fence = Staged::new();
        fence.renamed(&installed)?;
        fence.directory(&self.directory)?;
        if journal || commit == Commit::Ordered {
            fence.ordered()?;
        } else {
            fence.durable()?;
        }
        #[cfg(test)]
        self.fail_at(FaultPoint::AfterBaseSync)?;
        if journal {
            fs::remove_file(&self.journal)?;
        }
        #[cfg(test)]
        self.fail_at(FaultPoint::AfterJournalRemove)?;
        if journal && commit == Commit::Durable {
            let mut fence = Staged::new();
            fence.directory(&self.directory)?;
            fence.durable()?;
        }
        self.journal_synced_identity = None;
        self.journal_bytes = 0;
        self.pending_bytes = 0;
        self.pending.clear();
        match commit {
            Commit::Durable => {
                self.fenced();
            }
            Commit::Ordered => self.unfenced = true,
        }
        Ok(())
    }

    fn quarantine(&mut self) -> Result<()> {
        self.check_paths()?;
        let suffix = uuid::Uuid::new_v4().simple().to_string();
        let mut copied = Vec::new();
        let result = (|| {
            for source in [&self.base, &self.journal] {
                if !regular_file(source, self.allow_symlinks)? {
                    continue;
                }
                let name = source.file_name().unwrap().to_string_lossy();
                let target = self.directory.join(format!("{name}.corrupt.{suffix}"));
                // Stream rather than allocate the corrupt file. Refuse enormous
                // quarantine copies while leaving every original untouched.
                let maximum = self
                    .limits
                    .base_bytes
                    .max(self.limits.journal_bytes)
                    .max(64 * 1024 * 1024);
                let input = File::open(source)?;
                if input.metadata()?.len() > maximum as u64 {
                    return Err(Error::Limit("quarantine bytes"));
                }
                // Tempfiles start private on Unix. A normal create_new copy
                // could loosen a 0600 state's permissions through the umask.
                let mut output = tempfile::Builder::new()
                    .prefix(".markitai-quarantine-")
                    .suffix(".tmp")
                    .tempfile_in(&self.directory)?;
                let count = io::copy(
                    &mut input.take(maximum.saturating_add(1) as u64),
                    &mut output,
                )?;
                if count > maximum as u64 {
                    return Err(Error::Limit("quarantine bytes"));
                }
                output.as_file().sync_all()?;
                let installed = platform::persist_noclobber(output, &target)
                    .map_err(|error| Error::Io(error.error))?;
                copied.push(target);
                platform::sync_renamed(&installed)?;
            }
            sync_directory(&self.directory)
        })();
        if result.is_err() {
            for path in copied {
                let _ = fs::remove_file(path);
            }
            self.poisoned = true;
        }
        result
    }

    /// The storage paths under one observation of their shared ancestors (or
    /// the caller's, when it holds one); each call otherwise observes afresh.
    fn check_paths(&self) -> Result<()> {
        let _paths = super::paths::Scope::enter();
        check_policy(self.scope.output_spelling(), self.allow_symlinks)?;
        if super::paths::resolve(self.scope.output_spelling())? != self.scope.output {
            return Err(Error::ForeignScope(
                "output path changed during recovery".into(),
            ));
        }
        check_policy(
            &self.scope.output_spelling().join(".markitai/states"),
            self.allow_symlinks,
        )?;
        check_policy(&self.directory, self.allow_symlinks)?;
        check_policy(&self.base, self.allow_symlinks)?;
        check_policy(&self.journal, self.allow_symlinks)
    }

    fn healthy(&self) -> Result<()> {
        if self.poisoned {
            return Err(Error::Invalid(
                "writer is poisoned after an I/O failure; reopen the checkpoint".into(),
            ));
        }
        Ok(())
    }

    fn writable(&self) -> Result<()> {
        self.healthy()?;
        if !self.begun {
            return Err(Error::Invalid(
                "begin must checkpoint merged state before recording events".into(),
            ));
        }
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn inject_fault(&mut self, point: FaultPoint) {
        self.fault = Some(point);
    }

    #[cfg(test)]
    fn take_fault(&mut self, point: FaultPoint) -> bool {
        if self.fault == Some(point) {
            self.fault = None;
            true
        } else {
            false
        }
    }

    #[cfg(test)]
    fn fail_at(&mut self, point: FaultPoint) -> Result<()> {
        if self.take_fault(point) {
            Err(injected_error())
        } else {
            Ok(())
        }
    }
}

fn contains_key(snapshot: &Snapshot, key: &ItemKey) -> bool {
    match key {
        ItemKey::File(key) => snapshot.documents.contains_key(key),
        ItemKey::Url(key) => snapshot.urls.contains_key(key),
    }
}

/// `markitai_core::output::check_path`, sharing an active path observation.
fn check_policy(path: &Path, allow_symlinks: bool) -> Result<()> {
    match super::paths::symlinks_permitted(path, allow_symlinks) {
        Ok(true) => Ok(()),
        _ => Err(Error::ForeignScope(
            "symbolic link access is disabled for recovery storage".into(),
        )),
    }
}

fn regular_file(path: &Path, allow_symlinks: bool) -> Result<bool> {
    check_policy(path, allow_symlinks)?;
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => fs::metadata(path)?,
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error.into()),
    };
    if !metadata.is_file() {
        return Err(Error::Invalid(
            "recovery storage entry is not a regular file".into(),
        ));
    }
    Ok(true)
}

fn read_limited(path: &Path, allow_symlinks: bool, limit: usize) -> Result<Option<Vec<u8>>> {
    if !regular_file(path, allow_symlinks)? {
        return Ok(None);
    }
    let mut file = File::open(path)?.take(limit.saturating_add(1) as u64);
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    Ok(Some(bytes))
}

fn sync_directory(path: &Path) -> Result<()> {
    Ok(platform::sync_directory(path)?)
}

/// Create the states directory chain. True when created directories were only
/// ordered and still await the durable fence of the checkpoint volume.
fn create_directory(path: &Path) -> Result<bool> {
    let mut missing = Vec::new();
    for ancestor in path.ancestors() {
        match fs::metadata(ancestor) {
            Ok(_) => break,
            Err(error) if error.kind() == io::ErrorKind::NotFound => missing.push(ancestor),
            Err(error) => return Err(error.into()),
        }
    }
    fs::create_dir_all(path)?;
    // Sync each new parent entry too: syncing only `states` does not make a
    // previously absent output/.markitai/states chain durable on Unix. All
    // creation precedes synchronization; one ordering fence per verified local
    // volume then puts the chain before any checkpoint file created in it
    // (per-object durable sync elsewhere). Nothing relies on the chain before
    // the checkpoint's own durable fence, which on the same volume persists it.
    // A chain that reaches another volume keeps the immediate durable fence.
    let Some(parent) = missing.last().and_then(|directory| directory.parent()) else {
        return Ok(false);
    };
    let volume = platform::followed_status(path)?.id().volume;
    let mut one_volume = true;
    let mut group = SyncGroup::new();
    for directory in missing.iter().copied().chain([parent]) {
        one_volume &= platform::followed_status(directory)?.id().volume == volume;
        group.stage_directory(directory)?;
    }
    if one_volume {
        group.commit_ordered()?;
        return Ok(true);
    }
    group.commit()?;
    Ok(false)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Commit {
    /// Later writes on the volume cannot reach stable storage first.
    Ordered,
    /// On stable storage when the call returns.
    Durable,
}

/// Checkpoint files and their directory: per-object synchronization, then one
/// ordering or durable fence per verified volume (the output claims'
/// synchronization group, which on Windows flushes each file durably and
/// persists a renamed checkpoint by flushing it after the rename).
struct Staged {
    group: SyncGroup,
}

impl Staged {
    fn new() -> Self {
        Self {
            group: SyncGroup::new(),
        }
    }

    fn file(&mut self, file: &File) -> Result<()> {
        Ok(self.group.stage(file)?)
    }

    fn renamed(&mut self, file: &File) -> Result<()> {
        Ok(self.group.stage_renamed(file)?)
    }

    fn directory(&mut self, path: &Path) -> Result<()> {
        Ok(self.group.stage_directory(path)?)
    }

    fn ordered(self) -> Result<()> {
        Ok(self.group.commit_ordered()?)
    }

    fn durable(self) -> Result<()> {
        Ok(self.group.commit()?)
    }
}

fn warn(warnings: &mut Vec<String>, message: &str) {
    if warnings.len() < 16 && !warnings.iter().any(|warning| warning == message) {
        warnings.push(message.into());
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FaultPoint {
    AfterTempSync,
    AfterBaseSync,
    PartialJournalAppend,
    BeforeJournalSync,
    BeforeJournalDirectorySync,
    AfterJournalRemove,
}

#[cfg(test)]
fn injected_error() -> Error {
    Error::Io(io::Error::other("injected recovery storage fault"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::run_state::{Entry, Mode, Status};
    use serde_json::json;

    fn setup() -> (tempfile::TempDir, Scope, Snapshot) {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(dir.path().join("input")).unwrap();
        let scope = Scope::new(
            Mode::Directory,
            &dir.path().join("input"),
            &dir.path().join("out"),
        )
        .unwrap();
        let mut snapshot = Snapshot {
            options: serde_json::from_value(
                json!({"input_dir": scope.input, "output_dir": scope.output}),
            )
            .unwrap(),
            ..Snapshot::default()
        };
        snapshot.documents.insert("a.txt".into(), Entry::default());
        (dir, scope, snapshot)
    }

    fn open(scope: &Scope) -> StateStore {
        StateStore::open(scope.clone(), "abc123", false, Limits::default()).unwrap()
    }

    fn loaded(store: &mut StateStore) -> (Snapshot, Vec<String>) {
        match store.load().unwrap() {
            LoadOutcome::Loaded { snapshot, warnings } => (*snapshot, warnings),
            other => panic!("Expected loaded state, got {other:?}"),
        }
    }

    fn file_key() -> ItemKey {
        ItemKey::File("a.txt".into())
    }

    #[test]
    fn stable_journal_append_replays_and_compaction_requires_a_new_namespace_fence() {
        let (_dir, scope, snapshot) = setup();
        let mut store = open(&scope);
        store.begin(snapshot).unwrap();
        store
            .record(file_key(), json!({"status":"failed","error":"first"}))
            .unwrap();
        assert_eq!(store.flush().unwrap(), 1);
        store.inject_fault(FaultPoint::BeforeJournalDirectorySync);
        store
            .record(file_key(), json!({"status":"completed","error":null}))
            .unwrap();
        assert_eq!(store.flush().unwrap(), 2);
        store.compact().unwrap();
        store
            .record(file_key(), json!({"status":"failed","error":"third"}))
            .unwrap();
        assert!(store.flush().is_err());
        assert_eq!(store.durable_sequence, 2);
        assert!(
            store
                .record(file_key(), json!({"status":"completed"}))
                .is_err()
        );
        // The last successful checkpoint remains complete even when the next
        // namespace fence fails; a visible but unacknowledged tail may replay.
        let base = codec::decode(
            &fs::read(&store.base).unwrap(),
            &scope,
            false,
            Limits::default(),
        )
        .unwrap();
        assert_eq!(base.documents["a.txt"].status, Status::Completed);
        assert_eq!(base.checkpoint.unwrap().applied_sequence, 2);
    }

    #[test]
    fn same_bytes_replacement_cannot_reuse_a_previous_journal_namespace_fence() {
        let (_dir, scope, snapshot) = setup();
        let mut store = open(&scope);
        store.begin(snapshot).unwrap();
        store
            .record(file_key(), json!({"status":"failed","error":"first"}))
            .unwrap();
        store.flush().unwrap();
        let original_inode = platform::status(&store.journal).unwrap().id();
        let mut replacement = tempfile::NamedTempFile::new_in(&store.directory).unwrap();
        replacement
            .write_all(&fs::read(&store.journal).unwrap())
            .unwrap();
        replacement.persist(&store.journal).unwrap();
        assert_ne!(
            original_inode,
            platform::status(&store.journal).unwrap().id()
        );
        store.inject_fault(FaultPoint::BeforeJournalDirectorySync);
        store
            .record(file_key(), json!({"status":"completed"}))
            .unwrap();
        assert!(store.flush().is_err());
        assert_eq!(store.durable_sequence, 1);
    }

    #[test]
    fn first_journal_namespace_failure_never_acknowledges_the_new_sequence() {
        let (_dir, scope, snapshot) = setup();
        let mut store = open(&scope);
        store.begin(snapshot).unwrap();
        store.inject_fault(FaultPoint::BeforeJournalDirectorySync);
        store
            .record(file_key(), json!({"status":"completed"}))
            .unwrap();
        assert!(store.flush().is_err());
        assert_eq!(store.durable_sequence, 0);
        assert!(store.flush().is_err());
    }

    #[test]
    fn legacy_takeover_preserves_raw_pair_before_replacing_it_and_only_once() {
        let (_dir, scope, snapshot) = setup();
        let mut store = open(&scope);
        let mut value: Value = serde_json::from_slice(
            &codec::encode(&snapshot, &scope, false, Limits::default()).unwrap(),
        )
        .unwrap();
        value["unrecognized_private_field"] = json!({"keep":"原始"});
        let original = format!("  {}\n\n", serde_json::to_string_pretty(&value).unwrap());
        let journal = b"{broken JSON\n{\"type\":\"file\",\"key\":\"a.txt\",\"data\":{\"status\":\"completed\"}}\n";
        fs::write(&store.base, &original).unwrap();
        fs::write(&store.journal, journal).unwrap();
        let (replayed, warnings) = loaded(&mut store);
        assert_eq!(replayed.documents["a.txt"].status, Status::Completed);
        assert!(!warnings.is_empty());
        store.begin(replayed).unwrap();
        let backup = store.legacy_backup().unwrap().to_owned();
        assert_eq!(
            fs::read(backup.join(store.base.file_name().unwrap())).unwrap(),
            original.as_bytes()
        );
        assert_eq!(
            fs::read(backup.join(store.journal.file_name().unwrap())).unwrap(),
            journal
        );
        assert!(!store.journal.exists());
        assert!(store.snapshot().unwrap().checkpoint.is_some());
        let before = fs::read_dir(&store.directory).unwrap().count();
        store.begin(store.snapshot().unwrap().clone()).unwrap();
        assert_eq!(fs::read_dir(&store.directory).unwrap().count(), before);
        drop(store);
        let mut reopened = open(&scope);
        let (saved, _) = loaded(&mut reopened);
        reopened.begin(saved).unwrap();
        assert!(reopened.legacy_backup().is_none());
        assert_eq!(fs::read_dir(&reopened.directory).unwrap().count(), before);
    }

    #[test]
    fn legacy_backup_survives_each_checkpoint_publication_failure_window() {
        for point in [
            FaultPoint::AfterTempSync,
            FaultPoint::AfterBaseSync,
            FaultPoint::AfterJournalRemove,
        ] {
            let (_dir, scope, snapshot) = setup();
            let mut store = open(&scope);
            let original = codec::encode(&snapshot, &scope, false, Limits::default()).unwrap();
            let journal =
                b"{\"type\":\"file\",\"key\":\"a.txt\",\"data\":{\"status\":\"completed\"}}\n";
            fs::write(&store.base, &original).unwrap();
            fs::write(&store.journal, journal).unwrap();
            let (saved, _) = loaded(&mut store);
            store.inject_fault(point);
            assert!(store.begin(saved).is_err(), "{point:?}");
            let backup = store.legacy_backup().unwrap().to_owned();
            assert_eq!(
                fs::read(backup.join(store.base.file_name().unwrap())).unwrap(),
                original
            );
            assert_eq!(
                fs::read(backup.join(store.journal.file_name().unwrap())).unwrap(),
                journal
            );
            drop(store);
            let (recovered, _) = loaded(&mut open(&scope));
            assert_eq!(recovered.documents["a.txt"].status, Status::Completed);
        }
    }

    #[test]
    fn invalid_new_scope_is_rejected_before_preserving_or_replacing_legacy_state() {
        let (_dir, scope, mut snapshot) = setup();
        let mut store = open(&scope);
        let original = codec::encode(&snapshot, &scope, false, Limits::default()).unwrap();
        fs::write(&store.base, &original).unwrap();
        fs::write(&store.journal, b"\n").unwrap();
        snapshot
            .options
            .insert("output_dir".into(), json!(scope.output.join("different")));
        assert!(matches!(store.begin(snapshot), Err(Error::ForeignScope(_))));
        assert_eq!(fs::read(&store.base).unwrap(), original);
        assert_eq!(fs::read(&store.journal).unwrap(), b"\n");
        assert!(store.legacy_backup().is_none());
        assert_eq!(fs::read_dir(&store.directory).unwrap().count(), 3);
    }

    #[test]
    fn refusing_oversized_legacy_backup_does_not_upgrade_or_truncate_state() {
        let (_dir, scope, snapshot) = setup();
        let limits = Limits {
            journal_bytes: 8,
            ..Limits::default()
        };
        let mut store = StateStore::open(scope.clone(), "abc123", false, limits).unwrap();
        let original = codec::encode(&snapshot, &scope, false, limits).unwrap();
        let journal = b"a journal larger than the configured limit\n";
        fs::write(&store.base, &original).unwrap();
        fs::write(&store.journal, journal).unwrap();
        assert!(store.begin(snapshot).is_err());
        assert_eq!(fs::read(&store.base).unwrap(), original);
        assert_eq!(fs::read(&store.journal).unwrap(), journal);
        assert!(store.legacy_backup().is_none());
        assert!(store.snapshot().is_none());
        assert_eq!(fs::read_dir(&store.directory).unwrap().count(), 3);
    }

    #[test]
    fn record_requires_checkpoint_and_flush_is_the_durable_acknowledgement() {
        let (_dir, scope, snapshot) = setup();
        let mut store = open(&scope);
        assert!(matches!(store.load().unwrap(), LoadOutcome::Missing));
        assert!(
            store
                .record(file_key(), json!({"status": "completed"}))
                .is_err()
        );
        store.begin(snapshot).unwrap();
        let base = fs::read(&store.base).unwrap();
        assert_eq!(
            store
                .record(file_key(), json!({"status": "completed"}))
                .unwrap(),
            1
        );
        assert_eq!(store.durable_sequence, 0);
        assert!(!store.journal.exists());
        assert_eq!(fs::read(&store.base).unwrap(), base);
        assert_eq!(store.flush().unwrap(), 1);
        assert_eq!(store.flush().unwrap(), 1);
        drop(store);
        let (snapshot, warnings) = loaded(&mut open(&scope));
        assert_eq!(snapshot.documents["a.txt"].status, Status::Completed);
        assert_eq!(snapshot.checkpoint.unwrap().applied_sequence, 1);
        assert!(warnings.is_empty());
    }

    #[test]
    fn drop_does_not_claim_to_flush_buffered_completion() {
        let (_dir, scope, snapshot) = setup();
        let mut store = open(&scope);
        store.begin(snapshot).unwrap();
        store
            .record(file_key(), json!({"status": "completed"}))
            .unwrap();
        drop(store);
        let (snapshot, _) = loaded(&mut open(&scope));
        assert_eq!(snapshot.documents["a.txt"].status, Status::Pending);
        assert_eq!(snapshot.checkpoint.unwrap().applied_sequence, 0);
    }

    #[test]
    fn invalid_or_oversized_mutations_leave_snapshot_and_pending_buffer_unchanged() {
        let (_dir, scope, snapshot) = setup();
        let mut store = StateStore::open(
            scope.clone(),
            "abc123",
            false,
            Limits {
                line_bytes: 256,
                ..Limits::default()
            },
        )
        .unwrap();
        store.begin(snapshot).unwrap();
        let before = store.snapshot().unwrap().clone();
        assert!(
            store
                .record(
                    ItemKey::File("unknown.txt".into()),
                    json!({"status": "completed"})
                )
                .is_err()
        );
        assert!(
            store
                .record(file_key(), json!({"status": "nonsense"}))
                .is_err()
        );
        assert!(matches!(
            store.record(
                file_key(),
                json!({"status": "failed", "error": "x".repeat(600)})
            ),
            Err(Error::Limit(_))
        ));
        assert!(
            store
                .record(
                    file_key(),
                    json!({"status": "in_progress", "target": scope.input.join("outside.md")})
                )
                .is_err()
        );
        assert_eq!(store.snapshot(), Some(&before));
        assert!(store.pending.is_empty());
        assert_eq!(store.flush().unwrap(), 0);
        assert_eq!(
            store
                .record(file_key(), json!({"status": "completed"}))
                .unwrap(),
            1
        );
    }

    #[test]
    fn new_items_must_be_checkpointed_before_events_can_reference_them() {
        let (_dir, scope, snapshot) = setup();
        let mut store = open(&scope);
        store.begin(snapshot).unwrap();
        let key = ItemKey::File("new.txt".into());
        assert!(
            store
                .record(key.clone(), json!({"status": "completed"}))
                .is_err()
        );
        let mut merged = store.snapshot().unwrap().clone();
        merged.documents.insert("new.txt".into(), Entry::default());
        store.begin(merged).unwrap();
        let saved: Value = serde_json::from_slice(&fs::read(&store.base).unwrap()).unwrap();
        assert_eq!(saved["documents"]["new.txt"]["status"], "pending");
        store.record(key, json!({"status": "completed"})).unwrap();
        store.flush().unwrap();
        drop(store);
        assert_eq!(
            loaded(&mut open(&scope)).0.documents["new.txt"].status,
            Status::Completed
        );
    }

    #[test]
    fn compacted_base_fence_survives_failure_before_old_journal_removal() {
        let (_dir, scope, snapshot) = setup();
        let mut store = open(&scope);
        store.begin(snapshot).unwrap();
        store
            .record(
                file_key(),
                json!({"status": "in_progress", "target": scope.output.join("a.txt.md")}),
            )
            .unwrap();
        store.flush().unwrap();
        let old_journal = fs::read(&store.journal).unwrap();
        store
            .record(
                file_key(),
                json!({"status": "completed", "output": scope.output.join("a.txt.md")}),
            )
            .unwrap();
        store.inject_fault(FaultPoint::AfterBaseSync);
        assert!(store.compact().is_err());
        assert_eq!(fs::read(&store.journal).unwrap(), old_journal);
        assert!(store.flush().is_err());
        assert!(
            store
                .record(file_key(), json!({"status": "failed"}))
                .is_err()
        );
        drop(store);
        let (snapshot, warnings) = loaded(&mut open(&scope));
        assert_eq!(snapshot.documents["a.txt"].status, Status::Completed);
        assert_eq!(snapshot.checkpoint.unwrap().applied_sequence, 2);
        assert!(warnings.is_empty());
    }

    #[test]
    fn failure_before_base_replace_keeps_old_base_and_durable_events() {
        let (_dir, scope, snapshot) = setup();
        let mut store = open(&scope);
        store.begin(snapshot).unwrap();
        store
            .record(
                file_key(),
                json!({"status": "failed", "error": "first failure"}),
            )
            .unwrap();
        store.flush().unwrap();
        let base = fs::read(&store.base).unwrap();
        let journal = fs::read(&store.journal).unwrap();
        store
            .record(file_key(), json!({"status": "completed"}))
            .unwrap();
        store.inject_fault(FaultPoint::AfterTempSync);
        assert!(store.compact().is_err());
        assert_eq!(fs::read(&store.base).unwrap(), base);
        assert_eq!(fs::read(&store.journal).unwrap(), journal);
        assert!(!fs::read_dir(&store.directory).unwrap().any(|entry| {
            entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .ends_with(".tmp")
        }));
        drop(store);
        let (snapshot, _) = loaded(&mut open(&scope));
        assert_eq!(snapshot.documents["a.txt"].status, Status::Failed);
        assert_eq!(
            snapshot.documents["a.txt"].error.as_deref(),
            Some("first failure")
        );
        assert_eq!(snapshot.checkpoint.unwrap().applied_sequence, 1);
    }

    #[test]
    fn partial_append_poisons_writer_and_begin_repairs_torn_suffix() {
        let (_dir, scope, snapshot) = setup();
        let mut store = open(&scope);
        store.begin(snapshot).unwrap();
        store
            .record(file_key(), json!({"status": "completed"}))
            .unwrap();
        store.inject_fault(FaultPoint::PartialJournalAppend);
        assert!(store.flush().is_err());
        let partial = fs::read(&store.journal).unwrap();
        assert!(!partial.is_empty());
        assert!(!partial.ends_with(b"\n"));
        assert!(store.flush().is_err());
        assert!(store.compact().is_err());
        assert_eq!(fs::read(&store.journal).unwrap(), partial);
        drop(store);
        let mut store = open(&scope);
        let (snapshot, warnings) = loaded(&mut store);
        assert!(!warnings.is_empty());
        assert_eq!(snapshot.checkpoint.as_ref().unwrap().applied_sequence, 0);
        assert!(
            store
                .record(file_key(), json!({"status": "completed"}))
                .is_err()
        );
        store.begin(snapshot).unwrap();
        assert!(!store.journal.exists());
        store
            .record(file_key(), json!({"status": "completed"}))
            .unwrap();
        store.flush().unwrap();
        drop(store);
        assert_eq!(
            loaded(&mut open(&scope)).0.documents["a.txt"].status,
            Status::Completed
        );
    }

    #[test]
    fn sync_failure_never_returns_durable_ack_and_requires_reopen() {
        let (_dir, scope, snapshot) = setup();
        let mut store = open(&scope);
        store.begin(snapshot).unwrap();
        store
            .record(file_key(), json!({"status": "completed"}))
            .unwrap();
        store.inject_fault(FaultPoint::BeforeJournalSync);
        assert!(store.flush().is_err());
        assert_eq!(store.durable_sequence, 0);
        assert_eq!(store.pending.len(), 1);
        assert!(store.begin(store.snapshot().unwrap().clone()).is_err());
        assert!(store.flush().is_err());
    }

    #[test]
    fn cleanup_sync_failure_retains_committed_base() {
        let (_dir, scope, snapshot) = setup();
        let mut store = open(&scope);
        store.begin(snapshot).unwrap();
        store
            .record(file_key(), json!({"status": "completed"}))
            .unwrap();
        store.flush().unwrap();
        store.inject_fault(FaultPoint::AfterJournalRemove);
        assert!(store.compact().is_err());
        assert!(!store.journal.exists());
        drop(store);
        assert_eq!(
            loaded(&mut open(&scope)).0.documents["a.txt"].status,
            Status::Completed
        );
    }

    #[test]
    fn corrupt_originals_are_quarantined_before_a_failed_replacement() {
        let (_dir, scope, snapshot) = setup();
        let mut store = open(&scope);
        fs::write(&store.base, b"{damaged base").unwrap();
        fs::write(&store.journal, b"damaged journal\xff").unwrap();
        assert!(matches!(store.load().unwrap(), LoadOutcome::Corrupt { .. }));
        store.inject_fault(FaultPoint::AfterTempSync);
        assert!(store.begin(snapshot).is_err());
        assert_eq!(fs::read(&store.base).unwrap(), b"{damaged base");
        assert_eq!(fs::read(&store.journal).unwrap(), b"damaged journal\xff");
        let mut preserved: Vec<_> = fs::read_dir(&store.directory)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| {
                path.file_name()
                    .unwrap()
                    .to_string_lossy()
                    .contains(".corrupt.")
            })
            .map(|path| fs::read(path).unwrap())
            .collect();
        preserved.sort();
        let mut expected = vec![b"{damaged base".to_vec(), b"damaged journal\xff".to_vec()];
        expected.sort();
        assert_eq!(preserved, expected);
    }

    #[cfg(unix)]
    #[test]
    fn quarantine_does_not_relax_private_state_permissions() {
        use std::os::unix::fs::PermissionsExt;
        let (_dir, scope, snapshot) = setup();
        let mut store = open(&scope);
        fs::write(&store.base, b"broken private base").unwrap();
        fs::write(&store.journal, b"private journal").unwrap();
        fs::set_permissions(&store.base, fs::Permissions::from_mode(0o600)).unwrap();
        fs::set_permissions(&store.journal, fs::Permissions::from_mode(0o600)).unwrap();
        store.begin(snapshot).unwrap();
        let copies: Vec<_> = fs::read_dir(&store.directory)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| {
                path.file_name()
                    .unwrap()
                    .to_string_lossy()
                    .contains(".corrupt.")
            })
            .collect();
        assert_eq!(copies.len(), 2);
        for path in copies {
            assert_eq!(
                fs::metadata(path).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
    }

    /// Fence kinds of each step: bytes before names, a checkpoint name before
    /// its journal's removal, and exactly one durable fence per acknowledgement.
    #[test]
    fn checkpoint_steps_are_ordered_and_each_acknowledgement_is_one_durable_fence() {
        use crate::output_claims::sync_group::take_commits;
        let (_dir, scope, snapshot) = setup();
        take_commits();
        let mut store = open(&scope);
        // The created output/.markitai/states chain precedes the checkpoint.
        assert_eq!(take_commits(), ["ordered"]);
        assert!(store.unfenced);
        store.begin(snapshot).unwrap();
        // Snapshot bytes, then its name; no acknowledgement yet.
        assert_eq!(take_commits(), ["ordered", "ordered"]);
        assert_eq!(store.durable_sequence, 0);
        // The first flush completes the checkpoint's fence even when idle.
        assert_eq!(store.flush().unwrap(), 0);
        assert_eq!(take_commits(), ["durable"]);
        assert_eq!(store.flush().unwrap(), 0);
        assert!(take_commits().is_empty());
        store
            .record(file_key(), json!({"status": "failed", "error": "first"}))
            .unwrap();
        // A new journal's bytes precede its directory entry.
        assert_eq!(store.flush().unwrap(), 1);
        assert_eq!(take_commits(), ["ordered", "durable"]);
        store
            .record(file_key(), json!({"status": "completed"}))
            .unwrap();
        assert_eq!(store.flush().unwrap(), 2);
        assert_eq!(take_commits(), ["durable"]);
        // Base bytes, base name before the journal's removal, then durable.
        store.compact().unwrap();
        assert_eq!(take_commits(), ["ordered", "ordered", "durable"]);
        assert!(!store.journal.exists());
        assert!(!store.unfenced);
        // Without a journal to remove, the name's fence is the durable one.
        store.compact().unwrap();
        assert_eq!(take_commits(), ["ordered", "durable"]);
        assert_eq!(store.flush().unwrap(), 2);
        assert!(take_commits().is_empty());
        store
            .record(file_key(), json!({"status": "completed"}))
            .unwrap();
        assert_eq!(store.flush().unwrap(), 3);
        assert_eq!(take_commits(), ["ordered", "durable"]);
        // Reopening an existing chain creates nothing and fences nothing.
        drop(store);
        let mut reopened = open(&scope);
        assert!(take_commits().is_empty());
        assert!(!reopened.unfenced);
        let (saved, _) = loaded(&mut reopened);
        assert_eq!(saved.documents["a.txt"].status, Status::Completed);
        assert_eq!(saved.checkpoint.as_ref().unwrap().applied_sequence, 3);
        reopened
            .record(file_key(), json!({"status": "failed"}))
            .unwrap_err();
        // A resumed checkpoint names its base before removing the journal,
        // whose removal then waits for the next durable fence.
        assert!(reopened.journal.exists());
        reopened.begin(saved).unwrap();
        assert_eq!(take_commits(), ["ordered", "ordered"]);
        assert!(!reopened.journal.exists());
        assert_eq!(reopened.durable_sequence, 3);
        reopened
            .record(file_key(), json!({"status": "failed", "error": "again"}))
            .unwrap();
        assert_eq!(reopened.flush().unwrap(), 4);
        assert_eq!(take_commits(), ["ordered", "durable"]);
    }

    /// An ordered checkpoint never advances the acknowledged sequence; only a
    /// durable fence does, including after a compaction forced by capacity.
    #[test]
    fn ordered_compaction_waits_for_the_next_flush_to_acknowledge() {
        use crate::output_claims::sync_group::take_commits;
        let (_dir, scope, snapshot) = setup();
        let limits = Limits {
            journal_bytes: 300,
            line_bytes: 299,
            ..Limits::default()
        };
        let mut store = StateStore::open(scope.clone(), "abc123", false, limits).unwrap();
        store.begin(snapshot).unwrap();
        assert_eq!(store.flush().unwrap(), 0);
        take_commits();
        store
            .record(
                file_key(),
                json!({"status": "failed", "error": "x".repeat(70)}),
            )
            .unwrap();
        assert!(take_commits().is_empty());
        // Capacity compacts the buffered sequence 1 into the base: ordered,
        // so it is not acknowledged until the next flush.
        store
            .record(file_key(), json!({"status": "completed"}))
            .unwrap();
        assert_eq!(take_commits(), ["ordered", "ordered"]);
        let base: Value = serde_json::from_slice(&fs::read(&store.base).unwrap()).unwrap();
        assert_eq!(base["_markitai"]["applied_sequence"], 1);
        assert_eq!(store.durable_sequence, 0);
        assert!(store.unfenced);
        assert_eq!(store.flush().unwrap(), 2);
        assert_eq!(take_commits(), ["ordered", "durable"]);
        assert!(!store.unfenced);
        drop(store);
        let mut reopened = StateStore::open(scope.clone(), "abc123", false, limits).unwrap();
        let (saved, warnings) = loaded(&mut reopened);
        assert_eq!(saved.documents["a.txt"].status, Status::Completed);
        assert_eq!(saved.checkpoint.unwrap().applied_sequence, 2);
        assert!(warnings.is_empty());
    }

    /// Store operations share one observation of their own paths, never one
    /// from an earlier call: an output parent replaced by a link is rejected
    /// by the next operation, and an ordinary directory is accepted again.
    #[cfg(unix)]
    #[test]
    fn each_storage_operation_observes_a_substituted_output_afresh() {
        use std::os::unix::fs::symlink;
        let (dir, scope, snapshot) = setup();
        let mut store = open(&scope);
        assert!(!crate::run_state::paths::active());
        store.begin(snapshot).unwrap();
        store
            .record(file_key(), json!({"status": "failed", "error": "first"}))
            .unwrap();
        assert!(!crate::run_state::paths::active());
        let output = dir.path().join("out");
        let moved = dir.path().join("moved");
        fs::rename(&output, &moved).unwrap();
        symlink(&moved, &output).unwrap();
        assert!(matches!(store.flush(), Err(Error::ForeignScope(_))));
        assert_eq!(store.durable_sequence, 0);
        assert!(!crate::run_state::paths::active());
        // The rejected flush poisoned this writer; a new one sees the link.
        drop(store);
        assert!(matches!(
            StateStore::open(scope.clone(), "abc123", false, Limits::default()),
            Err(Error::ForeignScope(_))
        ));
        fs::remove_file(&output).unwrap();
        fs::rename(&moved, &output).unwrap();
        let mut store = open(&scope);
        let (saved, _) = loaded(&mut store);
        store.begin(saved).unwrap();
        store
            .record(file_key(), json!({"status": "completed"}))
            .unwrap();
        // A link substituted for the states directory between a record and
        // its flush is caught at the flush, as before.
        let states = scope.output.join(".markitai/states");
        let held = dir.path().join("held-states");
        fs::rename(&states, &held).unwrap();
        symlink(&held, &states).unwrap();
        assert!(matches!(store.flush(), Err(Error::ForeignScope(_))));
        fs::remove_file(&states).unwrap();
        fs::rename(&held, &states).unwrap();
        drop(store);
        let (saved, _) = loaded(&mut open(&scope));
        assert_eq!(saved.documents["a.txt"].status, Status::Pending);
    }

    #[test]
    fn journal_capacity_compacts_buffer_before_accepting_next_event() {
        let (_dir, scope, snapshot) = setup();
        let limits = Limits {
            journal_bytes: 300,
            line_bytes: 299,
            ..Limits::default()
        };
        let mut store = StateStore::open(scope.clone(), "abc123", false, limits).unwrap();
        store.begin(snapshot).unwrap();
        store
            .record(
                file_key(),
                json!({"status": "failed", "error": "x".repeat(70)}),
            )
            .unwrap();
        store
            .record(file_key(), json!({"status": "completed"}))
            .unwrap();
        let base: Value = serde_json::from_slice(&fs::read(&store.base).unwrap()).unwrap();
        assert_eq!(base["_markitai"]["applied_sequence"], 1);
        assert_eq!(base["documents"]["a.txt"]["status"], "failed");
        assert_eq!(store.flush().unwrap(), 2);
        assert!(fs::metadata(&store.journal).unwrap().len() <= limits.journal_bytes as u64);
        drop(store);
        assert_eq!(
            loaded(&mut open(&scope)).0.documents["a.txt"].status,
            Status::Completed
        );
    }

    #[test]
    fn duplicate_sequence_after_initial_watermark_stops_before_later_mutation() {
        let (_dir, scope, snapshot) = setup();
        let mut store = open(&scope);
        store.begin(snapshot).unwrap();
        store
            .record(file_key(), json!({"status": "failed", "error": "first"}))
            .unwrap();
        store.flush().unwrap();
        let mut first = fs::read(&store.journal).unwrap();
        first.extend(first.clone());
        let checkpoint = store.snapshot().unwrap().checkpoint.as_ref().unwrap();
        let later = Event {
            key: file_key(),
            data: json!({"status": "completed"}),
            fence: Some(Fence {
                generation: checkpoint.generation.clone(),
                sequence: 2,
            }),
        };
        first.extend(codec::encode_event(&later).unwrap());
        first.push(b'\n');
        fs::write(&store.journal, first).unwrap();
        drop(store);
        let (snapshot, warnings) = loaded(&mut open(&scope));
        assert_eq!(snapshot.documents["a.txt"].status, Status::Failed);
        assert_eq!(snapshot.checkpoint.unwrap().applied_sequence, 1);
        assert!(warnings.iter().any(|warning| warning.contains("sequence")));
    }

    #[test]
    fn normal_file_checks_reject_directory_storage_without_changing_it() {
        let (_dir, scope, _snapshot) = setup();
        let mut store = open(&scope);
        fs::create_dir(&store.base).unwrap();
        fs::write(store.base.join("keep"), b"owned elsewhere").unwrap();
        assert!(matches!(store.load(), Err(Error::Invalid(_))));
        assert_eq!(
            fs::read(store.base.join("keep")).unwrap(),
            b"owned elsewhere"
        );
    }

    #[cfg(unix)]
    #[test]
    fn new_journal_keeps_the_same_private_permissions_as_the_base() {
        use std::os::unix::fs::PermissionsExt;
        let (_dir, scope, snapshot) = setup();
        let mut store = open(&scope);
        store.begin(snapshot).unwrap();
        store
            .record(
                file_key(),
                json!({"status": "failed", "error": "private error detail"}),
            )
            .unwrap();
        store.flush().unwrap();
        for path in [&store.base, &store.journal] {
            assert_eq!(
                fs::metadata(path).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
    }

    #[test]
    fn custom_large_line_limit_has_the_same_write_and_replay_boundary() {
        let (_dir, scope, snapshot) = setup();
        let limit = Limits::default().line_bytes + 4096;
        let limits = Limits {
            line_bytes: limit,
            ..Limits::default()
        };
        let error = "e".repeat(Limits::default().line_bytes + 32);
        let mut store = StateStore::open(scope.clone(), "abc123", false, limits).unwrap();
        store.begin(snapshot).unwrap();
        assert_eq!(
            store
                .record(file_key(), json!({"status": "failed", "error": error}))
                .unwrap(),
            1
        );
        assert_eq!(store.flush().unwrap(), 1);
        assert!(fs::metadata(&store.journal).unwrap().len() > Limits::default().line_bytes as u64);
        drop(store);
        let mut reopened = StateStore::open(scope, "abc123", false, limits).unwrap();
        let (snapshot, warnings) = loaded(&mut reopened);
        assert_eq!(
            snapshot.documents["a.txt"].error.as_deref(),
            Some(error.as_str())
        );
        assert_eq!(snapshot.checkpoint.unwrap().applied_sequence, 1);
        assert!(warnings.is_empty());
    }

    #[cfg(unix)]
    fn relative_to_current_dir(target: &Path) -> PathBuf {
        use std::path::Component;
        let cwd = std::env::current_dir().unwrap();
        let mut relative = PathBuf::new();
        for component in cwd.components() {
            if matches!(component, Component::Normal(_)) {
                relative.push("..");
            }
        }
        for component in target.components() {
            if let Component::Normal(name) = component {
                relative.push(name);
            }
        }
        assert!(!relative.is_absolute());
        assert_eq!(
            crate::report_store::resolve_path(&relative).unwrap(),
            target
        );
        relative
    }

    #[cfg(unix)]
    #[test]
    fn relative_destinations_are_durable_across_a_different_process_cwd() {
        let (_dir, scope, snapshot) = setup();
        let target = scope.output.join("a.txt.md");
        let relative = relative_to_current_dir(&target);
        let mut store = open(&scope);
        store.begin(snapshot).unwrap();
        store
            .record(
                file_key(),
                json!({"status": "in_progress", "target": relative, "output": null}),
            )
            .unwrap();
        store
            .record(
                file_key(),
                json!({"status": "completed", "output": relative}),
            )
            .unwrap();
        assert_eq!(store.flush().unwrap(), 2);
        let journal = fs::read_to_string(&store.journal).unwrap();
        let lines: Vec<Value> = journal
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(lines[0]["data"]["target"], json!(target));
        assert!(lines[0]["data"].get("output").unwrap().is_null());
        assert_eq!(lines[1]["data"]["output"], json!(target));
        assert!(lines[1]["data"].get("target").is_none());
        drop(store);
        let different_cwd = tempfile::tempdir().unwrap();
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "run_state::store::tests::relative_destination_replay_child",
                "--ignored",
                "--nocapture",
            ])
            .current_dir(different_cwd.path())
            .env("MARKITAI_TEST_STATE_REPLAY_INPUT", &scope.input)
            .env("MARKITAI_TEST_STATE_REPLAY_OUTPUT", &scope.output)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "child replay failed: {}{}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(String::from_utf8_lossy(&result.stdout).contains("1 passed"));
    }

    #[cfg(unix)]
    #[test]
    #[ignore = "subprocess helper for changed-cwd replay"]
    fn relative_destination_replay_child() {
        let input = PathBuf::from(
            std::env::var_os("MARKITAI_TEST_STATE_REPLAY_INPUT").expect("private child input"),
        );
        let output = PathBuf::from(
            std::env::var_os("MARKITAI_TEST_STATE_REPLAY_OUTPUT").expect("private child output"),
        );
        let scope = Scope::new(Mode::Directory, &input, &output).unwrap();
        let (snapshot, warnings) = loaded(&mut open(&scope));
        let entry = &snapshot.documents["a.txt"];
        assert_eq!(entry.status, Status::Completed);
        assert_eq!(
            entry.output.as_deref(),
            Some(output.join("a.txt.md").as_path())
        );
        assert_eq!(
            entry.target.as_deref(),
            Some(output.join("a.txt.md").as_path())
        );
        assert_eq!(snapshot.checkpoint.unwrap().applied_sequence, 2);
        assert!(warnings.is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn native_url_checkpoint_anchors_relative_scope_and_list_provenance() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(dir.path().join("lists")).unwrap();
        let input = dir.path().join("lists/jobs.urls");
        fs::write(&input, "https://example.invalid/page\n").unwrap();
        let scope = Scope::new(Mode::UrlList, &input, &dir.path().join("out")).unwrap();
        let mut snapshot = Snapshot {
            options: serde_json::from_value(json!({
                "input_dir": relative_to_current_dir(scope.input.parent().unwrap()),
                "output_dir": relative_to_current_dir(&scope.output)
            }))
            .unwrap(),
            ..Snapshot::default()
        };
        snapshot.urls.insert(
            "https://example.invalid/page".into(),
            Entry {
                status: Status::Failed,
                source_file: Some(
                    relative_to_current_dir(&scope.input)
                        .to_string_lossy()
                        .into_owned(),
                ),
                target: Some(relative_to_current_dir(&scope.output.join("page.md"))),
                error: Some("interrupted earlier".into()),
                ..Entry::default()
            },
        );
        let mut store = open(&scope);
        store.begin(snapshot).unwrap();
        let saved: Value = serde_json::from_slice(&fs::read(&store.base).unwrap()).unwrap();
        assert_eq!(
            saved["options"]["input_dir"],
            json!(scope.input.parent().unwrap())
        );
        assert_eq!(saved["options"]["output_dir"], json!(scope.output));
        assert_eq!(
            saved["urls"]["https://example.invalid/page"]["source_file"],
            json!(scope.input)
        );
        assert_eq!(
            saved["urls"]["https://example.invalid/page"]["target"],
            json!(scope.output.join("page.md"))
        );
        store
            .record(
                ItemKey::Url("https://example.invalid/page".into()),
                json!({
                    "status": "failed",
                    "error": "retried after checkpoint",
                    "source_file": relative_to_current_dir(&scope.input),
                    "target": relative_to_current_dir(&scope.output.join("page.md"))
                }),
            )
            .unwrap();
        assert_eq!(store.flush().unwrap(), 1);
        let event: Value = serde_json::from_slice(&fs::read(&store.journal).unwrap()).unwrap();
        assert_eq!(event["data"]["source_file"], json!(scope.input));
        assert_eq!(event["data"]["target"], json!(scope.output.join("page.md")));
        drop(store);
        let different_cwd = tempfile::tempdir().unwrap();
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "run_state::store::tests::relative_url_checkpoint_child",
                "--ignored",
                "--nocapture",
            ])
            .current_dir(different_cwd.path())
            .env("MARKITAI_TEST_STATE_REPLAY_INPUT", &scope.input)
            .env("MARKITAI_TEST_STATE_REPLAY_OUTPUT", &scope.output)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "child checkpoint load failed: {}{}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(String::from_utf8_lossy(&result.stdout).contains("1 passed"));
    }

    #[cfg(unix)]
    #[test]
    #[ignore = "subprocess helper for relative URL checkpoint provenance"]
    fn relative_url_checkpoint_child() {
        let input = PathBuf::from(
            std::env::var_os("MARKITAI_TEST_STATE_REPLAY_INPUT").expect("private child input"),
        );
        let output = PathBuf::from(
            std::env::var_os("MARKITAI_TEST_STATE_REPLAY_OUTPUT").expect("private child output"),
        );
        let scope = Scope::new(Mode::UrlList, &input, &output).unwrap();
        let (snapshot, warnings) = loaded(&mut open(&scope));
        let entry = &snapshot.urls["https://example.invalid/page"];
        assert_eq!(entry.status, Status::Failed);
        assert_eq!(entry.source_file.as_deref(), Some(input.to_str().unwrap()));
        assert_eq!(
            entry.target.as_deref(),
            Some(output.join("page.md").as_path())
        );
        assert_eq!(entry.error.as_deref(), Some("retried after checkpoint"));
        assert_eq!(
            snapshot.options["input_dir"],
            json!(input.parent().unwrap())
        );
        assert_eq!(snapshot.options["output_dir"], json!(output));
        assert_eq!(snapshot.checkpoint.unwrap().applied_sequence, 1);
        assert!(warnings.is_empty());
    }
}
