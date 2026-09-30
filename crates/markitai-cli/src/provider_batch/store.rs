//! Private supplier-job evidence. This store never publishes document outputs.
use crate::output_claims::Owner;
use markitai_core::{
    ConversionUsage,
    provider_batch::{Batch, RemoteIdentity, UploadedInput},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs::{self, File, OpenOptions, TryLockError};
use std::io::{self, Read, Write};
use std::path::{Component, Path, PathBuf};
use tempfile::NamedTempFile;

pub(crate) type Result<T> = std::result::Result<T, Error>;
#[derive(Debug)]
pub(crate) enum Error {
    Invalid(&'static str),
    NotFound,
    Io,
    Busy,
    Overlap,
    Conflict,
    Durability,
    Unsupported,
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Invalid(message) => message,
            Self::NotFound => "Provider batch job was not found",
            Self::Io => "Provider batch storage operation failed",
            Self::Busy => "Another process is using this provider batch store",
            Self::Overlap => "An unfinished provider batch already owns part of this output family; collect or resolve it before submitting again",
            Self::Conflict => "Provider batch evidence conflicts with previously recorded data",
            Self::Durability => "Provider batch state was published but directory synchronization failed; reopen the store before continuing",
            Self::Unsupported => "Durable provider batch ownership is not supported on this platform",
        })
    }
}
impl std::error::Error for Error {}
impl From<io::Error> for Error {
    fn from(_: io::Error) -> Self {
        Self::Io
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Limits {
    pub state_bytes: usize,
    pub blob_bytes: usize,
    pub request_bytes: usize,
    pub total_bytes: u64,
    pub items: usize,
    pub jobs: usize,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            state_bytes: 16 * 1024 * 1024,
            blob_bytes: 8 * 1024 * 1024,
            request_bytes: 200_000_000,
            total_bytes: 1024 * 1024 * 1024,
            items: 10_000,
            jobs: 256,
        }
    }
}
impl Limits {
    fn check(self) -> Result<Self> {
        let max = Self::default();
        if self.state_bytes == 0
            || self.state_bytes > max.state_bytes
            || self.blob_bytes == 0
            || self.blob_bytes > max.blob_bytes
            || self.request_bytes == 0
            || self.request_bytes > max.request_bytes
            || self.total_bytes == 0
            || self.total_bytes > max.total_bytes
            || self.items == 0
            || self.items > max.items
            || self.jobs == 0
            || self.jobs > max.jobs
        {
            return Err(Error::Invalid("Provider batch limits are invalid"));
        }
        Ok(self)
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct Endpoint {
    pub provider: String,
    pub api_base: String,
    pub model: String,
}

pub(crate) struct NewJob {
    pub id: String,
    pub input_root: PathBuf,
    pub endpoint: Endpoint,
    pub items: Vec<NewItem>,
}
pub(crate) struct NewItem {
    pub custom_id: String,
    pub source: String,
    pub key: String,
    pub base: PathBuf,
    pub enhanced: PathBuf,
    pub base_sha256: String,
    pub owner: Owner,
    pub plan: Value,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Phase {
    Prepared,
    Uploaded,
    Creating,
    CreateUncertain,
    Rejected,
    Submitted,
    Collected,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct Blob {
    pub path: PathBuf,
    pub bytes: u64,
    pub sha256: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RecordedResult {
    pub blob: Blob,
    pub usage: ConversionUsage,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct Published {
    pub path: PathBuf,
    pub bytes: u64,
    pub sha256: String,
    /// Canonical validated receipt encoding hash (Claim::evidence_digest), not raw disk bytes.
    /// This records evidence metadata, never authority to bypass Claim checks.
    pub receipt_sha256: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Item {
    pub custom_id: String,
    pub source: String,
    pub key: String,
    pub base: PathBuf,
    pub enhanced: PathBuf,
    pub base_sha256: String,
    pub owner: Owner,
    pub plan: Blob,
    pub result: Option<RecordedResult>,
    pub finalized: Option<Published>,
    pub error: Option<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Job {
    pub version: u8,
    pub id: String,
    pub input_root: PathBuf,
    pub output_root: PathBuf,
    pub endpoint: Endpoint,
    pub phase: Phase,
    pub items: Vec<Item>,
    pub requests: Option<Blob>,
    pub uploaded: Option<UploadedInput>,
    pub batch: Option<Batch>,
}
#[derive(Clone, Debug)]
pub(crate) struct PendingSummary {
    pub id: String,
    pub batch_id: Option<String>,
    pub input_root: PathBuf,
    pub phase: Phase,
}
pub(crate) struct RecordResult {
    pub inserted: bool,
    pub usage: ConversionUsage,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Identity {
    device: u64,
    inode: u64,
}
struct Directory {
    path: PathBuf,
    identity: Identity,
    private: bool,
}
struct HeldLock {
    path: PathBuf,
    file: File,
    identity: Identity,
}
struct Root {
    output: PathBuf,
    path: PathBuf,
    directories: Vec<Directory>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ResumeStep {
    PrepareRequests,
    Upload,
    Create,
    Reconcile,
    Collect,
    Done,
}

pub(crate) struct Preparation {
    root: Root,
    lock: HeldLock,
}
impl Preparation {
    pub(crate) fn validate(&self) -> Result<()> {
        self.root.validate()?;
        self.lock.validate()
    }

    /// Select frozen work while holding submission exclusion, then acquire its collector lock.
    /// Input is an optional resolved scope identity; it need not exist anymore.
    pub(crate) fn open_frozen(
        &self,
        input_scope: Option<&Path>,
        limits: Limits,
    ) -> Result<Option<Store>> {
        self.validate()?;
        let limits = limits.check()?;
        if input_scope.is_some_and(|input| !absolute_path(input)) {
            return Err(Error::Invalid(
                "Provider batch input scope must be an absolute resolved path",
            ));
        }
        let global = self.root.lock()?;
        let jobs = scan(&self.root, limits)?;
        let mut pending = jobs
            .iter()
            .filter(|job| !matches!(job.phase, Phase::Collected | Phase::Rejected));
        let Some(job) = pending.next() else {
            return Ok(None);
        };
        if pending.next().is_some() {
            return Err(Error::Invalid(
                "Multiple unfinished provider batches require explicit collection",
            ));
        }
        if input_scope.is_some_and(|input| input != job.input_root) {
            return Err(Error::Conflict);
        }
        let id = job.id.clone();
        global.validate()?;
        drop(global);
        let store = Store::open_id(Root::open(&self.root.output, false, false)?, &id, limits)?;
        self.validate()?;
        // Another collector may have completed it between the snapshot and job-lock acquisition.
        if matches!(store.state.phase, Phase::Collected | Phase::Rejected) {
            return Ok(None);
        }
        Ok(Some(store))
    }
}
pub(crate) struct Store {
    root: Root,
    path: PathBuf,
    directory: Directory,
    lock: HeldLock,
    state: Job,
    limits: Limits,
    poisoned: bool,
    #[cfg(test)]
    fault: Option<Fault>,
}
#[cfg(test)]
#[derive(Clone, Copy, PartialEq, Eq)]
enum Fault {
    BeforeStateReplace,
    AfterStateReplace,
}

impl Store {
    pub(crate) fn lock_preparation(output: &Path, allow_symlinks: bool) -> Result<Preparation> {
        let root = Root::open(output, allow_symlinks, true)?;
        let lock = HeldLock::open(&root.path.join("submission.lock"))?;
        let preparation = Preparation { root, lock };
        preparation.validate()?;
        Ok(preparation)
    }
    pub(crate) fn create(
        output: &Path,
        job: NewJob,
        allow_symlinks: bool,
        limits: Limits,
    ) -> Result<Self> {
        let limits = limits.check()?;
        let root = Root::open(output, allow_symlinks, true)?;
        let global = root.lock()?;
        validate_new(&job, &root.output, limits)?;
        let existing = scan(&root, limits)?;
        if existing.len() >= limits.jobs {
            return Err(Error::Invalid("Provider batch job limit reached"));
        }
        let members: HashSet<_> = job
            .items
            .iter()
            .flat_map(|item| [&item.base, &item.enhanced])
            .collect();
        if existing
            .iter()
            .filter(|state| !matches!(state.phase, Phase::Collected | Phase::Rejected))
            .any(|state| {
                state
                    .items
                    .iter()
                    .any(|item| members.contains(&item.base) || members.contains(&item.enhanced))
            })
        {
            return Err(Error::Overlap);
        }
        let path = root.path.join(format!("attempt-{}", job.id));
        new_directory(&path)?;
        let directory = Directory::read(&path, true)?;
        new_directory(&path.join("plans"))?;
        new_directory(&path.join("results"))?;
        let lock = HeldLock::open(&path.join("job.lock"))?;
        let mut items = Vec::with_capacity(job.items.len());
        let mut stored = 0u64;
        for (index, item) in job.items.into_iter().enumerate() {
            let relative = PathBuf::from(format!("plans/{index}.json"));
            let allowance = usize::try_from(limits.total_bytes.saturating_sub(stored))
                .unwrap_or(usize::MAX)
                .min(limits.blob_bytes);
            let blob = write_json_blob(&path, &relative, &item.plan, allowance)?;
            stored = stored
                .checked_add(blob.bytes)
                .ok_or(Error::Invalid("Provider batch storage limit reached"))?;
            if stored > limits.total_bytes {
                return Err(Error::Invalid("Provider batch storage limit reached"));
            }
            items.push(Item {
                custom_id: item.custom_id,
                source: item.source,
                key: item.key,
                base: item.base,
                enhanced: item.enhanced,
                base_sha256: item.base_sha256,
                owner: item.owner,
                plan: blob,
                result: None,
                finalized: None,
                error: None,
            });
        }
        let state = Job {
            version: 1,
            id: job.id,
            input_root: job.input_root,
            output_root: root.output.clone(),
            endpoint: job.endpoint,
            phase: Phase::Prepared,
            items,
            requests: None,
            uploaded: None,
            batch: None,
        };
        let mut store = Self {
            root,
            path,
            directory,
            lock,
            state,
            limits,
            poisoned: false,
            #[cfg(test)]
            fault: None,
        };
        store.replace_state(store.state.clone())?;
        global.validate()?;
        Ok(store)
    }

    pub(crate) fn open_by_batch(
        output: &Path,
        batch_id: &str,
        allow_symlinks: bool,
        limits: Limits,
    ) -> Result<Self> {
        let limits = limits.check()?;
        if !opaque(batch_id, "batch") {
            return Err(Error::Invalid("Provider batch ID is invalid"));
        }
        let root = Root::open(output, allow_symlinks, false)?;
        // Index is merely a locator. State below must independently match its job ID.
        let global = root.lock()?;
        let index_path = root.path.join("by-id").join(format!("{batch_id}.json"));
        let index = if index_path.try_exists()? {
            read_json::<Index>(&index_path, limits.state_bytes)?
        } else {
            let jobs = scan(&root, limits)?;
            let mut matching = jobs
                .iter()
                .filter(|job| job.batch.as_ref().is_some_and(|batch| batch.id == batch_id));
            let job = matching.next().ok_or(Error::NotFound)?;
            if matching.next().is_some() {
                return Err(Error::Conflict);
            }
            let index = Index {
                version: 1,
                batch_id: batch_id.into(),
                job_id: job.id.clone(),
            };
            write_new_json(&index_path, &index, limits.state_bytes)?;
            index
        };
        global.validate()?;
        drop(global);
        if index.version != 1 || index.batch_id != batch_id || !uuid(&index.job_id) {
            return Err(Error::Conflict);
        }
        let store = Self::open_id(root, &index.job_id, limits)?;
        if store
            .state
            .batch
            .as_ref()
            .is_none_or(|batch| batch.id != batch_id)
        {
            return Err(Error::Conflict);
        }
        Ok(store)
    }

    #[cfg(test)]
    #[cfg_attr(not(unix), allow(dead_code))] // Used by Unix-only tests.
    pub(crate) fn open_by_id(
        output: &Path,
        id: &str,
        allow_symlinks: bool,
        limits: Limits,
    ) -> Result<Self> {
        let limits = limits.check()?;
        Self::open_id(Root::open(output, allow_symlinks, false)?, id, limits)
    }
    fn open_id(root: Root, id: &str, limits: Limits) -> Result<Self> {
        if !uuid(id) {
            return Err(Error::Invalid("Provider batch local ID is invalid"));
        }
        let path = root.path.join(format!("attempt-{id}"));
        let directory = Directory::read(&path, true)?;
        let lock = HeldLock::open(&path.join("job.lock"))?;
        let state: Job = read_json(&path.join("state.json"), limits.state_bytes)?;
        validate_job(&state, &root.output, limits)?;
        if state.id != id {
            return Err(Error::Conflict);
        }
        let store = Self {
            root,
            path,
            directory,
            lock,
            state,
            limits,
            poisoned: false,
            #[cfg(test)]
            fault: None,
        };
        store.validate()?;
        store.used_bytes()?;
        // State/blob tampering must be detected before a caller can send another request.
        for item in &store.state.items {
            store.read_blob(&item.plan, limits.blob_bytes)?;
            if let Some(result) = &item.result {
                store.read_blob(&result.blob, limits.blob_bytes)?;
            }
        }
        if let Some(requests) = &store.state.requests {
            store.verify_blob(requests, limits.request_bytes)?;
        }
        Ok(store)
    }

    pub(crate) fn pending(
        output: &Path,
        allow_symlinks: bool,
        limits: Limits,
    ) -> Result<Vec<PendingSummary>> {
        let limits = limits.check()?;
        // Missing stores do not create directories or locks during a resume inspection.
        let output = physical_output(output, allow_symlinks)?;
        if !output.join(".markitai/provider-batches").try_exists()? {
            return Ok(Vec::new());
        }
        let root = Root::open(&output, allow_symlinks, false)?;
        Ok(scan(&root, limits)?
            .into_iter()
            .filter(|state| !matches!(state.phase, Phase::Collected | Phase::Rejected))
            .map(|state| PendingSummary {
                id: state.id,
                batch_id: state.batch.map(|batch| batch.id),
                input_root: state.input_root,
                phase: state.phase,
            })
            .collect())
    }
    pub(crate) fn state(&self) -> &Job {
        &self.state
    }

    /// An unchanged typed phase is the only authority to choose the next network operation.
    /// This deliberately provides no transition from uncertain creation back to create.
    pub(crate) fn resume_step(&self) -> Result<ResumeStep> {
        self.validate()?;
        validate_job(&self.state, &self.root.output, self.limits)?;
        for item in &self.state.items {
            self.verify_blob(&item.plan, self.limits.blob_bytes)?;
        }
        if let Some(requests) = &self.state.requests {
            self.verify_blob(requests, self.limits.request_bytes)?;
        }
        match self.state.phase {
            Phase::Prepared if self.state.requests.is_none() => Ok(ResumeStep::PrepareRequests),
            Phase::Prepared => Ok(ResumeStep::Upload),
            Phase::Uploaded => Ok(ResumeStep::Create),
            Phase::Creating | Phase::CreateUncertain => Ok(ResumeStep::Reconcile),
            Phase::Submitted => Ok(ResumeStep::Collect),
            Phase::Collected => Ok(ResumeStep::Done),
            Phase::Rejected => Err(Error::Invalid(
                "A rejected provider submission cannot be resumed; retain its evidence and start a new run",
            )),
        }
    }

    /// Called with transport evidence from a strict manual lookup or complete reconciliation.
    /// Provider-returned IDs alone never let a caller attach an unrelated paid job.
    pub(crate) fn bind_reconciled(&mut self, identity: RemoteIdentity) -> Result<()> {
        if self.resume_step()? != ResumeStep::Reconcile {
            return Err(Error::Conflict);
        }
        let uploaded = self.state.uploaded.as_ref().ok_or(Error::Conflict)?;
        if identity.api_base() != self.state.endpoint.api_base
            || !identity.matches(uploaded, &self.state.id)
        {
            return Err(Error::Conflict);
        }
        self.bind_batch(identity.into_batch())
    }
    pub(crate) fn directory(&self) -> &Path {
        &self.path
    }
    pub(crate) fn request_path(&self) -> Result<PathBuf> {
        self.validate()?;
        let blob = self.state.requests.as_ref().ok_or(Error::Invalid(
            "Provider batch requests have not been saved",
        ))?;
        self.verify_blob(blob, self.limits.request_bytes)?;
        Ok(self.path.join(&blob.path))
    }
    pub(crate) fn read_plan(&self, id: &str) -> Result<Value> {
        let item = self.item(id)?;
        serde_json::from_slice(&self.read_blob(&item.plan, self.limits.blob_bytes)?)
            .map_err(|_| Error::Conflict)
    }
    pub(crate) fn read_result(&self, id: &str) -> Result<Option<Vec<u8>>> {
        self.item(id)?
            .result
            .as_ref()
            .map(|record| self.read_blob(&record.blob, self.limits.blob_bytes))
            .transpose()
    }

    pub(crate) fn save_requests(&mut self, reader: &mut impl Read) -> Result<Blob> {
        self.validate()?;
        if self.state.phase != Phase::Prepared {
            return Err(Error::Conflict);
        }
        let blob = self.write_blob(
            Path::new("requests.jsonl"),
            reader,
            self.limits.request_bytes,
        )?;
        if self
            .state
            .requests
            .as_ref()
            .is_some_and(|prior| prior != &blob)
        {
            return Err(Error::Conflict);
        }
        let mut state = self.state.clone();
        state.requests = Some(blob.clone());
        self.replace_state(state)?;
        Ok(blob)
    }
    pub(crate) fn save_uploaded(&mut self, uploaded: UploadedInput) -> Result<()> {
        self.validate()?;
        let request = self.state.requests.as_ref().ok_or(Error::Conflict)?;
        if !opaque(&uploaded.file_id, "file")
            || uploaded.sha256 != request.sha256
            || uploaded.bytes != request.bytes
            || uploaded.model != self.state.endpoint.model
            || uploaded.custom_ids
                != self
                    .state
                    .items
                    .iter()
                    .map(|item| item.custom_id.clone())
                    .collect::<Vec<_>>()
        {
            return Err(Error::Conflict);
        }
        if self.state.phase == Phase::Uploaded {
            return if same_json(
                self.state.uploaded.as_ref().ok_or(Error::Conflict)?,
                &uploaded,
            )? {
                Ok(())
            } else {
                Err(Error::Conflict)
            };
        }
        if self.state.phase != Phase::Prepared {
            return Err(Error::Conflict);
        }
        let mut state = self.state.clone();
        state.uploaded = Some(uploaded);
        state.phase = Phase::Uploaded;
        self.replace_state(state)
    }
    /// This transition is durable BEFORE invoking the non-idempotent create POST.
    pub(crate) fn mark_creating(&mut self) -> Result<String> {
        self.validate()?;
        if self.state.phase != Phase::Uploaded {
            return Err(Error::Conflict);
        }
        let mut state = self.state.clone();
        state.phase = Phase::Creating;
        self.replace_state(state)?;
        Ok(self.state.id.clone())
    }
    pub(crate) fn mark_uncertain(&mut self) -> Result<()> {
        self.validate()?;
        if self.state.phase == Phase::CreateUncertain {
            return Ok(());
        }
        if self.state.phase != Phase::Creating {
            return Err(Error::Conflict);
        }
        let mut state = self.state.clone();
        state.phase = Phase::CreateUncertain;
        self.replace_state(state)
    }
    pub(crate) fn mark_rejected(&mut self) -> Result<()> {
        self.validate()?;
        if self.state.phase == Phase::Rejected {
            return Ok(());
        }
        if self.state.phase != Phase::Creating {
            return Err(Error::Conflict);
        }
        let mut state = self.state.clone();
        state.phase = Phase::Rejected;
        self.replace_state(state)
    }
    pub(crate) fn bind_batch(&mut self, batch: Batch) -> Result<()> {
        self.validate()?;
        if !matches!(
            self.state.phase,
            Phase::Creating | Phase::CreateUncertain | Phase::Submitted
        ) || !opaque(&batch.id, "batch")
            || self
                .state
                .uploaded
                .as_ref()
                .is_none_or(|uploaded| uploaded.file_id != batch.input_file_id)
            || self.state.batch.as_ref().is_some_and(|prior| {
                prior.id != batch.id || (prior.status.terminal() && prior.status != batch.status)
            })
        {
            return Err(Error::Conflict);
        }
        let global = self.root.lock()?;
        let index_path = self
            .root
            .path
            .join("by-id")
            .join(format!("{}.json", batch.id));
        let index = Index {
            version: 1,
            batch_id: batch.id.clone(),
            job_id: self.state.id.clone(),
        };
        if index_path.try_exists()?
            && read_json::<Index>(&index_path, self.limits.state_bytes)? != index
        {
            return Err(Error::Conflict);
        }
        let mut state = self.state.clone();
        state.phase = Phase::Submitted;
        state.batch = Some(batch);
        self.replace_state(state)?;
        // State first: a crash before the locator permits safe repair with the local ID.
        if !index_path.try_exists()? {
            write_new_json(&index_path, &index, self.limits.state_bytes)?;
        }
        global.validate()?;
        Ok(())
    }

    pub(crate) fn record_result(
        &mut self,
        id: &str,
        raw: &[u8],
        usage: ConversionUsage,
    ) -> Result<RecordResult> {
        self.validate()?;
        if self.state.phase != Phase::Submitted || raw.len() > self.limits.blob_bytes {
            return Err(Error::Conflict);
        }
        validate_usage(&usage)?;
        let value: Value = serde_json::from_slice(raw)
            .map_err(|_| Error::Invalid("Provider result is not valid JSON"))?;
        if value.get("custom_id").and_then(Value::as_str) != Some(id) {
            return Err(Error::Conflict);
        }
        let index = self.index(id)?;
        let hash = hex(raw);
        if let Some(prior) = &self.state.items[index].result {
            if prior.blob.sha256 != hash || prior.blob.bytes != raw.len() as u64 {
                return Err(Error::Conflict);
            }
            self.read_blob(&prior.blob, self.limits.blob_bytes)?;
            return Ok(RecordResult {
                inserted: false,
                usage: prior.usage.clone(),
            });
        }
        let mut reader = raw;
        let blob = self.write_blob(
            Path::new(&format!("results/{index}.json")),
            &mut reader,
            self.limits.blob_bytes,
        )?;
        let mut state = self.state.clone();
        state.items[index].result = Some(RecordedResult {
            blob,
            usage: usage.clone(),
        });
        self.replace_state(state)?;
        Ok(RecordResult {
            inserted: true,
            usage,
        })
    }
    /// Called only after normal Claim publication/recovery has verified native authority.
    pub(crate) fn mark_finalized(&mut self, id: &str, published: Published) -> Result<()> {
        self.validate()?;
        if self.state.phase != Phase::Submitted {
            return Err(Error::Conflict);
        }
        let index = self.index(id)?;
        let item = &self.state.items[index];
        validate_published(&published, item)?;
        if let Some(prior) = &item.finalized {
            return if prior == &published {
                Ok(())
            } else {
                Err(Error::Conflict)
            };
        }
        let mut state = self.state.clone();
        state.items[index].finalized = Some(published);
        state.items[index].error = None;
        self.replace_state(state)
    }
    /// Persist failure independently of raw usage, without pretending output was published.
    pub(crate) fn mark_failed(&mut self, id: &str, message: &str) -> Result<()> {
        self.validate()?;
        if self.state.phase != Phase::Submitted
            || message.is_empty()
            || message.len() > 4096
            || message.contains('\0')
        {
            return Err(Error::Conflict);
        }
        let index = self.index(id)?;
        if self.state.items[index].finalized.is_some() {
            return Err(Error::Conflict);
        }
        let mut state = self.state.clone();
        state.items[index].error = Some(message.into());
        self.replace_state(state)
    }
    pub(crate) fn finish(&mut self) -> Result<()> {
        self.validate()?;
        if self.state.phase == Phase::Collected {
            return Ok(());
        }
        if self.state.phase != Phase::Submitted
            || self
                .state
                .batch
                .as_ref()
                .is_none_or(|batch| !batch.status.terminal())
            || self.state.items.iter().any(|item| item.finalized.is_none())
        {
            return Err(Error::Conflict);
        }
        let mut state = self.state.clone();
        state.phase = Phase::Collected;
        self.replace_state(state)
    }
    /// Totals are derived from immutable per-request observations, never incremented on replay.
    pub(crate) fn usage(&self) -> Result<ConversionUsage> {
        let mut usage = ConversionUsage::default();
        for record in self
            .state
            .items
            .iter()
            .filter_map(|item| item.result.as_ref())
        {
            merge_usage(&mut usage, &record.usage)?;
        }
        Ok(usage)
    }

    fn item(&self, id: &str) -> Result<&Item> {
        self.state.items.get(self.index(id)?).ok_or(Error::Conflict)
    }
    fn index(&self, id: &str) -> Result<usize> {
        self.state
            .items
            .iter()
            .position(|item| item.custom_id == id)
            .ok_or(Error::Invalid("Unknown provider batch item"))
    }
    fn validate(&self) -> Result<()> {
        if self.poisoned {
            return Err(Error::Durability);
        }
        self.root.validate()?;
        self.directory.validate()?;
        self.lock.validate()?;
        Ok(())
    }
    fn replace_state(&mut self, state: Job) -> Result<()> {
        self.validate()?;
        validate_job(&state, &self.root.output, self.limits)?;
        let bytes = encode(&state, self.limits.state_bytes)?;
        if self
            .used_bytes()?
            .checked_add(bytes.len() as u64)
            .is_none_or(|sum| sum > self.limits.total_bytes)
        {
            return Err(Error::Invalid("Provider batch storage limit reached"));
        }
        let mut temporary = private_temp(&self.path)?;
        temporary.write_all(&bytes)?;
        temporary.as_file().sync_all()?;
        #[cfg(test)]
        if self.fault == Some(Fault::BeforeStateReplace) {
            self.fault = None;
            return Err(Error::Io);
        }
        self.validate()?;
        temporary
            .persist(self.path.join("state.json"))
            .map_err(|_| Error::Io)?;
        self.state = state;
        #[cfg(test)]
        if self.fault == Some(Fault::AfterStateReplace) {
            self.fault = None;
            self.poisoned = true;
            return Err(Error::Durability);
        }
        if sync(&self.path).is_err() {
            self.poisoned = true;
            return Err(Error::Durability);
        }
        Ok(())
    }
    fn write_blob(&self, path: &Path, reader: &mut impl Read, limit: usize) -> Result<Blob> {
        self.validate()?;
        let remaining = self
            .limits
            .total_bytes
            .checked_sub(self.used_bytes()?)
            .ok_or(Error::Invalid("Provider batch storage limit reached"))?;
        write_blob(
            &self.path,
            path,
            reader,
            limit.min(usize::try_from(remaining).unwrap_or(usize::MAX)),
        )
    }
    fn read_blob(&self, blob: &Blob, limit: usize) -> Result<Vec<u8>> {
        self.validate()?;
        validate_blob(blob, limit)?;
        let path = self.path.join(&blob.path);
        Directory::read(path.parent().ok_or(Error::Conflict)?, true)?;
        let bytes = read_private(&path, limit)?;
        if bytes.len() as u64 != blob.bytes || hex(&bytes) != blob.sha256 {
            return Err(Error::Conflict);
        }
        Ok(bytes)
    }
    fn verify_blob(&self, blob: &Blob, limit: usize) -> Result<()> {
        self.validate()?;
        validate_blob(blob, limit)?;
        let path = self.path.join(&blob.path);
        Directory::read(path.parent().ok_or(Error::Conflict)?, true)?;
        let (bytes, hash) = hash_file(&path, limit)?;
        if bytes != blob.bytes || hash != blob.sha256 {
            return Err(Error::Conflict);
        }
        Ok(())
    }
    fn used_bytes(&self) -> Result<u64> {
        inventory(&self.path, self.limits)
    }
}

#[derive(Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Index {
    version: u8,
    batch_id: String,
    job_id: String,
}

impl Root {
    fn open(output: &Path, allow_symlinks: bool, create: bool) -> Result<Self> {
        if !cfg!(unix) {
            return Err(Error::Unsupported);
        }
        let physical = physical_output(output, allow_symlinks)?;
        if create {
            fs::create_dir_all(&physical)?;
        }
        let mut directories = vec![Directory::read(&physical, false)?];
        let metadata = physical.join(".markitai");
        if create {
            ensure_directory(&metadata, false)?;
        }
        directories.push(Directory::read(&metadata, false)?);
        let path = metadata.join("provider-batches");
        if create {
            ensure_directory(&path, true)?;
        }
        directories.push(Directory::read(&path, true)?);
        let index = path.join("by-id");
        if create {
            ensure_directory(&index, true)?;
        }
        directories.push(Directory::read(&index, true)?);
        markitai_core::output::check_path(output, allow_symlinks)
            .map_err(|_| Error::Invalid("Provider batch output path is not allowed"))?;
        if physical_output(output, allow_symlinks)? != physical {
            return Err(Error::Conflict);
        }
        Ok(Self {
            output: physical,
            path,
            directories,
        })
    }
    fn validate(&self) -> Result<()> {
        for directory in &self.directories {
            directory.validate()?;
        }
        Ok(())
    }
    fn lock(&self) -> Result<HeldLock> {
        self.validate()?;
        HeldLock::open(&self.path.join("index.lock"))
    }
}
impl Directory {
    fn read(path: &Path, private: bool) -> Result<Self> {
        let metadata = fs::symlink_metadata(path)?;
        if !metadata.is_dir() || metadata.file_type().is_symlink() {
            return Err(Error::Invalid(
                "Provider batch directory is not a regular directory",
            ));
        }
        check_permissions(&metadata, private)?;
        Ok(Self {
            path: path.to_owned(),
            identity: identity(&metadata)?,
            private,
        })
    }
    fn validate(&self) -> Result<()> {
        let now = Self::read(&self.path, self.private)?;
        if now.identity != self.identity {
            return Err(Error::Conflict);
        }
        Ok(())
    }
}
impl Drop for HeldLock {
    fn drop(&mut self) {
        // Job, index and submission ownership ends here even when a concurrently
        // forked child temporarily retains this open file description.
        let _ = self.file.unlock();
    }
}

impl HeldLock {
    fn open(path: &Path) -> Result<Self> {
        if let Ok(metadata) = fs::symlink_metadata(path) {
            regular(&metadata)?;
        }
        let mut options = OpenOptions::new();
        options.read(true).write(true).create(true).truncate(false);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options
                .mode(0o600)
                .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
        }
        let file = options.open(path)?;
        let metadata = file.metadata()?;
        regular(&metadata)?;
        if metadata.len() != 0 {
            return Err(Error::Invalid("Provider batch lock is not empty"));
        }
        let identity = identity(&metadata)?;
        file.try_lock().map_err(|error| match error {
            TryLockError::WouldBlock => Error::Busy,
            TryLockError::Error(_) => Error::Io,
        })?;
        let held = Self {
            path: path.into(),
            file,
            identity,
        };
        held.validate()?;
        held.file.sync_all()?;
        sync(path.parent().ok_or(Error::Io)?)?;
        Ok(held)
    }
    fn validate(&self) -> Result<()> {
        let opened = self.file.metadata()?;
        let current = fs::symlink_metadata(&self.path)?;
        regular(&opened)?;
        regular(&current)?;
        if opened.len() != 0
            || current.len() != 0
            || identity(&opened)? != self.identity
            || identity(&current)? != self.identity
        {
            return Err(Error::Conflict);
        }
        Ok(())
    }
}
fn identity(metadata: &fs::Metadata) -> Result<Identity> {
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
        Err(Error::Unsupported)
    }
}
fn check_permissions(metadata: &fs::Metadata, private: bool) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};
        if private
            && (metadata.permissions().mode() & 0o077 != 0
                || metadata.uid() != unsafe { libc::geteuid() })
        {
            return Err(Error::Invalid(
                "Provider batch evidence must be private to the current user",
            ));
        }
    }
    #[cfg(not(unix))]
    {
        let _ = (metadata, private);
        Err(Error::Unsupported)
    }
    #[cfg(unix)]
    {
        Ok(())
    }
}
fn regular(metadata: &fs::Metadata) -> Result<()> {
    if !metadata.is_file() || metadata.file_type().is_symlink() {
        return Err(Error::Invalid(
            "Provider batch evidence must be a regular file",
        ));
    }
    check_permissions(metadata, true)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if metadata.nlink() != 1 {
            return Err(Error::Invalid(
                "Provider batch evidence must not have multiple hard links",
            ));
        }
    }
    Ok(())
}
fn physical_output(output: &Path, allow_symlinks: bool) -> Result<PathBuf> {
    markitai_core::output::check_path(output, allow_symlinks)
        .map_err(|_| Error::Invalid("Provider batch output path is not allowed"))?;
    crate::report_store::resolve_path(output).map_err(Into::into)
}
fn new_directory(path: &Path) -> Result<()> {
    #[cfg_attr(not(unix), allow(unused_mut))] // Only Unix sets a mode.
    let mut builder = fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(path)?;
    sync(path.parent().ok_or(Error::Io)?)?;
    Ok(())
}
fn ensure_directory(path: &Path, private: bool) -> Result<()> {
    match fs::symlink_metadata(path) {
        Ok(_) => {
            Directory::read(path, private)?;
            Ok(())
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => new_directory(path),
        Err(_) => Err(Error::Io),
    }
}
fn sync(path: &Path) -> Result<()> {
    File::open(path)?.sync_all().map_err(Into::into)
}
fn open_private(path: &Path) -> Result<File> {
    let before = fs::symlink_metadata(path)?;
    regular(&before)?;
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let file = options.open(path)?;
    let after = file.metadata()?;
    regular(&after)?;
    if identity(&before)? != identity(&after)? {
        return Err(Error::Conflict);
    }
    Ok(file)
}
fn read_private(path: &Path, limit: usize) -> Result<Vec<u8>> {
    let file = open_private(path)?;
    let before = file.metadata()?;
    if before.len() > limit as u64 {
        return Err(Error::Invalid(
            "Provider batch evidence exceeds its byte limit",
        ));
    }
    let mut bytes = Vec::new();
    (&file).take(limit as u64 + 1).read_to_end(&mut bytes)?;
    let after = file.metadata()?;
    if bytes.len() > limit
        || before.len() != after.len()
        || before.modified().ok() != after.modified().ok()
        || identity(&fs::symlink_metadata(path)?)? != identity(&before)?
    {
        return Err(Error::Conflict);
    }
    Ok(bytes)
}
fn hash_file(path: &Path, limit: usize) -> Result<(u64, String)> {
    let mut file = open_private(path)?;
    let before = file.metadata()?;
    if before.len() > limit as u64 {
        return Err(Error::Invalid(
            "Provider batch evidence exceeds its byte limit",
        ));
    }
    let mut hash = Sha256::new();
    let mut bytes = 0u64;
    let mut buffer = [0; 65536];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        bytes = bytes.checked_add(read as u64).ok_or(Error::Conflict)?;
        if bytes > limit as u64 {
            return Err(Error::Invalid(
                "Provider batch evidence exceeds its byte limit",
            ));
        }
        hash.update(&buffer[..read]);
    }
    let after = file.metadata()?;
    if before.len() != bytes
        || after.len() != bytes
        || before.modified().ok() != after.modified().ok()
        || identity(&fs::symlink_metadata(path)?)? != identity(&before)?
    {
        return Err(Error::Conflict);
    }
    Ok((bytes, markitai_core::hex(hash.finalize())))
}
fn read_json<T: for<'de> Deserialize<'de>>(path: &Path, limit: usize) -> Result<T> {
    serde_json::from_slice(&read_private(path, limit)?)
        .map_err(|_| Error::Invalid("Provider batch state is invalid JSON"))
}
fn private_temp(parent: &Path) -> Result<NamedTempFile> {
    Directory::read(parent, true)?;
    let temporary = tempfile::Builder::new()
        .prefix(".pending-")
        .tempfile_in(parent)?;
    regular(&temporary.as_file().metadata()?)?;
    Ok(temporary)
}
struct Bounded<W> {
    writer: W,
    remaining: usize,
    bytes: u64,
    hash: Sha256,
}
impl<W: Write> Write for Bounded<W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > self.remaining {
            return Err(io::Error::other("provider batch byte limit"));
        }
        let written = self.writer.write(bytes)?;
        self.remaining -= written;
        self.bytes += written as u64;
        self.hash.update(&bytes[..written]);
        Ok(written)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }
}
fn encode(value: &impl Serialize, limit: usize) -> Result<Vec<u8>> {
    let mut output = Bounded {
        writer: Vec::new(),
        remaining: limit,
        bytes: 0,
        hash: Sha256::new(),
    };
    serde_json::to_writer(&mut output, value).map_err(|_| {
        Error::Invalid("Provider batch state exceeds its limit or cannot be serialized")
    })?;
    Ok(output.writer)
}
fn write_new_json(path: &Path, value: &impl Serialize, limit: usize) -> Result<()> {
    let bytes = encode(value, limit)?;
    let parent = path.parent().ok_or(Error::Io)?;
    let mut temporary = private_temp(parent)?;
    temporary.write_all(&bytes)?;
    temporary.as_file().sync_all()?;
    temporary
        .persist_noclobber(path)
        .map_err(|_| Error::Conflict)?;
    sync(parent)
}
fn write_json_blob(directory: &Path, relative: &Path, value: &Value, limit: usize) -> Result<Blob> {
    let bytes = encode(value, limit)?;
    write_blob(directory, relative, &mut bytes.as_slice(), limit)
}
fn write_blob(
    directory: &Path,
    relative: &Path,
    reader: &mut impl Read,
    limit: usize,
) -> Result<Blob> {
    relative_path(relative)?;
    let destination = directory.join(relative);
    let parent = destination.parent().ok_or(Error::Io)?;
    let temporary = private_temp(parent)?;
    let mut output = Bounded {
        writer: temporary,
        remaining: limit,
        bytes: 0,
        hash: Sha256::new(),
    };
    let mut buffer = [0; 65536];
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        output
            .write_all(&buffer[..count])
            .map_err(|_| Error::Invalid("Provider batch blob exceeds its byte limit"))?;
    }
    output.flush()?;
    output.writer.as_file().sync_all()?;
    let blob = Blob {
        path: relative.into(),
        bytes: output.bytes,
        sha256: markitai_core::hex(output.hash.finalize()),
    };
    if destination.try_exists()? {
        let (bytes, hash) = hash_file(&destination, limit)?;
        if bytes != blob.bytes || hash != blob.sha256 {
            return Err(Error::Conflict);
        }
    } else {
        output
            .writer
            .persist_noclobber(&destination)
            .map_err(|_| Error::Conflict)?;
        sync(parent)?;
    }
    Ok(blob)
}
fn inventory(directory: &Path, limits: Limits) -> Result<u64> {
    Directory::read(directory, true)?;
    let mut paths = vec![directory.to_owned()];
    let mut bytes = 0u64;
    let mut count = 0usize;
    while let Some(parent) = paths.pop() {
        for entry in fs::read_dir(parent)? {
            let entry = entry?;
            count += 1;
            if count > limits.items.saturating_mul(2).saturating_add(32) {
                return Err(Error::Invalid("Provider batch evidence file limit reached"));
            }
            let path = entry.path();
            let metadata = fs::symlink_metadata(&path)?;
            if metadata.is_dir() && !metadata.file_type().is_symlink() {
                if path.parent() != Some(directory)
                    || !matches!(entry.file_name().to_str(), Some("plans" | "results"))
                {
                    return Err(Error::Invalid(
                        "Unexpected provider batch evidence directory",
                    ));
                }
                Directory::read(&path, true)?;
                paths.push(path);
            } else {
                regular(&metadata)?;
                bytes = bytes.checked_add(metadata.len()).ok_or(Error::Conflict)?;
                if bytes > limits.total_bytes {
                    return Err(Error::Invalid("Provider batch storage limit reached"));
                }
            }
        }
    }
    Ok(bytes)
}
fn scan(root: &Root, limits: Limits) -> Result<Vec<Job>> {
    root.validate()?;
    let mut paths = Vec::new();
    for entry in fs::read_dir(&root.path)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_str().ok_or(Error::Conflict)?;
        if name == "by-id" {
            continue;
        }
        if matches!(name, "index.lock" | "submission.lock") {
            read_private(&entry.path(), 0)?;
            continue;
        }
        let id = name
            .strip_prefix("attempt-")
            .filter(|id| uuid(id))
            .ok_or(Error::Invalid("Unexpected provider batch store entry"))?;
        if paths.len() >= limits.jobs {
            return Err(Error::Invalid("Provider batch job limit reached"));
        }
        Directory::read(&entry.path(), true)?;
        paths.push((id.to_owned(), entry.path()));
    }
    paths.sort_by(|left, right| left.0.cmp(&right.0));
    let mut jobs = Vec::new();
    for (id, path) in paths {
        // Only create() writes before this file exists, and create() performs no HTTP.
        if !path.join("state.json").try_exists()? {
            continue;
        }
        let job: Job = read_json(&path.join("state.json"), limits.state_bytes)?;
        validate_job(&job, &root.output, limits)?;
        if job.id != id {
            return Err(Error::Conflict);
        }
        jobs.push(job);
    }
    Ok(jobs)
}

fn uuid(value: &str) -> bool {
    uuid::Uuid::parse_str(value).is_ok_and(|parsed| parsed.hyphenated().to_string() == value)
}
fn hex(bytes: &[u8]) -> String {
    markitai_core::hex(Sha256::digest(bytes))
}
fn hash(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn custom_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 64
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-'))
}
fn opaque(value: &str, prefix: &str) -> bool {
    value.len() <= 256
        && value.strip_prefix(prefix).is_some_and(|tail| {
            tail.len() > 1
                && matches!(tail.as_bytes()[0], b'_' | b'-')
                && tail
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-'))
        })
}
fn relative_path(path: &Path) -> Result<()> {
    if path.as_os_str().is_empty()
        || path
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
        || path.as_os_str().len() > 4096
    {
        return Err(Error::Invalid("Provider batch relative path is invalid"));
    }
    Ok(())
}
fn absolute_path(path: &Path) -> bool {
    path.is_absolute()
        && !path
            .components()
            .any(|part| matches!(part, Component::CurDir | Component::ParentDir))
}
fn validate_endpoint(endpoint: &Endpoint) -> Result<()> {
    let url = url::Url::parse(&endpoint.api_base)
        .map_err(|_| Error::Invalid("Provider batch endpoint is invalid"))?;
    if endpoint.provider != "openai"
        || endpoint.api_base.len() > 8192
        || endpoint.api_base.trim() != endpoint.api_base
        || endpoint.api_base.chars().any(char::is_control)
        || !matches!(url.scheme(), "http" | "https")
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || endpoint.model.trim().is_empty()
        || endpoint.model.len() > 256
        || endpoint.model.chars().any(char::is_control)
    {
        return Err(Error::Invalid(
            "Provider batch endpoint identity is invalid",
        ));
    }
    Ok(())
}
fn validate_identity(
    id: &str,
    roots: (&Path, &Path),
    key: &str,
    base: &Path,
    enhanced: &Path,
    base_sha: &str,
    owner: &Owner,
) -> Result<()> {
    let (input, output) = roots;
    relative_path(base)?;
    relative_path(enhanced)?;
    if base == enhanced
        || base.parent() != enhanced.parent()
        || !hash(base_sha)
        || key.is_empty()
        || key.len() > 8192
        || key.contains('\0')
        || owner.generation != id
        || owner.mode != "directory"
        || !matches!(owner.kind.as_str(), "file" | "url")
        || owner.key != key
        || owner.input != input
        || owner.output != output
        || !absolute_path(input)
        || !absolute_path(output)
    {
        return Err(Error::Invalid("Provider batch item ownership is invalid"));
    }
    if owner.kind == "file" {
        relative_path(Path::new(key))?;
    }
    let name = base
        .file_name()
        .and_then(|name| name.to_str())
        .and_then(|name| name.strip_suffix(".md"))
        .ok_or(Error::Invalid("Provider batch base must be Markdown"))?;
    if enhanced.file_name().and_then(|name| name.to_str())
        != Some(format!("{name}.llm.md").as_str())
    {
        return Err(Error::Invalid(
            "Provider batch enhanced path is not the base's output family",
        ));
    }
    Ok(())
}
fn validate_new(job: &NewJob, output: &Path, limits: Limits) -> Result<()> {
    if !uuid(&job.id) || job.items.is_empty() || job.items.len() > limits.items {
        return Err(Error::Invalid(
            "Provider batch job identity or size is invalid",
        ));
    }
    validate_endpoint(&job.endpoint)?;
    let mut ids = HashSet::new();
    let mut members = HashSet::new();
    for item in &job.items {
        if !custom_id(&item.custom_id)
            || !ids.insert(&item.custom_id)
            || item.source.len() > 8192
            || item.source.contains('\0')
            || !members.insert(&item.base)
            || !members.insert(&item.enhanced)
        {
            return Err(Error::Invalid(
                "Provider batch items are invalid or overlap",
            ));
        }
        validate_identity(
            &job.id,
            (&job.input_root, output),
            &item.key,
            &item.base,
            &item.enhanced,
            &item.base_sha256,
            &item.owner,
        )?;
    }
    Ok(())
}
fn validate_blob(blob: &Blob, limit: usize) -> Result<()> {
    relative_path(&blob.path)?;
    if !hash(&blob.sha256) || blob.bytes > limit as u64 {
        return Err(Error::Invalid("Provider batch blob identity is invalid"));
    }
    Ok(())
}
fn validate_published(published: &Published, item: &Item) -> Result<()> {
    if (published.path != item.enhanced && published.path != item.base)
        || !hash(&published.sha256)
        || !hash(&published.receipt_sha256)
    {
        return Err(Error::Invalid(
            "Provider batch publication evidence is invalid",
        ));
    }
    Ok(())
}
fn validate_job(job: &Job, output: &Path, limits: Limits) -> Result<()> {
    if job.version != 1
        || !uuid(&job.id)
        || job.output_root != output
        || !absolute_path(&job.input_root)
        || job.items.is_empty()
        || job.items.len() > limits.items
    {
        return Err(Error::Invalid("Provider batch state scope is invalid"));
    }
    validate_endpoint(&job.endpoint)?;
    let mut ids = HashSet::new();
    let mut members = HashSet::new();
    let mut total = 0u64;
    for (index, item) in job.items.iter().enumerate() {
        if !custom_id(&item.custom_id)
            || !ids.insert(&item.custom_id)
            || item.source.len() > 8192
            || item.source.contains('\0')
            || !members.insert(&item.base)
            || !members.insert(&item.enhanced)
        {
            return Err(Error::Invalid("Provider batch state items are invalid"));
        }
        validate_identity(
            &job.id,
            (&job.input_root, &job.output_root),
            &item.key,
            &item.base,
            &item.enhanced,
            &item.base_sha256,
            &item.owner,
        )?;
        validate_blob(&item.plan, limits.blob_bytes)?;
        if item.plan.path != Path::new(&format!("plans/{index}.json")) {
            return Err(Error::Conflict);
        }
        total = total.checked_add(item.plan.bytes).ok_or(Error::Conflict)?;
        if let Some(result) = &item.result {
            validate_blob(&result.blob, limits.blob_bytes)?;
            validate_usage(&result.usage)?;
            if result.blob.path != Path::new(&format!("results/{index}.json")) {
                return Err(Error::Conflict);
            }
            total = total
                .checked_add(result.blob.bytes)
                .ok_or(Error::Conflict)?;
        }
        if let Some(published) = &item.finalized {
            validate_published(published, item)?;
        }
        if item
            .error
            .as_ref()
            .is_some_and(|error| error.is_empty() || error.len() > 4096 || error.contains('\0'))
            || (item.finalized.is_some() && item.error.is_some())
        {
            return Err(Error::Conflict);
        }
    }
    if let Some(request) = &job.requests {
        validate_blob(request, limits.request_bytes)?;
        if request.path != Path::new("requests.jsonl") {
            return Err(Error::Conflict);
        }
        total = total.checked_add(request.bytes).ok_or(Error::Conflict)?;
    }
    if total > limits.total_bytes {
        return Err(Error::Invalid(
            "Provider batch state storage budget exceeded",
        ));
    }
    if let Some(uploaded) = &job.uploaded {
        let request = job.requests.as_ref().ok_or(Error::Conflict)?;
        if !opaque(&uploaded.file_id, "file")
            || uploaded.bytes != request.bytes
            || uploaded.sha256 != request.sha256
            || uploaded.model != job.endpoint.model
            || uploaded.custom_ids
                != job
                    .items
                    .iter()
                    .map(|item| item.custom_id.clone())
                    .collect::<Vec<_>>()
        {
            return Err(Error::Conflict);
        }
    }
    if let Some(batch) = &job.batch
        && (!opaque(&batch.id, "batch")
            || job
                .uploaded
                .as_ref()
                .is_none_or(|upload| upload.file_id != batch.input_file_id)
            || batch
                .output_file_id
                .as_ref()
                .is_some_and(|id| !opaque(id, "file"))
            || batch
                .error_file_id
                .as_ref()
                .is_some_and(|id| !opaque(id, "file")))
    {
        return Err(Error::Conflict);
    }

    match job.phase {
        Phase::Prepared if job.uploaded.is_none() && job.batch.is_none() => (),
        Phase::Uploaded | Phase::Creating | Phase::CreateUncertain | Phase::Rejected
            if job.uploaded.is_some() && job.batch.is_none() => {}
        Phase::Submitted if job.batch.is_some() => (),
        Phase::Collected
            if job
                .batch
                .as_ref()
                .is_some_and(|batch| batch.status.terminal())
                && job.items.iter().all(|item| item.finalized.is_some()) => {}
        _ => return Err(Error::Conflict),
    }
    if !matches!(job.phase, Phase::Submitted | Phase::Collected)
        && job
            .items
            .iter()
            .any(|item| item.result.is_some() || item.finalized.is_some() || item.error.is_some())
    {
        return Err(Error::Conflict);
    }
    Ok(())
}
fn same_json(left: &impl Serialize, right: &impl Serialize) -> Result<bool> {
    Ok(serde_json::to_value(left).map_err(|_| Error::Conflict)?
        == serde_json::to_value(right).map_err(|_| Error::Conflict)?)
}
fn validate_usage(usage: &ConversionUsage) -> Result<()> {
    if !usage.cost_usd.is_finite() || usage.cost_usd < 0.0 {
        return Err(Error::Invalid("Provider batch usage is invalid"));
    }
    if crate::diagnostics::observed(usage) {
        crate::diagnostics::AttemptDiagnostics::completed(
            crate::diagnostics::Operation::Enhance,
            usage.clone(),
        )
        .ok_or(Error::Conflict)?
        .validate()
        .map_err(|_| Error::Invalid("Provider batch usage is inconsistent"))?;
    } else if usage.cost_usd != 0.0 {
        return Err(Error::Invalid(
            "Provider batch cost lacks an observed request",
        ));
    }
    Ok(())
}
fn merge_usage(total: &mut ConversionUsage, value: &ConversionUsage) -> Result<()> {
    total.requests = total
        .requests
        .checked_add(value.requests)
        .ok_or(Error::Conflict)?;
    total.input_tokens = total
        .input_tokens
        .checked_add(value.input_tokens)
        .ok_or(Error::Conflict)?;
    total.output_tokens = total
        .output_tokens
        .checked_add(value.output_tokens)
        .ok_or(Error::Conflict)?;
    total.cost_usd += value.cost_usd;
    if !total.cost_usd.is_finite() {
        return Err(Error::Conflict);
    }
    crate::pricing::merge_models(&mut total.by_model, &value.by_model)
        .map_err(|_| Error::Conflict)?;
    Ok(())
}

#[cfg(test)]
#[path = "store_tests.rs"]
mod tests;

#[cfg(all(test, target_os = "linux"))]
use crate::file_lock_test as fork_lock_test;

#[cfg(all(test, target_os = "linux"))]
mod inherited_lock_tests {
    use super::*;
    #[test]
    fn batch_lock_kinds_release_before_inherited_child_exits() {
        let root = tempfile::tempdir().unwrap();
        for name in ["job.lock", "index.lock", "submission.lock"] {
            let path = root.path().join(name);
            let held = HeldLock::open(&path).unwrap();
            assert!(matches!(HeldLock::open(&path), Err(Error::Busy)));
            let mut child = fork_lock_test::InheritedChild::start();
            drop(held);
            let reopened = HeldLock::open(&path);
            child.finish();
            assert!(reopened.is_ok(), "batch lock outlived its owner: {name}");
        }
    }
    #[test]
    fn validation_failure_still_releases_the_owned_batch_inode() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("job.lock");
        let held = HeldLock::open(&path).unwrap();
        let mut child = fork_lock_test::InheritedChild::start();
        let moved = root.path().join("previous.lock");
        fs::rename(&path, &moved).unwrap();
        assert!(held.validate().is_err());
        drop(held);
        let recovered = HeldLock::open(&moved);
        child.finish();
        assert!(
            recovered.is_ok(),
            "validation failure pinned the original inode"
        );
    }
}
