use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt;

pub const ENDPOINT: &str = "/v1/chat/completions";
/// The identity Anthropic Message Batches give their remote jobs.
pub const MESSAGES_ENDPOINT: &str = "/v1/messages/batches";
/// Anthropic batches carry their requests inline in the create call, so there
/// is no uploaded file; this stands for that input wherever state names one.
pub const INLINE_INPUT: &str = "file-inline";

/// The Batch API a client speaks, chosen by the configured provider.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Api {
    /// Upload a JSONL file, then create a batch that names it.
    OpenAi,
    /// Message Batches: requests travel inline; results stream back by ID.
    Anthropic,
}
impl Api {
    pub fn for_provider(provider: &str) -> Option<Self> {
        match provider {
            "openai" => Some(Self::OpenAi),
            "anthropic" => Some(Self::Anthropic),
            _ => None,
        }
    }
}

/// The custom ID of the request at `index` of the submission `nonce`.
/// Anthropic's carry the nonce: a Message Batch keeps no metadata, so the IDs
/// in its results are what proves an uncertain creation later.
pub fn request_id(api: Api, nonce: &str, index: usize) -> String {
    match api {
        Api::OpenAi => format!("doc-{index}"),
        Api::Anthropic => format!("{}{index}", nonce_prefix(nonce)),
    }
}

/// The custom-ID prefix of an Anthropic submission: its nonce without hyphens.
pub(super) fn nonce_prefix(nonce: &str) -> String {
    format!("{}-", nonce.replace('-', ""))
}

/// A provider batch ID: OpenAI's `batch_…` or Anthropic's `msgbatch_…`.
pub fn batch_id(value: &str) -> bool {
    identifier(value, "batch") || identifier(value, "msgbatch")
}

/// Byte limits apply before JSON parsing, including decompressed response bytes.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub upload_bytes: usize,
    pub result_bytes: usize,
    pub line_bytes: usize,
    pub control_bytes: usize,
    pub requests: usize,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            upload_bytes: 200_000_000,
            result_bytes: 256 * 1024 * 1024,
            line_bytes: 8 * 1024 * 1024,
            control_bytes: 1024 * 1024,
            requests: 50_000,
        }
    }
}
impl Limits {
    pub(super) fn validate(self) -> Result<Self, Error> {
        let maximum = Self::default();
        if self.upload_bytes == 0
            || self.upload_bytes > maximum.upload_bytes
            || self.result_bytes == 0
            || self.result_bytes > maximum.result_bytes
            || self.line_bytes == 0
            || self.line_bytes > maximum.line_bytes
            || self.control_bytes == 0
            || self.control_bytes > maximum.control_bytes
            || self.requests == 0
            || self.requests > maximum.requests
        {
            return Err(Error::Invalid("Batch limits are invalid"));
        }
        Ok(self)
    }
}

/// No variant holds an endpoint, credential, provider message or document body.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Invalid(&'static str),
    #[error("{0}")]
    Limit(&'static str),
    #[error("Batch input could not be read or staged")]
    Input,
    #[error("Batch API transport failed")]
    Transport,
    #[error("Batch API rejected the request (HTTP {0})")]
    Http(u16),
    #[error("Batch API returned an invalid response")]
    Protocol,
    #[error(
        "Batch submission outcome is unknown; retain the input file ID and submission nonce, and do not automatically submit again"
    )]
    CreateUncertain,
    #[error("Batch is still processing; results are not ready")]
    Pending,
}

/// Contains the identity of exactly the immutable bytes uploaded, never auth.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UploadedInput {
    pub file_id: String,
    pub model: String,
    pub custom_ids: Vec<String>,
    pub bytes: u64,
    pub sha256: String,
}
impl fmt::Debug for UploadedInput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UploadedInput")
            .field("requests", &self.custom_ids.len())
            .field("bytes", &self.bytes)
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BatchStatus {
    Validating,
    InProgress,
    Finalizing,
    Completed,
    Failed,
    Expired,
    Cancelling,
    Cancelled,
}
impl BatchStatus {
    pub fn terminal(self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Failed | Self::Expired | Self::Cancelled
        )
    }
}

/// File IDs are endpoint-relative opaque identifiers, never downloadable URLs.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Batch {
    pub id: String,
    pub input_file_id: String,
    pub status: BatchStatus,
    pub output_file_id: Option<String>,
    pub error_file_id: Option<String>,
    pub total: Option<u64>,
    pub completed: Option<u64>,
    pub failed: Option<u64>,
}

/// A bounded read-only search. Exhaustion never establishes permission to create again.
#[derive(Clone, Copy, Debug)]
pub struct ReconcileLimits {
    pub pages: usize,
    pub bytes: usize,
    pub timeout: std::time::Duration,
}
impl Default for ReconcileLimits {
    fn default() -> Self {
        Self {
            pages: 20,
            bytes: 16 * 1024 * 1024,
            timeout: std::time::Duration::from_secs(120),
        }
    }
}
impl ReconcileLimits {
    pub(super) fn validate(self) -> Result<Self, Error> {
        let maximum = Self::default();
        if self.pages == 0
            || self.pages > maximum.pages
            || self.bytes == 0
            || self.bytes > maximum.bytes
            || self.timeout.is_zero()
            || self.timeout > maximum.timeout
        {
            return Err(Error::Invalid("Batch reconciliation limits are invalid"));
        }
        Ok(self)
    }
}

/// Only transport parsing constructs this evidence; arbitrary remote metadata is discarded.
#[derive(Clone)]
pub struct RemoteIdentity {
    pub(super) batch: Batch,
    pub(super) endpoint: String,
    pub(super) api_base: String,
    pub(super) submission_nonce: Option<String>,
}
impl RemoteIdentity {
    pub fn batch(&self) -> &Batch {
        &self.batch
    }
    pub fn api_base(&self) -> &str {
        &self.api_base
    }
    pub fn into_batch(self) -> Batch {
        self.batch
    }
    pub fn matches(&self, uploaded: &UploadedInput, nonce: &str) -> bool {
        (self.endpoint == ENDPOINT || self.endpoint == MESSAGES_ENDPOINT)
            && self.batch.input_file_id == uploaded.file_id
            && self.submission_nonce.as_deref() == Some(nonce)
    }
}
impl fmt::Debug for RemoteIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RemoteIdentity")
            .field("status", &self.batch.status)
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
pub enum Reconciliation {
    Found(Box<RemoteIdentity>),
    NotFound,
    /// At least two distinct matching IDs; no first/newest choice is made.
    Ambiguous(Vec<String>),
    /// Pagination, byte or time bounds were reached without a complete unique proof.
    Incomplete,
}

/// Raw provider bodies include observed usage even when HTTP/semantic results fail.
/// They are deliberately neither Serialize nor detailed Debug/Display values.
#[derive(Clone)]
pub struct ResultItem {
    pub custom_id: String,
    pub http_status: Option<u16>,
    pub body: Option<Value>,
    pub error: Option<Value>,
    pub request_id: Option<String>,
}
impl fmt::Debug for ResultItem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ResultItem")
            .field("http_status", &self.http_status)
            .field("has_body", &self.body.is_some())
            .field("has_error", &self.error.is_some())
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
pub struct Results {
    pub status: BatchStatus,
    /// Entries and missing IDs use the original submitted order.
    pub items: Vec<ResultItem>,
    pub missing: Vec<String>,
    pub bytes: usize,
}

/// A later malformed/oversized/interrupted file must not erase prior paid results.
#[derive(Debug)]
pub struct DownloadFailure {
    pub error: Error,
    pub partial: Results,
}
impl fmt::Display for DownloadFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.error, f)
    }
}
impl std::error::Error for DownloadFailure {}

pub(super) fn custom_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 64
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-'))
}
pub(super) fn identifier(value: &str, prefix: &str) -> bool {
    let Some(tail) = value.strip_prefix(prefix) else {
        return false;
    };
    tail.len() > 1
        && value.len() <= 256
        && matches!(tail.as_bytes()[0], b'_' | b'-')
        && tail
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-'))
}
