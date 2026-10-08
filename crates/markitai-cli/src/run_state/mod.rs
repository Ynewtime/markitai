//! Recovery storage primitives. CLI scheduling is connected in the next stage.
pub(crate) mod codec;
mod paths;
mod store;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::fmt;
use std::io;
use std::path::{Path, PathBuf};
pub(crate) use store::StateStore;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Mode {
    Directory,
    UrlList,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Scope {
    pub(crate) mode: Mode,
    pub(crate) input: PathBuf,
    pub(crate) output: PathBuf,
    #[serde(skip)]
    output_spelling: Option<PathBuf>,
}

impl PartialEq for Scope {
    fn eq(&self, other: &Self) -> bool {
        self.mode == other.mode && self.input == other.input && self.output == other.output
    }
}

impl Eq for Scope {}

impl Scope {
    pub(crate) fn new(mode: Mode, input: &Path, output: &Path) -> Result<Self> {
        Ok(Self {
            mode,
            input: crate::report_store::resolve_path(input)?,
            output: crate::report_store::resolve_path(output)?,
            output_spelling: Some(std::path::absolute(output)?),
        })
    }

    pub(crate) fn output_spelling(&self) -> &Path {
        self.output_spelling.as_deref().unwrap_or(&self.output)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Status {
    Pending,
    InProgress,
    Completed,
    Failed,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum ItemKey {
    File(String),
    Url(String),
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Entry {
    pub(crate) status: Status,
    pub(crate) source_file: Option<String>,
    pub(crate) url: Option<String>,
    pub(crate) output: Option<PathBuf>,
    pub(crate) target: Option<PathBuf>,
    pub(crate) error: Option<String>,
    /// Legacy measurements remain optional; missing data is never measured zero.
    pub(crate) observations: Map<String, Value>,
}

impl Default for Entry {
    fn default() -> Self {
        Self {
            status: Status::Pending,
            source_file: None,
            url: None,
            output: None,
            target: None,
            error: None,
            observations: Map::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Checkpoint {
    pub(crate) generation: String,
    pub(crate) applied_sequence: u64,
    pub(crate) scope: Scope,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Snapshot {
    pub(crate) options: IndexMap<String, Value>,
    pub(crate) documents: BTreeMap<String, Entry>,
    pub(crate) urls: IndexMap<String, Entry>,
    pub(crate) checkpoint: Option<Checkpoint>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Fence {
    pub(crate) generation: String,
    pub(crate) sequence: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Event {
    pub(crate) key: ItemKey,
    pub(crate) data: Value,
    pub(crate) fence: Option<Fence>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Limits {
    pub(crate) base_bytes: usize,
    pub(crate) journal_bytes: usize,
    pub(crate) line_bytes: usize,
    pub(crate) entries: usize,
}

impl Default for Limits {
    fn default() -> Self {
        // The base holds what the journal holds once compacted, so both share
        // one bound; 100,000 completed entries with diagnostics fit within it.
        Self {
            base_bytes: 64 * 1024 * 1024,
            journal_bytes: 64 * 1024 * 1024,
            line_bytes: 1024 * 1024,
            entries: 100_000,
        }
    }
}

#[derive(Debug)]
pub(crate) enum LoadOutcome {
    Missing,
    Loaded {
        snapshot: Box<Snapshot>,
        warnings: Vec<String>,
    },
    Corrupt {
        reason: String,
    },
}

#[derive(Debug)]
pub(crate) enum Error {
    Io(io::Error),
    Invalid(String),
    JsonSyntax(String),
    ForeignScope(String),
    Busy,
    Limit(&'static str),
    Sequence(String),
}

pub(crate) type Result<T> = std::result::Result<T, Error>;

impl From<io::Error> for Error {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "State I/O failed: {error}"),
            Self::Invalid(message) => write!(f, "Invalid recovery state: {message}"),
            Self::JsonSyntax(message) => write!(f, "Invalid recovery JSON: {message}"),
            Self::ForeignScope(message) => write!(f, "Recovery state scope mismatch: {message}"),
            Self::Busy => f.write_str("Another process owns this recovery checkpoint"),
            Self::Limit(kind) => write!(f, "Recovery state {kind} limit exceeded"),
            Self::Sequence(message) => write!(f, "Recovery journal sequence error: {message}"),
        }
    }
}

impl std::error::Error for Error {}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod audit;
