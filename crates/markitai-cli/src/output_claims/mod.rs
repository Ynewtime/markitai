//! Cooperating CLI writers claim real document members before conversion.
mod leases;
mod namespace;
mod receipts;
pub(crate) mod sync_group;
pub(crate) mod v2;
pub(crate) use receipts::group::{PreparedDocument, PublicationGroup, RenderedMember};

pub(crate) use leases::{
    MemberLeases, existing_keys, prepare_namespace_parent, prepare_output_ancestors, reserve_keys,
};
pub(crate) use namespace::{MAX_NAMESPACE_PARENTS, NamespaceBatch, PreparedNamespaces};
pub(crate) use receipts::{adopt_owner, reservation_members};
use serde::{Deserialize, Serialize};
use std::io;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Owner {
    pub(crate) generation: String,
    pub(crate) mode: String,
    pub(crate) input: PathBuf,
    pub(crate) output: PathBuf,
    pub(crate) kind: String,
    pub(crate) key: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Policy {
    NoClobber,
    Overwrite,
    RetryOwned,
}

#[derive(Debug)]
pub(crate) enum Error {
    Io(io::Error),
    Invalid(String),
    Busy,
    Ownership(String),
}

pub(crate) type Result<T> = std::result::Result<T, Error>;

impl From<io::Error> for Error {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "Output publication failed: {error}"),
            Self::Invalid(message) => write!(f, "Invalid output claim: {message}"),
            Self::Busy => f.write_str("Another process owns the requested output member"),
            Self::Ownership(message) => write!(f, "Output ownership mismatch: {message}"),
        }
    }
}
impl std::error::Error for Error {}

pub(crate) struct Claim {
    leases: MemberLeases,
    owner: Option<Owner>,
    policy: Policy,
    skip: bool,
}

impl Claim {
    /// Observing an existing result is not a write claim. No locks, receipt,
    /// ownership directories or provider work are needed for this decision.
    pub(crate) fn skipped(parent: &Path, allow_symlinks: bool) -> Result<Self> {
        Ok(Self {
            leases: MemberLeases::skipped(parent, allow_symlinks)?,
            owner: None,
            policy: Policy::NoClobber,
            skip: true,
        })
    }

    pub(crate) fn new(
        leases: MemberLeases,
        owner: Option<Owner>,
        policy: Policy,
        skip: bool,
    ) -> Result<Self> {
        if policy == Policy::RetryOwned {
            receipts::verify_owned(
                &leases,
                owner
                    .as_ref()
                    .ok_or_else(|| Error::Invalid("retry requires a native owner".into()))?,
            )?;
        }
        Ok(Self {
            leases,
            owner,
            policy,
            skip,
        })
    }
    pub(crate) fn parent(&self) -> &Path {
        self.leases.parent()
    }
    pub(crate) fn is_skip(&self) -> bool {
        self.skip
    }
    pub(crate) fn members(&self) -> &[String] {
        self.leases.members()
    }
    pub(crate) fn epoch(&self) -> Option<std::sync::Arc<v2::Epoch>> {
        self.leases.epoch()
    }
    pub(crate) fn keys(&self) -> Vec<markitai_core::platform::FileId> {
        self.leases.keys()
    }
    pub(crate) fn evidence_digest(&self) -> Result<String> {
        receipts::evidence_digest(
            &self.leases,
            self.owner.as_ref().ok_or_else(|| {
                Error::Invalid("publication evidence requires a native owner".into())
            })?,
        )
    }
}

impl markitai_core::output::Publication for Claim {
    fn skip_existing(&self) -> bool {
        self.skip
    }
    fn publish(&self, path: &Path, bytes: &[u8]) -> markitai_core::Result<()> {
        if self.skip {
            return Err(markitai_core::Error::InvalidInput(
                "A skipped output claim cannot publish".into(),
            ));
        }
        receipts::publish(&self.leases, self.owner.as_ref(), self.policy, path, bytes)
            .map_err(|error| markitai_core::Error::Conversion(error.to_string()))
    }
}
