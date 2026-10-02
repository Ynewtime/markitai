//! Names survive the entire invocation. Native identity indices survive only
//! while the corresponding parent epoch is pinned; evicted indices rebuild
//! every family, including completed outputs that have since disappeared.
use super::{ClaimError, ItemKey};
use crate::output_claims::{self, Claim, NamespaceBatch, v2};
use markitai_core::platform::{self, FileId};
use std::collections::{BTreeSet, HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

const IDLE_PARENTS: usize = 16;

struct Family {
    members: Vec<String>,
    item: ItemKey,
}
struct LogicalParent {
    parent: PathBuf,
    originals: BTreeSet<PathBuf>,
    families: Vec<Family>,
}
struct Index {
    keys: HashMap<FileId, BTreeSet<ItemKey>>,
    epoch: Option<Arc<v2::Epoch>>,
    legacy_directory: Option<FileId>,
}
impl Drop for Index {
    fn drop(&mut self) {
        self.keys.clear(); // Never release a pin with a usable inode index.
        self.epoch.take();
    }
}
#[derive(Default)]
struct State {
    logical: HashMap<FileId, LogicalParent>,
    hot: HashMap<FileId, Index>,
    lru: VecDeque<FileId>,
    bindings: HashMap<PathBuf, FileId>,
}
#[derive(Default)]
pub(crate) struct Reservations {
    inner: Mutex<State>,
}

impl Reservations {
    pub(super) fn new() -> Self {
        Self::default()
    }

    /// Recovery registration stores exact names only. No probe, lock or epoch
    /// is opened until this parent is admitted or tested for a conflict.
    pub(super) fn reserve(
        &self,
        parent: &Path,
        members: &[String],
        item: ItemKey,
        allow: bool,
    ) -> Result<(), ClaimError> {
        let original = std::path::absolute(parent)?;
        let (parent, id) = observed_parent(parent, allow)?;
        let mut state = self
            .inner
            .lock()
            .map_err(|_| ClaimError::Invalid("reservation coordinator poisoned".into()))?;
        state.bind(&original, id)?;
        state.bind(&parent, id)?;
        let group = state.logical.entry(id).or_insert_with(|| LogicalParent {
            parent: parent.clone(),
            originals: BTreeSet::new(),
            families: Vec::new(),
        });
        group.originals.insert(original);
        group.families.push(Family {
            members: members.to_vec(),
            item,
        });
        // Recovery registrations happen before admission. A later registration
        // must discard the whole index before releasing its epoch.
        if let Some(index) = state.hot.remove(&id) {
            drop(index);
        }
        state.lru.retain(|key| *key != id);
        Ok(())
    }

    pub(super) fn epoch(
        &self,
        parent: &Path,
        allow: bool,
    ) -> Result<Option<Arc<v2::Epoch>>, ClaimError> {
        let mut state = self
            .inner
            .lock()
            .map_err(|_| ClaimError::Invalid("reservation coordinator poisoned".into()))?;
        let id = state.activate(parent, allow)?;
        Ok(state.hot[&id].epoch.clone())
    }

    pub(super) fn blocked(
        &self,
        parent: &Path,
        names: &[String],
        item: &ItemKey,
        allow: bool,
    ) -> Result<bool, ClaimError> {
        let mut state = self
            .inner
            .lock()
            .map_err(|_| ClaimError::Invalid("reservation coordinator poisoned".into()))?;
        // A pure skip in an unreserved directory performs no metadata writes.
        let (_, id) = observed_parent(parent, allow)?;
        if !state.logical.contains_key(&id) {
            return Ok(false);
        }
        state.activate(parent, allow)?;
        let index = &state.hot[&id];
        let keys = match &index.epoch {
            Some(epoch) => epoch.keys(names)?,
            None => output_claims::existing_keys(parent, names)?,
        };
        Ok(blocked(index, &keys, item))
    }

    pub(super) fn blocks_keys(
        &self,
        parent: &Path,
        keys: &[FileId],
        item: &ItemKey,
    ) -> Result<bool, ClaimError> {
        let id = platform::status(parent)?.id();
        let state = self
            .inner
            .lock()
            .map_err(|_| ClaimError::Invalid("reservation coordinator poisoned".into()))?;
        Ok(state
            .hot
            .get(&id)
            .is_some_and(|index| blocked(index, keys, item)))
    }

    pub(super) fn retain(&self, claim: &Claim, item: ItemKey) -> Result<(), ClaimError> {
        let (parent, id) = observed_parent(claim.parent(), true)?;
        let mut state = self
            .inner
            .lock()
            .map_err(|_| ClaimError::Invalid("reservation coordinator poisoned".into()))?;
        let group = state.logical.entry(id).or_insert_with(|| LogicalParent {
            parent: parent.clone(),
            originals: BTreeSet::new(),
            families: Vec::new(),
        });
        group.originals.insert(parent);
        group.families.push(Family {
            members: claim.members().to_vec(),
            item: item.clone(),
        });
        let index = state
            .hot
            .get_mut(&id)
            .ok_or_else(|| ClaimError::Invalid("claim has no active reservation index".into()))?;
        for key in claim.keys() {
            index.keys.entry(key).or_default().insert(item.clone());
        }
        // active claim and index must pin the same epoch, not merely same path.
        match (&index.epoch, claim.epoch()) {
            (None, None) => (),
            (Some(a), Some(b)) if Arc::ptr_eq(a, &b) => (),
            _ => {
                return Err(ClaimError::Invalid(
                    "claim changed reservation epoch".into(),
                ));
            }
        }
        Ok(())
    }
}

fn blocked(index: &Index, keys: &[FileId], item: &ItemKey) -> bool {
    keys.iter().any(|key| {
        index
            .keys
            .get(key)
            .is_some_and(|owners| owners.iter().any(|owner| owner != item))
    })
}

fn observed_parent(parent: &Path, allow: bool) -> Result<(PathBuf, FileId), ClaimError> {
    markitai_core::output::check_user_directory(parent, allow)
        .map_err(|error| ClaimError::Invalid(error.to_string()))?;
    let resolved = crate::report_store::resolve_path(parent)?;
    let status = platform::status(&resolved)?;
    if !status.metadata().is_dir() {
        return Err(ClaimError::Invalid(
            "reservation parent is not a directory".into(),
        ));
    }
    Ok((resolved, status.id()))
}

impl State {
    fn bind(&mut self, path: &Path, id: FileId) -> Result<(), ClaimError> {
        if self.bindings.get(path).is_some_and(|known| *known != id) {
            return Err(ClaimError::Invalid(
                "logical reservation parent identity changed".into(),
            ));
        }
        self.bindings.insert(path.to_owned(), id);
        Ok(())
    }

    fn activate(&mut self, requested: &Path, allow: bool) -> Result<FileId, ClaimError> {
        // Missing output parents are established by the same namespace fence
        // before a probe can exist. Existing hot parents avoid repeat flushes.
        let original = std::path::absolute(requested)?;
        let observed = observed_parent(requested, allow);
        if let Ok((parent, id)) = &observed {
            self.bind(&original, *id)?;
            self.bind(parent, *id)?;
        }
        if let Ok((_, id)) = &observed
            && let Some(index) = self.hot.get(id)
        {
            if let Some(epoch) = &index.epoch {
                epoch.validate()?;
            } else {
                let members = crate::report_store::resolve_path(requested)?
                    .join(".markitai/ownership/members");
                let status = platform::status(&members)?;
                if !status.metadata().is_dir()
                    || !status.private()
                    || !status.owned_by_current_user()
                    || Some(status.id()) != index.legacy_directory
                {
                    return Err(ClaimError::Invalid(
                        "legacy reservation namespace changed".into(),
                    ));
                }
            }
            self.lru.retain(|key| key != id);
            self.lru.push_back(*id);
            self.prune();
            return Ok(*id);
        }
        let mut batch = NamespaceBatch::new();
        batch.prepare(requested, allow)?;
        batch.commit()?.validate(requested, allow)?;
        let (parent, id) = observed_parent(requested, allow)?;
        self.bind(&original, id)?;
        self.bind(&parent, id)?;
        if let Some(group) = self.logical.get(&id) {
            for original in &group.originals {
                if observed_parent(original, allow)?.1 != id {
                    return Err(ClaimError::Invalid(
                        "logical reservation parent changed".into(),
                    ));
                }
            }
            if observed_parent(&group.parent, allow)?.1 != id {
                return Err(ClaimError::Invalid(
                    "logical reservation parent identity changed".into(),
                ));
            }
        }
        let members = parent.join(".markitai/ownership/members");
        let epoch = if platform::status(&members)?.metadata().is_dir() {
            None
        } else {
            Some(v2::Epoch::new(v2::Parent::open(&parent)?)?)
        };
        let mut keys: HashMap<FileId, BTreeSet<ItemKey>> = HashMap::new();
        if let Some(group) = self.logical.get(&id) {
            let families: Vec<_> = group
                .families
                .iter()
                .map(|family| family.members.clone())
                .collect();
            let identities = match &epoch {
                Some(epoch) => epoch.family_keys(&families)?,
                None => families
                    .iter()
                    .map(|names| output_claims::reserve_keys(&parent, names, allow))
                    .collect::<Result<Vec<_>, _>>()?,
            };
            for (family, ids) in group.families.iter().zip(identities) {
                for key in ids {
                    keys.entry(key).or_default().insert(family.item.clone());
                }
            }
        }
        // Only the complete rebuilt index becomes visible.
        let legacy_directory = if epoch.is_none() {
            Some(platform::status(&members)?.id())
        } else {
            None
        };
        self.hot.insert(
            id,
            Index {
                keys,
                epoch,
                legacy_directory,
            },
        );
        self.lru.retain(|key| *key != id);
        self.lru.push_back(id);
        self.prune();
        Ok(id)
    }

    fn prune(&mut self) {
        let idle: Vec<_> = self
            .lru
            .iter()
            .copied()
            .filter(|id| {
                self.hot.get(id).is_some_and(|index| {
                    index
                        .epoch
                        .as_ref()
                        .is_none_or(|epoch| Arc::strong_count(epoch) == 1)
                })
            })
            .collect();
        for id in idle.into_iter().rev().skip(IDLE_PARENTS) {
            self.lru.retain(|key| *key != id);
            if let Some(index) = self.hot.remove(&id) {
                drop(index);
            }
        }
    }
}

#[cfg(test)]
#[path = "reservations/tests.rs"]
mod tests;
