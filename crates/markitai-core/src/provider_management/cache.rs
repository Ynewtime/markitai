use super::discovery;
use crate::{Error, Result, llm::ProbeReason as Reason};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    sync::{Arc, Condvar, Mutex, OnceLock},
    time::{Duration, Instant},
};

const FRESH: Duration = Duration::from_secs(300);
const STALE: Duration = Duration::from_secs(24 * 60 * 60);
const WAIT: Duration = Duration::from_secs(20);
const MAX_CONNECTIONS: usize = 32;

#[derive(Clone)]
struct Entry {
    value: Value,
    fetched: Instant,
}
#[derive(Default)]
struct SlotState {
    running: bool,
    good: Option<Entry>,
    last: Option<Value>,
}
#[derive(Default)]
struct Slot {
    state: Mutex<SlotState>,
    ready: Condvar,
}
#[derive(Default)]
pub(super) struct Cache {
    entries: Mutex<HashMap<[u8; 32], Arc<Slot>>>,
}

pub(super) fn global() -> &'static Cache {
    static CACHE: OnceLock<Cache> = OnceLock::new();
    CACHE.get_or_init(Cache::default)
}
fn identity(provider: &str, base: &str, key: Option<&str>) -> [u8; 32] {
    let mut hash = Sha256::new();
    for item in [provider, base, key.unwrap_or("")] {
        hash.update((item.len() as u64).to_le_bytes());
        hash.update(item.as_bytes());
    }
    hash.finalize().into()
}
fn poisoned<T>(_: std::sync::PoisonError<T>) -> Error {
    Error::Conversion("Provider discovery cache is unavailable".into())
}
fn cached(mut value: Value) -> Value {
    value["cached"] = true.into();
    value
}
impl Cache {
    pub(super) fn discover(
        &self,
        provider: &str,
        base: &str,
        key: Option<&str>,
        refresh: bool,
        loader: impl FnOnce() -> std::result::Result<Value, Value>,
    ) -> Result<Value> {
        let id = identity(provider, base, key);
        let slot = {
            let mut entries = self.entries.lock().map_err(poisoned)?;
            if !entries.contains_key(&id) && entries.len() >= MAX_CONNECTIONS {
                let removable = entries
                    .iter()
                    .filter(|(_, slot)| Arc::strong_count(slot) == 1)
                    .filter_map(|(key, slot)| {
                        slot.state.lock().ok().and_then(|state| {
                            (!state.running)
                                .then_some((*key, state.good.as_ref().map(|e| e.fetched)))
                        })
                    })
                    .min_by_key(|(_, time)| *time)
                    .map(|(key, _)| key);
                if let Some(key) = removable {
                    entries.remove(&key);
                } else {
                    return Ok(discovery::failed(
                        provider,
                        Reason::ProviderError,
                        "Too many model discovery requests are active",
                        None,
                    ));
                }
            }
            entries.entry(id).or_default().clone()
        };
        let mut state = slot.state.lock().map_err(poisoned)?;
        if !refresh
            && let Some(good) = &state.good
            && good.fetched.elapsed() < FRESH
        {
            return Ok(cached(good.value.clone()));
        }
        if state.running {
            let (state, wait) = slot
                .ready
                .wait_timeout_while(state, WAIT, |state| state.running)
                .map_err(poisoned)?;
            if wait.timed_out() && state.running {
                return Ok(discovery::failed(
                    provider,
                    Reason::Timeout,
                    "Model discovery wait timed out",
                    None,
                ));
            }
            // A failure is passed on as it is, never as a cached result.
            return Ok(state
                .last
                .clone()
                .map(|value| {
                    if value["status"] == "unavailable" {
                        value
                    } else {
                        cached(value)
                    }
                })
                .unwrap_or_else(|| {
                    discovery::failed(
                        provider,
                        Reason::ProviderError,
                        "Model discovery did not complete",
                        None,
                    )
                }));
        }
        state.running = true;
        drop(state);
        // Always notify waiters, including an unexpected parser panic. A panic
        // is not converted into a cached successful provider response.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(loader));
        let mut state = slot.state.lock().map_err(poisoned)?;
        state.running = false;
        let value = match result {
            Ok(Ok(value)) => {
                state.good = Some(Entry {
                    value: value.clone(),
                    fetched: Instant::now(),
                });
                value
            }
            failure => {
                if let Some(good) = &state.good
                    && good.fetched.elapsed() < STALE
                {
                    let mut value = cached(good.value.clone());
                    value["stale"] = true.into();
                    value["status"] = "partial".into();
                    value["detail"] = "Refresh failed; showing previously discovered models".into();
                    value
                } else {
                    match failure {
                        Ok(Err(failed)) => failed,
                        _ => discovery::failed(
                            provider,
                            Reason::ProviderError,
                            "Model discovery failed; check endpoint, credentials and provider availability",
                            None,
                        ),
                    }
                }
            }
        };
        state.last = Some(value.clone());
        slot.ready.notify_all();
        Ok(value)
    }
    #[cfg(test)]
    pub(super) fn age(&self, provider: &str, base: &str, key: Option<&str>, seconds: u64) {
        let slot = self.entries.lock().unwrap()[&identity(provider, base, key)].clone();
        slot.state.lock().unwrap().good.as_mut().unwrap().fetched =
            Instant::now() - Duration::from_secs(seconds);
    }
}
