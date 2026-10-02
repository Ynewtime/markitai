//! In-flight semantic answers, scoped to one caller-owned runtime.
use super::*;
use sha2::{Digest, Sha256};
use std::io::Write;
use std::sync::{Condvar, atomic::AtomicUsize};

const MAX_ACTIVE: usize = 128;
const MAX_ANSWER: usize = 16 * 1024 * 1024;
const MAX_RETAINED: usize = 64 * 1024 * 1024;

// Neither request fingerprints nor the per-runtime salt implement Debug.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Key([u8; 32]);
pub(crate) struct Table {
    salt: [u8; 32],
    entries: Mutex<HashMap<Key, Arc<Entry>>>,
    retained: Arc<AtomicUsize>,
}
struct Entry {
    state: Mutex<State>,
    changed: Condvar,
}
#[derive(Default)]
struct State {
    done: bool,
    answer: Option<Arc<Stored>>,
}
struct Stored {
    value: Value,
    bytes: usize,
    retained: Arc<AtomicUsize>,
}
impl Drop for Stored {
    fn drop(&mut self) {
        self.retained.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}
pub(crate) struct Owner {
    table: Arc<Table>,
    key: Key,
    entry: Arc<Entry>,
    finished: bool,
}
enum Joined {
    Owner(Owner),
    Waiting(Arc<Entry>),
    Bypass,
}
impl Table {
    pub(crate) fn new() -> Self {
        let mut salt = [0; 32];
        salt[..16].copy_from_slice(&random_ticket().to_le_bytes());
        salt[16..].copy_from_slice(&random_ticket().to_le_bytes());
        Self {
            salt,
            entries: Mutex::new(HashMap::new()),
            retained: Arc::new(AtomicUsize::new(0)),
        }
    }
    fn join(self: &Arc<Self>, key: Key) -> Joined {
        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(entry) = entries.get(&key) {
            return Joined::Waiting(entry.clone());
        }
        if entries.len() >= MAX_ACTIVE {
            return Joined::Bypass;
        }
        let entry = Arc::new(Entry {
            state: Mutex::new(State::default()),
            changed: Condvar::new(),
        });
        entries.insert(key, entry.clone());
        Joined::Owner(Owner {
            table: self.clone(),
            key,
            entry,
            finished: false,
        })
    }
}
impl Owner {
    fn publish(mut self, value: Value) {
        struct Count(usize);
        impl Write for Count {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.0 = self.0.saturating_add(bytes.len());
                if self.0 > MAX_ANSWER {
                    return Err(std::io::Error::other("shared answer limit"));
                }
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let mut count = Count(0);
        if serde_json::to_writer(&mut count, &value).is_err() {
            return;
        }
        // Rust 1.99 renames this `try_update`; the minimum supported Rust predates the new name.
        #[allow(deprecated)]
        let reserved =
            self.table
                .retained
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                    used.checked_add(count.0)
                        .filter(|total| *total <= MAX_RETAINED)
                });
        if reserved.is_err() {
            return;
        }
        let stored = Arc::new(Stored {
            value,
            bytes: count.0,
            retained: self.table.retained.clone(),
        });
        self.finish(Some(stored));
    }
    fn finish(&mut self, answer: Option<Arc<Stored>>) {
        // Remove only this generation. No table lock crosses HTTP/cache I/O.
        let mut entries = self.table.entries.lock().unwrap_or_else(|e| e.into_inner());
        if entries
            .get(&self.key)
            .is_some_and(|entry| Arc::ptr_eq(entry, &self.entry))
        {
            entries.remove(&self.key);
        }
        let mut state = self.entry.state.lock().unwrap_or_else(|e| e.into_inner());
        state.answer = answer;
        state.done = true;
        self.finished = true;
        drop(state);
        drop(entries);
        self.entry.changed.notify_all();
    }
}
impl Drop for Owner {
    fn drop(&mut self) {
        if !self.finished {
            self.finish(None);
        }
    }
}

/// Exact rendered inputs and resolved routing policy are hashed only in memory.
/// Persistent cache keys keep their established credential-independent meaning.
pub(super) fn key<'a>(
    runtime: &LlmRuntime,
    semantic_key: Option<&str>,
    context: &str,
    prompts: &Prompts,
    cfg: &Value,
    env: &HashMap<String, String>,
    images: impl IntoIterator<Item = (usize, &'a str, &'a [u8])>,
) -> Option<Key> {
    let semantic_key = semantic_key?;
    if !cfg
        .pointer("/cache/enabled")
        .and_then(Value::as_bool)
        .unwrap_or(true)
        || llm_cache::bypasses(cfg, context)
    {
        return None;
    }
    let entries = deployments(cfg, env).ok()?;
    let groups = fallback_groups(cfg, &entries).ok()?;
    let mut hash = Sha256::new();
    hash.update(runtime.flights().salt);
    fn part(hash: &mut Sha256, bytes: &[u8]) {
        hash.update((bytes.len() as u64).to_le_bytes());
        hash.update(bytes);
    }
    for text in [
        "typed-flight-v1",
        semantic_key,
        &prompts.system,
        &prompts.user,
        &prompts.cache_scope,
    ] {
        part(&mut hash, text.as_bytes());
    }
    // serde_json maps have stable key ordering. Serialize directly into the
    // digest rather than retaining another secret-bearing configuration string.
    struct Writer<'a>(&'a mut Sha256);
    impl Write for Writer<'_> {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.update(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    serde_json::to_writer(
        Writer(&mut hash),
        &json!({"llm":cfg.get("llm"),"cache":cfg.get("cache"),"groups":groups}),
    )
    .ok()?;
    for entry in entries {
        for text in [
            &entry.id,
            &entry.group,
            &entry.model,
            &entry.provider,
            &entry.endpoint,
        ] {
            part(&mut hash, text.as_bytes());
        }
        part(&mut hash, entry.key.as_deref().unwrap_or("").as_bytes());
        hash.update(entry.weight.to_le_bytes());
        hash.update([match entry.protocol {
            Protocol::Chat => 0,
            Protocol::Anthropic => 1,
            Protocol::Azure => 2,
        }]);
        hash.update(entry.max_tokens.unwrap_or(0).to_le_bytes());
        hash.update([match entry.supports_vision {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        }]);
    }
    for (number, mime, bytes) in images {
        hash.update((number as u64).to_le_bytes());
        part(&mut hash, mime.as_bytes());
        part(&mut hash, bytes);
    }
    Some(Key(hash.finalize().into()))
}

/// Only the caller that sends requests owns their attempts and paid usage.
/// Failed owners publish nothing: waiting documents retry under their own scope.
pub(super) fn execute<T>(
    runtime: &LlmRuntime,
    key: Option<Key>,
    stop: Option<&std::sync::atomic::AtomicBool>,
    load: impl Fn() -> Option<T>,
    validate: impl Fn(&Value) -> Result<T>,
    value: impl Fn(&T) -> Value,
    operation: impl FnOnce() -> (std::result::Result<T, VisionFailure>, Option<String>),
) -> (std::result::Result<T, VisionFailure>, Option<String>, bool) {
    let Some(key) = key else {
        let (result, warning) = operation();
        return (result, warning, false);
    };
    let cancelled = || stop.is_some_and(|stop| stop.load(Ordering::Acquire));
    loop {
        if cancelled() {
            return (
                Err(VisionFailure::blocked(Error::Conversion(
                    "LLM processing was cancelled before a shared request completed".into(),
                ))),
                None,
                false,
            );
        }
        match runtime.flights().join(key) {
            Joined::Bypass => break,
            Joined::Owner(owner) => {
                if let Some(answer) = load() {
                    owner.publish(value(&answer));
                    return (Ok(answer), None, true);
                }
                let (answer, warning) = operation();
                if let Ok(answer) = &answer {
                    owner.publish(value(answer));
                }
                return (answer, warning, false);
            }
            Joined::Waiting(entry) => {
                let mut state = entry.state.lock().unwrap_or_else(|e| e.into_inner());
                while !state.done && !cancelled() {
                    state = entry
                        .changed
                        .wait_timeout(state, Duration::from_millis(50))
                        .unwrap_or_else(|e| e.into_inner())
                        .0;
                }
                if cancelled() {
                    continue;
                }
                let answer = state.answer.clone();
                drop(state);
                if let Some(answer) = answer {
                    if let Ok(validated) = validate(&answer.value) {
                        return (Ok(validated), None, true);
                    }
                    // A caller's source guard takes priority over shared data.
                    break;
                }
            }
        }
    }
    let (answer, warning) = operation();
    (answer, warning, false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{atomic::AtomicBool, mpsc};
    use std::time::Instant;

    fn owner(table: &Arc<Table>, key: Key) -> Owner {
        match table.join(key) {
            Joined::Owner(owner) => owner,
            _ => panic!("expected owner"),
        }
    }
    fn attached(entry: &Arc<Entry>, baseline: usize) {
        let deadline = Instant::now() + Duration::from_secs(2);
        while Arc::strong_count(entry) <= baseline {
            assert!(Instant::now() < deadline, "waiter failed to attach");
            std::thread::yield_now();
        }
    }
    #[test]
    fn successful_owner_wakes_all_and_reclaims_both_table_and_answer_bytes() {
        let table = Arc::new(Table::new());
        let key = Key([1; 32]);
        let owner = owner(&table, key);
        let mut waiters = Vec::new();
        for _ in 0..8 {
            let Joined::Waiting(entry) = table.join(key) else {
                panic!("duplicate owner");
            };
            waiters.push(entry);
        }
        owner.publish(json!({"body":"complete"}));
        assert!(table.entries.lock().unwrap().is_empty());
        assert!(table.retained.load(Ordering::Acquire) > 0);
        for entry in &waiters {
            let state = entry.state.lock().unwrap();
            assert!(state.done);
            assert_eq!(state.answer.as_ref().unwrap().value["body"], "complete");
        }
        drop(waiters);
        assert_eq!(table.retained.load(Ordering::Acquire), 0);
    }
    #[test]
    fn unwound_owner_is_replaced_without_sharing_failure_or_spend() {
        let runtime = LlmRuntime::new(1).unwrap();
        let key = Key([2; 32]);
        let first = owner(runtime.flights(), key);
        let entry = first.entry.clone();
        let baseline = Arc::strong_count(&entry);
        let clone = runtime.clone();
        let worker = std::thread::spawn(move || {
            execute(
                &clone,
                Some(key),
                None,
                || None,
                |value| Ok(value.clone()),
                Clone::clone,
                || (Ok(json!({"new_owner":true})), None),
            )
        });
        attached(&entry, baseline);
        let failed = std::panic::catch_unwind(|| {
            let _guard = first;
            panic!("owner panic");
        });
        assert!(failed.is_err());
        let (answer, warning, reused) = worker.join().unwrap();
        assert_eq!(answer.unwrap()["new_owner"], true);
        assert!(warning.is_none());
        assert!(!reused);
        assert!(runtime.flights().entries.lock().unwrap().is_empty());
    }
    #[test]
    fn cancelled_waiter_detaches_without_cancelling_owner_or_other_waiters() {
        let runtime = LlmRuntime::new(1).unwrap();
        let key = Key([3; 32]);
        let first = owner(runtime.flights(), key);
        let entry = first.entry.clone();
        let cancelled = Arc::new(AtomicBool::new(false));
        let stop = cancelled.clone();
        let clone = runtime.clone();
        let baseline = Arc::strong_count(&entry);
        let (sent, received) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            let result = execute(
                &clone,
                Some(key),
                Some(&stop),
                || None,
                |value| Ok(value.clone()),
                Clone::clone,
                || panic!("cancelled waiter sent a request"),
            );
            sent.send(result.0.is_err()).unwrap();
        });
        attached(&entry, baseline);
        cancelled.store(true, Ordering::Release);
        assert!(received.recv_timeout(Duration::from_secs(2)).unwrap());
        worker.join().unwrap();
        assert!(!entry.state.lock().unwrap().done);
        assert_eq!(runtime.flights().entries.lock().unwrap().len(), 1);
        first.publish(json!({"still":"complete"}));
        assert_eq!(
            entry.state.lock().unwrap().answer.as_ref().unwrap().value["still"],
            "complete"
        );
    }
    #[test]
    fn bounded_table_and_large_results_bypass_without_evicting_active_owners() {
        let table = Arc::new(Table::new());
        let owners: Vec<_> = (0..MAX_ACTIVE)
            .map(|index| {
                let mut key = [0; 32];
                key[..8].copy_from_slice(&(index as u64).to_le_bytes());
                owner(&table, Key(key))
            })
            .collect();
        assert!(matches!(table.join(Key([255; 32])), Joined::Bypass));
        assert!(matches!(table.join(Key([0; 32])), Joined::Waiting(_)));
        drop(owners);
        let large = owner(&table, Key([9; 32]));
        let entry = large.entry.clone();
        large.publish(Value::String("x".repeat(MAX_ANSWER)));
        assert!(entry.state.lock().unwrap().done);
        assert!(entry.state.lock().unwrap().answer.is_none());
        assert_eq!(table.retained.load(Ordering::Acquire), 0);
        assert!(table.entries.lock().unwrap().is_empty());
    }
    #[test]
    fn keys_isolate_resolved_identity_and_all_rendered_inputs_without_debug_secrets() {
        let runtime = LlmRuntime::new(2).unwrap();
        let config = json!({"cache":{"enabled":true},"llm":{"model_list":[{"litellm_params":{"model":"openai/gpt-4.1","api_key":"env:FLIGHT_KEY","api_base":"http://127.0.0.1:9000/v1"}}]}});
        let prompt = Prompts {
            system: "system".into(),
            user: "source".into(),
            cache_scope: "v1".into(),
            image: None,
        };
        let env = HashMap::from([("FLIGHT_KEY".into(), "secret-one".into())]);
        let make = |cfg: &Value, env: &HashMap<String, String>, prompt: &Prompts, bytes: &[u8]| {
            key(
                &runtime,
                Some("semantic"),
                "source.md",
                prompt,
                cfg,
                env,
                [(1, "image/png", bytes)],
            )
            .unwrap()
        };
        let base = make(&config, &env, &prompt, b"pixels");
        assert!(base == make(&config, &env, &prompt, b"pixels"));
        let rotated = HashMap::from([("FLIGHT_KEY".into(), "secret-two".into())]);
        assert!(base != make(&config, &rotated, &prompt, b"pixels"));
        let mut changed = config.clone();
        changed["llm"]["model_list"][0]["litellm_params"]["api_base"] =
            json!("http://127.0.0.1:9001/v1");
        assert!(base != make(&changed, &env, &prompt, b"pixels"));
        assert!(base != make(&config, &env, &prompt, b"changed pixels"));
        let prompt2 = Prompts {
            user: "changed source".into(),
            ..prompt
        };
        assert!(base != make(&config, &env, &prompt2, b"pixels"));
        for patch in [
            json!({"enabled":false}),
            json!({"enabled":true,"no_cache":true}),
            json!({"enabled":true,"no_cache_patterns":["*.md"]}),
        ] {
            let mut cfg = config.clone();
            cfg["cache"] = patch;
            assert!(
                key(
                    &runtime,
                    Some("semantic"),
                    "source.md",
                    &prompt2,
                    &cfg,
                    &env,
                    std::iter::empty()
                )
                .is_none()
            );
        }
        let debug = format!("{runtime:?}");
        assert!(!debug.contains("secret") && !debug.contains("9000"));
    }
}
