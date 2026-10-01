//! Adaptive routing state belongs to a caller-owned runtime, never the process.
use super::{Deployment, Protocol, random_ticket};
use crate::{Error, Result};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const MAX_IDLE: usize = 4096;
const LATENCY_SAMPLES: usize = 10;
const USAGE_TTL: Duration = Duration::from_secs(60);
const LATENCY_TTL: Duration = Duration::from_secs(3600);

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Strategy {
    Shuffle,
    LeastBusy,
    Usage,
    Latency,
}
impl Strategy {
    pub(crate) fn parse(value: &str) -> Result<Self> {
        match value {
            "simple-shuffle" => Ok(Self::Shuffle),
            "least-busy" => Ok(Self::LeastBusy),
            "usage-based-routing" => Ok(Self::Usage),
            "latency-based-routing" => Ok(Self::Latency),
            _ => Err(Error::Unsupported("Unknown LLM routing strategy".into())),
        }
    }
    pub(crate) fn adaptive(self) -> bool {
        self != Self::Shuffle
    }
    pub(crate) fn measured(self) -> bool {
        matches!(self, Self::Usage | Self::Latency)
    }
}

// Salted resolved credentials are never exposed via Debug or serialization.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct Key([u8; 32]);
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct MetricKey {
    group: Key,
    deployment: Key,
}

#[derive(Default)]
pub(crate) enum Observation {
    Success {
        elapsed: Duration,
        total_tokens: Option<u64>,
        output_tokens: Option<u64>,
    },
    Timeout,
    #[default]
    OtherFailure,
}

#[derive(Clone, Copy)]
struct Stamp {
    minute: u64,
    tick: Duration,
}
struct Decision<'a> {
    strategy: Strategy,
    group: Option<Key>,
    keys: &'a [Key],
    candidates: &'a [usize],
    ticket: u128,
}
struct Group {
    strategy: Strategy,
    minute: u64,
    updated: Duration,
}
struct Metric {
    tokens: u128,
    samples: VecDeque<f64>,
    first_seen: u64,
    touched: u64,
}
#[derive(Default)]
struct State {
    active: HashMap<Key, usize>,
    active_metrics: HashMap<MetricKey, usize>,
    metrics: HashMap<MetricKey, Metric>,
    groups: HashMap<Key, Group>,
    sequence: u64,
}

pub(crate) struct Table {
    salt: [u8; 32],
    started: Instant,
    state: Mutex<State>,
    // Deployments that failed authentication beside a sibling stay out of
    // every later selection of this runtime. Only salted identities are kept.
    excluded: Mutex<HashSet<Key>>,
}
impl Table {
    pub(crate) fn new() -> Self {
        let mut salt = [0; 32];
        salt[..16].copy_from_slice(&random_ticket().to_le_bytes());
        salt[16..].copy_from_slice(&random_ticket().to_le_bytes());
        Self {
            salt,
            started: Instant::now(),
            state: Mutex::new(State::default()),
            excluded: Mutex::new(HashSet::new()),
        }
    }
    /// Returns true only for the call that newly excludes the deployment, so
    /// concurrent requests report each exclusion once.
    pub(crate) fn exclude(&self, key: Key) -> bool {
        self.excluded
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .insert(key)
    }
    pub(crate) fn included(&self, keys: &[Key], candidates: &[usize]) -> Vec<usize> {
        let excluded = self
            .excluded
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        candidates
            .iter()
            .copied()
            .filter(|&index| !excluded.contains(&keys[index]))
            .collect()
    }
    fn stamp(&self) -> Stamp {
        Stamp {
            minute: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
                / 60,
            tick: self.started.elapsed(),
        }
    }
    pub(super) fn key(&self, entry: &Deployment) -> Key {
        let mut hash = Sha256::new();
        hash.update(self.salt);
        for field in [
            entry.explicit_id.as_deref(),
            Some(entry.group.as_str()),
            Some(entry.provider.as_str()),
            Some(entry.model.as_str()),
            Some(entry.endpoint.as_str()),
            entry.key.as_deref(),
        ] {
            match field {
                Some(value) => {
                    hash.update([1]);
                    hash.update((value.len() as u64).to_le_bytes());
                    hash.update(value.as_bytes());
                }
                None => hash.update([0]),
            }
        }
        hash.update([match entry.protocol {
            Protocol::Chat => 0,
            Protocol::Anthropic => 1,
            Protocol::Azure => 2,
        }]);
        Key(hash.finalize().into())
    }
    pub(crate) fn group_key(
        &self,
        strategy: Strategy,
        visual: bool,
        keys: &[Key],
        candidates: &[usize],
    ) -> Option<Key> {
        if !strategy.measured() {
            return None;
        }
        let mut pool: Vec<_> = candidates.iter().map(|&index| keys[index]).collect();
        // Equal keys are identical, so a stable sort orders them as an
        // unstable one does.
        crate::sort::by(&mut pool, Key::cmp);
        pool.dedup();
        let mut hash = Sha256::new();
        hash.update(self.salt);
        hash.update([
            if strategy == Strategy::Usage { 1 } else { 2 },
            u8::from(visual),
        ]);
        for key in pool {
            hash.update(key.0);
        }
        Some(Key(hash.finalize().into()))
    }
    pub(crate) fn select<'a>(
        &'a self,
        strategy: Strategy,
        group: Option<Key>,
        keys: &[Key],
        candidates: &[usize],
        ticket: u128,
    ) -> (usize, Lease<'a>) {
        self.select_at(
            Decision {
                strategy,
                group,
                keys,
                candidates,
                ticket,
            },
            self.stamp(),
        )
    }
    fn select_at<'a>(&'a self, decision: Decision<'_>, now: Stamp) -> (usize, Lease<'a>) {
        let Decision {
            strategy,
            group,
            keys,
            candidates,
            ticket,
        } = decision;
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        state.expire(now);
        let metric = |index: usize| {
            group.map(|group| MetricKey {
                group,
                deployment: keys[index],
            })
        };
        let selected = match strategy {
            Strategy::LeastBusy => *candidates
                .iter()
                .min_by_key(|&&index| state.active.get(&keys[index]).copied().unwrap_or(0))
                .expect("eligible deployment"),
            Strategy::Usage => *candidates
                .iter()
                .min_by_key(|&&index| {
                    metric(index)
                        .and_then(|key| state.metrics.get(&key))
                        .map(|record| (record.tokens, record.first_seen))
                        .unwrap_or((0, u64::MAX))
                })
                .expect("eligible deployment"),
            Strategy::Latency => {
                let mut minimum = f64::INFINITY;
                let mut ties = Vec::new();
                for &index in candidates {
                    let score = metric(index)
                        .and_then(|key| state.metrics.get(&key))
                        .map_or(0.0, |record| {
                            if record.samples.is_empty() {
                                0.0
                            } else {
                                record.samples.iter().sum::<f64>() / record.samples.len() as f64
                            }
                        });
                    if score < minimum {
                        minimum = score;
                        ties.clear();
                    }
                    if score == minimum {
                        ties.push(index);
                    }
                }
                ties[(ticket % ties.len() as u128) as usize]
            }
            Strategy::Shuffle => unreachable!("weighted selection does not use metric state"),
        };
        let key = keys[selected];
        *state.active.entry(key).or_default() += 1;
        let metric = metric(selected);
        if let Some(metric) = metric {
            *state.active_metrics.entry(metric).or_default() += 1;
            state.sequence = state.sequence.saturating_add(1);
            let sequence = state.sequence;
            if let Some(record) = state.metrics.get_mut(&metric) {
                record.touched = sequence;
            }
        }
        (
            selected,
            Lease {
                table: self,
                key,
                metric,
                strategy,
            },
        )
    }
    #[cfg(test)]
    fn reserve<'a>(&'a self, keys: &[Key], candidates: &[usize]) -> (usize, Lease<'a>) {
        self.select(Strategy::LeastBusy, None, keys, candidates, 0)
    }
}
impl State {
    fn expire(&mut self, now: Stamp) {
        let previous = self.groups.len();
        self.groups.retain(|_, group| {
            let ttl = if group.strategy == Strategy::Usage {
                USAGE_TTL
            } else {
                LATENCY_TTL
            };
            (group.strategy != Strategy::Usage || group.minute == now.minute)
                && now.tick.saturating_sub(group.updated) < ttl
        });
        if self.groups.len() != previous {
            self.metrics
                .retain(|key, _| self.groups.contains_key(&key.group));
        }
    }
    fn observe(
        &mut self,
        key: MetricKey,
        strategy: Strategy,
        observation: Observation,
        now: Stamp,
    ) {
        self.expire(now);
        let (tokens, sample) = match (strategy, observation) {
            (
                Strategy::Usage,
                Observation::Success {
                    total_tokens: Some(tokens),
                    ..
                },
            ) => (Some(tokens), None),
            (
                Strategy::Latency,
                Observation::Success {
                    elapsed,
                    output_tokens,
                    ..
                },
            ) => {
                let seconds = elapsed.as_secs_f64();
                let sample = output_tokens
                    .filter(|&value| value > 0)
                    .map_or(seconds, |value| seconds / value as f64);
                (None, Some(sample))
            }
            (Strategy::Latency, Observation::Timeout) => (None, Some(1000.0)),
            _ => return,
        };
        self.groups.insert(
            key.group,
            Group {
                strategy,
                minute: now.minute,
                updated: now.tick,
            },
        );
        self.sequence = self.sequence.saturating_add(1);
        let sequence = self.sequence;
        let record = self.metrics.entry(key).or_insert_with(|| Metric {
            tokens: 0,
            samples: VecDeque::with_capacity(LATENCY_SAMPLES),
            first_seen: sequence,
            touched: sequence,
        });
        record.touched = sequence;
        if let Some(tokens) = tokens {
            record.tokens = record.tokens.saturating_add(u128::from(tokens));
        }
        if let Some(sample) = sample.filter(|value| value.is_finite() && *value >= 0.0) {
            if record.samples.len() == LATENCY_SAMPLES {
                record.samples.pop_front();
            }
            record.samples.push_back(sample);
        }
        self.prune();
    }
    fn prune(&mut self) {
        let mut removed = false;
        if self.metrics.len() > MAX_IDLE {
            let mut idle: Vec<_> = self
                .metrics
                .iter()
                .filter(|(key, _)| !self.active_metrics.contains_key(*key))
                .map(|(key, record)| (*key, record.touched))
                .collect();
            if idle.len() > MAX_IDLE {
                removed = true;
                // Ties keep the map's iteration order, which was already
                // arbitrary.
                crate::sort::by_key(&mut idle, |(_, touched)| *touched);
                for (key, _) in idle.iter().take(idle.len() - MAX_IDLE) {
                    self.metrics.remove(key);
                }
            }
        }
        if removed {
            let live: HashSet<_> = self.metrics.keys().map(|key| key.group).collect();
            self.groups.retain(|key, _| live.contains(key));
        }
    }
}

pub(crate) struct Lease<'a> {
    table: &'a Table,
    key: Key,
    metric: Option<MetricKey>,
    strategy: Strategy,
}
impl Lease<'_> {
    pub(crate) fn observe(&self, observation: Observation) {
        if let Some(metric) = self.metric {
            self.table
                .state
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .observe(metric, self.strategy, observation, self.table.stamp());
        }
    }
}
impl Drop for Lease<'_> {
    fn drop(&mut self) {
        let mut state = self
            .table
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some(count) = state.active.get_mut(&self.key) {
            *count -= 1;
            if *count == 0 {
                state.active.remove(&self.key);
            }
        }
        if let Some(metric) = self.metric {
            if let Some(count) = state.active_metrics.get_mut(&metric) {
                *count -= 1;
                if *count == 0 {
                    state.active_metrics.remove(&metric);
                }
            }
            state.prune();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LlmRuntime;

    #[test]
    fn occupancy_is_atomic_and_unwind_releases_without_retaining_history() {
        let table = Table::new();
        let keys = [Key([1; 32]), Key([2; 32])];
        let (first, lease) = table.reserve(&keys, &[0, 1]);
        assert_eq!(first, 0);
        let caught = std::panic::catch_unwind(|| {
            let (second, _lease) = table.reserve(&keys, &[0, 1]);
            assert_eq!(second, 1);
            panic!("authored request unwind");
        });
        assert!(caught.is_err());
        let (next, next_lease) = table.reserve(&keys, &[0, 1]);
        assert_eq!(next, 1);
        drop(next_lease);
        drop(lease);
        assert!(table.state.lock().unwrap().active.is_empty());
    }

    #[test]
    fn every_resolved_identity_component_separates_occupancy() {
        let table = Table::new();
        let cfg = serde_json::json!({"llm":{"model_list":[{"model_name":"default","litellm_params":{"model":"openai/test","api_base":"http://127.0.0.1:9","api_key":"authored-key"}}]}});
        let entry = super::super::deployments(&cfg, &HashMap::new())
            .unwrap()
            .remove(0);
        let original = table.key(&entry);
        let variants: Vec<_> = (0..6)
            .map(|field| {
                let mut changed = entry.clone();
                match field {
                    0 => changed.explicit_id = Some("deployment-b".into()),
                    1 => changed.endpoint.push_str("?other=1"),
                    2 => changed.key = Some("other-authored-key".into()),
                    3 => changed.model.push_str("-other"),
                    4 => changed.group = "backup".into(),
                    _ => changed.protocol = Protocol::Anthropic,
                }
                table.key(&changed)
            })
            .collect();
        for key in variants {
            assert!(key != original);
        }
        let mut weighted = entry;
        weighted.weight = 100;
        assert!(table.key(&weighted) == original);
        assert!(!format!("{:?}", LlmRuntime::new(2).unwrap()).contains("authored-key"));
    }

    fn at(minute: u64, seconds: u64) -> Stamp {
        Stamp {
            minute,
            tick: Duration::from_secs(seconds),
        }
    }
    fn measured(table: &Table, strategy: Strategy, keys: &[Key]) -> Key {
        table
            .group_key(strategy, false, keys, &(0..keys.len()).collect::<Vec<_>>())
            .unwrap()
    }
    fn record(
        table: &Table,
        group: Key,
        key: Key,
        strategy: Strategy,
        observation: Observation,
        now: Stamp,
    ) {
        table.state.lock().unwrap().observe(
            MetricKey {
                group,
                deployment: key,
            },
            strategy,
            observation,
            now,
        );
    }
    fn choose(
        table: &Table,
        strategy: Strategy,
        group: Key,
        keys: &[Key],
        ticket: u128,
        now: Stamp,
    ) -> usize {
        let candidates = (0..keys.len()).collect::<Vec<_>>();
        let (index, _lease) = table.select_at(
            Decision {
                strategy,
                group: Some(group),
                keys,
                candidates: &candidates,
                ticket,
            },
            now,
        );
        index
    }
    fn success(seconds: u64, total: Option<u64>, output: Option<u64>) -> Observation {
        Observation::Success {
            elapsed: Duration::from_secs(seconds),
            total_tokens: total,
            output_tokens: output,
        }
    }

    #[test]
    fn usage_known_zero_unknown_and_fixed_minute_rollover_are_distinct() {
        let table = Table::new();
        let keys = [Key([1; 32]), Key([2; 32])];
        let group = measured(&table, Strategy::Usage, &keys);
        record(
            &table,
            group,
            keys[1],
            Strategy::Usage,
            success(1, None, None),
            at(100, 0),
        );
        assert!(table.state.lock().unwrap().metrics.is_empty());
        record(
            &table,
            group,
            keys[1],
            Strategy::Usage,
            success(1, Some(0), Some(0)),
            at(100, 1),
        );
        // Known zero has an observed insertion order; unknown is appended after it.
        assert_eq!(
            choose(&table, Strategy::Usage, group, &keys, 0, at(100, 2)),
            1
        );
        record(
            &table,
            group,
            keys[1],
            Strategy::Usage,
            success(1, Some(20), Some(2)),
            at(100, 3),
        );
        assert_eq!(
            choose(&table, Strategy::Usage, group, &keys, 0, at(100, 4)),
            0
        );
        record(
            &table,
            group,
            keys[0],
            Strategy::Usage,
            success(1, Some(30), Some(2)),
            at(100, 5),
        );
        assert_eq!(
            choose(&table, Strategy::Usage, group, &keys, 0, at(100, 6)),
            1
        );
        assert_eq!(
            choose(&table, Strategy::Usage, group, &keys, 0, at(101, 7)),
            0
        );
        assert!(table.state.lock().unwrap().metrics.is_empty());
        record(
            &table,
            group,
            keys[0],
            Strategy::Usage,
            success(1, Some(30), Some(2)),
            at(101, 8),
        );
        assert_eq!(
            choose(&table, Strategy::Usage, group, &keys, 0, at(99, 9)),
            0
        );
        assert!(table.state.lock().unwrap().metrics.is_empty());
    }

    #[test]
    fn usage_expiry_is_bounded_even_when_wall_clock_does_not_advance() {
        let table = Table::new();
        let keys = [Key([1; 32]), Key([2; 32])];
        let group = measured(&table, Strategy::Usage, &keys);
        record(
            &table,
            group,
            keys[0],
            Strategy::Usage,
            success(1, Some(9), Some(2)),
            at(7, 0),
        );
        assert_eq!(
            choose(&table, Strategy::Usage, group, &keys, 0, at(7, 59)),
            1
        );
        assert_eq!(
            choose(&table, Strategy::Usage, group, &keys, 0, at(7, 60)),
            0
        );
        assert!(table.state.lock().unwrap().groups.is_empty());
    }

    #[test]
    fn latency_normalizes_tokens_retains_ten_samples_and_penalizes_only_timeouts() {
        let table = Table::new();
        let keys = [Key([1; 32]), Key([2; 32])];
        let group = measured(&table, Strategy::Latency, &keys);
        record(
            &table,
            group,
            keys[0],
            Strategy::Latency,
            success(10, None, Some(10)),
            at(1, 1),
        );
        record(
            &table,
            group,
            keys[1],
            Strategy::Latency,
            success(20, None, Some(100)),
            at(1, 2),
        );
        assert_eq!(
            choose(&table, Strategy::Latency, group, &keys, 0, at(1, 3)),
            1
        );
        record(
            &table,
            group,
            keys[1],
            Strategy::Latency,
            Observation::OtherFailure,
            at(1, 4),
        );
        assert_eq!(
            choose(&table, Strategy::Latency, group, &keys, 0, at(1, 5)),
            1
        );
        record(
            &table,
            group,
            keys[1],
            Strategy::Latency,
            Observation::Timeout,
            at(1, 6),
        );
        assert_eq!(
            choose(&table, Strategy::Latency, group, &keys, 0, at(1, 7)),
            0
        );
        for time in 8..18 {
            record(
                &table,
                group,
                keys[1],
                Strategy::Latency,
                success(1, None, Some(10)),
                at(1, time),
            );
        }
        assert_eq!(
            choose(&table, Strategy::Latency, group, &keys, 0, at(1, 18)),
            1
        );
        let key = MetricKey {
            group,
            deployment: keys[1],
        };
        assert_eq!(table.state.lock().unwrap().metrics[&key].samples.len(), 10);
        for output in [None, Some(0)] {
            record(
                &table,
                group,
                keys[1],
                Strategy::Latency,
                success(2, None, output),
                at(1, 19),
            );
            assert_eq!(
                table.state.lock().unwrap().metrics[&key].samples.back(),
                Some(&2.0)
            );
        }
    }

    #[test]
    fn latency_group_ttl_refresh_and_minimum_ties_match_the_declared_contract() {
        let table = Table::new();
        let keys = [Key([1; 32]), Key([2; 32])];
        let group = measured(&table, Strategy::Latency, &keys);
        record(
            &table,
            group,
            keys[0],
            Strategy::Latency,
            success(1, None, None),
            at(1, 0),
        );
        record(
            &table,
            group,
            keys[1],
            Strategy::Latency,
            success(2, None, None),
            at(2, 3500),
        );
        assert_eq!(
            choose(&table, Strategy::Latency, group, &keys, 1, at(3, 3601)),
            0
        );
        assert_eq!(
            choose(&table, Strategy::Latency, group, &keys, 1, at(4, 7100)),
            1
        );
        assert!(table.state.lock().unwrap().metrics.is_empty());
        for ticket in 0..8 {
            assert_eq!(
                choose(&table, Strategy::Latency, group, &keys, ticket, at(4, 7101)),
                (ticket % 2) as usize
            );
        }
        assert!(table.group_key(Strategy::Latency, true, &keys, &[0, 1]) != Some(group));
        assert!(table.group_key(Strategy::Usage, false, &keys, &[0, 1]) != Some(group));
        assert!(table.group_key(Strategy::Latency, false, &keys, &[1, 0]) == Some(group));
    }

    #[test]
    fn idle_metric_bound_preserves_active_records_and_removes_orphan_groups() {
        let table = Table::new();
        let key = Key([255; 32]);
        let group = Key([254; 32]);
        let (_, lease) = table.select_at(
            Decision {
                strategy: Strategy::Usage,
                group: Some(group),
                keys: &[key],
                candidates: &[0],
                ticket: 0,
            },
            at(1, 0),
        );
        record(
            &table,
            group,
            key,
            Strategy::Usage,
            success(1, Some(1), None),
            at(1, 0),
        );
        for index in 0..MAX_IDLE + 5 {
            let mut bytes = [0; 32];
            bytes[..8].copy_from_slice(&(index as u64).to_le_bytes());
            record(
                &table,
                Key(bytes),
                Key(bytes),
                Strategy::Usage,
                success(1, Some(1), None),
                at(1, 1),
            );
        }
        {
            let state = table.state.lock().unwrap();
            assert_eq!(state.metrics.len(), MAX_IDLE + 1);
            assert!(state.metrics.contains_key(&MetricKey {
                group,
                deployment: key
            }));
            // The least recently touched idle metrics went first.
            let metric = |index: u64| {
                let mut bytes = [0; 32];
                bytes[..8].copy_from_slice(&index.to_le_bytes());
                MetricKey {
                    group: Key(bytes),
                    deployment: Key(bytes),
                }
            };
            assert!(!state.metrics.contains_key(&metric(0)));
            assert!(state.metrics.contains_key(&metric(MAX_IDLE as u64 + 4)));
            assert_eq!(state.groups.len(), MAX_IDLE + 1);
        }
        drop(lease);
        let state = table.state.lock().unwrap();
        assert_eq!(state.metrics.len(), MAX_IDLE);
        assert_eq!(state.groups.len(), MAX_IDLE);
        assert!(state.active.is_empty());
        assert!(state.active_metrics.is_empty());
    }
}
