//! Acceptance of persistent state through the codec/store boundary. These tests
//! never invoke conversion or claim that the CLI resume scheduler is connected.
use super::{
    Checkpoint, Entry, Error, Event, Fence, ItemKey, Limits, LoadOutcome, Mode, Scope, Snapshot,
    StateStore, Status, codec,
};
use serde_json::{Value, json};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const HASH: &str = "abc123";
const GENERATION: &str = "12345678-1234-4234-8234-123456789abc";

struct Fixture {
    root: tempfile::TempDir,
    scope: Scope,
}
impl Fixture {
    fn directory() -> Self {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(root.path().join("input/nested")).unwrap();
        std::fs::create_dir_all(root.path().join("output/nested")).unwrap();
        let scope = Scope::new(
            Mode::Directory,
            &root.path().join("input"),
            &root.path().join("output"),
        )
        .unwrap();
        Self { root, scope }
    }
    fn open(&self) -> StateStore {
        self.open_with(Limits::default())
    }
    fn open_with(&self, limits: Limits) -> StateStore {
        StateStore::open(self.scope.clone(), HASH, false, limits).unwrap()
    }
    fn directory_path(&self) -> PathBuf {
        self.scope.output.join(".markitai/states")
    }
    fn base(&self) -> PathBuf {
        self.directory_path()
            .join(format!("markitai.{HASH}.state.json"))
    }
    fn journal(&self) -> PathBuf {
        self.directory_path()
            .join(format!("markitai.{HASH}.state.jsonl"))
    }
    fn lock(&self) -> PathBuf {
        self.directory_path()
            .join(format!("markitai.{HASH}.state.lock"))
    }
    fn output(&self, key: &str) -> PathBuf {
        self.scope.output.join(format!("{key}.md"))
    }
    fn seed(&self, snapshot: &Snapshot, journal: &[u8]) {
        std::fs::create_dir_all(self.directory_path()).unwrap();
        let bytes = codec::encode(snapshot, &self.scope, false, Limits::default()).unwrap();
        std::fs::write(self.base(), bytes).unwrap();
        std::fs::write(self.journal(), journal).unwrap();
    }
}
fn discovered(keys: &[&str]) -> Snapshot {
    let mut snapshot = Snapshot::default();
    for key in keys {
        snapshot
            .documents
            .insert((*key).to_owned(), Entry::default());
    }
    snapshot
}
fn loaded(store: &mut StateStore) -> (Snapshot, Vec<String>) {
    match store.load().unwrap() {
        LoadOutcome::Loaded { snapshot, warnings } => (*snapshot, warnings),
        other => panic!("expected loaded checkpoint, received {other:?}"),
    }
}
fn event(key: &str, data: Value, fence: Option<(&str, u64)>) -> Event {
    Event {
        key: ItemKey::File(key.into()),
        data,
        fence: fence.map(|(generation, sequence)| Fence {
            generation: generation.into(),
            sequence,
        }),
    }
}
fn journal(events: &[Event]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for event in events {
        bytes.extend(codec::encode_event(event).unwrap());
        bytes.push(b'\n');
    }
    bytes
}
fn completed(fixture: &Fixture, key: &str) -> Value {
    json!({"status":"completed", "output":fixture.output(key)})
}
fn fenced(mut snapshot: Snapshot, scope: &Scope, sequence: u64) -> Snapshot {
    snapshot.checkpoint = Some(Checkpoint {
        generation: GENERATION.into(),
        applied_sequence: sequence,
        scope: scope.clone(),
    });
    snapshot
}
fn same_directory_files(fixture: &Fixture) -> Vec<PathBuf> {
    let mut paths: Vec<_> = std::fs::read_dir(fixture.directory_path())
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect();
    paths.sort();
    paths
}

#[test]
fn fresh_reopen_distinguishes_flushed_events_from_buffered_drop() {
    let fixture = Fixture::directory();
    let mut store = fixture.open();
    assert!(matches!(store.load().unwrap(), LoadOutcome::Missing));
    store.begin(discovered(&["a.txt", "nested/b.txt"])).unwrap();
    let initial = store.snapshot().unwrap().clone();
    let checkpoint = initial.checkpoint.as_ref().unwrap();
    uuid::Uuid::parse_str(&checkpoint.generation).unwrap();
    assert_eq!(checkpoint.applied_sequence, 0);
    assert_eq!(checkpoint.scope, fixture.scope);
    let disk = codec::decode(
        &std::fs::read(fixture.base()).unwrap(),
        &fixture.scope,
        false,
        Limits::default(),
    )
    .unwrap();
    assert_eq!(disk, initial);
    assert_eq!(
        store
            .record(
                ItemKey::File("a.txt".into()),
                json!({"status":"in_progress","target":fixture.output("a.txt")})
            )
            .unwrap(),
        1
    );
    assert_eq!(store.flush().unwrap(), 1);
    assert_eq!(
        store
            .record(ItemKey::File("a.txt".into()), completed(&fixture, "a.txt"))
            .unwrap(),
        2
    );
    assert_eq!(
        store.snapshot().unwrap().documents["a.txt"].status,
        Status::Completed
    );
    drop(store);

    let mut reopened = fixture.open();
    let (recovered, _) = loaded(&mut reopened);
    assert_eq!(recovered.documents["a.txt"].status, Status::Failed);
    assert_eq!(
        recovered.documents["a.txt"].target.as_ref(),
        Some(&fixture.output("a.txt"))
    );
    assert_eq!(recovered.documents["nested/b.txt"].status, Status::Pending);
    assert_eq!(recovered.checkpoint.as_ref().unwrap().applied_sequence, 1);
    assert_eq!(
        recovered.checkpoint.as_ref().unwrap().generation,
        checkpoint.generation
    );
    reopened.begin(recovered).unwrap();
    assert_eq!(
        reopened
            .record(ItemKey::File("a.txt".into()), completed(&fixture, "a.txt"))
            .unwrap(),
        2
    );
    assert_eq!(reopened.flush().unwrap(), 2);
    drop(reopened);
    let (final_state, _) = loaded(&mut fixture.open());
    assert_eq!(final_state.documents["a.txt"].status, Status::Completed);
    assert_eq!(
        final_state.documents["a.txt"].output.as_ref(),
        Some(&fixture.output("a.txt"))
    );
}

#[test]
fn legacy_replay_finishes_before_interrupted_items_are_normalized() {
    let fixture = Fixture::directory();
    let mut snapshot = discovered(&["a.txt", "nested/b.txt"]);
    snapshot.documents.get_mut("a.txt").unwrap().status = Status::InProgress;
    snapshot.documents.get_mut("a.txt").unwrap().target = Some(fixture.output("a.txt"));
    let events = [
        event("a.txt", completed(&fixture, "a.txt"), None),
        event(
            "nested/b.txt",
            json!({"status":"in_progress","target":fixture.output("nested/b.txt")}),
            None,
        ),
    ];
    fixture.seed(&snapshot, &journal(&events));
    let (recovered, _) = loaded(&mut fixture.open());
    assert_eq!(recovered.documents["a.txt"].status, Status::Completed);
    assert_eq!(
        recovered.documents["a.txt"].output.as_ref(),
        Some(&fixture.output("a.txt"))
    );
    assert_eq!(recovered.documents["nested/b.txt"].status, Status::Failed);
    assert!(recovered.documents["a.txt"].observations.is_empty());
}

#[test]
fn compacted_checkpoint_rejects_stale_log_regression_and_accepts_next_sequence() {
    let fixture = Fixture::directory();
    let mut store = fixture.open();
    store.begin(discovered(&["a.txt", "nested/b.txt"])).unwrap();
    let generation = store
        .snapshot()
        .unwrap()
        .checkpoint
        .as_ref()
        .unwrap()
        .generation
        .clone();
    store
        .record(
            ItemKey::File("a.txt".into()),
            json!({"status":"in_progress","target":fixture.output("a.txt")}),
        )
        .unwrap();
    store.flush().unwrap();
    store
        .record(ItemKey::File("a.txt".into()), completed(&fixture, "a.txt"))
        .unwrap();
    store.flush().unwrap();
    let mut stale_log = std::fs::read(fixture.journal()).unwrap();
    store.compact().unwrap();
    assert_eq!(
        store
            .snapshot()
            .unwrap()
            .checkpoint
            .as_ref()
            .unwrap()
            .applied_sequence,
        2
    );
    drop(store);
    // Recreate the observable crash window after the new base was published but
    // before the old journal was removed, including an unrelated old writer.
    stale_log.extend(journal(&[
        event(
            "a.txt",
            json!({"status":"failed","error":"old legacy writer"}),
            None,
        ),
        event(
            "a.txt",
            json!({"status":"pending"}),
            Some((GENERATION, 999)),
        ),
        event(
            "nested/b.txt",
            completed(&fixture, "nested/b.txt"),
            Some((&generation, 3)),
        ),
    ]));
    std::fs::write(fixture.journal(), stale_log).unwrap();
    let (recovered, _) = loaded(&mut fixture.open());
    assert_eq!(recovered.documents["a.txt"].status, Status::Completed);
    assert_eq!(
        recovered.documents["nested/b.txt"].status,
        Status::Completed
    );
    assert_eq!(recovered.checkpoint.unwrap().applied_sequence, 3);
}

#[test]
fn fresh_generation_cannot_replay_a_previous_runs_leftover_journal() {
    let fixture = Fixture::directory();
    let mut first = fixture.open();
    first.begin(discovered(&["a.txt", "nested/b.txt"])).unwrap();
    let old_generation = first
        .snapshot()
        .unwrap()
        .checkpoint
        .as_ref()
        .unwrap()
        .generation
        .clone();
    first
        .record(ItemKey::File("a.txt".into()), completed(&fixture, "a.txt"))
        .unwrap();
    first.flush().unwrap();
    let mut old_log = std::fs::read(fixture.journal()).unwrap();
    drop(first);
    let mut next = fixture.open();
    next.begin(discovered(&["a.txt", "nested/b.txt"])).unwrap();
    let new_generation = next
        .snapshot()
        .unwrap()
        .checkpoint
        .as_ref()
        .unwrap()
        .generation
        .clone();
    assert_ne!(new_generation, old_generation);
    assert_eq!(
        next.record(
            ItemKey::File("nested/b.txt".into()),
            completed(&fixture, "nested/b.txt")
        )
        .unwrap(),
        1
    );
    next.flush().unwrap();
    old_log.extend(std::fs::read(fixture.journal()).unwrap());
    drop(next);
    std::fs::write(fixture.journal(), old_log).unwrap();
    let (recovered, _) = loaded(&mut fixture.open());
    assert_eq!(recovered.documents["a.txt"].status, Status::Pending);
    assert_eq!(
        recovered.documents["nested/b.txt"].status,
        Status::Completed
    );
    assert_eq!(
        recovered.checkpoint.as_ref().unwrap().generation,
        new_generation
    );
    assert_eq!(recovered.checkpoint.unwrap().applied_sequence, 1);
}

#[test]
fn merged_new_key_is_checkpointed_before_its_first_event() {
    let fixture = Fixture::directory();
    let mut first = fixture.open();
    first.begin(discovered(&["a.txt"])).unwrap();
    first
        .record(ItemKey::File("a.txt".into()), completed(&fixture, "a.txt"))
        .unwrap();
    first.flush().unwrap();
    drop(first);
    let mut next = fixture.open();
    let (mut existing, _) = loaded(&mut next);
    codec::merge(
        &mut existing,
        &discovered(&["a.txt", "nested/b.txt"]),
        &[],
        Limits::default(),
    )
    .unwrap();
    next.begin(existing).unwrap();
    let durable = codec::decode(
        &std::fs::read(fixture.base()).unwrap(),
        &fixture.scope,
        false,
        Limits::default(),
    )
    .unwrap();
    assert_eq!(durable.documents.len(), 2);
    assert_eq!(durable.documents["a.txt"].status, Status::Completed);
    assert_eq!(durable.documents["nested/b.txt"].status, Status::Pending);
    next.record(
        ItemKey::File("nested/b.txt".into()),
        json!({"status":"in_progress","target":fixture.output("nested/b.txt")}),
    )
    .unwrap();
    next.flush().unwrap();
    drop(next);
    let (recovered, _) = loaded(&mut fixture.open());
    assert_eq!(recovered.documents["a.txt"].status, Status::Completed);
    assert_eq!(recovered.documents["nested/b.txt"].status, Status::Failed);
    assert_eq!(
        recovered.documents["nested/b.txt"].target.as_ref(),
        Some(&fixture.output("nested/b.txt"))
    );
}

#[test]
fn legacy_native_event_mismatch_stops_replay_without_upgrading_the_base() {
    let fixture = Fixture::directory();
    fixture.seed(
        &discovered(&["a.txt", "nested/b.txt"]),
        &journal(&[
            event("a.txt", completed(&fixture, "a.txt"), Some((GENERATION, 1))),
            event("nested/b.txt", completed(&fixture, "nested/b.txt"), None),
        ]),
    );
    let (recovered, warnings) = loaded(&mut fixture.open());
    assert!(!warnings.is_empty());
    assert!(recovered.checkpoint.is_none());
    assert!(
        recovered
            .documents
            .values()
            .all(|entry| entry.status == Status::Pending)
    );
}

#[test]
fn legacy_syntax_errors_skip_but_semantic_and_utf8_errors_stop_at_the_valid_prefix() {
    for invalid_utf8 in [false, true] {
        let fixture = Fixture::directory();
        let mut bytes = journal(&[event("a.txt", completed(&fixture, "a.txt"), None)]);
        if invalid_utf8 {
            bytes.extend_from_slice(b"\xff\n");
        } else {
            bytes.extend_from_slice(b"{broken JSON\n");
            bytes.extend(journal(&[event(
                "nested/b.txt",
                completed(&fixture, "nested/b.txt"),
                None,
            )]));
            bytes.extend(journal(&[event(
                "a.txt",
                json!({"status":"private-invalid-status","error":"DO_NOT_LEAK_TOKEN"}),
                None,
            )]));
        }
        bytes.extend(journal(&[event(
            "c.txt",
            completed(&fixture, "c.txt"),
            None,
        )]));
        fixture.seed(&discovered(&["a.txt", "nested/b.txt", "c.txt"]), &bytes);
        let (recovered, warnings) = loaded(&mut fixture.open());
        assert_eq!(recovered.documents["a.txt"].status, Status::Completed);
        assert_eq!(
            recovered.documents["nested/b.txt"].status,
            if invalid_utf8 {
                Status::Pending
            } else {
                Status::Completed
            }
        );
        assert_eq!(recovered.documents["c.txt"].status, Status::Pending);
        assert!(!warnings.is_empty());
        assert!(
            warnings
                .iter()
                .all(|warning| !warning.contains("DO_NOT_LEAK_TOKEN")
                    && !warning.contains("private-invalid-status"))
        );
    }
}

#[test]
fn native_duplicate_reversed_and_gap_sequences_stop_before_later_valid_events() {
    for sequence_order in [vec![1, 1, 2], vec![1, 2, 1, 3], vec![1, 3, 2], vec![2, 1]] {
        let fixture = Fixture::directory();
        let mut events = Vec::new();
        for (index, sequence) in sequence_order.iter().enumerate() {
            let key = match index {
                0 => "a.txt",
                1 => "nested/b.txt",
                _ => "c.txt",
            };
            events.push(event(
                key,
                completed(&fixture, key),
                Some((GENERATION, *sequence)),
            ));
        }
        fixture.seed(
            &fenced(
                discovered(&["a.txt", "nested/b.txt", "c.txt"]),
                &fixture.scope,
                0,
            ),
            &journal(&events),
        );
        let (recovered, warnings) = loaded(&mut fixture.open());
        let accepted = if sequence_order[0] != 1 {
            0
        } else if sequence_order[1] == 2 {
            2
        } else {
            1
        };
        assert_eq!(
            recovered.checkpoint.as_ref().unwrap().applied_sequence,
            accepted
        );
        assert_eq!(
            recovered.documents["a.txt"].status,
            if accepted >= 1 {
                Status::Completed
            } else {
                Status::Pending
            }
        );
        assert_eq!(
            recovered.documents["nested/b.txt"].status,
            if accepted >= 2 {
                Status::Completed
            } else {
                Status::Pending
            }
        );
        assert_eq!(recovered.documents["c.txt"].status, Status::Pending);
        assert!(
            !warnings.is_empty(),
            "missing sequence diagnostic for {sequence_order:?}"
        );
    }
}

#[test]
fn unknown_native_key_stops_prefix_while_unknown_legacy_key_is_ignored() {
    for native in [false, true] {
        let fixture = Fixture::directory();
        let snapshot = discovered(&["a.txt", "nested/b.txt"]);
        let snapshot = if native {
            fenced(snapshot, &fixture.scope, 0)
        } else {
            snapshot
        };
        let fence = |sequence| native.then_some((GENERATION, sequence));
        let events = [
            event("a.txt", completed(&fixture, "a.txt"), fence(1)),
            event(
                "not-discovered.txt",
                completed(&fixture, "not-discovered.txt"),
                fence(2),
            ),
            event(
                "nested/b.txt",
                completed(&fixture, "nested/b.txt"),
                fence(3),
            ),
        ];
        fixture.seed(&snapshot, &journal(&events));
        let (recovered, warnings) = loaded(&mut fixture.open());
        assert_eq!(recovered.documents.len(), 2);
        assert_eq!(recovered.documents["a.txt"].status, Status::Completed);
        assert_eq!(
            recovered.documents["nested/b.txt"].status,
            if native {
                Status::Pending
            } else {
                Status::Completed
            }
        );
        if native {
            assert_eq!(recovered.checkpoint.unwrap().applied_sequence, 1);
            assert!(!warnings.is_empty());
        }
    }
}

#[test]
fn begin_repairs_torn_suffix_before_appending_new_events() {
    let fixture = Fixture::directory();
    let mut bytes = journal(&[event("a.txt", completed(&fixture, "a.txt"), None)]);
    let complete = codec::encode_event(&event(
        "nested/b.txt",
        completed(&fixture, "nested/b.txt"),
        None,
    ))
    .unwrap();
    bytes.extend_from_slice(&complete[..complete.len() / 2]);
    fixture.seed(&discovered(&["a.txt", "nested/b.txt"]), &bytes);
    let mut store = fixture.open();
    let (recovered, warnings) = loaded(&mut store);
    assert_eq!(recovered.documents["a.txt"].status, Status::Completed);
    assert_eq!(recovered.documents["nested/b.txt"].status, Status::Pending);
    assert!(!warnings.is_empty());
    store.begin(recovered).unwrap();
    store
        .record(
            ItemKey::File("nested/b.txt".into()),
            completed(&fixture, "nested/b.txt"),
        )
        .unwrap();
    store.flush().unwrap();
    drop(store);
    let (final_state, warnings) = loaded(&mut fixture.open());
    assert!(
        final_state
            .documents
            .values()
            .all(|entry| entry.status == Status::Completed)
    );
    assert!(
        warnings.is_empty(),
        "repaired journal still has replay errors: {warnings:?}"
    );
}

#[test]
fn explicit_foreign_scope_is_neither_quarantined_nor_overwritten_by_begin() {
    for native in [false, true] {
        let fixture = Fixture::directory();
        let other = Fixture::directory();
        let mut foreign = discovered(&["a.txt"]);
        if native {
            foreign = fenced(foreign, &other.scope, 0);
        } else {
            foreign.options = serde_json::from_value(
                json!({"input_dir":other.scope.input,"output_dir":other.scope.output}),
            )
            .unwrap();
        }
        let bytes = codec::encode(&foreign, &other.scope, false, Limits::default()).unwrap();
        std::fs::create_dir_all(fixture.directory_path()).unwrap();
        std::fs::write(fixture.base(), &bytes).unwrap();
        let sidecar = b"foreign sidecar must remain untouched\n";
        std::fs::write(fixture.journal(), sidecar).unwrap();
        let mut store = fixture.open();
        let before = same_directory_files(&fixture);
        assert!(matches!(store.load(), Err(Error::ForeignScope(_))));
        assert!(matches!(
            store.begin(discovered(&["a.txt"])),
            Err(Error::ForeignScope(_))
        ));
        assert_eq!(std::fs::read(fixture.base()).unwrap(), bytes);
        assert_eq!(std::fs::read(fixture.journal()).unwrap(), sidecar);
        assert_eq!(same_directory_files(&fixture), before);
    }
}

#[test]
fn rejected_unknown_oversized_and_unsafe_events_do_not_mutate_or_consume_sequence() {
    let fixture = Fixture::directory();
    let limits = Limits {
        line_bytes: 1024,
        ..Limits::default()
    };
    let mut store = fixture.open_with(limits);
    store.begin(discovered(&["a.txt", "nested/b.txt"])).unwrap();
    let initial = store.snapshot().unwrap().clone();
    let base = std::fs::read(fixture.base()).unwrap();
    assert!(
        store
            .record(
                ItemKey::File("unknown.txt".into()),
                completed(&fixture, "unknown.txt")
            )
            .is_err()
    );
    assert!(matches!(
        store.record(
            ItemKey::File("a.txt".into()),
            json!({"status":"failed","error":"x".repeat(2048)})
        ),
        Err(Error::Limit(_))
    ));
    for path in [
        fixture.root.path().join("outside.md"),
        fixture.scope.output.join(".markitai/secret.md"),
        fixture.output("nested/wrong-parent.txt"),
    ] {
        assert!(matches!(
            store.record(
                ItemKey::File("a.txt".into()),
                json!({"status":"completed","output":path})
            ),
            Err(Error::ForeignScope(_))
        ));
        assert_eq!(store.snapshot().unwrap(), &initial);
    }
    assert_eq!(std::fs::read(fixture.base()).unwrap(), base);
    assert_eq!(store.flush().unwrap(), 0);
    assert_eq!(
        store
            .record(ItemKey::File("a.txt".into()), completed(&fixture, "a.txt"))
            .unwrap(),
        1
    );
    assert_eq!(store.flush().unwrap(), 1);
}

#[test]
fn corrupt_base_and_sidecar_are_quarantined_byte_exact_without_replacing_prior_evidence() {
    let fixture = Fixture::directory();
    let corrupt = b"{\"secret\":\"DO_NOT_LEAK_TOKEN\"";
    let sidecar = b"\xfftruncated journal evidence\n";
    for attempt in 1..=2 {
        std::fs::create_dir_all(fixture.directory_path()).unwrap();
        std::fs::write(fixture.base(), corrupt).unwrap();
        std::fs::write(fixture.journal(), sidecar).unwrap();
        let mut store = fixture.open();
        match store.load().unwrap() {
            LoadOutcome::Corrupt { reason } => assert!(!reason.contains("DO_NOT_LEAK_TOKEN")),
            other => panic!("expected corrupt status, received {other:?}"),
        }
        assert_eq!(std::fs::read(fixture.base()).unwrap(), corrupt);
        assert_eq!(std::fs::read(fixture.journal()).unwrap(), sidecar);
        store.begin(discovered(&["a.txt"])).unwrap();
        let fresh = codec::decode(
            &std::fs::read(fixture.base()).unwrap(),
            &fixture.scope,
            false,
            Limits::default(),
        )
        .unwrap();
        assert_eq!(fresh.documents["a.txt"].status, Status::Pending);
        assert!(fresh.checkpoint.is_some());
        let evidence: Vec<_> = same_directory_files(&fixture)
            .into_iter()
            .filter(|path| {
                path != &fixture.base() && path != &fixture.journal() && path != &fixture.lock()
            })
            .map(|path| std::fs::read(path).unwrap())
            .collect();
        assert_eq!(
            evidence
                .iter()
                .filter(|bytes| bytes.as_slice() == corrupt)
                .count(),
            attempt
        );
        assert_eq!(
            evidence
                .iter()
                .filter(|bytes| bytes.as_slice() == sidecar)
                .count(),
            attempt
        );
        assert_eq!(evidence.len(), 2 * attempt);
    }
}

#[test]
fn bounded_journal_compaction_preserves_all_acknowledged_sequences() {
    let fixture = Fixture::directory();
    let limits = Limits {
        base_bytes: 4096,
        journal_bytes: 1024,
        line_bytes: 512,
        entries: 2,
    };
    let mut store = fixture.open_with(limits);
    store.begin(discovered(&["a.txt"])).unwrap();
    for sequence in 1..=20 {
        assert_eq!(store.record(ItemKey::File("a.txt".into()), json!({"status":"failed","error":format!("attempt {sequence}: {}", "x".repeat(160))})).unwrap(), sequence);
    }
    assert_eq!(store.flush().unwrap(), 20);
    assert!(std::fs::metadata(fixture.base()).unwrap().len() <= limits.base_bytes as u64);
    if fixture.journal().exists() {
        assert!(std::fs::metadata(fixture.journal()).unwrap().len() <= limits.journal_bytes as u64);
    }
    drop(store);
    let (recovered, warnings) = loaded(&mut fixture.open_with(limits));
    assert!(warnings.is_empty());
    assert_eq!(recovered.checkpoint.unwrap().applied_sequence, 20);
    assert_eq!(
        recovered.documents["a.txt"].error.as_deref(),
        Some(format!("attempt 20: {}", "x".repeat(160)).as_str())
    );
}

#[test]
fn absent_base_does_not_adopt_an_orphan_journal_and_oversized_base_is_corrupt() {
    let fixture = Fixture::directory();
    std::fs::create_dir_all(fixture.directory_path()).unwrap();
    std::fs::write(
        fixture.journal(),
        journal(&[event("a.txt", completed(&fixture, "a.txt"), None)]),
    )
    .unwrap();
    let mut store = fixture.open();
    assert!(matches!(store.load().unwrap(), LoadOutcome::Missing));
    store.begin(discovered(&["a.txt"])).unwrap();
    drop(store);
    let (recovered, _) = loaded(&mut fixture.open());
    assert_eq!(recovered.documents["a.txt"].status, Status::Pending);
    std::fs::write(fixture.base(), vec![b' '; 1025]).unwrap();
    let limits = Limits {
        base_bytes: 1024,
        ..Limits::default()
    };
    let mut store = fixture.open_with(limits);
    assert!(matches!(store.load().unwrap(), LoadOutcome::Corrupt { .. }));
    assert_eq!(std::fs::metadata(fixture.base()).unwrap().len(), 1025);
}

#[cfg(unix)]
#[test]
fn original_symlink_spelling_is_enforced_even_after_scope_resolution() {
    let fixture = Fixture::directory();
    let link = fixture.root.path().join("output-link");
    std::os::unix::fs::symlink(&fixture.scope.output, &link).unwrap();
    let scope = Scope::new(Mode::Directory, &fixture.scope.input, &link).unwrap();
    assert_eq!(scope.output, fixture.scope.output);
    assert!(StateStore::open(scope.clone(), HASH, false, Limits::default()).is_err());
    assert!(!fixture.directory_path().exists());
    let mut allowed = StateStore::open(scope.clone(), HASH, true, Limits::default()).unwrap();
    allowed.begin(discovered(&["a.txt"])).unwrap();
    drop(allowed);
    assert!(StateStore::open(scope.clone(), HASH, false, Limits::default()).is_err());
    let (recovered, _) =
        loaded(&mut StateStore::open(scope, HASH, true, Limits::default()).unwrap());
    assert_eq!(recovered.documents["a.txt"].status, Status::Pending);
}

struct ChildGuard(Child);
impl Drop for ChildGuard {
    fn drop(&mut self) {
        if self.0.try_wait().ok().flatten().is_none() {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
}
fn child(root: &Path, action: &str) -> ChildGuard {
    let mut command = Command::new(std::env::current_exe().unwrap());
    command.env_clear();
    for name in ["PATH", "SYSTEMROOT", "TMPDIR", "TEMP", "TMP"] {
        if let Some(value) = std::env::var_os(name) {
            command.env(name, value);
        }
    }
    ChildGuard(
        command
            .current_dir(root)
            .env("MARKITAI_HOME", root.join("home"))
            .env("MARKITAI_TEST_STATE_ROOT", root)
            .env("MARKITAI_TEST_STATE_ACTION", action)
            .args([
                "--exact",
                "run_state::tests::process_lock_child",
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap(),
    )
}
fn wait_exit(child: &mut ChildGuard) {
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        if let Some(status) = child.0.try_wait().unwrap() {
            assert!(status.success(), "recovery child exited with {status}");
            return;
        }
        assert!(Instant::now() < deadline, "recovery child did not finish");
        std::thread::sleep(Duration::from_millis(5));
    }
}
fn wait_ready(child: &mut ChildGuard, path: &Path) {
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        if path.exists() {
            return;
        }
        assert!(
            child.0.try_wait().unwrap().is_none(),
            "recovery lock holder exited before readiness"
        );
        assert!(
            Instant::now() < deadline,
            "recovery lock holder did not become ready"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn exclusive_lock_competes_across_processes_and_kill_releases_the_same_inode() {
    let fixture = Fixture::directory();
    let mut owner = fixture.open();
    owner.begin(discovered(&["a.txt"])).unwrap();
    let generation = owner
        .snapshot()
        .unwrap()
        .checkpoint
        .as_ref()
        .unwrap()
        .generation
        .clone();
    let original_inode = markitai_core::platform::status(&fixture.lock())
        .unwrap()
        .id();
    let mut contender = child(fixture.root.path(), "expect-busy");
    wait_exit(&mut contender);
    assert_eq!(
        std::fs::read(fixture.root.path().join("busy-confirmed")).unwrap(),
        b"busy"
    );
    drop(owner);
    let mut holder = child(fixture.root.path(), "hold");
    wait_ready(&mut holder, &fixture.root.path().join("lock-ready"));
    assert!(matches!(
        StateStore::open(fixture.scope.clone(), HASH, false, Limits::default()),
        Err(Error::Busy)
    ));
    holder.0.kill().unwrap();
    assert!(!holder.0.wait().unwrap().success());
    let mut recovered = fixture.open();
    let (snapshot, _) = loaded(&mut recovered);
    assert_eq!(snapshot.checkpoint.unwrap().generation, generation);
    assert_eq!(
        markitai_core::platform::status(&fixture.lock())
            .unwrap()
            .id(),
        original_inode
    );
}

#[test]
#[ignore = "invoked only as an isolated child by the process-lock acceptance test"]
fn process_lock_child() {
    let root =
        PathBuf::from(std::env::var_os("MARKITAI_TEST_STATE_ROOT").expect("private test root"));
    let scope = Scope::new(Mode::Directory, &root.join("input"), &root.join("output")).unwrap();
    match std::env::var("MARKITAI_TEST_STATE_ACTION")
        .unwrap()
        .as_str()
    {
        "expect-busy" => {
            assert!(matches!(
                StateStore::open(scope, HASH, false, Limits::default()),
                Err(Error::Busy)
            ));
            std::fs::write(root.join("busy-confirmed"), b"busy").unwrap();
        }
        "hold" => {
            let _guard = StateStore::open(scope, HASH, false, Limits::default()).unwrap();
            std::fs::write(root.join("lock-ready"), b"held").unwrap();
            // The pipe remains open in the parent. This child is intentionally
            // killed so OS release, rather than Rust Drop, is what unlocks it.
            let mut byte = [0u8; 1];
            let _ = std::io::stdin().read_exact(&mut byte);
        }
        action => panic!("unknown test-only child action: {action}"),
    }
}

#[test]
fn url_list_roundtrip_and_merge_preserve_encounter_order_and_raw_named_identity() {
    let mut fixture = Fixture::directory();
    let list = fixture.scope.input.join("sources.urls");
    std::fs::write(&list, b"# No fetches are performed by state storage.\n").unwrap();
    fixture.scope = Scope::new(Mode::UrlList, &list, &fixture.scope.output).unwrap();
    let z = "https://example.invalid/z raw-z";
    let a = "https://example.invalid/a raw-a.md";
    let m = "https://example.invalid/m later";
    let entry = |url: &str| Entry {
        source_file: Some(list.to_string_lossy().into_owned()),
        url: Some(url.into()),
        ..Entry::default()
    };
    let mut snapshot = Snapshot::default();
    snapshot
        .options
        .insert("z-extension".into(), json!("first custom field"));
    snapshot.options.insert("concurrency".into(), json!(2));
    snapshot
        .options
        .insert("a-extension".into(), json!("second custom field"));
    let mut first = entry("https://example.invalid/z");
    first.status = Status::Completed;
    first.output = Some(fixture.scope.output.join("raw-z.md"));
    snapshot.urls.insert(z.into(), first);
    snapshot
        .urls
        .insert(a.into(), entry("https://example.invalid/a"));
    let mut store = fixture.open();
    store.begin(snapshot).unwrap();
    drop(store);
    let mut reopened = fixture.open();
    let (mut loaded, _) = loaded(&mut reopened);
    assert_eq!(
        loaded.urls.keys().map(String::as_str).collect::<Vec<_>>(),
        [z, a]
    );
    assert_eq!(loaded.urls[z].status, Status::Completed);
    assert_eq!(
        loaded
            .options
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["concurrency", "z-extension", "a-extension"]
    );
    assert_eq!(
        loaded.urls[a].url.as_deref(),
        Some("https://example.invalid/a")
    );
    let raw = std::fs::read_to_string(fixture.base()).unwrap();
    let z_offset = raw.find(&serde_json::to_string(z).unwrap()).unwrap();
    let a_offset = raw.find(&serde_json::to_string(a).unwrap()).unwrap();
    assert!(
        z_offset < a_offset,
        "URL serialization changed encounter order"
    );
    let custom_z = raw.find("\"z-extension\"").unwrap();
    let custom_a = raw.find("\"a-extension\"").unwrap();
    assert!(
        custom_z < custom_a,
        "custom option fields changed encounter order"
    );
    let mut discovered = Snapshot::default();
    discovered
        .urls
        .insert(m.into(), entry("https://example.invalid/m"));
    discovered
        .urls
        .insert(a.into(), entry("https://example.invalid/a"));
    discovered
        .urls
        .insert(z.into(), entry("https://example.invalid/z"));
    codec::merge(
        &mut loaded,
        &discovered,
        &[m.into(), a.into(), z.into()],
        Limits::default(),
    )
    .unwrap();
    assert_eq!(
        loaded.urls.keys().map(String::as_str).collect::<Vec<_>>(),
        [z, a, m]
    );
    assert_eq!(loaded.urls[z].status, Status::Completed);
    reopened.begin(loaded).unwrap();
    drop(reopened);
    let mut final_store = fixture.open();
    let (final_snapshot, _) = match final_store.load().unwrap() {
        LoadOutcome::Loaded { snapshot, warnings } => (*snapshot, warnings),
        other => panic!("expected merged URL-list checkpoint, received {other:?}"),
    };
    assert_eq!(
        final_snapshot
            .urls
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        [z, a, m]
    );
    assert_eq!(final_snapshot.urls[z].status, Status::Completed);
    assert_eq!(final_snapshot.urls[m].status, Status::Pending);
}
