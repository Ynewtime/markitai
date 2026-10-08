//! Store decisions run on every platform; mode bits and substituted symbolic
//! links are Unix fixtures.
use super::*;
use markitai_core::provider_batch::BatchStatus;
use serde_json::json;
#[cfg(unix)]
use std::os::unix::fs::{PermissionsExt, symlink};

struct Fixture {
    _directory: tempfile::TempDir,
    input: PathBuf,
    output: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let root = platform::canonicalize(directory.path()).unwrap();
        let input = root.join("input");
        let output = root.join("output");
        fs::create_dir(&input).unwrap();
        fs::create_dir(&output).unwrap();
        Self {
            _directory: directory,
            input,
            output,
        }
    }
    fn job(&self, names: &[&str]) -> NewJob {
        let id = uuid::Uuid::new_v4().to_string();
        let items = names.iter().enumerate().map(|(index, name)| {
            let key = format!("{name}.txt");
            NewItem {
                custom_id: format!("item_{index}"), source: self.input.join(&key).to_string_lossy().into_owned(),
                key: key.clone(), base: format!("{name}.md").into(), enhanced: format!("{name}.llm.md").into(),
                base_sha256: sha256_hex(b"base body"),
                owner: Owner {generation: id.clone(), mode: "directory".into(), input: self.input.clone(),
                    output: self.output.clone(), kind: "file".into(), key},
                plan: json!({"frozen_content":format!("original {name}"),"model":"fixture-model"}),
            }
        }).collect();
        NewJob {
            id,
            input_root: self.input.clone(),
            endpoint: Endpoint {
                provider: "openai".into(),
                api_base: "http://127.0.0.1:1234/v1".into(),
                model: "fixture-model".into(),
            },
            items,
        }
    }
    fn create(&self, names: &[&str]) -> Store {
        Store::create(&self.output, self.job(names), false, Limits::default()).unwrap()
    }
    fn reopen(&self, id: &str) -> Store {
        Store::open_by_id(&self.output, id, false, Limits::default()).unwrap()
    }
}
fn uploaded(store: &mut Store) {
    let blob = store
        .save_requests(&mut b"frozen validated request bytes\n".as_slice())
        .unwrap();
    store
        .save_uploaded(UploadedInput {
            file_id: "file_input".into(),
            model: "fixture-model".into(),
            custom_ids: store
                .state()
                .items
                .iter()
                .map(|item| item.custom_id.clone())
                .collect(),
            bytes: blob.bytes,
            sha256: blob.sha256,
        })
        .unwrap();
}
fn submitted(store: &mut Store, status: BatchStatus) {
    uploaded(store);
    store.mark_creating().unwrap();
    store
        .bind_batch(Batch {
            id: "batch_fixture".into(),
            input_file_id: "file_input".into(),
            status,
            output_file_id: Some("file_output".into()),
            error_file_id: None,
            total: Some(store.state().items.len() as u64),
            completed: None,
            failed: None,
        })
        .unwrap();
}
fn usage(priced: bool) -> ConversionUsage {
    let model = if priced {
        json!({"requests":1,"input_tokens":10,"output_tokens":2,"cost_usd":0.02,"cached_input_tokens":3,
            "priced_requests":1,"unpriced_requests":0,"cost_status":"complete","pricing_snapshot":"fixture-tariff"})
    } else {
        json!({"requests":1,"input_tokens":10,"output_tokens":2,"cost_usd":0.0,"cached_input_tokens":1,
            "priced_requests":0,"unpriced_requests":1,"cost_status":"unknown"})
    };
    ConversionUsage {
        requests: 1,
        input_tokens: 10,
        output_tokens: 2,
        cost_usd: if priced { 0.02 } else { 0.0 },
        by_model: serde_json::from_value(json!({"fixture-model":model})).unwrap(),
    }
}
fn raw(id: &str, text: &str) -> Vec<u8> {
    serde_json::to_vec(&json!({"custom_id":id,"http_status":200,"body":{"text":text}})).unwrap()
}
fn published(store: &Store, index: usize) -> Published {
    Published {
        path: store.state().items[index].enhanced.clone(),
        bytes: 8,
        sha256: sha256_hex(b"enhanced"),
        receipt_sha256: sha256_hex(b"canonical receipt"),
    }
}

#[test]
fn prepared_job_retains_frozen_plan_after_original_source_disappears() {
    let fixture = Fixture::new();
    fs::write(fixture.input.join("notes.llm.txt"), "source").unwrap();
    let store = fixture.create(&["notes.llm"]);
    let id = store.state().id.clone();
    assert_eq!(store.state().items[0].base, Path::new("notes.llm.md"));
    assert_eq!(
        store.state().items[0].enhanced,
        Path::new("notes.llm.llm.md")
    );
    let directory = store.directory().to_owned();
    drop(store);
    fs::remove_dir_all(&fixture.input).unwrap();
    let restored = fixture.reopen(&id);
    assert_eq!(
        restored.read_plan("item_0").unwrap()["frozen_content"],
        "original notes.llm"
    );
    for path in [directory.clone(), directory.join("state.json")] {
        let status = platform::status(&path).unwrap();
        assert!(
            status.private() && status.owned_by_current_user(),
            "{path:?}"
        );
    }
    #[cfg(unix)]
    {
        assert_eq!(
            fs::metadata(&directory).unwrap().permissions().mode() & 0o777,
            0o700
        );
        assert_eq!(
            fs::metadata(directory.join("state.json"))
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
    }
    let pending = Store::pending(&fixture.output, false, Limits::default()).unwrap();
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].input_root, fixture.input);
    assert_eq!(pending[0].id, id);
}

#[test]
fn creating_reopen_remains_uncertain_and_cannot_submit_again() {
    let fixture = Fixture::new();
    let mut store = fixture.create(&["a"]);
    uploaded(&mut store);
    let id = store.mark_creating().unwrap();
    drop(store);
    let mut restored = fixture.reopen(&id);
    assert_eq!(restored.state().phase, Phase::Creating);
    assert!(matches!(restored.mark_creating(), Err(Error::Conflict)));
    restored.mark_uncertain().unwrap();
    assert!(matches!(
        Store::create(
            &fixture.output,
            fixture.job(&["a"]),
            false,
            Limits::default()
        ),
        Err(Error::Overlap)
    ));
    assert_eq!(
        Store::pending(&fixture.output, false, Limits::default()).unwrap()[0].phase,
        Phase::CreateUncertain
    );
}

#[test]
fn explicit_rejection_preserves_evidence_but_releases_pending_family() {
    let fixture = Fixture::new();
    let mut store = fixture.create(&["a"]);
    uploaded(&mut store);
    store.mark_creating().unwrap();
    let request = fs::read(store.request_path().unwrap()).unwrap();
    store.mark_rejected().unwrap();
    store.mark_rejected().unwrap();
    assert!(store.mark_creating().is_err());
    assert_eq!(fs::read(store.request_path().unwrap()).unwrap(), request);
    assert!(
        Store::pending(&fixture.output, false, Limits::default())
            .unwrap()
            .is_empty()
    );
    let replacement = Store::create(
        &fixture.output,
        fixture.job(&["a"]),
        false,
        Limits::default(),
    )
    .unwrap();
    assert_ne!(replacement.state().id, store.state().id);
}

#[test]
fn preparation_lock_serializes_preflight_through_creation_without_index_deadlock() {
    let fixture = Fixture::new();
    let guard = Store::lock_preparation(&fixture.output, false).unwrap();
    let output = fixture.output.clone();
    let contender = std::thread::spawn(move || {
        matches!(Store::lock_preparation(&output, false), Err(Error::Busy))
    });
    assert!(contender.join().unwrap());
    assert!(
        Store::pending(&fixture.output, false, Limits::default())
            .unwrap()
            .is_empty()
    );
    let store = fixture.create(&["a"]);
    guard.validate().unwrap();
    assert_eq!(
        Store::pending(&fixture.output, false, Limits::default())
            .unwrap()
            .len(),
        1
    );
    drop(guard);
    Store::lock_preparation(&fixture.output, false)
        .unwrap()
        .validate()
        .unwrap();
    assert_eq!(store.state().phase, Phase::Prepared);
}

#[test]
fn collectors_hold_one_stable_job_lock_and_reject_replaced_lock_paths() {
    let fixture = Fixture::new();
    let store = fixture.create(&["a"]);
    let id = store.state().id.clone();
    assert!(matches!(
        Store::open_by_id(&fixture.output, &id, false, Limits::default()),
        Err(Error::Busy)
    ));
    let lock = store.directory().join("job.lock");
    fs::rename(&lock, store.directory().join("old.lock")).unwrap();
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&lock)
        .unwrap();
    #[cfg(unix)]
    file.set_permissions(fs::Permissions::from_mode(0o600))
        .unwrap();
    drop(file);
    assert!(matches!(store.validate(), Err(Error::Conflict)));
}

#[test]
fn missing_batch_locator_is_rebuilt_from_one_committed_state() {
    let fixture = Fixture::new();
    let mut store = fixture.create(&["a"]);
    submitted(&mut store, BatchStatus::InProgress);
    let id = store.state().id.clone();
    let locator = fixture
        .output
        .join(".markitai/provider-batches/by-id/batch_fixture.json");
    fs::remove_file(&locator).unwrap();
    drop(store);
    let restored =
        Store::open_by_batch(&fixture.output, "batch_fixture", false, Limits::default()).unwrap();
    assert_eq!(restored.state().id, id);
    let index: Value = serde_json::from_slice(&fs::read(&locator).unwrap()).unwrap();
    assert_eq!(index["job_id"], id);
    assert!(platform::status(&locator).unwrap().private());
    #[cfg(unix)]
    assert_eq!(
        fs::metadata(locator).unwrap().permissions().mode() & 0o777,
        0o600
    );
}

#[test]
fn result_ledger_is_immutable_idempotent_and_keeps_pricing_coverage() {
    let fixture = Fixture::new();
    let mut store = fixture.create(&["a", "b"]);
    submitted(&mut store, BatchStatus::Completed);
    let first = raw("item_0", "one");
    assert!(
        store
            .record_result("item_0", &first, usage(true))
            .unwrap()
            .inserted
    );
    let replay = store.record_result("item_0", &first, usage(false)).unwrap();
    assert!(!replay.inserted);
    assert_eq!(replay.usage.cost_usd, 0.02);
    assert!(matches!(
        store.record_result("item_0", &raw("item_0", "changed"), usage(true)),
        Err(Error::Conflict)
    ));
    assert!(
        store
            .record_result("unknown", &raw("unknown", "text"), usage(true))
            .is_err()
    );
    store
        .record_result("item_1", &raw("item_1", "two"), usage(false))
        .unwrap();
    let id = store.state().id.clone();
    drop(store);
    let restored = fixture.reopen(&id);
    assert_eq!(restored.read_result("item_0").unwrap().unwrap(), first);
    let totals = restored.usage().unwrap();
    assert_eq!(
        (totals.requests, totals.input_tokens, totals.output_tokens),
        (2, 20, 4)
    );
    assert_eq!(totals.cost_usd, 0.02);
    let model = &totals.by_model["fixture-model"];
    assert_eq!(model["cached_input_tokens"], 4);
    assert_eq!(model["priced_requests"], 1);
    assert_eq!(model["unpriced_requests"], 1);
    assert_eq!(model["cost_status"], "partial");
    assert_eq!(model["pricing_snapshot"], "fixture-tariff");
}

#[test]
fn failed_result_state_commit_preserves_old_state_and_reuses_exact_orphan_blob() {
    let fixture = Fixture::new();
    let mut store = fixture.create(&["a"]);
    submitted(&mut store, BatchStatus::Completed);
    let state_path = store.directory().join("state.json");
    let before = fs::read(&state_path).unwrap();
    let id = store.state().id.clone();
    let response = raw("item_0", "paid but malformed text");
    store.fault = Some(Fault::BeforeStateReplace);
    assert!(
        store
            .record_result("item_0", &response, usage(true))
            .is_err()
    );
    assert_eq!(fs::read(&state_path).unwrap(), before);
    assert!(store.state().items[0].result.is_none());
    drop(store);
    let mut restored = fixture.reopen(&id);
    assert!(
        restored
            .record_result("item_0", &response, usage(true))
            .unwrap()
            .inserted
    );
    assert_eq!(restored.usage().unwrap().requests, 1);
}

#[test]
fn state_rename_before_sync_requires_reopen_and_never_recreates_paid_job() {
    let fixture = Fixture::new();
    let mut store = fixture.create(&["a"]);
    uploaded(&mut store);
    let id = store.state().id.clone();
    store.fault = Some(Fault::AfterStateReplace);
    assert!(matches!(store.mark_creating(), Err(Error::Durability)));
    assert!(matches!(store.mark_creating(), Err(Error::Durability)));
    drop(store);
    let mut restored = fixture.reopen(&id);
    assert_eq!(restored.state().phase, Phase::Creating);
    assert!(matches!(restored.mark_creating(), Err(Error::Conflict)));
}

#[test]
fn partial_terminal_results_keep_pending_until_each_publication_is_finalized() {
    let fixture = Fixture::new();
    let mut store = fixture.create(&["a", "b"]);
    submitted(&mut store, BatchStatus::Expired);
    store
        .record_result("item_0", &raw("item_0", "paid result"), usage(true))
        .unwrap();
    store
        .mark_finalized("item_0", published(&store, 0))
        .unwrap();
    store
        .mark_failed("item_1", "No supplier result for this request")
        .unwrap();
    assert!(store.finish().is_err());
    let id = store.state().id.clone();
    drop(store);
    let mut restored = fixture.reopen(&id);
    assert_eq!(
        restored.state().items[1].error.as_deref(),
        Some("No supplier result for this request")
    );
    let mut fallback = published(&restored, 1);
    fallback.path = restored.state().items[1].base.clone();
    restored.mark_finalized("item_1", fallback.clone()).unwrap();
    restored.mark_finalized("item_1", fallback).unwrap();
    restored.finish().unwrap();
    assert!(
        Store::pending(&fixture.output, false, Limits::default())
            .unwrap()
            .is_empty()
    );
    assert_eq!(restored.usage().unwrap().requests, 1);
}

#[test]
fn bounded_requests_and_tampered_plan_fail_without_advancing_state() {
    let fixture = Fixture::new();
    let limits = Limits {
        request_bytes: 8,
        ..Limits::default()
    };
    let mut store = Store::create(&fixture.output, fixture.job(&["a"]), false, limits).unwrap();
    let before = fs::read(store.directory().join("state.json")).unwrap();
    assert!(store.save_requests(&mut b"123456789".as_slice()).is_err());
    assert!(!store.directory().join("requests.jsonl").exists());
    assert_eq!(
        fs::read(store.directory().join("state.json")).unwrap(),
        before
    );
    let id = store.state().id.clone();
    let plan = store.directory().join("plans/0.json");
    fs::write(&plan, b"{}").unwrap();
    drop(store);
    assert!(Store::open_by_id(&fixture.output, &id, false, limits).is_err());
}

#[cfg(unix)]
#[test]
fn internal_symlinks_public_evidence_and_foreign_output_families_are_rejected() {
    let fixture = Fixture::new();
    let mut malformed = fixture.job(&["a"]);
    malformed.items[0].enhanced = "../elsewhere.llm.md".into();
    assert!(Store::create(&fixture.output, malformed, false, Limits::default()).is_err());
    let store = fixture.create(&["a"]);
    let id = store.state().id.clone();
    let plan = store.directory().join("plans/0.json");
    let original = store.directory().join("saved-plan.json");
    fs::rename(&plan, &original).unwrap();
    symlink(&original, &plan).unwrap();
    assert!(store.read_plan("item_0").is_err());
    fs::remove_file(&plan).unwrap();
    fs::rename(&original, &plan).unwrap();
    fs::set_permissions(&plan, fs::Permissions::from_mode(0o644)).unwrap();
    drop(store);
    assert!(Store::open_by_id(&fixture.output, &id, false, Limits::default()).is_err());
}

#[test]
fn frozen_selection_survives_deleted_input_and_keeps_request_bytes_and_identity() {
    let fixture = Fixture::new();
    let mut store = fixture.create(&["a"]);
    let requests = b"exact old requests\n";
    store.save_requests(&mut requests.as_slice()).unwrap();
    let id = store.state().id.clone();
    drop(store);
    fs::remove_dir_all(&fixture.input).unwrap();
    let preparation = Store::lock_preparation(&fixture.output, false).unwrap();
    let restored = preparation
        .open_frozen(Some(&fixture.input), Limits::default())
        .unwrap()
        .unwrap();
    assert_eq!(restored.state().id, id);
    assert_eq!(restored.resume_step().unwrap(), ResumeStep::Upload);
    assert_eq!(
        fs::read(restored.request_path().unwrap()).unwrap(),
        requests
    );
    assert_eq!(
        restored.read_plan("item_0").unwrap()["frozen_content"],
        "original a"
    );
    preparation.validate().unwrap();
}

#[test]
fn frozen_selection_requires_one_job_correct_scope_and_an_exclusive_collector() {
    let fixture = Fixture::new();
    let preparation = Store::lock_preparation(&fixture.output, false).unwrap();
    assert!(
        preparation
            .open_frozen(None, Limits::default())
            .unwrap()
            .is_none()
    );
    let first = fixture.create(&["a"]);
    assert!(matches!(
        preparation.open_frozen(None, Limits::default()),
        Err(Error::Busy)
    ));
    drop(first);
    assert!(matches!(
        preparation.open_frozen(Some(&fixture.output), Limits::default()),
        Err(Error::Conflict)
    ));
    let restored = preparation
        .open_frozen(None, Limits::default())
        .unwrap()
        .unwrap();
    assert_eq!(restored.resume_step().unwrap(), ResumeStep::PrepareRequests);
    drop(restored);
    let second = fixture.create(&["b"]);
    drop(second);
    assert!(
        preparation
            .open_frozen(Some(&fixture.input), Limits::default())
            .is_err()
    );
}

#[test]
fn resume_step_cannot_turn_uncertain_creation_into_a_new_create() {
    let fixture = Fixture::new();
    let mut store = fixture.create(&["a"]);
    assert_eq!(store.resume_step().unwrap(), ResumeStep::PrepareRequests);
    store
        .save_requests(&mut b"frozen validated request bytes\n".as_slice())
        .unwrap();
    assert_eq!(store.resume_step().unwrap(), ResumeStep::Upload);
    uploaded(&mut store);
    assert_eq!(store.resume_step().unwrap(), ResumeStep::Create);
    store.mark_creating().unwrap();
    assert_eq!(store.resume_step().unwrap(), ResumeStep::Reconcile);
    store.mark_uncertain().unwrap();
    assert_eq!(store.resume_step().unwrap(), ResumeStep::Reconcile);
    assert!(store.mark_creating().is_err());
    assert!(store.save_requests(&mut b"different".as_slice()).is_err());
}

#[test]
fn resume_step_revalidates_frozen_bytes_and_rejected_jobs_are_not_selected() {
    let fixture = Fixture::new();
    let mut store = fixture.create(&["a"]);
    uploaded(&mut store);
    store.mark_creating().unwrap();
    store.mark_rejected().unwrap();
    assert!(store.resume_step().is_err());
    drop(store);
    let preparation = Store::lock_preparation(&fixture.output, false).unwrap();
    assert!(
        preparation
            .open_frozen(None, Limits::default())
            .unwrap()
            .is_none()
    );
    let mut store = fixture.create(&["a"]);
    uploaded(&mut store);
    fs::write(
        store.directory().join("plans/0.json"),
        b"tampered after open",
    )
    .unwrap();
    assert!(store.resume_step().is_err());
    assert_eq!(store.state().phase, Phase::Uploaded);
}

fn remote_identity(nonce: &str, file_id: &str) -> markitai_core::provider_batch::RemoteIdentity {
    use std::net::TcpListener;
    use std::time::Duration;
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}/v1", listener.local_addr().unwrap());
    let response = serde_json::to_vec(&json!({"id":"batch_proven","input_file_id":file_id,
        "endpoint":"/v1/chat/completions","status":"in_progress","metadata":{"markitai_submission":nonce}})).unwrap();
    let worker = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        stream
            .set_write_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        let mut headers = Vec::new();
        while !headers.ends_with(b"\r\n\r\n") {
            let mut byte = [0];
            stream.read_exact(&mut byte).unwrap();
            headers.push(byte[0]);
            assert!(headers.len() < 65536);
        }
        assert!(headers.starts_with(b"GET /v1/batches/batch_proven "));
        write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            response.len()
        )
        .unwrap();
        stream.write_all(&response).unwrap();
    });
    let client = markitai_core::provider_batch::Client::new(
        &base,
        "fixture-only-key",
        Duration::from_secs(3),
        Default::default(),
    )
    .unwrap();
    let evidence = client.inspect("batch_proven").unwrap();
    worker.join().unwrap();
    evidence
}

#[test]
fn reconciled_binding_rechecks_api_base_nonce_file_and_phase_without_mutation() {
    for mismatch in 0..4 {
        let fixture = Fixture::new();
        let mut job = fixture.job(&["a"]);
        let nonce = if mismatch == 0 {
            "wrong_nonce"
        } else {
            &job.id
        };
        let file = if mismatch == 1 {
            "file_wrong"
        } else {
            "file_input"
        };
        let evidence = remote_identity(nonce, file);
        if mismatch != 2 {
            job.endpoint.api_base = evidence.api_base().into();
        }
        let mut store = Store::create(&fixture.output, job, false, Limits::default()).unwrap();
        uploaded(&mut store);
        if mismatch != 3 {
            store.mark_creating().unwrap();
        }
        let before = fs::read(store.directory().join("state.json")).unwrap();
        assert!(
            matches!(store.bind_reconciled(evidence), Err(Error::Conflict)),
            "mismatch {mismatch}"
        );
        assert_eq!(
            fs::read(store.directory().join("state.json")).unwrap(),
            before
        );
        assert!(store.state().batch.is_none());
    }
}

#[test]
fn proven_binding_preserves_create_evidence_across_state_commit_failure() {
    let fixture = Fixture::new();
    let mut job = fixture.job(&["a"]);
    let evidence = remote_identity(&job.id, "file_input");
    job.endpoint.api_base = evidence.api_base().into();
    let mut store = Store::create(&fixture.output, job, false, Limits::default()).unwrap();
    uploaded(&mut store);
    store.mark_creating().unwrap();
    let id = store.state().id.clone();
    store.fault = Some(Fault::BeforeStateReplace);
    assert!(store.bind_reconciled(evidence.clone()).is_err());
    assert_eq!(store.state().phase, Phase::Creating);
    drop(store);
    let mut restored = fixture.reopen(&id);
    restored.bind_reconciled(evidence).unwrap();
    assert_eq!(restored.resume_step().unwrap(), ResumeStep::Collect);
    drop(restored);
    let restored =
        Store::open_by_batch(&fixture.output, "batch_proven", false, Limits::default()).unwrap();
    assert_eq!(restored.state().id, id);
    assert_eq!(
        restored.state().uploaded.as_ref().unwrap().file_id,
        "file_input"
    );
    assert_eq!(restored.usage().unwrap().requests, 0);
}
