//! Message Batches transport against a loopback server answering with the
//! documented response shapes in `tests/fixtures/anthropic-batches`.
use super::tests::{Reply, Server};
use super::*;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::time::Duration;

const NONCE: &str = "0f8fad5b-d9cb-469f-a165-70867728950e";
const ID: &str = "msgbatch_01HkcTjaV5uDC8jWR4ZsDV8d";

fn fixture(name: &str) -> String {
    std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/anthropic-batches")
            .join(name),
    )
    .unwrap()
}
fn fixture_json(name: &str) -> Value {
    serde_json::from_str(&fixture(name)).unwrap()
}
fn client(server: &Server) -> Client {
    Client::for_api(
        Api::Anthropic,
        &server.base(),
        "fixture-only-secret",
        Duration::from_secs(3),
        Limits::default(),
    )
    .unwrap()
}
fn request(index: usize) -> Value {
    json!({"custom_id":request_id(Api::Anthropic, NONCE, index),"params":{"model":"claude-haiku-5-5",
        "max_tokens":8192,"system":"Clean the document.","messages":[{"role":"user","content":"Private source Ω 文本"}]}})
}
fn write_requests(root: &Path, rows: &[Value]) -> PathBuf {
    let path = root.join("requests.jsonl");
    let text: String = rows
        .iter()
        .map(|row| format!("{}\n", serde_json::to_string(row).unwrap()))
        .collect();
    std::fs::write(&path, text).unwrap();
    path
}
fn ids() -> Vec<String> {
    (0..2)
        .map(|index| request_id(Api::Anthropic, NONCE, index))
        .collect()
}

#[test]
fn custom_ids_carry_the_nonce_and_stay_within_the_documented_pattern() {
    let id = request_id(Api::Anthropic, NONCE, 49_999);
    assert_eq!(id, "0f8fad5bd9cb469fa16570867728950e-49999");
    assert!(id.len() <= 64 && types::custom_id(&id));
    assert_eq!(request_id(Api::OpenAi, NONCE, 3), "doc-3");
    assert!(batch_id(ID) && batch_id("batch_abc") && !batch_id("msg_abc"));
}

#[test]
fn upload_is_local_and_create_sends_the_saved_rows_inline_once() {
    let root = tempfile::tempdir().unwrap();
    let path = write_requests(root.path(), &[request(0), request(1)]);
    let server = Server::new(|request, _| {
        assert_eq!(request.method, "POST");
        assert_eq!(request.path, "/prefix/v1/messages/batches");
        Reply::text(fixture("batch-created.json"))
    });
    let client = client(&server);
    let uploaded = client.upload(&path).unwrap();
    assert!(
        server.requests().is_empty(),
        "upload must not leave the machine"
    );
    assert_eq!(uploaded.file_id, INLINE_INPUT);
    assert_eq!(uploaded.custom_ids, ids());
    assert_eq!(uploaded.model, "claude-haiku-5-5");
    let batch = client.create(&uploaded, NONCE, &path).unwrap();
    assert_eq!(batch.id, ID);
    assert_eq!(batch.status, BatchStatus::InProgress);
    assert_eq!(batch.input_file_id, INLINE_INPUT);
    assert_eq!(
        (batch.total, batch.completed, batch.failed),
        (Some(2), Some(0), Some(0))
    );
    let requests = server.requests();
    assert_eq!(requests.len(), 1);
    let headers = requests[0].headers.to_ascii_lowercase();
    assert!(
        headers.contains("x-api-key: fixture-only-secret"),
        "{headers}"
    );
    assert!(
        headers.contains("anthropic-version: 2023-06-01"),
        "{headers}"
    );
    assert!(!headers.contains("authorization:"), "{headers}");
    let body: Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(body, json!({"requests":[request(0), request(1)]}));
}

#[test]
fn create_refuses_rows_changed_after_validation_or_without_the_nonce() {
    let root = tempfile::tempdir().unwrap();
    let path = write_requests(root.path(), &[request(0), request(1)]);
    let server = Server::new(|_, _| panic!("no request may be sent"));
    let client = client(&server);
    let uploaded = client.upload(&path).unwrap();
    let mut changed = request(1);
    changed["params"]["system"] = json!("Something else.");
    write_requests(root.path(), &[request(0), changed]);
    assert!(matches!(
        client.create(&uploaded, NONCE, &path),
        Err(Error::Invalid(_))
    ));
    // Rows whose IDs lack this nonce could not be proven after an uncertain create.
    let foreign = write_requests(
        root.path(),
        &[json!({"custom_id":"doc-0","params":request(0)["params"]})],
    );
    let uploaded = client.upload(&foreign).unwrap();
    assert!(matches!(
        client.create(&uploaded, NONCE, &foreign),
        Err(Error::Invalid(_))
    ));
    // Anthropic has no default output limit.
    let mut unlimited = request(0);
    unlimited["params"]
        .as_object_mut()
        .unwrap()
        .remove("max_tokens");
    let path = write_requests(root.path(), &[unlimited]);
    assert!(matches!(client.upload(&path), Err(Error::Invalid(_))));
    // An OpenAI row is not a Message Batch row.
    let path = write_requests(
        root.path(),
        &[json!({"custom_id":"a","method":"POST","url":"/v1/chat/completions","body":{}})],
    );
    assert!(matches!(client.upload(&path), Err(Error::Invalid(_))));
    assert!(server.requests().is_empty());
}

#[test]
fn create_outcomes_separate_rejection_from_uncertainty() {
    let root = tempfile::tempdir().unwrap();
    let path = write_requests(root.path(), &[request(0)]);
    for (reply, rejected) in [
        (Reply::Bytes(400, br#"{"type":"error"}"#.to_vec()), true),
        (Reply::Bytes(429, br#"{"type":"error"}"#.to_vec()), true),
        (Reply::Bytes(529, br#"{"type":"error"}"#.to_vec()), false),
        (Reply::Disconnect, false),
        (Reply::text("{\"id\":\"not-a-batch\"}".into()), false),
    ] {
        let reply = std::sync::Mutex::new(Some(reply));
        let server = Server::new(move |_, _| reply.lock().unwrap().take().unwrap());
        let client = client(&server);
        let uploaded = client.upload(&path).unwrap();
        let error = client.create(&uploaded, NONCE, &path).unwrap_err();
        if rejected {
            assert!(matches!(error, Error::Http(_)), "{error:?}");
        } else {
            assert_eq!(error, Error::CreateUncertain);
        }
    }
}

#[test]
fn ended_batches_map_their_counts_and_results_to_the_shared_shape() {
    let server = Server::new(|request, _| match request.path.as_str() {
        "/prefix/v1/messages/batches/msgbatch_01HkcTjaV5uDC8jWR4ZsDV8d" => {
            Reply::text(fixture("batch-ended.json"))
        }
        "/prefix/v1/messages/batches/msgbatch_01HkcTjaV5uDC8jWR4ZsDV8d/results" => {
            Reply::Chunked(fixture("results.jsonl").into_bytes())
        }
        other => panic!("unexpected path {other}"),
    });
    let client = client(&server);
    let batch = client.retrieve(ID).unwrap();
    assert_eq!(batch.status, BatchStatus::Completed);
    assert_eq!(
        (batch.total, batch.completed, batch.failed),
        (Some(2), Some(1), Some(1))
    );
    assert!(client.retrieve("batch_openai").is_err());
    let results = client.download_results(&batch, &ids()).unwrap();
    assert!(results.missing.is_empty());
    // Submitted order, not file order.
    assert_eq!(results.items[0].custom_id, ids()[0]);
    assert_eq!(results.items[0].http_status, Some(200));
    assert_eq!(
        results.items[0].request_id.as_deref(),
        Some("msg_014VwiXbi91y3JMjcpyGBHX5")
    );
    let body = results.items[0].body.as_ref().unwrap();
    assert_eq!(body["usage"]["input_tokens"], 1200);
    assert_eq!(results.items[1].http_status, None);
    assert!(results.items[1].body.is_none());
    assert_eq!(
        results.items[1].error.as_ref().unwrap()["error"]["type"],
        "invalid_request_error"
    );
    // The documented results_url is never followed: both reads use the ID.
    assert!(
        server
            .requests()
            .iter()
            .all(|request| request.method == "GET" && !request.path.contains("://"))
    );
}

#[test]
fn canceled_and_expired_results_are_unbilled_errors_and_statuses_follow_the_counts() {
    let canceled = format!(
        "{}\n{}\n",
        json!({"custom_id":ids()[0],"result":{"type":"canceled"}}),
        json!({"custom_id":ids()[1],"result":{"type":"expired"}})
    );
    let server = Server::new(move |_, _| Reply::text(canceled.clone()));
    let client = client(&server);
    let mut batch = super::transport::tests_parse(fixture_json("batch-ended.json")).unwrap();
    batch.status = BatchStatus::Cancelled;
    let results = client.download_results(&batch, &ids()).unwrap();
    assert_eq!(results.items[0].error.as_ref().unwrap()["type"], "canceled");
    assert_eq!(results.items[1].error.as_ref().unwrap()["type"], "expired");
    assert!(results.items.iter().all(|item| item.body.is_none()));

    let mut value = fixture_json("batch-ended.json");
    value["cancel_initiated_at"] = json!("2026-10-11T08:10:00Z");
    assert_eq!(
        super::transport::tests_parse(value).unwrap().status,
        BatchStatus::Cancelled
    );
    let mut value = fixture_json("batch-ended.json");
    value["request_counts"] =
        json!({"processing":0,"succeeded":0,"errored":0,"canceled":0,"expired":2});
    assert_eq!(
        super::transport::tests_parse(value).unwrap().status,
        BatchStatus::Expired
    );
    let mut value = fixture_json("batch-created.json");
    value["processing_status"] = json!("canceling");
    assert_eq!(
        super::transport::tests_parse(value).unwrap().status,
        BatchStatus::Cancelling
    );
    let mut value = fixture_json("batch-created.json");
    value["processing_status"] = json!("finished");
    assert!(super::transport::tests_parse(value).is_err());
}

#[test]
fn foreign_or_duplicate_result_ids_stop_collection() {
    let server = Server::new(|_, index| {
        let row = if index == 0 {
            json!({"custom_id":"another-submission-0","result":{"type":"canceled"}})
        } else {
            json!({"custom_id":request_id(Api::Anthropic, NONCE, 0),"result":{"type":"canceled"}})
        };
        Reply::text(format!("{row}\n{row}\n"))
    });
    let client = client(&server);
    let batch = super::transport::tests_parse(fixture_json("batch-ended.json")).unwrap();
    let failure = client.download_results(&batch, &ids()).unwrap_err();
    assert!(matches!(failure.error, Error::Invalid(_)));
    let failure = client.download_results(&batch, &ids()).unwrap_err();
    assert!(matches!(failure.error, Error::Invalid(_)));
    // The first, attributable row survives the duplicate.
    assert_eq!(failure.partial.items.len(), 1);
}

#[test]
fn reconciliation_proves_only_an_ended_batch_whose_results_carry_the_nonce() {
    let root = tempfile::tempdir().unwrap();
    let path = write_requests(root.path(), &[request(0), request(1)]);
    let server = Server::new(|request, _| match request.path.as_str() {
        "/prefix/v1/messages/batches?limit=100" => Reply::text(fixture("list.json")),
        "/prefix/v1/messages/batches/msgbatch_03OtherSameSize0000000000/results" => {
            Reply::text(format!(
                "{}\n",
                json!({"custom_id":"ffffffffffffffffffffffffffffffff-0","result":{"type":"canceled"}})
            ))
        }
        "/prefix/v1/messages/batches/msgbatch_01HkcTjaV5uDC8jWR4ZsDV8d/results" => {
            Reply::text(fixture("results.jsonl"))
        }
        "/prefix/v1/messages/batches/msgbatch_01HkcTjaV5uDC8jWR4ZsDV8d" => {
            Reply::text(fixture("batch-ended.json"))
        }
        other => panic!("unexpected path {other}"),
    });
    let client = client(&server);
    let uploaded = client.upload(&path).unwrap();
    let Reconciliation::Found(identity) = client
        .reconcile(&uploaded, NONCE, ReconcileLimits::default())
        .unwrap()
    else {
        panic!("the batch whose results carry the nonce must be found");
    };
    assert_eq!(identity.batch().id, ID);
    assert!(identity.matches(&uploaded, NONCE));
    assert!(!identity.matches(&uploaded, "another-nonce"));
    // The unrelated three-request batch was never opened; only GETs were sent.
    let requests = server.requests();
    assert!(requests.iter().all(|request| request.method == "GET"));
    assert!(
        !requests
            .iter()
            .any(|request| request.path.contains("msgbatch_02Unrelated"))
    );
}

#[test]
fn reconciliation_stays_incomplete_while_a_same_size_batch_is_processing() {
    let root = tempfile::tempdir().unwrap();
    let path = write_requests(root.path(), &[request(0), request(1)]);
    let server = Server::new(|request, _| {
        assert_eq!(request.path, "/prefix/v1/messages/batches?limit=100");
        let mut processing = fixture_json("batch-created.json");
        processing["id"] = json!("msgbatch_04Processing00000000000");
        Reply::json(json!({"data":[processing],"has_more":false,
            "first_id":"msgbatch_04Processing00000000000","last_id":"msgbatch_04Processing00000000000"}))
    });
    let client = client(&server);
    let uploaded = client.upload(&path).unwrap();
    assert!(matches!(
        client
            .reconcile(&uploaded, NONCE, ReconcileLimits::default())
            .unwrap(),
        Reconciliation::Incomplete
    ));
    let empty = Server::new(|_, _| {
        Reply::json(json!({"data":[],"has_more":false,"first_id":null,"last_id":null}))
    });
    assert!(matches!(
        client_for(&empty)
            .reconcile(&uploaded, NONCE, ReconcileLimits::default())
            .unwrap(),
        Reconciliation::NotFound
    ));
}

fn client_for(server: &Server) -> Client {
    client(server)
}

#[test]
fn manual_binding_waits_for_the_end_and_then_requires_the_nonce() {
    let root = tempfile::tempdir().unwrap();
    let path = write_requests(root.path(), &[request(0), request(1)]);
    let processing = Server::new(|_, _| Reply::text(fixture("batch-created.json")));
    let client = client(&processing);
    let uploaded = client.upload(&path).unwrap();
    assert_eq!(
        client.verify_binding(ID, &uploaded, NONCE).unwrap_err(),
        Error::Pending
    );
    let ended = Server::new(|request, _| {
        if request.path.ends_with("/results") {
            Reply::text(fixture("results.jsonl"))
        } else {
            Reply::text(fixture("batch-ended.json"))
        }
    });
    let client = client_for(&ended);
    let identity = client.verify_binding(ID, &uploaded, NONCE).unwrap();
    assert_eq!(identity.batch().id, ID);
    assert!(matches!(
        client.verify_binding(ID, &uploaded, "11111111-2222-3333-4444-555555555555"),
        Err(Error::Invalid(_))
    ));
}
