use super::*;
use serde_json::{Value, json};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::thread::{self, JoinHandle};
use std::time::Duration;

#[derive(Debug, Clone)]
struct Request {
    method: String,
    path: String,
    headers: String,
    body: Vec<u8>,
}
enum Reply {
    Bytes(u16, Vec<u8>),
    Redirect(String),
    Truncated(Vec<u8>),
    Chunked(Vec<u8>),
    Disconnect,
}
impl Reply {
    fn json(value: Value) -> Self {
        Self::Bytes(200, serde_json::to_vec(&value).unwrap())
    }
    fn text(value: String) -> Self {
        Self::Bytes(200, value.into_bytes())
    }
}
struct Server {
    address: String,
    requests: Arc<Mutex<Vec<Request>>>,
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
}
impl Server {
    fn new(reply: impl Fn(&Request, usize) -> Reply + Send + 'static) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let address = listener.local_addr().unwrap().to_string();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let observed = requests.clone();
        let stopped = stop.clone();
        let worker = thread::spawn(move || {
            while !stopped.load(Ordering::Acquire) {
                let mut stream = match listener.accept() {
                    Ok((stream, _)) => stream,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(3));
                        continue;
                    }
                    Err(error) => panic!("fixture accept: {error}"),
                };
                // macOS can inherit the listener's nonblocking socket flag.
                stream.set_nonblocking(false).unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(3)))
                    .unwrap();
                stream
                    .set_write_timeout(Some(Duration::from_secs(3)))
                    .unwrap();
                let mut headers = Vec::new();
                loop {
                    let mut byte = [0];
                    if stream.read_exact(&mut byte).is_err() {
                        break;
                    }
                    headers.push(byte[0]);
                    assert!(headers.len() <= 65536);
                    if headers.ends_with(b"\r\n\r\n") {
                        break;
                    }
                }
                if headers.is_empty() {
                    continue;
                }
                let headers = String::from_utf8(headers).unwrap();
                let mut first = headers.lines().next().unwrap().split_whitespace();
                let method = first.next().unwrap().to_owned();
                let path = first.next().unwrap().to_owned();
                let length = headers
                    .lines()
                    .find_map(|line| {
                        let (key, value) = line.split_once(':')?;
                        key.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().unwrap())
                    })
                    .unwrap_or(0);
                assert!(length <= 2 * 1024 * 1024);
                let mut body = vec![0; length];
                stream.read_exact(&mut body).unwrap();
                let request = Request {
                    method,
                    path,
                    headers,
                    body,
                };
                let index = {
                    let mut values = observed.lock().unwrap();
                    let index = values.len();
                    values.push(request.clone());
                    index
                };
                let response = reply(&request, index);
                match response {
                    Reply::Bytes(status, body) => {
                        write!(stream, "HTTP/1.1 {status} Fixture\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len()).unwrap();
                        stream.write_all(&body).unwrap();
                    }
                    Reply::Redirect(url) => {
                        write!(stream, "HTTP/1.1 307 Temporary Redirect\r\nLocation: {url}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").unwrap();
                    }
                    Reply::Truncated(body) => {
                        write!(
                            stream,
                            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            body.len() + 100
                        )
                        .unwrap();
                        stream.write_all(&body).unwrap();
                    }
                    Reply::Chunked(body) => {
                        stream.write_all(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n").unwrap();
                        write!(stream, "{:x}\r\n", body.len()).unwrap();
                        stream.write_all(&body).unwrap();
                        stream.write_all(b"\r\n0\r\n\r\n").unwrap();
                    }
                    Reply::Disconnect => (),
                }
            }
        });
        Self {
            address,
            requests,
            stop,
            worker: Some(worker),
        }
    }
    fn base(&self) -> String {
        format!("http://{}/prefix/v1", self.address)
    }
    fn client(&self) -> Client {
        self.limited(Limits::default())
    }
    fn limited(&self, limits: Limits) -> Client {
        Client::new(
            &self.base(),
            "fixture-only-secret",
            Duration::from_secs(3),
            limits,
        )
        .unwrap()
    }
    fn requests(&self) -> Vec<Request> {
        self.requests.lock().unwrap().clone()
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        let _ = TcpStream::connect(&self.address);
        if let Some(worker) = self.worker.take() {
            let result = worker.join();
            if !thread::panicking() {
                result.unwrap();
            }
        }
    }
}
fn request(id: &str) -> Value {
    json!({"custom_id":id,"method":"POST","url":"/v1/chat/completions",
        "body":{"model":"model-a","messages":[{"role":"user","content":"Private source Ω 文本"}]}})
}
fn rows(values: &[Value]) -> String {
    values
        .iter()
        .map(|value| format!("{}\n", serde_json::to_string(value).unwrap()))
        .collect()
}
fn write_input(root: &Path, values: &[Value]) -> std::path::PathBuf {
    let path = root.join("input.jsonl");
    std::fs::write(&path, rows(values)).unwrap();
    path
}
fn batch_value(status: &str) -> Value {
    json!({"id":"batch_fixture","input_file_id":"file_input","endpoint":"/v1/chat/completions",
        "status":status,"output_file_id":"file_output","error_file_id":"file_errors",
        "request_counts":{"total":3,"completed":2,"failed":1}})
}
fn batch(status: BatchStatus) -> Batch {
    Batch {
        id: "batch_fixture".into(),
        input_file_id: "file_input".into(),
        status,
        output_file_id: Some("file_output".into()),
        error_file_id: Some("file_errors".into()),
        total: Some(3),
        completed: Some(2),
        failed: Some(1),
    }
}
fn uploaded() -> UploadedInput {
    UploadedInput {
        file_id: "file_input".into(),
        model: "model-a".into(),
        custom_ids: vec!["doc_a".into()],
        bytes: 123,
        sha256: "a".repeat(64),
    }
}
fn answer(id: &str, status: u16, text: &str) -> Value {
    json!({"custom_id":id,"response":{"status_code":status,"request_id":"request_fixture",
        "body":{"model":"model-a","choices":[{"message":{"content":text}}],
        "usage":{"prompt_tokens":7,"completion_tokens":3}}},"error":null})
}
fn ids() -> Vec<String> {
    ["doc_a", "doc_b", "doc_c"].map(str::to_owned).to_vec()
}

#[test]
fn upload_submits_the_validated_snapshot_even_if_original_is_replaced() {
    let root = tempfile::tempdir().unwrap();
    let original = rows(&[request("doc_a"), request("doc_b")]);
    let path = write_input(root.path(), &[request("doc_a"), request("doc_b")]);
    let snapshot = super::input::snapshot(&path, Limits::default()).unwrap();
    std::fs::write(&path, b"unvalidated replacement").unwrap();
    let server = Server::new(|request, _| {
        assert_eq!(request.path, "/prefix/v1/files");
        Reply::json(json!({"id":"file_input","purpose":"batch"}))
    });
    let uploaded = server.client().upload_snapshot(snapshot).unwrap();
    let requests = server.requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].method, "POST");
    assert!(
        requests[0]
            .headers
            .to_ascii_lowercase()
            .contains("authorization: bearer fixture-only-secret")
    );
    let body = String::from_utf8(requests[0].body.clone()).unwrap();
    assert!(body.contains("name=\"purpose\"\r\n\r\nbatch\r\n"));
    assert!(body.contains("filename=\"requests.jsonl\""));
    assert!(body.contains(&original));
    assert!(!body.contains("unvalidated replacement"));
    assert_eq!(uploaded.bytes, original.len() as u64);
    assert_eq!(uploaded.model, "model-a");
    assert_eq!(uploaded.custom_ids, ["doc_a", "doc_b"]);
    use sha2::{Digest, Sha256};
    assert_eq!(
        uploaded.sha256,
        crate::hex(Sha256::digest(original.as_bytes()))
    );
    assert!(!format!("{uploaded:?}").contains("Private source"));
}

#[test]
fn invalid_jsonl_is_rejected_before_any_http_request() {
    let root = tempfile::tempdir().unwrap();
    let server = Server::new(|_, _| panic!("invalid input was sent"));
    let mut wrong_model = request("doc_b");
    wrong_model["body"]["model"] = json!("other-model");
    let mut wrong_method = request("doc_a");
    wrong_method["method"] = json!("GET");
    let mut wrong_url = request("doc_a");
    wrong_url["url"] = json!("https://example.invalid/stolen");
    let mut stream = request("doc_a");
    stream["body"]["stream"] = json!(true);
    let mut unknown = request("doc_a");
    unknown["extra"] = json!(true);
    let mut missing_messages = request("doc_a");
    missing_messages["body"]["messages"] = json!([]);
    for values in [
        vec![request("doc_a"), request("doc_a")],
        vec![request("doc_a"), wrong_model],
        vec![wrong_method],
        vec![wrong_url],
        vec![stream],
        vec![unknown],
        vec![missing_messages],
        vec![request("bad/id")],
    ] {
        let path = write_input(root.path(), &values);
        assert!(server.client().upload(&path).is_err());
    }
    let path = root.path().join("input.jsonl");
    for bytes in ["", "\n", "not json\n"] {
        std::fs::write(&path, bytes).unwrap();
        assert!(server.client().upload(&path).is_err());
    }
    assert!(server.requests().is_empty());
}

#[test]
fn input_line_total_and_request_limits_are_enforced_before_upload() {
    let root = tempfile::tempdir().unwrap();
    let path = write_input(root.path(), &[request("doc_a"), request("doc_b")]);
    let server = Server::new(|_, _| panic!("oversized input was sent"));
    for limits in [
        Limits {
            requests: 1,
            ..Limits::default()
        },
        Limits {
            line_bytes: 16,
            ..Limits::default()
        },
        Limits {
            upload_bytes: 16,
            ..Limits::default()
        },
    ] {
        assert!(matches!(
            server.limited(limits).upload(&path),
            Err(Error::Limit(_))
        ));
    }
    assert!(server.requests().is_empty());
}

#[cfg(unix)]
#[test]
fn input_symlink_and_fifo_are_rejected_without_blocking_or_network() {
    use std::os::unix::fs::symlink;
    let root = tempfile::tempdir().unwrap();
    let path = write_input(root.path(), &[request("doc_a")]);
    let link = root.path().join("alias");
    symlink(&path, &link).unwrap();
    let fifo = root.path().join("pipe");
    use std::os::unix::ffi::OsStrExt;
    let name = std::ffi::CString::new(fifo.as_os_str().as_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
    let server = Server::new(|_, _| panic!("unsafe input was sent"));
    assert!(server.client().upload(&link).is_err());
    assert!(server.client().upload(&fifo).is_err());
    assert!(server.requests().is_empty());
}

#[test]
fn create_and_retrieve_preserve_identity_without_following_result_urls() {
    let trap = Server::new(|_, _| panic!("foreign result URL followed"));
    let foreign = format!("{}/secret", trap.base());
    let server = Server::new(move |request, index| {
        assert_eq!(
            request.path,
            if index == 0 {
                "/prefix/v1/batches"
            } else {
                "/prefix/v1/batches/batch_fixture"
            }
        );
        if index == 0 {
            assert_eq!(
                serde_json::from_slice::<Value>(&request.body).unwrap(),
                json!({
                "input_file_id":"file_input","endpoint":"/v1/chat/completions", "completion_window":"24h",
                "metadata":{"markitai_submission":"attempt_123"}})
            );
        }
        let mut batch = batch_value(if index == 0 {
            "validating"
        } else {
            "completed"
        });
        batch["results_url"] = json!(foreign);
        Reply::json(batch)
    });
    let client = server.client();
    let created = client.create(&uploaded(), "attempt_123").unwrap();
    assert_eq!(created.status, BatchStatus::Validating);
    let retrieved = client.retrieve(&created.id).unwrap();
    assert_eq!(retrieved.status, BatchStatus::Completed);
    assert_eq!(
        (retrieved.total, retrieved.completed, retrieved.failed),
        (Some(3), Some(2), Some(1))
    );
    assert_eq!(server.requests().len(), 2);
    assert!(trap.requests().is_empty());
}

#[test]
fn ambiguous_create_disconnect_invalid_success_and_5xx_are_never_retried() {
    for kind in 0..4 {
        let server = Server::new(move |_, _| match kind {
            0 => Reply::Disconnect,
            1 => Reply::Bytes(200, b"not-json".to_vec()),
            2 => Reply::Bytes(503, b"provider secret diagnostic".to_vec()),
            _ => Reply::json(
                json!({"id":"batch_other","input_file_id":"file_wrong","status":"validating"}),
            ),
        });
        let error = server
            .client()
            .create(&uploaded(), "attempt_123")
            .unwrap_err();
        assert_eq!(error, Error::CreateUncertain);
        assert_eq!(server.requests().len(), 1);
        assert!(!error.to_string().contains("provider secret"));
    }
}

#[test]
fn explicit_create_rejection_is_distinct_from_unknown_and_never_retried() {
    let server =
        Server::new(|_, _| Reply::Bytes(401, b"fixture-only-secret unauthorized".to_vec()));
    assert_eq!(
        server
            .client()
            .create(&uploaded(), "attempt_123")
            .unwrap_err(),
        Error::Http(401)
    );
    assert_eq!(server.requests().len(), 1);
}

#[test]
fn redirects_do_not_forward_auth_or_repeat_post() {
    let trap = Server::new(|_, _| panic!("redirect followed"));
    let destination = trap.base();
    let server = Server::new(move |_, _| Reply::Redirect(destination.clone()));
    let root = tempfile::tempdir().unwrap();
    let path = write_input(root.path(), &[request("doc_a")]);
    let client = server.client();
    assert_eq!(client.upload(&path).unwrap_err(), Error::Http(307));
    assert_eq!(
        client.create(&uploaded(), "attempt_123").unwrap_err(),
        Error::CreateUncertain
    );
    assert_eq!(
        client.retrieve("batch_fixture").unwrap_err(),
        Error::Http(307)
    );
    let error = client
        .download_results(&batch(BatchStatus::Completed), &ids())
        .unwrap_err();
    assert_eq!(error.error, Error::Http(307));
    assert!(error.partial.items.is_empty());
    assert_eq!(server.requests().len(), 4);
    assert!(trap.requests().is_empty());
}

#[test]
fn expired_job_downloads_success_and_error_files_in_submission_order() {
    let server = Server::new(|request, _| match request.path.as_str() {
        "/prefix/v1/files/file_output/content" => Reply::text(rows(&[
            answer("doc_c", 200, "not structurally valid but paid"),
            answer("doc_a", 200, "okay"),
        ])),
        "/prefix/v1/files/file_errors/content" => {
            Reply::text(rows(&[json!({"custom_id":"doc_b", "response":null,
            "error":{"code":"batch_expired","message":"not executed"}})]))
        }
        _ => panic!("unexpected route"),
    });
    let result = server
        .client()
        .download_results(&batch(BatchStatus::Expired), &ids())
        .unwrap();
    assert_eq!(result.status, BatchStatus::Expired);
    assert_eq!(
        result
            .items
            .iter()
            .map(|item| item.custom_id.as_str())
            .collect::<Vec<_>>(),
        ["doc_a", "doc_b", "doc_c"]
    );
    assert!(result.missing.is_empty());
    assert_eq!(
        result.items[2].body.as_ref().unwrap()["usage"]["prompt_tokens"],
        7
    );
    assert_eq!(
        result.items[1].error.as_ref().unwrap()["code"],
        "batch_expired"
    );
    assert_eq!(
        result.items[0].request_id.as_deref(),
        Some("request_fixture")
    );
    assert_eq!(server.requests().len(), 2);
}

#[test]
fn http_error_bodies_with_usage_remain_attributed_and_missing_stays_explicit() {
    let server =
        Server::new(|_, _| Reply::text(rows(&[answer("doc_b", 500, "paid invalid output")])));
    let mut job = batch(BatchStatus::Cancelled);
    job.error_file_id = None;
    let result = server.client().download_results(&job, &ids()).unwrap();
    assert_eq!(result.items.len(), 1);
    assert_eq!(result.items[0].http_status, Some(500));
    assert_eq!(
        result.items[0].body.as_ref().unwrap()["usage"]["completion_tokens"],
        3
    );
    assert_eq!(result.missing, ["doc_a", "doc_c"]);
    assert!(result.bytes > 0);
    assert!(!format!("{:?}", result.items[0]).contains("paid invalid output"));
}

#[test]
fn duplicate_or_unknown_custom_id_does_not_replace_already_received_results() {
    for second in ["doc_a", "unknown"] {
        let server = Server::new(move |_, _| {
            Reply::text(rows(&[
                answer("doc_a", 200, "first paid answer"),
                answer(second, 200, "must not replace"),
            ]))
        });
        let failure = server
            .client()
            .download_results(&batch(BatchStatus::Completed), &ids())
            .unwrap_err();
        assert!(matches!(failure.error, Error::Invalid(_)));
        assert_eq!(failure.partial.items.len(), 1);
        assert_eq!(
            failure.partial.items[0].body.as_ref().unwrap()["choices"][0]["message"]["content"],
            "first paid answer"
        );
        assert_eq!(failure.partial.missing, ["doc_b", "doc_c"]);
        assert_eq!(server.requests().len(), 1);
    }
}

#[test]
fn later_error_file_failure_retains_already_attributed_paid_output() {
    let server = Server::new(|request, _| {
        if request.path.ends_with("file_output/content") {
            Reply::text(rows(&[answer("doc_a", 200, "paid")]))
        } else {
            Reply::Truncated(b"{\"custom_id\":\"doc_b\"".to_vec())
        }
    });
    let failure = server
        .client()
        .download_results(&batch(BatchStatus::Completed), &ids())
        .unwrap_err();
    assert_eq!(failure.error, Error::Transport);
    assert_eq!(failure.partial.items.len(), 1);
    assert_eq!(failure.partial.items[0].custom_id, "doc_a");
    assert_eq!(server.requests().len(), 2);
}

#[test]
fn combined_result_limit_and_individual_line_limit_are_independent() {
    let first = rows(&[answer("doc_a", 200, "small")]);
    let allowance = first.len() + 10;
    let server = Server::new(move |request, _| {
        if request.path.ends_with("file_output/content") {
            Reply::text(first.clone())
        } else {
            Reply::text(rows(&[answer(
                "doc_b",
                200,
                "bigger than remaining allowance",
            )]))
        }
    });
    let limited = Limits {
        result_bytes: allowance,
        ..Limits::default()
    };
    let failure = server
        .limited(limited)
        .download_results(&batch(BatchStatus::Completed), &ids())
        .unwrap_err();
    assert!(matches!(failure.error, Error::Limit(_)));
    assert_eq!(failure.partial.items.len(), 1);
    let failure = server
        .limited(Limits {
            line_bytes: 16,
            ..Limits::default()
        })
        .download_results(&batch(BatchStatus::Completed), &ids())
        .unwrap_err();
    assert!(matches!(failure.error, Error::Limit(_)));
    assert!(failure.partial.items.is_empty());
}

#[test]
fn invalid_file_ids_and_pending_jobs_issue_no_download_requests() {
    let server = Server::new(|_, _| panic!("unsafe or premature download"));
    for bad in [
        "../../elsewhere",
        "https://example.invalid/private",
        "file_%2fsecret",
        "file_x?key=secret",
    ] {
        let mut job = batch(BatchStatus::Completed);
        job.output_file_id = Some(bad.into());
        assert_eq!(
            server
                .client()
                .download_results(&job, &ids())
                .unwrap_err()
                .error,
            Error::Protocol
        );
    }
    assert_eq!(
        server
            .client()
            .download_results(&batch(BatchStatus::InProgress), &ids())
            .unwrap_err()
            .error,
        Error::Pending
    );
    let mut no_files = batch(BatchStatus::Failed);
    no_files.output_file_id = None;
    no_files.error_file_id = None;
    let result = server.client().download_results(&no_files, &ids()).unwrap();
    assert!(result.items.is_empty());
    assert_eq!(result.missing, ids());
    assert!(server.requests().is_empty());
}

#[test]
fn credentials_urls_and_control_body_limits_fail_without_exposing_content() {
    for base in [
        "file:///tmp/private",
        "https://user:password@example.invalid/v1",
        "https://example.invalid/v1?key=secret",
        "https://example.invalid/v1#fragment",
    ] {
        let error = Client::new(
            base,
            "fixture-only-secret",
            Duration::from_secs(1),
            Limits::default(),
        )
        .unwrap_err();
        assert!(!error.to_string().contains("password"));
        assert!(!error.to_string().contains("secret"));
    }
    let server = Server::new(|_, _| Reply::Bytes(200, vec![b'x'; 64]));
    let error = server
        .limited(Limits {
            control_bytes: 16,
            ..Limits::default()
        })
        .retrieve("batch_fixture")
        .unwrap_err();
    assert!(matches!(error, Error::Limit(_)));
    assert!(!format!("{:?}", server.client()).contains("fixture-only-secret"));
}

#[test]
fn chunked_result_and_control_bodies_are_bounded_without_content_length() {
    let first = rows(&[answer("doc_a", 200, "first paid answer")]);
    let allowance = first.len() + 8;
    let server = Server::new(move |request, _| {
        if request.path.contains("/files/") {
            Reply::Chunked(
                format!(
                    "{first}{}",
                    rows(&[answer("doc_b", 200, "another response")])
                )
                .into_bytes(),
            )
        } else {
            Reply::Chunked(vec![b'x'; 64])
        }
    });
    let client = server.limited(Limits {
        result_bytes: allowance,
        control_bytes: 16,
        ..Limits::default()
    });
    let failure = client
        .download_results(&batch(BatchStatus::Completed), &ids())
        .unwrap_err();
    assert!(
        matches!(failure.error, Error::Limit(_)),
        "unexpected chunked download error: {:?}; observed requests: {}",
        failure.error,
        server.requests().len()
    );
    assert_eq!(failure.partial.items.len(), 1);
    assert_eq!(failure.partial.items[0].custom_id, "doc_a");
    assert!(matches!(
        client.retrieve("batch_fixture"),
        Err(Error::Limit(_))
    ));
}

#[path = "reconcile_tests.rs"]
mod reconciliation;
