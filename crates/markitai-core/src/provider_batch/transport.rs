use super::input;
use super::types::nonce_prefix;
use super::types::*;
use reqwest::blocking::{Client as HttpClient, RequestBuilder, Response, multipart};
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderName, HeaderValue};
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
use std::io::BufReader;
use std::path::Path;
use std::time::{Duration, Instant};
use url::Url;

/// Anthropic's documented API version for Message Batches.
const ANTHROPIC_VERSION: &str = "2023-06-01";
/// Anthropic batches whose results are opened to prove an uncertain creation.
const MAX_PROOFS: usize = 4;

/// No environment/config loading, redirects, automatic retries or persisted auth.
pub struct Client {
    http: HttpClient,
    base: Url,
    api: Api,
    /// `Authorization: Bearer …` for OpenAI, `x-api-key` for Anthropic.
    credential: (HeaderName, HeaderValue),
    limits: Limits,
    timeout: Duration,
}
impl std::fmt::Debug for Client {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchClient")
            .field("limits", &self.limits)
            .finish_non_exhaustive()
    }
}

impl Client {
    /// An OpenAI Batch API client.
    pub fn new(
        base: &str,
        api_key: &str,
        timeout: Duration,
        limits: Limits,
    ) -> Result<Self, Error> {
        Self::for_api(Api::OpenAi, base, api_key, timeout, limits)
    }

    pub fn for_api(
        api: Api,
        base: &str,
        api_key: &str,
        timeout: Duration,
        limits: Limits,
    ) -> Result<Self, Error> {
        let limits = limits.validate()?;
        if base.len() > 8192
            || base.trim() != base
            || base.chars().any(char::is_control)
            || api_key.is_empty()
            || api_key.len() > 8192
            || api_key.chars().any(char::is_control)
            || timeout.is_zero()
            || timeout > Duration::from_secs(1800)
        {
            return Err(Error::Invalid(
                "Batch endpoint, credential or timeout is invalid",
            ));
        }
        let mut base = Url::parse(base).map_err(|_| Error::Invalid("Batch endpoint is invalid"))?;
        if !matches!(base.scheme(), "http" | "https")
            || base.host_str().is_none()
            || !base.username().is_empty()
            || base.password().is_some()
            || base.query().is_some()
            || base.fragment().is_some()
        {
            return Err(Error::Invalid(
                "Batch endpoint must be HTTP(S) without user information, query or fragment",
            ));
        }
        if !base.path().ends_with('/') {
            let path = format!("{}/", base.path());
            base.set_path(&path);
        }
        let (name, value) = match api {
            Api::OpenAi => (AUTHORIZATION, format!("Bearer {api_key}")),
            Api::Anthropic => (HeaderName::from_static("x-api-key"), api_key.to_owned()),
        };
        let mut value = HeaderValue::from_str(&value)
            .map_err(|_| Error::Invalid("Batch credential is invalid"))?;
        value.set_sensitive(true);
        let http = crate::trust::client_builder()
            .timeout(timeout)
            .connect_timeout(timeout.min(Duration::from_secs(15)))
            .redirect(reqwest::redirect::Policy::none())
            .retry(reqwest::retry::never())
            // An implicit proxy is not part of this explicitly supplied identity.
            .no_proxy()
            .gzip(false)
            .brotli(false)
            .build()
            .map_err(|_| Error::Transport)?;
        Ok(Self {
            http,
            base,
            api,
            credential: (name, value),
            limits,
            timeout,
        })
    }

    pub fn api(&self) -> Api {
        self.api
    }

    /// The credential and, for Anthropic, its API version on one request.
    fn authorized(&self, request: RequestBuilder) -> RequestBuilder {
        let request = request.header(self.credential.0.clone(), self.credential.1.clone());
        match self.api {
            Api::OpenAi => request,
            Api::Anthropic => request.header("anthropic-version", ANTHROPIC_VERSION),
        }
    }

    fn endpoint(&self, relative: &str) -> Result<Url, Error> {
        self.base
            .join(relative)
            .map_err(|_| Error::Invalid("Batch endpoint is invalid"))
    }

    /// OpenAI: upload the validated snapshot. Anthropic: validate it only; its
    /// requests travel in the create call, so nothing leaves the machine here.
    pub fn upload(&self, source: &Path) -> Result<UploadedInput, Error> {
        let input = input::snapshot(source, self.limits, self.api)?;
        match self.api {
            Api::OpenAi => self.upload_snapshot(input),
            Api::Anthropic => Ok(UploadedInput {
                file_id: INLINE_INPUT.into(),
                model: input.model,
                custom_ids: input.custom_ids,
                bytes: input.bytes,
                sha256: input.sha256,
            }),
        }
    }

    pub(super) fn upload_snapshot(&self, input: input::Input) -> Result<UploadedInput, Error> {
        let input::Input {
            file,
            model,
            custom_ids,
            bytes,
            sha256,
        } = input;
        let part = multipart::Part::reader_with_length(file, bytes)
            .file_name("requests.jsonl")
            .mime_str("application/jsonl")
            .map_err(|_| Error::Invalid("Batch multipart content type is invalid"))?;
        let form = multipart::Form::new()
            .text("purpose", "batch")
            .part("file", part);
        let response = self
            .authorized(self.http.post(self.endpoint("files")?))
            .multipart(form)
            .send()
            .map_err(|_| Error::Transport)?;
        let value = self.control(response)?;
        let id = value
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| identifier(id, "file"))
            .ok_or(Error::Protocol)?;
        if value
            .get("purpose")
            .is_some_and(|purpose| purpose != "batch")
            || value
                .get("bytes")
                .is_some_and(|reported| reported.as_u64() != Some(bytes))
        {
            return Err(Error::Protocol);
        }
        Ok(UploadedInput {
            file_id: id.to_owned(),
            model,
            custom_ids,
            bytes,
            sha256,
        })
    }

    /// Exactly one create attempt. Network/5xx/malformed success cannot prove rejection.
    /// The caller must durably retain UploadedInput and nonce BEFORE this method.
    /// `requests` is the saved JSONL: an Anthropic batch sends it inline, after
    /// checking that it is still exactly the validated snapshot.
    pub fn create(
        &self,
        input: &UploadedInput,
        nonce: &str,
        requests: &Path,
    ) -> Result<Batch, Error> {
        if !identifier(&input.file_id, "file")
            || !custom_id(nonce)
            || input.bytes == 0
            || input.bytes > self.limits.upload_bytes as u64
            || input.custom_ids.is_empty()
            || input.custom_ids.len() > self.limits.requests
            || input.custom_ids.iter().any(|id| !custom_id(id))
            || input.custom_ids.iter().collect::<HashSet<_>>().len() != input.custom_ids.len()
            || input.model.trim().is_empty()
            || input.model.len() > 256
            || input.model.chars().any(char::is_control)
            || input.sha256.len() != 64
            || !input.sha256.bytes().all(|b| b.is_ascii_hexdigit())
        {
            return Err(Error::Invalid("Batch submission identity is invalid"));
        }
        let (url, body) = match self.api {
            Api::OpenAi => (
                self.endpoint("batches")?,
                json!({"input_file_id":input.file_id,"endpoint":ENDPOINT,
                    "completion_window":"24h","metadata":{"markitai_submission":nonce}})
                .to_string()
                .into_bytes(),
            ),
            Api::Anthropic => (
                self.endpoint("messages/batches")?,
                self.inline_requests(input, nonce, requests)?,
            ),
        };
        let response = self
            .authorized(self.http.post(url))
            .header(CONTENT_TYPE, "application/json")
            .body(body)
            .send()
            .map_err(|_| Error::CreateUncertain)?;
        let status = response.status().as_u16();
        if (400..500).contains(&status) && !matches!(status, 408 | 409) {
            return Err(Error::Http(status));
        }
        if !response.status().is_success() {
            return Err(Error::CreateUncertain);
        }
        let value = self.control(response).map_err(|_| Error::CreateUncertain)?;
        let batch = self.parse(value).map_err(|_| Error::CreateUncertain)?;
        if batch.input_file_id != input.file_id {
            return Err(Error::CreateUncertain);
        }
        Ok(batch)
    }

    /// The create body of a Message Batch: the saved request rows, re-read and
    /// checked against the validated snapshot. Every custom ID carries the
    /// submission nonce, which is what later proves an uncertain creation.
    fn inline_requests(
        &self,
        input: &UploadedInput,
        nonce: &str,
        requests: &Path,
    ) -> Result<Vec<u8>, Error> {
        let prefix = nonce_prefix(nonce);
        if input.file_id != INLINE_INPUT
            || input.custom_ids.iter().any(|id| !id.starts_with(&prefix))
        {
            return Err(Error::Invalid("Batch submission identity is invalid"));
        }
        let staged = input::snapshot(requests, self.limits, Api::Anthropic)?;
        if staged.sha256 != input.sha256
            || staged.bytes != input.bytes
            || staged.model != input.model
            || staged.custom_ids != input.custom_ids
        {
            return Err(Error::Invalid(
                "Batch requests changed after they were validated",
            ));
        }
        let mut body = Vec::with_capacity(staged.bytes as usize + 16);
        body.extend_from_slice(b"{\"requests\":[");
        let mut reader = BufReader::new(staged.file);
        let mut first = true;
        loop {
            let row = input::line(&mut reader, self.limits.line_bytes).map_err(|_| Error::Input)?;
            if row.is_empty() {
                break;
            }
            if !first {
                body.push(b',');
            }
            first = false;
            body.extend_from_slice(row.trim_ascii_end());
        }
        body.extend_from_slice(b"]}");
        Ok(body)
    }

    fn valid_id(&self, batch_id: &str) -> bool {
        match self.api {
            Api::OpenAi => identifier(batch_id, "batch"),
            Api::Anthropic => identifier(batch_id, "msgbatch"),
        }
    }

    fn batch_path(&self, batch_id: &str) -> String {
        match self.api {
            Api::OpenAi => format!("batches/{batch_id}"),
            Api::Anthropic => format!("messages/batches/{batch_id}"),
        }
    }

    fn parse(&self, value: Value) -> Result<Batch, Error> {
        match self.api {
            Api::OpenAi => parse_batch(value),
            Api::Anthropic => parse_message_batch(value),
        }
    }

    pub fn retrieve(&self, batch_id: &str) -> Result<Batch, Error> {
        if !self.valid_id(batch_id) {
            return Err(Error::Invalid("Batch ID is invalid"));
        }
        let response = self
            .authorized(self.http.get(self.endpoint(&self.batch_path(batch_id))?))
            .send()
            .map_err(|_| Error::Transport)?;
        let batch = self.parse(self.control(response)?)?;
        if batch.id != batch_id {
            return Err(Error::Protocol);
        }
        Ok(batch)
    }

    /// Retrieve identity evidence, without deriving URLs from response metadata.
    pub fn inspect(&self, batch_id: &str) -> Result<RemoteIdentity, Error> {
        self.inspect_with_budget(batch_id, self.timeout, self.limits.control_bytes)
            .map(|(identity, _)| identity)
    }

    /// The manual-ID path verifies exactly the same proof as automatic reconciliation.
    pub fn verify_binding(
        &self,
        batch_id: &str,
        uploaded: &UploadedInput,
        nonce: &str,
    ) -> Result<RemoteIdentity, Error> {
        validate_recovery(uploaded, nonce)?;
        let mut identity = self.inspect(batch_id)?;
        if self.api == Api::Anthropic {
            // A Message Batch keeps no submission metadata: only its results,
            // whose custom IDs carry the nonce, can prove it, after it ends.
            if !identity.batch.status.terminal() {
                return Err(Error::Pending);
            }
            if self.proven(&identity.batch, uploaded, nonce)? {
                identity.submission_nonce = Some(nonce.to_owned());
            }
        }
        if !identity.matches(uploaded, nonce) {
            return Err(Error::Invalid(
                "Batch identity does not match the frozen submission",
            ));
        }
        Ok(identity)
    }

    /// Whether an ended Message Batch is the submission with this nonce: its
    /// first result names one of the submission's custom IDs, which no other
    /// submission can use. Reads one bounded result line.
    fn proven(&self, batch: &Batch, uploaded: &UploadedInput, nonce: &str) -> Result<bool, Error> {
        if !identifier(&batch.id, "msgbatch") {
            return Err(Error::Protocol);
        }
        let response = self
            .authorized(
                self.http
                    .get(self.endpoint(&format!("messages/batches/{}/results", batch.id))?),
            )
            .send()
            .map_err(|_| Error::Transport)?;
        if !response.status().is_success() {
            return Err(Error::Http(response.status().as_u16()));
        }
        let row = input::line(&mut BufReader::new(response), self.limits.line_bytes)?;
        if row.is_empty() {
            return Ok(false);
        }
        let value: Value = serde_json::from_slice(&row).map_err(|_| Error::Protocol)?;
        let id = value
            .get("custom_id")
            .and_then(Value::as_str)
            .ok_or(Error::Protocol)?;
        Ok(id.starts_with(&nonce_prefix(nonce)) && uploaded.custom_ids.iter().any(|own| own == id))
    }

    /// Only GET requests are possible here, including after empty or ambiguous searches.
    pub fn reconcile(
        &self,
        uploaded: &UploadedInput,
        nonce: &str,
        limits: ReconcileLimits,
    ) -> Result<Reconciliation, Error> {
        validate_recovery(uploaded, nonce)?;
        let limits = limits.validate()?;
        if self.api == Api::Anthropic {
            return self.reconcile_messages(uploaded, nonce, limits);
        }
        let deadline = Instant::now() + limits.timeout;
        let mut remaining_bytes = limits.bytes;
        let mut cursor: Option<String> = None;
        let mut seen = HashSet::new();
        let mut matching: Option<RemoteIdentity> = None;
        for _ in 0..limits.pages {
            let Some(remaining_time) = deadline
                .checked_duration_since(Instant::now())
                .filter(|time| !time.is_zero())
            else {
                return Ok(Reconciliation::Incomplete);
            };
            if remaining_bytes == 0 {
                return Ok(Reconciliation::Incomplete);
            }
            let mut url = self.endpoint("batches")?;
            url.query_pairs_mut().append_pair("limit", "100");
            if let Some(cursor) = &cursor {
                url.query_pairs_mut().append_pair("after", cursor);
            }
            let response = match self
                .authorized(self.http.get(url))
                .timeout(remaining_time.min(self.timeout))
                .send()
            {
                Ok(response) => response,
                Err(_) if Instant::now() >= deadline => return Ok(Reconciliation::Incomplete),
                Err(_) => return Err(Error::Transport),
            };
            let (value, consumed) = match self
                .control_limited(response, remaining_bytes.min(self.limits.control_bytes))
            {
                Ok(value) => value,
                Err(Error::Limit(_)) => return Ok(Reconciliation::Incomplete),
                Err(_) if Instant::now() >= deadline => return Ok(Reconciliation::Incomplete),
                Err(error) => return Err(error),
            };
            remaining_bytes = remaining_bytes
                .checked_sub(consumed)
                .ok_or(Error::Protocol)?;
            if Instant::now() >= deadline {
                return Ok(Reconciliation::Incomplete);
            }
            let object = value.as_object().ok_or(Error::Protocol)?;
            if object.get("object").and_then(Value::as_str) != Some("list") {
                return Err(Error::Protocol);
            }
            let data = object
                .get("data")
                .and_then(Value::as_array)
                .ok_or(Error::Protocol)?;
            let has_more = object
                .get("has_more")
                .and_then(Value::as_bool)
                .ok_or(Error::Protocol)?;
            if data.len() > 100 {
                return Err(Error::Protocol);
            }
            let mut last_id = None;
            for row in data {
                let id = row
                    .get("id")
                    .and_then(Value::as_str)
                    .filter(|id| identifier(id, "batch"))
                    .ok_or(Error::Protocol)?;
                if !seen.insert(id.to_owned()) {
                    return Ok(Reconciliation::Incomplete);
                }
                last_id = Some(id);
                if remote_nonce(row)?.as_deref() != Some(nonce) {
                    continue;
                }
                let identity = parse_remote(row.clone(), self.base.as_str().trim_end_matches('/'))?;
                if !identity.matches(uploaded, nonce) {
                    return Err(Error::Invalid(
                        "Batch identity does not match the frozen submission",
                    ));
                }
                if let Some(prior) = &matching {
                    return Ok(Reconciliation::Ambiguous(vec![
                        prior.batch.id.clone(),
                        identity.batch.id,
                    ]));
                }
                matching = Some(identity);
            }
            if !has_more {
                let Some(found) = matching else {
                    return Ok(Reconciliation::NotFound);
                };
                let Some(remaining_time) = deadline
                    .checked_duration_since(Instant::now())
                    .filter(|time| !time.is_zero())
                else {
                    return Ok(Reconciliation::Incomplete);
                };
                if remaining_bytes == 0 {
                    return Ok(Reconciliation::Incomplete);
                }
                let checked = self.inspect_with_budget(
                    &found.batch.id,
                    remaining_time.min(self.timeout),
                    remaining_bytes.min(self.limits.control_bytes),
                );
                let (checked, _) = match checked {
                    Ok(value) => value,
                    Err(Error::Limit(_)) => return Ok(Reconciliation::Incomplete),
                    Err(_) if Instant::now() >= deadline => return Ok(Reconciliation::Incomplete),
                    Err(error) => return Err(error),
                };
                if Instant::now() >= deadline {
                    return Ok(Reconciliation::Incomplete);
                }
                if !checked.matches(uploaded, nonce) {
                    return Err(Error::Invalid(
                        "Batch identity changed during reconciliation",
                    ));
                }
                return Ok(Reconciliation::Found(Box::new(checked)));
            }
            let next = object
                .get("last_id")
                .and_then(Value::as_str)
                .filter(|id| identifier(id, "batch"))
                .ok_or(Error::Protocol)?;
            if last_id != Some(next) {
                return Ok(Reconciliation::Incomplete);
            }
            cursor = Some(next.to_owned());
        }
        Ok(Reconciliation::Incomplete)
    }

    /// Message Batches carry no submission metadata, so a listed batch of the
    /// same size only becomes evidence once it has ended and its first result
    /// names one of this submission's nonce-bearing custom IDs. A same-size
    /// batch still processing keeps the search incomplete; absence of every
    /// candidate never authorizes another create.
    fn reconcile_messages(
        &self,
        uploaded: &UploadedInput,
        nonce: &str,
        limits: ReconcileLimits,
    ) -> Result<Reconciliation, Error> {
        let deadline = Instant::now() + limits.timeout;
        let mut remaining_bytes = limits.bytes;
        let mut cursor: Option<String> = None;
        let mut seen = HashSet::new();
        let mut matching: Option<Batch> = None;
        let mut unproven = false;
        let mut proofs = 0;
        let expected = uploaded.custom_ids.len() as u64;
        let mut complete = false;
        for _ in 0..limits.pages {
            let Some(remaining_time) = deadline
                .checked_duration_since(Instant::now())
                .filter(|time| !time.is_zero())
            else {
                return Ok(Reconciliation::Incomplete);
            };
            if remaining_bytes == 0 {
                return Ok(Reconciliation::Incomplete);
            }
            let mut url = self.endpoint("messages/batches")?;
            url.query_pairs_mut().append_pair("limit", "100");
            if let Some(cursor) = &cursor {
                url.query_pairs_mut().append_pair("after_id", cursor);
            }
            let response = match self
                .authorized(self.http.get(url))
                .timeout(remaining_time.min(self.timeout))
                .send()
            {
                Ok(response) => response,
                Err(_) if Instant::now() >= deadline => return Ok(Reconciliation::Incomplete),
                Err(_) => return Err(Error::Transport),
            };
            let (value, consumed) = match self
                .control_limited(response, remaining_bytes.min(self.limits.control_bytes))
            {
                Ok(value) => value,
                Err(Error::Limit(_)) => return Ok(Reconciliation::Incomplete),
                Err(_) if Instant::now() >= deadline => return Ok(Reconciliation::Incomplete),
                Err(error) => return Err(error),
            };
            remaining_bytes = remaining_bytes
                .checked_sub(consumed)
                .ok_or(Error::Protocol)?;
            let object = value.as_object().ok_or(Error::Protocol)?;
            let data = object
                .get("data")
                .and_then(Value::as_array)
                .ok_or(Error::Protocol)?;
            let has_more = object
                .get("has_more")
                .and_then(Value::as_bool)
                .ok_or(Error::Protocol)?;
            if data.len() > 100 {
                return Err(Error::Protocol);
            }
            let mut last_id = None;
            for row in data {
                let batch = parse_message_batch(row.clone())?;
                if !seen.insert(batch.id.clone()) {
                    return Ok(Reconciliation::Incomplete);
                }
                last_id = Some(batch.id.clone());
                if batch.total != Some(expected) {
                    continue;
                }
                if !batch.status.terminal() {
                    unproven = true;
                    continue;
                }
                if proofs == MAX_PROOFS || Instant::now() >= deadline {
                    return Ok(Reconciliation::Incomplete);
                }
                proofs += 1;
                if self.proven(&batch, uploaded, nonce)? {
                    if let Some(prior) = &matching {
                        return Ok(Reconciliation::Ambiguous(vec![prior.id.clone(), batch.id]));
                    }
                    matching = Some(batch);
                }
            }
            if !has_more {
                complete = true;
                break;
            }
            let next = object
                .get("last_id")
                .and_then(Value::as_str)
                .filter(|id| identifier(id, "msgbatch"))
                .ok_or(Error::Protocol)?;
            if last_id.as_deref() != Some(next) {
                return Ok(Reconciliation::Incomplete);
            }
            cursor = Some(next.to_owned());
        }
        let Some(found) = matching else {
            return Ok(if complete && !unproven {
                Reconciliation::NotFound
            } else {
                Reconciliation::Incomplete
            });
        };
        let Some(remaining_time) = deadline
            .checked_duration_since(Instant::now())
            .filter(|time| !time.is_zero())
        else {
            return Ok(Reconciliation::Incomplete);
        };
        let (mut checked, _) = match self.inspect_with_budget(
            &found.id,
            remaining_time.min(self.timeout),
            remaining_bytes.clamp(1, self.limits.control_bytes),
        ) {
            Ok(value) => value,
            Err(Error::Limit(_)) => return Ok(Reconciliation::Incomplete),
            Err(_) if Instant::now() >= deadline => return Ok(Reconciliation::Incomplete),
            Err(error) => return Err(error),
        };
        if !checked.batch.status.terminal() {
            return Err(Error::Invalid(
                "Batch identity changed during reconciliation",
            ));
        }
        checked.submission_nonce = Some(nonce.to_owned());
        if !checked.matches(uploaded, nonce) {
            return Err(Error::Invalid(
                "Batch identity changed during reconciliation",
            ));
        }
        Ok(Reconciliation::Found(Box::new(checked)))
    }

    fn inspect_with_budget(
        &self,
        batch_id: &str,
        timeout: Duration,
        bytes: usize,
    ) -> Result<(RemoteIdentity, usize), Error> {
        if !self.valid_id(batch_id) {
            return Err(Error::Invalid("Batch ID is invalid"));
        }
        let response = self
            .authorized(self.http.get(self.endpoint(&self.batch_path(batch_id))?))
            .timeout(timeout)
            .send()
            .map_err(|_| Error::Transport)?;
        let (value, bytes) = self.control_limited(response, bytes)?;
        let api_base = self.base.as_str().trim_end_matches('/');
        let identity = match self.api {
            Api::OpenAi => parse_remote(value, api_base)?,
            Api::Anthropic => RemoteIdentity {
                batch: parse_message_batch(value)?,
                endpoint: MESSAGES_ENDPOINT.into(),
                api_base: api_base.into(),
                submission_nonce: None,
            },
        };
        if identity.batch.id != batch_id {
            return Err(Error::Protocol);
        }
        Ok((identity, bytes))
    }

    /// Download both files, including partial results of expired/cancelled jobs.
    /// A failure retains earlier attributed results; the caller owns persistence.
    pub fn download_results(
        &self,
        batch: &Batch,
        expected_ids: &[String],
    ) -> Result<Results, DownloadFailure> {
        let mut collection = match Collection::new(batch.status, expected_ids, self.limits.requests)
        {
            Ok(collection) => collection,
            Err(error) => {
                return Err(DownloadFailure {
                    error,
                    partial: Results {
                        status: batch.status,
                        items: Vec::new(),
                        missing: Vec::new(),
                        bytes: 0,
                    },
                });
            }
        };
        let result = self.download_into(batch, &mut collection);
        let partial = collection.finish();
        match result {
            Ok(()) => Ok(partial),
            Err(error) => Err(DownloadFailure { error, partial }),
        }
    }

    fn download_into(&self, batch: &Batch, collection: &mut Collection<'_>) -> Result<(), Error> {
        if !batch.status.terminal() {
            return Err(Error::Pending);
        }
        if self.api == Api::Anthropic {
            return self.download_messages(batch, collection);
        }
        if !identifier(&batch.id, "batch")
            || !identifier(&batch.input_file_id, "file")
            || batch
                .output_file_id
                .as_ref()
                .is_some_and(|id| !identifier(id, "file"))
            || batch
                .error_file_id
                .as_ref()
                .is_some_and(|id| !identifier(id, "file"))
            || (batch.output_file_id.is_some() && batch.output_file_id == batch.error_file_id)
        {
            return Err(Error::Protocol);
        }
        for file in [&batch.output_file_id, &batch.error_file_id]
            .into_iter()
            .flatten()
        {
            // Never accept/follow a provider-supplied results_url.
            let response = self
                .authorized(
                    self.http
                        .get(self.endpoint(&format!("files/{file}/content"))?),
                )
                .send()
                .map_err(|_| Error::Transport)?;
            if !response.status().is_success() {
                return Err(Error::Http(response.status().as_u16()));
            }
            let remaining = self.limits.result_bytes - collection.bytes;
            if response
                .content_length()
                .is_some_and(|length| length > remaining as u64)
            {
                return Err(Error::Limit(
                    "Batch result files exceed their combined byte limit",
                ));
            }
            let mut response = BufReader::new(response);
            loop {
                let remaining = self.limits.result_bytes - collection.bytes;
                let row = input::line(&mut response, self.limits.line_bytes.min(remaining))?;
                if row.is_empty() {
                    break;
                }
                collection.bytes += row.len();
                let value: Value = serde_json::from_slice(&row).map_err(|_| Error::Protocol)?;
                collection.push(value)?;
            }
        }
        Ok(())
    }

    /// Message Batch results stream from a path built from the validated ID,
    /// never from the batch's `results_url`. Each row becomes the OpenAI
    /// result shape the collector already reads.
    fn download_messages(
        &self,
        batch: &Batch,
        collection: &mut Collection<'_>,
    ) -> Result<(), Error> {
        if !identifier(&batch.id, "msgbatch")
            || batch.input_file_id != INLINE_INPUT
            || batch.output_file_id.is_some()
            || batch.error_file_id.is_some()
        {
            return Err(Error::Protocol);
        }
        let response = self
            .authorized(
                self.http
                    .get(self.endpoint(&format!("messages/batches/{}/results", batch.id))?),
            )
            .send()
            .map_err(|_| Error::Transport)?;
        if !response.status().is_success() {
            return Err(Error::Http(response.status().as_u16()));
        }
        let remaining = self.limits.result_bytes - collection.bytes;
        if response
            .content_length()
            .is_some_and(|length| length > remaining as u64)
        {
            return Err(Error::Limit(
                "Batch result files exceed their combined byte limit",
            ));
        }
        let mut response = BufReader::new(response);
        loop {
            let remaining = self.limits.result_bytes - collection.bytes;
            let row = input::line(&mut response, self.limits.line_bytes.min(remaining))?;
            if row.is_empty() {
                break;
            }
            collection.bytes += row.len();
            if row.trim_ascii().is_empty() {
                continue;
            }
            let value: Value = serde_json::from_slice(&row).map_err(|_| Error::Protocol)?;
            collection.push(message_result(value)?)?;
        }
        Ok(())
    }

    fn control(&self, response: Response) -> Result<Value, Error> {
        self.control_limited(response, self.limits.control_bytes)
            .map(|(value, _)| value)
    }

    fn control_limited(&self, response: Response, limit: usize) -> Result<(Value, usize), Error> {
        if !response.status().is_success() {
            return Err(Error::Http(response.status().as_u16()));
        }
        if response
            .content_length()
            .is_some_and(|size| size > limit as u64)
        {
            return Err(Error::Limit(
                "Batch control response exceeds its byte limit",
            ));
        }
        let bytes = crate::platform::read_limited(response, limit as u64)
            .map_err(|_| Error::Transport)?
            .ok_or(Error::Limit(
                "Batch control response exceeds its byte limit",
            ))?;
        let value = serde_json::from_slice(&bytes).map_err(|_| Error::Protocol)?;
        Ok((value, bytes.len()))
    }
}

/// One Message Batch result row in the OpenAI result shape: a succeeded
/// message is a 200 response; errored, canceled and expired requests (which
/// Anthropic does not bill) are errors without a body.
fn message_result(value: Value) -> Result<Value, Error> {
    let object = value.as_object().ok_or(Error::Protocol)?;
    let id = object
        .get("custom_id")
        .and_then(Value::as_str)
        .ok_or(Error::Protocol)?;
    let result = object
        .get("result")
        .and_then(Value::as_object)
        .ok_or(Error::Protocol)?;
    match result.get("type").and_then(Value::as_str) {
        Some("succeeded") => {
            let message = result
                .get("message")
                .filter(|message| message.is_object())
                .ok_or(Error::Protocol)?;
            let mut response = json!({"status_code":200,"body":message});
            if let Some(request) = message.get("id").and_then(Value::as_str) {
                response["request_id"] = json!(request);
            }
            Ok(json!({"custom_id":id,"response":response}))
        }
        Some("errored") => {
            let error = result
                .get("error")
                .filter(|error| error.is_object())
                .ok_or(Error::Protocol)?;
            Ok(json!({"custom_id":id,"error":error}))
        }
        Some(kind @ ("canceled" | "expired")) => Ok(json!({"custom_id":id,"error":{"type":kind}})),
        _ => Err(Error::Protocol),
    }
}

/// A Message Batch as the shared batch record. Its requests were inline, so
/// its input is [`INLINE_INPUT`]; its results have no file IDs. An ended batch
/// is completed (even with failed requests, as their results say), cancelled
/// when cancellation began, or expired when nothing ran before the deadline.
fn parse_message_batch(value: Value) -> Result<Batch, Error> {
    let object = value.as_object().ok_or(Error::Protocol)?;
    if object
        .get("type")
        .is_some_and(|kind| kind != "message_batch")
    {
        return Err(Error::Protocol);
    }
    let id = object
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| identifier(id, "msgbatch"))
        .ok_or(Error::Protocol)?
        .to_owned();
    let counts = object
        .get("request_counts")
        .and_then(Value::as_object)
        .ok_or(Error::Protocol)?;
    let count = |name| -> Result<u64, Error> {
        match counts.get(name) {
            None | Some(Value::Null) => Ok(0),
            Some(value) => value.as_u64().ok_or(Error::Protocol),
        }
    };
    let (processing, succeeded, errored, canceled, expired) = (
        count("processing")?,
        count("succeeded")?,
        count("errored")?,
        count("canceled")?,
        count("expired")?,
    );
    let failed = errored
        .checked_add(canceled)
        .and_then(|value| value.checked_add(expired))
        .ok_or(Error::Protocol)?;
    let total = processing
        .checked_add(succeeded)
        .and_then(|value| value.checked_add(failed))
        .ok_or(Error::Protocol)?;
    let cancelled = match object.get("cancel_initiated_at") {
        None | Some(Value::Null) => false,
        Some(Value::String(_)) => true,
        _ => return Err(Error::Protocol),
    };
    let status = match object.get("processing_status").and_then(Value::as_str) {
        Some("in_progress") => BatchStatus::InProgress,
        Some("canceling") => BatchStatus::Cancelling,
        Some("ended") if cancelled => BatchStatus::Cancelled,
        Some("ended") if succeeded == 0 && errored == 0 && expired > 0 => BatchStatus::Expired,
        Some("ended") => BatchStatus::Completed,
        _ => return Err(Error::Protocol),
    };
    Ok(Batch {
        id,
        input_file_id: INLINE_INPUT.into(),
        status,
        output_file_id: None,
        error_file_id: None,
        total: Some(total),
        completed: Some(succeeded),
        failed: Some(failed),
    })
}

fn validate_recovery(uploaded: &UploadedInput, nonce: &str) -> Result<(), Error> {
    if !identifier(&uploaded.file_id, "file") || !custom_id(nonce) {
        return Err(Error::Invalid("Batch recovery identity is invalid"));
    }
    Ok(())
}
fn remote_nonce(value: &Value) -> Result<Option<String>, Error> {
    let metadata = match value.get("metadata") {
        None | Some(Value::Null) => return Ok(None),
        Some(Value::Object(metadata)) => metadata,
        _ => return Err(Error::Protocol),
    };
    match metadata.get("markitai_submission") {
        None => Ok(None),
        Some(Value::String(nonce)) if nonce.len() <= 2048 => Ok(Some(nonce.clone())),
        _ => Err(Error::Protocol),
    }
}
fn parse_remote(value: Value, api_base: &str) -> Result<RemoteIdentity, Error> {
    let endpoint = value
        .get("endpoint")
        .and_then(Value::as_str)
        .filter(|endpoint| *endpoint == ENDPOINT)
        .ok_or(Error::Protocol)?
        .to_owned();
    let submission_nonce = remote_nonce(&value)?;
    Ok(RemoteIdentity {
        batch: parse_batch(value)?,
        endpoint,
        api_base: api_base.into(),
        submission_nonce,
    })
}

fn parse_batch(value: Value) -> Result<Batch, Error> {
    let object = value.as_object().ok_or(Error::Protocol)?;
    let id = object
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| identifier(id, "batch"))
        .ok_or(Error::Protocol)?
        .to_owned();
    let input_file_id = object
        .get("input_file_id")
        .and_then(Value::as_str)
        .filter(|id| identifier(id, "file"))
        .ok_or(Error::Protocol)?
        .to_owned();
    if object
        .get("endpoint")
        .is_some_and(|value| value != ENDPOINT)
    {
        return Err(Error::Protocol);
    }
    let status = serde_json::from_value(object.get("status").cloned().ok_or(Error::Protocol)?)
        .map_err(|_| Error::Protocol)?;
    let file_id = |name| -> Result<Option<String>, Error> {
        match object.get(name) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::String(id)) if identifier(id, "file") => Ok(Some(id.clone())),
            _ => Err(Error::Protocol),
        }
    };
    let count = |name| -> Result<Option<u64>, Error> {
        match object.get("request_counts") {
            None | Some(Value::Null) => Ok(None),
            Some(Value::Object(counts)) => counts
                .get(name)
                .map(|value| value.as_u64().ok_or(Error::Protocol))
                .transpose(),
            _ => Err(Error::Protocol),
        }
    };
    let total = count("total")?;
    let completed = count("completed")?;
    let failed = count("failed")?;
    if let (Some(total), Some(completed), Some(failed)) = (total, completed, failed)
        && completed
            .checked_add(failed)
            .is_none_or(|done| done > total)
    {
        return Err(Error::Protocol);
    }
    Ok(Batch {
        id,
        input_file_id,
        status,
        output_file_id: file_id("output_file_id")?,
        error_file_id: file_id("error_file_id")?,
        total,
        completed,
        failed,
    })
}

struct Collection<'a> {
    status: BatchStatus,
    expected: &'a [String],
    positions: HashMap<&'a str, usize>,
    slots: Vec<Option<ResultItem>>,
    bytes: usize,
}
impl<'a> Collection<'a> {
    fn new(status: BatchStatus, expected: &'a [String], max: usize) -> Result<Self, Error> {
        if expected.is_empty() || expected.len() > max {
            return Err(Error::Invalid("Batch expected request count is invalid"));
        }
        let mut positions = HashMap::with_capacity(expected.len());
        for (index, id) in expected.iter().enumerate() {
            if !custom_id(id) || positions.insert(id.as_str(), index).is_some() {
                return Err(Error::Invalid(
                    "Batch expected IDs are invalid or duplicated",
                ));
            }
        }
        Ok(Self {
            status,
            expected,
            positions,
            slots: vec![None; expected.len()],
            bytes: 0,
        })
    }
    fn push(&mut self, value: Value) -> Result<(), Error> {
        let object = value.as_object().ok_or(Error::Protocol)?;
        let id = object
            .get("custom_id")
            .and_then(Value::as_str)
            .ok_or(Error::Protocol)?;
        let position = *self
            .positions
            .get(id)
            .ok_or(Error::Invalid("Batch result contains an unknown custom_id"))?;
        if self.slots[position].is_some() {
            return Err(Error::Invalid(
                "Batch result contains a duplicate custom_id",
            ));
        }
        let error = object
            .get("error")
            .filter(|value| !value.is_null())
            .cloned();
        if error.as_ref().is_some_and(|value| !value.is_object()) {
            return Err(Error::Protocol);
        }
        let response = object.get("response").filter(|value| !value.is_null());
        let (http_status, body, request_id) = match response {
            Some(Value::Object(response)) => {
                let status = response
                    .get("status_code")
                    .and_then(Value::as_u64)
                    .filter(|code| (100..600).contains(code))
                    .ok_or(Error::Protocol)?;
                let body = response.get("body").filter(|body| !body.is_null()).cloned();
                let request_id = match response.get("request_id") {
                    Some(Value::String(id))
                        if id.len() <= 256 && !id.chars().any(char::is_control) =>
                    {
                        Some(id.clone())
                    }
                    // Optional provider metadata must not erase an attributable paid body.
                    _ => None,
                };
                (Some(status as u16), body, request_id)
            }
            None if error.is_some() => (None, None, None),
            _ => return Err(Error::Protocol),
        };
        self.slots[position] = Some(ResultItem {
            custom_id: id.to_owned(),
            http_status,
            body,
            error,
            request_id,
        });
        Ok(())
    }
    fn finish(self) -> Results {
        let mut items = Vec::new();
        let mut missing = Vec::new();
        for (index, item) in self.slots.into_iter().enumerate() {
            match item {
                Some(item) => items.push(item),
                None => missing.push(self.expected[index].clone()),
            }
        }
        Results {
            status: self.status,
            items,
            missing,
            bytes: self.bytes,
        }
    }
}

#[cfg(test)]
pub(super) fn tests_parse(value: Value) -> Result<Batch, Error> {
    parse_message_batch(value)
}
