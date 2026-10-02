//! Remote extraction services: defuddle.md, Jina Reader and Cloudflare
//! Browser Rendering (in [`super::cloudflare`]).
//!
//! Each call waits for its service's requests-per-minute window
//! (`fetch.defuddle.rpm`, `fetch.jina.rpm`). A failure reads
//! `HTTP <status> from the <service> service`, then the service's own words
//! when it gave a JSON reason and a hint for the common refusals; it never
//! contains a token, an account id or the service's endpoint.

use super::consent::Gate;
use super::{FetchContent, FetchOutcome, client, request_error, send};
use crate::{Document, Error, Result, config, output};
use reqwest::blocking::Response;
use serde_json::{Value, json};
use std::collections::{HashMap, VecDeque};
use std::io::Read;
use std::sync::{Condvar, LazyLock, Mutex, PoisonError};
use std::time::{Duration, Instant};
use url::Url;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Service {
    Defuddle,
    Jina,
    Cloudflare,
}

impl Service {
    pub(crate) fn named(name: &str) -> Option<Self> {
        match name {
            "defuddle" => Some(Self::Defuddle),
            "jina" => Some(Self::Jina),
            "cloudflare" => Some(Self::Cloudflare),
            _ => None,
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Defuddle => "defuddle",
            Self::Jina => "jina",
            Self::Cloudflare => "cloudflare",
        }
    }
}

const DEFUDDLE: &str = "https://defuddle.md";
const JINA: &str = "https://r.jina.ai";
pub(crate) const CLOUDFLARE: &str = "https://api.cloudflare.com/client/v4";

/// Where remote requests go and the process facts they read. Production uses
/// the real endpoints and environment; tests point every field at loopback
/// fixtures, so no test reads the user's environment or `.env` files.
pub(crate) struct Services<'a> {
    pub defuddle: String,
    pub jina: String,
    pub cloudflare: String,
    /// The environment, with the `.env` files configuration reads.
    pub vars: HashMap<String, String>,
    /// Whether a host name is local or private without resolving it.
    pub private_name: fn(&Url) -> bool,
    /// Resolves the host and refuses non-public addresses.
    pub public_addresses: fn(&Url) -> Result<()>,
    pub browser_ready: fn() -> bool,
    pub limits: &'a Limits,
    pub slots: &'a Slots,
    pub gate: &'a Gate,
    /// The first pause after a Cloudflare 429; the second is twice as long.
    pub retry_pause: Duration,
}

static LIMITS: LazyLock<Limits> = LazyLock::new(|| Limits::new(Duration::from_secs(60)));
/// Cloudflare's free plan runs two browsers at a time.
static SLOTS: Slots = Slots::new(2);

impl Services<'static> {
    pub(crate) fn production() -> Self {
        Self {
            defuddle: DEFUDDLE.into(),
            jina: JINA.into(),
            cloudflare: CLOUDFLARE.into(),
            vars: config::environment(),
            private_name: super::policy::private_name,
            public_addresses: super::public_addresses,
            browser_ready: crate::browser::available,
            limits: &LIMITS,
            slots: &SLOTS,
            gate: Gate::installed(),
            retry_pause: Duration::from_secs(2),
        }
    }
}

/// Services for tests: every endpoint under one loopback origin (`/defuddle`,
/// `/jina`, `/cloudflare`), an empty environment, no browser, no name or
/// address refused, and short pauses.
#[cfg(test)]
pub(crate) struct Fixture {
    pub limits: Limits,
    pub slots: Slots,
    pub gate: Gate,
}

#[cfg(test)]
impl Fixture {
    pub(crate) fn new(gate: Gate) -> Self {
        Self {
            limits: Limits::new(Duration::from_secs(60)),
            slots: Slots::new(2),
            gate,
        }
    }

    pub(crate) fn services(&self, origin: &str) -> Services<'_> {
        Services {
            defuddle: format!("{origin}/defuddle"),
            jina: format!("{origin}/jina"),
            cloudflare: format!("{origin}/cloudflare"),
            vars: HashMap::new(),
            private_name: |_| false,
            public_addresses: |_| Ok(()),
            browser_ready: || false,
            limits: &self.limits,
            slots: &self.slots,
            gate: &self.gate,
            retry_pause: Duration::from_millis(5),
        }
    }
}

// ---- pacing ---------------------------------------------------------------

/// A sliding window of request times per service.
pub(crate) struct Limits {
    window: Duration,
    sent: Mutex<HashMap<&'static str, VecDeque<Instant>>>,
}

impl Limits {
    pub(crate) fn new(window: Duration) -> Self {
        Self {
            window,
            sent: Mutex::new(HashMap::new()),
        }
    }

    /// Wait until fewer than `per_window` requests went to `service` within
    /// the window, then count this one.
    pub(crate) fn acquire(&self, service: &'static str, per_window: u64) {
        let per_window = usize::try_from(per_window.max(1)).unwrap_or(usize::MAX);
        loop {
            let wait = {
                let mut sent = self.sent.lock().unwrap_or_else(PoisonError::into_inner);
                let times = sent.entry(service).or_default();
                let now = Instant::now();
                while times
                    .front()
                    .is_some_and(|time| now.duration_since(*time) >= self.window)
                {
                    times.pop_front();
                }
                if times.len() < per_window {
                    times.push_back(now);
                    return;
                }
                self.window
                    .saturating_sub(now.duration_since(*times.front().expect("a full window")))
            };
            std::thread::sleep(wait.max(Duration::from_millis(1)));
        }
    }
}

/// A counting semaphore for concurrent remote browsers.
pub(crate) struct Slots {
    used: Mutex<usize>,
    freed: Condvar,
    capacity: usize,
}

pub(crate) struct Slot<'a>(&'a Slots);

impl Slots {
    pub(crate) const fn new(capacity: usize) -> Self {
        Self {
            used: Mutex::new(0),
            freed: Condvar::new(),
            capacity,
        }
    }

    pub(crate) fn hold(&self) -> Slot<'_> {
        let mut used = self.used.lock().unwrap_or_else(PoisonError::into_inner);
        while *used >= self.capacity {
            used = self
                .freed
                .wait(used)
                .unwrap_or_else(PoisonError::into_inner);
        }
        *used += 1;
        Slot(self)
    }
}

impl Drop for Slot<'_> {
    fn drop(&mut self) {
        let mut used = self.0.used.lock().unwrap_or_else(PoisonError::into_inner);
        *used -= 1;
        self.0.freed.notify_one();
    }
}

// ---- answers --------------------------------------------------------------

/// Bytes of a failed answer read for the service's own words.
const REASON_BYTES: u64 = 8 * 1024;

/// Every occurrence of a secret replaced, for text a service sent back.
pub(crate) fn without_secrets(text: &str, secrets: &[&str]) -> String {
    let mut text = text.to_owned();
    for secret in secrets.iter().filter(|secret| secret.len() >= 4) {
        text = text.replace(secret, "REDACTED");
    }
    text
}

/// The reason a service gave in a JSON answer: Jina's `readableMessage` or
/// `message`, Cloudflare's `errors[0]`, or the usual error fields.
pub(crate) fn service_said(body: &[u8], secrets: &[&str]) -> Option<String> {
    let value: Value = serde_json::from_slice(body).ok()?;
    let clean = |text: &str| {
        let text: String = text
            .chars()
            .filter(|ch| !ch.is_control() || ch.is_whitespace())
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        let text = without_secrets(&text, secrets);
        let mut said: String = text.chars().take(160).collect();
        if said.chars().count() < text.chars().count() {
            said.push('…');
        }
        said
    };
    let first_error = value.pointer("/errors/0");
    let message = ["/readableMessage", "/errors/0/message"]
        .iter()
        .find_map(|pointer| value.pointer(pointer).and_then(Value::as_str))
        .filter(|text| !text.trim().is_empty());
    match message {
        Some(message) => {
            let code = first_error
                .and_then(|error| error.get("code"))
                .and_then(Value::as_i64);
            Some(match code {
                Some(code) => format!("{} (code {code})", clean(message)),
                None => clean(message),
            })
        }
        None => super::sites::json_refusal(body).map(|said| clean(&said)),
    }
}

fn hint(service: Service, status: u16) -> Option<&'static str> {
    match (service, status) {
        (Service::Jina, 401 | 402 | 429 | 451) => Some(
            "set fetch.jina.api_key or JINA_API_KEY; anonymous requests are rate limited and refused for some sites",
        ),
        (Service::Cloudflare, 401 | 403) => Some(
            "check fetch.cloudflare.api_token (it needs Account / Browser Rendering / Edit for -s cloudflare, Account / Workers AI / Read for -b cloudflare) and fetch.cloudflare.account_id",
        ),
        (_, 429) => Some("rate limited; try again later"),
        (_, 500..=599) => Some("the service had a server error"),
        _ => None,
    }
}

/// The failure a service's status reports.
pub(crate) fn status_failure(service: Service, response: Response, secrets: &[&str]) -> Error {
    let status = response.status().as_u16();
    let mut body = Vec::new();
    let _ = response.take(REASON_BYTES).read_to_end(&mut body);
    let mut message = format!("HTTP {status} from the {} service", service.name());
    let parts: Vec<String> = service_said(&body, secrets)
        .into_iter()
        .chain(hint(service, status).map(str::to_owned))
        .collect();
    if !parts.is_empty() {
        message.push_str(": ");
        message.push_str(&parts.join("; "));
    }
    Error::Fetch(message)
}

/// The body of a successful answer, or the failure its status reports.
pub(crate) fn answer(service: Service, response: Response, secrets: &[&str]) -> Result<Vec<u8>> {
    if !response.status().is_success() {
        return Err(status_failure(service, response, secrets));
    }
    super::body(response)
}

/// A request error that names its service, without the endpoint (it can hold
/// an account id).
pub(crate) fn transport(service: Service, error: reqwest::Error) -> Error {
    match request_error(error) {
        Error::Fetch(message) => Error::Fetch(format!(
            "The {} service could not be reached: {message}",
            service.name()
        )),
        other => other,
    }
}

fn seconds(cfg: &Value, pointer: &str) -> u64 {
    cfg.pointer(pointer)
        .and_then(Value::as_u64)
        .unwrap_or(30)
        .clamp(1, 600)
}

fn per_minute(cfg: &Value, pointer: &str) -> u64 {
    cfg.pointer(pointer)
        .and_then(Value::as_u64)
        .unwrap_or(20)
        .max(1)
}

// ---- services -------------------------------------------------------------

/// The page through `service`, as a document whose `fetch_strategy` names it.
pub(crate) fn fetch(
    service: Service,
    source: &str,
    url: &Url,
    cfg: &Value,
    services: &Services<'_>,
) -> Result<Document> {
    let mut document = match service {
        Service::Defuddle => defuddle(source, cfg, services)?,
        Service::Jina => jina(source, cfg, services)?,
        Service::Cloudflare => super::cloudflare::render(source, url, cfg, services)?,
    };
    if document.markdown.trim().is_empty() {
        return Err(Error::Fetch(format!(
            "The {} service returned empty content",
            service.name()
        )));
    }
    document
        .metadata
        .insert("fetch_strategy".into(), json!(service.name()));
    Ok(document)
}

pub(crate) fn outcome(document: Document) -> FetchOutcome {
    FetchOutcome {
        content: FetchContent::Document(document),
        cache_hit: false,
        screenshots: Vec::new(),
    }
}

fn defuddle(source: &str, cfg: &Value, services: &Services<'_>) -> Result<Document> {
    services
        .limits
        .acquire("defuddle", per_minute(cfg, "/fetch/defuddle/rpm"));
    let endpoint = format!(
        "{}/{}",
        services.defuddle.trim_end_matches('/'),
        url::form_urlencoded::byte_serialize(source.as_bytes()).collect::<String>()
    );
    let client = client(seconds(cfg, "/fetch/defuddle/timeout"))?;
    let response = send(client.get(endpoint)).map_err(|error| named(Service::Defuddle, error))?;
    let bytes = answer(Service::Defuddle, response, &[])?;
    let text = String::from_utf8(bytes).map_err(|_| {
        Error::Fetch("The defuddle service returned Markdown that is not UTF-8".into())
    })?;
    let (metadata, markdown) = output::split_frontmatter(&text);
    Ok(Document {
        markdown: markdown.trim().into(),
        metadata,
        ..Default::default()
    })
}

fn jina(source: &str, cfg: &Value, services: &Services<'_>) -> Result<Document> {
    let text = |pointer: &str| {
        cfg.pointer(pointer)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
    };
    let key = config::resolve_optional(
        text("/fetch/jina/api_key"),
        Some("JINA_API_KEY"),
        &services.vars,
        false,
    )?
    .filter(|key| !key.trim().is_empty());
    if key
        .as_deref()
        .is_some_and(|key| key.chars().any(|ch| ch.is_control() || ch.is_whitespace()))
    {
        return Err(Error::Config(
            "fetch.jina.api_key is not a valid API key".into(),
        ));
    }
    services
        .limits
        .acquire("jina", per_minute(cfg, "/fetch/jina/rpm"));
    let client = client(seconds(cfg, "/fetch/jina/timeout"))?;
    let mut request = client
        .get(format!("{}/{source}", services.jina.trim_end_matches('/')))
        .header(reqwest::header::ACCEPT, "application/json");
    if let Some(key) = &key {
        request = request.bearer_auth(key);
    }
    if config::enabled(cfg, "/fetch/jina/no_cache") {
        request = request.header("X-No-Cache", "true");
    }
    for (pointer, header) in [
        ("/fetch/jina/target_selector", "X-Target-Selector"),
        ("/fetch/jina/wait_for_selector", "X-Wait-For-Selector"),
    ] {
        if let Some(selector) = text(pointer) {
            request = request.header(header, selector);
        }
    }
    let secrets: Vec<&str> = key.as_deref().into_iter().collect();
    let response = send(request).map_err(|error| named(Service::Jina, error))?;
    let bytes = answer(Service::Jina, response, &secrets)?;
    let value: Value = serde_json::from_slice(&bytes)
        .map_err(|_| Error::Fetch("The jina service returned an answer that is not JSON".into()))?;
    let Some(data) = value.get("data").filter(|data| data.is_object()) else {
        return Err(Error::Fetch(match service_said(&bytes, &secrets) {
            Some(said) => format!("The jina service returned no page: {said}"),
            None => "The jina service returned no page".into(),
        }));
    };
    let mut document = Document {
        markdown: data
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        ..Default::default()
    };
    if let Some(title) = data
        .get("title")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|title| !title.is_empty())
    {
        document.metadata.insert("title".into(), json!(title));
    }
    Ok(document)
}

/// A request failure that names the service it was sent to.
fn named(service: Service, error: Error) -> Error {
    match error {
        Error::Fetch(message) if !message.contains(service.name()) => Error::Fetch(format!(
            "The {} service could not be reached: {message}",
            service.name()
        )),
        other => other,
    }
}
