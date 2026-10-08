//! Remote extraction services: defuddle.md, Jina Reader and Cloudflare
//! Browser Rendering (in [`super::cloudflare`]).
//!
//! Each call waits for its service's requests-per-minute window
//! (`fetch.defuddle.rpm`, `fetch.jina.rpm`). A failure reads
//! `HTTP <status> from the <service> service`, then the service's own words
//! when it gave a JSON reason and a hint for the common refusals; it never
//! contains a token, an account id or the service's endpoint.

use super::consent::Gate;
use super::{FetchContent, FetchOutcome, client, request_error, send, sites};
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
        Service::Defuddle => defuddle(url, cfg, services)?,
        Service::Jina => jina(source, url, cfg, services)?,
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

/// Whether this conversion asked for a fresh reading of `source`: markitai's
/// own page cache is bypassed for it (`--no-cache`, or a `--no-cache-for` /
/// `cache.no_cache_patterns` entry that matches), so a remote service's own
/// cache should not answer either.
fn fresh_reading(cfg: &Value, source: &str) -> bool {
    if config::enabled(cfg, "/cache/no_cache") {
        return true;
    }
    let patterns: Vec<String> = cfg
        .pointer("/cache/no_cache_patterns")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect();
    crate::fetch_cache::url_matches_patterns(source, &patterns)
}

/// A reading that is the site's refusal page (a verification or login page,
/// a challenge, a notice that asks for JavaScript) or the site's JSON error
/// answer is the service failing, judged as the local readers judge markup
/// (see [`sites::Shown::read_by`]). `url` is where the service says it read
/// the page.
fn judge(service: Service, url: &Url, document: &Document) -> Result<()> {
    let title = document.metadata.get("title").and_then(Value::as_str);
    match sites::Shown::markdown(url, title, &document.markdown).read_by(service.name()) {
        Some(failure) => Err(Error::Fetch(failure)),
        None => Ok(()),
    }
}

/// The failure for the status the page itself answered a service with
/// (Jina's `httpStatus`, Cloudflare's `meta.status`), read as local fetching
/// reads a status: a refusal (401, 403, 418, 429) is the site turning the
/// service away, any other status from 400 up is the page's failure. `url`
/// is where the service says it read the page; `None` below 400.
pub(crate) fn origin_status(service: Service, status: u16, url: &Url) -> Option<Error> {
    if status < 400 {
        return None;
    }
    let name = service.name();
    Some(Error::Fetch(if super::refusal_status(status) {
        sites::refused_status(name, status)
    } else {
        match super::status_hint(status, url, false) {
            Some(hint) => {
                format!("The {name} service received HTTP {status} from the site: {hint}")
            }
            None => format!("The {name} service received HTTP {status} from the site"),
        }
    }))
}

/// The address a service says it read, when it is an http(s) URL; else the
/// requested one.
pub(crate) fn read_at(said: Option<&str>, url: &Url) -> Url {
    said.and_then(|said| Url::parse(said.trim()).ok())
        .filter(|read| matches!(read.scheme(), "http" | "https") && read.host_str().is_some())
        .unwrap_or_else(|| url.clone())
}

fn defuddle(url: &Url, cfg: &Value, services: &Services<'_>) -> Result<Document> {
    services
        .limits
        .acquire("defuddle", per_minute(cfg, "/fetch/defuddle/rpm"));
    let endpoint = format!(
        "{}/{}",
        services.defuddle.trim_end_matches('/'),
        url::form_urlencoded::byte_serialize(url.as_str().as_bytes()).collect::<String>()
    );
    let client = client(seconds(cfg, "/fetch/defuddle/timeout"))?;
    // defuddle.md documents no cache opt-out; its answers carry
    // `Cache-Control: s-maxage=300`, so a reading can be minutes old.
    let response = send(client.get(endpoint)).map_err(|error| named(Service::Defuddle, error))?;
    let bytes = answer(Service::Defuddle, response, &[])?;
    let text = String::from_utf8(bytes).map_err(|_| {
        Error::Fetch("The defuddle service returned Markdown that is not UTF-8".into())
    })?;
    let (metadata, markdown) = output::split_frontmatter(&text);
    let document = Document {
        markdown: markdown.trim().into(),
        metadata,
        ..Default::default()
    };
    judge(Service::Defuddle, url, &document)?;
    Ok(document)
}

/// What Jina Reader said about a page: its JSON `data` fields, or the header
/// lines of its text answer (`Title:`, `URL Source:`, `Warning:`, …, then
/// `Markdown Content:` and the page).
#[derive(Default)]
struct JinaReading {
    title: Option<String>,
    url: Option<String>,
    warnings: Vec<String>,
    /// The status the page itself answered Jina with.
    status: Option<u16>,
    content: String,
}

/// A header name of Jina's text answer: a few capitalized words.
fn jina_header(key: &str) -> bool {
    (1..=40).contains(&key.len())
        && key.starts_with(|ch: char| ch.is_ascii_uppercase())
        && key
            .chars()
            .all(|ch| ch.is_ascii_alphabetic() || ch == ' ' || ch == '-')
}

/// Jina's text answer, when `text` is one: header lines (blank lines between
/// them allowed), then a `… Content:` line after which the page follows.
/// Anything else is not taken for headers, so a page is never cut.
fn jina_text(text: &str) -> Option<JinaReading> {
    let mut reading = JinaReading::default();
    let mut rest = text.trim_start_matches('\u{feff}');
    while !rest.is_empty() {
        let (line, after) = rest.split_once('\n').unwrap_or((rest, ""));
        rest = after;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let (key, value) = line.split_once(':')?;
        let (key, value) = (key.trim(), value.trim());
        if !jina_header(key) {
            return None;
        }
        if key.ends_with(" Content") {
            let content = if value.is_empty() {
                after.to_owned()
            } else {
                format!("{value}\n{after}")
            };
            reading.content = content.replace("\r\n", "\n");
            return Some(reading);
        }
        if value.is_empty() {
            continue;
        }
        match key.to_ascii_lowercase().as_str() {
            "title" => reading.title = Some(value.to_owned()),
            "url source" => reading.url = Some(value.to_owned()),
            "warning" => reading.warnings.push(value.to_owned()),
            _ => {}
        }
    }
    None
}

/// The status in a Jina warning such as `Target URL returned error 403: Forbidden`.
fn warned_status(warning: &str) -> Option<u16> {
    let (_, after) = warning.split_once("returned error ")?;
    after
        .split(|ch: char| !ch.is_ascii_digit())
        .next()?
        .parse()
        .ok()
        .filter(|status| (100..600).contains(status))
}

/// Jina's answer: JSON (`{"data": {"title", "url", "content", "warning",
/// "httpStatus", …}}`), or the text form when the JSON was not honoured.
fn jina_reading(bytes: &[u8], secrets: &[&str]) -> Result<JinaReading> {
    let Ok(value) = serde_json::from_slice::<Value>(bytes) else {
        return std::str::from_utf8(bytes)
            .ok()
            .and_then(jina_text)
            .ok_or_else(|| {
                Error::Fetch("The jina service returned an answer that is not JSON".into())
            });
    };
    let Some(data) = value.get("data").filter(|data| data.is_object()) else {
        return Err(Error::Fetch(match service_said(bytes, secrets) {
            Some(said) => format!("The jina service returned no page: {said}"),
            None => "The jina service returned no page".into(),
        }));
    };
    let text = |key: &str| {
        data.get(key)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .map(str::to_owned)
    };
    let content = data
        .get("content")
        .and_then(Value::as_str)
        .unwrap_or_default();
    // Header lines of the text form inside the content are Jina's, not the page's.
    let mut reading = jina_text(content)
        .filter(|_| content.trim_start().starts_with("Title:"))
        .unwrap_or_else(|| JinaReading {
            content: content.to_owned(),
            ..Default::default()
        });
    if let Some(title) = text("title") {
        reading.title = Some(title);
    }
    if let Some(url) = text("url") {
        reading.url = Some(url);
    }
    reading.warnings.extend(text("warning"));
    reading.status = data
        .get("httpStatus")
        .and_then(Value::as_u64)
        .and_then(|status| u16::try_from(status).ok());
    Ok(reading)
}

fn jina(source: &str, url: &Url, cfg: &Value, services: &Services<'_>) -> Result<Document> {
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
        .get(format!("{}/{url}", services.jina.trim_end_matches('/')))
        .header(reqwest::header::ACCEPT, "application/json");
    if let Some(key) = &key {
        request = request.bearer_auth(key);
    }
    let fresh = config::enabled(cfg, "/fetch/jina/no_cache") || fresh_reading(cfg, source);
    if fresh {
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
    let reading = jina_reading(&bytes, &secrets)?;
    let read_at = read_at(reading.url.as_deref(), url);
    // The page's own status, read as local fetching reads it: a refusal is
    // the site turning Jina away, any other failure is the page's.
    if let Some(failure) = reading
        .status
        .or_else(|| {
            reading
                .warnings
                .iter()
                .find_map(|warning| warned_status(warning))
        })
        .and_then(|status| origin_status(Service::Jina, status, &read_at))
    {
        return Err(failure);
    }
    let mut document = Document {
        markdown: reading.content,
        ..Default::default()
    };
    if let Some(title) = reading.title {
        document.metadata.insert("title".into(), json!(title));
    }
    judge(Service::Jina, &read_at, &document)?;
    for warning in reading.warnings {
        let warning = without_secrets(&warning, &secrets);
        let mut said = format!("The jina service said: {warning}");
        if !fresh && warning.to_ascii_lowercase().contains("cached snapshot") {
            said.push_str(
                " Run with --no-cache (or set fetch.jina.no_cache) to ask Jina for a fresh reading.",
            );
        }
        document.warnings.push(said);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jina_text_header_lines_are_read_and_anything_else_is_left_alone() {
        let reading = jina_text("\u{feff}Title: A: B\r\n\r\nURL Source: https://example.com/a\r\nPublished Time: 2026-01-01\r\nWarning: first\r\nWarning: second\r\n\r\nMarkdown Content:\r\n# A\r\n\r\nBody.").unwrap();
        assert_eq!(reading.title.as_deref(), Some("A: B"));
        assert_eq!(reading.url.as_deref(), Some("https://example.com/a"));
        assert_eq!(reading.warnings, ["first", "second"]);
        assert_eq!(reading.content, "# A\n\nBody.");
        // Other content markers, and content on the marker's line.
        assert_eq!(
            jina_text("Title: T\nText Content: one line")
                .unwrap()
                .content,
            "one line\n"
        );
        for not_headers in [
            "# A heading\n\nTitle: later",
            "Title: Only headers\nURL Source: https://example.com/",
            "title: lower case\nMarkdown Content:\nx",
            "Plain text without a colon",
            "",
        ] {
            assert!(jina_text(not_headers).is_none(), "{not_headers}");
        }
        assert_eq!(
            warned_status("Target URL returned error 403: Forbidden"),
            Some(403)
        );
        assert_eq!(warned_status("Target URL returned error 99999"), None);
        assert_eq!(warned_status("This is a cached snapshot"), None);
    }

    #[test]
    fn a_fresh_reading_follows_the_page_cache_bypass() {
        let url = "https://example.com/docs/page";
        assert!(!fresh_reading(&json!({}), url));
        assert!(fresh_reading(&json!({"cache": {"no_cache": true}}), url));
        assert!(fresh_reading(
            &json!({"cache": {"no_cache_patterns": ["example.com"]}}),
            url
        ));
        assert!(fresh_reading(
            &json!({"cache": {"no_cache_patterns": ["https://example.com/docs/*"]}}),
            url
        ));
        assert!(!fresh_reading(
            &json!({"cache": {"no_cache_patterns": ["other.test"]}}),
            url
        ));
        // A cache that is off is not a request for a fresh reading.
        assert!(!fresh_reading(&json!({"cache": {"enabled": false}}), url));
    }
}
