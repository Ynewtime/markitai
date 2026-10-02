use crate::{
    Asset, Document, Error, Result, browser, config, fetch_cache, formats, output, spa_domains,
};
use reqwest::blocking::{Client, Response};
use serde_json::{Value, json};
use std::io::{Read, Write};
use std::net::{IpAddr, ToSocketAddrs};
use std::time::Duration;
use url::Url;

mod sites;

const MAX_RESPONSE: u64 = 100 * 1024 * 1024;

pub(crate) fn client(timeout: u64) -> Result<Client> {
    crate::proxy::http(
        Client::builder()
            .timeout(Duration::from_secs(timeout))
            .connect_timeout(Duration::from_secs(15))
            .redirect(reqwest::redirect::Policy::limited(10))
            .user_agent(concat!("markitai/", env!("CARGO_PKG_VERSION"))),
    )?
    .build()
    .map_err(request_error)
}

/// A request failure with the causes reqwest keeps behind its summary
/// ("error sending request" alone gives no reason, such as a refused
/// connection or a name that does not resolve), without the URL, which can
/// carry credentials.
pub(crate) fn request_error(error: reqwest::Error) -> Error {
    let error = error.without_url();
    let mut message = error.to_string();
    let mut source = std::error::Error::source(&error);
    while let Some(cause) = source {
        let cause_text = cause.to_string();
        if !message.contains(&cause_text) {
            message.push_str(": ");
            message.push_str(&cause_text);
        }
        source = cause.source();
    }
    Error::Fetch(message)
}

/// Pauses before the second and third attempt of a GET whose connection failed
/// before the server replied (see [`send`]).
const RETRY_PAUSES: [Duration; 2] = [Duration::from_millis(250), Duration::from_millis(750)];

/// Send a body-less GET, repeating it when the connection could not be
/// established because the peer reset it or cut it off, such as a TLS handshake
/// that ends early. Nothing was sent then, so the repeat cannot duplicate
/// work. Nothing else is retried: not a timeout, a refused connection, a name
/// that does not resolve, a certificate failure, a connection that fails after
/// the request went out or any HTTP status, so a 4xx or 5xx answer is reported
/// at once.
fn send(request: reqwest::blocking::RequestBuilder) -> Result<Response> {
    let mut request = request;
    let mut pauses = RETRY_PAUSES.iter();
    loop {
        let spare = request.try_clone();
        match request.send() {
            Ok(response) => return Ok(response),
            Err(error) => match (connection_dropped(&error), pauses.next(), spare) {
                (true, Some(pause), Some(spare)) => {
                    std::thread::sleep(*pause);
                    request = spare;
                }
                _ => return Err(request_error(error)),
            },
        }
    }
}

/// Whether a request failed while connecting (TCP or TLS) because the peer
/// reset the connection or cut it off early, which the next attempt usually
/// does not repeat.
fn connection_dropped(error: &reqwest::Error) -> bool {
    if !error.is_connect() || error.is_timeout() {
        return false;
    }
    let mut cause: Option<&(dyn std::error::Error + 'static)> = Some(error);
    while let Some(current) = cause {
        if current.downcast_ref::<std::io::Error>().is_some_and(|io| {
            use std::io::ErrorKind::{
                BrokenPipe, ConnectionAborted, ConnectionReset, UnexpectedEof,
            };
            matches!(
                io.kind(),
                BrokenPipe | ConnectionAborted | ConnectionReset | UnexpectedEof
            )
        }) {
            return true;
        }
        let text = current.to_string().to_ascii_lowercase();
        if text.contains("connection reset")
            || text.contains("connection aborted")
            || text.contains("broken pipe")
            || text
                .split(|c: char| !c.is_ascii_alphanumeric())
                .any(|word| word == "eof")
        {
            return true;
        }
        cause = current.source();
    }
    false
}

/// A URL for an error message: no userinfo or fragment, and no query value
/// that is a known or probable secret.
fn shown_url(url: &Url) -> String {
    const SECRET_NAMES: [&str; 10] = [
        "auth", "sig", "sid", "code", "pass", "pwd", "jwt", "otp", "sess", "ssid",
    ];
    const SECRET_PARTS: [&str; 7] = [
        "session", "bearer", "cookie", "ticket", "nonce", "access", "oauth",
    ];
    let mut url = url.clone();
    url.set_fragment(None);
    // The project's own redaction covers token, key, secret, password,
    // signature and credential names; the rest of the query is screened here.
    let Ok(mut url) = Url::parse(&output::redact_url(url.as_str())) else {
        return String::new();
    };
    let pairs: Vec<(String, String)> = url
        .query_pairs()
        .map(|(key, value)| {
            let name = key.to_ascii_lowercase();
            let opaque = value.len() >= 24
                && value.bytes().any(|byte| byte.is_ascii_digit())
                && value.bytes().any(|byte| byte.is_ascii_alphabetic())
                && value
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || b"-_+/=.%".contains(&byte));
            let secret = opaque
                || SECRET_NAMES.contains(&name.as_str())
                || SECRET_PARTS.iter().any(|part| name.contains(part));
            (
                key.into_owned(),
                if secret {
                    "REDACTED".into()
                } else {
                    value.into_owned()
                },
            )
        })
        .collect();
    if !pairs.is_empty() {
        url.query_pairs_mut().clear().extend_pairs(pairs);
    }
    let shown = url.to_string();
    if shown.chars().count() > 200 {
        let cut: String = shown.chars().take(199).collect();
        format!("{cut}…")
    } else {
        shown
    }
}

/// What a failed response showed besides its status.
#[derive(Default)]
struct Evidence {
    /// The answer says it is a Cloudflare bot check (`cf-mitigated: challenge`).
    cloudflare_challenge: bool,
    /// The words of a JSON refusal body.
    said: Option<String>,
}

/// Bytes of a failed answer read to find what it says.
const EVIDENCE_BYTES: u64 = 8 * 1024;

/// `HTTP <status>`, then what was asked and a short hint. The status stays
/// first: the web interface recognizes the message by it. A site that is
/// known to turn automated clients away is named, with what works instead
/// (see [`sites`]).
fn http_failure(
    status: reqwest::StatusCode,
    service: Option<&str>,
    url: &Url,
    evidence: &Evidence,
) -> Error {
    let mut message = format!("HTTP {}", status.as_u16());
    match service {
        Some(service) => message.push_str(&format!(" from the {service} service")),
        None => {
            message.push_str(" for ");
            message.push_str(&shown_url(url));
            let refused = matches!(status.as_u16(), 401 | 403 | 418 | 429);
            let site = refused
                .then(|| sites::refusal_hint(url, evidence.cloudflare_challenge))
                .flatten();
            let hint = match status.as_u16() {
                _ if site.is_some() => site,
                404 | 410 => Some("the page may have been removed or is not public".into()),
                401 | 403 => Some(
                    "the site refused access; it may block automated clients or need a login"
                        .into(),
                ),
                429 => Some("rate limited; try again later".into()),
                500..=599 => Some("the site had a server error".into()),
                _ => None,
            };
            let said = evidence
                .said
                .as_ref()
                .map(|said| format!("the site said: {said}"));
            let parts: Vec<String> = hint.into_iter().chain(said).collect();
            if !parts.is_empty() {
                message.push_str(": ");
                message.push_str(&parts.join("; "));
            }
        }
    }
    Error::Fetch(message)
}

/// The evidence a refused answer gives: its Cloudflare header and, for a
/// client error other than "not found", the words of a JSON body.
fn failure_evidence(response: Response) -> Evidence {
    let cloudflare_challenge = response
        .headers()
        .get("cf-mitigated")
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.trim().eq_ignore_ascii_case("challenge"));
    let status = response.status().as_u16();
    let said = if matches!(status, 400..=499) && !matches!(status, 404 | 410) {
        let mut bytes = Vec::new();
        response
            .take(EVIDENCE_BYTES)
            .read_to_end(&mut bytes)
            .ok()
            .and_then(|_| sites::json_refusal(&bytes))
    } else {
        None
    };
    Evidence {
        cloudflare_challenge,
        said,
    }
}

/// A browser's `Browser navigation returned HTTP <status>` as the failure a
/// static request reports, so a refused page reads alike on either route and
/// names the site (the browser does not give the answer's words).
fn browser_failure(url: &str, error: Error) -> Error {
    if let Error::Fetch(message) = &error
        && let Some(status) = sites::browser_status(message)
        && let Ok(status) = reqwest::StatusCode::from_u16(status)
        && let Ok(url) = Url::parse(url)
    {
        return http_failure(status, None, &url, &Evidence::default());
    }
    error
}

pub(crate) fn body(response: Response) -> Result<Vec<u8>> {
    read_body(response, None)
}

/// The body of a response, or the failure its status reports. `service` names
/// a remote extraction service whose own status is not about the page.
fn read_body(response: Response, service: Option<&str>) -> Result<Vec<u8>> {
    if !response.status().is_success() {
        let status = response.status();
        let url = response.url().clone();
        let evidence = if service.is_none() {
            failure_evidence(response)
        } else {
            Evidence::default()
        };
        return Err(http_failure(status, service, &url, &evidence));
    }
    if response.content_length().is_some_and(|n| n > MAX_RESPONSE) {
        return Err(Error::Fetch("Response exceeds 100 MiB".into()));
    }
    let mut bytes = Vec::new();
    response
        .take(MAX_RESPONSE + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| Error::Fetch(format!("Cannot read HTTP response: {e}")))?;
    if bytes.len() as u64 > MAX_RESPONSE {
        return Err(Error::Fetch("Response exceeds 100 MiB".into()));
    }
    Ok(bytes)
}

fn is_private(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v) => {
            v.is_private()
                || v.is_loopback()
                || v.is_link_local()
                || v.is_unspecified()
                || v.is_broadcast()
                || v.is_documentation()
                || v.is_multicast()
                || v.octets()[0] == 0
                || v.octets()[0] >= 240
                || (v.octets()[0] == 100 && (64..=127).contains(&v.octets()[1]))
        }
        IpAddr::V6(v) => {
            v.is_loopback()
                || v.is_unspecified()
                || v.is_unique_local()
                || v.is_unicast_link_local()
                || v.is_multicast()
                || v.to_ipv4_mapped()
                    .is_some_and(|v| is_private(IpAddr::V4(v)))
        }
    }
}

fn remote_allowed(url: &Url, cfg: &Value) -> Result<()> {
    let env = config::environment();
    if env
        .get("MARKITAI_NO_REMOTE_FETCH")
        .is_some_and(|s| ["1", "true", "yes", "on"].contains(&s.to_lowercase().as_str()))
        || cfg.pointer("/fetch/remote_consent").and_then(Value::as_str) != Some("always")
    {
        return Err(Error::Fetch("Remote fetching is disabled by policy".into()));
    }
    let sensitive_query = url.query_pairs().any(|(key, _)| {
        [
            "token",
            "key",
            "secret",
            "password",
            "signature",
            "credential",
        ]
        .iter()
        .any(|part| key.to_lowercase().contains(part))
    });
    if !url.username().is_empty() || url.password().is_some() || sensitive_query {
        return Err(Error::Fetch(
            "Credentialed URLs cannot be sent to remote extraction services".into(),
        ));
    }
    let host = url
        .host_str()
        .ok_or_else(|| Error::Fetch("URL has no hostname".into()))?;
    if !host.contains('.')
        || host.ends_with(".local")
        || host.ends_with(".localhost")
        || host.ends_with(".internal")
    {
        return Err(Error::Fetch(
            "Local URLs cannot be sent to remote extraction services".into(),
        ));
    }
    let addresses: Vec<_> = (host, url.port_or_known_default().unwrap_or(443))
        .to_socket_addrs()
        .map_err(|_| Error::Fetch("Cannot resolve URL hostname for remote policy check".into()))?
        .collect();
    if addresses.is_empty() || addresses.iter().any(|a| is_private(a.ip())) {
        return Err(Error::Fetch(
            "Private URLs cannot be sent to remote extraction services".into(),
        ));
    }
    Ok(())
}

pub(crate) enum FetchContent {
    Document(Document),
    Pdf(DownloadedPdf),
}

pub(crate) struct DownloadedPdf {
    pub bytes: Vec<u8>,
    pub final_url: String,
    pub warnings: Vec<String>,
    pub strategy: &'static str,
}

impl FetchContent {
    fn warnings_mut(&mut self) -> &mut Vec<String> {
        match self {
            Self::Document(document) => &mut document.warnings,
            Self::Pdf(pdf) => &mut pdf.warnings,
        }
    }
}

pub(crate) struct FetchOutcome {
    pub content: FetchContent,
    pub cache_hit: bool,
    pub screenshots: Vec<Asset>,
}

#[cfg(test)]
pub fn fetch(source: &str, cfg: &Value) -> Result<Document> {
    match fetch_with_context(source, cfg, None, true)?.content {
        FetchContent::Document(document) => Ok(document),
        FetchContent::Pdf(_) => Err(Error::Unsupported(
            "The document-only test helper cannot consume deferred PDF bytes".into(),
        )),
    }
}

/// Explicit strategy provenance scopes page-cache keys and suppresses learned
/// routing hints for explicitly selected auto, matching the reference caller.
#[cfg(test)]
fn fetch_with_context(
    source: &str,
    cfg: &Value,
    explicit_strategy: Option<&str>,
    output_available: bool,
) -> Result<FetchOutcome> {
    fetch_with_runtime(source, cfg, explicit_strategy, output_available, None)
}

pub(crate) fn fetch_with_runtime(
    source: &str,
    cfg: &Value,
    explicit_strategy: Option<&str>,
    output_available: bool,
    runtime: Option<&crate::BrowserRuntime>,
) -> Result<FetchOutcome> {
    let url = Url::parse(source).map_err(|_| Error::InvalidInput("Invalid URL".into()))?;
    if !["http", "https"].contains(&url.scheme()) || url.host_str().is_none() {
        return Err(Error::InvalidInput(
            "An HTTP(S) URL with a hostname is required".into(),
        ));
    }
    // An X post named through a mirror host answers with a redirect page; fetch the post itself.
    let canonical = formats::canonical_status_url(&url);
    let (url, source) = match &canonical {
        Some(canonical) => (canonical.clone(), canonical.as_str()),
        None => (url, source),
    };
    let strategy = cfg
        .pointer("/fetch/strategy")
        .and_then(Value::as_str)
        .unwrap_or("auto");
    let capture = config::enabled(cfg, "/screenshot/enabled")
        || config::enabled(cfg, "/screenshot/screenshot_only");
    let anonymous = !browser::identity_configured(cfg)
        && url.username().is_empty()
        && url.password().is_none()
        && cfg
            .pointer("/fetch/playwright/session_mode")
            .and_then(Value::as_str)
            .unwrap_or("isolated")
            == "isolated"
        && !url.query_pairs().any(|(key, _)| {
            [
                "token",
                "secret",
                "password",
                "signature",
                "credential",
                "api_key",
            ]
            .iter()
            .any(|part| key.to_ascii_lowercase().contains(part))
        });
    let learn = strategy == "auto" && anonymous;
    let mut learning_unavailable = false;
    if learn && explicit_strategy.is_none() && browser::available() {
        match spa_domains::take_hint(cfg, &url) {
            Ok(true) => return fetch_browser(source, cfg, capture, output_available, runtime),
            Ok(false) => {}
            Err(_) => learning_unavailable = true,
        }
    }
    let mut result = match strategy {
        "playwright" => fetch_browser(source, cfg, capture, output_available, runtime),
        // A configured browser identity must not read an anonymous cache entry
        // or send an unauthenticated PDF probe before challenge authentication.
        "auto" if browser::identity_configured(cfg) => {
            fetch_browser(source, cfg, capture, output_available, runtime)
        }
        "auto" if capture => match probe_pdf(source, &url, cfg, explicit_strategy, learn)? {
            (Some(outcome), _) => Ok(outcome),
            (None, needs_javascript) => {
                require_capture_output(cfg, output_available)?;
                fetch_browser_and_learn(
                    source,
                    &url,
                    cfg,
                    true,
                    output_available,
                    learn && needs_javascript,
                    runtime,
                )
            }
        },
        "static" if capture => {
            // Explicit static keeps its chosen text representation, even when
            // Chromium is additionally needed for a rendered screenshot.
            let mut outcome = match fetch_static(source, &url, cfg, explicit_strategy) {
                Err(error) if visual_only(cfg) && browser_quality_failure(&error) => {
                    require_capture_output(cfg, output_available)?;
                    return fetch_browser(source, cfg, true, output_available, runtime);
                }
                result => result.map_err(static_javascript_error)?,
            };
            if matches!(&outcome.content, FetchContent::Document(_)) {
                require_capture_output(cfg, output_available)?;
                attach_screenshot(source, cfg, &mut outcome, runtime)?;
            }
            Ok(outcome)
        }
        "auto" => auto_result(
            fetch_static(source, &url, cfg, explicit_strategy),
            browser::available,
            |needs_javascript| {
                fetch_browser_and_learn(
                    source,
                    &url,
                    cfg,
                    false,
                    output_available,
                    learn && needs_javascript,
                    runtime,
                )
            },
        ),
        "static" => {
            fetch_static(source, &url, cfg, explicit_strategy).map_err(static_javascript_error)
        }
        "defuddle" | "jina" => {
            require_capture_output(cfg, output_available)?;
            remote_allowed(&url, cfg)?;
            let client = client(30)?;
            let remote = if strategy == "defuddle" {
                format!(
                    "https://defuddle.md/{}",
                    url::form_urlencoded::byte_serialize(source.as_bytes()).collect::<String>()
                )
            } else {
                format!("https://r.jina.ai/{source}")
            };
            let mut request = client.get(remote);
            if strategy == "jina" {
                request = request.header("Accept", "application/json");
                if let Some(key) = config::environment().get("JINA_API_KEY") {
                    request = request.bearer_auth(key);
                }
            }
            let bytes = read_body(send(request)?, Some(strategy))?;
            let mut doc = if strategy == "jina" {
                let value: Value = serde_json::from_slice(&bytes)?;
                let markdown = value
                    .pointer("/data/content")
                    .and_then(Value::as_str)
                    .ok_or_else(|| Error::Fetch("Jina returned no content".into()))?
                    .to_owned();
                let mut doc = Document {
                    markdown,
                    ..Default::default()
                };
                if let Some(title) = value.pointer("/data/title") {
                    doc.metadata.insert("title".into(), title.clone());
                }
                doc
            } else {
                let text = String::from_utf8(bytes)
                    .map_err(|_| Error::Fetch("Remote Markdown is not UTF-8".into()))?;
                let (metadata, markdown) = output::split_frontmatter(&text);
                Document {
                    markdown: markdown.into(),
                    metadata,
                    ..Default::default()
                }
            };
            if doc.markdown.trim().is_empty() {
                return Err(Error::Fetch("Remote service returned empty content".into()));
            }
            doc.metadata
                .insert("fetch_strategy".into(), json!(strategy));
            let mut outcome = FetchOutcome {
                content: FetchContent::Document(doc),
                cache_hit: false,
                screenshots: Vec::new(),
            };
            if capture {
                attach_screenshot(source, cfg, &mut outcome, runtime)?;
            }
            Ok(outcome)
        }
        _ => Err(Error::Unsupported(format!(
            "Fetch strategy '{strategy}' is not implemented in this development build"
        ))),
    };
    if learning_unavailable && let Ok(outcome) = &mut result {
        add_learning_warning(outcome);
    }
    result
}

const LEARNING_WARNING: &str =
    "Learned browser-domain store is unavailable; routing knowledge could not be used or saved.";
/// The page asks for JavaScript in its visible text.
const JS_REQUIRED: &str = "The page needs JavaScript to show its content";
/// The page has no text of its own and builds its content with scripts.
const JS_EMPTY: &str = "The page has no text without JavaScript";
/// Static extraction kept the little text there was, and the page's markup
/// says it is rendered by scripts.
const JS_SHELL: &str = "The page needs JavaScript to show most of its content, so this text may be incomplete; use -s playwright, or the default auto strategy with Chrome or Chromium installed, to render it.";
const STATIC_NEEDS_JAVASCRIPT: &str =
    "The page needs JavaScript; use -s playwright or the default auto strategy";
const NO_BROWSER_FOR_JAVASCRIPT: &str = "The page needs JavaScript, and no local browser (Chrome or Chromium) was found; install one or set MARKITAI_BROWSER_EXECUTABLE (see 'markitai doctor')";

fn needs_javascript(error: &Error) -> bool {
    matches!(error, Error::Fetch(reason) if reason == JS_REQUIRED || reason == JS_EMPTY)
}

/// An explicit `static` fetch cannot render the page; say what to use.
fn static_javascript_error(error: Error) -> Error {
    if needs_javascript(&error) {
        Error::Fetch(STATIC_NEEDS_JAVASCRIPT.into())
    } else {
        error
    }
}

/// What `auto` makes of its static result. A page that needs JavaScript goes
/// to the local browser, learning the authority when the page said so in its
/// own text; one that merely looks like a script-rendered shell keeps its
/// static text, with a warning, whenever the browser is missing or fails.
fn auto_result(
    result: Result<FetchOutcome>,
    browser_ready: impl FnOnce() -> bool,
    render: impl FnOnce(bool) -> Result<FetchOutcome>,
) -> Result<FetchOutcome> {
    match result {
        Err(error) if browser_quality_failure(&error) => {
            if browser_ready() {
                let rendered =
                    render(matches!(&error, Error::Fetch(reason) if reason == JS_REQUIRED));
                // A site's verification page is what the reader needs to hear
                // about, not only that the browser then failed in its own way.
                match (&error, rendered) {
                    (Error::Fetch(before), Err(Error::Fetch(after)))
                        if before.contains(sites::VERIFICATION_PAGE)
                            && !after.contains(sites::VERIFICATION_PAGE)
                            && !after.starts_with("HTTP ") =>
                    {
                        Err(Error::Fetch(format!(
                            "{before}; the local browser failed as well ({after})"
                        )))
                    }
                    (_, rendered) => rendered,
                }
            } else if needs_javascript(&error) {
                Err(Error::Fetch(NO_BROWSER_FOR_JAVASCRIPT.into()))
            } else {
                Err(error)
            }
        }
        Ok(mut outcome)
            if matches!(&outcome.content, FetchContent::Document(document)
                if document.warnings.iter().any(|warning| warning == JS_SHELL)) =>
        {
            if !browser_ready() {
                return Ok(outcome);
            }
            match render(false) {
                Ok(rendered) => Ok(rendered),
                Err(Error::Fetch(reason)) => {
                    outcome.content.warnings_mut().push(format!(
                        "Browser rendering failed ({reason}); the static text was kept."
                    ));
                    Ok(outcome)
                }
                Err(error) => Err(error),
            }
        }
        result => result,
    }
}

fn add_learning_warning(outcome: &mut FetchOutcome) {
    let warnings = outcome.content.warnings_mut();
    if !warnings.iter().any(|warning| warning == LEARNING_WARNING) {
        warnings.push(LEARNING_WARNING.into());
    }
}

fn fetch_browser_and_learn(
    source: &str,
    url: &Url,
    cfg: &Value,
    capture: bool,
    output_available: bool,
    learn: bool,
    runtime: Option<&crate::BrowserRuntime>,
) -> Result<FetchOutcome> {
    let response = browser::fetch_with_runtime(source, cfg, capture, runtime)
        .map_err(|error| browser_failure(source, error))?;
    // A challenge or still-unrendered shell is not evidence that this domain
    // has a usable browser representation. PDFs never teach HTML routing.
    let admissible = learn
        && matches!(&response, browser::BrowserResponse::Page(page) if html_rejection(&page.html).is_none());
    let mut outcome = browser_response_outcome(response, cfg, output_available)?;
    if admissible
        && matches!(&outcome.content, FetchContent::Document(document) if !document.markdown.trim().is_empty())
        && spa_domains::record_success(cfg, url).is_err()
    {
        add_learning_warning(&mut outcome);
    }
    Ok(outcome)
}

fn require_capture_output(cfg: &Value, output_available: bool) -> Result<()> {
    if visual_only(cfg) && !config::enabled(cfg, "/llm/enabled") && !output_available {
        return Err(Error::InvalidInput(
            "Screenshot-only conversion without LLM requires output_dir to retain captured images"
                .into(),
        ));
    }
    Ok(())
}

fn visual_only(cfg: &Value) -> bool {
    config::enabled(cfg, "/screenshot/screenshot_only")
        && !(config::enabled(cfg, "/llm/enabled") && config::enabled(cfg, "/llm/pure"))
}

fn browser_quality_failure(error: &Error) -> bool {
    match error {
        Error::Fetch(message) => {
            message == JS_REQUIRED
                || message == JS_EMPTY
                || message.contains("HTML challenge page cannot be extracted")
                || message.contains(sites::VERIFICATION_PAGE)
                || message == "URL returned no extractable content"
        }
        Error::Conversion(message) => message == "HTML contains no extractable content",
        _ => false,
    }
}

fn attach_screenshot(
    source: &str,
    cfg: &Value,
    outcome: &mut FetchOutcome,
    runtime: Option<&crate::BrowserRuntime>,
) -> Result<()> {
    match browser::fetch_with_runtime(source, cfg, true, runtime) {
        Ok(browser::BrowserResponse::Page(page)) => {
            outcome.screenshots = page.screenshots;
            outcome.content.warnings_mut().extend(page.warnings);
        }
        Ok(browser::BrowserResponse::Pdf(_)) if visual_only(cfg) => {
            return Err(Error::Fetch("Browser received a PDF instead of the selected webpage; use the playwright strategy to convert that PDF".into()));
        }
        Ok(browser::BrowserResponse::Pdf(_)) => outcome.content.warnings_mut().push(
            "Browser received a PDF instead of a webpage screenshot; retained the selected text representation.".into(),
        ),
        Err(error) if visual_only(cfg) => return Err(error),
        Err(_) => outcome
            .content
            .warnings_mut()
            .push("Browser screenshot failed; retained the selected text representation.".into()),
    }
    Ok(())
}

fn fetch_browser(
    source: &str,
    cfg: &Value,
    capture: bool,
    output_available: bool,
    runtime: Option<&crate::BrowserRuntime>,
) -> Result<FetchOutcome> {
    browser_response_outcome(
        browser::fetch_with_runtime(source, cfg, capture, runtime)
            .map_err(|error| browser_failure(source, error))?,
        cfg,
        output_available,
    )
}

fn browser_response_outcome(
    response: browser::BrowserResponse,
    cfg: &Value,
    output_available: bool,
) -> Result<FetchOutcome> {
    match response {
        browser::BrowserResponse::Page(page) => {
            require_capture_output(cfg, output_available)?;
            browser_outcome(page, cfg)
        }
        browser::BrowserResponse::Pdf(pdf) => Ok(FetchOutcome {
            content: FetchContent::Pdf(DownloadedPdf {
                bytes: pdf.bytes,
                final_url: pdf.final_url,
                warnings: pdf.warnings,
                strategy: "playwright",
            }),
            cache_hit: false,
            screenshots: Vec::new(),
        }),
    }
}

fn browser_outcome(page: browser::BrowserPage, cfg: &Value) -> Result<FetchOutcome> {
    if let Ok(url) = Url::parse(&page.final_url)
        && let Some(message) = sites::verification_page(&url, &page.html)
    {
        return Err(Error::Fetch(message));
    }
    let visual_only = visual_only(cfg);
    let mut document = match formats::extract_html(&page.html, Some(&page.final_url)) {
        Err(Error::Conversion(message))
            if message == "HTML contains no extractable content"
                && visual_only
                && !page.screenshots.is_empty() =>
        {
            Document::default()
        }
        result => result?,
    };
    if document.markdown.trim().is_empty() && !(visual_only && !page.screenshots.is_empty()) {
        return Err(Error::Fetch(
            "Browser returned no extractable content".into(),
        ));
    }
    if !page.title.is_empty() {
        document
            .metadata
            .entry("title")
            .or_insert_with(|| json!(page.title));
    }
    document
        .metadata
        .insert("fetch_strategy".into(), json!("playwright"));
    document
        .metadata
        .insert("renderer".into(), json!("playwright"));
    document.metadata.insert(
        "source_url".into(),
        json!(output::redact_url(&page.final_url)),
    );
    document.warnings.extend(page.warnings);
    Ok(FetchOutcome {
        content: FetchContent::Document(document),
        cache_hit: false,
        screenshots: page.screenshots,
    })
}

const STATIC_ACCEPT: &str = "text/markdown, text/html;q=0.9, */*;q=0.5";
const CACHE_WARNING: &str =
    "Persistent URL fetch cache is unavailable; caching could not be completed.";

struct StaticPage {
    content: FetchContent,
    cache_eligible: bool,
    final_url: String,
    etag: Option<String>,
    last_modified: Option<String>,
    /// Where a nearly empty page sent the reader with a `<meta>` refresh
    /// within two seconds; `content` is then only a placeholder.
    refresh: Option<Url>,
}

/// Meta refreshes followed after one request.
const MAX_REFRESHES: usize = 5;

/// Follow `<meta http-equiv="refresh">` pages, which a browser would have left
/// at once, the way HTTP redirects are followed: only to http(s) URLs, with no
/// credentials (the request carries none, even to the same origin), and a
/// bounded number of times. The cache validators of the first response do
/// not describe the page that was finally read, so none are kept.
fn follow_meta_refresh(client: &Client, page: &mut StaticPage, defer_pdf: bool) -> Result<()> {
    let mut hops = 0;
    while let Some(target) = page.refresh.take() {
        hops += 1;
        if hops > MAX_REFRESHES {
            return Err(Error::Fetch(format!(
                "Too many <meta> refresh redirects (more than {MAX_REFRESHES})"
            )));
        }
        *page = decode_static(
            send(
                client
                    .get(target)
                    .header(reqwest::header::ACCEPT, STATIC_ACCEPT),
            )?,
            defer_pdf,
        )?;
        page.etag = None;
        page.last_modified = None;
    }
    Ok(())
}

fn cache_warning(content: &mut FetchContent, unavailable: bool) {
    if unavailable {
        content.warnings_mut().push(CACHE_WARNING.into());
    }
}

fn cached_document(entry: fetch_cache::Entry) -> Document {
    let mut metadata = entry.metadata;
    if let Some(title) = entry.title {
        metadata.entry("title").or_insert_with(|| title.into());
    }
    metadata.insert("fetch_strategy".into(), entry.strategy_used.into());
    Document {
        markdown: entry.content,
        metadata,
        ..Default::default()
    }
}

fn fetch_static(
    source: &str,
    url: &Url,
    cfg: &Value,
    explicit_strategy: Option<&str>,
) -> Result<FetchOutcome> {
    let cache = fetch_cache::Cache::from_config(cfg);
    let mut cache_unavailable = false;
    let entry = match cache
        .as_ref()
        .map(|cache| cache.get(source, explicit_strategy))
    {
        Some(Ok(entry)) => entry.filter(|entry| !entry.content.trim().is_empty()),
        Some(Err(_)) => {
            cache_unavailable = true;
            None
        }
        None => None,
    };
    let entry = match entry {
        Some(entry)
            if entry.etag.as_deref().is_none_or(str::is_empty)
                && entry.last_modified.as_deref().is_none_or(str::is_empty) =>
        {
            // The store enforces TTL and updates access time on reads.
            let mut content = FetchContent::Document(cached_document(entry));
            cache_warning(&mut content, cache_unavailable);
            return Ok(FetchOutcome {
                content,
                cache_hit: true,
                screenshots: Vec::new(),
            });
        }
        entry => entry,
    };

    let client = client(30)?;
    let mut fresh = None;
    if let Some(entry) = entry {
        let mut request = client
            .get(url.clone())
            .header(reqwest::header::ACCEPT, STATIC_ACCEPT);
        if let Some(etag) = &entry.etag {
            request = request.header(reqwest::header::IF_NONE_MATCH, etag);
        }
        if let Some(modified) = &entry.last_modified {
            request = request.header(reqwest::header::IF_MODIFIED_SINCE, modified);
        }
        if let Ok(response) = request.send() {
            if response.status() == reqwest::StatusCode::NOT_MODIFIED {
                if let Some(cache) = &cache {
                    cache_unavailable |= cache.touch(source, explicit_strategy).is_err();
                }
                let mut content = FetchContent::Document(cached_document(entry));
                cache_warning(&mut content, cache_unavailable);
                return Ok(FetchOutcome {
                    content,
                    cache_hit: true,
                    screenshots: Vec::new(),
                });
            }
            // A failed status, unreadable body or unusable extracted page takes
            // the same unconditional path as a failed conditional request.
            fresh = decode_static(response, defer_pdf(cfg)).ok();
        }
    }
    let mut page = match fresh {
        Some(page) => page,
        None => decode_static(
            send(
                client
                    .get(url.clone())
                    .header(reqwest::header::ACCEPT, STATIC_ACCEPT),
            )?,
            defer_pdf(cfg),
        )?,
    };
    follow_meta_refresh(&client, &mut page, defer_pdf(cfg))?;
    if let Some(cache) = cache {
        if page.cache_eligible
            && let FetchContent::Document(document) = &page.content
            && document.assets.is_empty()
            && document.warnings.is_empty()
        {
            let entry = fetch_cache::Entry {
                content: document.markdown.clone(),
                metadata: document.metadata.clone(),
                strategy_used: "static".into(),
                title: document
                    .metadata
                    .get("title")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                final_url: Some(page.final_url),
                etag: page.etag,
                last_modified: page.last_modified,
                ..Default::default()
            };
            cache_unavailable |= cache.set(source, explicit_strategy, &entry).is_err();
        } else {
            // A successfully changed representation must not leave an old
            // validator or still-fresh TTL row able to replay different content.
            cache_unavailable |= cache.remove(source, explicit_strategy).is_err();
        }
    }
    cache_warning(&mut page.content, cache_unavailable);
    Ok(FetchOutcome {
        content: page.content,
        cache_hit: false,
        screenshots: Vec::new(),
    })
}

fn header_text(response: &Response, name: reqwest::header::HeaderName) -> Option<String> {
    response
        .headers()
        .get(name)
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.trim().is_empty())
        .map(str::to_owned)
}

/// The text of a response body and an optional warning about its encoding.
///
/// HTML follows a BOM, then the declarations its bytes fit and, failing them,
/// detection (see `formats::decode_fetched_html`). Plain text follows a BOM,
/// then the HTTP charset, then UTF-8, and malformed sequences become
/// replacement characters.
fn decode_text<'a>(
    bytes: &'a [u8],
    content_type: &str,
    html: bool,
) -> (std::borrow::Cow<'a, str>, Option<String>) {
    let charset = content_type.split(';').find_map(|parameter| {
        let (key, value) = parameter.trim().split_once('=')?;
        key.trim()
            .eq_ignore_ascii_case("charset")
            .then_some(value.trim())
    });
    if html {
        return formats::decode_fetched_html(bytes, charset);
    }
    let encoding = charset
        .and_then(|label| {
            encoding_rs::Encoding::for_label(label.trim_matches(['\"', '\'']).as_bytes())
        })
        .unwrap_or(encoding_rs::UTF_8);
    (encoding.decode(bytes).0, None)
}

// Restrict checks to HTML and visible non-code text. Vendor names in an article,
// literal code examples or a plain-text document are not challenge evidence.
fn html_rejection(html: &str) -> Option<&'static str> {
    let tree = scraper::Html::parse_document(html);
    let mut text = String::new();
    let mut title = String::new();
    let mut widget = false;
    let mut authored_article = false;
    for node in tree.tree.nodes() {
        if let Some(element) = scraper::ElementRef::wrap(node) {
            let name = element.value().name();
            if name == "article" {
                authored_article |= element.text().any(|value| !value.trim().is_empty());
            }
            if name == "title" {
                title = element.text().collect::<String>().to_lowercase();
            }
            for attr in ["id", "class", "src", "action"] {
                if let Some(value) = element.value().attr(attr) {
                    let value = value.to_ascii_lowercase();
                    widget |= [
                        "cf-browser-verification",
                        "cf-chl-",
                        "cf_chl_",
                        "/cdn-cgi/challenge-platform/",
                        "google.com/recaptcha",
                        "hcaptcha.com",
                        "g-recaptcha",
                        "h-captcha",
                    ]
                    .iter()
                    .any(|marker| value.contains(marker));
                }
            }
        } else if let scraper::Node::Text(value) = node.value() {
            let ignored = node
                .ancestors()
                .filter_map(scraper::ElementRef::wrap)
                .any(|element| {
                    matches!(
                        element.value().name(),
                        "head" | "script" | "style" | "pre" | "code" | "template"
                    )
                });
            if !ignored {
                text.push_str(value);
                text.push(' ');
            }
        }
    }
    let text = text
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase();
    let short = text.len() <= 2000;
    let challenge_instruction = [
        "checking your browser before accessing",
        "verify you are human",
        "verify that you are human",
        "智能验证检测中",
        "由极验提供技术支持",
    ]
    .iter()
    .any(|phrase| text.starts_with(phrase));
    let challenge_title = [
        "just a moment",
        "attention required",
        "security verification",
        "verify you are human",
    ]
    .iter()
    .any(|phrase| title.trim_start().starts_with(phrase));
    if short
        && ((!authored_article && challenge_instruction)
            || (widget && (challenge_title || text.is_empty())))
    {
        return Some(
            "HTML challenge page cannot be extracted; browser challenge fallback is not implemented",
        );
    }
    if short
        && !authored_article
        && [
            "please enable javascript",
            "javascript is disabled",
            "javascript is not available",
            "you need to enable javascript",
            "enable javascript to continue",
        ]
        .iter()
        .any(|phrase| text.starts_with(phrase))
    {
        return Some(JS_REQUIRED);
    }
    None
}

fn defer_pdf(cfg: &Value) -> bool {
    config::enabled(cfg, "/ocr/enabled")
        || config::enabled(cfg, "/screenshot/enabled")
        || config::enabled(cfg, "/screenshot/screenshot_only")
}

struct StaticResponse {
    effective_url: Url,
    content_type: String,
    mime: String,
    etag: Option<String>,
    last_modified: Option<String>,
    bytes: Vec<u8>,
}

impl StaticResponse {
    fn read(response: Response) -> Result<Self> {
        let effective_url = response.url().clone();
        let content_type =
            header_text(&response, reqwest::header::CONTENT_TYPE).unwrap_or_default();
        let mime = content_type
            .split(';')
            .next()
            .unwrap_or("")
            .trim()
            .to_ascii_lowercase();
        let etag = header_text(&response, reqwest::header::ETAG);
        let last_modified = header_text(&response, reqwest::header::LAST_MODIFIED);
        let bytes = body(response)?;
        Ok(Self {
            effective_url,
            content_type,
            mime,
            etag,
            last_modified,
            bytes,
        })
    }

    fn kind(&self) -> StaticKind {
        let prefix = self
            .bytes
            .iter()
            .copied()
            .skip_while(u8::is_ascii_whitespace)
            .take(32)
            .map(|byte| byte.to_ascii_lowercase())
            .collect::<Vec<_>>();
        // Explicit text keeps literal HTML/PDF examples; actual HTML remains
        // authoritative even when a download URL or PDF MIME says otherwise.
        let explicit_text = self.mime.starts_with("text/") && !self.mime.contains("html");
        if self.mime.contains("html")
            || (!explicit_text
                && (prefix.starts_with(b"<!doctype html") || prefix.starts_with(b"<html")))
        {
            StaticKind::Html
        } else if self.mime.starts_with("text/") && !self.mime.contains("xml") {
            StaticKind::Text
        } else {
            StaticKind::Other
        }
    }

    fn is_pdf(&self) -> bool {
        if self.kind() != StaticKind::Other || self.mime.starts_with("text/") {
            return false;
        }
        // An authoritative MIME is a representation change even when its bytes
        // are malformed. Its later reader failure must not replay cached HTML
        // or cause another download of the same accepted response.
        if matches!(self.mime.as_str(), "application/pdf" | "application/x-pdf") {
            return true;
        }
        let generic = matches!(
            self.mime.as_str(),
            "" | "application/octet-stream"
                | "binary/octet-stream"
                | "application/binary"
                | "application/download"
        );
        let path_hint = std::path::Path::new(self.effective_url.path())
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("pdf"));
        (generic || path_hint)
            && self.bytes[..self.bytes.len().min(1024)]
                .windows(8)
                .any(|header| {
                    &header[..5] == b"%PDF-"
                        && matches!(header[5], b'1' | b'2')
                        && header[6] == b'.'
                        && header[7].is_ascii_digit()
                })
    }

    fn into_pdf(self) -> DownloadedPdf {
        DownloadedPdf {
            strategy: "static",
            bytes: self.bytes,
            final_url: self.effective_url.into(),
            warnings: Vec::new(),
        }
    }
}

#[derive(PartialEq, Eq)]
enum StaticKind {
    Html,
    Text,
    Other,
}

/// Auto capture must inspect the response before invoking a browser: extensions
/// and cached HTML cannot identify an opaque URL that now downloads a PDF.
fn probe_pdf(
    source: &str,
    url: &Url,
    cfg: &Value,
    explicit_strategy: Option<&str>,
    inspect_learning: bool,
) -> Result<(Option<FetchOutcome>, bool)> {
    let response = StaticResponse::read(send(
        client(30)?
            .get(url.clone())
            .header(reqwest::header::ACCEPT, STATIC_ACCEPT),
    )?)?;
    if !response.is_pdf() {
        let needs_javascript = inspect_learning
            && response.kind() == StaticKind::Html
            && html_rejection(&decode_text(&response.bytes, &response.content_type, true).0)
                == Some(JS_REQUIRED);
        return Ok((None, needs_javascript));
    }
    let mut content = FetchContent::Pdf(response.into_pdf());
    if let Some(cache) = fetch_cache::Cache::from_config(cfg) {
        cache_warning(
            &mut content,
            cache.remove(source, explicit_strategy).is_err(),
        );
    }
    Ok((
        Some(FetchOutcome {
            content,
            cache_hit: false,
            screenshots: Vec::new(),
        }),
        false,
    ))
}

fn decode_static(response: Response, defer_pdf: bool) -> Result<StaticPage> {
    let response = StaticResponse::read(response)?;
    if defer_pdf && response.is_pdf() {
        let final_url = response.effective_url.to_string();
        return Ok(StaticPage {
            content: FetchContent::Pdf(response.into_pdf()),
            cache_eligible: false,
            final_url,
            etag: None,
            last_modified: None,
            refresh: None,
        });
    }
    let kind = response.kind();
    let cache_eligible = kind != StaticKind::Other;
    let mut doc = match kind {
        StaticKind::Html => {
            let (html, decode_warning) = decode_text(&response.bytes, &response.content_type, true);
            if let Some(message) = sites::verification_page(&response.effective_url, &html) {
                return Err(Error::Fetch(message));
            }
            if let Some(reason) = html_rejection(&html) {
                return Err(Error::Fetch(reason.into()));
            }
            let extracted = formats::extract_html(&html, Some(response.effective_url.as_str()));
            let words = extracted
                .as_ref()
                .ok()
                .map(|document| text_words(&document.markdown));
            let empty =
                matches!(&extracted, Err(Error::Conversion(message)) if message == NO_CONTENT);
            let short = words.is_some_and(|words| words < SHORT_PAGE_WORDS);
            if (empty || short)
                && let Some(target) = meta_refresh(&html, &response.effective_url)
            {
                return Ok(StaticPage {
                    content: FetchContent::Document(Document::default()),
                    cache_eligible: false,
                    final_url: response.effective_url.into(),
                    etag: None,
                    last_modified: None,
                    refresh: Some(target),
                });
            }
            let mut document = match extracted {
                Err(_) if empty && script_rendered(&html) => {
                    return Err(Error::Fetch(JS_EMPTY.into()));
                }
                extracted => extracted?,
            };
            if short && script_rendered(&html) {
                document.warnings.push(JS_SHELL.into());
            }
            document.warnings.extend(decode_warning);
            document
        }
        StaticKind::Text => Document {
            markdown: decode_text(&response.bytes, &response.content_type, false)
                .0
                .into_owned(),
            ..Default::default()
        },
        StaticKind::Other => {
            let extension = if response.mime.contains("pdf") {
                "pdf"
            } else {
                std::path::Path::new(response.effective_url.path())
                    .extension()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
            };
            if !formats::supports_extension(extension) {
                return Err(Error::Unsupported(format!(
                    "Unsupported URL content type: {}",
                    response.mime
                )));
            }
            let mut file = tempfile::Builder::new()
                .prefix("markitai-fetch-")
                .suffix(&format!(".{extension}"))
                .tempfile()?;
            file.write_all(&response.bytes)?;
            formats::extract(file.path())?
        }
    };
    if doc.markdown.trim().is_empty() {
        return Err(Error::Fetch("URL returned no extractable content".into()));
    }
    doc.metadata
        .insert("fetch_strategy".into(), json!("static"));
    Ok(StaticPage {
        content: FetchContent::Document(doc),
        cache_eligible,
        final_url: response.effective_url.into(),
        etag: response.etag,
        last_modified: response.last_modified,
        refresh: None,
    })
}

const NO_CONTENT: &str = "HTML contains no extractable content";
/// Fewer words than this, in the Markdown a page yields, make it a candidate
/// for a stub that only redirects or a shell that scripts fill in.
const SHORT_PAGE_WORDS: usize = 30;
/// Inline script text, in bytes, that outweighs a page this short: the data
/// or code that builds the page rather than an analytics snippet.
const SHELL_SCRIPT_BYTES: usize = 3000;
/// The longest delay, in seconds, of a `<meta>` refresh that is followed.
const MAX_REFRESH_SECONDS: f64 = 2.0;

/// Words of extracted Markdown; the destinations of links are not words.
fn text_words(markdown: &str) -> usize {
    use std::sync::LazyLock;
    static DESTINATION: LazyLock<regex::Regex> =
        LazyLock::new(|| regex::Regex::new(r"\]\([^)]*\)").unwrap());
    formats::word_count(&DESTINATION.replace_all(markdown, "]"))
}

/// Whether the markup says the page is built by scripts: a large inline
/// script, or beside any script an empty mount point for a front-end
/// framework or a `<noscript>` text about JavaScript.
fn script_rendered(html: &str) -> bool {
    const MOUNTS: [&str; 9] = [
        "root",
        "app",
        "__next",
        "__nuxt",
        "___gatsby",
        "svelte",
        "react-root",
        "q-app",
        "app-root",
    ];
    let tree = scraper::Html::parse_document(html);
    let (mut scripts, mut inline, mut mount, mut noscript) = (0, 0, false, false);
    for node in tree.tree.nodes() {
        let Some(element) = scraper::ElementRef::wrap(node) else {
            continue;
        };
        let name = element.value().name();
        match name {
            "script" => {
                scripts += 1;
                let kind = element
                    .value()
                    .attr("type")
                    .unwrap_or("")
                    .to_ascii_lowercase();
                // Structured data and templates describe visible content.
                if element.value().attr("src").is_none()
                    && !kind.contains("ld+json")
                    && !kind.contains("template")
                {
                    inline += element.text().map(str::len).sum::<usize>();
                }
            }
            "noscript" => {
                noscript |= element
                    .text()
                    .any(|text| text.to_ascii_lowercase().contains("javascript"));
            }
            _ => {}
        }
        let mount_point = name == "app-root"
            || element
                .value()
                .id()
                .is_some_and(|id| MOUNTS.contains(&id.to_ascii_lowercase().as_str()));
        if mount_point && !has_page_text(element) {
            mount = true;
        }
    }
    scripts > 0 && (inline >= SHELL_SCRIPT_BYTES || mount || noscript)
}

/// Text of an element that a reader would see, not script or style source.
fn has_page_text(element: scraper::ElementRef<'_>) -> bool {
    element.descendants().any(|node| {
        matches!(node.value(), scraper::Node::Text(text) if !text.trim().is_empty())
            && !node
                .ancestors()
                .filter_map(scraper::ElementRef::wrap)
                .any(|ancestor| {
                    matches!(
                        ancestor.value().name(),
                        "script" | "style" | "noscript" | "template"
                    )
                })
    })
}

/// The target of a `<meta http-equiv="refresh">` that a browser would follow
/// within [`MAX_REFRESH_SECONDS`], when it is another http(s) URL.
fn meta_refresh(html: &str, base: &Url) -> Option<Url> {
    use std::sync::LazyLock;
    static META: LazyLock<scraper::Selector> =
        LazyLock::new(|| scraper::Selector::parse("meta[http-equiv][content]").unwrap());
    let tree = scraper::Html::parse_document(html);
    tree.select(&META).find_map(|meta| {
        let element = meta.value();
        element
            .attr("http-equiv")?
            .trim()
            .eq_ignore_ascii_case("refresh")
            .then(|| refresh_target(element.attr("content")?, base))
            .flatten()
    })
}

/// `seconds`, `seconds; url=target` or `seconds, target`, as the HTML standard
/// reads a refresh's `content`: a delay of at most [`MAX_REFRESH_SECONDS`],
/// then a target other than the page itself.
fn refresh_target(content: &str, base: &Url) -> Option<Url> {
    let content = content.trim_start();
    let end = content
        .find(|c: char| !(c.is_ascii_digit() || c == '.'))
        .unwrap_or(content.len());
    let seconds: f64 = content[..end].parse().ok()?;
    if seconds > MAX_REFRESH_SECONDS {
        return None;
    }
    let rest =
        content[end..].trim_start_matches(|c: char| c == ';' || c == ',' || c.is_whitespace());
    let rest = if rest.len() >= 3
        && rest.is_char_boundary(3)
        && rest[..3].eq_ignore_ascii_case("url")
        && let Some(after) = rest[3..].trim_start().strip_prefix('=')
    {
        after.trim_start()
    } else {
        rest
    };
    let rest = match rest.chars().next() {
        Some(quote @ ('"' | '\'')) => rest[1..].split(quote).next().unwrap_or_default(),
        _ => rest,
    }
    .trim();
    if rest.is_empty() {
        return None;
    }
    let mut target = base.join(rest).ok()?;
    // Credentials in a page's own markup are not ours to send either.
    let _ = target.set_username("");
    let _ = target.set_password(None);
    (matches!(target.scheme(), "http" | "https") && target.host_str().is_some() && target != *base)
        .then_some(target)
}

#[cfg(test)]
impl FetchOutcome {
    fn document(&self) -> &Document {
        match &self.content {
            FetchContent::Document(document) => document,
            FetchContent::Pdf(_) => panic!("expected extracted text, received deferred PDF"),
        }
    }
    fn document_mut(&mut self) -> &mut Document {
        match &mut self.content {
            FetchContent::Document(document) => document,
            FetchContent::Pdf(_) => panic!("expected extracted text, received deferred PDF"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn a_failed_request_says_why_without_its_url() {
        // A port nothing listens on: the summary alone would only say that
        // sending the request failed.
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let url = format!("http://user:secret@127.0.0.1:{port}/private/path");
        let error = client(5)
            .unwrap()
            .get(&url)
            .send()
            .map_err(request_error)
            .unwrap_err();
        let Error::Fetch(message) = error else {
            panic!("{error:?}")
        };
        assert!(message.to_lowercase().contains("refused"), "{message}");
        assert!(
            !message.contains("secret") && !message.contains("/private/path"),
            "{message}"
        );
    }
    #[test]
    fn private_and_credentialed_remote_targets_are_rejected() {
        let cfg = config::defaults();
        for target in [
            "http://localhost/x",
            "http://127.0.0.1/x",
            "http://[::1]/x",
            "https://user:pass@example.com/x",
            "https://example.com/?api_key=secret",
        ] {
            assert!(remote_allowed(&Url::parse(target).unwrap(), &cfg).is_err());
        }
    }
    #[test]
    fn browser_final_url_metadata_is_redacted_without_changing_relative_link_base() {
        let cfg = config::defaults();
        let page = browser::BrowserPage {
            html: "<article><p>Read <a href='next'>next page</a>.</p></article>".into(),
            final_url: "https://example.test/section/current?token=private-value&view=full".into(),
            title: "Result".into(),
            screenshots: Vec::new(),
            warnings: Vec::new(),
        };
        let mut result = browser_outcome(page, &cfg).unwrap();
        assert!(
            result
                .document()
                .markdown
                .contains("https://example.test/section/next")
        );
        let prepared = output::prepare(
            "https://example.test/start",
            "start",
            result.document_mut(),
            &cfg,
        );
        let metadata = serde_json::to_string(&prepared.frontmatter).unwrap();
        assert!(!metadata.contains("private-value"));
        assert_eq!(
            prepared.frontmatter["source_url"],
            "https://example.test/section/current?token=REDACTED&view=full"
        );
    }

    #[test]
    fn static_html_runs_against_local_http() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0; 4096];
            let _ = stream.read(&mut request);
            let html = "<html><title>Local title</title><article><h1>Local title</h1><p>Hello native network.</p></article></html>";
            write!(stream,"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{html}",html.len()).unwrap();
        });
        let mut cfg = config::defaults();
        cfg["cache"]["enabled"] = json!(false);
        let doc = fetch(&format!("http://{addr}/page"), &cfg).unwrap();
        server.join().unwrap();
        assert!(doc.markdown.contains("Hello native network."));
        assert_eq!(doc.metadata["fetch_strategy"], "static");
    }
}

#[cfg(test)]
mod cache_tests {
    use super::*;
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    };

    #[derive(Clone)]
    pub(super) struct Reply {
        status: u16,
        headers: Vec<(String, String)>,
        body: Vec<u8>,
    }

    impl Reply {
        /// Read the request, then close the connection without answering.
        pub(super) fn dropped() -> Self {
            Self {
                status: 0,
                headers: Vec::new(),
                body: Vec::new(),
            }
        }
        /// Read the request, then reset the connection (an RST, where the
        /// platform allows it) without answering. Only the Unix tests use it.
        #[cfg(unix)]
        pub(super) fn reset() -> Self {
            Self {
                status: 1,
                ..Self::dropped()
            }
        }
        pub(super) fn bytes(mime: &str, bytes: &[u8]) -> Self {
            Self {
                status: 200,
                headers: vec![("Content-Type".into(), mime.into())],
                body: bytes.to_vec(),
            }
        }
        pub(super) fn text(body: &str) -> Self {
            Self {
                status: 200,
                headers: vec![("Content-Type".into(), "text/plain".into())],
                body: body.as_bytes().to_vec(),
            }
        }
        pub(super) fn html(body: &str) -> Self {
            Self::text(body).header("Content-Type", "text/html")
        }
        pub(super) fn header(mut self, name: &str, value: &str) -> Self {
            self.headers
                .retain(|(key, _)| !key.eq_ignore_ascii_case(name));
            self.headers.push((name.into(), value.into()));
            self
        }
        pub(super) fn status(mut self, status: u16) -> Self {
            self.status = status;
            self
        }
    }

    /// Make closing the stream send a reset instead of an orderly shutdown.
    pub(super) fn reset_on_close(stream: &std::net::TcpStream) {
        #[cfg(unix)]
        {
            use std::os::fd::AsRawFd;
            let linger = libc::linger {
                l_onoff: 1,
                l_linger: 0,
            };
            // SAFETY: a valid socket descriptor, and a `linger` of the size given.
            let result = unsafe {
                libc::setsockopt(
                    stream.as_raw_fd(),
                    libc::SOL_SOCKET,
                    libc::SO_LINGER,
                    (&raw const linger).cast(),
                    std::mem::size_of::<libc::linger>() as libc::socklen_t,
                )
            };
            assert_eq!(result, 0);
        }
        #[cfg(not(unix))]
        let _ = stream;
    }

    pub(super) struct Server {
        origin: String,
        requests: Arc<Mutex<Vec<String>>>,
        stop: Arc<AtomicBool>,
        thread: Option<std::thread::JoinHandle<()>>,
    }

    impl Server {
        pub(super) fn new(replies: Vec<Reply>) -> Self {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let origin = format!("http://{}", listener.local_addr().unwrap());
            listener.set_nonblocking(true).unwrap();
            let requests = Arc::new(Mutex::new(Vec::new()));
            let stop = Arc::new(AtomicBool::new(false));
            let log = Arc::clone(&requests);
            let stopped = Arc::clone(&stop);
            let thread = std::thread::spawn(move || {
                while !stopped.load(Ordering::Relaxed) {
                    let (mut stream, _) = match listener.accept() {
                        Ok(connection) => connection,
                        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                            std::thread::sleep(Duration::from_millis(2));
                            continue;
                        }
                        Err(error) => panic!("loopback server: {error}"),
                    };
                    stream.set_nonblocking(false).unwrap();
                    stream
                        .set_read_timeout(Some(Duration::from_secs(5)))
                        .unwrap();
                    let mut request_reader = bounded_fixture_io::Reader::new(
                        &stream,
                        std::time::Instant::now() + Duration::from_secs(5),
                    );
                    let mut request = Vec::new();
                    let mut chunk = [0; 1024];
                    while !request.windows(4).any(|bytes| bytes == b"\r\n\r\n") {
                        let size = request_reader.read(&mut chunk).unwrap();
                        if size == 0 {
                            break;
                        }
                        request.extend_from_slice(&chunk[..size]);
                        assert!(request.len() <= 64 * 1024);
                    }
                    let index = {
                        let mut log = log.lock().unwrap();
                        let index = log.len();
                        log.push(String::from_utf8(request).unwrap());
                        index
                    };
                    let reply = replies
                        .get(index)
                        .cloned()
                        .unwrap_or_else(|| Reply::text("unexpected request").status(500));
                    if reply.status < 100 {
                        // No reply is written: the connection is closed cleanly
                        // (0) or reset (1).
                        if reply.status == 1 {
                            reset_on_close(&stream);
                        }
                        continue;
                    }
                    write!(
                        stream,
                        "HTTP/1.1 {} Test\r\nContent-Length: {}\r\nConnection: close\r\n",
                        reply.status,
                        reply.body.len()
                    )
                    .unwrap();
                    for (name, value) in reply.headers {
                        write!(stream, "{name}: {value}\r\n").unwrap();
                    }
                    stream.write_all(b"\r\n").unwrap();
                    stream.write_all(&reply.body).unwrap();
                }
            });
            Self {
                origin,
                requests,
                stop,
                thread: Some(thread),
            }
        }
        pub(super) fn url(&self, path: &str) -> String {
            format!("{}{path}", self.origin)
        }
        pub(super) fn requests(&self) -> Vec<String> {
            self.requests.lock().unwrap().clone()
        }
    }
    impl Drop for Server {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::Relaxed);
            let result = self.thread.take().unwrap().join();
            if !std::thread::panicking() {
                result.expect("loopback server failed");
            }
        }
    }
    pub(super) fn settings() -> (tempfile::TempDir, Value) {
        let directory = tempfile::tempdir().unwrap();
        let mut cfg = config::defaults();
        // These fixtures test HTTP validators and store state, without a browser
        // retry consuming a response intended for the next cache operation.
        cfg["fetch"]["strategy"] = json!("static");
        cfg["cache"]["global_dir"] = json!(directory.path());
        (directory, cfg)
    }
    fn run(server: &Server, cfg: &Value) -> FetchOutcome {
        fetch_with_context(&server.url("/page"), cfg, None, true).unwrap()
    }
    fn header(request: &str, name: &str) -> Option<String> {
        request
            .lines()
            .filter_map(|line| line.split_once(':'))
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.trim().into())
    }

    #[test]
    fn fresh_unvalidated_hits_reopen_the_store_and_ttl_does_not_restart() {
        let (directory, mut cfg) = settings();
        let server = Server::new(vec![
            Reply::text("first"),
            Reply::text("second"),
            Reply::text("third"),
        ]);
        assert!(!run(&server, &cfg).cache_hit);
        assert!(run(&server, &cfg).cache_hit);
        assert_eq!(server.requests().len(), 1);
        let db = rusqlite::Connection::open(directory.path().join("fetch_cache.db")).unwrap();
        db.execute("UPDATE fetch_cache SET created_at=1", [])
            .unwrap();
        let second = run(&server, &cfg);
        assert!(!second.cache_hit);
        assert_eq!(second.document().markdown, "second");
        cfg["cache"]["fetch_ttl_seconds"] = json!(0);
        let third = run(&server, &cfg);
        assert!(!third.cache_hit);
        assert_eq!(third.document().markdown, "third");
        assert_eq!(server.requests().len(), 3);
        for request in server.requests() {
            assert_eq!(header(&request, "Accept").as_deref(), Some(STATIC_ACCEPT));
            assert!(header(&request, "If-None-Match").is_none());
        }
    }

    #[test]
    fn validators_are_sent_each_time_independent_of_ttl_and_304_keeps_creation() {
        let (directory, mut cfg) = settings();
        cfg["cache"]["fetch_ttl_seconds"] = json!(0);
        let modified = "Mon, 28 Sep 2026 09:00:00 GMT";
        let server = Server::new(vec![
            Reply::html("<title>First title</title><p>First body.</p>")
                .header("ETag", "\"v1\"")
                .header("Last-Modified", modified),
            Reply::text("").status(304).header("ETag", "\"ignored\""),
            Reply::html("<title>Second title</title><p>Second body.</p>").header("ETag", "\"v2\""),
            Reply::text("").status(304),
        ]);
        let first = run(&server, &cfg);
        let db = rusqlite::Connection::open(directory.path().join("fetch_cache.db")).unwrap();
        db.execute("UPDATE fetch_cache SET created_at=1", [])
            .unwrap();
        let cached = run(&server, &cfg);
        assert!(cached.cache_hit);
        assert_eq!(cached.document().markdown, first.document().markdown);
        assert_eq!(cached.document().metadata, first.document().metadata);
        assert_eq!(
            db.query_row("SELECT created_at FROM fetch_cache", [], |row| row
                .get::<_, i64>(0))
                .unwrap(),
            1
        );
        let fresh = run(&server, &cfg);
        assert!(!fresh.cache_hit);
        assert!(fresh.document().markdown.contains("Second body."));
        assert_eq!(fresh.document().metadata["title"], "Second title");
        assert!(run(&server, &cfg).cache_hit);
        let requests = server.requests();
        assert_eq!(requests.len(), 4);
        for request in &requests[1..3] {
            assert_eq!(header(request, "If-None-Match").as_deref(), Some("\"v1\""));
            assert_eq!(
                header(request, "If-Modified-Since").as_deref(),
                Some(modified)
            );
        }
        assert_eq!(
            header(&requests[3], "If-None-Match").as_deref(),
            Some("\"v2\"")
        );
        assert!(header(&requests[3], "If-Modified-Since").is_none());
    }

    #[test]
    fn last_modified_alone_revalidates_and_failed_conditionals_fetch_normally() {
        let (_directory, cfg) = settings();
        let modified = "Mon, 28 Sep 2026 09:00:00 GMT";
        let server = Server::new(vec![
            Reply::text("old").header("Last-Modified", modified),
            Reply::text("failure").status(503),
            Reply::text("replacement"),
        ]);
        assert!(!run(&server, &cfg).cache_hit);
        let fresh = run(&server, &cfg);
        assert!(!fresh.cache_hit);
        assert_eq!(fresh.document().markdown, "replacement");
        assert!(run(&server, &cfg).cache_hit);
        let requests = server.requests();
        assert_eq!(requests.len(), 3);
        assert_eq!(
            header(&requests[1], "If-Modified-Since").as_deref(),
            Some(modified)
        );
        assert!(header(&requests[1], "If-None-Match").is_none());
        assert!(header(&requests[2], "If-Modified-Since").is_none());
    }

    #[test]
    fn redirects_and_declared_charsets_share_the_same_decoder() {
        let (directory, cfg) = settings();
        let mut page = Reply::html("")
            .header("Content-Type", "TEXT/HTML; Charset=\"windows-1252\"")
            .header("ETag", "W/\"page\"");
        page.body =
            b"<title>Caf\xe9</title><article><p>Caf\xe9 <a href=\"child\">link</a></p></article>"
                .to_vec();
        let server = Server::new(vec![
            Reply::text("")
                .status(302)
                .header("Location", "/folder/page"),
            page,
            Reply::text("")
                .status(302)
                .header("Location", "/folder/page"),
            Reply::text("").status(304),
        ]);
        let first = run(&server, &cfg);
        assert!(first.document().markdown.contains("Café"));
        assert!(
            first
                .document()
                .markdown
                .contains(&server.url("/folder/child"))
        );
        let second = run(&server, &cfg);
        assert!(second.cache_hit);
        assert_eq!(second.document().markdown, first.document().markdown);
        let db = rusqlite::Connection::open(directory.path().join("fetch_cache.db")).unwrap();
        let final_url: String = db
            .query_row("SELECT final_url FROM fetch_cache", [], |row| row.get(0))
            .unwrap();
        assert_eq!(final_url, server.url("/folder/page"));
        assert_eq!(server.requests().len(), 4);
    }

    #[test]
    fn html_meta_charset_applies_without_a_header_charset() {
        let (_directory, mut cfg) = settings();
        cfg["cache"]["enabled"] = json!(false);
        let body = |charset: &str| {
            let mut page = Reply::html("");
            page.body = [
                format!("<meta charset={charset}><title>t</title><article><p>").as_bytes(),
                b"\xd6\xd0\xce\xc4\xc4\xda\xc8\xdd</p></article>",
            ]
            .concat();
            page
        };
        let server = Server::new(vec![
            body("gbk"),
            // The HTTP charset wins over the declaration.
            body("gbk").header("Content-Type", "text/html; charset=windows-1252"),
        ]);
        assert!(run(&server, &cfg).document().markdown.contains("中文内容"));
        assert!(run(&server, &cfg).document().markdown.contains("ÖÐÎÄÄÚÈÝ"));
    }

    #[test]
    fn bypass_refreshes_then_reuses_and_patterns_apply_to_real_requests() {
        let (_directory, mut cfg) = settings();
        let server = Server::new(vec![
            Reply::text("original").header("ETag", "v1"),
            Reply::text("refresh"),
            Reply::text("pattern refresh"),
        ]);
        run(&server, &cfg);
        cfg["cache"]["no_cache"] = json!(true);
        assert_eq!(run(&server, &cfg).document().markdown, "refresh");
        assert!(header(&server.requests()[1], "If-None-Match").is_none());
        cfg["cache"]["no_cache"] = json!(false);
        assert!(run(&server, &cfg).cache_hit);
        cfg["cache"]["no_cache_patterns"] = json!(["**/page"]);
        assert_eq!(run(&server, &cfg).document().markdown, "pattern refresh");
        cfg["cache"]["no_cache_patterns"] = json!([]);
        assert!(run(&server, &cfg).cache_hit);
        assert_eq!(server.requests().len(), 3);
    }

    #[test]
    fn strategy_provenance_uses_distinct_keys_without_bypassing_guards() {
        let (_directory, mut cfg) = settings();
        cfg["fetch"]["strategy"] = json!("auto");
        let server = Server::new(vec![
            Reply::text("unscoped"),
            Reply::text("explicit static"),
        ]);
        assert!(!run(&server, &cfg).cache_hit);
        cfg["fetch"]["strategy"] = json!("static");
        assert!(run(&server, &cfg).cache_hit);
        let explicit =
            fetch_with_context(&server.url("/page"), &cfg, Some("static"), true).unwrap();
        assert!(!explicit.cache_hit);
        assert_eq!(explicit.document().markdown, "explicit static");
        let auto = fetch_with_context(&server.url("/page"), &cfg, Some("auto"), true).unwrap();
        assert!(auto.cache_hit);
        assert_eq!(auto.document().markdown, "unscoped");
        cfg["fetch"]["strategy"] = json!("playwright");
        cfg["fetch"]["playwright"]["session_mode"] = json!("domain_persistent");
        // Persistent sessions are supported; invalid browser settings must
        // still fail before launch instead of reusing the anonymous cache.
        cfg["fetch"]["playwright"]["session_ttl_seconds"] = json!(59);
        assert!(matches!(
            fetch_with_context(&server.url("/page"), &cfg, None, true),
            Err(Error::Config(message)) if message.contains("session_ttl_seconds")
        ));
        cfg["fetch"]["strategy"] = json!("jina");
        assert!(fetch_with_context(&server.url("/page"), &cfg, None, true).is_err());
        assert_eq!(server.requests().len(), 2);
    }

    #[test]
    fn disabled_and_unwritable_storage_never_damage_successful_fetches() {
        let (directory, mut cfg) = settings();
        let blocked = directory.path().join("private-token-path");
        std::fs::write(&blocked, "file instead of directory").unwrap();
        cfg["cache"]["global_dir"] = json!(blocked);
        cfg["cache"]["enabled"] = json!(false);
        let server = Server::new(vec![
            Reply::text("first"),
            Reply::text("second"),
            Reply::text("third"),
        ]);
        let first = run(&server, &cfg);
        assert!(!first.cache_hit);
        assert!(first.document().warnings.is_empty());
        assert_eq!(run(&server, &cfg).document().markdown, "second");
        cfg["cache"]["enabled"] = json!(true);
        let third = run(&server, &cfg);
        assert_eq!(third.document().markdown, "third");
        assert_eq!(third.document().warnings, [CACHE_WARNING]);
        assert!(!third.document().warnings[0].contains("private-token-path"));
        assert!(!directory.path().join("fetch_cache.db").exists());
        assert_eq!(server.requests().len(), 3);
    }

    #[test]
    fn empty_error_and_unsolicited_304_responses_do_not_create_cache() {
        for reply in [
            Reply::text("  \n"),
            Reply::text("failure").status(500),
            Reply::text("").status(304),
        ] {
            let (directory, cfg) = settings();
            let server = Server::new(vec![reply]);
            assert!(fetch_with_context(&server.url("/page"), &cfg, None, true).is_err());
            assert!(!directory.path().join("fetch_cache.db").exists());
            assert_eq!(server.requests().len(), 1);
        }
    }

    #[test]
    fn unusable_revalidation_retains_good_entry_without_silent_stale_success() {
        for invalid in [
            Reply::html(
                "<title>Just a moment...</title><script src=\"/cdn-cgi/challenge-platform/check\"></script><p>Checking your browser before accessing this site.</p>",
            ),
            Reply::html("<p>Please enable JavaScript to continue.</p>"),
            Reply::text(" \n"),
            Reply::text("failure").status(503),
        ] {
            let (_directory, cfg) = settings();
            let server = Server::new(vec![
                Reply::text("saved").header("ETag", "v1"),
                invalid.clone(),
                invalid,
                Reply::text("").status(304),
            ]);
            run(&server, &cfg);
            assert!(fetch_with_context(&server.url("/page"), &cfg, None, true).is_err());
            let saved = run(&server, &cfg);
            assert!(saved.cache_hit);
            assert_eq!(saved.document().markdown, "saved");
            let requests = server.requests();
            assert_eq!(requests.len(), 4);
            assert!(header(&requests[2], "If-None-Match").is_none());
            assert_eq!(header(&requests[3], "If-None-Match").as_deref(), Some("v1"));
        }
    }

    #[test]
    fn challenge_signatures_in_documents_and_code_are_not_rejected() {
        for reply in [
            Reply::text("Checking your browser before accessing: g-recaptcha cf_chl_ hcaptcha.com"),
            Reply::html(
                "<title>CAPTCHA guide</title><article><p>Explain CAPTCHA integrations.</p><pre><code>Checking your browser before accessing: cf_chl_</code></pre></article>",
            ),
            Reply::html(
                "<title>Contact us</title><p>Send your message.</p><div class=\"g-recaptcha\"></div>",
            ),
            Reply::html(
                "<article><h1>Verify you are human</h1><p>This tutorial explains how identity challenges work.</p></article>",
            ),
            Reply::html(
                "<article><h1>Please enable JavaScript</h1><p>This guide explains the browser setting and progressive enhancement.</p></article>",
            ),
        ] {
            let (_directory, cfg) = settings();
            let server = Server::new(vec![reply]);
            assert!(!run(&server, &cfg).cache_hit);
            assert!(run(&server, &cfg).cache_hit);
            assert_eq!(server.requests().len(), 1);
        }
    }

    #[test]
    fn explicit_text_types_preserve_literal_html_challenge_examples() {
        for content_type in ["text/plain", "text/markdown"] {
            let (_directory, cfg) = settings();
            let example = "<html><body><p>Checking your browser before accessing this site.</p></body></html>";
            let server = Server::new(vec![
                Reply::text(example).header("Content-Type", content_type),
            ]);
            let fresh = run(&server, &cfg);
            assert_eq!(fresh.document().markdown, example);
            assert!(!fresh.cache_hit);
            let cached = run(&server, &cfg);
            assert_eq!(cached.document().markdown, example);
            assert!(cached.cache_hit);
            assert_eq!(server.requests().len(), 1);
        }
    }

    #[test]
    fn downloaded_documents_with_owned_assets_bypass_the_page_cache() {
        let (directory, cfg) = settings();
        let reply = downloadable_email();
        let server = Server::new(vec![reply.clone(), reply]);
        for _ in 0..2 {
            let outcome =
                fetch_with_context(&server.url("/document.eml"), &cfg, None, true).unwrap();
            assert!(!outcome.cache_hit);
            assert_eq!(outcome.document().assets.len(), 1);
            assert_eq!(outcome.document().assets[0].bytes, [0, 1, 2, 255]);
            assert!(
                outcome
                    .document()
                    .markdown
                    .contains(&outcome.document().assets[0].name)
            );
        }
        assert_eq!(server.requests().len(), 2);
        assert!(header(&server.requests()[1], "If-None-Match").is_none());
        assert!(!directory.path().join("fetch_cache.db").exists());
    }

    fn downloadable_email() -> Reply {
        let mail = "Subject: Download\r\nMIME-Version: 1.0\r\nContent-Type: multipart/mixed; boundary=x\r\n\r\n--x\r\nContent-Type: text/plain\r\n\r\nDocument body.\r\n--x\r\nContent-Type: application/octet-stream\r\nContent-Disposition: attachment; filename=data.bin\r\nContent-Transfer-Encoding: base64\r\n\r\nAAEC/w==\r\n--x--\r\n";
        Reply::text(mail)
            .header("Content-Type", "application/octet-stream")
            .header("ETag", "mail-v1")
    }

    #[test]
    fn successful_binary_transition_invalidates_previous_page_and_validator() {
        for validators in [true, false] {
            let (directory, mut cfg) = settings();
            let mut initial = Reply::text("old page without attachments");
            if validators {
                initial = initial.header("ETag", "old-page");
            }
            let reply = downloadable_email();
            let server = Server::new(vec![initial, reply.clone(), reply]);
            let url = server.url("/document.eml");
            assert!(
                !fetch_with_context(&url, &cfg, None, true)
                    .unwrap()
                    .cache_hit
            );
            // Without validators, force a refresh while the old TTL is fresh.
            cfg["cache"]["no_cache"] = json!(!validators);
            let fresh = fetch_with_context(&url, &cfg, None, true).unwrap();
            assert!(!fresh.cache_hit);
            assert_eq!(fresh.document().assets[0].bytes, [0, 1, 2, 255]);
            cfg["cache"]["no_cache"] = json!(false);
            let next = fetch_with_context(&url, &cfg, None, true).unwrap();
            assert!(!next.cache_hit);
            assert_eq!(next.document().assets[0].bytes, [0, 1, 2, 255]);
            let requests = server.requests();
            assert_eq!(requests.len(), 3);
            assert!(header(&requests[2], "If-None-Match").is_none());
            let db = rusqlite::Connection::open(directory.path().join("fetch_cache.db")).unwrap();
            assert_eq!(
                db.query_row("SELECT COUNT(*) FROM fetch_cache", [], |row| row
                    .get::<_, i64>(0))
                    .unwrap(),
                0
            );
        }
    }
}

#[cfg(test)]
mod pdf_tests;

#[cfg(test)]
mod robustness_tests;

#[cfg(test)]
mod bounded_fixture_io {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/support/bounded_read.rs"
    ));
}
