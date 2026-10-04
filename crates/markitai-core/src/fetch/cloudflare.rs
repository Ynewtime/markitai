//! Cloudflare, with the user's own account: Browser Rendering for
//! `-s cloudflare` (the rendered HTML goes through the same native extraction
//! and site readers as every other strategy) and Workers AI `toMarkdown` for
//! the `-b cloudflare` file backend (`fetch.cloudflare.convert_enabled`).
//!
//! Credentials come from `fetch.cloudflare.api_token` and `account_id`, each
//! a literal or an `env:NAME` reference, and otherwise from
//! `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID`. No message carries the
//! token, the account id or an endpoint (which holds the account id).

use super::consent::{self, Consent};
use super::remote::{self, Service, Services};
use super::{client, sites};
use crate::{Document, Error, Result, config, formats};
use reqwest::blocking::multipart;
use serde_json::{Map, Value, json};
use std::collections::HashMap;
use std::path::Path;
use url::Url;

pub(crate) struct Credentials {
    token: String,
    account: String,
}

impl Credentials {
    fn secrets(&self) -> [&str; 2] {
        [&self.token, &self.account]
    }

    fn endpoint(&self, base: &str, path: &str) -> Result<Url> {
        Url::parse(&format!(
            "{}/accounts/{}/{path}",
            base.trim_end_matches('/'),
            self.account
        ))
        .map_err(|_| Error::Config("The Cloudflare endpoint could not be formed".into()))
    }
}

/// The account's credentials; `None` when either is missing. With `strict`,
/// an `env:NAME` reference to a variable that is not set is an error.
pub(crate) fn credentials(
    cfg: &Value,
    vars: &HashMap<String, String>,
    strict: bool,
) -> Result<Option<Credentials>> {
    let text = |pointer: &str| {
        cfg.pointer(pointer)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
    };
    let token = config::resolve_optional(
        text("/fetch/cloudflare/api_token"),
        Some("CLOUDFLARE_API_TOKEN"),
        vars,
        strict,
    )?
    .map(|token| token.trim().to_owned())
    .filter(|token| !token.is_empty());
    let account = config::resolve_optional(
        text("/fetch/cloudflare/account_id"),
        Some("CLOUDFLARE_ACCOUNT_ID"),
        vars,
        strict,
    )?
    .map(|account| account.trim().to_owned())
    .filter(|account| !account.is_empty());
    let (Some(token), Some(account)) = (token, account) else {
        return Ok(None);
    };
    if token
        .chars()
        .any(|ch| ch.is_control() || ch.is_whitespace())
    {
        return Err(Error::Config(
            "fetch.cloudflare.api_token is not a valid API token".into(),
        ));
    }
    if account.len() > 64
        || !account
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_')
    {
        return Err(Error::Config(
            "fetch.cloudflare.account_id is not a Cloudflare account ID".into(),
        ));
    }
    Ok(Some(Credentials { token, account }))
}

/// Whether the auto chain can use Cloudflare: both credentials resolve.
pub(crate) fn configured(cfg: &Value, vars: &HashMap<String, String>) -> bool {
    matches!(credentials(cfg, vars, false), Ok(Some(_)))
}

/// Local prerequisites only: this never contacts Cloudflare, verifies account
/// permissions, or exposes credentials. The caller controls the configuration.
pub fn cloudflare_capabilities(cfg: &Value) -> Value {
    capabilities_with_vars(cfg, &config::environment())
}

fn capabilities_with_vars(cfg: &Value, vars: &HashMap<String, String>) -> Value {
    let (configured, configuration_reason) = match credentials(cfg, vars, true) {
        Ok(Some(_)) => (true, None),
        Ok(None) => (false, Some("not_configured")),
        Err(_) => (false, Some("invalid_configuration")),
    };
    let reason = if consent::hard_off(vars) || consent::configured(cfg) == Consent::Never {
        Some("disabled_by_policy")
    } else {
        configuration_reason
    };
    let available = reason.is_none();
    json!({
        "configured": configured,
        "available": available,
        "reason": reason,
        "browser_rendering": available,
        "file_conversion": available,
        "file_extensions": FORMATS.iter().map(|(extension, _)| *extension).collect::<Vec<_>>()
    })
}

pub(crate) fn missing_credentials() -> Error {
    Error::Config("Cloudflare needs an API token and an account ID: set fetch.cloudflare.api_token and fetch.cloudflare.account_id (a value or an env:NAME reference), or CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID. Create the token at dash.cloudflare.com/profile/api-tokens with Account / Browser Rendering / Edit (for -s cloudflare) and Account / Workers AI / Read (for -b cloudflare); the account ID is on the account's home page in the dashboard.".into())
}

// ---- Browser Rendering ----------------------------------------------------

/// Requests for these resources are dropped unless
/// `fetch.cloudflare.reject_resource_patterns` says otherwise.
const DEFAULT_REJECTED: [&str; 5] = [
    "/\\.css$/",
    "/\\.woff2?$/",
    "/\\.ttf$/",
    "/\\.eot$/",
    "/\\.otf$/",
];

fn payload(source: &str, settings: &Value, vars: &HashMap<String, String>) -> Result<Value> {
    let mut payload = json!({"url": source});
    let mut goto = Map::new();
    if let Some(timeout) = settings
        .get("timeout")
        .and_then(Value::as_u64)
        .filter(|timeout| *timeout > 0)
    {
        goto.insert("timeout".into(), json!(timeout));
    }
    if let Some(wait) = settings
        .get("wait_until")
        .and_then(Value::as_str)
        .filter(|wait| !wait.is_empty())
    {
        goto.insert("waitUntil".into(), json!(wait));
    }
    if !goto.is_empty() {
        payload["gotoOptions"] = Value::Object(goto);
    }
    payload["rejectRequestPattern"] = match settings.get("reject_resource_patterns") {
        Some(Value::Array(patterns)) => Value::Array(patterns.clone()),
        _ => json!(DEFAULT_REJECTED),
    };
    if let Some(agent) = settings
        .get("user_agent")
        .and_then(Value::as_str)
        .filter(|agent| !agent.is_empty())
    {
        payload["userAgent"] = json!(agent);
    }
    if let Some(cookies) = settings
        .get("cookies")
        .and_then(Value::as_array)
        .filter(|cookies| !cookies.is_empty())
    {
        payload["cookies"] = Value::Array(cookies.clone());
    }
    if let Some(selector) = settings
        .get("wait_for_selector")
        .and_then(Value::as_str)
        .filter(|selector| !selector.is_empty())
    {
        payload["waitForSelector"] = json!({"selector": selector});
    }
    if let Some(credentials) = settings.get("http_credentials").and_then(Value::as_object) {
        let mut authenticate = Map::new();
        for (key, value) in credentials {
            if let Some(value) = value.as_str() {
                let value = config::resolve_env_value(value, vars, true)?.unwrap_or_default();
                authenticate.insert(key.clone(), json!(value));
            }
        }
        payload["authenticate"] = Value::Object(authenticate);
    }
    Ok(payload)
}

/// `X-Browser-Ms-Used` (`2378.702880859375`) as a whole number of
/// milliseconds, the way other durations are recorded (`duration_ms`).
fn whole_milliseconds(header: &str) -> Option<u64> {
    let milliseconds: f64 = header.trim().parse().ok()?;
    // Rounded, and bounded to what a u64 holds exactly.
    (milliseconds.is_finite() && (0.0..9.0e15).contains(&milliseconds))
        .then(|| milliseconds.round() as u64)
}

/// The page rendered by Cloudflare Browser Rendering's `/content`, read by the
/// native HTML extraction. A 429 is repeated twice, after a pause that
/// doubles; at most two renders run at a time.
pub(crate) fn render(
    source: &str,
    url: &Url,
    cfg: &Value,
    services: &Services<'_>,
) -> Result<Document> {
    let credentials = credentials(cfg, &services.vars, true)?.ok_or_else(missing_credentials)?;
    let settings = cfg
        .pointer("/fetch/cloudflare")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let payload = payload(source, &settings, &services.vars)?;
    let mut endpoint = credentials.endpoint(&services.cloudflare, "browser-rendering/content")?;
    if let Some(ttl) = settings
        .get("cache_ttl")
        .and_then(Value::as_u64)
        .filter(|ttl| *ttl > 0)
    {
        endpoint
            .query_pairs_mut()
            .append_pair("cacheTTL", &ttl.to_string());
    }
    let timeout_ms = settings
        .get("timeout")
        .and_then(Value::as_u64)
        .unwrap_or(30_000);
    let client = client((timeout_ms / 1000 + 10).clamp(60, 600))?;
    let secrets = credentials.secrets();
    let (bytes, browser_ms) = {
        let _slot = services.slots.hold();
        let mut pause = services.retry_pause;
        let mut attempts = 1;
        let response = loop {
            let response = client
                .post(endpoint.clone())
                .bearer_auth(&credentials.token)
                .json(&payload)
                .send()
                .map_err(|error| remote::transport(Service::Cloudflare, error))?;
            if response.status().as_u16() == 429 && attempts < 3 {
                attempts += 1;
                std::thread::sleep(pause);
                pause *= 2;
                continue;
            }
            break response;
        };
        let browser_ms = response
            .headers()
            .get("x-browser-ms-used")
            .and_then(|value| value.to_str().ok())
            .and_then(whole_milliseconds);
        (
            remote::answer(Service::Cloudflare, response, &secrets)?,
            browser_ms,
        )
    };
    let envelope: Value = serde_json::from_slice(&bytes).map_err(|_| {
        Error::Fetch("The cloudflare service returned an answer that is not JSON".into())
    })?;
    if envelope.get("success").and_then(Value::as_bool) != Some(true) {
        return Err(Error::Fetch(match remote::service_said(&bytes, &secrets) {
            Some(said) => format!("The cloudflare service could not render the page: {said}"),
            None => "The cloudflare service could not render the page".into(),
        }));
    }
    // `meta` (optional, as each of its fields) describes the origin's answer:
    // `finalUrl` after redirects and `status`, the HTTP status the origin
    // returned, read like Jina's `httpStatus`. Without it the page is judged
    // by its markup alone.
    let read_at = remote::read_at(
        envelope.pointer("/meta/finalUrl").and_then(Value::as_str),
        url,
    );
    if let Some(failure) = envelope
        .pointer("/meta/status")
        .and_then(Value::as_u64)
        .and_then(|status| u16::try_from(status).ok())
        .and_then(|status| remote::origin_status(Service::Cloudflare, status, &read_at))
    {
        return Err(failure);
    }
    let html = envelope
        .get("result")
        .and_then(Value::as_str)
        .filter(|html| !html.trim().is_empty())
        .ok_or_else(|| Error::Fetch("The cloudflare service returned no content".into()))?;
    // The same judgement as every other reader's page: a verification or
    // login page, a challenge, a JavaScript notice or a JSON error answer the
    // browser showed as the page is a failure.
    if let Some(failure) = sites::Shown::html(&read_at, html).read_by(Service::Cloudflare.name()) {
        return Err(Error::Fetch(failure));
    }
    let mut document = match formats::extract_html(html, Some(source)) {
        Err(Error::Conversion(_)) => {
            return Err(Error::Fetch(
                "The cloudflare service rendered a page with no extractable content".into(),
            ));
        }
        result => result?,
    };
    document
        .metadata
        .insert("renderer".into(), json!("cloudflare"));
    if let Some(browser_ms) = browser_ms {
        document
            .metadata
            .insert("browser_ms_used".into(), json!(browser_ms));
    }
    Ok(document)
}

// ---- Workers AI toMarkdown ------------------------------------------------

/// Extensions Workers AI converts, with their media types.
const FORMATS: [(&str, &str); 17] = [
    ("pdf", "application/pdf"),
    (
        "docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ),
    (
        "xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ),
    ("xlsm", "application/vnd.ms-excel.sheet.macroEnabled.12"),
    (
        "xlsb",
        "application/vnd.ms-excel.sheet.binary.macroEnabled.12",
    ),
    ("xls", "application/vnd.ms-excel"),
    ("et", "application/vnd.ms-excel"),
    ("jpeg", "image/jpeg"),
    ("jpg", "image/jpeg"),
    ("png", "image/png"),
    ("webp", "image/webp"),
    ("svg", "image/svg+xml"),
    ("csv", "text/csv"),
    ("xml", "application/xml"),
    ("ods", "application/vnd.oasis.opendocument.spreadsheet"),
    ("odt", "application/vnd.oasis.opendocument.text"),
    ("numbers", "application/vnd.apple.numbers"),
];

const IMAGES: [&str; 5] = ["jpeg", "jpg", "png", "webp", "svg"];

fn media_type(extension: &str) -> Option<&'static str> {
    let extension = extension.to_ascii_lowercase();
    FORMATS
        .iter()
        .find(|(known, _)| *known == extension)
        .map(|(_, mime)| *mime)
}

/// Whether the Cloudflare backend converts this local file: the backend is on
/// and Workers AI reads the format. Other files keep the native readers.
pub(crate) fn converts(cfg: &Value, path: &Path, extension: &str) -> bool {
    config::enabled(cfg, "/fetch/cloudflare/convert_enabled")
        && media_type(extension).is_some()
        && path.is_file()
}

/// A local file converted by Workers AI `toMarkdown` in the user's account.
pub(crate) fn convert_file(path: &Path, extension: &str, cfg: &Value) -> Result<Document> {
    convert_file_with(path, extension, cfg, &Services::production())
}

pub(crate) fn convert_file_with(
    path: &Path,
    extension: &str,
    cfg: &Value,
    services: &Services<'_>,
) -> Result<Document> {
    if consent::hard_off(&services.vars) || consent::configured(cfg) == Consent::Never {
        return Err(Error::Config("Cloudflare file conversion sends the file to Cloudflare, which --no-remote-fetch, MARKITAI_NO_REMOTE_FETCH or fetch.remote_consent=never forbids; use -b native".into()));
    }
    let credentials = credentials(cfg, &services.vars, true)?.ok_or_else(missing_credentials)?;
    let mime = media_type(extension).ok_or_else(|| {
        Error::Unsupported(format!(
            "Cloudflare does not convert .{extension} files; use -b native"
        ))
    })?;
    let name = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| format!("document.{extension}"));
    let bytes = std::fs::read(path)?;
    let part = multipart::Part::bytes(bytes)
        .file_name(name.clone())
        .mime_str(mime)
        .map_err(|error| remote::transport(Service::Cloudflare, error))?;
    let form = multipart::Form::new().part("files", part);
    let endpoint = credentials.endpoint(&services.cloudflare, "ai/tomarkdown")?;
    let secrets = credentials.secrets();
    let failed = |error: Error| match error {
        Error::Fetch(message) => Error::Conversion(message),
        other => other,
    };
    let response = client(60)?
        .post(endpoint)
        .bearer_auth(&credentials.token)
        .multipart(form)
        .send()
        .map_err(|error| failed(remote::transport(Service::Cloudflare, error)))?;
    let body = remote::answer(Service::Cloudflare, response, &secrets).map_err(failed)?;
    let envelope: Value = serde_json::from_slice(&body).map_err(|_| {
        Error::Conversion("The cloudflare service returned an answer that is not JSON".into())
    })?;
    let result = envelope
        .pointer("/result/0")
        .filter(|result| result.is_object())
        .ok_or_else(|| {
            Error::Conversion(match remote::service_said(&body, &secrets) {
                Some(said) => format!("The cloudflare service converted nothing: {said}"),
                None => "The cloudflare service converted nothing".into(),
            })
        })?;
    if result.get("format").and_then(Value::as_str) == Some("error") {
        let reason = result
            .get("error")
            .and_then(Value::as_str)
            .map(|reason| remote::without_secrets(reason, &secrets))
            .unwrap_or_else(|| "no reason given".into());
        return Err(Error::Conversion(format!(
            "The cloudflare service could not convert the file: {reason}"
        )));
    }
    let markdown = result
        .get("data")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if markdown.trim().is_empty() {
        return Err(Error::Conversion(
            "The cloudflare service returned an empty conversion".into(),
        ));
    }
    let (markdown, metadata) = unwrapped(markdown, &name);
    let mut document = Document {
        markdown,
        metadata,
        ..Default::default()
    };
    document
        .metadata
        .insert("converter".into(), json!("cloudflare-tomarkdown"));
    if let Some(tokens) = result.get("tokens").filter(|tokens| tokens.is_number()) {
        document.metadata.insert("tokens".into(), tokens.clone());
    }
    if IMAGES.contains(&extension.to_ascii_lowercase().as_str()) {
        document.warnings.push(
            "Cloudflare converts images with a Workers AI model, which uses the account's Neurons allowance".into(),
        );
    }
    Ok(document)
}

// ---- the toMarkdown wrapper -----------------------------------------------

/// Workers AI's Markdown without the frame it puts around every document:
/// a `# <file name>` heading, a `## Metadata` list of the file's properties
/// (`- PDFFormatVersion=1.4`, `- Creator=Writer`, …) and a `## Contents`
/// heading over the content. Of the properties, the title, the author and the
/// creation date become metadata under the names the native readers use
/// (`title`, `author`, `date`); the rest describe the file, not the document.
/// A PDF's `### Page N` headings become the native PDF reader's page markers
/// (`<!-- Page number: N -->`), and their number `pages`: a page is not a
/// section of the document, so it is no heading, and a file reads alike with
/// either backend. Markdown without the frame is returned as it came.
fn unwrapped(markdown: &str, name: &str) -> (String, Map<String, Value>) {
    let mut metadata = Map::new();
    let lines: Vec<&str> = markdown.lines().collect();
    let mut at = 0;
    let skip_blank = |at: &mut usize| {
        while lines.get(*at).is_some_and(|line| line.trim().is_empty()) {
            *at += 1;
        }
    };
    skip_blank(&mut at);
    let wrapper = lines
        .get(at)
        .and_then(|line| line.strip_prefix("# "))
        .is_some_and(|heading| heading.trim().replace('\\', "").eq_ignore_ascii_case(name));
    if !wrapper {
        return (markdown.to_owned(), metadata);
    }
    at += 1;
    skip_blank(&mut at);
    let mut properties = Vec::new();
    if lines
        .get(at)
        .is_some_and(|line| line.trim() == "## Metadata")
    {
        at += 1;
        // Only the list itself: whatever follows it is the document's.
        while let Some(line) = lines.get(at).map(|line| line.trim()) {
            if line.is_empty() {
                at += 1;
                continue;
            }
            let Some((key, value)) = line
                .strip_prefix("- ")
                .or_else(|| line.strip_prefix("* "))
                .and_then(|property| {
                    property
                        .split_once('=')
                        .or_else(|| property.split_once(": "))
                })
            else {
                break;
            };
            properties.push((key.trim().to_owned(), value.trim().to_owned()));
            at += 1;
        }
    }
    let contents = lines
        .get(at)
        .is_some_and(|line| line.trim() == "## Contents");
    if contents {
        at += 1;
    }
    let mut pages = 0;
    let body: Vec<String> = lines[at..]
        .iter()
        .map(|line| {
            if contents && line.trim() == format!("### Page {}", pages + 1) {
                pages += 1;
                format!("<!-- Page number: {pages} -->")
            } else {
                (*line).to_owned()
            }
        })
        .collect();
    for (key, value) in properties {
        let value = value.trim_matches('"').trim();
        if value.is_empty() {
            continue;
        }
        match key.to_ascii_lowercase().as_str() {
            "title" if !file_title(value, name) => {
                metadata.insert("title".into(), json!(value));
            }
            "author" => {
                metadata.insert("author".into(), json!(value));
            }
            "creationdate" => {
                if let Some(date) = pdf_date(value) {
                    metadata.insert("date".into(), json!(date));
                }
            }
            _ => {}
        }
    }
    if pages > 0 {
        metadata.insert("pages".into(), json!(pages));
    }
    (body.join("\n").trim().to_owned(), metadata)
}

/// A title property that only names the file the document was made from
/// (`report.docx`, `Microsoft Word - report.doc`) or this file, which the
/// document's first heading names better.
fn file_title(title: &str, name: &str) -> bool {
    let lower = title.to_ascii_lowercase();
    let stem = name.rsplit_once('.').map_or(name, |(stem, _)| stem);
    lower == name.to_ascii_lowercase()
        || lower == stem.to_ascii_lowercase()
        || [
            "microsoft word - ",
            "microsoft powerpoint - ",
            "microsoft excel - ",
        ]
        .iter()
        .any(|prefix| lower.starts_with(prefix))
        || lower.rsplit_once('.').is_some_and(|(stem, extension)| {
            !stem.trim().is_empty()
                && (media_type(extension).is_some()
                    || matches!(
                        extension,
                        "doc" | "html" | "htm" | "txt" | "md" | "pptx" | "ppt" | "rtf"
                    ))
        })
}

/// A PDF date (`D:20170816144228+02'00'`, any trailing part optional) as
/// RFC 3339 (`2017-08-16T14:42:28+02:00`), or the date alone when it has no
/// time. `None` when it is not one.
fn pdf_date(value: &str) -> Option<String> {
    let text = value.strip_prefix("D:").unwrap_or(value);
    let digits: String = text.chars().take_while(char::is_ascii_digit).collect();
    let rest = &text[digits.len()..];
    let part = |from: usize, to: usize, default: u32| -> Option<u32> {
        match digits.get(from..to) {
            Some(part) => part.parse().ok(),
            None if digits.len() <= from => Some(default),
            None => None,
        }
    };
    if digits.len() < 4 {
        return None;
    }
    let year = part(0, 4, 0)?;
    let (month, day) = (part(4, 6, 1)?, part(6, 8, 1)?);
    let (hour, minute, second) = (part(8, 10, 0)?, part(10, 12, 0)?, part(12, 14, 0)?);
    chrono::NaiveDate::from_ymd_opt(i32::try_from(year).ok()?, month, day)?;
    if hour > 23 || minute > 59 || second > 60 {
        return None;
    }
    let date = format!("{year:04}-{month:02}-{day:02}");
    if digits.len() <= 8 {
        return Some(date);
    }
    let offset = match rest.chars().next() {
        Some('Z') => "Z".to_owned(),
        Some(sign @ ('+' | '-')) => {
            // `HH'mm'`, as the PDF format writes it, or `HHmm`.
            let groups: Vec<&str> = rest[1..]
                .split(|ch: char| !ch.is_ascii_digit())
                .filter(|group| !group.is_empty())
                .collect();
            let (hours, minutes): (u32, u32) = match groups.as_slice() {
                [both] if both.len() == 4 => (both[..2].parse().ok()?, both[2..].parse().ok()?),
                [hours] => (hours.parse().ok()?, 0),
                [hours, minutes, ..] => (hours.parse().ok()?, minutes.parse().ok()?),
                [] => return None,
            };
            if hours > 23 || minutes > 59 {
                return None;
            }
            format!("{sign}{hours:02}:{minutes:02}")
        }
        _ => String::new(),
    };
    Some(format!("{date}T{hour:02}:{minute:02}:{second:02}{offset}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pdf_dates_become_rfc_3339() {
        assert_eq!(
            pdf_date("D:20170816144228+02'00'").as_deref(),
            Some("2017-08-16T14:42:28+02:00")
        );
        assert_eq!(
            pdf_date("D:20240102030405Z").as_deref(),
            Some("2024-01-02T03:04:05Z")
        );
        assert_eq!(
            pdf_date("D:20240102030405-0530").as_deref(),
            Some("2024-01-02T03:04:05-05:30")
        );
        assert_eq!(
            pdf_date("D:202401020304").as_deref(),
            Some("2024-01-02T03:04:00")
        );
        assert_eq!(pdf_date("D:20240102").as_deref(), Some("2024-01-02"));
        assert_eq!(pdf_date("D:2024").as_deref(), Some("2024-01-01"));
        for not_a_date in [
            "",
            "D:",
            "D:12",
            "D:20241302",
            "D:20240230",
            "D:2024010225",
            "D:20240102030405+25'00'",
            "yesterday",
        ] {
            assert!(pdf_date(not_a_date).is_none(), "{not_a_date}");
        }
    }

    #[test]
    fn browser_time_is_a_whole_number_of_milliseconds() {
        assert_eq!(whole_milliseconds("2378.702880859375"), Some(2379));
        assert_eq!(whole_milliseconds(" 1234 "), Some(1234));
        assert_eq!(whole_milliseconds("0.4"), Some(0));
        for not_one in ["", "-1", "abc", "1e30", "NaN", "inf"] {
            assert_eq!(whole_milliseconds(not_one), None, "{not_one}");
        }
    }

    #[test]
    fn markdown_without_the_frame_is_left_alone() {
        let (text, metadata) = unwrapped("# Report\n\nConverted by Workers AI.", "report.pdf");
        assert_eq!(text, "# Report\n\nConverted by Workers AI.");
        assert!(metadata.is_empty());
        // The wrapper alone, as for a DOCX file.
        let (text, _) = unwrapped("# notes.docx\n\n# Notes\n\nText.", "notes.docx");
        assert_eq!(text, "# Notes\n\nText.");
        // A property list followed by the content, without `## Contents`.
        let (text, metadata) = unwrapped(
            "# data.csv\n\n## Metadata\n\n- Rows=2\n\n| a | b |\n|---|---|\n| 1 | 2 |",
            "data.csv",
        );
        assert_eq!(text, "| a | b |\n|---|---|\n| 1 | 2 |");
        assert!(metadata.is_empty());
        assert!(file_title("report.docx", "report.pdf"));
        assert!(file_title("Report", "report.pdf"));
        assert!(file_title("Microsoft Word - Q3.doc", "q3.pdf"));
        assert!(!file_title("Version 2.0 notes", "q3.pdf"));
        assert!(!file_title("Node.js in practice", "book.pdf"));
    }
}

#[cfg(test)]
mod capability_tests {
    use super::*;
    #[test]
    fn local_readiness_never_returns_credentials_or_claims_permissions() {
        let vars = HashMap::new();
        let cfg = json!({"fetch":{"remote_consent":"ask","cloudflare":{
            "api_token":"synthetic-token","account_id":"synthetic-account"}}});
        let value = capabilities_with_vars(&cfg, &vars);
        assert_eq!(value["available"], true);
        assert_eq!(value["configured"], true);
        assert_eq!(value["browser_rendering"], true);
        assert_eq!(value["file_conversion"], true);
        assert!(
            value["file_extensions"]
                .as_array()
                .unwrap()
                .contains(&json!("pdf"))
        );
        let text = value.to_string();
        for secret in [
            "synthetic-token",
            "synthetic-account",
            "api_token",
            "account_id",
            "https:",
        ] {
            assert!(!text.contains(secret));
        }
        assert!(value.get("permissions_verified").is_none());
    }
    #[test]
    fn missing_invalid_never_and_hard_off_are_distinct_without_network() {
        let mut cfg = json!({"fetch":{"remote_consent":"ask","cloudflare":{}}});
        let mut vars = HashMap::new();
        assert_eq!(
            capabilities_with_vars(&cfg, &vars)["reason"],
            "not_configured"
        );
        cfg["fetch"]["cloudflare"] =
            json!({"api_token":"synthetic token","account_id":"synthetic-account"});
        assert_eq!(
            capabilities_with_vars(&cfg, &vars)["reason"],
            "invalid_configuration"
        );
        cfg["fetch"]["cloudflare"]["api_token"] = json!("synthetic-token");
        cfg["fetch"]["remote_consent"] = json!("never");
        assert_eq!(
            capabilities_with_vars(&cfg, &vars)["reason"],
            "disabled_by_policy"
        );
        cfg["fetch"]["remote_consent"] = json!("always");
        vars.insert("MARKITAI_NO_REMOTE_FETCH".into(), "true".into());
        assert_eq!(
            capabilities_with_vars(&cfg, &vars)["reason"],
            "disabled_by_policy"
        );
        assert_eq!(capabilities_with_vars(&cfg, &vars)["available"], false);
    }
}
