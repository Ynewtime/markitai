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
use super::{client, html_rejection, sites};
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
            .map(str::to_owned);
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
    let html = envelope
        .get("result")
        .and_then(Value::as_str)
        .filter(|html| !html.trim().is_empty())
        .ok_or_else(|| Error::Fetch("The cloudflare service returned no content".into()))?;
    // The same checks as the local browser's page: a verification or
    // challenge page is a failure, not the content.
    if let Some(message) = sites::verification_page(url, html) {
        return Err(Error::Fetch(message));
    }
    if html_rejection(html).is_some() {
        return Err(Error::Fetch(
            "The cloudflare service was shown a challenge or JavaScript notice instead of the content"
                .into(),
        ));
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
        .file_name(name)
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
    let mut document = Document {
        markdown: markdown.to_owned(),
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
