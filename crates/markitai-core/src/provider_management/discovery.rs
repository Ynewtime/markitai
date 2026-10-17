use super::checked_url;
use crate::{Error, Result, llm::ProbeReason as Reason};
use reqwest::{blocking::Client, redirect::Policy};
use serde_json::{Value, json};
use std::{collections::HashSet, io::Read, time::Duration};
const MAX_RESPONSE: u64 = 2 * 1024 * 1024;
const MAX_MODELS: usize = 1000;
/// How long a discovery request may take (shorter under test).
pub(super) const TIMEOUT: Duration = if cfg!(test) {
    Duration::from_secs(3)
} else {
    Duration::from_secs(10)
};
/// The provider's own words in a refusal, at most this many characters.
const MAX_MESSAGE: usize = 300;
fn failure(message: &str) -> Error {
    Error::Conversion(message.into())
}
/// A discovery that listed nothing and why: `reason` is one of the reasons
/// a failed connection test reports, `key_variable` the variable a missing
/// key would come from.
pub(super) fn failed(
    provider: &str,
    reason: Reason,
    detail: &str,
    key_variable: Option<&str>,
) -> Value {
    let mut value = json!({"provider":provider,"status":"unavailable","source":"live_api","authoritative":false,"cached":false,"stale":false,"models":[],"detail":detail,"reason":reason.name()});
    if let Some(variable) = key_variable {
        value["key_variable"] = variable.into();
    }
    value
}

/// The provider's message in a refused request's body: a JSON
/// `error.message`, `message` or `error` string, else a short text snippet
/// without markup or control characters. The key sent is never repeated.
fn provider_message(body: &[u8], key: Option<&str>) -> String {
    let parsed: Option<Value> = serde_json::from_slice(body).ok();
    let text = parsed
        .as_ref()
        .and_then(|value| {
            ["/error/message", "/message", "/error", "/detail"]
                .iter()
                .find_map(|pointer| value.pointer(pointer).and_then(Value::as_str))
                .map(str::to_owned)
        })
        .unwrap_or_else(|| {
            if parsed.is_some() {
                return String::new();
            }
            let text = String::from_utf8_lossy(body);
            let mut plain = String::with_capacity(text.len().min(4096));
            let mut markup = false;
            for character in text.chars().take(64 * 1024) {
                match character {
                    '<' => markup = true,
                    '>' if markup => {
                        markup = false;
                        plain.push(' ');
                    }
                    _ if markup => {}
                    other => plain.push(other),
                }
            }
            plain
        });
    let mut words = text
        .split(|c: char| c.is_whitespace() || c.is_control())
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    if let Some(key) = key.filter(|key| key.len() >= 4) {
        words = words.replace(key, "…");
    }
    let mut message: String = words.chars().take(MAX_MESSAGE).collect();
    if words.chars().count() > MAX_MESSAGE {
        message.push('…');
    }
    message
}

/// How a refused discovery request is reported: 401/403 is a rejected key,
/// or a missing one when none was sent (`keyless_variable` names the variable
/// it would come from); 429 a rate limit; a redirect, which discovery never
/// follows, a network failure; anything else, 404 included (the address
/// lists no models), a provider error.
fn refused(status: u16, keyless: bool) -> Reason {
    match status {
        401 | 403 if keyless => Reason::NoKey,
        401 | 403 => Reason::KeyRejected,
        429 => Reason::RateLimited,
        300..=399 => Reason::Network,
        _ => Reason::ProviderError,
    }
}

/// List a provider's models, or say why not as [`failed`] does.
pub(super) fn load(
    provider: &str,
    base: &str,
    key: Option<&str>,
    keyless_variable: Option<&str>,
) -> std::result::Result<Value, Value> {
    let fail = |reason: Reason, detail: &str| failed(provider, reason, detail, None);
    let mut url = checked_url(base).map_err(|_| {
        fail(
            Reason::ProviderError,
            "Provider endpoint is not a valid HTTP(S) URL",
        )
    })?;
    let path = url.path().trim_end_matches('/').to_owned();
    let suffix = match provider {
        "ollama" => "api/tags",
        "azure" if !path.ends_with("/v1") => "openai/models",
        _ => "models",
    };
    url.set_path(&format!("{path}/{suffix}"));
    match provider {
        "anthropic" => {
            url.query_pairs_mut().append_pair("limit", "1000");
        }
        "gemini" => {
            url.query_pairs_mut().append_pair("pageSize", "1000");
        }
        "azure" if !path.ends_with("/v1") => {
            url.query_pairs_mut()
                .append_pair("api-version", "2024-10-21");
        }
        _ => {}
    }
    let client = crate::proxy::http_api(
        Client::builder()
            .redirect(Policy::none())
            .connect_timeout(Duration::from_secs(3))
            .timeout(TIMEOUT),
    )
    .map_err(|_| fail(Reason::ProviderError, "Proxy settings are malformed"))?
    .build()
    .map_err(|_| {
        fail(
            Reason::ProviderError,
            "Cannot create model discovery client",
        )
    })?;
    let mut request = client.get(url).header("accept", "application/json");
    if let Some(key) = key {
        request = match provider {
            "anthropic" => request.header("x-api-key", key),
            "azure" => request.header("api-key", key),
            // A header rather than `?key=`, which request logs and proxies record.
            "gemini" => request.header("x-goog-api-key", key),
            "ollama" => request,
            _ => request.bearer_auth(key),
        };
    }
    if provider == "anthropic" {
        request = request.header("anthropic-version", "2023-06-01");
    }
    let response = request.send().map_err(|error| {
        if error.is_timeout() {
            return fail(Reason::Timeout, "Model discovery timed out");
        }
        match crate::llm::transport_cause(error) {
            Some(cause) => fail(
                Reason::Network,
                &format!("Model discovery request failed: {cause}"),
            ),
            None => fail(Reason::Network, "Model discovery request failed"),
        }
    })?;
    let status = response.status().as_u16();
    if (300..400).contains(&status) {
        return Err(fail(
            Reason::Network,
            &format!(
                "Model discovery request failed: the address redirected (HTTP {status}), which discovery does not follow"
            ),
        ));
    }
    if !response.status().is_success() {
        let mut body = Vec::new();
        let _ = response.take(64 * 1024).read_to_end(&mut body);
        let message = provider_message(&body, key);
        let detail = if message.is_empty() {
            format!("Model discovery returned HTTP {status}")
        } else {
            format!("Model discovery returned HTTP {status}: {message}")
        };
        // Ollama is sent no key even when one is configured.
        let keyless = key.is_none() || provider == "ollama";
        let reason = refused(status, keyless);
        let variable = (reason == Reason::NoKey)
            .then_some(keyless_variable)
            .flatten();
        return Err(failed(provider, reason, &detail, variable));
    }
    let unreadable = |detail: &str| fail(Reason::ProviderError, detail);
    if response
        .content_length()
        .is_some_and(|length| length > MAX_RESPONSE)
    {
        return Err(unreadable("Model discovery response exceeds 2 MiB"));
    }
    let bytes = crate::platform::read_limited(response, MAX_RESPONSE)
        .map_err(|error| {
            if error.kind() == std::io::ErrorKind::TimedOut {
                fail(Reason::Timeout, "Model discovery timed out")
            } else {
                unreadable("Cannot read model discovery response")
            }
        })?
        .ok_or_else(|| unreadable("Model discovery response exceeds 2 MiB"))?;
    let data: Value = serde_json::from_slice(&bytes)
        .map_err(|_| unreadable("Model discovery response is not JSON"))?;
    parse(provider, &data).map_err(|error| {
        let detail = match error {
            Error::Conversion(detail) => detail,
            _ => "Model discovery response has no model list".into(),
        };
        unreadable(&detail)
    })
}
pub(super) fn parse(provider: &str, data: &Value) -> Result<Value> {
    let records = data
        .get(if matches!(provider, "gemini" | "ollama") {
            "models"
        } else {
            "data"
        })
        .and_then(Value::as_array)
        // Together AI answers with the bare array of model records.
        .or_else(|| data.as_array().filter(|_| provider == "together_ai"))
        .ok_or_else(|| failure("Model discovery response has no model list"))?;
    if records.len() > MAX_MODELS {
        return Err(failure("Model discovery exceeds 1000 records"));
    }
    let mut models = Vec::new();
    let mut seen = HashSet::new();
    for record in records {
        if provider == "gemini"
            && let Some(methods) = record
                .get("supportedGenerationMethods")
                .and_then(Value::as_array)
            && !methods.is_empty()
            && !methods.iter().any(|method| method == "generateContent")
        {
            continue;
        }
        let Some(raw) = record
            .get(if matches!(provider, "gemini" | "ollama") {
                "name"
            } else {
                "id"
            })
            .and_then(Value::as_str)
        else {
            continue;
        };
        let raw = if provider == "gemini" {
            raw.strip_prefix("models/").unwrap_or(raw)
        } else {
            raw
        };
        if raw.is_empty() {
            continue;
        }
        if raw.len() > 1024 || raw.chars().any(char::is_control) {
            return Err(failure("Model identifier is invalid or too long"));
        }
        let prefix = if provider == "custom" {
            "openai"
        } else {
            provider
        };
        let model = format!("{prefix}/{raw}");
        if !seen.insert(model.clone()) {
            continue;
        }
        let label = record
            .get("display_name")
            .or_else(|| record.get("displayName"))
            .or_else(|| record.get("name"))
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .unwrap_or(raw);
        let label = label
            .chars()
            .filter(|c| !c.is_control())
            .take(1024)
            .collect::<String>();
        let vision = provider == "anthropic"
            || provider == "openrouter"
                && record
                    .pointer("/architecture/input_modalities")
                    .and_then(Value::as_array)
                    .is_some_and(|items| items.iter().any(|item| item == "image"));
        models.push(json!({"model":model,"label":label,"supports_vision":vision}));
    }
    let paginated = data.get("has_more").and_then(Value::as_bool) == Some(true)
        || data
            .get("nextPageToken")
            .and_then(Value::as_str)
            .is_some_and(|s| !s.is_empty());
    let authoritative = provider != "azure" && !paginated;
    let mut result = json!({"provider":provider,"status":if authoritative{"ok"}else{"partial"},"source":"live_api","authoritative":authoritative,"cached":false,"stale":false,"models":models});
    if provider == "azure" {
        result["detail"] =
            "Azure lists regional base models; routing still requires a deployment name".into();
    } else if paginated {
        result["detail"] =
            "The provider has more models; only its bounded first page is shown".into();
    }
    Ok(result)
}
