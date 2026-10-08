use super::checked_url;
use crate::{Error, Result};
use reqwest::{blocking::Client, redirect::Policy};
use serde_json::{Value, json};
use std::{collections::HashSet, time::Duration};
const MAX_RESPONSE: u64 = 2 * 1024 * 1024;
const MAX_MODELS: usize = 1000;
fn failure(message: &str) -> Error {
    Error::Conversion(message.into())
}
pub(super) fn unavailable(provider: &str, detail: &str) -> Value {
    json!({"provider":provider,"status":"unavailable","source":"live_api","authoritative":false,"cached":false,"stale":false,"models":[],"detail":detail})
}
pub(super) fn load(provider: &str, base: &str, key: Option<&str>) -> Result<Value> {
    let mut url = checked_url(base)?;
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
    let client = Client::builder()
        .redirect(Policy::none())
        .connect_timeout(Duration::from_secs(3))
        .timeout(Duration::from_secs(10))
        .build()
        .map_err(|_| failure("Cannot create model discovery client"))?;
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
    let response = request
        .send()
        .map_err(|_| failure("Model discovery transport failed"))?;
    if !response.status().is_success() {
        return Err(failure(
            "Model discovery returned an unsuccessful HTTP status",
        ));
    }
    if response
        .content_length()
        .is_some_and(|length| length > MAX_RESPONSE)
    {
        return Err(failure("Model discovery response exceeds 2 MiB"));
    }
    let bytes = crate::platform::read_limited(response, MAX_RESPONSE)
        .map_err(|_| failure("Cannot read model discovery response"))?
        .ok_or_else(|| failure("Model discovery response exceeds 2 MiB"))?;
    let data: Value = serde_json::from_slice(&bytes)
        .map_err(|_| failure("Model discovery response is not JSON"))?;
    parse(provider, &data)
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
