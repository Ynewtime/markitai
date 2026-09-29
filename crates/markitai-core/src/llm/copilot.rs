//! Official subscription runtime integration; no HTTP endpoint or dollar estimate.
use super::*;
use crate::subscription;
use sha2::{Digest, Sha256};
use std::sync::atomic::AtomicBool;

pub(super) const WARNING: &str = "Copilot subscription dollar cost and unreported token counts are unknown. Persistent response caching and cross-call merging are disabled for pools containing Copilot because the official runtime account may change.";

pub(super) fn pool_has_copilot(entries: &[Value]) -> bool {
    entries.iter().any(|entry| {
        entry
            .pointer("/litellm_params/weight")
            .and_then(Value::as_u64)
            .unwrap_or(1)
            > 0
            && entry
                .pointer("/litellm_params/model")
                .and_then(Value::as_str)
                .is_some_and(|model| model.starts_with("copilot/"))
    })
}
pub(super) fn configured(cfg: &Value) -> Option<bool> {
    cfg.pointer("/llm/model_list")
        .and_then(Value::as_array)
        .filter(|entries| !entries.is_empty())
        .map(|entries| pool_has_copilot(entries))
}
pub(super) fn deployment(
    entry: &Value,
    env: &HashMap<String, String>,
    grouped: bool,
) -> Result<Deployment> {
    let params = &entry["litellm_params"];
    let id = nonempty(params.get("model"))
        .ok_or_else(|| Error::Config("Copilot model is required".into()))?;
    let model = id
        .strip_prefix("copilot/")
        .filter(|model| {
            !model.is_empty() && model.len() <= 512 && !model.chars().any(char::is_control)
        })
        .ok_or_else(|| Error::Config("Copilot model identifier is invalid".into()))?;
    for field in ["api_key", "api_base", "max_tokens"] {
        if params
            .get(field)
            .is_some_and(|value| !value.is_null() && value.as_str() != Some(""))
        {
            return Err(Error::Unsupported(format!(
                "Copilot subscription runtime does not accept {field}; use official CLI authentication and model defaults"
            )));
        }
    }
    if entry
        .pointer("/model_info/max_tokens")
        .is_some_and(|value| !value.is_null())
    {
        return Err(Error::Unsupported(
            "Copilot subscription runtime cannot enforce model_info.max_tokens".into(),
        ));
    }
    let config = subscription::CopilotConfig::from_env(env)?;
    // This digest isolates adaptive routing metrics. It is not an authenticated
    // account identity and must never enable persistent cache or single-flight.
    let mut hash = Sha256::new();
    let executable = config.executable().to_string_lossy();
    for value in [
        executable.as_ref(),
        env.get("COPILOT_HOME").map(String::as_str).unwrap_or(""),
        env.get("HOME").map(String::as_str).unwrap_or(""),
    ] {
        hash.update((value.len() as u64).to_le_bytes());
        hash.update(value.as_bytes());
    }
    let token = ["COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"]
        .iter()
        .find_map(|key| env.get(*key).filter(|value| !value.is_empty()));
    if let Some(token) = token {
        hash.update(token.len().to_le_bytes());
        hash.update(token.as_bytes());
    }
    let endpoint = format!("copilot-cli://{}", crate::hex(hash.finalize()));
    Ok(Deployment {
        id: id.into(),
        explicit_id: nonempty(entry.pointer("/model_info/id")).map(str::to_owned),
        group: if grouped {
            entry
                .get("model_name")
                .and_then(Value::as_str)
                .unwrap_or("default")
                .into()
        } else {
            "default".into()
        },
        model: model.into(),
        provider: "copilot".into(),
        weight: params.get("weight").and_then(Value::as_u64).unwrap_or(1),
        key: None,
        endpoint,
        protocol: Protocol::Chat,
        max_tokens: None,
        supports_vision: entry
            .pointer("/model_info/supports_vision")
            .and_then(Value::as_bool),
    })
}
fn observed(
    usage: &mut ConversionUsage,
    entry: &Deployment,
    evidence: &subscription::UsageEvidence,
) {
    for call in &evidence.calls {
        let mut data = json!({"model":format!("copilot/{}",call.model),"usage":{}});
        let fields = [
            ("prompt_tokens", call.input_tokens),
            ("completion_tokens", call.output_tokens),
        ];
        for (name, value) in fields {
            if let Some(value) = value {
                data["usage"][name] = json!(value);
            }
        }
        if let Some(value) = call.cache_read_tokens {
            data["usage"]["prompt_tokens_details"] = json!({"cached_tokens":value});
        }
        // The subscription identity has no catalog tariff; the runtime multiplier
        // is intentionally absent. Each distinct observed API call is unpriced.
        record_usage(usage, entry, &data);
    }
}
fn failure(error: subscription::Failure) -> Failure {
    use subscription::FailureKind as K;
    let kind = match error.kind {
        K::Refusal => FailureKind::Refusal,
        K::Truncated => FailureKind::Truncated,
        K::InvalidRequest | K::Unsupported => FailureKind::InvalidRequest,
        K::Permission | K::ResourceLimit | K::Cancelled => FailureKind::Blocked,
        _ => FailureKind::Transport,
    };
    let fatal = matches!(error.kind, K::Permission | K::ResourceLimit | K::Cancelled);
    Failure {
        kind,
        error: error.error,
        retryable: false,
        fatal,
        document_fatal: true,
        retry_after: None,
    }
}
pub(super) fn request(
    entry: &Deployment,
    prompts: &Prompts,
    env: &HashMap<String, String>,
    timeout: Duration,
    stop: Option<&AtomicBool>,
    usage: &mut ConversionUsage,
    observation: Option<&mut routing::Observation>,
) -> std::result::Result<String, Failure> {
    let config = subscription::CopilotConfig::from_env(env).map_err(|error| Failure {
        kind: FailureKind::InvalidRequest,
        error,
        retryable: false,
        fatal: false,
        document_fatal: true,
        retry_after: None,
    })?;
    let mut decoded = Vec::new();
    let mut size = 0usize;
    for (mime, encoded) in prompts.image.as_deref().unwrap_or(&[]) {
        // Avoid allocating an oversized decoded buffer before the adapter check.
        if encoded.len() > 90 * 1024 * 1024 {
            return Err(Failure::resource_limit(
                "Copilot image payload is oversized",
            ));
        }
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .map_err(|_| Failure::resource_limit("Copilot image payload is invalid"))?;
        size = size
            .checked_add(bytes.len())
            .ok_or_else(|| Failure::resource_limit("Copilot image payload is oversized"))?;
        if size > 64 * 1024 * 1024 {
            return Err(Failure::resource_limit(
                "Copilot image payload exceeds 64 MiB",
            ));
        }
        decoded.push((mime.as_str(), bytes));
    }
    let images: Vec<_> = decoded
        .iter()
        .map(|(mime, bytes)| (*mime, bytes.as_slice()))
        .collect();
    let started = std::time::Instant::now();
    let result = subscription::complete(
        &config,
        subscription::Request {
            model: &entry.id,
            system: &prompts.system,
            user: &prompts.user,
            images: &images,
            timeout,
            cancel: stop,
        },
    );
    let evidence = match &result {
        Ok(value) => &value.usage,
        Err(error) => &error.usage,
    };
    observed(usage, entry, evidence);
    if let Some(observation) = observation {
        if let Ok(value) = &result {
            let input = value
                .usage
                .calls
                .iter()
                .try_fold(0u64, |sum, call| sum.checked_add(call.input_tokens?));
            let output = value
                .usage
                .calls
                .iter()
                .try_fold(0u64, |sum, call| sum.checked_add(call.output_tokens?));
            let known = !value.usage.calls.is_empty();
            *observation = routing::Observation::Success {
                elapsed: started.elapsed(),
                total_tokens: if known {
                    input.zip(output).and_then(|(a, b)| a.checked_add(b))
                } else {
                    None
                },
                output_tokens: if known { output } else { None },
            };
        } else if matches!(
            result.as_ref().err().map(|error| error.kind),
            Some(subscription::FailureKind::Timeout)
        ) {
            *observation = routing::Observation::Timeout;
        }
    }
    result.map(|value| value.text).map_err(failure)
}
