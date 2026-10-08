//! Official subscription runtime integration; no HTTP endpoint or dollar estimate.
use super::*;
use crate::subscription;
use sha2::{Digest, Sha256};
use std::sync::atomic::AtomicBool;

pub(super) const WARNING: &str = "Subscription dollar cost and unreported request or token counts are unknown. Persistent response caching and cross-call merging are disabled for pools containing official subscription runtimes because their account may change.";
pub(super) fn subscription_pool(entries: &[Value]) -> bool {
    copilot::pool_has_copilot(entries)
        || entries.iter().any(|entry| {
            entry
                .pointer("/litellm_params/weight")
                .and_then(Value::as_u64)
                .unwrap_or(1)
                > 0
                && entry
                    .pointer("/litellm_params/model")
                    .and_then(Value::as_str)
                    .is_some_and(|model| {
                        model.starts_with("claude-agent/") || model.starts_with("chatgpt/")
                    })
        })
}
pub(super) fn deployment(
    entry: &Value,
    env: &HashMap<String, String>,
    grouped: bool,
) -> Result<Deployment> {
    let params = &entry["litellm_params"];
    let id = nonempty(params.get("model"))
        .ok_or_else(|| Error::Config("Claude model is required".into()))?;
    let model = id
        .strip_prefix("claude-agent/")
        .filter(|model| {
            !model.is_empty() && model.len() <= 512 && !model.chars().any(char::is_control)
        })
        .ok_or_else(|| Error::Config("Claude model identifier is invalid".into()))?;
    for field in ["api_key", "api_base", "max_tokens"] {
        if params
            .get(field)
            .is_some_and(|value| !value.is_null() && value.as_str() != Some(""))
        {
            return Err(Error::Unsupported(format!(
                "Claude subscription runtime does not accept {field}; use official CLI authentication and model defaults"
            )));
        }
    }
    if entry
        .pointer("/model_info/max_tokens")
        .is_some_and(|value| !value.is_null())
    {
        return Err(Error::Unsupported(
            "Claude subscription runtime cannot enforce model_info.max_tokens".into(),
        ));
    }
    let config = subscription::claude::Config::from_env(env)?;
    // This digest isolates adaptive routing metrics. It is not an authenticated
    // account identity and must never enable persistent cache or single-flight.
    let mut hash = Sha256::new();
    let executable = config.executable().to_string_lossy();
    for value in [
        executable.as_ref(),
        env.get("CLAUDE_CONFIG_DIR")
            .map(String::as_str)
            .unwrap_or(""),
        env.get("HOME").map(String::as_str).unwrap_or(""),
    ] {
        hash.update((value.len() as u64).to_le_bytes());
        hash.update(value.as_bytes());
    }
    let endpoint = format!("claude-cli://{}", crate::hex(hash.finalize()));
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
        provider: "claude-agent".into(),
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
fn route_tokens(evidence: &subscription::claude::UsageEvidence) -> (Option<u64>, Option<u64>) {
    use subscription::claude::TokenCounts;
    fn totals<'a>(mut items: impl Iterator<Item = &'a TokenCounts>) -> Option<(u64, u64)> {
        items.try_fold((0u64, 0u64), |(input, output), row| {
            Some((
                input
                    .checked_add(row.input_tokens?)?
                    .checked_add(row.cache_read_tokens?)?
                    .checked_add(row.cache_write_tokens?)?,
                output.checked_add(row.output_tokens?)?,
            ))
        })
    }
    let known = if let Some(aggregate) = &evidence.aggregate {
        if aggregate.by_model.is_empty() {
            totals(std::iter::once(&aggregate.totals))
        } else {
            totals(aggregate.by_model.values())
        }
    } else if !evidence.calls.is_empty() {
        evidence
            .calls
            .iter()
            .try_fold((0u64, 0u64), |(input, output), call| {
                Some((
                    input
                        .checked_add(call.input_tokens?)?
                        .checked_add(call.cache_read_tokens?)?
                        .checked_add(call.cache_write_tokens?)?,
                    output.checked_add(call.output_tokens?)?,
                ))
            })
    } else {
        None
    };
    (
        known.and_then(|(input, output)| input.checked_add(output)),
        known.map(|(_, output)| output),
    )
}
fn failure(error: subscription::claude::Failure) -> Failure {
    use subscription::FailureKind as K;
    let kind = match error.kind {
        K::Authentication => FailureKind::Authentication,
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
    let config = subscription::claude::Config::from_env(env).map_err(|error| Failure {
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
        if encoded.len() > 10 * 1024 * 1024 {
            return Err(Failure::resource_limit("Claude image payload is oversized"));
        }
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .map_err(|_| Failure::resource_limit("Claude image payload is invalid"))?;
        size = size
            .checked_add(bytes.len())
            .ok_or_else(|| Failure::resource_limit("Claude image payload is oversized"))?;
        if size > 15 * 1024 * 1024 / 2 {
            return Err(Failure::resource_limit(
                "Claude image payload exceeds 7.5 MiB",
            ));
        }
        decoded.push((mime.as_str(), bytes));
    }
    let images: Vec<_> = decoded
        .iter()
        .map(|(mime, bytes)| (*mime, bytes.as_slice()))
        .collect();
    let started = std::time::Instant::now();
    let result = subscription::claude::complete(
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
    subscription_accounting::observe_claude(usage, entry, evidence).map_err(|error| Failure {
        kind: FailureKind::Blocked,
        error,
        retryable: false,
        fatal: true,
        document_fatal: true,
        retry_after: None,
    })?;
    if let Some(observation) = observation {
        if let Ok(value) = &result {
            let (total_tokens, output_tokens) = route_tokens(&value.usage);
            *observation = routing::Observation::Success {
                elapsed: started.elapsed(),
                total_tokens,
                output_tokens,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::subscription::{
        ObservedCall,
        claude::{AggregateUsage, TokenCounts, UsageEvidence},
    };
    #[test]
    fn adaptive_token_metrics_never_add_overlapping_totals_or_guess_missing_fields() {
        let complete = TokenCounts {
            input_tokens: Some(11),
            output_tokens: Some(7),
            cache_read_tokens: Some(3),
            cache_write_tokens: Some(2),
        };
        let mut evidence = UsageEvidence {
            calls: vec![ObservedCall {
                model: "claude-agent/model".into(),
                input_tokens: Some(11),
                output_tokens: Some(7),
                cache_read_tokens: Some(3),
                cache_write_tokens: Some(2),
            }],
            aggregate: Some(AggregateUsage {
                totals: complete.clone(),
                by_model: Default::default(),
            }),
        };
        assert_eq!(route_tokens(&evidence), (Some(23), Some(7)));
        evidence
            .aggregate
            .as_mut()
            .unwrap()
            .totals
            .cache_write_tokens = None;
        assert_eq!(route_tokens(&evidence), (None, None));
        evidence.aggregate = None;
        assert_eq!(route_tokens(&evidence), (Some(23), Some(7)));
        evidence.calls.clear();
        assert_eq!(route_tokens(&evidence), (None, None));
    }
    #[test]
    fn subscription_pool_checks_all_enabled_fallbacks_without_claiming_account_identity() {
        let rows = vec![
            json!({"litellm_params":{"model":"openai/gpt-4.1"}}),
            json!({"model_name":"fallback","litellm_params":{"model":"claude-agent/sonnet"}}),
        ];
        assert!(subscription_pool(&rows));
        let mut disabled = rows.clone();
        disabled[1]["litellm_params"]["weight"] = json!(0);
        assert!(!subscription_pool(&disabled));
        assert!(subscription_pool(&[
            json!({"litellm_params":{"model":"copilot/fixture"}})
        ]));
    }
}
