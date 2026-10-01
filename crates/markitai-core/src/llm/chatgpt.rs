//! Subscription-only official Codex execution under the common LLM budget.
use super::*;
use crate::subscription;
use sha2::{Digest, Sha256};
use std::sync::atomic::AtomicBool;

pub(super) fn deployment(
    entry: &Value,
    env: &HashMap<String, String>,
    grouped: bool,
) -> Result<Deployment> {
    let params = &entry["litellm_params"];
    let id = nonempty(params.get("model"))
        .ok_or_else(|| Error::Config("ChatGPT model is required".into()))?;
    let model = id
        .strip_prefix("chatgpt/")
        .filter(|model| *model == subscription::chatgpt::MODEL)
        .ok_or_else(|| {
            Error::Unsupported("The installed Codex adapter has not validated this model".into())
        })?;
    for field in ["api_key", "api_base", "max_tokens"] {
        if params
            .get(field)
            .is_some_and(|v| !v.is_null() && v.as_str() != Some(""))
        {
            return Err(Error::Unsupported(format!(
                "ChatGPT subscription runtime does not accept {field}; use official Codex authentication and model defaults"
            )));
        }
    }
    if entry
        .pointer("/model_info/max_tokens")
        .is_some_and(|v| !v.is_null())
    {
        return Err(Error::Unsupported(
            "ChatGPT subscription runtime cannot enforce model_info.max_tokens".into(),
        ));
    }
    let config = subscription::chatgpt::Config::from_env(env)?;
    let mut hash = Sha256::new();
    let executable = config.executable().to_string_lossy();
    for value in [
        executable.as_ref(),
        env.get("CODEX_HOME").map(String::as_str).unwrap_or(""),
        env.get("HOME").map(String::as_str).unwrap_or(""),
    ] {
        hash.update((value.len() as u64).to_le_bytes());
        hash.update(value.as_bytes());
    }
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
        provider: "chatgpt".into(),
        weight: params.get("weight").and_then(Value::as_u64).unwrap_or(1),
        key: None,
        endpoint: format!("codex-cli://{}", crate::hex(hash.finalize())),
        protocol: Protocol::Chat,
        max_tokens: None,
        supports_vision: entry
            .pointer("/model_info/supports_vision")
            .and_then(Value::as_bool),
    })
}
fn observe(
    usage: &mut ConversionUsage,
    entry: &Deployment,
    evidence: &subscription::chatgpt::UsageEvidence,
) {
    let Some(total) = &evidence.aggregate else {
        return;
    };
    let mut delta = ConversionUsage {
        input_tokens: total.input_tokens,
        output_tokens: total.output_tokens,
        ..Default::default()
    };
    delta.by_model.insert(entry.id.clone(),json!({"requests":0,"input_tokens":total.input_tokens,"output_tokens":total.output_tokens,"cached_input_tokens":total.cached_input_tokens,"cache_creation_input_tokens":total.cache_creation_input_tokens,"cost_usd":0.0,"priced_requests":0,"unpriced_requests":0,"cost_status":"unknown","incomplete_request_observations":1}));
    DOCUMENT_ACCOUNTING.with(|slot| {
        if let Some(state) = slot.borrow().as_ref() {
            let mut state = state.lock().unwrap_or_else(|e| e.into_inner());
            state
                .dollars
                .observe(pricing::Quote::Unknown(pricing::UnknownPrice::Provider));
            merge_usage(&mut state.usage, &delta);
        }
    });
    merge_usage(usage, &delta);
}
fn failure(error: subscription::chatgpt::Failure) -> Failure {
    use subscription::FailureKind as K;
    let kind = match error.kind {
        K::Authentication => FailureKind::Authentication,
        K::Refusal => FailureKind::Refusal,
        K::Truncated => FailureKind::Truncated,
        K::InvalidRequest | K::Unsupported => FailureKind::InvalidRequest,
        K::Permission | K::ResourceLimit | K::Cancelled => FailureKind::Blocked,
        _ => FailureKind::Transport,
    };
    Failure {
        kind,
        error: error.error,
        retryable: false,
        fatal: matches!(error.kind, K::Permission | K::ResourceLimit | K::Cancelled),
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
    let config = subscription::chatgpt::Config::from_env(env).map_err(|error| Failure {
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
        if encoded.len() > 44 * 1024 * 1024 {
            return Err(Failure::resource_limit("Codex image payload is oversized"));
        }
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .map_err(|_| Failure::resource_limit("Codex image payload is invalid"))?;
        size = size
            .checked_add(bytes.len())
            .ok_or_else(|| Failure::resource_limit("Codex image payload is oversized"))?;
        if size > 32 * 1024 * 1024 {
            return Err(Failure::resource_limit(
                "Codex image payload exceeds 32 MiB",
            ));
        }
        decoded.push((mime.as_str(), bytes));
    }
    let images: Vec<_> = decoded
        .iter()
        .map(|(mime, bytes)| (*mime, bytes.as_slice()))
        .collect();
    let started = std::time::Instant::now();
    let result = subscription::chatgpt::complete(
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
    observe(
        usage,
        entry,
        match &result {
            Ok(value) => &value.usage,
            Err(error) => &error.usage,
        },
    );
    if let Some(observation) = observation {
        if let Ok(value) = &result {
            let known = value
                .usage
                .aggregate
                .as_ref()
                .filter(|t| t.input_tokens > 0 || t.output_tokens > 0);
            *observation = routing::Observation::Success {
                elapsed: started.elapsed(),
                total_tokens: known.and_then(|t| t.input_tokens.checked_add(t.output_tokens)),
                output_tokens: known.map(|t| t.output_tokens),
            };
        } else if matches!(
            result.as_ref().err().map(|e| e.kind),
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
    fn entry() -> Deployment {
        Deployment {
            id: "chatgpt/gpt-5.5".into(),
            explicit_id: None,
            group: "default".into(),
            model: "gpt-5.5".into(),
            provider: "chatgpt".into(),
            weight: 1,
            key: None,
            endpoint: "codex-cli://fixture".into(),
            protocol: Protocol::Chat,
            max_tokens: None,
            supports_vision: None,
        }
    }
    #[test]
    fn a_signed_out_runtime_is_an_authentication_refusal_and_policy_stays_fatal() {
        use subscription::FailureKind as K;
        let classify = |kind| {
            failure(subscription::chatgpt::Failure {
                kind,
                error: Error::Conversion("authored runtime failure".into()),
                usage: Default::default(),
            })
        };
        let refused = classify(K::Authentication);
        assert_eq!(refused.kind, FailureKind::Authentication);
        assert!(!refused.fatal && !refused.retryable && refused.document_fatal);
        let policy = classify(K::Permission);
        assert_eq!(policy.kind, FailureKind::Blocked);
        assert!(policy.fatal);
    }
    #[test]
    fn aggregate_totals_do_not_double_count_cached_input_or_invent_requests() {
        let mut usage = ConversionUsage::default();
        let evidence = subscription::chatgpt::UsageEvidence {
            aggregate: Some(subscription::chatgpt::TokenTotals {
                input_tokens: 11,
                output_tokens: 5,
                cached_input_tokens: 3,
                ..Default::default()
            }),
        };
        observe(&mut usage, &entry(), &evidence);
        assert_eq!(
            (usage.requests, usage.input_tokens, usage.output_tokens),
            (0, 11, 5)
        );
        assert!(!usage.cost_complete());
        assert_eq!(
            usage.by_model["chatgpt/gpt-5.5"]["incomplete_request_observations"],
            1
        );
        let before = usage.clone();
        observe(&mut usage, &entry(), &evidence);
        let delta = usage_difference(&usage, &before);
        assert_eq!(
            (delta.requests, delta.input_tokens, delta.output_tokens),
            (0, 11, 5)
        );
        observe(
            &mut usage,
            &entry(),
            &subscription::chatgpt::UsageEvidence::default(),
        );
        assert_eq!(usage.input_tokens, 22);
    }
    #[test]
    fn zero_totals_remain_unknown_and_whole_fallback_pool_bypasses_cache() {
        let mut usage = ConversionUsage::default();
        observe(
            &mut usage,
            &entry(),
            &subscription::chatgpt::UsageEvidence {
                aggregate: Some(Default::default()),
            },
        );
        assert!(!usage.cost_complete());
        assert_eq!(usage.requests, 0);
        let rows = vec![
            json!({"litellm_params":{"model":"openai/gpt-4.1"}}),
            json!({"model_name":"fallback","litellm_params":{"model":"chatgpt/gpt-5.5"}}),
        ];
        assert!(claude::subscription_pool(&rows));
        let mut disabled = rows;
        disabled[1]["litellm_params"]["weight"] = json!(0);
        assert!(!claude::subscription_pool(&disabled));
    }
}
