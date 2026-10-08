//! Bounded provider discovery and one-deployment connection checks.
//!
//! Environment-backed calls read the conversion environment snapshot (process
//! variables plus `.env` files, see `config::environment`); explicit calls read
//! none. They do not load configuration, prompt files or document caches.
//! Copilot delegates its auth store to the installed official runtime; this
//! module never parses that store.
mod cache;
mod discovery;
#[cfg(test)]
mod tests;

use crate::{Error, Result};
use serde_json::{Value, json};
use std::collections::HashMap;

const MAX_FIELD: usize = 8192;
const DETECTED: [(&str, &str, &str); 5] = [
    ("openai", "OPENAI_API_KEY", "openai/gpt-5.6-luna"),
    (
        "anthropic",
        "ANTHROPIC_API_KEY",
        "anthropic/claude-haiku-4-5",
    ),
    (
        "gemini",
        "GEMINI_API_KEY",
        "gemini/gemini-flash-lite-latest",
    ),
    ("deepseek", "DEEPSEEK_API_KEY", "deepseek/deepseek-v4-flash"),
    (
        "openrouter",
        "OPENROUTER_API_KEY",
        "openrouter/google/gemini-3.1-flash-lite",
    ),
];

/// Conventional labels, without credential or account information.
pub fn provider_label(provider: &str) -> String {
    match provider {
        "openai" => "OpenAI",
        "anthropic" => "Anthropic",
        "gemini" => "Gemini",
        "deepseek" => "DeepSeek",
        "openrouter" => "OpenRouter",
        "azure" => "Azure",
        "ollama" | "ollama_chat" => "Ollama",
        "groq" => "Groq",
        "mistral" => "Mistral AI",
        "xai" => "xAI",
        "together_ai" => "Together AI",
        "perplexity" => "Perplexity",
        "cerebras" => "Cerebras",
        "fireworks_ai" => "Fireworks AI",
        "deepinfra" => "DeepInfra",
        "nebius" => "Nebius",
        "moonshot" => "Moonshot AI",
        "sambanova" => "SambaNova",
        "zai" => "Z.ai",
        "nvidia_nim" => "NVIDIA NIM",
        "novita" => "Novita AI",
        "hosted_vllm" => "vLLM",
        "lm_studio" => "LM Studio",
        "custom" => "Custom",
        "claude-agent" => "Claude Agent",
        "copilot" => "Copilot",
        "chatgpt" => "ChatGPT",
        _ => "Unknown provider",
    }
    .into()
}

/// The providers first offered by the settings surfaces, in the reference's
/// order; `custom` is any other OpenAI-compatible endpoint.
const POPULAR: [&str; 8] = [
    "openai",
    "anthropic",
    "gemini",
    "ollama",
    "deepseek",
    "openrouter",
    "azure",
    "custom",
];
/// OpenAI-compatible prefixes whose documentation names no model list
/// endpoint (checked 2026-10-02): their model IDs are entered by hand.
const MANUAL_MODELS: [&str; 3] = ["perplexity", "zai", "fireworks_ai"];

/// One provider the settings surfaces can connect, with what its card shows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProviderEntry {
    /// The model prefix, or `custom` for an OpenAI-compatible endpoint.
    pub provider: &'static str,
    /// The documented inference endpoint; `None` when every server has its own.
    pub default_base: Option<&'static str>,
    /// Environment variables that replace the endpoint, in order.
    pub base_variables: &'static [&'static str],
    /// Environment variables read for the key, in order (empty for `custom`).
    pub key_variables: &'static [&'static str],
    /// A local server that works without a key.
    pub key_optional: bool,
    /// `discover` can list its models; otherwise model IDs are entered by hand.
    pub discovery: bool,
    /// One of the OpenAI-compatible prefixes added after the reference's list.
    pub compatible: bool,
}

/// Every connectable provider: the reference's popular list first, then the
/// further OpenAI-compatible prefixes in the order of the routing table.
pub fn catalog() -> Vec<ProviderEntry> {
    let listed: Vec<_> = crate::llm::providers::listings().collect();
    let entry = |name: &'static str, compatible: bool| {
        let listing = listed.iter().find(|listing| listing.prefix == name);
        ProviderEntry {
            provider: name,
            default_base: listing.and_then(|listing| listing.base),
            base_variables: listing.map_or(&[], |listing| listing.base_vars),
            key_variables: listing.map_or(&[], |listing| listing.key_vars),
            key_optional: listing.is_some_and(|listing| listing.key_optional),
            discovery: !MANUAL_MODELS.contains(&name),
            compatible,
        }
    };
    let mut entries: Vec<_> = POPULAR.iter().map(|name| entry(name, false)).collect();
    for listing in &listed {
        if listing.openai_compatible
            && !POPULAR.contains(&listing.prefix)
            && listing.prefix != "ollama_chat"
        {
            entries.push(entry(listing.prefix, true));
        }
    }
    entries
}

fn compatible_entry(provider: &str) -> Option<ProviderEntry> {
    catalog()
        .into_iter()
        .find(|entry| entry.compatible && entry.provider == provider)
}

/// Discovery bases; Gemini and Ollama inference use different endpoint paths.
/// The added OpenAI-compatible prefixes list `/models` beside their inference
/// endpoint.
pub fn provider_default_base(provider: &str) -> Option<&'static str> {
    match provider {
        "openai" => Some("https://api.openai.com/v1"),
        "anthropic" => Some("https://api.anthropic.com/v1"),
        "gemini" => Some("https://generativelanguage.googleapis.com/v1beta"),
        "deepseek" => Some("https://api.deepseek.com"),
        "openrouter" => Some("https://openrouter.ai/api/v1"),
        "ollama" => Some("http://127.0.0.1:11434"),
        _ => compatible_entry(provider).and_then(|entry| entry.default_base),
    }
}

/// Quick-add candidates from nonempty process environment keys, without I/O.
pub fn detected() -> Vec<Value> {
    DETECTED.iter().filter(|(_, key, _)| std::env::var(key).is_ok_and(|v| !v.trim().is_empty()))
        .map(|(provider, _, model)| json!({"provider":provider,"model":model,"label":provider_label(provider),"requires_api_key":false})).collect()
}

pub(super) fn field<'a>(request: &'a Value, key: &str) -> Result<Option<&'a str>> {
    match request.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) if value.len() <= MAX_FIELD && !value.contains('\0') => {
            Ok(Some(value))
        }
        _ => Err(Error::InvalidInput(
            "Provider request contains an invalid or oversized field".into(),
        )),
    }
}
fn validate(request: &Value, keys: &[&str]) -> Result<()> {
    let object = request
        .as_object()
        .ok_or_else(|| Error::InvalidInput("Provider request must be an object".into()))?;
    if object.keys().any(|key| !keys.contains(&key.as_str())) {
        return Err(Error::InvalidInput(
            "Provider request contains an unknown field".into(),
        ));
    }
    Ok(())
}
pub(super) fn resolve(
    value: Option<&str>,
    fallback: Option<&str>,
    env: &HashMap<String, String>,
) -> Result<Option<String>> {
    let resolved = crate::config::resolve_optional(value, fallback, env, true).map_err(|_| {
        Error::InvalidInput(
            "Provider credential or endpoint environment reference is unavailable".into(),
        )
    })?;
    if resolved
        .as_ref()
        .is_some_and(|value| value.len() > MAX_FIELD || value.chars().any(char::is_control))
    {
        return Err(Error::InvalidInput(
            "Provider credential or endpoint is invalid or oversized".into(),
        ));
    }
    Ok(resolved)
}
pub(super) fn checked_url(base: &str) -> Result<url::Url> {
    let value = url::Url::parse(base).map_err(|_| {
        Error::InvalidInput("Provider endpoint must be an absolute HTTP(S) URL".into())
    })?;
    if !matches!(value.scheme(), "http" | "https")
        || value.host_str().is_none()
        || !value.username().is_empty()
        || value.password().is_some()
        || value.fragment().is_some()
    {
        return Err(Error::InvalidInput(
            "Provider endpoint must use HTTP(S) without user information or a fragment".into(),
        ));
    }
    Ok(value)
}

fn discovery_base_variable(provider: &str, env: &HashMap<String, String>) -> Option<&'static str> {
    catalog()
        .into_iter()
        .find(|entry| entry.provider == provider)
        .and_then(|entry| {
            entry
                .base_variables
                .iter()
                .copied()
                .find(|name| env.get(*name).is_some_and(|value| !value.trim().is_empty()))
        })
}

/// The first of the provider's key variables holding a value in `env`, for
/// reporting that a connection without a stored key is served by the
/// environment (process variables and dotenv files).
pub fn environment_key_variable(
    provider: &str,
    env: &HashMap<String, String>,
) -> Option<&'static str> {
    catalog()
        .into_iter()
        .find(|entry| entry.provider == provider)
        .and_then(|entry| {
            entry
                .key_variables
                .iter()
                .copied()
                .find(|name| env.get(*name).is_some_and(|value| !value.trim().is_empty()))
        })
}

/// Discover model identifiers using a credential-isolated in-memory cache.
/// The input contains provider, optional api_key/api_base and optional refresh.
pub fn discover(request: &Value) -> Result<Value> {
    discover_with_environment(request, true)
}

/// Discover with literal request credentials only; no process key or endpoint fallback.
/// Configuration-backed callers should use `discover` instead.
pub fn discover_explicit(request: &Value) -> Result<Value> {
    discover_with_environment(request, false)
}

fn discover_with_environment(request: &Value, allow_environment: bool) -> Result<Value> {
    validate(request, &["provider", "api_key", "api_base", "refresh"])?;
    let provider = field(request, "provider")?
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| Error::InvalidInput("Provider is required".into()))?
        .trim()
        .to_ascii_lowercase();
    let compatible = compatible_entry(&provider);
    if compatible.is_none()
        && !matches!(
            provider.as_str(),
            "openai"
                | "anthropic"
                | "gemini"
                | "deepseek"
                | "openrouter"
                | "azure"
                | "ollama"
                | "custom"
                | "claude-agent"
                | "copilot"
                | "chatgpt"
        )
    {
        return Err(Error::InvalidInput("Unknown provider".into()));
    }
    let refresh = match request.get("refresh") {
        None => false,
        Some(Value::Bool(value)) => *value,
        _ => return Err(Error::InvalidInput("refresh must be a boolean".into())),
    };
    // The same snapshot conversion reads: process variables, then `.env` files.
    let env: HashMap<String, String> = if allow_environment {
        crate::config::environment()
    } else {
        HashMap::new()
    };
    if provider == "copilot" {
        if field(request, "api_key")?.is_some_and(|value| !value.is_empty())
            || field(request, "api_base")?.is_some_and(|value| !value.is_empty())
        {
            return Err(Error::InvalidInput("Copilot discovery uses official CLI authentication, not API credentials or a base URL".into()));
        }
        // A stored login can change within one HOME. No discovery cache keyed by
        // a filesystem path can establish account identity, so query each time.
        let result = crate::subscription::CopilotConfig::from_env(&env).and_then(|config| {
            crate::subscription::models(&config, std::time::Duration::from_secs(15))
                .map_err(|failure| failure.error)
        });
        return Ok(match result {
            Ok(models) => {
                json!({"provider":"copilot","status":"ok","source":"official_cli","authoritative":true,"cached":false,"stale":false,"models":models,"detail":"Models reported by the authenticated official Copilot runtime"})
            }
            Err(_) => {
                json!({"provider":"copilot","status":"unavailable","source":"official_cli","authoritative":false,"cached":false,"stale":false,"models":[],"detail":"Official Copilot runtime or authentication is unavailable"})
            }
        });
    }
    if provider == "claude-agent" {
        if field(request, "api_key")?.is_some_and(|value| !value.is_empty())
            || field(request, "api_base")?.is_some_and(|value| !value.is_empty())
        {
            return Err(Error::InvalidInput("Claude discovery uses official CLI authentication, not API credentials or a base URL".into()));
        }
        // A stored subscription can change accounts independently of this process.
        let result = crate::subscription::claude::Config::from_env(&env).and_then(|config| {
            crate::subscription::claude::models(&config, std::time::Duration::from_secs(15))
                .map_err(|failure| failure.error)
        });
        return Ok(match result {
            Ok(models) => {
                json!({"provider":"claude-agent","status":"ok","source":"official_cli","authoritative":true,"cached":false,"stale":false,"models":models,"detail":"Models reported by the authenticated official Claude runtime"})
            }
            Err(_) => {
                json!({"provider":"claude-agent","status":"unavailable","source":"official_cli","authoritative":false,"cached":false,"stale":false,"models":[],"detail":"Official Claude runtime or subscription authentication is unavailable"})
            }
        });
    }
    if provider == "chatgpt" {
        if field(request, "api_key")?.is_some_and(|value| !value.is_empty())
            || field(request, "api_base")?.is_some_and(|value| !value.is_empty())
        {
            return Err(Error::InvalidInput("Codex discovery uses official CLI authentication, not API credentials or a base URL".into()));
        }
        // A stored subscription can change accounts independently of this process.
        let result = crate::subscription::chatgpt::Config::from_env(&env).and_then(|config| {
            crate::subscription::chatgpt::models(&config, std::time::Duration::from_secs(15))
                .map_err(|failure| failure.error)
        });
        return Ok(match result {
            Ok(models) => {
                json!({"provider":"chatgpt","status":"ok","source":"adapter_allowlist","authoritative":false,"cached":false,"stale":false,"models":models,"detail":"Only the pinned gpt-5.5 adapter capability is listed; account entitlement is not discovered"})
            }
            Err(_) => {
                json!({"provider":"chatgpt","status":"unavailable","source":"official_cli","authoritative":false,"cached":false,"stale":false,"models":[],"detail":"Official Codex runtime or subscription authentication is unavailable"})
            }
        });
    }
    if let Some(entry) = compatible.as_ref().filter(|entry| !entry.discovery) {
        // Validate the request as for any other provider, then say why no
        // list follows: the client offers manual model entry instead.
        resolve(field(request, "api_key")?, None, &env)?;
        if let Some(base) = resolve(field(request, "api_base")?, None, &env)? {
            checked_url(&base)?;
        }
        return Ok(
            json!({"provider":entry.provider,"status":"unavailable","source":"manual","authoritative":false,"cached":false,"stale":false,"models":[],"detail":"This provider does not publish a model list; enter model IDs manually"}),
        );
    }
    let variable = match provider.as_str() {
        "openai" => Some("OPENAI_API_KEY"),
        "anthropic" => Some("ANTHROPIC_API_KEY"),
        "gemini" => Some("GEMINI_API_KEY"),
        "deepseek" => Some("DEEPSEEK_API_KEY"),
        "openrouter" => Some("OPENROUTER_API_KEY"),
        "azure" => Some("AZURE_API_KEY"),
        "ollama" => Some("OLLAMA_API_KEY"),
        // The first of the provider's key variables that holds a value.
        _ => compatible.as_ref().and_then(|entry| {
            entry
                .key_variables
                .iter()
                .find(|name| {
                    env.get(**name)
                        .is_some_and(|value| !value.trim().is_empty())
                })
                .or(entry.key_variables.first())
                .copied()
        }),
    };
    let key = resolve(field(request, "api_key")?, variable, &env)?;
    // Built-in and additional providers share inference's endpoint variables.
    let base_variable = discovery_base_variable(&provider, &env);
    let base = resolve(field(request, "api_base")?, base_variable, &env)?
        .or_else(|| provider_default_base(&provider).map(str::to_owned));
    if let Some(base) = &base {
        checked_url(base)?;
    }
    let Some(base) = base.filter(|value| !value.trim().is_empty()) else {
        return Ok(discovery::unavailable(
            &provider,
            "An API endpoint is required",
        ));
    };
    cache::global().discover(&provider, &base, key.as_deref(), refresh, || {
        discovery::load(&provider, &base, key.as_deref())
    })
}

/// Make one short request to exactly the requested deployment, without retry,
/// fallback, prompt loading, document conversion or persistent cache access.
pub fn probe(request: &Value) -> Result<Value> {
    probe_with_environment(request, true)
}

/// Probe literal request credentials without inheriting process keys or endpoints.
pub fn probe_explicit(request: &Value) -> Result<Value> {
    probe_with_environment(request, false)
}

fn probe_with_environment(request: &Value, allow_environment: bool) -> Result<Value> {
    validate(request, &["model", "api_key", "api_base"])?;
    let model = field(request, "model")?
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| Error::InvalidInput("Model is required".into()))?;
    field(request, "api_key")?;
    field(request, "api_base")?;
    if model.chars().any(char::is_control) || model.len() > 1024 {
        return Err(Error::InvalidInput(
            "Model contains invalid characters or exceeds 1024 bytes".into(),
        ));
    }
    match crate::llm::service_probe(request, allow_environment) {
        Ok(()) => Ok(
            json!({"ok":true,"detail":format!("{} responded", model.chars().take(280).collect::<String>())}),
        ),
        Err(error) => {
            // This helper's errors are deliberately fixed text or HTTP status,
            // never reqwest URLs, response bodies or user-supplied credentials.
            let detail: String = match error {
                Error::Unsupported(_) => {
                    "This model provider is not supported by the native runtime".into()
                }
                Error::Config(_) | Error::NoModelConfigured | Error::InvalidInput(_) => {
                    "Model credentials or endpoint configuration are invalid or unavailable".into()
                }
                Error::Conversion(message) => message.chars().take(300).collect(),
                _ => "Model connection test failed".into(),
            };
            Ok(json!({"ok":false,"detail":detail}))
        }
    }
}
