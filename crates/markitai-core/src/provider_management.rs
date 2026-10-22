//! Bounded provider discovery and one-deployment connection checks.
//!
//! Environment-backed calls read the conversion environment snapshot (process
//! variables plus `.env` files, see `config::environment`); explicit calls read
//! none. They do not load configuration, prompt files or document caches.
//! Copilot delegates its auth store to the installed official runtime; this
//! module never parses that store.
mod cache;
mod discovery;
mod runtime;
#[cfg(test)]
mod tests;

use crate::{
    Error, Result,
    llm::{ProbeFailure, ProbeReason},
};
use serde_json::{Value, json};
use std::collections::HashMap;

const MAX_FIELD: usize = 8192;

/// The key a connection or deployment saved through the settings API stores
/// when it has no key but names an address (the same word Python markitai
/// 1.2.1 stores, so a configuration file means the same to both). It is no
/// credential: it is sent as written, which a server that checks no key
/// accepts, and an entry holding it never uses the environment's
/// credentials, so the server's own key variable cannot reach an address a
/// settings caller chose. Model discovery sends no key for it.
pub const NO_KEY: &str = "markitai-no-key";

/// Whether `key` is a credential: a nonempty value other than [`NO_KEY`].
pub fn real_key(key: Option<&str>) -> bool {
    key.is_some_and(|key| !key.is_empty() && key != NO_KEY)
}

/// Conventional labels, without credential or account information.
pub fn provider_label(provider: &str) -> String {
    lookup(provider)
        .map_or("Unknown provider", |entry| entry.label)
        .into()
}

/// The reference's popular providers; `custom` is any other OpenAI-compatible
/// endpoint. Their cards are `common` in the settings API, like the
/// providers that are not OpenAI-compatible HTTP APIs.
const REFERENCE: [&str; 8] = [
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

/// The groups of the provider picker, in its order: the providers most
/// people start with, then subscription runtimes, model makers' own APIs,
/// cloud platforms, inference hosts, gateways and local servers.
pub const PROVIDER_GROUPS: [&str; 7] = [
    "common",
    "subscriptions",
    "makers",
    "clouds",
    "inference",
    "gateways",
    "local",
];

/// One provider the settings surfaces can connect, with what its card shows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProviderEntry {
    /// The model prefix, or `custom` for an OpenAI-compatible endpoint.
    pub provider: &'static str,
    /// The provider's name as the settings surfaces show it.
    pub label: &'static str,
    /// One of [`PROVIDER_GROUPS`].
    pub group: &'static str,
    /// The brand mark the workbench draws (a key of its vendored marks), or
    /// `None` for a monogram.
    pub icon: Option<&'static str>,
    /// Further model prefixes routed the same way, then other names the
    /// provider is searched by.
    pub aliases: &'static [&'static str],
    /// The documented inference endpoint; `None` when every server has its own.
    pub default_base: Option<&'static str>,
    /// Environment variables that replace the endpoint, in order.
    pub base_variables: &'static [&'static str],
    /// Environment variables read for the key, in order (empty when the
    /// provider takes none, as for `custom`).
    pub key_variables: &'static [&'static str],
    /// No key needs to be entered: a local server, a subscription runtime, or
    /// a cloud that also reads its own credentials.
    pub key_optional: bool,
    /// `discover` can list its models; otherwise model IDs are entered by hand.
    pub discovery: bool,
    /// An OpenAI-compatible prefix beyond the reference's popular list.
    pub compatible: bool,
}

/// How the settings surfaces present one provider.
struct Shown {
    provider: &'static str,
    label: &'static str,
    group: &'static str,
    icon: Option<&'static str>,
    aliases: &'static [&'static str],
}

const fn shown(
    provider: &'static str,
    label: &'static str,
    group: &'static str,
    icon: Option<&'static str>,
    aliases: &'static [&'static str],
) -> Shown {
    Shown {
        provider,
        label,
        group,
        icon,
        aliases,
    }
}

/// Every connectable provider in picker order: the common eight as the
/// workbench shows them, then each further group by label. Icons are keys of
/// the lobehub marks the workbench vendors (`src/components/brand-marks.ts`).
#[rustfmt::skip]
static SHOWN: [Shown; 95] = [
    shown("openai", "OpenAI", "common", Some("openai"), &[]),
    shown("anthropic", "Anthropic", "common", Some("anthropic"), &["claude"]),
    shown("gemini", "Gemini", "common", Some("gemini"), &["google", "google ai studio"]),
    shown("deepseek", "DeepSeek", "common", Some("deepseek"), &[]),
    shown("openrouter", "OpenRouter", "common", Some("openrouter"), &[]),
    shown("azure", "Azure OpenAI", "common", Some("azure"), &["microsoft"]),
    shown("ollama", "Ollama", "common", Some("ollama"), &["ollama_chat"]),
    shown("mistral", "Mistral AI", "common", Some("mistral"), &[]),
    shown("chatgpt", "ChatGPT", "subscriptions", Some("openai"), &["codex", "openai"]),
    shown("claude-agent", "Claude", "subscriptions", Some("claude"), &["claude code", "anthropic"]),
    shown("copilot", "GitHub Copilot", "subscriptions", Some("githubcopilot"), &["github"]),
    shown("ai21", "AI21 Labs", "makers", Some("ai21"), &["ai21_chat", "jamba"]),
    shown("amazon_nova", "Amazon Nova", "makers", Some("nova"), &["amazon", "aws"]),
    shown("codestral", "Codestral", "makers", Some("mistral"), &["mistral"]),
    shown("cognition", "Cognition", "makers", None, &["swe"]),
    shown("cohere", "Cohere", "makers", Some("cohere"), &["cohere_chat", "command"]),
    shown("compactifai", "CompactifAI", "makers", None, &["multiverse computing"]),
    shown("inception", "Inception", "makers", Some("inception"), &["mercury"]),
    shown("meta", "Meta Llama API", "makers", Some("meta"), &["llama"]),
    shown("minimax", "MiniMax", "makers", Some("minimax"), &[]),
    shown("moonshot", "Moonshot AI", "makers", Some("moonshot"), &["kimi"]),
    shown("morph", "Morph", "makers", Some("morph"), &[]),
    shown("perplexity", "Perplexity", "makers", Some("perplexity"), &["sonar"]),
    shown("sarvam", "Sarvam AI", "makers", None, &[]),
    shown("xai", "xAI", "makers", Some("xai"), &["grok"]),
    shown("xiaomi_mimo", "Xiaomi MiMo", "makers", None, &["xiaomi"]),
    shown("zai", "Z.ai", "makers", Some("zai"), &["zhipu", "glm", "bigmodel"]),
    shown("dashscope", "Alibaba Cloud Model Studio", "clouds", Some("bailian"), &["bailian", "aliyun", "qwen"]),
    shown("bedrock", "Amazon Bedrock", "clouds", Some("bedrock"), &["aws"]),
    shown("azure_ai", "Azure AI Foundry", "clouds", Some("azureai"), &["microsoft", "foundry"]),
    shown("bedrock_mantle", "Bedrock Mantle", "clouds", Some("bedrock"), &["aws", "amazon bedrock"]),
    shown("cloudflare", "Cloudflare Workers AI", "clouds", Some("workersai"), &["workers ai"]),
    shown("databricks", "Databricks", "clouds", None, &[]),
    shown("datarobot", "DataRobot", "clouds", None, &[]),
    shown("gradient_ai", "DigitalOcean Gradient", "clouds", Some("digitalocean"), &["digitalocean"]),
    shown("heroku", "Heroku", "clouds", None, &[]),
    shown("ovhcloud", "OVHcloud", "clouds", None, &["ovh"]),
    shown("qwen_ai_platform", "Qwen AI Platform", "clouds", Some("qwen"), &["alibaba", "dashscope"]),
    shown("qwencloud", "Qwen Cloud", "clouds", Some("qwen"), &["alibaba", "dashscope"]),
    shown("scaleway", "Scaleway", "clouds", None, &[]),
    shown("tencent", "Tencent Cloud", "clouds", Some("tencentcloud"), &["hunyuan", "tokenhub"]),
    shown("vertex_ai", "Vertex AI", "clouds", Some("vertexai"), &["vertex_ai_beta", "google cloud", "gcp"]),
    shown("volcengine", "Volcano Engine", "clouds", Some("volcengine"), &["ark", "doubao", "byteplus", "bytedance"]),
    shown("abliteration", "Abliteration", "inference", None, &[]),
    shown("baseten", "Baseten", "inference", Some("baseten"), &[]),
    shown("cerebras", "Cerebras", "inference", Some("cerebras"), &[]),
    shown("chutes", "Chutes", "inference", Some("chutes"), &[]),
    shown("crusoe", "Crusoe", "inference", Some("crusoe"), &[]),
    shown("darkbloom", "Darkbloom", "inference", None, &[]),
    shown("deepinfra", "DeepInfra", "inference", Some("deepinfra"), &[]),
    shown("empiriolabs", "Empirio Labs", "inference", None, &[]),
    shown("featherless_ai", "Featherless", "inference", Some("featherless"), &[]),
    shown("fireworks_ai", "Fireworks AI", "inference", Some("fireworks"), &[]),
    shown("friendliai", "FriendliAI", "inference", Some("friendli"), &[]),
    shown("galadriel", "Galadriel", "inference", None, &[]),
    shown("gmi", "GMI Cloud", "inference", Some("gmicloud"), &[]),
    shown("groq", "Groq", "inference", Some("groq"), &[]),
    shown("libertai", "LibertAI", "inference", None, &[]),
    shown("modelscope", "ModelScope", "inference", Some("modelscope"), &[]),
    shown("nebius", "Nebius", "inference", Some("nebius"), &["token factory"]),
    shown("neosantara", "Neosantara", "inference", None, &[]),
    shown("novita", "Novita AI", "inference", Some("novita"), &[]),
    shown("nscale", "Nscale", "inference", None, &[]),
    shown("nvidia_nim", "NVIDIA NIM", "inference", Some("nvidia"), &[]),
    shown("parasail", "Parasail", "inference", Some("parasail"), &[]),
    shown("pinstripes", "Pinstripes", "inference", None, &[]),
    shown("publicai", "PublicAI", "inference", None, &[]),
    shown("sambanova", "SambaNova", "inference", Some("sambanova"), &[]),
    shown("scx-ai", "SCX.ai", "inference", None, &[]),
    shown("synthetic", "Synthetic", "inference", None, &[]),
    shown("tensormesh", "Tensormesh", "inference", None, &[]),
    shown("together_ai", "Together AI", "inference", Some("together"), &[]),
    shown("veniceai", "Venice AI", "inference", Some("venice"), &[]),
    shown("wandb", "W&B Inference", "inference", Some("wandb"), &["weights & biases", "coreweave"]),
    shown("aiml", "AI/ML API", "gateways", None, &["aimlapi"]),
    shown("aihubmix", "AiHubMix", "gateways", Some("aihubmix"), &[]),
    shown("apertis", "Apertis", "gateways", Some("apertis"), &["stima"]),
    shown("assemblyai", "AssemblyAI", "gateways", Some("assemblyai"), &[]),
    shown("cometapi", "CometAPI", "gateways", Some("cometapi"), &[]),
    shown("edenai", "Eden AI", "gateways", None, &[]),
    shown("helicone", "Helicone", "gateways", None, &[]),
    shown("huggingface", "Hugging Face", "gateways", Some("huggingface"), &["hf"]),
    shown("litellm_proxy", "LiteLLM Proxy", "gateways", None, &["litellm"]),
    shown("nadir", "Nadir", "gateways", None, &[]),
    shown("nano-gpt", "NanoGPT", "gateways", None, &[]),
    shown("poe", "Poe", "gateways", Some("poe"), &[]),
    shown("vercel_ai_gateway", "Vercel AI Gateway", "gateways", Some("vercel"), &[]),
    shown("docker_model_runner", "Docker Model Runner", "local", None, &["docker"]),
    shown("lemonade", "Lemonade", "local", None, &[]),
    shown("llamafile", "llamafile", "local", None, &[]),
    shown("lm_studio", "LM Studio", "local", Some("lmstudio"), &[]),
    shown("custom", "OpenAI-compatible endpoint", "local", None, &["openai_like", "custom", "self-hosted", "llama.cpp"]),
    shown("oobabooga", "text-generation-webui", "local", None, &["oobabooga", "textgen"]),
    shown("hosted_vllm", "vLLM", "local", Some("vllm"), &[]),
    shown("xinference", "Xinference", "local", Some("xinference"), &[]),
];

static CATALOG: std::sync::LazyLock<Vec<ProviderEntry>> = std::sync::LazyLock::new(|| {
    let listed: Vec<_> = crate::llm::providers::listings().collect();
    SHOWN
        .iter()
        .map(|shown| {
            let name = shown.provider;
            let listing = listed.iter().find(|listing| listing.prefix == name);
            // Bedrock also signs with AWS credentials and Vertex AI uses
            // Google's; neither publishes a model list this build reads.
            let (base_variables, key_variables): (&[&str], &[&str]) = match name {
                "bedrock" => (
                    &["AWS_BEDROCK_RUNTIME_ENDPOINT"],
                    &["AWS_BEARER_TOKEN_BEDROCK"],
                ),
                _ => listing.map_or((&[], &[]), |listing| (listing.base_vars, listing.key_vars)),
            };
            ProviderEntry {
                provider: name,
                label: shown.label,
                group: shown.group,
                icon: shown.icon,
                aliases: shown.aliases,
                default_base: listing.and_then(|listing| listing.base),
                base_variables,
                key_variables,
                key_optional: listing.is_some_and(|listing| listing.key_optional)
                    || matches!(shown.group, "subscriptions")
                    || matches!(name, "bedrock" | "vertex_ai"),
                discovery: !MANUAL_MODELS.contains(&name)
                    && !matches!(name, "bedrock" | "vertex_ai"),
                compatible: listing.is_some_and(|listing| listing.openai_compatible)
                    && !REFERENCE.contains(&name),
            }
        })
        .collect()
});

/// Every connectable provider, in the picker's order (see [`PROVIDER_GROUPS`]).
pub fn catalog() -> Vec<ProviderEntry> {
    CATALOG.clone()
}

/// The catalog entry of a provider, or of the provider a further prefix
/// (`ollama_chat/`, `cohere_chat/`, …) is routed with.
pub fn provider_entry(provider: &str) -> Option<ProviderEntry> {
    lookup(provider).cloned()
}

fn lookup(provider: &str) -> Option<&'static ProviderEntry> {
    CATALOG
        .iter()
        .find(|entry| entry.provider == provider)
        .or_else(|| {
            CATALOG.iter().find(|entry| {
                entry.aliases.contains(&provider) && crate::llm::providers::supported(provider)
            })
        })
}

/// Discovery bases; Gemini and Ollama inference use different endpoint paths.
/// The other OpenAI-compatible prefixes list `/models` beside their inference
/// endpoint.
pub fn provider_default_base(provider: &str) -> Option<&'static str> {
    match provider {
        "openai" => Some("https://api.openai.com/v1"),
        "anthropic" => Some("https://api.anthropic.com/v1"),
        "gemini" => Some("https://generativelanguage.googleapis.com/v1beta"),
        "deepseek" => Some("https://api.deepseek.com"),
        "openrouter" => Some("https://openrouter.ai/api/v1"),
        "ollama" => Some("http://127.0.0.1:11434"),
        _ => CATALOG
            .iter()
            .find(|entry| entry.provider == provider)
            .and_then(|entry| entry.default_base),
    }
}

/// Quick-add candidates: the default models whose API keys the conversion
/// environment sets, in routing priority.
pub fn detected() -> Vec<Value> {
    crate::llm::keyed_defaults(&crate::config::environment())
        .map(|(provider, model)| json!({"provider":provider,"model":model,"label":provider_label(provider),"requires_api_key":false}))
        .collect()
}

/// The deployments conversions use while `llm.model_list` is empty: `MODEL`,
/// else [`detected`]'s models.
pub fn automatic_deployments() -> Vec<Value> {
    crate::llm::automatic_entries(&crate::config::environment())
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
    CATALOG
        .iter()
        .find(|entry| entry.provider == provider)
        .and_then(|entry| {
            entry
                .base_variables
                .iter()
                .copied()
                .find(|name| env.get(*name).is_some_and(|value| !value.trim().is_empty()))
        })
}

/// The provider's key variable a request reads in `env`, for reporting that
/// a connection without a stored key is served by the environment (process
/// variables and dotenv files).
pub fn environment_key_variable(
    provider: &str,
    env: &HashMap<String, String>,
) -> Option<&'static str> {
    CATALOG
        .iter()
        .find(|entry| entry.provider == provider)
        .and_then(|entry| crate::llm::providers::held_key_variable(entry.key_variables, env))
}

/// The variable among `variables` a request reads its key from, given which
/// ones hold a value: the first, except that `GOOGLE_API_KEY` comes first.
pub fn held_key_variable(
    variables: &[&'static str],
    held: impl Fn(&str) -> bool,
) -> Option<&'static str> {
    crate::llm::providers::held_key_variable_by(variables, held)
}

/// Discover model identifiers using a credential-isolated in-memory cache.
/// The input contains provider, optional api_key/api_base and optional refresh.
pub fn discover(request: &Value) -> Result<Value> {
    discover_with_environment(request, true, false)
}

/// Discover with literal request credentials only; no process key or endpoint fallback.
/// Configuration-backed callers should use `discover` instead.
pub fn discover_explicit(request: &Value) -> Result<Value> {
    discover_with_environment(request, false, false)
}

/// [`discover_explicit`] for a service's user: its request reaches public
/// addresses only (see `public_net`) and never shares the result cache.
pub fn discover_public(request: &Value) -> Result<Value> {
    discover_with_environment(request, false, true)
}

fn discover_with_environment(
    request: &Value,
    allow_environment: bool,
    public_only: bool,
) -> Result<Value> {
    validate(request, &["provider", "api_key", "api_base", "refresh"])?;
    let provider = field(request, "provider")?
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| Error::InvalidInput("Provider is required".into()))?
        .trim()
        .to_ascii_lowercase();
    let Some(entry) = CATALOG.iter().find(|entry| entry.provider == provider) else {
        return Err(Error::InvalidInput("Unknown provider".into()));
    };
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
        return Ok(runtime::discover(&provider, &env));
    }
    if provider == "claude-agent" {
        if field(request, "api_key")?.is_some_and(|value| !value.is_empty())
            || field(request, "api_base")?.is_some_and(|value| !value.is_empty())
        {
            return Err(Error::InvalidInput("Claude discovery uses official CLI authentication, not API credentials or a base URL".into()));
        }
        return Ok(runtime::discover(&provider, &env));
    }
    if provider == "chatgpt" {
        if field(request, "api_key")?.is_some_and(|value| !value.is_empty())
            || field(request, "api_base")?.is_some_and(|value| !value.is_empty())
        {
            return Err(Error::InvalidInput("Codex discovery uses official CLI authentication, not API credentials or a base URL".into()));
        }
        return Ok(runtime::discover(&provider, &env));
    }
    if !entry.discovery {
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
    // A key the provider needs and nothing supplies: say so before sending
    // anything, as a connection test does.
    if let Some(variable) = missing_key(
        entry,
        field(request, "api_key")?,
        field(request, "api_base")?,
        &env,
        allow_environment,
    ) {
        let detail = match &variable {
            Some(variable) => format!("{variable} is not set on the server, so nothing was sent"),
            None => "Model discovery needs an API key, and none is set, so nothing was sent".into(),
        };
        return Ok(discovery::failed(
            &provider,
            ProbeReason::NoKey,
            &detail,
            variable.as_deref(),
        ));
    }
    // The provider's key variable that holds a value, else its first one.
    let variable = crate::llm::providers::held_key_variable(entry.key_variables, &env)
        .or(entry.key_variables.first().copied());
    // The keyless placeholder is no key to send, and never the environment's.
    let key = resolve(field(request, "api_key")?, variable, &env)?.filter(|key| key != NO_KEY);
    // A gateway token is a bearer token, as for inference.
    let bearer = provider == "anthropic"
        && match field(request, "api_key")?.filter(|value| !value.is_empty()) {
            Some(value) => value == "env:ANTHROPIC_AUTH_TOKEN",
            None => variable == Some("ANTHROPIC_AUTH_TOKEN"),
        };
    // Built-in and additional providers share inference's endpoint variables.
    let base_variable = discovery_base_variable(&provider, &env);
    let base = resolve(field(request, "api_base")?, base_variable, &env)?
        .or_else(|| provider_default_base(&provider).map(str::to_owned));
    if let Some(base) = &base {
        checked_url(base)?;
    }
    let Some(base) = base.filter(|value| !value.trim().is_empty()) else {
        return Ok(discovery::failed(
            &provider,
            ProbeReason::ProviderError,
            "An API endpoint is required",
            None,
        ));
    };
    // A keyless request refused for want of a key names the variable the
    // server would read it from (none when it reads no environment).
    let keyless_variable = allow_environment
        .then(|| entry.key_variables.first().copied())
        .flatten();
    if public_only {
        // Never answered from, nor stored for, another caller's list.
        return Ok(discovery::load(
            &provider,
            &base,
            key.as_deref(),
            bearer,
            keyless_variable,
            true,
        )
        .unwrap_or_else(|failed| failed));
    }
    cache::global().discover(&provider, &base, key.as_deref(), refresh, || {
        discovery::load(
            &provider,
            &base,
            key.as_deref(),
            bearer,
            keyless_variable,
            false,
        )
    })
}

/// A discovery that has no key while its provider needs one, found before
/// any request, by the rule a connection test applies: `Some` with the
/// variable the key would come from, if one is named. An unset or empty
/// `env:` reference always counts. Without a key, it counts unless the
/// provider takes none (local servers, subscriptions, clouds with their own
/// credentials, OpenAI-compatible endpoints) or the request is aimed at an
/// address of its own, which is tried. The keyless placeholder is a key.
fn missing_key(
    entry: &ProviderEntry,
    key: Option<&str>,
    base: Option<&str>,
    env: &HashMap<String, String>,
    allow_environment: bool,
) -> Option<Option<String>> {
    let set = |name: &str| env.get(name).is_some_and(|value| !value.trim().is_empty());
    if let Some(key) = key.filter(|key| !key.is_empty()) {
        return key
            .strip_prefix("env:")
            .filter(|name| !set(name))
            .map(|name| {
                let named = (1..=128).contains(&name.len())
                    && name
                        .bytes()
                        .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_');
                named.then(|| name.to_owned())
            });
    }
    if entry.key_optional || entry.key_variables.is_empty() {
        return None;
    }
    let addressed = entry.default_base.is_some()
        && (base.is_some_and(|base| !base.is_empty())
            || allow_environment && entry.base_variables.iter().any(|name| set(name)));
    if addressed || allow_environment && entry.key_variables.iter().any(|name| set(name)) {
        return None;
    }
    Some(
        allow_environment
            .then(|| entry.key_variables.first().map(|name| (*name).to_owned()))
            .flatten(),
    )
}

/// Make one short request to exactly the requested deployment, without retry,
/// fallback, prompt loading, document conversion or persistent cache access.
pub fn probe(request: &Value) -> Result<Value> {
    probe_with_environment(request, true, false)
}

/// Probe literal request credentials without inheriting process keys or endpoints.
pub fn probe_explicit(request: &Value) -> Result<Value> {
    probe_with_environment(request, false, false)
}

/// [`probe_explicit`] for a service's user: its request reaches public
/// addresses only (see `public_net`).
pub fn probe_public(request: &Value) -> Result<Value> {
    probe_with_environment(request, false, true)
}

fn probe_with_environment(
    request: &Value,
    allow_environment: bool,
    public_only: bool,
) -> Result<Value> {
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
    match crate::llm::service_probe(request, allow_environment, public_only) {
        Ok(()) => Ok(
            json!({"ok":true,"detail":format!("{} responded", model.chars().take(280).collect::<String>())}),
        ),
        Err(ProbeFailure {
            reason,
            error,
            key_variable,
        }) => {
            // This helper's errors are deliberately fixed text, an HTTP status
            // with the provider's error type or code, or a transport cause;
            // never reqwest URLs, response bodies or user-supplied credentials.
            // A missing key names only its variable.
            let detail: String = match (reason, &key_variable, error) {
                (ProbeReason::NoKey, Some(variable), _) => {
                    format!("{variable} is not set on the server")
                }
                (ProbeReason::NoKey, None, _) => {
                    "This model needs an API key, and none is set".into()
                }
                (_, _, Error::Unsupported(_)) => {
                    "This model provider is not supported by the native runtime".into()
                }
                (
                    _,
                    _,
                    Error::Config(_)
                    | Error::NoModelConfigured
                    | Error::MissingApiKey(_)
                    | Error::InvalidInput(_),
                ) => {
                    "Model credentials or endpoint configuration are invalid or unavailable".into()
                }
                (_, _, Error::Conversion(message)) => message.chars().take(300).collect(),
                _ => "Model connection test failed".into(),
            };
            let mut result = json!({"ok":false,"detail":detail,"reason":reason.name()});
            if let Some(variable) = key_variable {
                result["key_variable"] = variable.into();
            }
            Ok(result)
        }
    }
}

/// Every `reason` a failed connection test reports, in the order the
/// settings API documents them.
pub fn probe_reasons() -> impl Iterator<Item = &'static str> {
    ProbeReason::ALL.into_iter().map(ProbeReason::name)
}

/// The further `reason`s a subscription runtime's discovery reports instead
/// of `provider_error` (see [`runtime`]).
pub const RUNTIME_REASONS: [&str; 4] = [
    "runtime_missing",
    "runtime_unsupported",
    "not_signed_in",
    "runtime_failed",
];
