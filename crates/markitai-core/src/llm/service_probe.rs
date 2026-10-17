//! A probe uses the conversion protocol but not its retry or fallback router.
use super::{
    Deployment, MODEL_UNAVAILABLE, Prompts, Protocol, REGION_UNAVAILABLE, deployments, diagnosis,
    payload, providers, refusal_reason,
};
use crate::{Error, provider_management};
use reqwest::{blocking::Client, redirect::Policy};
use serde_json::{Value, json};
use std::{collections::HashMap, io::Read, time::Duration};
const MAX_RESPONSE: u64 = 1024 * 1024;

/// Why a connection test failed, for clients that explain it in their own
/// words; the failure's error keeps the English detail.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Reason {
    /// The deployment has no key to send, and its provider needs one.
    NoKey,
    /// HTTP 401 or 403.
    KeyRejected,
    /// HTTP 404, or an error that names the model as unknown or unavailable.
    ModelNotFound,
    /// HTTP 429, or an exhausted quota or balance.
    RateLimited,
    /// No HTTP response: name resolution, connection, TLS, or a redirect,
    /// which a probe never follows.
    Network,
    Timeout,
    ProviderError,
}
impl Reason {
    pub(crate) const ALL: [Reason; 7] = [
        Reason::NoKey,
        Reason::KeyRejected,
        Reason::ModelNotFound,
        Reason::RateLimited,
        Reason::Network,
        Reason::Timeout,
        Reason::ProviderError,
    ];
    pub(crate) fn name(self) -> &'static str {
        match self {
            Reason::NoKey => "no_key",
            Reason::KeyRejected => "key_rejected",
            Reason::ModelNotFound => "model_not_found",
            Reason::RateLimited => "rate_limited",
            Reason::Network => "network",
            Reason::Timeout => "timeout",
            Reason::ProviderError => "provider_error",
        }
    }
}

#[derive(Debug)]
pub(crate) struct Failure {
    pub(crate) reason: Reason,
    pub(crate) error: Error,
    /// For [`Reason::NoKey`], the environment variable the key would come
    /// from, when one is named.
    pub(crate) key_variable: Option<String>,
}
impl From<Error> for Failure {
    fn from(error: Error) -> Self {
        Self {
            reason: Reason::ProviderError,
            error,
            key_variable: None,
        }
    }
}

pub(crate) fn probe(request: &Value, allow_environment: bool) -> Result<(), Failure> {
    // The same snapshot conversion reads: process variables, then `.env` files.
    let env: HashMap<String, String> = if allow_environment {
        crate::config::environment()
    } else {
        HashMap::new()
    };
    let model = request["model"]
        .as_str()
        .ok_or_else(|| Error::InvalidInput("Model is required".into()))?;
    if model.starts_with("copilot/") {
        for name in ["api_key", "api_base"] {
            if request
                .get(name)
                .is_some_and(|value| !value.is_null() && value.as_str() != Some(""))
            {
                return Err(Error::InvalidInput("Copilot probe uses official CLI authentication, not API credentials or a base URL".into()).into());
            }
        }
        let config = crate::subscription::CopilotConfig::from_env(&env)?;
        return crate::subscription::complete(
            &config,
            crate::subscription::Request {
                model,
                system: "Reply briefly without tools.",
                user: "Reply with exactly OK.",
                images: &[],
                timeout: Duration::from_secs(15),
                cancel: None,
            },
        )
        .map(|_| ())
        .map_err(|failure| subscription(failure.kind, failure.error));
    }
    if model.starts_with("claude-agent/") {
        for name in ["api_key", "api_base"] {
            if request
                .get(name)
                .is_some_and(|value| !value.is_null() && value.as_str() != Some(""))
            {
                return Err(Error::InvalidInput("Claude subscription probe uses official CLI authentication, not API credentials or a base URL".into()).into());
            }
        }
        let config = crate::subscription::claude::Config::from_env(&env)?;
        return crate::subscription::claude::complete(
            &config,
            crate::subscription::Request {
                model,
                system: "Reply briefly without tools.",
                user: "Reply with exactly OK.",
                images: &[],
                timeout: Duration::from_secs(15),
                cancel: None,
            },
        )
        .map(|_| ())
        .map_err(|failure| subscription(failure.kind, failure.error));
    }
    if model.starts_with("chatgpt/") {
        for name in ["api_key", "api_base"] {
            if request
                .get(name)
                .is_some_and(|value| !value.is_null() && value.as_str() != Some(""))
            {
                return Err(Error::InvalidInput("ChatGPT subscription probe uses official CLI authentication, not API credentials or a base URL".into()).into());
            }
        }
        let config = crate::subscription::chatgpt::Config::from_env(&env)?;
        return crate::subscription::chatgpt::complete(
            &config,
            crate::subscription::Request {
                model,
                system: "Reply briefly without tools.",
                user: "Reply with exactly OK.",
                images: &[],
                timeout: Duration::from_secs(15),
                cancel: None,
            },
        )
        .map(|_| ())
        .map_err(|failure| subscription(failure.kind, failure.error));
    }
    if let Some(variable) = missing_key(model, request, &env, allow_environment) {
        return Err(Failure {
            reason: Reason::NoKey,
            error: Error::Config("Model API key is not available".into()),
            key_variable: variable,
        });
    }
    // An explicit probe reads no environment at all: no variables (`env` is
    // empty) and no credential the server's own login holds (gcloud).
    let mut params = json!({"model":model,"max_tokens":16,"weight":1,"use_environment_credentials":allow_environment});
    for name in ["api_key", "api_base"] {
        if let Some(value) = request.get(name) {
            params[name] = value.clone();
        }
    }
    // The native Azure protocol needs an explicit version, including probes.
    if model.starts_with("azure/") {
        params["api_version"] = env
            .get("AZURE_API_VERSION")
            .map_or("2024-10-21", String::as_str)
            .into();
    }
    let cfg = json!({"llm":{"model_list":[{"model_name":"default","litellm_params":params}]}});
    let mut entries = deployments(&cfg, &env)?;
    if entries.len() != 1 {
        return Err(Error::Config("Connection test requires exactly one deployment".into()).into());
    }
    let entry = entries.remove(0);
    provider_management::checked_url(&entry.endpoint)?;
    perform(&entry).map_err(|mut failure| {
        // Refused without a key: name the variable that would have held it.
        if failure.reason == Reason::NoKey && nonempty(request, "api_key").is_none() {
            failure.key_variable = needed_variable(model, allow_environment);
        }
        failure
    })
}

/// A deployment that has no key while its provider needs one, found before
/// any request: `Some` with the variable the key would come from, if one is
/// named. An unset or empty `env:` reference always counts. A deployment
/// without a stored key counts unless its provider takes none (local
/// servers; subscriptions and clouds are not in the runtime's table) or it
/// is aimed at an address of its own, which may well check no key: it is
/// tried, and a refusal of the keyless request says so then. The keyless
/// placeholder is a stored key.
fn missing_key(
    model: &str,
    request: &Value,
    env: &HashMap<String, String>,
    allow_environment: bool,
) -> Option<Option<String>> {
    let set = |name: &str| env.get(name).is_some_and(|value| !value.trim().is_empty());
    if let Some(key) = nonempty(request, "api_key") {
        return key
            .strip_prefix("env:")
            .filter(|name| !set(name))
            .map(|name| variable_name(name).then(|| name.to_owned()));
    }
    let prefix = model.split_once('/').map_or("openai", |(prefix, _)| prefix);
    let known = providers::find(prefix).filter(|known| !known.key_optional)?;
    let addressed = nonempty(request, "api_base").is_some()
        || allow_environment && known.base_vars.iter().any(|name| set(name));
    if addressed || allow_environment && known.key_vars.iter().any(|name| set(name)) {
        return None;
    }
    Some(needed_variable(model, allow_environment))
}

/// The variable a deployment without a stored key reads its key from: none
/// for an explicit probe, which reads no environment.
fn needed_variable(model: &str, allow_environment: bool) -> Option<String> {
    let prefix = model.split_once('/').map_or("openai", |(prefix, _)| prefix);
    allow_environment
        .then(|| providers::find(prefix))
        .flatten()
        .and_then(|known| known.key_vars.first())
        .map(|name| (*name).to_owned())
}

/// Whether an `env:` reference names a variable, not a pasted key, so it can
/// be shown.
fn variable_name(name: &str) -> bool {
    (1..=128).contains(&name.len())
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
}

fn nonempty<'a>(request: &'a Value, name: &str) -> Option<&'a str> {
    request[name].as_str().filter(|value| !value.is_empty())
}

/// A subscription runtime's failure: it has no key of this kind, so its
/// sign-in problems stay provider errors with their own detail.
fn subscription(kind: crate::subscription::FailureKind, error: Error) -> Failure {
    use crate::subscription::FailureKind;
    let reason = match kind {
        FailureKind::Timeout => Reason::Timeout,
        FailureKind::Transport => Reason::Network,
        FailureKind::RateLimit => Reason::RateLimited,
        _ => Reason::ProviderError,
    };
    Failure {
        reason,
        error,
        key_variable: None,
    }
}

fn perform(entry: &Deployment) -> Result<(), Failure> {
    let client = crate::proxy::http_api(
        Client::builder()
            .redirect(Policy::none())
            .connect_timeout(Duration::from_secs(3))
            .timeout(Duration::from_secs(15)),
    )
    .map_err(|_| failure(Reason::ProviderError, "Proxy settings are malformed"))?
    .build()
    .map_err(|_| {
        failure(
            Reason::ProviderError,
            "Cannot create model connection client",
        )
    })?;
    let prompts = Prompts {
        system: String::new(),
        user: "Reply with exactly OK.".into(),
        image: None,
        cache_scope: String::new(),
    };
    let mut body = payload(entry, &prompts);
    // The shared payload supplies the provider's token field and content shape.
    // Probes have no conversion system prompt at all.
    if entry.protocol == Protocol::Anthropic {
        body.as_object_mut()
            .expect("payload object")
            .remove("system");
    } else if let Some(messages) = body["messages"].as_array_mut() {
        messages.retain(|message| message["role"] != "system");
    }
    let mut call = client
        .post(&entry.endpoint)
        .header("accept", "application/json");
    if let Some(cloud) = entry.cloud.as_deref() {
        call = super::cloud::prepare(cloud, &client, call, entry, body).map_err(|_| {
            failure(
                Reason::ProviderError,
                "Model connection credentials could not be used",
            )
        })?;
    } else {
        if let Some(key) = &entry.key {
            call = match entry.protocol {
                Protocol::Anthropic => call.header("x-api-key", key),
                Protocol::Azure => call.header("api-key", key),
                Protocol::Chat => call.bearer_auth(key),
            };
        }
        if entry.protocol == Protocol::Anthropic {
            call = call.header("anthropic-version", "2023-06-01");
        }
        call = call.json(&body);
    }
    // A failure names its cause as conversion's does (diagnosis), but the
    // provider's own wording is never shown here: only its error type or
    // code, without the endpoint, the key or the probe's prompt.
    let response = call.send().map_err(|error| {
        if error.is_timeout() {
            return failure(Reason::Timeout, "Model connection test timed out");
        }
        match diagnosis::cause(error, crate::proxy::describe(&entry.endpoint).as_deref()) {
            Some(cause) => failure(
                Reason::Network,
                &format!("Model connection request failed: {cause}"),
            ),
            None => failure(Reason::Network, "Model connection request failed"),
        }
    })?;
    let status = response.status().as_u16();
    if !(200..300).contains(&status) {
        let mut bytes = Vec::new();
        let _ = response.take(64 * 1024).read_to_end(&mut bytes);
        let body = String::from_utf8_lossy(&bytes).to_ascii_lowercase();
        let refused = refusal_reason(status, &body);
        let reason = match (refused, status) {
            (Some(MODEL_UNAVAILABLE), _) | (None, 404) => Reason::ModelNotFound,
            (Some(REGION_UNAVAILABLE), _) => Reason::ProviderError,
            // The only other recognized refusal: the account's quota or billing.
            (Some(_), _) | (None, 429) => Reason::RateLimited,
            (None, 401 | 403)
                if entry.cloud.is_none()
                    && !provider_management::real_key(entry.key.as_deref()) =>
            {
                Reason::NoKey
            }
            (None, 401 | 403) => Reason::KeyRejected,
            (None, 300..=399) => Reason::Network,
            _ => Reason::ProviderError,
        };
        return Err(failure(
            reason,
            &match refused {
                Some(refused) => format!("Model connection returned HTTP {status}: {refused}"),
                None => format!(
                    "Model connection returned HTTP {status}{}",
                    diagnosis::refusal_code(status, &bytes, entry)
                ),
            },
        ));
    }
    if response
        .content_length()
        .is_some_and(|length| length > MAX_RESPONSE)
    {
        return Err(failure(
            Reason::ProviderError,
            "Model connection response exceeds 1 MiB",
        ));
    }
    let bytes = crate::platform::read_limited(response, MAX_RESPONSE)
        .map_err(|_| {
            failure(
                Reason::ProviderError,
                "Cannot read model connection response",
            )
        })?
        .ok_or_else(|| {
            failure(
                Reason::ProviderError,
                "Model connection response exceeds 1 MiB",
            )
        })?;
    let data: Value = serde_json::from_slice(&bytes).map_err(|_| {
        failure(
            Reason::ProviderError,
            "Model connection response is not valid JSON",
        )
    })?;
    let data = match entry.cloud.as_deref() {
        Some(cloud) => super::cloud::response(cloud, data, entry),
        None => data,
    };
    let complete = if entry.protocol == Protocol::Anthropic {
        data.get("content").and_then(Value::as_array).is_some()
    } else {
        data.get("choices")
            .and_then(Value::as_array)
            .is_some_and(|choices| {
                !choices.is_empty()
                    && choices
                        .iter()
                        .any(|choice| choice.get("message").is_some_and(Value::is_object))
            })
    };
    if !complete || data.get("error").is_some_and(|value| !value.is_null()) {
        return Err(failure(
            Reason::ProviderError,
            "Model connection returned an invalid completion",
        ));
    }
    Ok(())
}
/// Why a request that was not timed out received no HTTP response, as a
/// connection test names it (model discovery says the same).
pub(crate) fn transport_cause(error: reqwest::Error, proxy: Option<&str>) -> Option<String> {
    diagnosis::cause(error, proxy)
}

fn failure(reason: Reason, message: &str) -> Failure {
    Failure {
        reason,
        error: Error::Conversion(message.into()),
        key_variable: None,
    }
}

#[cfg(test)]
mod tests {
    use super::missing_key;
    use crate::provider_management::probe_explicit;
    use serde_json::json;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    fn detail(base: &str, key: &str) -> String {
        let result =
            probe_explicit(&json!({"model":"openai/probe-test","api_key":key,"api_base":base}))
                .unwrap();
        assert_eq!(result["ok"], false, "{result}");
        result["detail"].as_str().unwrap().to_owned()
    }

    #[test]
    fn a_failed_connection_test_says_why_without_the_key_or_endpoint() {
        let refused = TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap();
        assert_eq!(
            detail(&format!("http://{refused}/v1"), "probe-key-must-not-leak"),
            "Model connection request failed: connection refused"
        );

        // A provider that quotes the key back in its refusal.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = Vec::new();
            let mut buffer = [0; 4096];
            while !request.ends_with(b"}") {
                match stream.read(&mut buffer) {
                    Ok(0) | Err(_) => break,
                    Ok(read) => request.extend_from_slice(&buffer[..read]),
                }
            }
            let body = json!({"error":{"type":"invalid_request_error","code":"invalid_api_key",
                "message":"Incorrect API key provided: probe-key-must-not-leak. See http://private-host/keys"}})
            .to_string();
            let _ = write!(
                stream,
                "HTTP/1.1 401 Unauthorized\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
        });
        let said = detail(&format!("http://{address}/v1"), "probe-key-must-not-leak");
        server.join().unwrap();
        assert_eq!(
            said,
            "Model connection returned HTTP 401 (invalid_request_error/invalid_api_key)"
        );
    }

    #[test]
    fn a_missing_key_is_found_from_the_runtime_provider_table() {
        let env = |pairs: &[(&str, &str)]| -> std::collections::HashMap<String, String> {
            pairs
                .iter()
                .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
                .collect()
        };
        let named = |name: &str| Some(Some(name.to_owned()));
        let base = json!({"api_base":"http://127.0.0.1:9/v1"});
        for (model, request, environment, allow, expected) in [
            (
                "deepseek/x",
                json!({}),
                env(&[]),
                true,
                named("DEEPSEEK_API_KEY"),
            ),
            (
                "deepseek/x",
                json!({"api_key":""}),
                env(&[("DEEPSEEK_API_KEY", " ")]),
                true,
                named("DEEPSEEK_API_KEY"),
            ),
            (
                "deepseek/x",
                json!({}),
                env(&[("DEEPSEEK_API_KEY", "k")]),
                true,
                None,
            ),
            // The server's key variable is not read by an explicit test.
            (
                "deepseek/x",
                json!({}),
                env(&[("DEEPSEEK_API_KEY", "k")]),
                false,
                Some(None),
            ),
            // An address of its own, named or from the environment, is tried.
            ("deepseek/x", base.clone(), env(&[]), true, None),
            (
                "deepseek/x",
                json!({}),
                env(&[("DEEPSEEK_API_BASE", "http://127.0.0.1:9")]),
                true,
                None,
            ),
            ("gpt-x", json!({}), env(&[]), true, named("OPENAI_API_KEY")),
            ("azure/x", json!({}), env(&[]), true, named("AZURE_API_KEY")),
            (
                "openai/x",
                json!({"api_key":"env:SYNTHETIC_KEY"}),
                env(&[]),
                true,
                named("SYNTHETIC_KEY"),
            ),
            (
                "openai/x",
                json!({"api_key":"env:SYNTHETIC_KEY"}),
                env(&[("SYNTHETIC_KEY", "")]),
                true,
                named("SYNTHETIC_KEY"),
            ),
            (
                "openai/x",
                json!({"api_key":"env:SYNTHETIC_KEY"}),
                env(&[("SYNTHETIC_KEY", "k")]),
                true,
                None,
            ),
            // A reference that holds no variable name is not repeated.
            (
                "openai/x",
                json!({"api_key":"env:sk-pasted-1234"}),
                env(&[]),
                true,
                Some(None),
            ),
            (
                "openai/x",
                json!({"api_key":"literal"}),
                env(&[]),
                true,
                None,
            ),
            (
                "deepseek/x",
                json!({"api_key":"markitai-no-key"}),
                env(&[]),
                true,
                None,
            ),
            ("ollama/x", json!({}), env(&[]), true, None),
            ("ollama_chat/x", json!({}), env(&[]), false, None),
            ("lm_studio/x", json!({}), env(&[]), true, None),
            ("hosted_vllm/x", json!({}), env(&[]), true, None),
            ("bedrock/x", json!({}), env(&[]), true, None),
            ("vertex_ai/x", json!({}), env(&[]), true, None),
            ("copilot/x", json!({}), env(&[]), true, None),
            ("claude-agent/x", json!({}), env(&[]), true, None),
            ("chatgpt/x", json!({}), env(&[]), true, None),
        ] {
            assert_eq!(
                missing_key(model, &request, &environment, allow),
                expected,
                "{model} {request} {allow}"
            );
        }
    }
}
