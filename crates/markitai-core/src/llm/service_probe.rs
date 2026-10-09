//! A probe uses the conversion protocol but not its retry or fallback router.
use super::{Deployment, Prompts, Protocol, deployments, diagnosis, payload, refusal_reason};
use crate::{Error, Result, provider_management};
use reqwest::{blocking::Client, redirect::Policy};
use serde_json::{Value, json};
use std::{collections::HashMap, io::Read, time::Duration};
const MAX_RESPONSE: u64 = 1024 * 1024;

pub(crate) fn probe(request: &Value, allow_environment: bool) -> Result<()> {
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
                return Err(Error::InvalidInput("Copilot probe uses official CLI authentication, not API credentials or a base URL".into()));
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
        .map_err(|failure| failure.error);
    }
    if model.starts_with("claude-agent/") {
        for name in ["api_key", "api_base"] {
            if request
                .get(name)
                .is_some_and(|value| !value.is_null() && value.as_str() != Some(""))
            {
                return Err(Error::InvalidInput("Claude subscription probe uses official CLI authentication, not API credentials or a base URL".into()));
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
        .map_err(|failure| failure.error);
    }
    if model.starts_with("chatgpt/") {
        for name in ["api_key", "api_base"] {
            if request
                .get(name)
                .is_some_and(|value| !value.is_null() && value.as_str() != Some(""))
            {
                return Err(Error::InvalidInput("ChatGPT subscription probe uses official CLI authentication, not API credentials or a base URL".into()));
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
        .map_err(|failure| failure.error);
    }
    let mut params = json!({"model":model,"max_tokens":16,"weight":1});
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
        return Err(Error::Config(
            "Connection test requires exactly one deployment".into(),
        ));
    }
    let entry = entries.remove(0);
    provider_management::checked_url(&entry.endpoint)?;
    perform(&entry)
}
fn perform(entry: &Deployment) -> Result<()> {
    let client = Client::builder()
        .redirect(Policy::none())
        .connect_timeout(Duration::from_secs(3))
        .timeout(Duration::from_secs(15))
        .build()
        .map_err(|_| failure("Cannot create model connection client"))?;
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
    // Failures are worded as conversion's are (diagnosis), without the
    // endpoint, the key or the probe's prompt.
    let response = call.json(&body).send().map_err(|error| {
        if error.is_timeout() {
            return failure("Model connection test timed out");
        }
        match diagnosis::cause(error) {
            Some(cause) => failure(&format!("Model connection request failed: {cause}")),
            None => failure("Model connection request failed"),
        }
    })?;
    let status = response.status().as_u16();
    if !(200..300).contains(&status) {
        let mut bytes = Vec::new();
        let _ = response.take(64 * 1024).read_to_end(&mut bytes);
        let body = String::from_utf8_lossy(&bytes).to_ascii_lowercase();
        return Err(failure(&match refusal_reason(status, &body) {
            Some(reason) => format!("Model connection returned HTTP {status}: {reason}"),
            None => format!(
                "Model connection returned HTTP {status}{}",
                diagnosis::refusal(status, &bytes, entry, &prompts)
            ),
        }));
    }
    if response
        .content_length()
        .is_some_and(|length| length > MAX_RESPONSE)
    {
        return Err(failure("Model connection response exceeds 1 MiB"));
    }
    let bytes = crate::platform::read_limited(response, MAX_RESPONSE)
        .map_err(|_| failure("Cannot read model connection response"))?
        .ok_or_else(|| failure("Model connection response exceeds 1 MiB"))?;
    let data: Value = serde_json::from_slice(&bytes)
        .map_err(|_| failure("Model connection response is not valid JSON"))?;
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
        return Err(failure("Model connection returned an invalid completion"));
    }
    Ok(())
}
fn failure(message: &str) -> Error {
    Error::Conversion(message.into())
}

#[cfg(test)]
mod tests {
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
                "message":"Incorrect API key provided: probe-key-must-not-leak."}})
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
            "Model connection returned HTTP 401 (invalid_request_error/invalid_api_key): Incorrect API key provided: [REDACTED]."
        );
    }
}
