use super::{ApiError, ApiResult, identity, invalid, missing};
pub(super) use markitai_core::provider_management::{NO_KEY, real_key};
use serde_json::{Map, Value, json};
use std::collections::HashMap;

/// The refusal of a change that would leave a keyed connection keyless at
/// another host; the same words as Python markitai 1.2.1's.
pub(super) const KEYLESS_MOVE: &str = "a connection with a key cannot become one without a key at another address; add a new connection for that server instead";

/// Model prefixes whose requests sign in with the server's own cloud login
/// whatever key an entry carries, with the product and credential the
/// refusal names: Vertex AI with the Google login (Rust and Python's
/// LiteLLM), watsonx and GigaChat with their token variables (Python's
/// LiteLLM; Rust does not route them, and refuses them the same way so a
/// configuration file means the same to both). With an address that login
/// would go there; Bedrock is not here, as it sends the keyless placeholder as
/// its bearer key and never signs with the AWS chain once it has one.
const AMBIENT_LOGIN: [(&str, &str, &str); 5] = [
    ("vertex_ai", "Vertex AI", "Google credentials"),
    ("vertex_ai_beta", "Vertex AI", "Google credentials"),
    ("watsonx", "watsonx", "WATSONX_TOKEN"),
    ("watsonx_text", "watsonx", "WATSONX_TOKEN"),
    ("gigachat", "GigaChat", "GIGACHAT_ACCESS_TOKEN"),
];

/// Refuse an entry the settings API writes or tests that would carry the
/// server's own cloud login to an address: such a model may only use its
/// cloud's own endpoint here (owner decision, contract v1.9). The detail is
/// Python markitai 1.2.1's, which the workbench translates by its wording.
pub(super) fn refuse_ambient_login_address(model: &str, base: Option<&str>) -> ApiResult<()> {
    if base.is_none_or(|base| base.trim().is_empty()) {
        return Ok(());
    }
    let Some((prefix, _)) = model.split_once('/') else {
        return Ok(());
    };
    let prefix = prefix.trim().to_ascii_lowercase();
    let Some((_, label, credential)) = AMBIENT_LOGIN.iter().find(|(name, ..)| *name == prefix)
    else {
        return Ok(());
    };
    Err(invalid(&format!(
        "api_base: {label} models send the server's own {credential} to any address, so here they can only use {label}'s own endpoint; set another address in the config file"
    )))
}

/// The key to store for an entry the settings API writes: `key`, or
/// [`NO_KEY`] when it has none while `base` names an address and the entry
/// may not use the environment's credentials, so that no backend reading the
/// configuration (Python markitai's LiteLLM included) sends the server's own
/// key variable there. Subscription runtimes take neither key nor address.
pub(super) fn keyless(key: Value, base: &Value, provider: &str, allow_environment: bool) -> Value {
    if allow_environment
        || local(provider)
        || key.as_str().is_some_and(|key| !key.is_empty())
        || base.as_str().is_none_or(str::is_empty)
    {
        return key;
    }
    json!(NO_KEY)
}

/// Scheme, host and port the requests of an entry go to: its address, else
/// the provider's documented one; None when unknown (an `env:` reference).
fn endpoint_host(base: Option<&str>, provider: &str) -> Option<(String, String, u16)> {
    let base = base
        .filter(|base| !base.is_empty())
        .or_else(|| markitai_core::provider_management::provider_default_base(provider))?;
    let url = url::Url::parse(base).ok()?;
    Some((
        url.scheme().to_owned(),
        url.host_str()?.to_ascii_lowercase(),
        url.port_or_known_default()?,
    ))
}

/// Refuse one change that turns a connection or deployment with a key into
/// one without a key aimed at another host it names: its requests would go
/// to a server the owner never paired a key with. Clearing the key in place,
/// or together with the address (back to the provider's own endpoint), is
/// allowed.
pub(super) fn refuse_keyless_move(old: &Value, next: &Value, provider: &str) -> ApiResult<()> {
    if !real_key(old["api_key"].as_str()) || real_key(next["api_key"].as_str()) {
        return Ok(());
    }
    let Some(base) = next["api_base"].as_str().filter(|base| !base.is_empty()) else {
        return Ok(());
    };
    let host = endpoint_host(Some(base), provider);
    if host.is_some() && host == endpoint_host(old["api_base"].as_str(), provider) {
        return Ok(());
    }
    Err(invalid(KEYLESS_MOVE))
}

pub(super) fn object<'a>(body: &'a Value, allowed: &[&str]) -> ApiResult<&'a Map<String, Value>> {
    let map = body
        .as_object()
        .ok_or_else(|| invalid("request must be an object"))?;
    if map.keys().any(|key| !allowed.contains(&key.as_str())) {
        return Err(invalid("unknown request field"));
    }
    Ok(map)
}
pub(super) fn string(
    body: &Value,
    key: &str,
    required: bool,
    trim: bool,
) -> ApiResult<Option<String>> {
    match body.get(key) {
        None | Some(Value::Null) if !required => Ok(None),
        Some(Value::String(value)) => {
            if value.contains('…') {
                return Err(invalid("masked values cannot be saved"));
            }
            let value = if trim { value.trim() } else { value.as_str() };
            if required && value.is_empty() {
                return Err(invalid("required string cannot be blank"));
            }
            Ok(Some(value.to_owned()))
        }
        _ => Err(invalid("invalid or missing string field")),
    }
}

/// HTTP clients supply literal values; only the configuration file may refer to env.
pub(super) fn client_field(body: &Value, field: &str) -> ApiResult<Option<String>> {
    let value = string(body, field, false, false)?;
    if value
        .as_deref()
        .is_some_and(|v| v.trim_start().starts_with("env:"))
    {
        return Err(invalid(
            "environment references must be configured on the server",
        ));
    }
    // A masked value sent back would replace the secret with its own mask.
    if value.as_deref().is_some_and(|v| v.contains(MASK)) {
        return Err(invalid(&format!(
            "{field} is masked; send the whole new value, or leave {field} out to keep the saved one"
        )));
    }
    Ok(value)
}

/// `body` without its `api_base` when that is the masked form of the stored
/// address, unchanged: saving a form with the address left as shown keeps
/// the stored one.
pub(super) fn unchanged_base(body: &Value, stored: Option<&str>) -> Value {
    let mut body = body.clone();
    let unchanged = stored.is_some_and(|stored| {
        body.get("api_base")
            .and_then(Value::as_str)
            .is_some_and(|sent| sent != stored && sent == markitai_core::output::masked_url(stored))
    });
    if unchanged && let Some(fields) = body.as_object_mut() {
        fields.remove("api_base");
    }
    body
}

const MASK: char = '\u{2026}';

/// How a stored key is shown, as the reference shows it: the first two and
/// last four characters of a key of 12 or more, the last two of a shorter
/// one, nothing of one under four. An `env:` reference names a variable, not
/// a secret, and is shown as it is.
pub(super) fn masked_key(key: &str) -> String {
    if key.starts_with("env:") {
        return key.to_owned();
    }
    let characters = key.chars().collect::<Vec<_>>();
    let text = |part: &[char]| part.iter().collect::<String>();
    match characters.len() {
        length if length >= 12 => format!(
            "{}{MASK}{}",
            text(&characters[..2]),
            text(&characters[length - 4..])
        ),
        length if length >= 4 => format!("{MASK}{}", text(&characters[length - 2..])),
        _ => MASK.to_string(),
    }
}
pub(super) fn environment_connection(id: &str) -> ApiResult<Value> {
    let name = id.strip_prefix("env:").ok_or_else(missing)?;
    let allowed = markitai_core::provider_management::catalog()
        .into_iter()
        .any(|entry| entry.provider == name && !entry.key_variables.is_empty());
    if !allowed {
        return Err(invalid("unknown environment provider"));
    }
    Ok(json!({"provider":name,"use_environment_credentials":true}))
}
pub(super) fn same_base(left: Option<&str>, right: Option<&str>) -> bool {
    let normalize = |value: Option<&str>| {
        // An absent configured base may resolve a server environment variable;
        // it is not interchangeable with a client-supplied provider default.
        let value = value.filter(|v| !v.is_empty())?;
        let mut url = url::Url::parse(value).ok()?;
        // The network clients append routes after removing trailing slashes.
        let path = url.path().trim_end_matches('/').to_owned();
        url.set_path(&path);
        Some(url.to_string())
    };
    if left == right {
        return true;
    }
    match (normalize(left), normalize(right)) {
        (Some(left), Some(right)) => left == right,
        _ => false,
    }
}
pub(super) fn check_reuse(
    old_provider: &str,
    provider: &str,
    old_base: Option<&str>,
    base: Option<&str>,
    supplies_key: bool,
) -> ApiResult<()> {
    if !supplies_key && (old_provider != provider || !same_base(old_base, base)) {
        return Err(invalid(
            "changing a saved credential endpoint or provider requires an explicit API key (or explicit clear)",
        ));
    }
    Ok(())
}
pub(super) fn params(entry: &Value, providers: &[Value]) -> Value {
    let mut result = entry["litellm_params"].clone();
    if let Some(id) = linked(entry)
        && let Some(saved) = providers.iter().find(|p| p["id"] == id)
    {
        for field in ["api_key", "api_base", "use_environment_credentials"] {
            if result[field].is_null() {
                result[field] = saved[field].clone();
            }
        }
    }
    result
}

pub(super) fn provider(entry: &Value) -> String {
    entry
        .pointer("/litellm_params/model")
        .and_then(Value::as_str)
        .unwrap_or("")
        .split_once('/')
        .map_or("openai", |(provider, _)| provider)
        .to_ascii_lowercase()
}
/// The provider of a deployment's connection: its model prefix, or `custom`
/// for an `openai/` model linked to a saved OpenAI-compatible endpoint.
pub(super) fn connection_provider(entry: &Value, providers: &[Value]) -> String {
    let prefix = provider(entry);
    let custom = linked(entry)
        .and_then(|id| providers.iter().find(|p| p["id"] == id))
        .is_some_and(|saved| saved["provider"] == "custom");
    if prefix == "openai" && custom {
        "custom".into()
    } else {
        prefix
    }
}
/// `custom` names any OpenAI-compatible endpoint; its models use the `openai/`
/// prefix. Every other provider is its own model prefix.
fn serves(provider: &str, prefix: &str) -> bool {
    provider == prefix || (provider == "custom" && prefix == "openai")
}
pub(super) fn local(provider: &str) -> bool {
    matches!(provider, "claude-agent" | "copilot" | "chatgpt")
}
/// Whether an entry the settings API writes may use the server's own cloud
/// credentials (the AWS chain for Bedrock, the Google login for Vertex AI): a
/// Bedrock or Vertex AI entry with neither a key nor an address of its own,
/// whose requests go to the cloud's own endpoint, as Python markitai's LiteLLM
/// sends them (owner decision 2026-10-12). One with an address never does.
pub(super) fn cloud_credentials(provider: &str, key: &Value, base: &Value) -> bool {
    matches!(provider, "bedrock" | "vertex_ai")
        && !real_key(key.as_str())
        && base.as_str().is_none_or(str::is_empty)
}
pub(super) fn link(entry: &mut Value, id: &str) {
    if !entry["model_info"].is_object() {
        entry["model_info"] = json!({});
    }
    entry["model_info"]["provider_id"] = json!(id);
}
pub(super) fn linked(entry: &Value) -> Option<&str> {
    entry
        .pointer("/model_info/provider_id")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
}
fn same_connection(
    entry: &Value,
    name: &str,
    key: &Value,
    base: &Value,
    allow_environment: bool,
) -> bool {
    provider(entry) == name
        && entry["litellm_params"]["api_key"] == *key
        && entry["litellm_params"]["api_base"] == *base
        && (entry["litellm_params"]["use_environment_credentials"].as_bool() != Some(false))
            == allow_environment
}
fn ensure_provider(
    providers: &mut Vec<Value>,
    models: &mut [Value],
    name: &str,
    key: &Value,
    base: &Value,
    allow_environment: bool,
) -> Option<String> {
    if local(name) || (key.is_null() && base.is_null()) {
        return None;
    }
    let id = if let Some(saved) = providers.iter().find(|p| {
        p["provider"] == name
            && p["api_key"] == *key
            && p["api_base"] == *base
            && (p["use_environment_credentials"].as_bool() != Some(false)) == allow_environment
    }) {
        saved["id"].as_str()?.to_owned()
    } else {
        let id = uuid::Uuid::new_v4().to_string();
        let mut value =
            json!({"id":id,"provider":name,"use_environment_credentials":allow_environment});
        for (field, value_) in [("api_key", key), ("api_base", base)] {
            if !value_.is_null() {
                value[field] = value_.clone();
            }
        }
        providers.push(value);
        id
    };
    for entry in models {
        if same_connection(entry, name, key, base, allow_environment) {
            link(entry, &id);
        }
    }
    Some(id)
}
pub(super) fn find(
    models: &[Value],
    id: &str,
    mapping: &HashMap<String, String>,
) -> ApiResult<usize> {
    let id = mapping.get(id).map(String::as_str).unwrap_or(id);
    models
        .iter()
        .enumerate()
        .position(|(index, entry)| identity::id(entry, index) == id)
        .ok_or_else(missing)
}
pub(super) fn legacy(models: &[Value], name: &str) -> ApiResult<usize> {
    let mut found = models
        .iter()
        .enumerate()
        .filter(|(_, entry)| entry["model_name"] == name);
    let index = found.next().ok_or_else(missing)?.0;
    if found.next().is_some() {
        return Err(ApiError::structured(
            409,
            "ambiguous_legacy_model_name",
            json!({"code":"ambiguous_legacy_model_name","model_name":name}),
        ));
    }
    Ok(index)
}
pub(super) fn create(
    body: &Value,
    models: &mut Vec<Value>,
    providers: &mut Vec<Value>,
    credential_models: &[Value],
) -> ApiResult<()> {
    object(
        body,
        &[
            "model_name",
            "model",
            "provider",
            "api_key",
            "api_base",
            "weight",
            "credential_provider_id",
            "credential_deployment_id",
            "expected_revision",
        ],
    )?;
    let group = string(body, "model_name", true, true)?.unwrap();
    let model = string(body, "model", true, true)?.unwrap();
    let model_provider = model
        .split_once('/')
        .map_or("openai", |(provider, _)| provider)
        .to_ascii_lowercase();
    let requested = string(body, "provider", false, true)?.map(|name| name.to_lowercase());
    if requested
        .as_deref()
        .is_some_and(|name| name.is_empty() || !serves(name, &model_provider))
    {
        return Err(invalid("provider must match the model provider"));
    }
    let client_key = client_field(body, "api_key")?;
    let client_base = client_field(body, "api_base")?;
    let provider_ref = string(body, "credential_provider_id", false, false)?;
    let deployment_ref = string(body, "credential_deployment_id", false, false)?;
    if provider_ref.is_some() && deployment_ref.is_some() {
        return Err(invalid("send one credential reference"));
    }
    let mut provider_id = None;
    let fallback = if let Some(id) = provider_ref {
        if id.starts_with("env:") {
            environment_connection(&id)?
        } else {
            let value = providers
                .iter()
                .find(|p| p["id"] == id)
                .ok_or_else(missing)?;
            provider_id = Some(id);
            value.clone()
        }
    } else if let Some(id) = deployment_ref {
        let index = find(credential_models, &id, &HashMap::new())?;
        let entry = &credential_models[index];
        provider_id = linked(entry).map(str::to_owned);
        let mut value = params(entry, providers);
        value["provider"] = json!(connection_provider(entry, providers));
        value
    } else {
        json!({})
    };
    // An `openai/` model added from a saved OpenAI-compatible endpoint stays
    // with that endpoint when the request names no provider.
    let name = requested.unwrap_or_else(|| {
        if model_provider == "openai" && fallback["provider"] == "custom" {
            "custom".into()
        } else {
            model_provider.clone()
        }
    });
    if name.is_empty() {
        return Err(invalid("provider must match the model provider"));
    }
    let supplied_key = body.get("api_key").is_some();
    let key = if supplied_key {
        client_key.map(Value::String).unwrap_or(Value::Null)
    } else {
        fallback["api_key"].clone()
    };
    let base = if body.get("api_base").is_some() {
        client_base.map(Value::String).unwrap_or(Value::Null)
    } else {
        fallback["api_base"].clone()
    };
    refuse_ambient_login_address(&model, base.as_str())?;
    if let Some(old_provider) = fallback["provider"].as_str() {
        check_reuse(
            old_provider,
            &name,
            fallback["api_base"].as_str(),
            base.as_str(),
            supplied_key,
        )?;
    }
    if name == "custom" && base.as_str().is_none_or(str::is_empty) {
        return Err(invalid("an OpenAI-compatible endpoint needs its api_base"));
    }
    let allow_environment = (!supplied_key
        && fallback["provider"].is_string()
        && fallback["use_environment_credentials"].as_bool() != Some(false))
        || cloud_credentials(&name, &key, &base);
    // An address this request names, or a new connection's, never gets the
    // server's own key: without one it stores the placeholder. A saved
    // connection that may use the environment keeps its own pair.
    let key = keyless(key, &base, &name, allow_environment);
    if supplied_key {
        provider_id = None;
    }
    let weight = match body.get("weight") {
        None => 1,
        Some(value) => value
            .as_u64()
            .filter(|n| *n <= i64::MAX as u64)
            .ok_or_else(|| invalid("weight must be a nonnegative integer"))?,
    };
    if provider_id.is_none() {
        provider_id = ensure_provider(providers, models, &name, &key, &base, allow_environment);
    }
    let mut params = json!({"model":model,"use_environment_credentials":allow_environment});
    if !local(&provider(&json!({"litellm_params":params}))) {
        if !key.is_null() {
            params["api_key"] = key;
        }
        if !base.is_null() {
            params["api_base"] = base;
        }
    }
    if weight != 1 {
        params["weight"] = json!(weight);
    }
    let mut entry = json!({"model_name":group,"litellm_params":params,"model_info":{"id":uuid::Uuid::new_v4().to_string()}});
    if let Some(id) = provider_id {
        link(&mut entry, &id);
    }
    models.push(entry);
    Ok(())
}
pub(super) fn update(entry: &mut Value, body: &Value, providers: &[Value]) -> ApiResult<()> {
    let old = params(entry, providers);
    let body = &unchanged_base(body, old["api_base"].as_str());
    object(
        body,
        &[
            "model_name",
            "model",
            "api_key",
            "api_base",
            "weight",
            "expected_revision",
        ],
    )?;
    client_field(body, "api_key")?;
    let client_base = client_field(body, "api_base")?;
    let old_provider = connection_provider(entry, providers);
    let next_provider = body["model"]
        .as_str()
        .map(|name| {
            let prefix = name
                .split_once('/')
                .map_or("openai", |(provider, _)| provider)
                .to_ascii_lowercase();
            // Another model of the same OpenAI-compatible endpoint.
            if serves(&old_provider, &prefix) {
                old_provider.clone()
            } else {
                prefix
            }
        })
        .unwrap_or_else(|| old_provider.clone());
    let base = if body.get("api_base").is_some() {
        client_base.as_deref()
    } else {
        old["api_base"].as_str()
    };
    if body.get("api_base").is_some() || body.get("model").is_some() {
        let model = body["model"].as_str().or(old["model"].as_str());
        refuse_ambient_login_address(model.unwrap_or_default().trim(), base)?;
    }
    check_reuse(
        &old_provider,
        &next_provider,
        old["api_base"].as_str(),
        base,
        body.get("api_key").is_some(),
    )?;
    if body.get("api_key").is_some() {
        entry["litellm_params"]["use_environment_credentials"] = json!(false);
    }
    // An explicit clear must not silently re-inherit the same saved connection.
    if (body.get("api_key").is_some() || body.get("api_base").is_some_and(Value::is_null))
        && let Some(id) = linked(entry).map(str::to_owned)
    {
        if let Some(saved) = providers.iter().find(|p| p["id"] == id) {
            for field in ["api_key", "api_base"] {
                if body.get(field).is_none()
                    && entry["litellm_params"][field]
                        .as_str()
                        .is_none_or(str::is_empty)
                    && !saved[field].is_null()
                {
                    entry["litellm_params"][field] = saved[field].clone();
                }
            }
        }
        entry["model_info"]
            .as_object_mut()
            .unwrap()
            .remove("provider_id");
    }
    for field in ["model_name", "model"] {
        if body.get(field).is_some() {
            let value = string(body, field, true, true)?.unwrap();
            if field == "model_name" {
                entry[field] = json!(value);
            } else {
                entry["litellm_params"][field] = json!(value);
            }
        }
    }
    for field in ["api_key", "api_base"] {
        if body.get(field).is_some() {
            let value = string(body, field, false, false)?;
            replace(
                &mut entry["litellm_params"],
                field,
                value.map(Value::String).as_ref(),
            );
        }
    }
    if let Some(value) = body.get("weight").filter(|v| !v.is_null()) {
        if value.as_u64().filter(|n| *n <= i64::MAX as u64).is_none() {
            return Err(invalid("weight must be a nonnegative integer"));
        }
        entry["litellm_params"]["weight"] = value.clone();
    }
    if local(&provider(entry)) {
        let params = entry["litellm_params"].as_object_mut().unwrap();
        params.remove("api_key");
        params.remove("api_base");
    }
    if body.get("api_key").is_some() || body.get("api_base").is_some() {
        let mut next = params(entry, providers);
        let connection = connection_provider(entry, providers);
        refuse_keyless_move(&old, &next, &connection)?;
        // A cloud entry whose key is cleared and that has no address goes back
        // to the server's credentials at the cloud's own endpoint.
        if body.get("api_key").is_some()
            && cloud_credentials(&connection, &next["api_key"], &next["api_base"])
        {
            entry["litellm_params"]["use_environment_credentials"] = json!(true);
            next["use_environment_credentials"] = json!(true);
        }
        let allow_environment = next["use_environment_credentials"].as_bool() != Some(false);
        let key = keyless(
            next["api_key"].clone(),
            &next["api_base"],
            &connection,
            allow_environment,
        );
        if key != next["api_key"] {
            entry["litellm_params"]["api_key"] = key;
        }
    }
    Ok(())
}
fn replace(target: &mut Value, key: &str, value: Option<&Value>) {
    if let Some(value) = value.filter(|v| !v.is_null()) {
        target[key] = value.clone();
    } else {
        target.as_object_mut().unwrap().remove(key);
    }
}
pub(super) fn remove(models: &mut Vec<Value>, providers: &mut Vec<Value>, index: usize) {
    let entry = models[index].clone();
    if linked(&entry).is_none() {
        ensure_provider(
            providers,
            models,
            &provider(&entry),
            &entry["litellm_params"]["api_key"],
            &entry["litellm_params"]["api_base"],
            entry["litellm_params"]["use_environment_credentials"].as_bool() != Some(false),
        );
    }
    models.remove(index);
}
pub(super) fn mutate_provider(
    models: &mut Vec<Value>,
    providers: &mut Vec<Value>,
    id: &str,
    mapping: &HashMap<String, String>,
    body: Option<&Value>,
) -> ApiResult<()> {
    let saved = providers.iter().position(|p| p["id"] == id);
    let old = if let Some(index) = saved {
        providers[index].clone()
    } else if let Some(id) = id.strip_prefix("legacy:") {
        let entry = &models[find(models, id, mapping)?];
        json!({"provider":provider(entry),"api_key":entry["litellm_params"]["api_key"],"api_base":entry["litellm_params"]["api_base"],"use_environment_credentials":entry["litellm_params"]["use_environment_credentials"]})
    } else {
        return Err(missing());
    };
    let name = old["provider"].as_str().unwrap_or_default();
    let allow_environment = old["use_environment_credentials"].as_bool() != Some(false);
    let key = &old["api_key"];
    let base = &old["api_base"];
    if let Some(body) = body {
        object(body, &["api_key", "api_base", "expected_revision"])?;
        if body.get("api_key").is_none() && body.get("api_base").is_none() {
            return Err(invalid("api_key or api_base is required"));
        }
        let body = &unchanged_base(body, base.as_str());
        // Only the address, sent back as shown: nothing changes.
        if body.get("api_key").is_none() && body.get("api_base").is_none() {
            return Ok(());
        }
        client_field(body, "api_key")?;
        let requested_base = client_field(body, "api_base")?;
        let next_base = if body.get("api_base").is_some() {
            requested_base.as_deref()
        } else {
            old["api_base"].as_str()
        };
        if body.get("api_base").is_some() {
            refuse_ambient_login_address(&format!("{name}/"), next_base)?;
        }
        check_reuse(
            name,
            name,
            old["api_base"].as_str(),
            next_base,
            body.get("api_key").is_some(),
        )?;
        if body.get("api_base").is_some() && body.get("api_key").is_none() {
            // Linked deployments may override the provider's key and endpoint.
            // A provider-level no-op must not move one of those retained keys.
            for entry in models.iter().filter(|entry| {
                saved.is_some() && linked(entry) == old["id"].as_str()
                    || same_connection(entry, name, key, base, allow_environment)
            }) {
                let effective = params(entry, providers);
                let entry_provider = provider(entry);
                check_reuse(
                    &entry_provider,
                    &entry_provider,
                    effective["api_base"].as_str(),
                    next_base,
                    false,
                )?;
            }
        }
        let mut next = old.clone();
        if body.get("api_key").is_some() {
            next["use_environment_credentials"] = json!(false);
        }
        if saved.is_none() {
            next["id"] = json!(uuid::Uuid::new_v4().to_string());
        }
        for field in ["api_key", "api_base"] {
            if body.get(field).is_some() {
                let value = string(body, field, false, true)?;
                if value.as_deref() == Some("") {
                    return Err(invalid("credential fields cannot be blank"));
                }
                replace(&mut next, field, value.map(Value::String).as_ref());
            }
        }
        if body.get("api_key").is_some()
            && cloud_credentials(name, &next["api_key"], &next["api_base"])
        {
            next["use_environment_credentials"] = json!(true);
        }
        refuse_keyless_move(&old, &next, name)?;
        let placeholder = keyless(
            next["api_key"].clone(),
            &next["api_base"],
            name,
            next["use_environment_credentials"].as_bool() != Some(false),
        );
        // The placeholder is the connection's key now, for its models too.
        let key_changed = body.get("api_key").is_some() || placeholder != next["api_key"];
        if !placeholder.is_null() {
            next["api_key"] = placeholder;
        }
        let id = next["id"].as_str().unwrap().to_owned();
        for entry in models {
            if linked(entry) == Some(id.as_str())
                || same_connection(entry, name, key, base, allow_environment)
            {
                for field in ["api_key", "api_base"] {
                    if body.get(field).is_some() || field == "api_key" && key_changed {
                        replace(&mut entry["litellm_params"], field, next.get(field));
                    }
                }
                if key_changed {
                    entry["litellm_params"]["use_environment_credentials"] =
                        json!(next["use_environment_credentials"] == true);
                }
                link(entry, &id);
            }
        }
        if let Some(index) = saved {
            providers[index] = next;
        } else {
            providers.push(next);
        }
    } else {
        models.retain(|entry| {
            !(saved.is_some() && linked(entry) == old["id"].as_str()
                || same_connection(entry, name, key, base, allow_environment))
        });
        if let Some(index) = saved {
            providers.remove(index);
        }
    }
    Ok(())
}
