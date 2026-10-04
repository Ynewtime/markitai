use super::{ApiError, ApiResult, identity, invalid, missing};
use serde_json::{Map, Value, json};
use std::collections::HashMap;

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
    Ok(value)
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
pub(super) fn same_base(_provider: &str, left: Option<&str>, right: Option<&str>) -> bool {
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
    if !supplies_key && (old_provider != provider || !same_base(provider, old_base, base)) {
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
pub(super) fn local(provider: &str) -> bool {
    matches!(provider, "claude-agent" | "copilot" | "chatgpt")
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
    let name = string(body, "provider", false, true)?
        .unwrap_or_else(|| {
            model
                .split_once('/')
                .map_or("openai", |(provider, _)| provider)
                .to_owned()
        })
        .to_lowercase();
    let model_provider = model
        .split_once('/')
        .map_or("openai", |(provider, _)| provider);
    if name.is_empty() || name != model_provider.to_ascii_lowercase() {
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
        value["provider"] = json!(provider(entry));
        value
    } else {
        json!({})
    };
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
    if let Some(old_provider) = fallback["provider"].as_str() {
        check_reuse(
            old_provider,
            &name,
            fallback["api_base"].as_str(),
            base.as_str(),
            supplied_key,
        )?;
    }
    let allow_environment = !supplied_key
        && fallback["provider"].is_string()
        && fallback["use_environment_credentials"].as_bool() != Some(false);
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
    let old = params(entry, providers);
    let old_provider = provider(entry);
    let next_provider = body["model"]
        .as_str()
        .map(|name| {
            name.split_once('/')
                .map_or("openai", |(provider, _)| provider)
                .to_ascii_lowercase()
        })
        .unwrap_or_else(|| old_provider.clone());
    let base = if body.get("api_base").is_some() {
        client_base.as_deref()
    } else {
        old["api_base"].as_str()
    };
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
        client_field(body, "api_key")?;
        let requested_base = client_field(body, "api_base")?;
        let next_base = if body.get("api_base").is_some() {
            requested_base.as_deref()
        } else {
            old["api_base"].as_str()
        };
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
        let id = next["id"].as_str().unwrap().to_owned();
        for entry in models {
            if linked(entry) == Some(id.as_str())
                || same_connection(entry, name, key, base, allow_environment)
            {
                for field in ["api_key", "api_base"] {
                    if body.get(field).is_some() {
                        replace(&mut entry["litellm_params"], field, next.get(field));
                    }
                }
                if body.get("api_key").is_some() {
                    entry["litellm_params"]["use_environment_credentials"] = json!(false);
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
