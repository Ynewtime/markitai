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
pub(super) fn provider(entry: &Value) -> String {
    entry
        .pointer("/litellm_params/model")
        .and_then(Value::as_str)
        .unwrap_or("")
        .split('/')
        .next()
        .unwrap_or("")
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
fn same_connection(entry: &Value, name: &str, key: &Value, base: &Value) -> bool {
    provider(entry) == name
        && entry["litellm_params"]["api_key"] == *key
        && entry["litellm_params"]["api_base"] == *base
}
fn ensure_provider(
    providers: &mut Vec<Value>,
    models: &mut [Value],
    name: &str,
    key: &Value,
    base: &Value,
) -> Option<String> {
    if local(name) || (key.is_null() && base.is_null()) {
        return None;
    }
    let id = if let Some(saved) = providers
        .iter()
        .find(|p| p["provider"] == name && p["api_key"] == *key && p["api_base"] == *base)
    {
        saved["id"].as_str()?.to_owned()
    } else {
        let id = uuid::Uuid::new_v4().to_string();
        let mut value = json!({"id":id,"provider":name});
        for (field, value_) in [("api_key", key), ("api_base", base)] {
            if !value_.is_null() {
                value[field] = value_.clone();
            }
        }
        providers.push(value);
        id
    };
    for entry in models {
        if same_connection(entry, name, key, base) {
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
        .unwrap_or_else(|| model.split('/').next().unwrap_or("").to_owned())
        .to_lowercase();
    if name.is_empty() {
        return Err(invalid("provider cannot be blank"));
    }
    let provider_ref = string(body, "credential_provider_id", false, false)?;
    let deployment_ref = string(body, "credential_deployment_id", false, false)?;
    if provider_ref.is_some() && deployment_ref.is_some() {
        return Err(invalid("send one credential reference"));
    }
    let mut provider_id = None;
    let fallback = if let Some(id) = provider_ref {
        let value = providers
            .iter()
            .find(|p| p["id"] == id)
            .ok_or_else(missing)?;
        provider_id = Some(id);
        value.clone()
    } else if let Some(id) = deployment_ref {
        let index = find(credential_models, &id, &HashMap::new())?;
        let entry = &credential_models[index];
        provider_id = linked(entry).map(str::to_owned);
        entry["litellm_params"].clone()
    } else {
        json!({})
    };
    let key = string(body, "api_key", false, false)?
        .map(Value::String)
        .unwrap_or_else(|| fallback["api_key"].clone());
    let base = string(body, "api_base", false, false)?
        .map(Value::String)
        .unwrap_or_else(|| fallback["api_base"].clone());
    let weight = match body.get("weight") {
        None => 1,
        Some(value) => value
            .as_u64()
            .filter(|n| *n <= i64::MAX as u64)
            .ok_or_else(|| invalid("weight must be a nonnegative integer"))?,
    };
    if provider_id.is_none() {
        provider_id = ensure_provider(providers, models, &name, &key, &base);
    }
    let mut params = json!({"model":model});
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
    // An explicit clear must not silently re-inherit the same saved connection.
    if ["api_key", "api_base"]
        .iter()
        .any(|field| body.get(*field).is_some_and(Value::is_null))
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
        json!({"provider":provider(entry),"api_key":entry["litellm_params"]["api_key"],"api_base":entry["litellm_params"]["api_base"]})
    } else {
        return Err(missing());
    };
    let name = old["provider"].as_str().unwrap_or_default();
    let key = &old["api_key"];
    let base = &old["api_base"];
    if let Some(body) = body {
        object(body, &["api_key", "api_base", "expected_revision"])?;
        if body.get("api_key").is_none() && body.get("api_base").is_none() {
            return Err(invalid("api_key or api_base is required"));
        }
        let mut next = old.clone();
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
            if linked(entry) == Some(id.as_str()) || same_connection(entry, name, key, base) {
                for field in ["api_key", "api_base"] {
                    if body.get(field).is_some() {
                        replace(&mut entry["litellm_params"], field, next.get(field));
                    }
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
                || same_connection(entry, name, key, base))
        });
        if let Some(index) = saved {
            providers.remove(index);
        }
    }
    Ok(())
}
