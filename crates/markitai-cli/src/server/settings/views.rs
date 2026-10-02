use super::{SettingsSource, identity, model, store::Data};
use markitai_core::provider_management::{
    ProviderEntry, catalog, provider_default_base, provider_label,
};
use serde_json::{Value, json};
use std::collections::HashSet;

fn entry(name: &str) -> Option<ProviderEntry> {
    let name = if name == "ollama_chat" {
        "ollama"
    } else {
        name
    };
    catalog().into_iter().find(|entry| entry.provider == name)
}
fn set(variable: &str) -> bool {
    std::env::var(variable).is_ok_and(|value| !value.trim().is_empty())
}
/// What a card says about its provider beyond the connection: the documented
/// endpoint, the key variable, whether a key is needed, whether models can be
/// listed.
fn facts(name: &str, card: &mut Value) {
    let entry = entry(name);
    let variable = entry.as_ref().and_then(|entry| {
        entry
            .key_variables
            .iter()
            .find(|variable| set(variable))
            .or(entry.key_variables.first())
            .copied()
    });
    card["default_base"] = json!(entry.as_ref().and_then(|entry| entry.default_base));
    card["key_variable"] = json!(variable);
    card["key_optional"] = json!(entry.as_ref().is_some_and(|entry| entry.key_optional));
    if card
        .get("supports_discovery")
        .is_none_or(|value| value == true)
    {
        card["supports_discovery"] =
            json!(!model::local(name) && entry.as_ref().is_none_or(|entry| entry.discovery));
    }
}
fn nonempty(value: &Value) -> bool {
    value.as_str().is_some_and(|s| !s.is_empty())
}
fn resolved(value: &Value) -> Option<String> {
    value.as_str().and_then(|s| {
        if let Some(key) = s.strip_prefix("env:") {
            std::env::var(key).ok()
        } else {
            Some(s.to_owned())
        }
    })
}
fn origin(value: Option<&str>) -> Option<String> {
    let value = value?;
    let value = if let Some(key) = value.strip_prefix("env:") {
        std::env::var(key).ok()?
    } else {
        value.to_owned()
    };
    let url = url::Url::parse(&value).ok()?;
    if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
        return None;
    }
    Some(url.origin().ascii_serialization())
}
/// Whether a conversion could route an LLM request: what the conversion side
/// reports (subscription runtimes, models detected from the environment), or
/// a configured deployment whose key a linked provider record holds.
fn routable(data: &Data) -> bool {
    markitai_core::llm_capabilities(&data.cfg).routable || keyed_deployment(data)
}
fn keyed_deployment(data: &Data) -> bool {
    data.cfg["llm"]["model_list"]
        .as_array()
        .is_some_and(|models| {
            models.iter().any(|entry| {
                if entry["litellm_params"]["weight"].as_u64() == Some(0) {
                    return false;
                }
                let name = model::provider(entry);
                let Some(known) = self::entry(&name).filter(|known| known.provider != "custom")
                else {
                    return false;
                };
                let linked = model::linked(entry)
                    .and_then(|id| data.providers.iter().find(|p| p["id"] == id));
                let explicit = |field: &str| {
                    entry["litellm_params"]
                        .get(field)
                        .filter(|v| nonempty(v))
                        .or_else(|| linked.and_then(|p| p.get(field).filter(|v| nonempty(v))))
                };
                let usable = |value: &Value| resolved(value).is_some_and(|value| !value.is_empty());
                // A local server needs no key, only an address (vLLM has no default one).
                if known.key_optional {
                    return known.default_base.is_some()
                        || explicit("api_base").is_some_and(usable)
                        || known.base_variables.iter().any(|variable| set(variable));
                }
                // An explicit key, even an unresolvable reference, is the only candidate.
                match explicit("api_key") {
                    Some(value) => usable(value),
                    None => known.key_variables.iter().any(|variable| set(variable)),
                }
            })
        })
}
fn deployment(entry: &Value, index: usize, persisted: bool) -> Value {
    let params = &entry["litellm_params"];
    let local = model::local(&model::provider(entry));
    json!({"deployment_id":identity::id(entry,index),"routing_group":entry["model_name"].as_str().unwrap_or("default"),"model":params["model"],"weight":params["weight"].as_u64().unwrap_or(1),"api_key_configured":!local&&nonempty(&params["api_key"]),"api_base_configured":!local&&nonempty(&params["api_base"]),"api_base":if local{None}else{origin(params["api_base"].as_str())},"persisted":persisted})
}
pub(super) fn payload(data: &Data, source: &SettingsSource) -> Value {
    let path = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .and_then(|home| {
            source
                .path
                .strip_prefix(home)
                .ok()
                .map(|rest| format!("~/{}", rest.display()))
        })
        .unwrap_or_else(|| source.path.to_string_lossy().into_owned());
    json!({"configured":!data.models.is_empty(),"routable":routable(data),"source":if !data.models.is_empty(){"config"}else if !data.detected.is_empty(){"detected"}else{"none"},"config_path":path,"config_origin":source.origin,"revision":data.revision,"deployments":data.models.iter().enumerate().map(|(index,entry)|deployment(entry,index,true)).collect::<Vec<_>>(),"detected":data.detected.iter().enumerate().map(|(index,entry)|deployment(entry,index,false)).collect::<Vec<_>>()})
}
pub(super) fn providers(data: &Data) -> Value {
    let mut result = Vec::new();
    let catalog = catalog();
    for known in &catalog {
        let name = known.provider;
        let Some(variable) = known.key_variables.iter().copied().find(|v| set(v)) else {
            continue;
        };
        if !data
            .providers
            .iter()
            .filter(|p| p["provider"] == name)
            .any(|p| p["api_key"] == format!("env:{variable}"))
        {
            result.push(json!({"id":format!("env:{name}"),"provider":name,"label":provider_label(name),"kind":"environment","status":"ready","source":variable,"credential":format!("env:{variable}"),"supports_discovery":known.discovery}));
        }
    }
    // The reference's providers first, then the further OpenAI-compatible ones.
    for known in &catalog {
        let name = known.provider;
        if data.providers.iter().any(|p| p["provider"] == name)
            || data.models.iter().any(|p| model::provider(p) == name)
            || result.iter().any(|p| p["provider"] == name)
        {
            continue;
        }
        let label = match name {
            "azure" => "Azure OpenAI".to_owned(),
            "custom" => "OpenAI-compatible endpoint".to_owned(),
            _ => provider_label(name),
        };
        let status = if known.key_optional && known.default_base.is_some() {
            "unknown"
        } else {
            "needs_credentials"
        };
        result.push(json!({"id":format!("common:{name}"),"provider":name,"label":label,"kind":if known.compatible{"compatible"}else{"common"},"status":status,"source":"built_in","supports_discovery":known.discovery}));
    }
    for saved in &data.providers {
        let name = saved["provider"].as_str().unwrap_or_default();
        let id = saved["id"].as_str().unwrap_or_default();
        let count = data
            .models
            .iter()
            .filter(|entry| model::linked(entry) == Some(id))
            .count();
        result.push(json!({"id":format!("provider:{id}"),"provider_id":id,"provider":name,"label":provider_label(name),"kind":"configured","status":if !saved["api_key"].is_null()||!saved["api_base"].is_null(){"ready"}else{"needs_credentials"},"source":"config","api_key_configured":nonempty(&saved["api_key"]),"api_base_configured":nonempty(&saved["api_base"]),"api_base":origin(saved["api_base"].as_str().filter(|s|!s.is_empty()).or_else(||provider_default_base(name))),"model_count":count,"supports_discovery":!model::local(name)}));
    }
    let mut seen = HashSet::new();
    for (index, entry) in data.models.iter().enumerate() {
        let name = model::provider(entry);
        let params = &entry["litellm_params"];
        if model::local(&name)
            || data.providers.iter().any(|p| {
                model::linked(entry) == p["id"].as_str()
                    || p["provider"] == name
                        && p["api_key"] == params["api_key"]
                        && p["api_base"] == params["api_base"]
            })
        {
            continue;
        }
        let connection = json!([name, params["api_key"], params["api_base"]]).to_string();
        if !seen.insert(connection) {
            continue;
        }
        let count = data
            .models
            .iter()
            .filter(|other| {
                model::provider(other) == name
                    && other["litellm_params"]["api_key"] == params["api_key"]
                    && other["litellm_params"]["api_base"] == params["api_base"]
            })
            .count();
        let id = identity::id(entry, index);
        result.push(json!({"id":format!("provider:legacy:{id}"),"provider_id":format!("legacy:{id}"),"deployment_id":id,"provider":name,"label":provider_label(&name),"kind":"configured","status":if params["weight"].as_u64()==Some(0){"disabled"}else{"ready"},"source":"config","api_key_configured":nonempty(&params["api_key"]),"api_base_configured":nonempty(&params["api_base"]),"api_base":origin(params["api_base"].as_str().filter(|s|!s.is_empty()).or_else(||provider_default_base(&name))),"model_count":count,"supports_discovery":true}));
    }
    for card in &mut result {
        let name = card["provider"].as_str().unwrap_or_default().to_owned();
        facts(&name, card);
    }
    json!({"providers":result})
}
